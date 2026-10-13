"""Exercise the real PC1 engine/background adapter without CUDA/PyTorch."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "strategies/tide_engine/distributed_engine.py"
BACKEND_PATH = ROOT / "strategies/tide_engine/gsplat_backend.py"


class FakeTensor:
    """Use array operations to preserve the adapter's tensor/value semantics."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape

    def reshape(self, *shape):
        return FakeTensor(self.values.reshape(*shape))

    def expand(self, *shape):
        return FakeTensor(np.broadcast_to(self.values, shape))

    def contiguous(self):
        return FakeTensor(self.values.copy())

    def transpose(self, first, second):
        return FakeTensor(self.values.swapaxes(first, second))


def stack(tensors, dim=0):
    return FakeTensor(np.stack([tensor.values for tensor in tensors], axis=dim))


class RenderProbe:
    """Record the external renderer boundary after all actual adapter logic."""

    def __init__(self):
        self.calls = []
        self.result = object()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def load_functions(namespace, *, engine_source=None):
    """Compile complete production functions, including the nested PC1 submit."""
    engine = ast.parse(
        ENGINE_PATH.read_text(encoding="utf-8")
        if engine_source is None else engine_source
    )
    backend = ast.parse(BACKEND_PATH.read_text(encoding="utf-8"))
    pc1_submit = next(
        node for node in ast.walk(engine)
        if isinstance(node, ast.FunctionDef) and node.name == "_pc1_submit"
    )
    backend_functions = [
        node for node in backend.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_stack_cameras", "distributed_rasterize"}
    ]
    for filename, functions in (
        (ENGINE_PATH, [pc1_submit]),
        (BACKEND_PATH, backend_functions),
    ):
        module = ast.Module(body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *functions,
        ], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(filename), "exec"), namespace)
    return namespace["_pc1_submit"], namespace["distributed_rasterize"]


def make_case(camera_count, background, *, missing_layout=False):
    cameras = [
        SimpleNamespace(
            world_view_transform=FakeTensor(np.eye(4) * (index + 1)),
            K=FakeTensor(np.eye(3) * (index + 1)),
        )
        for index in range(camera_count)
    ]
    pc1_probe, off_probe = RenderProbe(), RenderProbe()
    render_inputs = tuple(object() for _ in range(4))
    leaves = {"xyz": object()}
    gaussians = SimpleNamespace(active_sh_degree=2)
    namespace = {
        "torch": SimpleNamespace(stack=stack),
        "background": background,
        "gaussians": gaussians,
        "leaves": leaves,
        "_transformed_leaves": lambda gaussian_model, tensors: render_inputs,
        "debug_projection_input": lambda **kwargs: None,
        "pc1_layout": None if missing_layout else SimpleNamespace(
            microbatch_cameras=lambda index: ("global-viewmats", "global-Ks"),
            N_world=[32, 32],
            microbatch_C_world=[camera_count, camera_count],
        ),
        "pc1_pipeline": SimpleNamespace(submit_projection=pc1_probe),
        "image_width": 12,
        "image_height": 8,
        "args": SimpleNamespace(radius_clip=0.25),
        "require_distributed_gsplat": lambda: SimpleNamespace(
            rasterization=off_probe
        ),
    }
    return namespace, cameras, pc1_probe, off_probe, render_inputs


class DistributedPipelineBackgroundTest(unittest.TestCase):
    def assert_background_matches_off(self, values):
        for camera_count in (1, 4):
            with self.subTest(camera_count=camera_count, values=values):
                background = None if values is None else FakeTensor(values)
                namespace, cameras, pc1, off, render_inputs = make_case(
                    camera_count, background
                )
                submit, rasterize = load_functions(namespace)
                handle, returned_inputs = submit(1, cameras)
                self.assertIs(handle, pc1.result)
                self.assertIs(returned_inputs, render_inputs)
                rasterize(
                    means=namespace["leaves"]["xyz"],
                    scales=render_inputs[0],
                    quats=render_inputs[1],
                    opacities=render_inputs[2],
                    sh_coefficients=render_inputs[3],
                    cameras=cameras,
                    width=12,
                    height=8,
                    sh_degree=2,
                    background=background,
                    radius_clip=0.25,
                )
                self.assertEqual(len(pc1.calls), 1)
                self.assertEqual(len(off.calls), 1)
                pc1_values, off_values = pc1.calls[0], off.calls[0]
                if values is None:
                    self.assertIsNone(pc1_values["backgrounds"])
                    self.assertIsNone(off_values["backgrounds"])
                else:
                    expected = np.repeat([values], camera_count, axis=0)
                    self.assertEqual(pc1_values["backgrounds"].shape, (camera_count, 3))
                    np.testing.assert_array_equal(
                        pc1_values["backgrounds"].values,
                        off_values["backgrounds"].values,
                    )
                    np.testing.assert_allclose(
                        pc1_values["backgrounds"].values, expected
                    )
                    np.testing.assert_array_equal(background.values, values)
                for name in ("viewmats", "Ks"):
                    np.testing.assert_array_equal(
                        pc1_values[name].values, off_values[name].values
                    )
                for name in (
                    "means", "quats", "scales", "opacities", "colors",
                    "width", "height", "sh_degree", "radius_clip",
                ):
                    self.assertEqual(pc1_values[name], off_values[name])

    def test_default_black_background_preserves_none(self):
        self.assert_background_matches_off(None)

    def test_white_background_matches_off(self):
        self.assert_background_matches_off([1.0, 1.0, 1.0])

    def test_custom_background_matches_off(self):
        self.assert_background_matches_off([0.125, 0.25, 0.75])

    def test_missing_layout_fails_before_renderer(self):
        namespace, cameras, pc1, off, _ = make_case(
            4, None, missing_layout=True
        )
        submit, _ = load_functions(namespace)
        with self.assertRaisesRegex(RuntimeError, "camera layout is not initialized"):
            submit(0, cameras)
        self.assertEqual(pc1.calls, [])
        self.assertEqual(off.calls, [])


if __name__ == "__main__":
    unittest.main()
