"""Exercise real PC1 camera slicing and validation without CUDA/PyTorch."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


PIPELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "submodules/gsplat/gsplat/distributed_pipeline.py"
)
CONFIGURATIONS = ((2, 2, 1), (2, 16, 4), (4, 16, 4), (8, 16, 4))


class FakeTensor:
    """Preserve actual tensor slicing/reshape semantics for camera metadata."""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.ndim = self.values.ndim
        self.device = SimpleNamespace(type="cuda")

    def reshape(self, *shape):
        return FakeTensor(self.values.reshape(*shape))

    def __getitem__(self, indices):
        return FakeTensor(self.values[indices])


class ProjectionBoundaryReached(Exception):
    pass


class ProjectionProbe:
    """Stop only at the native projection call, after all real shape checks."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise ProjectionBoundaryReached


def load_pipeline_layout(world_size, rank):
    """Load the complete actual class and projection preparation function."""

    source = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"))
    nodes = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *[
            node for node in source.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in {"BatchCameraLayout", "_projection_prepare"}
        ],
    ]
    probe = ProjectionProbe()
    namespace = {
        "__name__": __name__,
        "dataclass": dataclass,
        "torch": SimpleNamespace(
            is_tensor=lambda value: isinstance(value, FakeTensor),
            distributed=SimpleNamespace(
                get_world_size=lambda: world_size,
                get_rank=lambda: rank,
            ),
            cuda=SimpleNamespace(current_stream=lambda device: None),
        ),
        "_cuda_event": lambda: SimpleNamespace(record=lambda stream: None),
        "fully_fused_projection": probe,
    }
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(PIPELINE_PATH), "exec"), namespace)
    return namespace["BatchCameraLayout"], namespace["_projection_prepare"], probe


def make_layout(layout_class, world_size, local_cameras, microbatch):
    identifiers = np.array([
        100 * rank + camera
        for rank in range(world_size)
        for camera in range(local_cameras)
    ])
    return layout_class(
        N_world=[32 + rank for rank in range(world_size)],
        C_world=[local_cameras] * world_size,
        viewmats_global=FakeTensor(
            np.broadcast_to(identifiers[:, None, None], (len(identifiers), 4, 4))
        ),
        Ks_global=FakeTensor(
            np.broadcast_to(identifiers[:, None, None], (len(identifiers), 3, 3))
        ),
        local_camera_count=local_cameras,
        microbatch=microbatch,
    )


def projection_inputs(layout, rank, index, *, counts=None):
    global_viewmats, global_Ks = layout.microbatch_cameras(index)
    local_start = rank * layout.microbatch
    local_end = local_start + layout.microbatch
    gaussian_count = layout.N_world[rank]
    return {
        "means": FakeTensor(np.zeros((gaussian_count, 3))),
        "quats": FakeTensor(np.zeros((gaussian_count, 4))),
        "scales": FakeTensor(np.zeros((gaussian_count, 3))),
        "opacities": FakeTensor(np.zeros((gaussian_count,))),
        "colors": FakeTensor(np.zeros((gaussian_count, 16, 3))),
        "viewmats": global_viewmats[local_start:local_end],
        "Ks": global_Ks[local_start:local_end],
        "viewmats_global": global_viewmats,
        "Ks_global": global_Ks,
        "width": 8,
        "height": 8,
        "near_plane": 0.01,
        "far_plane": 1e10,
        "radius_clip": 0.0,
        "eps2d": 0.3,
        "sh_degree": 0,
        "backgrounds": FakeTensor(np.zeros((layout.microbatch, 3))),
        "tile_size": 16,
        "rasterize_mode": "classic",
        "N_world": layout.N_world,
        "C_world": layout.microbatch_C_world if counts is None else counts,
    }


class DistributedPipelineCameraLayoutTest(unittest.TestCase):
    def test_microbatch_counts_preserve_full_batch_counts(self):
        for world_size, local_cameras, microbatch in CONFIGURATIONS:
            with self.subTest(world_size=world_size, microbatch=microbatch):
                layout_class, _, _ = load_pipeline_layout(world_size, 0)
                layout = make_layout(layout_class, world_size, local_cameras, microbatch)
                self.assertEqual(layout.C_world, [local_cameras] * world_size)
                self.assertEqual(layout.global_camera_count, world_size * local_cameras)
                self.assertEqual(layout.microbatch_C_world, [microbatch] * world_size)
                self.assertEqual(layout.N_world, [32 + rank for rank in range(world_size)])

    def test_microbatch_slices_keep_rank_major_order(self):
        for world_size, local_cameras, microbatch in CONFIGURATIONS:
            layout_class, _, _ = load_pipeline_layout(world_size, 0)
            layout = make_layout(layout_class, world_size, local_cameras, microbatch)
            for index in range(local_cameras // microbatch):
                with self.subTest(world_size=world_size, index=index):
                    viewmats, Ks = layout.microbatch_cameras(index)
                    expected = [
                        100 * rank + camera
                        for rank in range(world_size)
                        for camera in range(index * microbatch, (index + 1) * microbatch)
                    ]
                    self.assertEqual(viewmats.shape, (world_size * microbatch, 4, 4))
                    self.assertEqual(Ks.shape, (world_size * microbatch, 3, 3))
                    np.testing.assert_array_equal(viewmats.values[:, 0, 0], expected)
                    np.testing.assert_array_equal(Ks.values[:, 0, 0], expected)

    def test_projection_accepts_microbatch_counts_on_every_rank(self):
        for world_size, local_cameras, microbatch in CONFIGURATIONS:
            for rank in range(world_size):
                with self.subTest(world_size=world_size, rank=rank):
                    layout_class, prepare, probe = load_pipeline_layout(world_size, rank)
                    layout = make_layout(layout_class, world_size, local_cameras, microbatch)
                    values = projection_inputs(layout, rank, 1)
                    with self.assertRaises(ProjectionBoundaryReached):
                        prepare(**values)
                    self.assertEqual(len(probe.calls), 1)
                    self.assertIs(probe.calls[0][0][4], values["viewmats_global"])
                    self.assertIs(probe.calls[0][0][5], values["Ks_global"])

    def test_projection_rejects_full_batch_counts_before_cuda(self):
        for world_size, local_cameras, microbatch in CONFIGURATIONS:
            for rank in range(world_size):
                with self.subTest(world_size=world_size, rank=rank):
                    layout_class, prepare, probe = load_pipeline_layout(world_size, rank)
                    layout = make_layout(layout_class, world_size, local_cameras, microbatch)
                    with self.assertRaisesRegex(ValueError, "local camera count"):
                        prepare(**projection_inputs(layout, rank, 0, counts=layout.C_world))
                    self.assertEqual(probe.calls, [])

    def test_projection_rejects_mismatched_global_camera_or_intrinsic_count(self):
        for field in ("viewmats_global", "Ks_global"):
            with self.subTest(field=field):
                layout_class, prepare, probe = load_pipeline_layout(2, 0)
                layout = make_layout(layout_class, 2, 16, 4)
                values = projection_inputs(layout, 0, 0)
                values[field] = values[field][:-1]
                with self.assertRaisesRegex(ValueError, "does not match gathered tensors"):
                    prepare(**values)
                self.assertEqual(probe.calls, [])

    def test_projection_preserves_gaussian_count_guard(self):
        layout_class, prepare, probe = load_pipeline_layout(2, 1)
        layout = make_layout(layout_class, 2, 16, 4)
        values = projection_inputs(layout, 1, 0)
        values["N_world"] = [32, 34]
        with self.assertRaisesRegex(ValueError, "local Gaussian count changed"):
            prepare(**values)
        self.assertEqual(probe.calls, [])

    def test_whole_batch_microbatch_keeps_full_camera_counts(self):
        layout_class, prepare, probe = load_pipeline_layout(2, 1)
        layout = make_layout(layout_class, 2, 16, 16)
        self.assertEqual(layout.microbatch_C_world, layout.C_world)
        with self.assertRaises(ProjectionBoundaryReached):
            prepare(**projection_inputs(layout, 1, 0))
        self.assertEqual(len(probe.calls), 1)

    def test_microbatch_slice_rejects_out_of_range_indices(self):
        layout_class, _, _ = load_pipeline_layout(2, 0)
        layout = make_layout(layout_class, 2, 16, 4)
        for index in (-1, 4):
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    layout.microbatch_cameras(index)


if __name__ == "__main__":
    unittest.main()
