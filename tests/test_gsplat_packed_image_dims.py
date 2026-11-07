"""Exercise the real gsplat Python shape checks without importing CUDA/PyTorch."""

import ast
import unittest
from pathlib import Path


WRAPPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "submodules/gsplat/gsplat/cuda/_wrapper.py"
)
RASTERIZERS = ("rasterize_to_pixels", "rasterize_to_pixels_2dgs")


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.device = "cpu"

    def size(self, index):
        return self.shape[index]

    def contiguous(self):
        return self


class ApplyProbe:
    """Replace only the native boundary, preserving the wrapper's validation."""

    def __init__(self, is_2dgs):
        self.is_2dgs = is_2dgs
        self.calls = []

    def apply(self, *args):
        self.calls.append(args)
        offsets = args[11 if self.is_2dgs else 9]
        width, height = args[8:10] if self.is_2dgs else args[6:8]
        shape = offsets.shape[:-2] + (height, width)
        outputs = [FakeTensor(shape + (3,)), FakeTensor(shape + (1,))]
        if self.is_2dgs:
            outputs.extend(
                [FakeTensor(shape + (3,)), FakeTensor(shape + (1,)),
                 FakeTensor(shape + (1,))]
            )
        return tuple(outputs)


def load_rasterizer(name, source=None):
    """Compile a complete actual wrapper function; replace its CUDA apply only."""
    tree = ast.parse(WRAPPER_PATH.read_text() if source is None else source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.returns = None
    for argument in (*function.args.posonlyargs, *function.args.args,
                     *function.args.kwonlyargs):
        argument.annotation = None
    function.decorator_list = []
    probe = ApplyProbe(name.endswith("_2dgs"))
    namespace = {
        "_RasterizeToPixels": probe,
        "_RasterizeToPixels2DGS": probe,
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(WRAPPER_PATH), "exec"), namespace)
    return namespace[name], probe


def inputs(name, image_dims, *, packed, nnz=7, background_shape=None):
    gaussian_dims = (nnz,) if packed else tuple(image_dims) + (nnz,)
    result = {
        "means2d": FakeTensor(gaussian_dims + (2,)),
        "colors": FakeTensor(gaussian_dims + (3,)),
        "opacities": FakeTensor(gaussian_dims),
        "image_width": 8,
        "image_height": 8,
        "tile_size": 16,
        "isect_offsets": FakeTensor(tuple(image_dims) + (1, 1)),
        "flatten_ids": FakeTensor((nnz,)),
        "backgrounds": FakeTensor(
            tuple(image_dims) + (3,) if background_shape is None else background_shape
        ),
        "packed": packed,
    }
    if name.endswith("_2dgs"):
        result.update(
            ray_transforms=FakeTensor(gaussian_dims + (3, 3)),
            normals=FakeTensor(gaussian_dims + (3,)),
            densify=FakeTensor(gaussian_dims + (2,)),
        )
    else:
        result["conics"] = FakeTensor(gaussian_dims + (3,))
    return result


class GsplatPackedImageDimsTest(unittest.TestCase):
    def assert_success(self, name, values):
        rasterizer, probe = load_rasterizer(name)
        output = rasterizer(**values)
        self.assertEqual(len(probe.calls), 1)
        self.assertIs(
            probe.calls[0][6 if probe.is_2dgs else 4], values["backgrounds"]
        )
        self.assertEqual(
            output[0].shape, values["isect_offsets"].shape[:-2] + (8, 8, 3)
        )

    def test_packed_single_camera_background(self):
        for name in RASTERIZERS:
            with self.subTest(rasterizer=name):
                self.assert_success(name, inputs(name, (1,), packed=True))

    def test_packed_four_camera_background(self):
        for name in RASTERIZERS:
            with self.subTest(rasterizer=name):
                self.assert_success(name, inputs(name, (4,), packed=True))

    def test_empty_packed_gaussians_keep_camera_axes(self):
        for name in RASTERIZERS:
            for cameras in (1, 4):
                with self.subTest(rasterizer=name, cameras=cameras):
                    self.assert_success(
                        name, inputs(name, (cameras,), packed=True, nnz=0)
                    )

    def test_packed_background_requires_camera_axis(self):
        for name in RASTERIZERS:
            for cameras in (1, 4):
                with self.subTest(rasterizer=name, cameras=cameras):
                    rasterizer, probe = load_rasterizer(name)
                    with self.assertRaises(AssertionError):
                        rasterizer(**inputs(
                            name, (cameras,), packed=True, background_shape=(3,)
                        ))
                    self.assertEqual(probe.calls, [])

    def test_packed_background_camera_count_must_match(self):
        for name in RASTERIZERS:
            with self.subTest(rasterizer=name):
                rasterizer, probe = load_rasterizer(name)
                with self.assertRaises(AssertionError):
                    rasterizer(**inputs(
                        name, (4,), packed=True, background_shape=(1, 3)
                    ))
                self.assertEqual(probe.calls, [])

    def test_dense_image_axes_are_preserved(self):
        for name in RASTERIZERS:
            for image_dims in ((), (1,), (4,), (2, 4)):
                with self.subTest(rasterizer=name, image_dims=image_dims):
                    self.assert_success(name, inputs(name, image_dims, packed=False))

    def test_dense_background_still_requires_image_axes(self):
        for name in RASTERIZERS:
            with self.subTest(rasterizer=name):
                rasterizer, probe = load_rasterizer(name)
                with self.assertRaises(AssertionError):
                    rasterizer(**inputs(
                        name, (4,), packed=False, background_shape=(3,)
                    ))
                self.assertEqual(probe.calls, [])


if __name__ == "__main__":
    unittest.main()
