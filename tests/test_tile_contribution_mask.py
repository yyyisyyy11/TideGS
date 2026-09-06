import math
import random
import sys
import unittest
from pathlib import Path


ALPHA_THRESHOLD = 1.0 / 255.0


def _clamp(value, lower, upper):
    return min(max(value, lower), upper)


def _quadratic(x, y, conic):
    a, b, c = conic
    return a * x * x + 2.0 * b * x * y + c * y * y


def _rectangle_min_quadratic(x0, x1, y0, y1, conic):
    a, b, c = conic
    if x0 <= 0.0 <= x1 and y0 <= 0.0 <= y1:
        return 0.0
    values = [
        _quadratic(x0, y0, conic),
        _quadratic(x0, y1, conic),
        _quadratic(x1, y0, conic),
        _quadratic(x1, y1, conic),
    ]
    if c > 0.0:
        values.append(_quadratic(x0, _clamp(-b * x0 / c, y0, y1), conic))
        values.append(_quadratic(x1, _clamp(-b * x1 / c, y0, y1), conic))
    if a > 0.0:
        values.append(_quadratic(_clamp(-b * y0 / a, x0, x1), y0, conic))
        values.append(_quadratic(_clamp(-b * y1 / a, x0, x1), y1, conic))
    return max(min(values), 0.0)


def _reference_one(
    mean,
    conic,
    opacity,
    radius,
    image_width,
    image_height,
    tile_size=16,
    alpha_threshold=ALPHA_THRESHOLD,
):
    radius_x, radius_y = (
        (int(radius), int(radius))
        if isinstance(radius, int)
        else (int(radius[0]), int(radius[1]))
    )
    if radius_x <= 0 or radius_y <= 0:
        return False, 0, 0
    mean_x, mean_y = mean
    tile_width = math.ceil(image_width / tile_size)
    tile_height = math.ceil(image_height / tile_size)
    min_x = _clamp(math.floor((mean_x - radius_x) / tile_size), 0, tile_width)
    min_y = _clamp(math.floor((mean_y - radius_y) / tile_size), 0, tile_height)
    max_x = _clamp(math.ceil((mean_x + radius_x) / tile_size), 0, tile_width)
    max_y = _clamp(math.ceil((mean_y + radius_y) / tile_size), 0, tile_height)
    candidates = max(0, max_x - min_x) * max(0, max_y - min_y)
    if candidates == 0 or not opacity >= alpha_threshold:
        return False, candidates, 0

    contributing = 0
    for tile_y in range(min_y, max_y):
        pixel_y0 = tile_y * tile_size
        pixel_y1 = min(image_height, (tile_y + 1) * tile_size) - 1
        y0 = pixel_y0 + 0.5 - mean_y
        y1 = pixel_y1 + 0.5 - mean_y
        for tile_x in range(min_x, max_x):
            pixel_x0 = tile_x * tile_size
            pixel_x1 = min(image_width, (tile_x + 1) * tile_size) - 1
            x0 = pixel_x0 + 0.5 - mean_x
            x1 = pixel_x1 + 0.5 - mean_x
            distance = _rectangle_min_quadratic(x0, x1, y0, y1, conic)
            if opacity * math.exp(-0.5 * distance) >= alpha_threshold:
                contributing += 1
    return contributing > 0, candidates, contributing


class TileContributionReferenceTest(unittest.TestCase):
    def test_threshold_opacity_radius_and_image_boundaries(self):
        cases = (
            ((8.5, 8.5), (1.0, 0.0, 1.0), ALPHA_THRESHOLD, 4, 16, 16,
             (True, 1, 1)),
            ((16.0, 16.0), (0.1, 0.0, 0.1), ALPHA_THRESHOLD * 0.5,
             (17, 9), 32, 32, (False, 4, 0)),
            ((8.5, 8.5), (1.0, 0.0, 1.0), 1.0, 0, 16, 16,
             (False, 0, 0)),
            ((-100.0, -100.0), (1.0, 0.0, 1.0), 1.0, 2, 16, 16,
             (False, 0, 0)),
            ((16.5, 16.5), (4.0, 0.0, 4.0), ALPHA_THRESHOLD,
             (1, 1), 17, 17, (True, 4, 1)),
        )
        for *inputs, expected in cases:
            with self.subTest(inputs=inputs):
                self.assertEqual(_reference_one(*inputs), expected)

    def test_non_diagonal_rotated_conic_finds_edge_minimum(self):
        angle = math.radians(37.0)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        lambda_x, lambda_y = 0.04, 1.7
        conic = (
            lambda_x * cosine * cosine + lambda_y * sine * sine,
            (lambda_x - lambda_y) * sine * cosine,
            lambda_x * sine * sine + lambda_y * cosine * cosine,
        )
        minimum = _rectangle_min_quadratic(2.0, 7.0, -4.0, 5.0, conic)
        edge_minimum = min(
            _quadratic(2.0, y / 1000.0, conic)
            for y in range(-4000, 5001)
        )
        self.assertAlmostEqual(minimum, edge_minimum, places=6)

    def test_continuous_rectangle_test_never_drops_a_contributing_pixel(self):
        generator = random.Random(19)
        for _ in range(100):
            width = generator.randint(5, 33)
            height = generator.randint(5, 33)
            mean = (
                generator.uniform(-3.0, width + 3.0),
                generator.uniform(-3.0, height + 3.0),
            )
            angle = generator.uniform(0.0, math.pi)
            cosine, sine = math.cos(angle), math.sin(angle)
            lambda_x = generator.uniform(0.01, 0.2)
            lambda_y = generator.uniform(0.2, 2.0)
            conic = (
                lambda_x * cosine * cosine + lambda_y * sine * sine,
                (lambda_x - lambda_y) * sine * cosine,
                lambda_x * sine * sine + lambda_y * cosine * cosine,
            )
            opacity = generator.uniform(ALPHA_THRESHOLD, 1.0)
            radius = (generator.randint(1, 12), generator.randint(1, 12))
            keep, _, _ = _reference_one(
                mean, conic, opacity, radius, width, height
            )
            x0 = max(0, math.floor((mean[0] - radius[0]) / 16) * 16)
            y0 = max(0, math.floor((mean[1] - radius[1]) / 16) * 16)
            x1 = min(width, math.ceil((mean[0] + radius[0]) / 16) * 16)
            y1 = min(height, math.ceil((mean[1] + radius[1]) / 16) * 16)
            discrete_contribution = any(
                opacity
                * math.exp(
                    -0.5
                    * _quadratic(
                        pixel_x + 0.5 - mean[0],
                        pixel_y + 0.5 - mean[1],
                        conic,
                    )
                )
                >= ALPHA_THRESHOLD
                for pixel_y in range(y0, y1)
                for pixel_x in range(x0, x1)
            )
            self.assertFalse(discrete_contribution and not keep)


class TileContributionCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ModuleNotFoundError as error:
            raise unittest.SkipTest("torch is unavailable") from error
        if not hasattr(torch, "cuda") or not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")
        package_root = Path(__file__).resolve().parents[1] / "submodules" / "clm_kernels"
        sys.path.insert(0, str(package_root))
        try:
            from clm_kernels import tile_contribution_mask
        except (ImportError, OSError) as error:
            raise unittest.SkipTest("clm_kernels CUDA extension is unavailable") from error
        cls.torch = torch
        cls.mask = staticmethod(tile_contribution_mask)

    def test_cuda_matches_cpu_reference_for_edge_cases_and_random_inputs(self):
        generator = random.Random(7)
        cases = [
            ((8.5, 8.5), (1.0, 0.0, 1.0), ALPHA_THRESHOLD, (4, 4)),
            ((16.0, 16.0), (0.1, 0.0, 0.1), ALPHA_THRESHOLD * 0.5, (17, 9)),
            ((16.5, 16.5), (4.0, 0.0, 4.0), ALPHA_THRESHOLD, (1, 1)),
            ((8.5, 8.5), (1.0, 0.4, 0.5), 0.75, (0, 7)),
        ]
        for _ in range(64):
            a = generator.uniform(0.05, 1.0)
            c = generator.uniform(0.05, 1.0)
            b = generator.uniform(-0.8, 0.8) * math.sqrt(a * c)
            cases.append(
                (
                    (generator.uniform(-4, 36), generator.uniform(-4, 36)),
                    (a, b, c),
                    generator.uniform(0.001, 1.0),
                    (generator.randint(0, 14), generator.randint(0, 14)),
                )
            )
        means, conics, opacities, radii = zip(*cases)
        torch = self.torch
        outputs = self.mask(
            torch.tensor(means, dtype=torch.float32, device="cuda"),
            torch.tensor(conics, dtype=torch.float32, device="cuda"),
            torch.tensor(opacities, dtype=torch.float32, device="cuda"),
            torch.tensor(radii, dtype=torch.int32, device="cuda"),
            32,
            32,
            16,
            ALPHA_THRESHOLD,
        )
        actual = [value.cpu().tolist() for value in outputs]
        expected = list(
            zip(
                *[
                    _reference_one(mean, conic, opacity, radius, 32, 32)
                    for mean, conic, opacity, radius in zip(
                        torch.tensor(means, dtype=torch.float32).tolist(),
                        torch.tensor(conics, dtype=torch.float32).tolist(),
                        torch.tensor(opacities, dtype=torch.float32).tolist(),
                        radii,
                    )
                ]
            )
        )
        self.assertEqual(actual[0], list(expected[0]))
        self.assertEqual(actual[1], list(expected[1]))
        self.assertEqual(actual[2], list(expected[2]))

    def test_empty_input(self):
        torch = self.torch
        keep, candidates, contributing = self.mask(
            torch.empty((0, 2), dtype=torch.float32, device="cuda"),
            torch.empty((0, 3), dtype=torch.float32, device="cuda"),
            torch.empty((0,), dtype=torch.float32, device="cuda"),
            torch.empty((0, 2), dtype=torch.int32, device="cuda"),
            32,
            32,
        )
        self.assertEqual(tuple(keep.shape), (0,))
        self.assertEqual(tuple(candidates.shape), (0,))
        self.assertEqual(tuple(contributing.shape), (0,))

    def test_all_masked_camera_matches_background_and_has_zero_gradients(self):
        try:
            from gsplat.cuda._wrapper import (
                isect_offset_encode,
                isect_tiles,
                rasterize_to_pixels,
            )
        except (ImportError, OSError) as error:
            self.skipTest(f"gsplat 1.5.3 CUDA extension is unavailable: {error}")

        torch = self.torch
        background = torch.tensor([[0.2, 0.3, 0.4]], device="cuda")

        def render(mode):
            def variable(value):
                return torch.tensor(
                    value, dtype=torch.float32, device="cuda", requires_grad=True
                )

            means = variable([[[18.0, 8.5]]])
            conics = variable([[[4.0, 0.0, 4.0]]])
            colors = variable([[[0.8, 0.1, 0.6]]])
            opacities = variable([[0.01]])
            radii = torch.tensor([[[4, 4]]], dtype=torch.int32, device="cuda")
            if mode != "off":
                keep, _, _ = self.mask(
                    means,
                    conics,
                    opacities,
                    radii,
                    16,
                    16,
                    16,
                    ALPHA_THRESHOLD,
                )
                self.assertFalse(bool(keep.item()))
                if mode == "apply":
                    radii = torch.where(
                        keep.unsqueeze(-1), radii, torch.zeros_like(radii)
                    )
            tiles, isect_ids, flatten_ids = isect_tiles(
                means,
                radii,
                torch.ones((1, 1), device="cuda"),
                16,
                1,
                1,
                packed=False,
            )
            offsets = isect_offset_encode(isect_ids, 1, 1, 1)
            rendered, alphas = rasterize_to_pixels(
                means,
                conics,
                colors,
                opacities,
                16,
                16,
                16,
                offsets,
                flatten_ids,
                backgrounds=background,
                packed=False,
            )
            loss = rendered.square().sum() + alphas.square().sum()
            gradients = torch.autograd.grad(loss, (means, conics, colors, opacities))
            return rendered, alphas, loss, gradients, tiles

        baseline = render("off")
        for mode in ("profile", "apply"):
            result = render(mode)
            torch.testing.assert_close(result[0], baseline[0], rtol=0, atol=0)
            torch.testing.assert_close(result[1], baseline[1], rtol=0, atol=0)
            torch.testing.assert_close(result[2], baseline[2], rtol=0, atol=0)
            for gradient, baseline_gradient in zip(result[3], baseline[3]):
                torch.testing.assert_close(
                    gradient, baseline_gradient, rtol=0, atol=0
                )
                self.assertEqual(int(torch.count_nonzero(gradient)), 0)
        torch.testing.assert_close(
            baseline[0],
            background[:, None, None, :].expand(1, 16, 16, 3),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
