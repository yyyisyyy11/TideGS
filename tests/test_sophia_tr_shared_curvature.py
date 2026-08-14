import ast
from pathlib import Path
import unittest

import torch

from strategies.tide_engine.sophia_tr_curvature import (
    build_fused_3dgs2_curvature_residuals,
    estimate_seeded_curvature_sample,
)


ROOT = Path(__file__).resolve().parents[1]


class SharedResidualTest(unittest.TestCase):
    @staticmethod
    def _ssim_map(image, target):
        return 1.0 - 0.25 * (image - target).square()

    def test_camera_chunking_does_not_change_residual_normalization(self):
        image = torch.tensor(
            [
                [[[0.0, 0.5], [0.25, 1.0]]],
                [[[0.8, 0.1], [0.4, 0.6]]],
            ],
            dtype=torch.float64,
        )
        target = torch.tensor(
            [
                [[[0.5, 0.25], [0.25, 0.5]]],
                [[[0.2, 0.1], [0.9, 0.3]]],
            ],
            dtype=torch.float64,
        )
        weight = 0.2

        chunked = build_fused_3dgs2_curvature_residuals(
            image,
            target,
            weight,
            ssim_map_builder=self._ssim_map,
        )
        separate = tuple(
            build_fused_3dgs2_curvature_residuals(
                image[index],
                target[index],
                weight,
                ssim_map_builder=self._ssim_map,
            )[0]
            for index in range(2)
        )

        self.assertEqual(len(chunked), 2)
        for actual, expected in zip(chunked, separate):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_loss = 0.5 * sum(value.square().sum() for value in chunked)
        ssim_map = self._ssim_map(image, target)
        expected_loss = sum(
            (1.0 - weight) * (image[index] - target[index]).abs().mean()
            + weight * (1.0 - ssim_map[index]).mean()
            for index in range(2)
        )
        torch.testing.assert_close(actual_loss, expected_loss)


class SharedProbeTest(unittest.TestCase):
    def test_seeded_sample_applies_exact_inverse_sample_count(self):
        parameters = torch.tensor(
            [0.25, -0.5, 1.5], dtype=torch.float64, requires_grad=True
        )
        matrix = torch.tensor(
            [[1.0, 2.0, -1.0], [0.5, -1.0, 3.0]], dtype=torch.float64
        )
        residuals = (matrix @ parameters,)
        kwargs = {
            "base_seed": 37,
            "optimizer_step": 11,
            "microbatch_index": 2,
            "sample_index": 0,
            "rank": 1,
        }

        unscaled = estimate_seeded_curvature_sample(
            residuals, (parameters,), sample_count=1, **kwargs
        )
        scaled = estimate_seeded_curvature_sample(
            residuals, (parameters,), sample_count=4, **kwargs
        )

        torch.testing.assert_close(scaled[0], unscaled[0] / 4.0, rtol=0, atol=0)
        repeated = estimate_seeded_curvature_sample(
            residuals, (parameters,), sample_count=4, **kwargs
        )
        torch.testing.assert_close(repeated[0], scaled[0], rtol=0, atol=0)


class SharedRuntimeContractTest(unittest.TestCase):
    def test_single_and_distributed_paths_call_shared_helpers(self):
        expected_calls = {
            "build_fused_3dgs2_curvature_residuals",
            "estimate_seeded_curvature_sample",
        }
        for relative_path, function_name in (
            ("strategies/tide_engine/engine.py", "_run_single_rank_sophia_curvature_batch"),
            ("strategies/tide_engine/distributed_engine.py", "train_distributed_tide_batch"),
        ):
            tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
            function = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == function_name
            )
            calls = {
                node.func.id
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            self.assertEqual(expected_calls.difference(calls), set())


if __name__ == "__main__":
    unittest.main()
