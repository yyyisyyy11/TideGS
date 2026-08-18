import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence
import unittest

from strategies.tide_engine.distributed_metrics import (
    GRAD_NEAR_ZERO_THRESHOLDS,
    GRAD_ZERO_VALUE_FIELDS,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None


ENGINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "strategies"
    / "tide_engine"
    / "distributed_engine.py"
)


def _load_touched_component_rows():
    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_touched_component_rows"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Dict": Dict,
        "torch": torch,
        "GRAD_NEAR_ZERO_THRESHOLDS": GRAD_NEAR_ZERO_THRESHOLDS,
        "GRAD_ZERO_VALUE_FIELDS": GRAD_ZERO_VALUE_FIELDS,
    }
    exec(compile(module, str(ENGINE_PATH), "exec"), namespace)
    return namespace["_touched_component_rows"]


def _load_local_ids_for_blocks():
    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_local_ids_for_blocks"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Sequence": Sequence, "torch": torch}
    exec(compile(module, str(ENGINE_PATH), "exec"), namespace)
    return namespace["_local_ids_for_blocks"]


def _load_gradient_zero_helpers():
    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"))
    names = {
        "_LEAF_NAMES",
        "_LEAF_WIDTHS",
        "_PARAMETERS_PER_GAUSSIAN",
        "_mark_projection_cull_survivors",
        "_projection_cull_gradient_zero_stats",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.FunctionDef)):
            targets = (
                [target.id for target in node.targets if isinstance(target, ast.Name)]
                if isinstance(node, ast.Assign)
                else [node.name]
            )
            if any(name in names for name in targets):
                nodes.append(node)
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Dict": Dict,
        "torch": torch,
        "GRAD_NEAR_ZERO_THRESHOLDS": GRAD_NEAR_ZERO_THRESHOLDS,
        "GRAD_ZERO_VALUE_FIELDS": GRAD_ZERO_VALUE_FIELDS,
    }
    exec(compile(module, str(ENGINE_PATH), "exec"), namespace)
    return (
        namespace["_mark_projection_cull_survivors"],
        namespace["_projection_cull_gradient_zero_stats"],
    )


@unittest.skipIf(torch is None, "torch is unavailable")
class DistributedOwnerActiveRowsTest(unittest.TestCase):
    def test_block_rows_follow_persistent_slot_order(self):
        local_ids_for_blocks = _load_local_ids_for_blocks()
        manager = SimpleNamespace(
            device=torch.device("cpu"),
            block_to_gpu_slice={
                1: slice(4, 8),
                2: slice(0, 4),
            },
        )

        rows = local_ids_for_blocks(manager, [1, 2, 1, 99])

        torch.testing.assert_close(
            rows,
            torch.arange(8, dtype=torch.long),
            rtol=0,
            atol=0,
        )

    def test_gradient_and_curvature_rows_are_selected_independently(self):
        _touched_component_rows = _load_touched_component_rows()

        gradient = {
            "xyz": torch.tensor(
                [[1.0, 0.0], [0.0, 0.0], [float("nan"), 0.0], [0.0, 0.0]]
            ),
            "opacity": torch.tensor([[0.0], [0.0], [0.0], [-2.0]]),
        }
        curvature = {
            "xyz": torch.tensor(
                [[0.0, 0.0], [3.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            )
        }

        gradient_rows = _touched_component_rows(
            gradient, 4, torch.device("cpu")
        )
        curvature_rows = _touched_component_rows(
            curvature, 4, torch.device("cpu")
        )

        torch.testing.assert_close(
            gradient_rows,
            torch.tensor([0, 2, 3], dtype=torch.long),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            curvature_rows,
            torch.tensor([1], dtype=torch.long),
            rtol=0,
            atol=0,
        )

    def test_empty_owner_ignores_collective_sentinel_row(self):
        _touched_component_rows = _load_touched_component_rows()

        sentinel = {
            "opacity": torch.tensor([[float("nan")]]),
        }
        rows = _touched_component_rows(sentinel, 0, torch.device("cpu"))
        self.assertEqual(int(rows.numel()), 0)

    def test_nonempty_owner_rejects_wrong_component_shape(self):
        _touched_component_rows = _load_touched_component_rows()

        with self.assertRaisesRegex(RuntimeError, "invalid leading rows"):
            _touched_component_rows(
                {"xyz": torch.zeros((2, 3))},
                3,
                torch.device("cpu"),
            )

    def test_projection_cull_gradient_zero_stats_are_exact_and_row_grouped(self):
        mark_survivors, gradient_zero_stats = _load_gradient_zero_helpers()
        mask = torch.zeros((4,), dtype=torch.bool)
        mark_survivors(
            mask,
            {
                "gaussian_ids": torch.tensor([0, 2, 2, 3]),
                "radii": torch.tensor([1, 0, 2, 3]),
            },
            active_count=4,
        )
        self.assertEqual(mask.tolist(), [True, False, True, True])

        components = {
            "xyz": torch.tensor([
                [0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0],
                [1e-9, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]),
            "opacity": torch.tensor([[0.0], [5.0], [0.0], [1.0]]),
            "scaling": torch.tensor([
                [0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0],
                [0.0, 2.0, 0.0],
                [1.0, 1.0, 1.0],
            ]),
            "rotation": torch.tensor([
                [0.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0, 5.0],
                [0.0, 0.0, 0.0, 3.0],
                [1.0, 1.0, 1.0, 1.0],
            ]),
            "features_dc": torch.tensor([
                [0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]),
            "features_rest": torch.stack([
                torch.zeros((15, 3)),
                torch.full((15, 3), 5.0),
                torch.cat([torch.ones(4), torch.zeros(41)]).reshape(15, 3),
                torch.ones((15, 3)),
            ]),
        }

        profile = gradient_zero_stats(
            components,
            mask,
            near_zero_threshold=1e-8,
            sample_rows=2,
            chunk_rows=2,
        )
        stats = dict(zip(GRAD_ZERO_VALUE_FIELDS, profile["values"].tolist()))
        self.assertEqual(stats["projection_cull_unique_gaussians"], 3)
        self.assertEqual(stats["projection_cull_parameter_elements"], 177)
        self.assertEqual(stats["projection_cull_zero_gradient_elements"], 111)
        self.assertEqual(stats["projection_cull_nonfinite_gradient_elements"], 0)
        self.assertEqual(stats["projection_cull_all_zero_gradient_gaussians"], 1)
        self.assertEqual(stats["raw_sampled_gaussians"], 2)
        self.assertEqual(stats["projection_cull_xyz_zero_gradient_elements"], 5)
        self.assertEqual(stats["projection_cull_features_rest_zero_gradient_elements"], 86)
        self.assertEqual(stats["projection_cull_abs_le_1em10_gradient_elements"], 111)
        self.assertEqual(stats["projection_cull_abs_le_1em8_gradient_elements"], 112)
        self.assertEqual(stats["projection_cull_xyz_abs_le_1em8_gradient_elements"], 6)
        self.assertEqual(stats["projection_cull_zero_gradient_elements_0_gaussians"], 1)
        self.assertEqual(stats["projection_cull_zero_gradient_elements_52_gaussians"], 1)
        self.assertEqual(stats["projection_cull_zero_gradient_elements_59_gaussians"], 1)
        self.assertEqual(stats["projection_cull_near_zero_elements_0_gaussians"], 1)
        self.assertEqual(stats["projection_cull_near_zero_elements_53_gaussians"], 1)
        self.assertEqual(stats["projection_cull_near_zero_elements_59_gaussians"], 1)
        self.assertEqual(tuple(profile["sample_gradients"].shape), (2, 59))
        self.assertEqual(profile["sample_owner_rows"].tolist(), [0, 2])

    def test_projection_cull_gradient_zero_stats_handles_empty_owner(self):
        _, gradient_zero_stats = _load_gradient_zero_helpers()
        values = gradient_zero_stats({}, torch.zeros((0,), dtype=torch.bool))["values"]
        self.assertEqual(tuple(values.shape), (len(GRAD_ZERO_VALUE_FIELDS),))
        self.assertEqual(int(values.sum()), 0)

    def test_projection_cull_gradient_sample_can_include_every_survivor(self):
        _, gradient_zero_stats = _load_gradient_zero_helpers()
        mask = torch.tensor([True, False, True])
        components = {
            name: torch.arange(3 * width, dtype=torch.float32).reshape(3, width)
            for name, width in {
                "xyz": 3,
                "opacity": 1,
                "scaling": 3,
                "rotation": 4,
                "features_dc": 3,
                "features_rest": 45,
            }.items()
        }

        profile = gradient_zero_stats(components, mask, sample_rows=-1)

        self.assertEqual(profile["sample_owner_rows"].tolist(), [0, 2])
        self.assertEqual(tuple(profile["sample_gradients"].shape), (2, 59))


if __name__ == "__main__":
    unittest.main()
