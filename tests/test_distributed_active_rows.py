import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence
import unittest

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
    namespace = {"Dict": Dict, "torch": torch}
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


if __name__ == "__main__":
    unittest.main()
