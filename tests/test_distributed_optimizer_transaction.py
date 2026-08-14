import ast
import copy
from pathlib import Path
import unittest
from unittest import mock


ENGINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "strategies"
    / "tide_engine"
    / "distributed_engine.py"
)


def _is_sophia_branch(node):
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "optimizer_algorithm"
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "3dgs2_tr"
    )


def _is_mark_dirty(node):
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "mark_dirty_blocks"
    )


def _load_optimizer_transaction(namespace):
    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "train_distributed_tide_batch"
    )
    start = next(
        index for index, node in enumerate(function.body) if _is_sophia_branch(node)
    )
    stop = next(
        index
        for index, node in enumerate(function.body[start:], start=start)
        if _is_mark_dirty(node)
    )
    body = copy.deepcopy(function.body[start : stop + 1])
    body.append(
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id="checkpoint", ctx=ast.Load()),
                args=[],
                keywords=[],
            )
        )
    )
    wrapper = ast.FunctionDef(
        name="run_optimizer_transaction",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=body,
        decorator_list=[],
    )
    module = ast.Module(body=[wrapper], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(ENGINE_PATH), "exec"), namespace)
    return namespace["run_optimizer_transaction"]


class DistributedOptimizerTransactionTest(unittest.TestCase):
    def test_peer_prepare_failure_aborts_without_commit_dirty_or_checkpoint(self):
        prepared = object()
        optimizer = mock.Mock()
        optimizer.prepare_step.return_value = prepared
        state = mock.Mock()
        checkpoint = mock.Mock()
        synchronized_phases = []

        def synchronize(_context, *, phase, error):
            synchronized_phases.append((phase, error))
            if phase == "3DGS2-TR optimizer prepare":
                raise RuntimeError("rank=1 RuntimeError: peer prepare failed")

        namespace = {
            "optimizer_algorithm": "3dgs2_tr",
            "optimizer": optimizer,
            "_synchronize_rank_errors": synchronize,
            "context": object(),
            "iteration": 1,
            "gaussians": mock.Mock(),
            "gradient_local": object(),
            "sparse_grad_components": object(),
            "curvature_local": object(),
            "sparse_curvature_components": object(),
            "curvature_due": True,
            "optimizer_step": 1,
            "optimizer_end": mock.Mock(),
            "state": state,
            "checkpoint": checkpoint,
        }
        run_optimizer_transaction = _load_optimizer_transaction(namespace)

        with self.assertRaisesRegex(RuntimeError, "peer prepare failed"):
            run_optimizer_transaction()

        optimizer.prepare_step.assert_called_once()
        optimizer.abort_step.assert_called_once_with(prepared)
        optimizer.commit_step.assert_not_called()
        state.mark_dirty_blocks.assert_not_called()
        checkpoint.assert_not_called()
        self.assertEqual(
            synchronized_phases,
            [("3DGS2-TR optimizer prepare", None)],
        )


if __name__ == "__main__":
    unittest.main()
