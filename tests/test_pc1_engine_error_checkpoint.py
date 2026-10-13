"""Ensure fatal publication precedes a potentially blocking rank checkpoint."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest


def load_checkpoint():
    path = Path(__file__).parents[1] / "strategies/tide_engine/distributed_engine.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"_error_description", "_synchronize_rank_errors"}]
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0,
    ), *nodes], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["_synchronize_rank_errors"]


class PC1EngineErrorCheckpointTest(unittest.TestCase):
    def test_local_error_is_published_without_entering_gloo(self):
        calls = []

        def fail(phase, error):
            calls.append((phase, error))
            raise RuntimeError("fatal worker exit")

        def gather(*args, **kwargs):
            self.fail("must not wait for a peer collective after local failure")

        context = SimpleNamespace(
            rank=0, pc1_error_controller=SimpleNamespace(fail=fail),
            all_gather_object=gather,
        )
        error = ValueError("C1 payload failed")
        with self.assertRaisesRegex(RuntimeError, "fatal worker exit"):
            load_checkpoint()(context, phase="C1", error=error, group=object())
        self.assertEqual(calls, [("C1", error)])

    def test_healthy_gloo_wait_itself_has_a_deadline(self):
        calls = []
        group = object()

        @contextmanager
        def operation(phase):
            calls.append(("enter", phase))
            yield
            calls.append(("exit", phase))

        def gather(payload, *, group):
            calls.append(("gather", group))
            return [payload, {"rank": 1, "error": None}]

        context = SimpleNamespace(
            rank=0, pc1_error_controller=SimpleNamespace(operation=operation),
            all_gather_object=gather,
        )
        load_checkpoint()(context, phase="forward", error=None, group=group)
        self.assertEqual(calls, [("enter", "forward"), ("gather", group), ("exit", "forward")])


if __name__ == "__main__":
    unittest.main()
