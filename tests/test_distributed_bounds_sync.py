import ast
from pathlib import Path
from typing import Optional, Sequence
import unittest


ENGINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "strategies"
    / "tide_engine"
    / "distributed_engine.py"
)


def _function_node(name):
    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing function {name}")


class _Array:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class _NumpyStub:
    float32 = "float32"

    @staticmethod
    def asarray(value, dtype=None):
        del dtype
        if isinstance(value, _Array):
            return value
        return _Array(value)


class _TorchStub:
    @staticmethod
    def is_tensor(value):
        del value
        return False


def _load_sync_bounds():
    nodes = [
        _function_node("_error_description"),
        _function_node("_synchronize_rank_errors"),
        _function_node("_sync_bounds"),
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "DistributedContext": object,
        "Optional": Optional,
        "Sequence": Sequence,
        "np": _NumpyStub,
        "torch": _TorchStub,
    }
    exec(compile(module, str(ENGINE_PATH), "exec"), namespace)
    return namespace["_sync_bounds"]


class _PendingBounds:
    def wait(self):
        return [7], [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]


class _Manager:
    def stage_block_bounds(self, block_ids):
        if block_ids != [7]:
            raise AssertionError(f"unexpected updated blocks: {block_ids}")
        return _PendingBounds()


class _ScriptedContext:
    world_size = 2

    def __init__(self, rank, *, application_failure=True):
        self.rank = rank
        self.application_failure = application_failure
        self.gathered = []

    def all_gather_object(self, value):
        self.gathered.append(value)
        call = len(self.gathered)
        if call == 1:
            return [
                {"rank": 0, "error": None},
                {"rank": 1, "error": None},
            ]
        if call == 2:
            return [
                ([7], [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]),
                ([], []),
            ]
        if call == 3:
            rank0_error = (
                "OSError: rank 0 bounds apply failed"
                if self.application_failure
                else None
            )
            return [
                {"rank": 0, "error": rank0_error},
                {"rank": 1, "error": None},
            ]
        raise AssertionError(f"unexpected all_gather_object call {call}")


class _Adapter:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.updates = []

    def update_block_bounds(self, block_ids, bounds):
        self.updates.append((list(block_ids), bounds.tolist()))
        if self.fail:
            raise OSError("rank 0 bounds apply failed")


class DistributedBoundsSyncTest(unittest.TestCase):
    def test_apply_failure_is_synchronized_on_failing_and_peer_ranks(self):
        sync_bounds = _load_sync_bounds()

        for rank in (0, 1):
            with self.subTest(rank=rank):
                context = _ScriptedContext(rank)
                adapter = _Adapter(fail=rank == 0)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Distributed TideGS bounds application failed: "
                    "rank=0 OSError: rank 0 bounds apply failed",
                ) as raised:
                    sync_bounds(
                        context=context,
                        storage_adapter=adapter,
                        manager=_Manager(),
                        updated_blocks=[7],
                    )

                self.assertEqual(len(context.gathered), 3)
                self.assertEqual(len(adapter.updates), 1)
                if rank == 0:
                    self.assertIsInstance(raised.exception.__cause__, OSError)
                    self.assertEqual(
                        context.gathered[2]["error"],
                        "OSError: rank 0 bounds apply failed",
                    )
                else:
                    self.assertIsNone(raised.exception.__cause__)
                    self.assertIsNone(context.gathered[2]["error"])

    def test_successful_apply_still_participates_in_error_sync(self):
        sync_bounds = _load_sync_bounds()
        context = _ScriptedContext(rank=0, application_failure=False)
        adapter = _Adapter()

        sync_bounds(
            context=context,
            storage_adapter=adapter,
            manager=_Manager(),
            updated_blocks=[7],
        )

        self.assertEqual(len(context.gathered), 3)
        self.assertEqual(context.gathered[2], {"rank": 0, "error": None})
        self.assertEqual(
            adapter.updates,
            [([7], [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])],
        )


if __name__ == "__main__":
    unittest.main()
