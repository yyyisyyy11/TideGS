import ast
from pathlib import Path
import unittest
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "storage" / "distributed_checkpoint.py"
)


class DistributedCheckpointSetupContractTest(unittest.TestCase):
    def test_directory_creation_error_is_gathered_before_first_barrier(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "write_distributed_incremental_checkpoint"
        )
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        gather_lines = [
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "all_gather_object"
        ]
        barrier_lines = [
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "barrier"
        ]
        mkdir_lines = [
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "mkdir"
        ]

        self.assertTrue(mkdir_lines)
        self.assertTrue(gather_lines)
        self.assertTrue(barrier_lines)
        self.assertLess(mkdir_lines[0], gather_lines[0])
        self.assertLess(gather_lines[0], barrier_lines[0])

    def test_rank0_directory_failure_stops_all_ranks_before_barrier(self):
        try:
            from storage.distributed_checkpoint import (
                write_distributed_incremental_checkpoint,
            )
        except ModuleNotFoundError as error:
            self.skipTest(f"runtime dependency unavailable: {error}")

        failure = {
            "ok": False,
            "rank": 0,
            "error_type": "OSError",
            "error_message": "disk unavailable",
        }

        class Context:
            world_size = 2

            def __init__(self, rank):
                self.rank = rank
                self.is_rank0 = rank == 0
                self.barrier_calls = 0

            def all_gather_object(self, local):
                peer = {"ok": True, "rank": 1}
                return [failure, peer]

            def barrier(self):
                self.barrier_calls += 1

        for rank in (0, 1):
            context = Context(rank)
            mkdir_patch = (
                mock.patch.object(Path, "mkdir", side_effect=OSError("disk unavailable"))
                if rank == 0
                else mock.patch.object(Path, "mkdir")
            )
            with mkdir_patch, self.assertRaisesRegex(
                RuntimeError, "directory setup failed"
            ):
                write_distributed_incremental_checkpoint(
                    context=context,
                    storage_adapter=None,
                    gaussians=None,
                    checkpoint_dir="unused",
                    iteration=1,
                    next_iteration=2,
                    args=None,
                    block_owner=None,
                )
            self.assertEqual(context.barrier_calls, 0)


if __name__ == "__main__":
    unittest.main()
