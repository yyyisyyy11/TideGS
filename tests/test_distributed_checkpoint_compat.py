import json
import tempfile
import types
import unittest
from pathlib import Path

from storage.distributed_checkpoint import load_distributed_checkpoint_manifest


class DistributedCheckpointCompatTest(unittest.TestCase):
    def _write_checkpoint(self, root, root_manifest):
        checkpoint = Path(root)
        rank_dir = checkpoint / "rank_0"
        rank_dir.mkdir(parents=True)
        (rank_dir / "pure_ssd_checkpoint.json").write_text(
            json.dumps(
                {
                    "checkpoint_type": "pure_ssd_incremental",
                    "base_file": "base.bin",
                    "total_points": 16,
                    "num_blocks": 4,
                    "block_size": 4,
                    "param_dim": 59,
                    "next_iteration": 33,
                }
            ),
            encoding="utf-8",
        )
        (checkpoint / "block_owner.npy").write_bytes(b"owner")
        (checkpoint / "block_bounds.npy").write_bytes(b"bounds")
        (checkpoint / "pure_ssd_distributed_checkpoint.json").write_text(
            json.dumps(root_manifest),
            encoding="utf-8",
        )
        return checkpoint

    def test_v1_capacity_is_converted_to_global_without_owner_reassignment(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._write_checkpoint(
                directory,
                {
                    "checkpoint_version": 1,
                    "world_size": 4,
                    "global_bsz": 32,
                    "next_iteration": 33,
                    "rank_manifests": ["rank_0"] * 4,
                    "block_owner": "block_owner.npy",
                    "block_bounds": "block_bounds.npy",
                    "per_rank_capacity_blocks": 2048,
                    "owner_policy": "visibility_weighted_lpt",
                },
            )
            manifest = load_distributed_checkpoint_manifest(
                checkpoint,
                rank=0,
                world_size=4,
                global_bsz=32,
            )
            self.assertEqual(manifest["_tide_global_capacity_blocks"], 8192)
            self.assertEqual(
                manifest["_tide_owner_policy"],
                "visibility_weighted_lpt",
            )
            self.assertEqual(
                manifest["_tide_block_owner_file"],
                str((checkpoint / "block_owner.npy").resolve()),
            )

    def test_v2_uses_explicit_global_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._write_checkpoint(
                directory,
                {
                    "checkpoint_version": 2,
                    "world_size": 4,
                    "global_bsz": 32,
                    "next_iteration": 33,
                    "rank_manifests": ["rank_0"] * 4,
                    "block_owner": "block_owner.npy",
                    "block_bounds": "block_bounds.npy",
                    "global_capacity_blocks": 8192,
                    "owner_policy": "stable_round_robin",
                },
            )
            manifest = load_distributed_checkpoint_manifest(
                checkpoint,
                rank=0,
                world_size=4,
                global_bsz=32,
            )
            self.assertEqual(manifest["_tide_global_capacity_blocks"], 8192)
            self.assertEqual(manifest["_tide_owner_policy"], "stable_round_robin")

    def test_v2_loader_rejects_3dgs2_tr_resume_but_accepts_adam(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._write_checkpoint(
                directory,
                {
                    "checkpoint_version": 2,
                    "world_size": 1,
                    "global_bsz": 8,
                    "next_iteration": 17,
                    "rank_manifests": ["rank_0"],
                    "block_owner": "block_owner.npy",
                    "block_bounds": "block_bounds.npy",
                    "global_capacity_blocks": 5,
                },
            )
            sophia_args = types.SimpleNamespace(
                paper_optimizer_algorithm="3dgs2_tr"
            )
            with self.assertRaisesRegex(ValueError, "only resume with Adam"):
                load_distributed_checkpoint_manifest(
                    checkpoint,
                    rank=0,
                    world_size=1,
                    global_bsz=8,
                    args=sophia_args,
                )

            load_distributed_checkpoint_manifest(
                checkpoint,
                rank=0,
                world_size=1,
                global_bsz=8,
                args=types.SimpleNamespace(paper_optimizer_algorithm="adam"),
            )


if __name__ == "__main__":
    unittest.main()
