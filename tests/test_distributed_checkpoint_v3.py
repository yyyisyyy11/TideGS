import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from storage.distributed_checkpoint import (
    DISTRIBUTED_CHECKPOINT_MANIFEST,
    load_distributed_checkpoint_manifest,
    load_distributed_checkpoint_root,
    planner_state_from_distributed_manifest,
    write_distributed_incremental_checkpoint,
)
from storage.pure_ssd_checkpoint import CHECKPOINT_MANIFEST
from strategies.tide_engine.checkpoint_validation import build_optimizer_provenance


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _args(**overrides):
    values = {
        "paper_optimizer_algorithm": "3dgs2_tr",
        "bsz": 8,
        "iterations": 100,
        "paper_optimizer_backend": "gpu_resident",
        "paper_optimizer_state_mode": "resident_blocks",
        "paper_sophia_beta1": 0.9,
        "paper_sophia_beta2": 0.999,
        "paper_sophia_curvature_interval": 10,
        "paper_sophia_hutchinson_samples": 1,
        "paper_sophia_curvature_estimator": "residual_vjp",
        "paper_sophia_gamma": 1.0,
        "paper_sophia_epsilon": 1e-15,
        "paper_sophia_curvature_seed": 7,
        "paper_tr_epsilon_init": 1e-6,
        "paper_tr_epsilon_final": 1e-8,
        "paper_tr_quat_norm": -1.0,
        "paper_resident_capacity_blocks": 5,
        "paper_resident_selection_policy": "topc_balanced",
        "paper_resident_lambda": 0.3,
        "paper_resident_recency_decay": 0.95,
        "paper_balanced_seed_fraction": 0.25,
        "lambda_dssim": 0.2,
        "ssd_schedule_ordering": "trajectory",
        "tide_camera_assignment": "equal",
        "tide_camera_microbatch": 2,
        "_tide_owner_policy": "stable_round_robin",
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _write_rank_manifest(rank_dir):
    rank_dir = Path(rank_dir)
    rank_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint_type": "pure_ssd_incremental",
        "base_file": "base.bin",
        "total_points": 8,
        "num_blocks": 2,
        "block_size": 4,
        "param_dim": 59,
        "next_iteration": 17,
    }
    manifest_file = rank_dir / CHECKPOINT_MANIFEST
    manifest_file.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest


def _write_v3_checkpoint(directory):
    checkpoint = Path(directory)
    rank_dir = checkpoint / "rank_0"
    _write_rank_manifest(rank_dir)
    owner_file = checkpoint / "block_owner.npy"
    bounds_file = checkpoint / "block_bounds.npy"
    np.save(owner_file, np.asarray([0, 0], dtype=np.int32))
    np.save(bounds_file, np.zeros((2, 6), dtype=np.float32))
    root = {
        "checkpoint_version": 3,
        "checkpoint_type": "pure_ssd_distributed_incremental",
        "iteration": 9,
        "next_iteration": 17,
        "world_size": 1,
        "global_bsz": 8,
        "global_capacity_blocks": 5,
        "capacity_semantics": "global",
        "owner_policy": "stable_round_robin",
        "block_version_semantics": "monotonic_gpu_cpu_ssd",
        "camera_assignment": "equal",
        "camera_microbatch": 2,
        "block_owner": owner_file.name,
        "block_owner_sha256": _sha256(owner_file),
        "block_bounds": bounds_file.name,
        "rank_manifests": ["rank_0"],
        "rank_manifest_sha256": [_sha256(rank_dir / CHECKPOINT_MANIFEST)],
        "optimizer_provenance": build_optimizer_provenance(_args()),
        "global_optimizer_step": 2,
        "optimizer_state_mode": "cold_start_per_rank",
        "optimizer_ema_saved": False,
        "optimizer_curvature_saved": False,
        "planner_state": {
            "resident": [0, 1],
            "active": [1],
            "recency": {"0": 0.5, "1": 1.0},
        },
        "total_points": 8,
        "num_blocks": 2,
        "block_size": 4,
        "param_dim": 59,
    }
    (checkpoint / DISTRIBUTED_CHECKPOINT_MANIFEST).write_text(
        json.dumps(root, sort_keys=True),
        encoding="utf-8",
    )
    return checkpoint


class _SingleRankContext:
    rank = 0
    world_size = 1
    is_rank0 = True

    def barrier(self):
        return None

    def all_gather_object(self, value):
        return [value]

    def broadcast_object(self, value):
        return value


class DistributedCheckpointV3Test(unittest.TestCase):
    def test_future_manifest_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _write_v3_checkpoint(directory)
            root_file = checkpoint / DISTRIBUTED_CHECKPOINT_MANIFEST
            root = json.loads(root_file.read_text(encoding="utf-8"))
            root["checkpoint_version"] = 4
            root_file.write_text(json.dumps(root), encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError,
                "Unsupported distributed checkpoint version: 4",
            ):
                load_distributed_checkpoint_root(checkpoint)

    def test_v3_load_verifies_hashes_and_exposes_resume_state(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _write_v3_checkpoint(directory)
            root = load_distributed_checkpoint_root(checkpoint)
            state = planner_state_from_distributed_manifest(root)
            self.assertEqual(state["resident"], [0, 1])
            self.assertEqual(state["active"], [1])
            self.assertEqual(state["recency"], {0: 0.5, 1: 1.0})

            manifest = load_distributed_checkpoint_manifest(
                checkpoint,
                rank=0,
                world_size=1,
                global_bsz=8,
                args=_args(),
            )
            self.assertEqual(manifest["_tide_global_optimizer_step"], 2)
            self.assertEqual(
                manifest["_tide_optimizer_state_mode"],
                "cold_start_per_rank",
            )
            self.assertEqual(manifest["_tide_planner_state"], state)

    def test_v3_resume_rejects_cli_mismatch_without_overwriting_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _write_v3_checkpoint(directory)
            args = _args(paper_sophia_beta1=0.8)

            with self.assertRaisesRegex(ValueError, "paper_sophia_beta1"):
                load_distributed_checkpoint_manifest(
                    checkpoint,
                    rank=0,
                    world_size=1,
                    global_bsz=8,
                    args=args,
                )

            self.assertEqual(args.paper_sophia_beta1, 0.8)

    def test_v3_rejects_tampered_rank_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _write_v3_checkpoint(directory)
            rank_manifest = checkpoint / "rank_0" / CHECKPOINT_MANIFEST
            rank_manifest.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "rank 0 manifest SHA-256 mismatch"):
                load_distributed_checkpoint_root(checkpoint)

    def test_v3_rejects_tampered_owner_map(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _write_v3_checkpoint(directory)
            (checkpoint / "block_owner.npy").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "block_owner SHA-256 mismatch"):
                load_distributed_checkpoint_root(checkpoint)

    def test_v3_rejects_member_path_outside_checkpoint_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _write_v3_checkpoint(directory)
            root_file = checkpoint / DISTRIBUTED_CHECKPOINT_MANIFEST
            root = json.loads(root_file.read_text(encoding="utf-8"))
            root["block_owner"] = str((checkpoint / "block_owner.npy").resolve())
            root_file.write_text(json.dumps(root, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "must be a relative path"):
                load_distributed_checkpoint_root(checkpoint)

    def test_v3_rejects_rank_topology_mismatch_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _write_v3_checkpoint(directory)
            rank_manifest_file = checkpoint / "rank_0" / CHECKPOINT_MANIFEST
            rank_manifest = json.loads(
                rank_manifest_file.read_text(encoding="utf-8")
            )
            rank_manifest["total_points"] = 4
            rank_manifest_file.write_text(
                json.dumps(rank_manifest, sort_keys=True),
                encoding="utf-8",
            )
            root_file = checkpoint / DISTRIBUTED_CHECKPOINT_MANIFEST
            root = json.loads(root_file.read_text(encoding="utf-8"))
            root["rank_manifest_sha256"] = [_sha256(rank_manifest_file)]
            root_file.write_text(json.dumps(root, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "rank 0 total_points mismatch"):
                load_distributed_checkpoint_root(checkpoint)

    def test_v3_rejects_planner_state_over_global_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _write_v3_checkpoint(directory)
            root_file = checkpoint / DISTRIBUTED_CHECKPOINT_MANIFEST
            root = json.loads(root_file.read_text(encoding="utf-8"))
            root["global_capacity_blocks"] = 1
            root_file.write_text(json.dumps(root, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exceeds global capacity"):
                load_distributed_checkpoint_root(checkpoint)

    def test_writer_emits_v3_provenance_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"

            def fake_rank_writer(**kwargs):
                manifest = _write_rank_manifest(kwargs["checkpoint_dir"])
                manifest.update({"patch_files": 3, "patch_bytes": 4096})
                return manifest

            storage_adapter = types.SimpleNamespace(
                block_bounds=np.zeros((2, 6), dtype=np.float32),
                num_points=8,
                num_blocks=2,
                block_size=4,
            )
            with mock.patch(
                "storage.distributed_checkpoint.write_pure_ssd_incremental_checkpoint",
                side_effect=fake_rank_writer,
            ):
                root = write_distributed_incremental_checkpoint(
                    context=_SingleRankContext(),
                    storage_adapter=storage_adapter,
                    gaussians=object(),
                    checkpoint_dir=checkpoint,
                    iteration=9,
                    next_iteration=17,
                    args=_args(),
                    block_owner=np.asarray([0, 0], dtype=np.int32),
                    planner_state={
                        "resident": [0, 1],
                        "active": [1],
                        "recency": {0: 0.5, 1: 1.0},
                    },
                    global_optimizer_step=2,
                )

            self.assertEqual(root["checkpoint_version"], 3)
            self.assertEqual(root["global_optimizer_step"], 2)
            self.assertEqual(root["optimizer_state_mode"], "cold_start_per_rank")
            self.assertFalse(root["optimizer_ema_saved"])
            self.assertEqual(root["rank_patch_files"], [3])
            self.assertFalse(Path(root["block_owner"]).is_absolute())
            self.assertFalse(Path(root["block_bounds"]).is_absolute())
            self.assertTrue(
                all(not Path(value).is_absolute() for value in root["rank_manifests"])
            )
            self.assertEqual(
                root["block_owner_sha256"],
                _sha256(checkpoint / root["block_owner"]),
            )
            self.assertEqual(
                root["rank_manifest_sha256"],
                [_sha256(checkpoint / "rank_0" / CHECKPOINT_MANIFEST)],
            )
            load_distributed_checkpoint_root(checkpoint)

    def test_writer_propagates_rank_failure_before_root_collectives(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "storage.distributed_checkpoint.write_pure_ssd_incremental_checkpoint",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "rank 0: OSError: disk full",
                ):
                    write_distributed_incremental_checkpoint(
                        context=_SingleRankContext(),
                        storage_adapter=types.SimpleNamespace(),
                        gaussians=object(),
                        checkpoint_dir=Path(directory) / "checkpoint",
                        iteration=9,
                        next_iteration=17,
                        args=_args(),
                        block_owner=np.asarray([0, 0], dtype=np.int32),
                        planner_state={
                            "resident": [],
                            "active": [],
                            "recency": {},
                        },
                    )

    def test_writer_propagates_root_manifest_validation_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            def fake_rank_writer(**kwargs):
                return _write_rank_manifest(kwargs["checkpoint_dir"])

            with mock.patch(
                "storage.distributed_checkpoint.write_pure_ssd_incremental_checkpoint",
                side_effect=fake_rank_writer,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "root manifest write failed.*planner_state",
                ):
                    write_distributed_incremental_checkpoint(
                        context=_SingleRankContext(),
                        storage_adapter=types.SimpleNamespace(
                            block_bounds=np.zeros((2, 6), dtype=np.float32),
                            num_points=8,
                            num_blocks=2,
                            block_size=4,
                        ),
                        gaussians=object(),
                        checkpoint_dir=Path(directory) / "checkpoint",
                        iteration=9,
                        next_iteration=17,
                        args=_args(),
                        block_owner=np.asarray([0, 0], dtype=np.int32),
                    )


if __name__ == "__main__":
    unittest.main()
