import math
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import tools.eval_pure_ssd_checkpoint as eval_checkpoint_module
from tools.eval_pure_ssd_checkpoint import (
    _close_evaluation_resources,
    _load_evaluation_checkpoint,
    _optimizer_provenance,
    _resolve_evaluation_resident_capacity,
    build_argparser as build_eval_argparser,
    shared_schedule_cache_dir,
)
from tools.predecode_test_images import build_argparser as build_predecode_argparser
from tools.pure_ssd_image_io import (
    load_evaluation_camera,
    normalize_gt_image,
    raw_cache_path,
)
from tools.pure_ssd_quality_utils import (
    CheckpointShardBlockReader,
    OwnerRoutedBlockReader,
    checkpoint_manifest_fingerprint,
    checkpoint_tree_fingerprint,
    compute_next_resident_transition,
    compute_psnr,
    fingerprint_camera_schedule,
    select_initial_resident_blocks,
    select_preview_indices,
    summarize_values,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None


class PreviewSelectionTest(unittest.TestCase):
    def test_uniform_64_indices_are_unique_and_cover_endpoints(self):
        indices = select_preview_indices(2584, 64)
        self.assertEqual(len(indices), 64)
        self.assertEqual(len(set(indices)), 64)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 2583)

    def test_preview_count_is_bounded_by_camera_count(self):
        self.assertEqual(select_preview_indices(3, 64), [0, 1, 2])
        self.assertEqual(select_preview_indices(0, 64), [])


class CameraScheduleTest(unittest.TestCase):
    def test_ab_outputs_share_schedule_cache(self):
        root = Path("/quality")
        self.assertEqual(
            shared_schedule_cache_dir(root / "adam"),
            root / "camera_schedule_cache",
        )
        self.assertEqual(
            shared_schedule_cache_dir(root / "candidate"),
            root / "camera_schedule_cache",
        )

    def test_schedule_fingerprint_is_order_sensitive(self):
        self.assertEqual(
            fingerprint_camera_schedule([3, 1, 2]),
            fingerprint_camera_schedule([3, 1, 2]),
        )
        self.assertNotEqual(
            fingerprint_camera_schedule([3, 1, 2]),
            fingerprint_camera_schedule([1, 2, 3]),
        )


class PredecodeCliTest(unittest.TestCase):
    def test_test_split_remains_the_default(self):
        args = build_predecode_argparser().parse_args(
            ["--run-dir", "/run", "--output", "/summary.json"]
        )
        self.assertEqual(args.split, "test")
        self.assertEqual(args.expected_count, 2584)

    def test_train_split_accepts_full_matrixcity_count(self):
        args = build_predecode_argparser().parse_args(
            [
                "--run-dir",
                "/run",
                "--output",
                "/summary.json",
                "--split",
                "train",
                "--expected-count",
                "49048",
            ]
        )
        self.assertEqual(args.split, "train")
        self.assertEqual(args.expected_count, 49048)
        self.assertEqual(args.min_free_gb_after, 128.0)

    def test_minimum_free_space_can_be_explicitly_disabled(self):
        args = build_predecode_argparser().parse_args(
            [
                "--run-dir",
                "/run",
                "--output",
                "/summary.json",
                "--min-free-gb-after",
                "0",
            ]
        )
        self.assertEqual(args.min_free_gb_after, 0.0)


class EvaluationCliTest(unittest.TestCase):
    def test_eval_batch_defaults_to_two_independently(self):
        args = build_eval_argparser().parse_args(
            ["--run-dir", "/run", "--output-dir", "/quality"]
        )
        self.assertEqual(args.eval_batch_size, 2)
        self.assertIsNone(args.resident_capacity_blocks)


class EvaluationCleanupTest(unittest.TestCase):
    def test_shutdown_without_compaction_runs_even_when_gpu_clear_fails(self):
        class WorkingSet:
            def clear(self):
                raise RuntimeError("clear failed")

        class Adapter:
            def __init__(self):
                self.calls = []

            def shutdown(self, *, compact_storage):
                self.calls.append(compact_storage)

        adapter = Adapter()
        gaussians = SimpleNamespace(gpu_working_set_manager=WorkingSet())

        with self.assertRaisesRegex(RuntimeError, "clear failed"):
            _close_evaluation_resources(gaussians, adapter)

        self.assertEqual(adapter.calls, [False])

    @unittest.skipUnless(torch is not None, "PyTorch is required")
    def test_failed_initialization_still_cleans_up_and_checks_fingerprint(self):
        events = []
        log_bindings = []

        class WorkingSet:
            def clear(self):
                events.append("clear")

        class GaussianModel:
            def __init__(self, *, sh_degree, only_for_rendering):
                self.gpu_working_set_manager = WorkingSet()

            def prepare_pure_ssd_checkpoint_resume(self, manifest, cameras_extent):
                events.append("prepare")

        class Adapter:
            def shutdown(self, *, compact_storage):
                events.append(("shutdown", compact_storage))

        adapter = Adapter()
        import utils.general_utils as general_utils

        fake_modules = {
            "lpips": SimpleNamespace(),
            "storage.tide_storage_adapter": SimpleNamespace(
                TideStorageAdapter=lambda **kwargs: adapter
            ),
            "strategies.base_engine": SimpleNamespace(calculate_filters=lambda *args: None),
            "strategies.tide_engine.gaussian_model": SimpleNamespace(
                TideGaussianModel=GaussianModel
            ),
            "utils.loss_utils": SimpleNamespace(ssim=lambda *args: None),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            checkpoint_dir = run_dir / "checkpoints" / "7"
            checkpoint_dir.mkdir(parents=True)
            checkpoint_file = checkpoint_dir / "manifest.json"
            checkpoint_file.write_text("before", encoding="utf-8")
            output_dir = root / "quality"
            manifest = {
                "block_size": 4096,
                "checkpoint_type": "pure_ssd_snapshot",
            }
            checkpoint = {
                "kind": "single_rank",
                "manifest": manifest,
                "rank_manifests": [manifest],
                "root_manifest": None,
                "block_owner": None,
                "world_size": 1,
            }
            training_args = SimpleNamespace(
                bsz=16,
                sh_degree=3,
                paper_resident_selection_policy="topc_balanced",
                paper_resident_capacity_blocks=14,
                paper_resident_lambda=0.3,
                paper_resident_recency_decay=0.95,
                paper_balanced_seed_fraction=0.25,
                num_clusters=4,
                max_ram_gb=1.0,
                use_6plane=True,
            )
            cli_args = SimpleNamespace(
                gpu=0,
                run_dir=str(run_dir),
                output_dir=str(output_dir),
                iteration=7,
                resident_capacity_blocks=None,
                eval_batch_size=2,
                camera_limit=-1,
                preview_count=1,
            )

            def fail_after_adapter(checkpoint_payload):
                checkpoint_file.write_text("changed", encoding="utf-8")
                raise RuntimeError("initialization failed")

            with mock.patch.dict(sys.modules, fake_modules), mock.patch.object(
                general_utils, "set_args"
            ), mock.patch.object(
                general_utils,
                "set_log_file",
                side_effect=lambda handle: log_bindings.append(handle),
            ), mock.patch.object(
                torch.cuda, "is_available", return_value=True
            ), mock.patch.object(torch.cuda, "set_device"), mock.patch.object(
                eval_checkpoint_module,
                "_load_evaluation_checkpoint",
                return_value=checkpoint,
            ), mock.patch.object(
                eval_checkpoint_module,
                "_load_run_args",
                return_value=training_args,
            ), mock.patch.object(
                eval_checkpoint_module,
                "load_test_scene_metadata",
                return_value=([], 1.0, 1, 1),
            ), mock.patch.object(
                eval_checkpoint_module,
                "_checkpoint_block_reader",
                side_effect=fail_after_adapter,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "checkpoint tree changed during failed read-only evaluation",
                ):
                    eval_checkpoint_module.evaluate_checkpoint(cli_args)

        self.assertEqual(events, ["prepare", "clear", ("shutdown", False)])
        self.assertEqual(len(log_bindings), 2)
        self.assertIsNone(log_bindings[-1])
        self.assertTrue(log_bindings[0].closed)


@unittest.skipUnless(torch is not None, "PyTorch package initialization is required")
class ResidentSelectionTest(unittest.TestCase):
    def test_topc_balanced_selection_is_deterministic_and_capacity_bounded(self):
        camera_blocks = {
            3: list(range(0, 18)),
            7: list(range(9, 30)),
            11: list(range(20, 41)),
        }
        visible = sorted({block for blocks in camera_blocks.values() for block in blocks})
        kwargs = {
            "capacity": 12,
            "resident_lambda": 0.3,
            "recency_decay": 0.95,
            "balanced_seed_fraction": 0.25,
        }
        first = select_initial_resident_blocks(visible, camera_blocks, **kwargs)
        second = select_initial_resident_blocks(visible, camera_blocks, **kwargs)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 12)

        transition = compute_next_resident_transition(
            current_blocks=visible,
            next_blocks=list(range(30, 80)),
            current_resident_blocks=first,
            next_camera_ids=[13, 17],
            next_camera_blocks={13: list(range(30, 55)), 17: list(range(50, 80))},
            previous_recency_scores={},
            **kwargs,
        )
        self.assertLessEqual(len(transition.next_resident_blocks), 12)


class SummaryTest(unittest.TestCase):
    def test_summary_statistics(self):
        summary = summarize_values([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["median"], 2.5)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 4.0)

    def test_non_finite_values_fail(self):
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            summarize_values([1.0, float("nan")])

    def test_distributed_optimizer_provenance_is_reported_from_checkpoint(self):
        checkpoint = {
            "manifest": {"optimizer_state_mode": "cold_start"},
            "root_manifest": {
                "optimizer_provenance": {
                    "paper_optimizer_algorithm": "3dgs2_tr",
                    "paper_sophia_curvature_interval": 10,
                },
                "global_optimizer_step": 11,
                "optimizer_state_mode": "cold_start_per_rank",
                "optimizer_ema_saved": False,
                "optimizer_curvature_saved": False,
            },
        }
        result = _optimizer_provenance(
            SimpleNamespace(paper_optimizer_algorithm="adam"),
            checkpoint,
        )
        self.assertEqual(result["algorithm"], "3dgs2_tr")
        self.assertEqual(result["global_optimizer_step"], 11)
        self.assertEqual(result["state_mode"], "cold_start_per_rank")


class EvaluationCameraSourceTest(unittest.TestCase):
    @staticmethod
    def make_context(root):
        args = SimpleNamespace(decode_dataset_path=str(root))
        camera_info = SimpleNamespace(image_name="test/camera_0001")
        return args, camera_info

    def test_existing_raw_uses_cache_loader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args, camera_info = self.make_context(root)
            raw_path = root / "dataset_raw" / "test" / "camera_0001.raw"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(bytes(range(12)))
            calls = []

            def raw_loader(*_):
                calls.append("raw")
                return "raw-camera"

            def source_loader(*_):
                calls.append("source")
                return "source-camera"

            result = load_evaluation_camera(
                args,
                0,
                camera_info,
                2,
                2,
                raw_loader=raw_loader,
                source_loader=source_loader,
            )
            self.assertEqual(result.camera, "raw-camera")
            self.assertEqual(result.source, "raw")
            self.assertEqual(calls, ["raw"])

    def test_raw_path_prefers_split_aware_image_cache_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(decode_dataset_path=temp_dir)
            camera_info = SimpleNamespace(
                image_name="camera_0001",
                image_cache_key="test/block_42/camera_0001",
            )
            self.assertEqual(
                raw_cache_path(args, camera_info),
                Path(temp_dir)
                / "dataset_raw"
                / "test"
                / "block_42"
                / "camera_0001.raw",
            )

    def test_explicit_source_mode_bypasses_usable_raw(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args, camera_info = self.make_context(root)
            raw_path = root / "dataset_raw" / "test" / "camera_0001.raw"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(bytes(range(12)))
            calls = []

            def raw_loader(*_):
                calls.append("raw")
                return "raw-camera"

            def source_loader(*_):
                calls.append("source")
                return "source-camera"

            result = load_evaluation_camera(
                args,
                0,
                camera_info,
                2,
                2,
                use_raw_cache=False,
                raw_loader=raw_loader,
                source_loader=source_loader,
            )
            self.assertEqual(result.camera, "source-camera")
            self.assertEqual(result.source, "source_image")
            self.assertEqual(result.fallback_reason, "raw_disabled")
            self.assertEqual(calls, ["source"])
            self.assertTrue(raw_path.exists())

    def test_missing_raw_falls_back_without_creating_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args, camera_info = self.make_context(root)
            raw_path = root / "dataset_raw" / "test" / "camera_0001.raw"

            result = load_evaluation_camera(
                args,
                0,
                camera_info,
                2,
                2,
                raw_loader=lambda *_: self.fail("raw loader should not run"),
                source_loader=lambda *_: "source-camera",
            )
            self.assertEqual(result.camera, "source-camera")
            self.assertEqual(result.source, "source_image")
            self.assertEqual(result.fallback_reason, "raw_missing")
            self.assertFalse(raw_path.exists())
            self.assertFalse((root / "dataset_raw").exists())

    def test_source_fallback_preserves_checkpoint_fingerprint_and_mtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args, camera_info = self.make_context(root)
            checkpoint = root / "pure_ssd_checkpoint.json"
            checkpoint.write_text('{"iteration": 32}\n')
            before = checkpoint_manifest_fingerprint(checkpoint)

            load_evaluation_camera(
                args,
                0,
                camera_info,
                2,
                2,
                source_loader=lambda *_: "source-camera",
            )

            after = checkpoint_manifest_fingerprint(checkpoint)
            self.assertEqual(after, before)


class CheckpointTreeFingerprintTest(unittest.TestCase):
    def test_nested_artifact_content_is_covered(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint"
            nested = checkpoint / "rank_1" / "ssd_delta" / "patches" / "patch.bin"
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b"before")
            (checkpoint / "block_owner.npy").write_bytes(b"owner")
            before = checkpoint_tree_fingerprint(checkpoint)

            nested.write_bytes(b"after!")
            after = checkpoint_tree_fingerprint(checkpoint)

            self.assertNotEqual(after, before)
            self.assertEqual(after["file_count"], 2)

    def test_empty_directory_is_covered(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint"
            checkpoint.mkdir()
            before = checkpoint_tree_fingerprint(checkpoint)

            (checkpoint / "unexpected_empty_directory").mkdir()
            after = checkpoint_tree_fingerprint(checkpoint)

            self.assertNotEqual(after, before)
            self.assertEqual(
                after["directory_count"],
                before["directory_count"] + 1,
            )


class EvaluationResidentCapacityTest(unittest.TestCase):
    def test_legacy_v1_capacity_is_converted_from_per_rank_to_global(self):
        checkpoint = {
            "root_manifest": {
                "checkpoint_version": 1,
                "world_size": 4,
                "per_rank_capacity_blocks": 2048,
            }
        }

        capacity = _resolve_evaluation_resident_capacity(
            checkpoint,
            SimpleNamespace(paper_resident_capacity_blocks=123),
            None,
        )

        self.assertEqual(capacity, 8192)

    def test_cli_capacity_override_has_priority(self):
        checkpoint = {
            "root_manifest": {
                "checkpoint_version": 1,
                "world_size": 4,
                "per_rank_capacity_blocks": 2048,
            }
        }
        self.assertEqual(
            _resolve_evaluation_resident_capacity(
                checkpoint,
                SimpleNamespace(paper_resident_capacity_blocks=123),
                17,
            ),
            17,
        )


class OwnerRoutingTest(unittest.TestCase):
    def test_reader_never_fetches_a_block_from_its_non_owner(self):
        class FakeReader:
            layout = "cache"
            total_gaussians = 4
            block_size = 2
            num_blocks = 2

            def __init__(self, rank):
                self.rank = rank
                self.requests = []

            def read_blocks(self, block_ids):
                self.requests.append(list(block_ids))
                return {block_id: f"rank-{self.rank}" for block_id in block_ids}

        readers = [FakeReader(0), FakeReader(1)]
        routed = OwnerRoutedBlockReader(readers, [1, 0])
        loaded = routed.read_blocks([0, 1])

        self.assertEqual(readers[0].requests, [[1]])
        self.assertEqual(readers[1].requests, [[0]])
        self.assertEqual(loaded, {0: "rank-1", 1: "rank-0"})

    def test_reader_rejects_unrequested_blocks_from_owner_shard(self):
        class FakeReader:
            layout = "cache"
            total_gaussians = 2
            block_size = 1
            num_blocks = 2

            def read_blocks(self, block_ids):
                return {0: "requested", 1: "unexpected"}

        routed = OwnerRoutedBlockReader([FakeReader()], [0, 0])
        with self.assertRaisesRegex(RuntimeError, "unrequested blocks"):
            routed.read_blocks([0])


@unittest.skipUnless(torch is not None, "PyTorch is required")
class EvaluationCheckpointDetectionTest(unittest.TestCase):
    def test_distributed_checkpoint_is_detected_and_all_ranks_loaded(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy is required")
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint"
            checkpoint.mkdir()
            np.save(checkpoint / "block_owner.npy", np.asarray([0, 1], dtype=np.int32))
            np.save(checkpoint / "block_bounds.npy", np.zeros((2, 6), dtype=np.float32))
            for rank in range(2):
                rank_dir = checkpoint / f"rank_{rank}"
                rank_dir.mkdir()
                (rank_dir / "pure_ssd_checkpoint.json").write_text(
                    json.dumps(
                        {
                            "checkpoint_type": "pure_ssd_snapshot",
                            "base_file": "base.bin",
                            "block_bounds": "bounds.npy",
                            "training_state": "training_state.pth",
                            "scene_min": [0.0, 0.0, 0.0],
                            "scene_max": [1.0, 1.0, 1.0],
                            "total_points": 4,
                            "num_blocks": 2,
                            "block_size": 2,
                            "param_dim": 59,
                            "next_iteration": 33,
                        }
                    ),
                    encoding="utf-8",
                )
            (checkpoint / "pure_ssd_distributed_checkpoint.json").write_text(
                json.dumps(
                    {
                        "checkpoint_version": 2,
                        "world_size": 2,
                        "global_bsz": 4,
                        "global_capacity_blocks": 2,
                        "next_iteration": 33,
                        "rank_manifests": ["rank_0", "rank_1"],
                        "block_owner": "block_owner.npy",
                        "block_bounds": "block_bounds.npy",
                    }
                ),
                encoding="utf-8",
            )

            loaded = _load_evaluation_checkpoint(checkpoint)

            self.assertEqual(loaded["kind"], "distributed")
            self.assertEqual(loaded["world_size"], 2)
            self.assertEqual(len(loaded["rank_manifests"]), 2)
            self.assertEqual(loaded["block_owner"].tolist(), [0, 1])
            self.assertEqual(
                loaded["manifest"]["block_bounds"],
                str((checkpoint / "block_bounds.npy").resolve()),
            )


@unittest.skipUnless(torch is not None, "PyTorch is required")
class ImageMetricTest(unittest.TestCase):
    def test_known_psnr_and_ssim(self):
        from utils.loss_utils import ssim

        zeros = torch.zeros(1, 3, 16, 16)
        ones = torch.ones_like(zeros)
        self.assertEqual(compute_psnr(zeros, ones), 0.0)
        self.assertTrue(math.isinf(compute_psnr(zeros, zeros)))
        self.assertAlmostEqual(float(ssim(zeros, zeros).item()), 1.0, places=6)

    def test_uint8_and_unit_float_gt_normalize_identically(self):
        uint8_image = torch.tensor([0, 64, 127, 255], dtype=torch.uint8).reshape(1, 2, 2)
        float_image = uint8_image.float() / 255.0
        torch.testing.assert_close(
            normalize_gt_image(uint8_image),
            normalize_gt_image(float_image),
        )

    def test_identical_image_lpips_is_near_zero_when_assets_are_available(self):
        try:
            import lpips
        except ModuleNotFoundError:
            self.skipTest("lpips is not installed")
        weights = Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth"
        if not weights.is_file():
            self.skipTest("offline AlexNet weights are not installed")
        metric = lpips.LPIPS(net="alex", version="0.1").eval()
        image = torch.linspace(0.0, 1.0, steps=3 * 64 * 64).reshape(1, 3, 64, 64)
        score = float(metric(image, image, normalize=True).item())
        self.assertLess(abs(score), 1e-6)


@unittest.skipUnless(torch is not None, "PyTorch is required")
class CheckpointIndexReadTest(unittest.TestCase):
    def test_storage_index_reads_patch_instead_of_base(self):
        from storage.log_storage_manager import LogStorageManager

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            source_dir.mkdir()
            storage = LogStorageManager(
                storage_dir=str(source_dir),
                block_size=2,
                num_blocks=2,
                point_dim=59,
                verbose=False,
                min_free_gb=0,
            )
            base = torch.arange(4 * 59, dtype=torch.float32).reshape(4, 59)
            (source_dir / "base_file.bin").write_bytes(base.numpy().tobytes())
            patch = torch.full((2, 59), 9.0, dtype=torch.float32)
            storage.write_patch({0: patch})
            index_path = root / "checkpoint" / "storage_index.json"
            storage.export_index_manifest(index_path, root / "checkpoint" / "patches")
            storage.close()

            active_dir = root / "active"
            reader = LogStorageManager(
                storage_dir=str(active_dir),
                block_size=2,
                num_blocks=2,
                point_dim=59,
                verbose=False,
                min_free_gb=0,
            )
            reader.load_index_manifest(index_path)
            loaded = reader.read_blocks([0])[0]
            torch.testing.assert_close(loaded, patch)
            self.assertFalse(torch.equal(loaded, base[:2]))
            reader.close()

    def test_read_only_shard_reader_uses_latest_index_location(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = torch.zeros((4, 59), dtype=torch.float32)
            patch = torch.full((2, 59), 7.0, dtype=torch.float32)
            base_path = root / "base.bin"
            patch_path = root / "patch.bin"
            base_path.write_bytes(base.numpy().tobytes())
            patch_path.write_bytes(patch.numpy().tobytes())
            index_path = root / "storage_index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "block_size": 2,
                        "num_blocks": 2,
                        "point_dim": 59,
                        "files": {
                            "0": {"path": str(base_path)},
                            "1": {"path": str(patch_path)},
                        },
                        "index": {
                            "0": {"file_id": 1, "offset": 0, "size": patch.numel() * 4},
                            "1": {
                                "file_id": 0,
                                "offset": 2 * 59 * 4,
                                "size": 2 * 59 * 4,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            reader = CheckpointShardBlockReader(
                {
                    "checkpoint_type": "pure_ssd_incremental",
                    "storage_index": str(index_path),
                    "base_file": str(base_path),
                    "total_points": 4,
                    "num_blocks": 2,
                    "block_size": 2,
                    "param_dim": 59,
                }
            )
            torch.testing.assert_close(reader.read_blocks([0])[0], patch)

    def test_read_only_shard_reader_trims_final_block_to_valid_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = torch.arange(4 * 59, dtype=torch.float32).reshape(4, 59)
            base_path = root / "base.bin"
            base_path.write_bytes(base.numpy().tobytes())
            reader = CheckpointShardBlockReader(
                {
                    "checkpoint_type": "pure_ssd_snapshot",
                    "base_file": str(base_path),
                    "total_points": 3,
                    "num_blocks": 2,
                    "block_size": 2,
                    "param_dim": 59,
                }
            )

            final_block = reader.read_blocks([1])[1]

            self.assertEqual(tuple(final_block.shape), (1, 59))
            torch.testing.assert_close(final_block, base[2:3])

class TrainWrapperCheckpointTest(unittest.TestCase):
    def test_explicit_checkpoint_iteration_applies_to_train_mode(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "train_matrixcity_1b.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(
                [
                    "bash",
                    str(script),
                    "--mode",
                    "train",
                    "--root",
                    str(root),
                    "--run-tag",
                    "quality_test",
                    "--src",
                    str(root / "dataset"),
                    "--ply",
                    str(root / "points.ply"),
                    "--iterations",
                    "500000",
                    "--checkpoint-iter",
                    "500000",
                    "--dry-run",
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            commands = (root / "output" / "runs" / "quality_test" / "commands.sh").read_text()
            self.assertIn("--checkpoint_iterations", commands)
            self.assertIn("500000", commands)


if __name__ == "__main__":
    unittest.main()
