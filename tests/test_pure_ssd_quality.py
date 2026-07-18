import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.pure_ssd_image_io import load_evaluation_camera, normalize_gt_image
from tools.pure_ssd_quality_utils import (
    checkpoint_manifest_fingerprint,
    compute_next_resident_transition,
    compute_psnr,
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
            )
            reader.load_index_manifest(index_path)
            loaded = reader.read_blocks([0])[0]
            torch.testing.assert_close(loaded, patch)
            self.assertFalse(torch.equal(loaded, base[:2]))
            reader.close()


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
