import errno
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import torch

from storage.log_storage_manager import LogStorageManager, StorageCapacityError
from storage.pure_ssd_checkpoint import prune_checkpoint_history


class LogStorageLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.storage_dir = self.root / "cache"
        self.storage = LogStorageManager(
            storage_dir=str(self.storage_dir),
            block_size=2,
            num_blocks=4,
            point_dim=3,
            verbose=False,
            max_patch_files=32,
            max_patch_gb=0,
            min_free_gb=0,
            idle_compaction_seconds=0,
        )
        base = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        self.storage.file_paths[0].write_bytes(base.numpy().tobytes())
        self.base_bytes = self.storage.file_paths[0].read_bytes()

    def tearDown(self):
        self.storage.close()
        self.tempdir.cleanup()

    @staticmethod
    def block(value):
        return torch.full((2, 3), float(value), dtype=torch.float32)

    def test_sparse_compaction_preserves_latest_data_versions_and_base(self):
        first = self.storage.write_patch({0: self.block(10), 1: self.block(11)})
        second = self.storage.write_patch({0: self.block(20), 2: self.block(12)})
        old_paths = [self.storage.file_paths[first], self.storage.file_paths[second]]
        versions = {block_id: location.version for block_id, location in self.storage.index.items()}

        self.assertTrue(self.storage.compact_patches(force=True))

        self.assertEqual(self.storage.file_paths[0].read_bytes(), self.base_bytes)
        self.assertEqual(len(self.storage.file_paths), 2)
        self.assertTrue(torch.equal(self.storage.read_blocks([0])[0], self.block(20)))
        self.assertTrue(torch.equal(self.storage.read_blocks([1])[1], self.block(11)))
        self.assertTrue(torch.equal(self.storage.read_blocks([2])[2], self.block(12)))
        self.assertEqual(
            {block_id: location.version for block_id, location in self.storage.index.items()},
            versions,
        )
        self.assertTrue(all(not path.exists() for path in old_paths))

    def test_explicit_stale_version_cannot_replace_newer_index(self):
        self.assertGreaterEqual(
            self.storage.write_patch(
                {0: self.block(50)},
                block_versions={0: 5},
            ),
            1,
        )
        self.assertEqual(
            self.storage.write_patch(
                {0: self.block(40)},
                block_versions={0: 4},
            ),
            -1,
        )
        self.assertEqual(self.storage.get_block_versions([0]), {0: 5})
        self.assertTrue(
            torch.equal(self.storage.read_blocks([0])[0], self.block(50))
        )
        blocks, versions = self.storage.read_blocks_with_versions([0])
        self.assertEqual(versions, {0: 5})
        self.assertTrue(torch.equal(blocks[0], self.block(50)))

    def test_checkpoint_hardlink_survives_runtime_patch_gc(self):
        self.storage.write_patch({0: self.block(10), 1: self.block(11)})
        self.storage.write_patch({0: self.block(20)})
        self.storage.compact_patches(force=True)
        runtime_patch = next(path for file_id, path in self.storage.file_paths.items() if file_id != 0)

        checkpoint_dir = self.root / "checkpoint"
        manifest_path = checkpoint_dir / "storage_index.json"
        manifest = self.storage.export_index_manifest(
            manifest_path=manifest_path,
            patches_dir=checkpoint_dir / "patches",
            patch_file_mode="hardlink",
        )
        checkpoint_patch = Path(next(
            info["path"] for file_id, info in manifest["files"].items() if file_id != "0"
        ))
        self.assertEqual(os.stat(runtime_patch).st_ino, os.stat(checkpoint_patch).st_ino)
        self.assertEqual(manifest["copied_patch_bytes"], 0)

        self.storage.write_patch({0: self.block(30)})
        self.storage.compact_patches(force=True)
        self.assertFalse(runtime_patch.exists())
        self.assertTrue(checkpoint_patch.exists())

        resume_dir = self.root / "resume"
        resume = LogStorageManager(
            storage_dir=str(resume_dir),
            block_size=2,
            num_blocks=4,
            point_dim=3,
            verbose=False,
            min_free_gb=0,
            idle_compaction_seconds=0,
        )
        try:
            resume.load_index_manifest(manifest_path)
            self.assertTrue(torch.equal(resume.read_blocks([0])[0], self.block(20)))
            self.assertTrue(torch.equal(resume.read_blocks([1])[1], self.block(11)))
        finally:
            resume.close()

    def test_checkpoint_hardlink_falls_back_to_copy(self):
        self.storage.write_patch({0: self.block(10)})
        checkpoint_dir = self.root / "copy_fallback"

        with mock.patch(
            "storage.log_storage_manager.os.link",
            side_effect=OSError(errno.EXDEV, "cross-device link"),
        ):
            manifest = self.storage.export_index_manifest(
                manifest_path=checkpoint_dir / "storage_index.json",
                patches_dir=checkpoint_dir / "patches",
                patch_file_mode="hardlink",
            )

        self.assertEqual(manifest["copied_patch_files"], 1)
        self.assertEqual(manifest["linked_patch_files"], 0)
        self.assertGreater(manifest["copied_patch_bytes"], 0)

    def test_failed_atomic_switch_keeps_old_index_and_patches(self):
        self.storage.write_patch({0: self.block(10)})
        self.storage.write_patch({0: self.block(20)})
        old_paths = dict(self.storage.file_paths)
        old_index = dict(self.storage.index)

        with mock.patch("storage.log_storage_manager.os.replace", side_effect=OSError("injected")):
            with self.assertRaisesRegex(OSError, "injected"):
                self.storage.compact_patches(force=True)

        self.assertEqual(self.storage.file_paths, old_paths)
        self.assertEqual(self.storage.index, old_index)
        self.assertTrue(all(path.exists() for path in old_paths.values()))
        self.assertEqual(list(self.storage_dir.glob(".tide_compact_*.tmp")), [])
        self.assertTrue(torch.equal(self.storage.read_blocks([0])[0], self.block(20)))

    def test_patch_write_never_runs_compaction_synchronously(self):
        self.storage.max_patch_files = 2
        self.storage.write_patch({0: self.block(10)})
        self.storage.write_patch({0: self.block(20), 1: self.block(11)})

        self.assertEqual(self.storage.get_stats()["num_patches"], 2)
        self.assertEqual(self.storage.get_stats()["compactions"], 0)

        self.assertTrue(self.storage.maybe_compact())
        self.assertEqual(self.storage.get_stats()["num_patches"], 1)
        self.assertEqual(self.storage.get_stats()["compactions"], 1)
        self.assertTrue(torch.equal(self.storage.read_blocks([0])[0], self.block(20)))

    def test_incremental_compaction_only_merges_oldest_patch_batch(self):
        self.storage.compaction_batch_files = 2
        patch_ids = [
            self.storage.write_patch({block_id: self.block(10 + block_id)})
            for block_id in range(4)
        ]
        old_paths = {
            patch_id: self.storage.file_paths[patch_id]
            for patch_id in patch_ids
        }

        self.assertTrue(self.storage.compact_patches(force=True))

        remaining_ids = set(self.storage.file_paths) - {0}
        self.assertEqual(len(remaining_ids), 3)
        self.assertNotIn(patch_ids[0], remaining_ids)
        self.assertNotIn(patch_ids[1], remaining_ids)
        self.assertIn(patch_ids[2], remaining_ids)
        self.assertIn(patch_ids[3], remaining_ids)
        self.assertFalse(old_paths[patch_ids[0]].exists())
        self.assertFalse(old_paths[patch_ids[1]].exists())
        self.assertTrue(old_paths[patch_ids[2]].exists())
        self.assertTrue(old_paths[patch_ids[3]].exists())
        for block_id in range(4):
            self.assertTrue(
                torch.equal(
                    self.storage.read_blocks([block_id])[block_id],
                    self.block(10 + block_id),
                )
            )

    def test_checkpoint_compaction_drains_bounded_batches(self):
        self.storage.compaction_batch_files = 3
        for version in range(7):
            self.storage.write_patch(
                {version % 4: self.block(20 + version)}
            )

        rounds = self.storage.compact_for_checkpoint()

        self.assertGreaterEqual(rounds, 3)
        self.assertEqual(self.storage.get_stats()["num_patches"], 1)

    def test_reader_priority_allows_new_reader_ahead_of_waiting_writer(self):
        lock = self.storage._storage_rwlock
        writer_acquired = threading.Event()
        second_reader_acquired = threading.Event()

        def writer():
            with lock.write_lock():
                writer_acquired.set()

        def second_reader():
            with lock.read_lock():
                second_reader_acquired.set()

        with lock.read_lock():
            writer_thread = threading.Thread(target=writer)
            writer_thread.start()
            deadline = time.monotonic() + 1.0
            while lock.waiting_writers == 0 and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(lock.waiting_writers, 1)

            reader_thread = threading.Thread(target=second_reader)
            reader_thread.start()
            self.assertTrue(second_reader_acquired.wait(timeout=1.0))
            self.assertFalse(writer_acquired.is_set())

        reader_thread.join(timeout=1.0)
        writer_thread.join(timeout=1.0)
        self.assertTrue(writer_acquired.is_set())

    def test_idle_worker_runs_incremental_compaction(self):
        idle_dir = self.root / "idle-cache"
        storage = LogStorageManager(
            storage_dir=str(idle_dir),
            block_size=2,
            num_blocks=4,
            point_dim=3,
            verbose=False,
            max_patch_files=2,
            max_patch_gb=0,
            min_free_gb=0,
            compaction_batch_files=2,
            idle_compaction_seconds=0.01,
        )
        base = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        storage.file_paths[0].write_bytes(base.numpy().tobytes())
        try:
            storage.write_patch({0: self.block(10)})
            storage.write_patch({1: self.block(11)})
            deadline = time.monotonic() + 2.0
            while (
                storage.get_stats()["idle_compaction_runs"] == 0
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            self.assertEqual(storage.get_stats()["idle_compaction_runs"], 1)
            self.assertEqual(storage.get_stats()["num_patches"], 1)
        finally:
            storage.close()

    def test_live_delta_size_does_not_retrigger_compaction(self):
        self.storage.max_stale_patch_bytes = 1
        self.storage.write_patch({0: self.block(10)})
        self.storage.write_patch({0: self.block(20)})
        self.assertTrue(self.storage.maybe_compact())
        self.assertEqual(self.storage.get_stats()["compactions"], 1)

        self.storage.write_patch({1: self.block(11)})
        self.assertFalse(self.storage.maybe_compact())

        stats = self.storage.get_stats()
        self.assertEqual(stats["compactions"], 1)
        self.assertEqual(stats["num_patches"], 2)
        self.assertEqual(stats["stale_patch_size_mb"], 0)

    def test_low_space_defers_maintenance_without_changing_data(self):
        self.storage.write_patch({0: self.block(10)})
        self.storage.write_patch({0: self.block(20)})
        self.storage.min_free_bytes = 1024
        with mock.patch(
            "storage.log_storage_manager.shutil.disk_usage",
            return_value=mock.Mock(total=2048, used=2048, free=0),
        ):
            self.assertFalse(self.storage.maybe_compact(force=True))

        self.assertEqual(self.storage.stats["compactions_deferred"], 1)
        self.assertTrue(torch.equal(self.storage.read_blocks([0])[0], self.block(20)))

    def test_free_space_reserve_rejects_write_before_creating_patch(self):
        self.storage.min_free_bytes = 1024
        before = set(self.storage_dir.iterdir())
        with mock.patch(
            "storage.log_storage_manager.shutil.disk_usage",
            return_value=mock.Mock(total=2048, used=2048, free=0),
        ):
            with self.assertRaises(StorageCapacityError):
                self.storage.write_patch({0: self.block(10)})
        self.assertEqual(set(self.storage_dir.iterdir()), before)

    def test_checkpoint_retention_keeps_newest_numeric_directories(self):
        model_path = self.root / "model"
        for iteration in (100, 200, 300):
            path = model_path / "checkpoints" / str(iteration)
            path.mkdir(parents=True)
            (path / "manifest.json").write_text(json.dumps({"iteration": iteration}))

        removed = prune_checkpoint_history(model_path, keep_last=2)

        self.assertEqual([path.name for path in removed], ["100"])
        self.assertEqual(
            sorted(path.name for path in (model_path / "checkpoints").iterdir()),
            ["200", "300"],
        )


if __name__ == "__main__":
    unittest.main()
