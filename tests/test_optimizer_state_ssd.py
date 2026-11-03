import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from storage.optimizer_state_ssd import SSDOptimizerStateStore
from strategies.tide_engine.gpu_resident_optimizer import GPUResidentAdam


def _state(step=3, rows=2):
    return {
        "row_count": rows,
        "step": step,
        "exp_avg": {
            "xyz": torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3),
            "opacity": torch.ones((rows, 1), dtype=torch.float32),
        },
        "exp_avg_sq": {
            "xyz": torch.full((rows, 3), 2.0, dtype=torch.float32),
            "opacity": torch.full((rows, 1), 4.0, dtype=torch.float32),
        },
    }


def _full_state(step=3, rows=2):
    widths = {
        name: width for name, _, width, _ in GPUResidentAdam.COMPONENT_SPECS
    }
    return {
        "row_count": rows,
        "step": step,
        "exp_avg": {
            name: torch.full((rows, width), float(step))
            for name, width in widths.items()
        },
        "exp_avg_sq": {
            name: torch.full((rows, width), float(step) + 1.0)
            for name, width in widths.items()
        },
    }


class SSDOptimizerStateStoreTest(unittest.TestCase):
    def test_async_write_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SSDOptimizerStateStore(temporary, reset=True)
            try:
                store.save_async(7, _state(step=11))
                store.flush()
                loaded = store.load(7)
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded["step"], 11)
                torch.testing.assert_close(loaded["exp_avg"]["xyz"], _state()["exp_avg"]["xyz"])
                self.assertEqual(store.stats()["blocks"], 1)
                self.assertGreater(store.stats()["bytes"], 0)
            finally:
                store.close()

    def test_checkpoint_export_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = SSDOptimizerStateStore(root / "working", reset=True)
            try:
                source.save_async(3, _state(step=5))
                source.flush()
                exported = source.export_checkpoint(
                    root / "checkpoint",
                    {9: _state(step=8, rows=1)},
                    block_size=4096,
                )
            finally:
                source.close()

            resumed = SSDOptimizerStateStore(root / "resumed", reset=True)
            try:
                resumed.import_checkpoint(exported["manifest"])
                shutil.rmtree(root / "checkpoint")
                self.assertEqual(resumed.load(3)["step"], 5)
                self.assertEqual(resumed.load(9)["step"], 8)
            finally:
                resumed.close()

    def test_checksum_failure_allows_cold_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SSDOptimizerStateStore(root, reset=True, allow_cold_start=True)
            try:
                store.save_async(4, _state())
                store.flush()
                index = json.loads((root / "index.json").read_text())
                state_path = root / index["blocks"]["4"]["path"]
                state_path.write_bytes(state_path.read_bytes() + b"corrupt")
                self.assertIsNone(store.load(4))
                self.assertGreaterEqual(store.stats()["warnings"], 1)
            finally:
                store.close()

    def test_repeated_writeback_keeps_one_stable_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SSDOptimizerStateStore(temporary, reset=True)
            try:
                store.save_async(4, _state(step=1))
                store.save_async(4, _state(step=2))
                store.flush()
                self.assertEqual(
                    sorted(path.name for path in (Path(temporary) / "blocks").glob("*.pt")),
                    ["block_4.pt"],
                )
                self.assertEqual(store.load(4)["step"], 2)
                self.assertEqual(store.state_version(4), 0)
            finally:
                store.close()

    def test_stale_version_cannot_replace_newer_writeback(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SSDOptimizerStateStore(temporary, reset=True)
            try:
                store.save_async(4, _state(step=2), state_version=2)
                store.save_async(4, _state(step=1), state_version=1)
                store.flush()
                self.assertEqual(store.load(4)["step"], 2)
                self.assertEqual(store.state_version(4), 2)
            finally:
                store.close()

    def test_checkpoint_is_snapshot_and_does_not_repoint_live_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SSDOptimizerStateStore(root / "working", reset=True)
            try:
                store.save_async(4, _state(step=7))
                store.flush()
                exported = store.export_checkpoint(root / "checkpoint", {})
                shutil.rmtree(root / "checkpoint")
                self.assertEqual(store.load(4)["step"], 7)
                self.assertTrue(Path(exported["manifest"]).is_relative_to(root / "checkpoint"))
            finally:
                store.close()

    def test_reset_removes_orphaned_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = SSDOptimizerStateStore(root, reset=True)
            first.save_async(4, _state())
            first.flush()
            first.close()
            self.assertTrue((root / "blocks" / "block_4.pt").exists())
            second = SSDOptimizerStateStore(root, reset=True)
            try:
                self.assertFalse((root / "blocks" / "block_4.pt").exists())
                self.assertIsNone(second.load(4))
            finally:
                second.close()


class GPUResidentAdamSSDTierTest(unittest.TestCase):
    def test_gpu_eviction_stays_in_cpu_until_gaussian_eviction(self):
        with tempfile.TemporaryDirectory() as temporary:
            optimizer = GPUResidentAdam(
                batch_size=1, block_size=2, capacity_blocks=1, device="cpu"
            )
            optimizer.enable_ssd_state_cache(temporary)
            optimizer.set_resident_blocks([4])
            slot = optimizer._block_to_slot[4]
            optimizer._slot_initialized[slot] = True
            optimizer._slot_row_counts[slot] = 2
            optimizer._slot_steps[slot] = 9
            optimizer.set_resident_blocks([])
            self.assertIn(4, optimizer._cpu_state_cache)
            self.assertEqual(optimizer._ssd_state_store.stats()["state_writes"], 0)
            optimizer.on_cpu_block_evicted(4)
            optimizer.flush_persistent_state()
            self.assertNotIn(4, optimizer._cpu_state_cache)
            self.assertEqual(optimizer._ssd_state_store.stats()["state_writes"], 1)
            optimizer.close()

    def test_cpu_hit_precedes_ssd_and_ssd_hit_restores_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            optimizer = GPUResidentAdam(
                batch_size=1, block_size=2, capacity_blocks=1, device="cpu"
            )
            optimizer.enable_ssd_state_cache(temporary)
            optimizer._ssd_state_store.save_async(4, _full_state(step=13, rows=2), state_version=3)
            optimizer._ssd_state_store.flush()
            optimizer.set_resident_blocks([4])
            slot = optimizer._block_to_slot[4]
            self.assertEqual(optimizer._slot_steps[slot], 13)
            self.assertEqual(optimizer.get_stats()["persistent_state_ssd_hits"], 1)
            optimizer.set_resident_blocks([])
            self.assertIn(4, optimizer._cpu_state_cache)
            before = optimizer.get_stats()["persistent_state_ssd_hits"]
            optimizer.set_resident_blocks([4])
            self.assertEqual(optimizer.get_stats()["persistent_state_ssd_hits"], before)
            self.assertEqual(optimizer.get_stats()["persistent_state_cpu_hits"], 1)
            optimizer.close()

    def test_sophia_rejects_ssd_state_cache(self):
        from strategies.tide_engine.gpu_resident_optimizer import GPUResidentSophiaTR

        optimizer = GPUResidentSophiaTR(
            batch_size=1, block_size=2, capacity_blocks=1, device="cpu"
        )
        with self.assertRaisesRegex(RuntimeError, "supports Adam only"):
            optimizer.enable_ssd_state_cache("/tmp/unused-optimizer-state")


if __name__ == "__main__":
    unittest.main()
