import json
import tempfile
import unittest
from pathlib import Path

import torch

from storage.optimizer_state_ssd import SSDOptimizerStateStore


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


if __name__ == "__main__":
    unittest.main()
