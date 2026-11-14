import threading
import unittest

import torch

from strategies.tide_engine.gpu_resident_optimizer import GPUResidentAdam


class FakeExtentCache:
    def __init__(self):
        self.state_payloads = {}
        self.state_payloads_lock = threading.RLock()

    def attach_block_state(self, block_id, payload, *, step=0, state_version=None):
        tensor = payload.detach().to("cpu").contiguous()
        version = int(state_version) if state_version is not None else 0
        with self.state_payloads_lock:
            self.state_payloads[int(block_id)] = (tensor, int(step), version)

    def state_prefetch(self, block_ids, *, allow_storage=True):
        result = {}
        with self.state_payloads_lock:
            for block_id in block_ids:
                entry = self.state_payloads.get(int(block_id))
                if entry is not None:
                    tensor, step, version = entry
                    result[int(block_id)] = {
                        "tensor": tensor,
                        "row_count": int(tensor.shape[0]),
                        "step": int(step),
                        "version": int(version),
                    }
        return result

    def take_block_state(self, block_id):
        with self.state_payloads_lock:
            entry = self.state_payloads.pop(int(block_id), None)
        if entry is None:
            return None
        tensor, step, version = entry
        return {"tensor": tensor, "step": int(step), "version": int(version)}


class StateExtentOptimizerTest(unittest.TestCase):
    def setUp(self):
        self.optimizer = GPUResidentAdam(
            batch_size=2, block_size=4, capacity_blocks=4, device="cpu"
        )
        self.cache = FakeExtentCache()

    def _fill_slot(self, slot, row_count, base):
        offset = 0
        for name, _, width, _ in self.optimizer.COMPONENT_SPECS:
            avg = self.optimizer._exp_avg[name].view(
                self.optimizer._allocated_slots, self.optimizer.block_size, width
            )
            sq = self.optimizer._exp_avg_sq[name].view(
                self.optimizer._allocated_slots, self.optimizer.block_size, width
            )
            values = torch.arange(width, dtype=torch.float32) + float(base + offset)
            avg[slot, :row_count] = values
            sq[slot, :row_count] = -values
            offset += 1

    def test_flat_snapshot_restore_round_trip(self):
        self.optimizer._ensure_storage(2)
        self.optimizer._slot_steps = [3, 7]
        self.optimizer._slot_row_counts = [4, 2]
        self.optimizer._slot_initialized = [True, True]
        self._fill_slot(1, 2, 10.0)
        expected = {
            name: (
                self.optimizer._exp_avg[name].view(
                    self.optimizer._allocated_slots, self.optimizer.block_size, width
                )[1, :2].clone(),
                self.optimizer._exp_avg_sq[name].view(
                    self.optimizer._allocated_slots, self.optimizer.block_size, width
                )[1, :2].clone(),
            )
            for name, _, width, _ in self.optimizer.COMPONENT_SPECS
        }

        flat, step = self.optimizer._snapshot_slot_flat(1, 1)
        self.assertEqual(step, 7)
        self.assertEqual(
            tuple(flat.shape), (2, 2 * self.optimizer.TOTAL_WIDTH)
        )

        for name, _, width, _ in self.optimizer.COMPONENT_SPECS:
            self.optimizer._exp_avg[name].view(
                self.optimizer._allocated_slots, self.optimizer.block_size, width
            ).zero_()
            self.optimizer._exp_avg_sq[name].view(
                self.optimizer._allocated_slots, self.optimizer.block_size, width
            ).zero_()

        self.optimizer._restore_slot_flat(1, 1, flat, 2)
        for name, _, width, _ in self.optimizer.COMPONENT_SPECS:
            restored_avg = self.optimizer._exp_avg[name].view(
                self.optimizer._allocated_slots, self.optimizer.block_size, width
            )[1, :2]
            restored_sq = self.optimizer._exp_avg_sq[name].view(
                self.optimizer._allocated_slots, self.optimizer.block_size, width
            )[1, :2]
            self.assertTrue(torch.equal(restored_avg, expected[name][0]))
            self.assertTrue(torch.equal(restored_sq, expected[name][1]))

    def test_resident_cycle_attaches_and_restores_state(self):
        self.optimizer.enable_state_extent(self.cache)
        self.optimizer.set_resident_blocks([0, 1])
        for block_id in [0, 1]:
            slot = self.optimizer._block_to_slot[block_id]
            self.optimizer._slot_row_counts[slot] = 4
            self.optimizer._slot_initialized[slot] = True
            self.optimizer._slot_steps[slot] = 5
            self._fill_slot(slot, 4, 100.0 * (block_id + 1))

        slot_zero = self.optimizer._block_to_slot[0]
        snapshot = {
            name: (
                self.optimizer._exp_avg[name].view(
                    self.optimizer._allocated_slots, self.optimizer.block_size, width
                )[slot_zero, :4].clone(),
                self.optimizer._exp_avg_sq[name].view(
                    self.optimizer._allocated_slots, self.optimizer.block_size, width
                )[slot_zero, :4].clone(),
            )
            for name, _, width, _ in self.optimizer.COMPONENT_SPECS
        }

        self.optimizer.set_resident_blocks([2, 3])
        self.assertIn(0, self.cache.state_payloads)
        self.assertIn(1, self.cache.state_payloads)

        self.optimizer.set_resident_blocks([0, 1])
        slot = self.optimizer._block_to_slot[0]
        self.assertNotIn(0, self.cache.state_payloads)
        self.assertNotIn(1, self.cache.state_payloads)
        for name, _, width, _ in self.optimizer.COMPONENT_SPECS:
            restored_avg = self.optimizer._exp_avg[name].view(
                self.optimizer._allocated_slots, self.optimizer.block_size, width
            )[slot, :4]
            restored_sq = self.optimizer._exp_avg_sq[name].view(
                self.optimizer._allocated_slots, self.optimizer.block_size, width
            )[slot, :4]
            self.assertTrue(torch.equal(restored_avg, snapshot[name][0]))
            self.assertTrue(torch.equal(restored_sq, snapshot[name][1]))


if __name__ == "__main__":
    unittest.main()
