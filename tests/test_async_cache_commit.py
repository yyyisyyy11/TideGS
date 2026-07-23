import threading
import time
import unittest
from queue import Queue

import torch

from storage.block_reader import TieredCacheBlockReader
from storage.tide_storage_adapter import TideStorageAdapter


class _Payload:
    def __init__(self, block_id, tensor):
        self.block_ids = [block_id]
        self.tensor = tensor
        self.ready = threading.Event()
        self.released = threading.Event()

    def wait_gpu(self):
        self.ready.wait(timeout=2.0)

    def wait(self):
        if not self.ready.wait(timeout=2.0):
            raise TimeoutError("payload was not released")
        return {self.block_ids[0]: self.tensor}

    def release(self):
        self.released.set()


class _Cache:
    def __init__(self, block_id, tensor):
        self.data = {block_id: tensor}
        self.future_hints = []

    def prefetch(self, block_ids):
        return {block_id: self.data[block_id] for block_id in block_ids}

    def prefetch_future(self, block_ids):
        self.future_hints.append(list(block_ids))
        return len(block_ids)


class _ResidentState:
    def __init__(self, block_ids):
        self._dirty = set(block_ids)

    def dirty_blocks(self):
        return sorted(self._dirty)

    def mark_blocks_written_back(self, block_ids):
        self._dirty.difference_update(block_ids)


class _WorkingSet:
    def __init__(self, payloads):
        self.payloads = payloads

    def stage_updated_blocks(self, block_ids):
        payload = _Payload(block_ids[0], torch.full((4, 59), 9.0))
        payload.ready.set()
        self.payloads.append(payload)
        return payload


class AsyncCacheCommitTest(unittest.TestCase):
    def setUp(self):
        self.adapter = TideStorageAdapter.__new__(TideStorageAdapter)
        self.adapter.block_owner = None
        self.adapter.owner_rank = None
        self.adapter.execution_metrics = {
            "paper_cache_sync_calls": 0,
            "paper_cache_sync_blocks": 0,
            "paper_cache_sync_staged_blocks": 0,
            "paper_bounds_refresh_blocks": 0,
            "background_cache_commit_jobs": 0,
            "background_cache_commit_blocks": 0,
            "background_cache_commit_waits": 0,
        }
        self.adapter._cache_commit_queue = Queue()
        self.adapter._cache_commit_lock = threading.RLock()
        self.adapter._pending_cache_commits = {}
        self.adapter._cache_commit_error = None
        self.adapter._cache_commit_thread = None
        self.payloads = []
        self.refresh_bounds_flags = []
        self.cache = _Cache(3, torch.zeros(4, 59))

        def sync_cache(updated, *, refresh_bounds=True):
            self.refresh_bounds_flags.append(bool(refresh_bounds))
            self.cache.data.update(
                {block_id: tensor.clone() for block_id, tensor in updated.items()}
            )
            return len(updated)

        self.adapter.sync_cache_from_cpu_views = sync_cache
        self.adapter._start_cache_commit_worker()

    def tearDown(self):
        for payload in self.payloads:
            payload.ready.set()
        self.adapter.drain_cache_writebacks()
        self.adapter._cache_commit_queue.put(None)
        self.adapter._cache_commit_queue.join()
        self.adapter._cache_commit_thread.join(timeout=2.0)

    def test_reader_waits_for_new_version_and_hint_skips_pending_block(self):
        updated = torch.full((4, 59), 7.0)
        payload = _Payload(3, updated)
        self.payloads.append(payload)
        self.assertEqual(self.adapter.submit_cache_writeback(payload), 1)

        reader = TieredCacheBlockReader(
            cache_manager=self.cache,
            total_gaussians=16,
            block_size=4,
            before_read=self.adapter.wait_for_cache_blocks,
            filter_hint=self.adapter.filter_cache_prefetch_candidates,
        )
        self.assertEqual(reader.hint_future([3]), 0)
        self.assertEqual(self.cache.future_hints, [])

        result = {}
        thread = threading.Thread(
            target=lambda: result.update(reader.read_blocks([3])),
        )
        thread.start()
        time.sleep(0.05)
        self.assertTrue(thread.is_alive())

        payload.ready.set()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(torch.equal(result[3], updated))
        self.assertTrue(payload.released.is_set())
        self.assertEqual(self.refresh_bounds_flags[-1], True)

    def test_training_commit_skips_duplicate_cpu_bounds_refresh(self):
        payload = _Payload(3, torch.full((4, 59), 5.0))
        self.payloads.append(payload)
        self.assertEqual(
            self.adapter.submit_cache_writeback(
                payload,
                bounds_managed_externally=True,
            ),
            1,
        )
        payload.ready.set()
        self.adapter.drain_cache_writebacks()
        self.assertEqual(self.refresh_bounds_flags[-1], False)

    def test_checkpoint_flushes_gpu_dirty_resident_blocks(self):
        resident = _ResidentState([3])
        working_set = _WorkingSet(self.payloads)
        self.adapter.bind_resident_writeback(resident, working_set)

        self.assertEqual(self.adapter.flush_resident_dirty(), 1)
        self.assertEqual(resident.dirty_blocks(), [])
        self.assertTrue(torch.equal(self.cache.data[3], torch.full((4, 59), 9.0)))
        self.assertTrue(self.payloads[-1].released.is_set())

    def test_checkpoint_waits_for_already_pending_gpu_commit(self):
        payload = _Payload(3, torch.full((4, 59), 6.0))
        self.payloads.append(payload)
        self.assertEqual(self.adapter.submit_cache_writeback(payload), 1)
        self.adapter.bind_resident_writeback(_ResidentState([]), _WorkingSet([]))

        thread = threading.Thread(target=self.adapter.flush_resident_dirty)
        thread.start()
        time.sleep(0.05)
        self.assertTrue(thread.is_alive())
        payload.ready.set()
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(torch.equal(self.cache.data[3], torch.full((4, 59), 6.0)))


if __name__ == "__main__":
    unittest.main()
