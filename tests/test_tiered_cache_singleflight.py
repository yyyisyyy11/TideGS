import threading
import time
import unittest
import importlib.util
import sys
import types
from collections import Counter
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

    class _FakeTensor:
        def __init__(self, shape, value):
            self.shape = tuple(shape)
            self.value = float(value)

        def clone(self):
            return _FakeTensor(self.shape, self.value)

        def numel(self):
            total = 1
            for dim in self.shape:
                total *= dim
            return total

        def element_size(self):
            return 4

        def dim(self):
            return len(self.shape)

    torch = types.SimpleNamespace(
        Tensor=_FakeTensor,
        full=lambda shape, value: _FakeTensor(shape, value),
        equal=lambda left, right: (
            isinstance(left, _FakeTensor)
            and isinstance(right, _FakeTensor)
            and left.shape == right.shape
            and left.value == right.value
        ),
        is_tensor=lambda value: isinstance(value, _FakeTensor),
    )
    sys.modules["torch"] = torch

sys.modules.setdefault("psutil", types.SimpleNamespace())

_MODULE_PATH = Path(__file__).resolve().parents[1] / "storage" / "tiered_cache_manager.py"
_SPEC = importlib.util.spec_from_file_location("tiered_cache_manager_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
TieredCacheManager = _MODULE.TieredCacheManager


class FakeStorage:
    def __init__(self, block_size=2, point_dim=3):
        self.block_size = block_size
        self.point_dim = point_dim
        self.data = {
            block_id: torch.full((block_size, point_dim), float(block_id))
            for block_id in range(32)
        }
        self.read_counts = Counter()
        self.call_count = 0
        self.lock = threading.Lock()
        self.block_first_read = False
        self.fail_first_read = False
        self.read_started = threading.Event()
        self.release_read = threading.Event()

    def read_blocks(self, block_ids):
        block_ids = [int(block_id) for block_id in block_ids]
        with self.lock:
            self.call_count += 1
            call_index = self.call_count
            for block_id in block_ids:
                self.read_counts[block_id] += 1

        if self.block_first_read and call_index == 1:
            self.read_started.set()
            if not self.release_read.wait(timeout=2.0):
                raise TimeoutError("test timed out waiting to release first read")

        if self.fail_first_read and call_index == 1:
            raise RuntimeError("injected read failure")

        return {block_id: self.data[block_id].clone() for block_id in block_ids}

    def write_patch(self, dirty_blocks, block_versions=None):
        return None


class VersionedBlockingStorage(FakeStorage):
    def __init__(self, block_size=2, point_dim=3):
        super().__init__(block_size=block_size, point_dim=point_dim)
        self.versions = {block_id: 1 for block_id in self.data}
        self.block_first_read = True

    def read_blocks_with_versions(self, block_ids):
        block_ids = [int(block_id) for block_id in block_ids]
        with self.lock:
            self.call_count += 1
            call_index = self.call_count
            for block_id in block_ids:
                self.read_counts[block_id] += 1
            blocks = {
                block_id: self.data[block_id].clone()
                for block_id in block_ids
            }
            versions = {
                block_id: int(self.versions[block_id])
                for block_id in block_ids
            }

        if call_index == 1:
            self.read_started.set()
            if not self.release_read.wait(timeout=2.0):
                raise TimeoutError("test timed out waiting to release first read")
        return blocks, versions

    def get_block_versions(self, block_ids):
        with self.lock:
            return {
                int(block_id): int(self.versions[int(block_id)])
                for block_id in block_ids
            }


class TieredCacheSingleFlightTest(unittest.TestCase):
    def make_cache(self, storage):
        cache = TieredCacheManager(
            storage,
            max_ram_gb=1.0,
            block_size=storage.block_size,
            point_dim=storage.point_dim,
            verbose=False,
        )
        self.addCleanup(cache.shutdown)
        return cache

    def run_prefetch_in_thread(self, cache, block_ids):
        result = {}
        errors = []

        def target():
            try:
                result.update(cache.prefetch(block_ids))
            except Exception as exc:  # pragma: no cover - asserted by tests
                errors.append(exc)

        thread = threading.Thread(target=target)
        thread.start()
        return thread, result, errors

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required")
    def test_dirty_batch_uses_independently_owned_blocks(self):
        storage = FakeStorage(block_size=2, point_dim=3)
        cache = self.make_cache(storage)
        source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        expected = source.clone()

        self.assertEqual(
            cache.upsert_dirty_block_batch({5: source[:2], 6: source[2:]}),
            2,
        )
        source.add_(100.0)

        self.assertTrue(torch.equal(cache.cache_data[5], expected[:2]))
        self.assertTrue(torch.equal(cache.cache_data[6], expected[2:]))
        self.assertNotEqual(
            cache.cache_data[5].untyped_storage().data_ptr(),
            cache.cache_data[6].untyped_storage().data_ptr(),
        )
        self.assertEqual(cache.dirty_set, {5, 6})
        self.assertEqual(cache._get_ram_usage(), expected.numel() * expected.element_size())

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required")
    def test_ram_usage_tracks_partial_blocks_and_lru_eviction(self):
        storage = FakeStorage(block_size=4, point_dim=3)
        cache = self.make_cache(storage)
        full = torch.ones((4, 3), dtype=torch.float32)
        partial = torch.ones((2, 3), dtype=torch.float32)
        cache.upsert_dirty_block_batch({5: full, 6: partial})
        expected_bytes = (full.numel() + partial.numel()) * full.element_size()
        self.assertEqual(cache._get_ram_usage(), expected_bytes)

        cache.dirty_set.clear()
        cache.evict_and_flush(
            target_ram_bytes=partial.numel() * partial.element_size()
        )
        self.assertEqual(
            cache._get_ram_usage(),
            partial.numel() * partial.element_size(),
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required")
    def test_managed_ram_deduplicates_cache_and_flushing_storage(self):
        storage = FakeStorage(block_size=2, point_dim=3)
        cache = self.make_cache(storage)
        source = torch.ones((2, 3), dtype=torch.float32)
        cache.upsert_dirty_block_batch({5: source})
        cached = cache.cache_data[5]
        with cache.flushing_lock:
            cache.flushing_buffer[5] = (cached, time.time(), 1)

        expected_bytes = cached.numel() * cached.element_size()
        self.assertEqual(cache._get_ram_usage(), expected_bytes)
        self.assertEqual(cache._get_flushing_ram_usage(), expected_bytes)
        self.assertEqual(cache._get_managed_ram_usage(), expected_bytes)

    def test_stale_cache_commit_cannot_overwrite_newer_version(self):
        storage = FakeStorage(block_size=2, point_dim=3)
        cache = self.make_cache(storage)
        newer = torch.full((2, 3), 20.0)
        stale = torch.full((2, 3), 10.0)

        self.assertEqual(
            cache.upsert_dirty_block_batch(
                {5: newer},
                block_versions={5: 2},
                origin_iteration=33,
            ),
            1,
        )
        self.assertEqual(
            cache.upsert_dirty_block_batch(
                {5: stale},
                block_versions={5: 1},
                origin_iteration=1,
            ),
            0,
        )
        self.assertTrue(torch.equal(cache.cache_data[5], newer))
        self.assertEqual(cache.get_block_versions([5]), {5: 2})
        self.assertEqual(cache.block_origin_iterations[5], 33)
        self.assertEqual(
            cache._get_ram_usage(),
            newer.numel() * newer.element_size(),
        )

    def test_future_read_serves_urgent_without_second_ssd_read(self):
        storage = FakeStorage()
        storage.block_first_read = True
        cache = self.make_cache(storage)

        self.assertEqual(cache.prefetch_future([7]), 1)
        self.assertTrue(storage.read_started.wait(timeout=2.0))

        thread, result, errors = self.run_prefetch_in_thread(cache, [7])
        time.sleep(0.05)
        self.assertTrue(thread.is_alive())

        storage.release_read.set()
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(torch.equal(result[7], storage.data[7]))
        self.assertEqual(storage.read_counts[7], 1)

        stats = cache.get_stats()
        self.assertEqual(stats["urgent_prefetch_blocks"], 0)
        self.assertEqual(stats["future_prefetch_reserved"], 1)
        self.assertGreaterEqual(stats["inflight_wait_blocks"], 1)
        events = cache.drain_io_events()
        self.assertEqual(
            [event["operation"] for event in events[-2:]],
            ["ssd_read_future", "cpu_materialize_future"],
        )

    def test_future_hint_skips_block_already_claimed_by_urgent(self):
        storage = FakeStorage()
        storage.block_first_read = True
        cache = self.make_cache(storage)

        thread, result, errors = self.run_prefetch_in_thread(cache, [3])
        self.assertTrue(storage.read_started.wait(timeout=2.0))

        self.assertEqual(cache.prefetch_future([3]), 0)

        storage.release_read.set()
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(torch.equal(result[3], storage.data[3]))
        self.assertEqual(storage.read_counts[3], 1)
        self.assertGreaterEqual(cache.get_stats()["future_prefetch_skipped"], 1)

    def test_two_urgent_prefetches_share_one_ssd_read(self):
        storage = FakeStorage()
        storage.block_first_read = True
        cache = self.make_cache(storage)

        thread_a, result_a, errors_a = self.run_prefetch_in_thread(cache, [5])
        self.assertTrue(storage.read_started.wait(timeout=2.0))

        thread_b, result_b, errors_b = self.run_prefetch_in_thread(cache, [5])
        time.sleep(0.05)
        self.assertTrue(thread_b.is_alive())

        storage.release_read.set()
        thread_a.join(timeout=2.0)
        thread_b.join(timeout=2.0)

        self.assertFalse(thread_a.is_alive())
        self.assertFalse(thread_b.is_alive())
        self.assertEqual(errors_a, [])
        self.assertEqual(errors_b, [])
        self.assertTrue(torch.equal(result_a[5], storage.data[5]))
        self.assertTrue(torch.equal(result_b[5], storage.data[5]))
        self.assertEqual(storage.read_counts[5], 1)
        self.assertEqual(cache.get_stats()["urgent_prefetch_blocks"], 1)

    def test_flushing_buffer_hit_does_not_touch_ssd_or_inflight(self):
        storage = FakeStorage()
        cache = self.make_cache(storage)
        tensor = torch.full((storage.block_size, storage.point_dim), 9.0)

        with cache.flushing_lock:
            cache.flushing_buffer[9] = (tensor, time.time(), 0)
        try:
            result = cache.prefetch([9])
            self.assertTrue(torch.equal(result[9], tensor))
            self.assertEqual(storage.read_counts[9], 0)
            self.assertEqual(cache.get_stats()["inflight_read_blocks"], 0)
        finally:
            with cache.flushing_lock:
                cache.flushing_buffer.pop(9, None)

    def test_dirty_eviction_publishes_flushing_before_releasing_cache_lock(self):
        storage = FakeStorage()
        cache = self.make_cache(storage)
        expected = torch.full(
            (storage.block_size, storage.point_dim),
            5.0,
        )
        cache.upsert_dirty_block_batch(
            {5: expected},
            block_versions={5: 2},
            origin_iteration=33,
        )
        popped = threading.Event()
        original_pop = cache._pop_lru_cache_block_locked

        def observed_pop():
            value = original_pop()
            popped.set()
            return value

        cache._pop_lru_cache_block_locked = observed_pop
        cache._enqueue_dirty_flush = lambda *args, **kwargs: True
        cache.flushing_lock.acquire()
        thread = threading.Thread(target=lambda: cache.evict_and_flush(num_blocks=1))
        thread.start()
        self.assertTrue(popped.wait(timeout=2.0))

        acquired = cache.cache_lock.acquire(blocking=False)
        if acquired:
            cache.cache_lock.release()
        self.assertFalse(acquired)

        cache.flushing_lock.release()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        tensor, source = cache._lookup_cached_or_flushing(5)
        self.assertEqual(source, "flushing")
        self.assertTrue(torch.equal(tensor, expected))

        with cache.cache_lock:
            cache.dirty_set.discard(5)
        with cache.flushing_lock:
            cache.flushing_buffer.pop(5, None)

    def test_urgent_stale_ssd_read_prefers_newer_flushing_version(self):
        storage = VersionedBlockingStorage()
        cache = self.make_cache(storage)
        thread, result, errors = self.run_prefetch_in_thread(cache, [5])
        self.assertTrue(storage.read_started.wait(timeout=2.0))

        newer = torch.full((storage.block_size, storage.point_dim), 20.0)
        with cache.cache_lock:
            cache.block_versions[5] = 2
            with cache.flushing_lock:
                cache.flushing_buffer[5] = (newer, time.time(), 2)
        storage.release_read.set()
        thread.join(timeout=2.0)

        self.assertEqual(errors, [])
        self.assertTrue(torch.equal(result[5], newer))
        self.assertEqual(storage.read_counts[5], 1)
        with cache.flushing_lock:
            cache.flushing_buffer.pop(5, None)

    def test_urgent_stale_ssd_read_retries_after_newer_ram_is_released(self):
        storage = VersionedBlockingStorage()
        cache = self.make_cache(storage)
        thread, result, errors = self.run_prefetch_in_thread(cache, [6])
        self.assertTrue(storage.read_started.wait(timeout=2.0))

        newer = torch.full((storage.block_size, storage.point_dim), 30.0)
        with storage.lock:
            storage.data[6] = newer
            storage.versions[6] = 2
        with cache.cache_lock:
            cache.block_versions[6] = 2
        storage.release_read.set()
        thread.join(timeout=2.0)

        self.assertEqual(errors, [])
        self.assertTrue(torch.equal(result[6], newer))
        self.assertEqual(storage.read_counts[6], 2)
        self.assertEqual(cache.get_block_versions([6]), {6: 2})

    def test_future_stale_ssd_read_prefers_newer_flushing_version(self):
        storage = VersionedBlockingStorage()
        cache = self.make_cache(storage)
        self.assertEqual(cache.prefetch_future([7], target_iteration=33), 1)
        self.assertTrue(storage.read_started.wait(timeout=2.0))

        newer = torch.full((storage.block_size, storage.point_dim), 40.0)
        with cache.cache_lock:
            cache.block_versions[7] = 2
            with cache.flushing_lock:
                cache.flushing_buffer[7] = (newer, time.time(), 2)
        storage.release_read.set()
        cache.future_prefetch_queue.join()

        result = cache.prefetch([7])
        self.assertTrue(torch.equal(result[7], newer))
        self.assertEqual(storage.read_counts[7], 1)
        with cache.flushing_lock:
            cache.flushing_buffer.pop(7, None)

    def test_partitioned_io_service_time_is_split_by_actual_bytes(self):
        storage = FakeStorage(block_size=2, point_dim=3)
        cache = self.make_cache(storage)
        full = torch.full((2, 3), 1.0)
        partial = torch.full((1, 3), 2.0)

        cache.record_partitioned_io_events(
            operation="ssd_write_async",
            tier="ssd",
            block_tensors={5: full, 6: partial},
            block_origin_iterations={5: 1, 6: 33},
            service_ms=9.0,
        )

        events = sorted(
            cache.drain_io_events(),
            key=lambda event: event["origin_iteration"],
        )
        self.assertEqual(events[0]["blocks"], 1)
        self.assertEqual(events[0]["bytes"], full.numel() * full.element_size())
        self.assertAlmostEqual(events[0]["service_ms"], 6.0)
        self.assertEqual(events[1]["blocks"], 1)
        self.assertEqual(
            events[1]["bytes"],
            partial.numel() * partial.element_size(),
        )
        self.assertAlmostEqual(events[1]["service_ms"], 3.0)

    def test_mixed_origin_dirty_flush_emits_one_event_per_origin(self):
        storage = FakeStorage(block_size=2, point_dim=3)
        cache = self.make_cache(storage)
        full = torch.full((2, 3), 1.0)
        partial = torch.full((1, 3), 2.0)
        cache.upsert_dirty_block_batch(
            {5: full, 6: partial},
            block_versions={5: 2, 6: 3},
            block_origin_iterations={5: 1, 6: 33},
        )

        self.assertEqual(cache.submit_dirty_blocks([5, 6]), 2)
        cache.flush_queue.join()

        events = sorted(
            (
                event
                for event in cache.drain_io_events()
                if event["operation"] == "ssd_write_async"
            ),
            key=lambda event: event["origin_iteration"],
        )
        self.assertEqual(
            [(event["origin_iteration"], event["blocks"]) for event in events],
            [(1, 1), (33, 1)],
        )
        self.assertEqual(events[0]["bytes"], full.numel() * full.element_size())
        self.assertEqual(
            events[1]["bytes"],
            partial.numel() * partial.element_size(),
        )

    def test_future_failure_wakes_urgent_and_allows_fallback_read(self):
        storage = FakeStorage()
        storage.block_first_read = True
        storage.fail_first_read = True
        cache = self.make_cache(storage)

        self.assertEqual(cache.prefetch_future([11]), 1)
        self.assertTrue(storage.read_started.wait(timeout=2.0))

        thread, result, errors = self.run_prefetch_in_thread(cache, [11])
        time.sleep(0.05)
        self.assertTrue(thread.is_alive())

        storage.release_read.set()
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(torch.equal(result[11], storage.data[11]))
        self.assertEqual(storage.read_counts[11], 2)

        stats = cache.get_stats()
        self.assertEqual(stats["future_prefetch_errors"], 1)
        self.assertGreaterEqual(stats["inflight_fallback_blocks"], 1)


if __name__ == "__main__":
    unittest.main()
