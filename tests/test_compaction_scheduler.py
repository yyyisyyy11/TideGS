import unittest

from storage.compaction_scheduler import (
    _rank_groups,
    crossed_periodic_iteration,
    resolve_emergency_free_gb,
    run_compaction_maintenance,
)


class _Context:
    rank = 0
    world_size = 1

    def __init__(self):
        self.barriers = 0

    def all_gather_object(self, value):
        return [value]

    def barrier(self):
        self.barriers += 1


class _Storage:
    min_free_bytes = 64 * (1024 ** 3)

    def __init__(self, *, patches=12, free_gb=256):
        self.patches = patches
        self.free_gb = free_gb

    def get_stats(self):
        return {
            "num_patches": self.patches,
            "free_space_gb": self.free_gb,
        }

    def estimate_next_compaction_output_bytes(self):
        return 4 * (1024 ** 3) if self.patches > 1 else 0

    def compact_to_patch_count(self, *, target_patch_files):
        before = self.patches
        self.patches = min(self.patches, target_patch_files)
        self.free_gb += 32
        return {
            "rounds": 2,
            "before_patches": before,
            "after_patches": self.patches,
            "input_bytes": 20,
            "output_bytes": 12,
            "reclaimed_bytes": 8,
        }


class _Adapter:
    def __init__(self, storage):
        self.storage = storage
        self.drains = 0
        self.read_waits = 0

    def drain_cache_writebacks(self):
        self.drains += 1

    def drain_storage_writebacks(self):
        self.drains += 1

    def wait_for_storage_reads(self):
        self.read_waits += 1

    def flush_dirty_cache_to_storage(self):
        self.drains += 1


class CompactionSchedulerTest(unittest.TestCase):
    def test_bsz32_crosses_5000_once_and_resume_does_not_retrigger(self):
        self.assertIsNone(
            crossed_periodic_iteration(
                iteration=4961,
                batch_size=32,
                interval_iterations=5000,
            )
        )
        self.assertEqual(
            crossed_periodic_iteration(
                iteration=4993,
                batch_size=32,
                interval_iterations=5000,
            ),
            5000,
        )
        self.assertIsNone(
            crossed_periodic_iteration(
                iteration=5025,
                batch_size=32,
                interval_iterations=5000,
            )
        )
        self.assertEqual(
            crossed_periodic_iteration(
                iteration=9985,
                batch_size=32,
                interval_iterations=5000,
            ),
            10000,
        )
        self.assertIsNone(
            crossed_periodic_iteration(
                iteration=4993,
                batch_size=32,
                interval_iterations=0,
            )
        )

    def test_rank_groups_use_two_waves_and_fall_back_to_single_rank(self):
        gib = 1024 ** 3
        states = [
            {
                "free_bytes": 256 * gib,
                "min_free_bytes": 64 * gib,
                "num_patches": 12,
                "estimated_output_bytes": 16 * gib,
            }
            for _ in range(4)
        ]
        self.assertEqual(
            _rank_groups(
                states,
                requested_concurrency=2,
                target_patch_files=8,
            ),
            [[0, 1], [2, 3]],
        )
        for state in states:
            state["free_bytes"] = 80 * gib
        self.assertEqual(
            _rank_groups(
                states,
                requested_concurrency=2,
                target_patch_files=8,
            ),
            [[0], [1], [2], [3]],
        )

    def test_periodic_and_emergency_compaction_reach_low_watermark(self):
        context = _Context()
        storage = _Storage()
        adapter = _Adapter(storage)
        result = run_compaction_maintenance(
            context=context,
            storage_adapter=adapter,
            iteration=4993,
            periodic_iteration=5000,
            target_patch_files=8,
            rank_concurrency=2,
            emergency_free_gb=128,
        )
        self.assertEqual(result["trigger"], "periodic")
        self.assertEqual(result["iteration"], 5000)
        self.assertEqual(result["after_patches"], 8)
        self.assertEqual(adapter.drains, 1)
        self.assertEqual(adapter.read_waits, 1)

        emergency_storage = _Storage(free_gb=100)
        emergency_adapter = _Adapter(emergency_storage)
        result = run_compaction_maintenance(
            context=_Context(),
            storage_adapter=emergency_adapter,
            iteration=3001,
            periodic_iteration=None,
            target_patch_files=8,
            rank_concurrency=2,
            emergency_free_gb=128,
        )
        self.assertEqual(result["trigger"], "emergency")
        self.assertGreaterEqual(result["free_space_gb_after"], 128)

    def test_emergency_threshold_defaults_to_twice_reserve(self):
        self.assertEqual(
            resolve_emergency_free_gb(
                configured_gb=-1,
                min_free_gb=64,
            ),
            128,
        )


if __name__ == "__main__":
    unittest.main()
