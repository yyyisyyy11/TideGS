import unittest

from storage.compaction_scheduler import (
    _rank_groups,
    crossed_periodic_iteration,
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
    def __init__(self, *, patches=12):
        self.patches = patches

    def get_stats(self):
        return {
            "num_patches": self.patches,
        }

    def estimate_next_compaction_output_bytes(self):
        return 4 * (1024 ** 3) if self.patches > 1 else 0

    def compact_to_patch_count(self, *, target_patch_files):
        before = self.patches
        self.patches = min(self.patches, target_patch_files)
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

    def test_rank_groups_split_into_waves(self):
        self.assertEqual(
            _rank_groups(4, requested_concurrency=2),
            [[0, 1], [2, 3]],
        )
        self.assertEqual(
            _rank_groups(4, requested_concurrency=1),
            [[0], [1], [2], [3]],
        )
        # Concurrency is clamped to the world size.
        self.assertEqual(
            _rank_groups(2, requested_concurrency=8),
            [[0, 1]],
        )
        # Requests below one are raised to one rather than producing empty waves.
        self.assertEqual(
            _rank_groups(3, requested_concurrency=0),
            [[0], [1], [2]],
        )

    def test_periodic_compaction_reaches_low_watermark(self):
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
        )
        self.assertEqual(result["trigger"], "periodic")
        self.assertEqual(result["iteration"], 5000)
        self.assertEqual(result["after_patches"], 8)
        self.assertEqual(adapter.drains, 1)
        self.assertEqual(adapter.read_waits, 1)

    def test_no_trigger_returns_none(self):
        """Without a periodic crossing or a forced trigger there is nothing to do.

        This is the only remaining reason the scheduler can no-op: the
        free-space emergency trigger was removed because it compared a
        filesystem-wide reading against a threshold no cluster could reach.
        """
        storage = _Storage()
        adapter = _Adapter(storage)
        self.assertIsNone(
            run_compaction_maintenance(
                context=_Context(),
                storage_adapter=adapter,
                iteration=3001,
                periodic_iteration=None,
                target_patch_files=8,
                rank_concurrency=2,
            )
        )
        self.assertEqual(adapter.drains, 0)

    def test_forced_trigger_runs_without_periodic_crossing(self):
        storage = _Storage()
        adapter = _Adapter(storage)
        result = run_compaction_maintenance(
            context=_Context(),
            storage_adapter=adapter,
            iteration=3001,
            periodic_iteration=None,
            target_patch_files=8,
            rank_concurrency=2,
            forced_trigger="shutdown",
            flush_dirty_cache=True,
        )
        self.assertEqual(result["trigger"], "shutdown")
        self.assertEqual(result["after_patches"], 8)


if __name__ == "__main__":
    unittest.main()
