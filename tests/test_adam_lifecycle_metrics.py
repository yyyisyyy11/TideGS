import unittest

from utils.adam_lifecycle_metrics import (
    AdamLifecycleHistogram,
    increment_usage_counts,
    summarize_usage_histogram,
)


class AdamLifecycleMetricsTest(unittest.TestCase):
    def test_same_row_can_be_counted_multiple_times(self):
        counts = [0, 0, 0]
        increment_usage_counts(counts, [1, 1, 2])
        self.assertEqual(counts, [0, 2, 1])

    def test_histogram_summary_uses_nearest_rank_percentiles(self):
        summary = summarize_usage_histogram({1: 2, 2: 1, 5: 1})

        self.assertEqual(summary['adam_lifecycle_samples'], 4)
        self.assertEqual(summary['adam_lifecycle_updates_total'], 9)
        self.assertAlmostEqual(summary['adam_lifecycle_mean_uses'], 2.25)
        self.assertEqual(summary['adam_lifecycle_max_uses'], 5)
        self.assertEqual(summary['adam_lifecycle_p50_uses'], 1)
        self.assertEqual(summary['adam_lifecycle_p90_uses'], 5)
        self.assertEqual(summary['adam_lifecycle_p95_uses'], 5)
        self.assertEqual(summary['adam_lifecycle_p99_uses'], 5)
        self.assertAlmostEqual(summary['adam_lifecycle_one_shot_pct'], 50.0)
        self.assertAlmostEqual(summary['adam_lifecycle_reuse_ge2_pct'], 50.0)
        self.assertAlmostEqual(summary['adam_lifecycle_reuse_ge5_pct'], 25.0)

    def test_completed_active_and_one_shot_lifetimes_do_not_duplicate(self):
        histogram = AdamLifecycleHistogram()
        histogram.finalize([3, 0])
        histogram.record_one_shot(1)
        active = [[1, 2, 0]]

        first = histogram.snapshot(active)
        second = histogram.snapshot(active)

        self.assertEqual(first, second)
        self.assertEqual(first['adam_lifecycle_samples'], 4)
        self.assertEqual(first['adam_lifecycle_updates_total'], 7)
        self.assertEqual(first['adam_lifecycle_max_uses'], 3)
        self.assertAlmostEqual(first['adam_lifecycle_one_shot_pct'], 50.0)

    def test_eviction_and_readmission_are_independent_lifetimes(self):
        histogram = AdamLifecycleHistogram()
        histogram.finalize([3])
        histogram.finalize([1])

        summary = histogram.snapshot()
        self.assertEqual(summary['adam_lifecycle_samples'], 2)
        self.assertEqual(summary['adam_lifecycle_updates_total'], 4)
        self.assertEqual(summary['adam_lifecycle_max_uses'], 3)
        self.assertAlmostEqual(summary['adam_lifecycle_mean_uses'], 2.0)

    def test_untouched_rows_are_excluded(self):
        histogram = AdamLifecycleHistogram()
        histogram.finalize([0] * 8)

        summary = histogram.snapshot()
        self.assertEqual(summary['adam_lifecycle_samples'], 0)
        self.assertEqual(summary['adam_lifecycle_max_uses'], 0)


if __name__ == '__main__':
    unittest.main()
