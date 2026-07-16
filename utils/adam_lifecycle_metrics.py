import math
from typing import Dict, Iterable

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def increment_usage_counts(counts, row_indices) -> None:
    if np is not None and isinstance(counts, np.ndarray):
        rows = np.asarray(row_indices, dtype=np.intp)
        np.add.at(counts, rows, 1)
        return
    for row_index in row_indices:
        index = int(row_index)
        counts[index] = int(counts[index]) + 1


def merge_positive_usage_counts(
    histogram: Dict[int, int],
    counts,
) -> None:
    if np is not None:
        values = np.asarray(counts, dtype=np.uint32).reshape(-1)
        values = values[values > 0]
        if values.size == 0:
            return
        unique, frequencies = np.unique(values, return_counts=True)
        for value, frequency in zip(unique.tolist(), frequencies.tolist()):
            key = int(value)
            histogram[key] = int(histogram.get(key, 0)) + int(frequency)
        return

    for value in counts:
        key = int(value)
        if key > 0:
            histogram[key] = int(histogram.get(key, 0)) + 1


def _nearest_rank_percentile(histogram: Dict[int, int], percentile: float) -> int:
    sample_count = sum(int(frequency) for frequency in histogram.values())
    if sample_count <= 0:
        return 0
    target_rank = max(1, int(math.ceil(float(percentile) * sample_count)))
    cumulative = 0
    for usage_count in sorted(histogram):
        cumulative += int(histogram[usage_count])
        if cumulative >= target_rank:
            return int(usage_count)
    return int(max(histogram))


def summarize_usage_histogram(histogram: Dict[int, int]) -> Dict[str, float]:
    normalized = {
        int(usage_count): int(frequency)
        for usage_count, frequency in histogram.items()
        if int(usage_count) > 0 and int(frequency) > 0
    }
    sample_count = sum(normalized.values())
    update_count = sum(
        usage_count * frequency
        for usage_count, frequency in normalized.items()
    )
    if sample_count <= 0:
        return {
            'adam_lifecycle_samples': 0,
            'adam_lifecycle_updates_total': 0,
            'adam_lifecycle_mean_uses': 0.0,
            'adam_lifecycle_max_uses': 0,
            'adam_lifecycle_p50_uses': 0,
            'adam_lifecycle_p90_uses': 0,
            'adam_lifecycle_p95_uses': 0,
            'adam_lifecycle_p99_uses': 0,
            'adam_lifecycle_one_shot_pct': 0.0,
            'adam_lifecycle_reuse_ge2_pct': 0.0,
            'adam_lifecycle_reuse_ge5_pct': 0.0,
        }

    one_shot = int(normalized.get(1, 0))
    reuse_ge2 = sum(
        frequency for usage_count, frequency in normalized.items()
        if usage_count >= 2
    )
    reuse_ge5 = sum(
        frequency for usage_count, frequency in normalized.items()
        if usage_count >= 5
    )
    return {
        'adam_lifecycle_samples': int(sample_count),
        'adam_lifecycle_updates_total': int(update_count),
        'adam_lifecycle_mean_uses': float(update_count) / float(sample_count),
        'adam_lifecycle_max_uses': int(max(normalized)),
        'adam_lifecycle_p50_uses': _nearest_rank_percentile(normalized, 0.50),
        'adam_lifecycle_p90_uses': _nearest_rank_percentile(normalized, 0.90),
        'adam_lifecycle_p95_uses': _nearest_rank_percentile(normalized, 0.95),
        'adam_lifecycle_p99_uses': _nearest_rank_percentile(normalized, 0.99),
        'adam_lifecycle_one_shot_pct': 100.0 * one_shot / sample_count,
        'adam_lifecycle_reuse_ge2_pct': 100.0 * reuse_ge2 / sample_count,
        'adam_lifecycle_reuse_ge5_pct': 100.0 * reuse_ge5 / sample_count,
    }


class AdamLifecycleHistogram:
    def __init__(self):
        self._completed: Dict[int, int] = {}

    def finalize(self, counts) -> None:
        merge_positive_usage_counts(self._completed, counts)

    def record_one_shot(self, count: int) -> None:
        count = int(count)
        if count > 0:
            self._completed[1] = int(self._completed.get(1, 0)) + count

    def snapshot(self, active_counts: Iterable = ()) -> Dict[str, float]:
        combined = dict(self._completed)
        for counts in active_counts:
            merge_positive_usage_counts(combined, counts)
        return summarize_usage_histogram(combined)
