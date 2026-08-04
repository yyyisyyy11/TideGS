import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from strategies.tide_engine import runtime
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    runtime = None


class _Range:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _Reader:
    def __init__(self):
        self.hints = []

    def hint_future(self, blocks):
        self.hints.append(list(blocks))
        return len(blocks)


class _DoubleBuffer:
    def __init__(self):
        self.prefetch_calls = []

    def start_prefetch(self, **kwargs):
        self.prefetch_calls.append(kwargs)


@unittest.skipIf(runtime is None, "requires the TideGS torch runtime")
class LegacyRuntimeMetricsTest(unittest.TestCase):
    def test_next_plan_returns_split_metrics_and_target_iteration(self):
        reader = _Reader()
        double_buffer = _DoubleBuffer()
        storage = SimpleNamespace(
            get_visible_blocks_batch=mock.Mock(return_value=(7, {10: [1, 2]})),
            wait_for_pending_gpu_copies=mock.Mock(),
        )
        block_sets = {
            "stream_in_blocks": [3],
            "next_resident_blocks": [1, 3],
            "keep_resident_blocks": [1],
            "evict_blocks": [2],
        }
        with mock.patch.object(
            runtime,
            "get_current_and_next_camera_batches",
            return_value=(None, SimpleNamespace(batch_indices=[10])),
        ), mock.patch.object(
            runtime,
            "compute_paper_block_sets",
            return_value=block_sets,
        ), mock.patch.object(runtime.torch.cuda.nvtx, "range", return_value=_Range()):
            result = runtime.plan_and_start_resident_prefetch(
                storage_adapter=storage,
                training_schedule=[0],
                iteration=1,
                batch_size=64,
                current_block_ids=[1],
                schedule_ordering="trajectory",
                current_resident_blocks=[1, 2],
                current_resident_recency_scores={},
                resident_selection_policy="topc_balanced",
                resident_lambda=0.3,
                resident_recency_decay=0.95,
                resident_capacity_blocks=8,
                balanced_seed_fraction=0.25,
                active_block_reader=reader,
                double_buffer=double_buffer,
            )

        metrics = result["next_plan_metrics"]
        self.assertEqual(metrics["next_plan_target_iteration"], 65)
        self.assertGreaterEqual(metrics["next_plan_cull_ms"], 0.0)
        self.assertGreaterEqual(metrics["next_plan_select_ms"], 0.0)
        self.assertGreaterEqual(metrics["next_plan_service_ms"], 0.0)
        self.assertEqual(
            set(result["next_plan_timeline"]),
            {"cull", "select", "hint_submit", "buffer_submit"},
        )
        self.assertEqual(reader.hints, [[3]])
        self.assertEqual(double_buffer.prefetch_calls[0]["iteration"], 65)


if __name__ == "__main__":
    unittest.main()
