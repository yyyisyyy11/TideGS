import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from strategies.tide_engine.distributed_metrics import (
    DistributedMetricsWriter,
    LegacyCudaMetricsCollector,
)


class _Context:
    rank = 0
    world_size = 2
    is_rank0 = True

    def all_gather_object(self, value):
        if value and all(isinstance(key, int) for key in value):
            peer = {
                33: {
                    "ssd_future_read_blocks": 3,
                    "ssd_future_read_bytes": 300,
                    "prefetch_ssd_ms": 6.0,
                    "gpu_d2h_blocks": 1,
                    "gpu_d2h_bytes": 50,
                    "gpu_d2h_ms": 2.0,
                }
            }
            return [value, peer]
        peer = dict(value)
        peer["rank"] = 1
        if "trigger" in value:
            peer["rounds"] = 3
            peer["input_bytes"] = 300
            peer["output_bytes"] = 200
            peer["reclaimed_bytes"] = 100
            peer["duration_ms"] = 8.0
            peer["free_space_gb_after"] = 140.0
        peer["optimizer_ms"] = 8.0
        return [value, peer]


class _SingleRankContext:
    rank = 0
    world_size = 1
    is_rank0 = True

    def all_gather_object(self, value):
        return [value]


class _Event:
    def __init__(self, elapsed_ms, ready=False):
        self.elapsed_ms = elapsed_ms
        self.ready = ready

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True

    def elapsed_time(self, end):
        assert end.ready
        return self.elapsed_ms


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class DistributedMetricsTest(unittest.TestCase):
    def test_legacy_collector_defers_cuda_read_and_writes_single_rank_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = DistributedMetricsWriter(
                args=SimpleNamespace(
                    log_folder=directory,
                    tide_detailed_metrics=True,
                ),
                context=_SingleRankContext(),
            )
            collector = LegacyCudaMetricsCollector(writer)
            forward_start = _Event(3.0, ready=True)
            forward_end = _Event(3.0, ready=False)
            backward_start = _Event(4.0, ready=True)
            backward_end = _Event(4.0, ready=False)
            adam_start = _Event(2.0, ready=True)
            adam_end = _Event(2.0, ready=False)
            collector.enqueue(
                {
                    "iteration": 1,
                    "local_cameras": 64,
                    "block_cull_backend": "cpu",
                    "next_plan_target_iteration": 65,
                    "next_plan_service_ms": 1.25,
                },
                {
                    "gsplat_forward_ms": [{
                        "start": forward_start,
                        "end": forward_end,
                        "start_ns": 100,
                        "name": "gaussian_cull_forward",
                    }],
                    "backward_ms": [{
                        "start": backward_start,
                        "end": backward_end,
                        "start_ns": 200,
                        "name": "backward",
                    }],
                    "optimizer_ms": [{
                        "start": adam_start,
                        "end": adam_end,
                        "start_ns": 300,
                        "name": "adam",
                    }],
                },
            )
            self.assertFalse((Path(directory) / "metrics_batch_rank0.tsv").exists())

            forward_end.ready = True
            backward_end.ready = True
            adam_end.ready = True
            collector.flush_ready()

            rank_rows = _read_rows(Path(directory) / "metrics_batch_rank0.tsv")
            global_rows = _read_rows(Path(directory) / "metrics_batch_global.tsv")
            self.assertEqual(rank_rows[0]["gsplat_forward_ms"], "3.0")
            self.assertEqual(rank_rows[0]["backward_ms"], "4.0")
            self.assertEqual(rank_rows[0]["optimizer_ms"], "2.0")
            self.assertEqual(rank_rows[0]["train_ms"], "9.0")
            self.assertEqual(rank_rows[0]["next_plan_target_iteration"], "65")
            self.assertEqual(global_rows[0]["world_size"], "1")
            self.assertEqual(global_rows[0]["next_plan_service_ms_max"], "1.25")

    def test_timeline_events_preserve_host_clock_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = DistributedMetricsWriter(
                args=SimpleNamespace(
                    log_folder=directory,
                    tide_detailed_metrics=True,
                ),
                context=_Context(),
            )
            writer.write_timeline_event(
                name="preview_plan",
                lane="cpu",
                start_ns=1_000,
                end_ns=1_025,
                iteration=1,
                target_iteration=5,
            )
            writer.write_io_events(
                [
                    {
                        "operation": "ssd_read_future",
                        "tier": "ssd",
                        "lane": "ssd",
                        "origin_iteration": 1,
                        "target_iteration": 5,
                        "blocks": 2,
                        "bytes": 200,
                        "service_ms": 0.02,
                        "start_ns": 2_000,
                        "end_ns": 2_020,
                        "thread_id": 99,
                    }
                ]
            )

            with open(
                Path(directory) / "timeline_events_rank0.jsonl",
                encoding="utf-8",
            ) as handle:
                events = [json.loads(line) for line in handle]

            self.assertEqual([event["name"] for event in events], [
                "preview_plan",
                "ssd_read_future",
            ])
            self.assertEqual(events[0]["duration_ns"], 25)
            self.assertEqual(events[1]["lane"], "ssd")
            self.assertEqual(events[1]["thread_id"], 99)

    def test_memory_points_are_rank_local_and_detailed_only(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = DistributedMetricsWriter(
                args=SimpleNamespace(
                    log_folder=directory,
                    tide_detailed_metrics=True,
                ),
                context=_SingleRankContext(),
            )
            writer.write_memory_point(
                iteration=65,
                phase="after_backward",
                microbatch_index=2,
                microbatch_count=4,
                camera_uids=[17, 23],
                active_gaussians=1234,
                rank_active_blocks=12,
                rank_resident_blocks=18,
                gpu_slot_capacity_blocks=32,
                cuda_allocated_bytes=100,
                cuda_reserved_bytes=200,
                cuda_peak_allocated_bytes=300,
                cuda_peak_reserved_bytes=400,
                cuda_free_bytes=500,
                cuda_total_bytes=600,
            )
            with open(
                Path(directory) / "timeline_events_rank0.jsonl",
                encoding="utf-8",
            ) as handle:
                events = [json.loads(line) for line in handle]

            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(event["name"], "cuda_memory_point")
            self.assertEqual(event["lane"], "gpu")
            self.assertEqual(event["duration_ns"], 0)
            self.assertEqual(event["phase"], "after_backward")
            self.assertEqual(event["microbatch_index"], 2)
            self.assertEqual(event["microbatch_count"], 4)
            self.assertEqual(event["camera_uids"], [17, 23])
            self.assertEqual(event["cuda_peak_allocated_bytes"], 300)

            disabled_writer = DistributedMetricsWriter(
                args=SimpleNamespace(
                    log_folder=directory,
                    tide_detailed_metrics=False,
                ),
                context=_SingleRankContext(),
            )
            disabled_writer.write_memory_point(
                iteration=65,
                phase="batch_start",
                cuda_allocated_bytes=1,
            )
            with open(
                Path(directory) / "timeline_events_rank0.jsonl",
                encoding="utf-8",
            ) as handle:
                self.assertEqual(len(list(handle)), 1)

    def test_async_io_uses_causal_iteration_and_global_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = DistributedMetricsWriter(
                args=SimpleNamespace(
                    log_folder=directory,
                    tide_detailed_metrics=True,
                ),
                context=_Context(),
            )
            writer.write_io_events(
                [
                    {
                        "operation": "ssd_read_future",
                        "tier": "ssd",
                        "origin_iteration": 1,
                        "target_iteration": 33,
                        "blocks": 2,
                        "bytes": 200,
                        "service_ms": 4.0,
                    },
                    {
                        "operation": "cpu_materialize_future",
                        "tier": "cpu_cache",
                        "origin_iteration": 1,
                        "target_iteration": 33,
                        "blocks": 2,
                        "bytes": 200,
                        "service_ms": 1.0,
                    }
                ]
            )
            row = {
                "iteration": 33,
                "predicted_stream_in_blocks": 5,
                "prediction_missing_blocks": 1,
                "prediction_extra_blocks": 2,
                "prediction_replanned": 1,
                "prediction_repair_ms": 4.0,
                "prediction_exact_plan_ms": 3.0,
                "block_cull_backend": "gpu",
                "block_cull_gpu_kernel_ms": 1.5,
                "block_cull_gpu_d2h_ms": 0.5,
                "block_cull_cache_hit_cameras": 2,
                "block_cull_gpu_cameras": 3,
                "block_cull_output_blocks": 12,
                "gpu_slot_capacity_blocks": 4,
                "gpu_slot_growth_blocks": 1,
                "h2d_bytes": 400,
                "block_reader_foreground_ms": 7.0,
                "ssd_urgent_read_ms": 1.5,
                "prefetch_inflight_wait_ms": 1.0,
                "optimizer_ms": 6.0,
            }
            writer.write_batch(row)
            writer.write_batch({**row, "iteration": 35})
            writer.write_io_events(
                [
                    {
                        "operation": "gpu_d2h",
                        "tier": "gpu_to_cpu",
                        "origin_iteration": 33,
                        "target_iteration": None,
                        "blocks": 2,
                        "bytes": 200,
                        "service_ms": 3.0,
                    }
                ]
            )
            writer.finalize_async_metrics()

            rank_rows = _read_rows(
                Path(directory) / "metrics_batch_rank0.tsv"
            )
            self.assertNotIn("ssd_future_read_blocks", rank_rows[0])

            global_rows = _read_rows(
                Path(directory) / "metrics_batch_global.tsv"
            )
            self.assertEqual(global_rows[0]["gpu_slot_capacity_blocks"], "8.0")
            self.assertEqual(global_rows[0]["gpu_slot_growth_blocks"], "2.0")
            self.assertEqual(global_rows[0]["h2d_bytes"], "800.0")
            self.assertEqual(global_rows[0]["predicted_stream_in_blocks"], "5.0")
            self.assertEqual(global_rows[0]["prediction_missing_blocks"], "1.0")
            self.assertEqual(global_rows[0]["prediction_extra_blocks"], "2.0")
            self.assertEqual(global_rows[0]["prediction_replanned"], "1.0")
            self.assertEqual(rank_rows[0]["block_cull_backend"], "gpu")
            self.assertEqual(global_rows[0]["block_cull_backend"], "gpu")
            self.assertEqual(global_rows[0]["block_cull_gpu_cameras"], "6.0")
            self.assertEqual(global_rows[0]["block_cull_output_blocks"], "24.0")
            self.assertEqual(global_rows[0]["block_cull_gpu_kernel_ms_max"], "1.5")
            self.assertEqual(global_rows[0]["block_cull_gpu_d2h_ms_mean"], "0.5")
            self.assertEqual(global_rows[0]["block_reader_foreground_ms_max"], "7.0")
            self.assertEqual(global_rows[0]["ssd_urgent_read_ms_max"], "1.5")
            self.assertEqual(global_rows[0]["prefetch_inflight_wait_ms_mean"], "1.0")
            self.assertEqual(global_rows[0]["prediction_repair_ms_max"], "4.0")
            self.assertEqual(
                global_rows[0]["prediction_exact_plan_ms_mean"], "3.0"
            )
            self.assertEqual(global_rows[0]["optimizer_ms_max"], "8.0")
            self.assertEqual(global_rows[0]["optimizer_ms_mean"], "7.0")

            async_rank = _read_rows(
                Path(directory) / "metrics_async_iteration_rank0.tsv"
            )
            self.assertEqual(async_rank[0]["iteration"], "33")
            self.assertEqual(async_rank[0]["ssd_future_read_blocks"], "2.0")
            self.assertEqual(async_rank[0]["ssd_future_read_bytes"], "200.0")
            self.assertEqual(async_rank[0]["prefetch_ssd_ms"], "4.0")
            self.assertEqual(async_rank[0]["prefetch_cpu_ms"], "1.0")
            self.assertEqual(async_rank[0]["gpu_d2h_ms"], "3.0")

            async_global = _read_rows(
                Path(directory) / "metrics_async_iteration_global.tsv"
            )
            self.assertEqual(async_global[0]["ssd_future_read_bytes"], "500.0")
            self.assertEqual(async_global[0]["prefetch_ssd_ms_max"], "6.0")
            self.assertEqual(async_global[0]["prefetch_ssd_ms_mean"], "5.0")
            self.assertEqual(async_global[0]["gpu_d2h_bytes"], "250.0")
            self.assertEqual(async_global[0]["gpu_d2h_ms_max"], "3.0")
            self.assertEqual(async_global[0]["gpu_d2h_ms_mean"], "2.5")

    def test_missing_required_attribution_fails_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = DistributedMetricsWriter(
                args=SimpleNamespace(
                    log_folder=directory,
                    tide_detailed_metrics=True,
                ),
                context=_Context(),
            )
            with self.assertRaisesRegex(ValueError, "target_iteration"):
                writer.write_io_events(
                    [
                        {
                            "operation": "ssd_read_future",
                            "tier": "ssd",
                            "origin_iteration": 1,
                            "target_iteration": None,
                            "blocks": 1,
                            "bytes": 100,
                            "service_ms": 2.0,
                        }
                    ]
                )

    def test_compaction_metrics_write_rank_and_global_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = DistributedMetricsWriter(
                args=SimpleNamespace(
                    log_folder=directory,
                    tide_detailed_metrics=True,
                ),
                context=_Context(),
            )
            writer.write_compaction(
                {
                    "trigger": "periodic",
                    "iteration": 5000,
                    "rounds": 2,
                    "before_patches": 12,
                    "after_patches": 8,
                    "input_bytes": 200,
                    "output_bytes": 120,
                    "reclaimed_bytes": 80,
                    "duration_ms": 6.0,
                    "actual_concurrency": 2,
                    "free_space_gb_before": 120.0,
                    "free_space_gb_after": 150.0,
                }
            )

            rank_row = _read_rows(
                Path(directory) / "metrics_compaction_rank0.tsv"
            )[0]
            self.assertEqual(rank_row["trigger"], "periodic")
            self.assertEqual(rank_row["iteration"], "5000")
            self.assertEqual(rank_row["output_bytes"], "120")

            global_row = _read_rows(
                Path(directory) / "metrics_compaction_global.tsv"
            )[0]
            self.assertEqual(global_row["rounds"], "5")
            self.assertEqual(global_row["input_bytes"], "500")
            self.assertEqual(global_row["output_bytes"], "320")
            self.assertEqual(global_row["duration_ms_max"], "8.0")
            self.assertEqual(global_row["duration_ms_mean"], "7.0")
            self.assertEqual(global_row["free_space_gb_after"], "140.0")


if __name__ == "__main__":
    unittest.main()
