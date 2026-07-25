import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from strategies.tide_engine.distributed_metrics import DistributedMetricsWriter


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
        peer["optimizer_ms"] = 8.0
        return [value, peer]


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class DistributedMetricsTest(unittest.TestCase):
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
                "gpu_slot_capacity_blocks": 4,
                "gpu_slot_growth_blocks": 1,
                "h2d_bytes": 400,
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


if __name__ == "__main__":
    unittest.main()
