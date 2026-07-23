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

    def all_gather_object(self, rank_row):
        peer = dict(rank_row)
        peer["rank"] = 1
        peer["ssd_future_read_bytes"] = 300
        peer["ssd_future_read_blocks"] = 3
        peer["optimizer_ms"] = 8.0
        return [rank_row, peer]


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class DistributedMetricsTest(unittest.TestCase):
    def test_future_io_is_attributed_to_target_batch_and_globally_summed(self):
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
                    }
                ]
            )
            row = {
                "iteration": 33,
                "ssd_future_read_blocks": 0,
                "ssd_future_read_bytes": 0,
                "optimizer_ms": 6.0,
            }
            writer.write_batch(row)

            rank_rows = _read_rows(
                Path(directory) / "metrics_batch_rank0.tsv"
            )
            self.assertEqual(rank_rows[0]["ssd_future_read_blocks"], "2.0")
            self.assertEqual(rank_rows[0]["ssd_future_read_bytes"], "200.0")

            global_rows = _read_rows(
                Path(directory) / "metrics_batch_global.tsv"
            )
            self.assertEqual(global_rows[0]["ssd_future_read_bytes"], "500.0")
            self.assertEqual(global_rows[0]["optimizer_ms_max"], "8.0")
            self.assertEqual(global_rows[0]["optimizer_ms_mean"], "7.0")


if __name__ == "__main__":
    unittest.main()
