import tempfile
import unittest
from pathlib import Path

from utils.optimizer_metrics import (
    OPTIMIZER_TIMING_FIELDS,
    write_optimizer_timing_metrics,
)


class OptimizerTimingMetricsTest(unittest.TestCase):
    def test_writes_header_rows_and_mean_updates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = write_optimizer_timing_metrics(
                model_path=tmpdir,
                iteration=1,
                batch_size=4,
                update_rule='resident_adam',
                updates_enabled=True,
                omega_wait_ms=1.25,
                optimizer_submit_ms=0.5,
                optimizer_cuda_ms=2.75,
                adam_lifecycle_accounting_ms=0.125,
                touched_rows=20,
                total_gaussians=100,
                session_row_updates_total=20,
            )
            second = write_optimizer_timing_metrics(
                model_path=tmpdir,
                iteration=5,
                batch_size=4,
                update_rule='resident_adam',
                updates_enabled=True,
                omega_wait_ms=0.0,
                optimizer_submit_ms=0.25,
                optimizer_cuda_ms=2.0,
                adam_lifecycle_accounting_ms=0.1,
                touched_rows=10,
                total_gaussians=100,
                session_row_updates_total=30,
            )

            self.assertAlmostEqual(first['session_mean_updates_per_gaussian'], 0.2)
            self.assertAlmostEqual(first['adam_lifecycle_accounting_ms'], 0.125)
            self.assertAlmostEqual(second['session_mean_updates_per_gaussian'], 0.3)
            lines = (Path(tmpdir) / 'optimizer_timing.tsv').read_text().splitlines()
            self.assertEqual(lines[0].split('\t'), list(OPTIMIZER_TIMING_FIELDS))
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[2].split('\t')[0], '1')


if __name__ == '__main__':
    unittest.main()
