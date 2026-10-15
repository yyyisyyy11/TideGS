"""CSV-to-JSON behavior for complete-batch PC1 target overlap summaries."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


_SHARED = Path(__file__).resolve().parents[3] / 'experiment/timeline/scripts'
_BUNDLED = Path(__file__).resolve().parents[1] / 'tools/pc1_timeline'
_SCRIPT_DIR = Path(os.environ.get('TIDE_TIMELINE_SCRIPTS', str(
    _SHARED if _SHARED.is_dir() else _BUNDLED)))
_SCRIPT = _SCRIPT_DIR / 'analyze_pc1_submission_order.py'
if not _SCRIPT.is_file():
    raise unittest.SkipTest('timeline submission script unavailable; set TIDE_TIMELINE_SCRIPTS')


PREFIX = 'target_next_projection_c1'


def batch(index, **fields):
    return dict(rank=0, trace_batch=index, compute_nccl_cross_stream_overlap_ms=100, **fields)


def pair(index, value, *, status='assessable', target_type='projection_c1'):
    return dict(rank=0, trace_batch=index, cross_stream_overlap_ms=value,
                status=status, target_type=target_type)


class SubmissionOrderSummaryTest(unittest.TestCase):
    @staticmethod
    def write_csv(path, rows, default_fields):
        fields = list(dict.fromkeys(key for row in rows for key in row)) or default_fields
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, separators=(',', ':'))
                                 if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})

    def run_summary(self, batches, pairs):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_csv(root / 'pc1-overlap.batches.csv', batches, ['rank', 'trace_batch'])
            self.write_csv(root / 'pc1-overlap.pairs.csv', pairs, ['rank', 'trace_batch'])
            self.write_csv(root / 'pc1-overlap.stages.csv', [], ['rank', 'trace_batch', 'phase'])
            self.write_csv(root / 'pc1-overlap.kernels.csv', [], ['rank', 'trace_batch', 'phase'])
            result = subprocess.run([
                sys.executable, str(_SCRIPT), '--derived-dir', str(root),
                '--output', str(root / 'submission'),
            ], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / 'submission.json').read_text())
        return report['ranks'][0]

    def test_all_unknown_remains_null_instead_of_zero(self):
        result = self.run_summary(
            [batch(1), batch(2)],
            [pair(1, None, status='missing_phase'), pair(2, None, status='missing_runtime')])
        self.assertIsNone(result[PREFIX + '_overlap_ms_per_batch'])
        self.assertEqual(result[PREFIX + '_complete_batches'], 0)
        self.assertEqual(result[PREFIX + '_unknown_batches'], 2)
        self.assertEqual(result[PREFIX + '_declared_batches'], 2)
        self.assertEqual(result[PREFIX + '_coverage_fraction'], 0)
        self.assertTrue(all(row['overlap_ms'] is None for row in result[PREFIX + '_batch_results']))

    def test_mixed_batches_use_only_complete_target_batches_as_denominator(self):
        result = self.run_summary(
            [batch(1), batch(2), batch(3)],
            [pair(1, 4.0), pair(2, None, status='missing_runtime')])
        self.assertEqual(result['analyzed_batches'], 3)
        self.assertEqual(result[PREFIX + '_overlap_ms_per_batch'], 4.0)
        self.assertEqual(result[PREFIX + '_complete_batches'], 1)
        self.assertEqual(result[PREFIX + '_unknown_batches'], 2)
        self.assertEqual(result[PREFIX + '_no_pair_batches'], 1)
        self.assertEqual(result[PREFIX + '_coverage_fraction'], 1 / 3)

    def test_any_unknown_pair_invalidates_the_whole_batch(self):
        result = self.run_summary(
            [batch(1), batch(2)],
            [pair(1, 4.0), pair(1, None, status='missing_runtime'), pair(2, 2.0), pair(2, 3.0)])
        self.assertEqual(result[PREFIX + '_overlap_ms_per_batch'], 5.0)
        first, second = result[PREFIX + '_batch_results']
        self.assertEqual(first['status'], 'unknown')
        self.assertIsNone(first['overlap_ms'])
        self.assertEqual(second['status'], 'complete')
        self.assertEqual(second['overlap_ms'], 5.0)

    def test_assessable_measured_zero_remains_zero(self):
        result = self.run_summary([batch(1)], [pair(1, 0)])
        self.assertEqual(result[PREFIX + '_overlap_ms_per_batch'], 0)
        self.assertEqual(result[PREFIX + '_complete_batches'], 1)
        self.assertEqual(result[PREFIX + '_unknown_batches'], 0)
        self.assertEqual(result[PREFIX + '_coverage_fraction'], 1)

    def test_legacy_missing_target_type_defaults_to_projection_c1(self):
        pairs = [pair(1, 2), pair(1, 3), pair(2, 5)]
        for row in pairs:
            row.pop('target_type')
        result = self.run_summary([batch(1), batch(2)], pairs)
        self.assertEqual(result[PREFIX + '_overlap_ms_per_batch'], 5)
        self.assertEqual(result[PREFIX + '_complete_batches'], 2)
        self.assertEqual(result[PREFIX + '_coverage_fraction'], 1)
        self.assertTrue(all(row['source'] == 'legacy_complete_pair_sum'
                            for row in result[PREFIX + '_batch_results']))

    def test_unrelated_or_absent_targets_do_not_establish_zero(self):
        result = self.run_summary([batch(1), batch(2)], [pair(1, 7, target_type='image_c1')])
        self.assertIsNone(result[PREFIX + '_overlap_ms_per_batch'])
        self.assertEqual(result[PREFIX + '_declared_batches'], 0)
        self.assertEqual(result[PREFIX + '_no_pair_batches'], 2)
        self.assertEqual(result[PREFIX + '_unknown_batches'], 2)
        self.assertIn('lower bound', result['fine_tag_owned_kernel_overlap_interpretation'])

    def test_new_schema_uses_json_batch_union_without_double_counting_pairs(self):
        result = self.run_summary(
            [batch(1, target_pair_samples={'projection_c1': 2},
                   target_overlap_ms={'projection_c1': 4})],
            [pair(1, 2), pair(1, 3)])
        self.assertEqual(result[PREFIX + '_overlap_ms_per_batch'], 4)
        self.assertEqual(result[PREFIX + '_batch_results'][0]['source'], 'recorded_batch_union')

    def test_declared_null_or_unknown_pair_is_never_overridden_by_another_source(self):
        for aggregate, rows in (
            (None, [pair(1, 7)]),
            (7, [pair(1, None, status='missing_runtime')]),
        ):
            with self.subTest(aggregate=aggregate):
                result = self.run_summary(
                    [batch(1, target_pair_samples={'projection_c1': 1},
                           target_overlap_ms={'projection_c1': aggregate})], rows)
                self.assertIsNone(result[PREFIX + '_overlap_ms_per_batch'])
                self.assertEqual(result[PREFIX + '_unknown_batches'], 1)

    def test_declared_pair_count_must_match_and_metadata_alone_is_not_a_pair(self):
        for expected, rows in ((2, [pair(1, 4)]), (1, [])):
            with self.subTest(expected=expected):
                result = self.run_summary(
                    [batch(1, target_pair_samples={'projection_c1': expected},
                           target_overlap_ms={'projection_c1': 4})], rows)
                self.assertIsNone(result[PREFIX + '_overlap_ms_per_batch'])
                self.assertEqual(result[PREFIX + '_complete_batches'], 0)
                self.assertEqual(result[PREFIX + '_declared_batches'], 1)

    def test_assessable_label_with_missing_or_nonfinite_numeric_evidence_is_unknown(self):
        for value in (None, 'null', 'nan', 'inf', -1):
            with self.subTest(value=value):
                result = self.run_summary([batch(1)], [pair(1, value)])
                self.assertIsNone(result[PREFIX + '_overlap_ms_per_batch'])
                self.assertEqual(result[PREFIX + '_unknown_batches'], 1)


if __name__ == '__main__':
    unittest.main()
