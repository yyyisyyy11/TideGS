"""Actual-kernel attribution across main, comm and autograd threads (no CUDA)."""

import importlib.util
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

SHARED_SCRIPT_DIR = Path(__file__).resolve().parents[3] / 'experiment/timeline/scripts'
BUNDLED_SCRIPT_DIR = Path(__file__).resolve().parents[1] / 'tools/micro_pipeline_timeline'
SCRIPT_DIR = Path(os.environ.get('TIDE_TIMELINE_SCRIPTS', str(
    SHARED_SCRIPT_DIR if SHARED_SCRIPT_DIR.is_dir() else BUNDLED_SCRIPT_DIR)))
SCRIPT = SCRIPT_DIR / 'analyze_micro_pipeline_overlap.py'
if not SCRIPT.is_file():
    raise unittest.SkipTest('shared timeline scripts unavailable; set TIDE_TIMELINE_SCRIPTS')
spec = importlib.util.spec_from_file_location('micro_pipeline_overlap_analysis', SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class MicroPipelineOverlapAnalysisTest(unittest.TestCase):
    def fixture(self):
        conn = sqlite3.connect(':memory:')
        self.addCleanup(conn.close)
        conn.executescript('''
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            CREATE TABLE PROCESSES(globalPid INTEGER, pid INTEGER);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,deviceId INTEGER,streamId INTEGER,shortName INTEGER);
        ''')
        pid = 123
        process = pid << 24
        main, comm, autograd = (process | tid for tid in (pid, 124, 125))
        conn.execute('INSERT INTO PROCESSES VALUES(?,?)', (process, pid))
        conn.executemany('INSERT INTO StringIds VALUES(?,?)', [(1, 'compute'), (2, 'ncclKernel_SendRecv'), (3, 'cudaLaunchKernel')])

        def span(start, end, tid, name):
            conn.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,NULL)', (start, end, tid, name))

        span(0, 1000, main, 'tide.batch.step')
        span(0, 950, main, 'tide.batch.compute')
        for start, end, tid, mb, phase in (
            (1, 10, main, 0, 'Projection'), (11, 20, main, 1, 'Projection'),
            (21, 30, comm, 0, 'C1:Metadata'), (31, 80, comm, 0, 'C1:Payload'),
            (81, 90, comm, 1, 'C1:Metadata'), (91, 130, comm, 1, 'C1:Payload'),
            (100, 110, main, 0, 'Render'), (111, 120, main, 0, 'Loss'),
            (121, 130, main, 0, 'IntermediateGrad'),
            (131, 140, main, 1, 'Render'), (141, 150, main, 1, 'Loss'),
            (151, 160, main, 1, 'IntermediateGrad'),
            (131, 170, comm, 0, 'C2:Payload'), (171, 200, comm, 1, 'C2:Payload'),
            (170, 190, main, 0, 'FinalGrad'), (175, 185, autograd, 0, 'FinalGrad:ProjectionBackward'),
            (191, 210, main, 1, 'FinalGrad'),
        ):
            span(start, end, tid, f'Tide:S1:MB{mb}:{phase}|rank=0,batch=1,slot={mb},schedule=micro-pipeline-window-v2')
        # A nested node on its own autograd thread must own this launch once.
        span(178, 184, autograd, 'Tide:S1:MB0:FinalGrad:SHBackward')
        rows = (
            (15, main, 50, 90, 1, 1), (25, comm, 30, 40, 2, 2),
            (35, comm, 40, 80, 2, 2), (85, comm, 100, 110, 2, 2),
            (100, comm, 110, 155, 2, 2), (105, main, 120, 145, 1, 1),
            (115, main, 145, 150, 1, 1), (125, main, 150, 155, 1, 1),
            (155, main, 165, 180, 1, 1), (140, comm, 160, 190, 2, 2),
            (180, comm, 195, 220, 2, 2), (180, autograd, 200, 230, 1, 1),
        )
        for correlation, (launch, tid, start, end, stream, name) in enumerate(rows, 1):
            conn.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,3)', (launch, launch + 1, tid, correlation))
            conn.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,0,?,?)', (start, end, process, correlation, stream, name))
        return conn, process, main, comm, autograd

    def test_four_targets_and_intermediate_subset_use_all_threads(self):
        conn, process, main, comm, autograd = self.fixture()
        result, audit = analysis.analyze_connection(conn, rank=0, pid=123, trim=0)
        pairs = {row['target_type']: row for row in result['pairs']}
        self.assertEqual(set(pairs), {'projection_c1', 'image_c1', 'image_intermediate_c1', 'intermediate_c2', 'final_c2'})
        expected = {'projection_c1': 30, 'image_c1': 35, 'image_intermediate_c1': 5, 'intermediate_c2': 15, 'final_c2': 20}
        for name, overlap in expected.items():
            self.assertEqual(pairs[name]['status'], 'assessable')
            self.assertAlmostEqual(pairs[name]['cross_stream_overlap_ms'], overlap / 1e6)
            self.assertEqual(result['summary']['target_summary'][name]['samples'], 1)
        self.assertEqual({row['runtime_global_tid'] for row in audit}, {main, comm, autograd})
        self.assertEqual(len(audit), len({row['kernel_rowid'] for row in audit}))
        final = next(row for row in audit if row['correlation_id'] == 12)
        self.assertEqual(final['nvtx_start_ns'], 178)

    def test_unknown_is_retained_without_nearest_time_attribution(self):
        conn, process, *_ = self.fixture()
        conn.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(300,320,?,999,0,1,1)', (process,))
        result, audit = analysis.analyze_connection(conn, rank=0, pid=123, trim=0)
        unknown = next(row for row in audit if row['correlation_id'] == 999)
        self.assertEqual(unknown['phase'], 'unknown')
        self.assertIsNone(unknown['nvtx_rowid'])
        self.assertEqual(result['summary']['correlation_diagnostics']['unknown_phase_kernel_rows'], 1)

    def test_same_stream_and_other_process_never_prove_overlap(self):
        conn, process, *_ = self.fixture()
        conn.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET streamId=1')
        conn.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(0,1000,?,1,0,2,2)', (456 << 24,))
        result, audit = analysis.analyze_connection(conn, rank=0, pid=123, trim=0)
        self.assertTrue(all(row['cross_stream_overlap_ms'] == 0 for row in result['pairs']))
        self.assertTrue(all(row['kernel_global_pid'] == process for row in audit))

    def test_missing_runtime_is_unassessable_and_csv_accepts_both_schemas(self):
        conn, *_ = self.fixture()
        conn.execute('DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=9')
        result, _ = analysis.analyze_connection(conn, rank=0, pid=123, trim=0)
        row = next(row for row in result['pairs'] if row['target_type'] == 'intermediate_c2')
        self.assertIsNone(row['cross_stream_overlap_ms'])
        self.assertIsNone(result['batches'][0]['target_overlap_ms']['intermediate_c2'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pairs.csv'
            analysis.write_csv(path, result['pairs'])
            self.assertIn('target_type', path.read_text())

    def test_missing_required_phase_never_reports_a_partial_target_as_complete(self):
        for microbatch, phase, targets in (
            (0, 'Render', ('image_c1',)),
            (0, 'Loss', ('image_c1',)),
            (0, 'IntermediateGrad', ('image_c1', 'image_intermediate_c1')),
            (1, 'C1:Metadata', ('image_c1', 'image_intermediate_c1')),
            (1, 'C1:Payload', ('image_c1', 'image_intermediate_c1')),
            (1, 'IntermediateGrad', ('intermediate_c2',)),
            (0, 'C2:Payload', ('intermediate_c2',)),
            (0, 'FinalGrad', ('final_c2',)),
            (1, 'C2:Payload', ('final_c2',)),
        ):
            with self.subTest(microbatch=microbatch, phase=phase):
                conn, *_ = self.fixture()
                prefix = f'Tide:S1:MB{microbatch}:{phase}'
                conn.execute('DELETE FROM NVTX_EVENTS WHERE text LIKE ? OR text LIKE ?',
                             (prefix + '|%', prefix + ':%'))
                result, _ = analysis.analyze_connection(conn, rank=0, pid=123, trim=0)
                rows = {row['target_type']: row for row in result['pairs']}
                for target in targets:
                    self.assertIn('missing_', rows[target]['status'])
                    self.assertIsNone(rows[target]['cross_stream_overlap_ms'])
                    self.assertIsNone(result['batches'][0]['target_overlap_ms'][target])

    def test_present_kernel_free_loss_scope_is_valid_without_borrowing_unknown_work(self):
        conn, process, *_ = self.fixture()
        # This scope genuinely launches no kernels. Other image phases still
        # have correlated compute; an uncorrelated kernel at the same GPU time
        # must remain unknown and must not replace its missing activity.
        conn.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=7')
        conn.execute('DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=7')
        conn.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(145,150,?,999,0,1,1)', (process,))
        result, audit = analysis.analyze_connection(conn, rank=0, pid=123, trim=0)
        row = next(row for row in result['pairs'] if row['target_type'] == 'image_c1')
        self.assertEqual(row['status'], 'assessable')
        self.assertAlmostEqual(row['cross_stream_overlap_ms'], 30 / 1e6)
        unknown = next(row for row in audit if row['correlation_id'] == 999)
        self.assertEqual(unknown['phase'], 'unknown')

    def test_launch_without_captured_kernel_makes_image_target_unassessable(self):
        conn, *_ = self.fixture()
        conn.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=7')
        result, _ = analysis.analyze_connection(conn, rank=0, pid=123, trim=0)
        row = next(row for row in result['pairs'] if row['target_type'] == 'image_c1')
        self.assertIn('unmatched_phase_kernel_launch_api', row['status'])
        self.assertIsNone(row['cross_stream_overlap_ms'])

    def test_legacy_c2_node_scope_remains_a_valid_communication_boundary(self):
        conn, *_ = self.fixture()
        conn.execute("UPDATE NVTX_EVENTS SET text=replace(text,'C2:Payload','C2:AlltoAllBackward')")
        result, _ = analysis.analyze_connection(conn, rank=0, pid=123, trim=0)
        rows = {row['target_type']: row for row in result['pairs']}
        for target, expected in (('intermediate_c2', 15), ('final_c2', 20)):
            self.assertEqual(rows[target]['status'], 'assessable')
            self.assertAlmostEqual(rows[target]['cross_stream_overlap_ms'], expected / 1e6)

    def test_legacy_reports_still_pass_builtin_self_test(self):
        analysis.self_test()

    def test_crossing_same_thread_scopes_fail(self):
        with self.assertRaisesRegex(ValueError, 'Crossing'):
            analysis.ThreadRangeIndex([
                {'rowid': 1, 'globalTid': 1, 'start': 0, 'end': 10},
                {'rowid': 2, 'globalTid': 1, 'start': 5, 'end': 15},
            ])

    def test_slot_wrap_starts_a_new_window_without_cross_window_targets(self):
        conn, *_ = self.fixture()
        conn.execute("UPDATE NVTX_EVENTS SET text=replace(text,'slot=1','slot=0') WHERE text LIKE 'Tide:S1:MB1:Projection%'")
        result, _ = analysis.analyze_connection(conn, rank=0, pid=123, trim=0)
        self.assertEqual(result['pairs'], [])

    def test_legacy_mode_and_schedule_emit_current_names(self):
        conn, *_ = self.fixture()
        conn.execute("UPDATE NVTX_EVENTS SET text=replace(text, 'micro-pipeline-window-v2', 'pc1-window-v2')")
        result, _ = analysis.analyze_connection(conn, rank=0, pid=123, trim=0, mode='pc1')
        summary = result['summary']
        self.assertEqual(summary['mode'], 'micro')
        self.assertGreater(summary['observed_micro_pipeline_pairs'], 0)
        self.assertIn('assessable_micro_pipeline_pairs', summary)
        self.assertNotIn('observed_pc1_pairs', summary)
        self.assertEqual(analysis.argument_parser().parse_args(['--mode', 'micro']).mode, 'micro')
        self.assertEqual(analysis.argument_parser().parse_args(['--mode', 'pc1']).mode, 'pc1')

    def test_submission_report_accepts_worker_payload_and_unknown_kernel(self):
        conn, process, *_ = self.fixture()
        conn.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(300,320,?,999,0,1,1)', (process,))
        result, audit = analysis.analyze_connection(conn, rank=0, pid=123, trim=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('pairs', 'batches', 'stages'):
                analysis.write_csv(root / f'micro-pipeline-overlap.{name}.csv', result[name])
            analysis.write_csv(root / 'micro-pipeline-overlap.kernels.csv', audit, analysis.KERNEL_AUDIT_FIELDS)
            run = subprocess.run([
                sys.executable, str(SCRIPT.with_name('analyze_micro_pipeline_submission_order.py')),
                '--derived-dir', str(root), '--output', str(root / 'submission'),
            ], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue((root / 'submission.pairs.csv').is_file())

    def test_cpu_parent_subtraction_is_per_thread(self):
        spec = importlib.util.spec_from_file_location('micro_pipeline_extract_comparison', SCRIPT.with_name('extract_superpod_comparison.py'))
        extractor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extractor)
        spans = [
            {'start_ns': 0, 'end_ns': 100, 'global_tid': 1, 'source_rowid': 1, 'name': 'tide.batch.compute'},
            {'start_ns': 20, 'end_ns': 80, 'global_tid': 2, 'source_rowid': 2, 'name': 'Tide:S1:MB0:C1:Payload|tid=2'},
        ]
        fragments = extractor.exclusive_cpu_phases(spans)
        self.assertEqual([(row['global_tid'], row['start_ns'], row['end_ns']) for row in fragments],
                         [(1, 0, 100), (2, 20, 80)])


if __name__ == '__main__':
    unittest.main()
