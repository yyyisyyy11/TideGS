"""Small behavior tests for the opt-in paired benchmark runner (no GPU)."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


PATH = Path(__file__).parents[1] / "tools/run_pc1_paired_benchmark.py"
SPEC = importlib.util.spec_from_file_location("pc1_paired_benchmark", PATH)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class PC1PairedBenchmarkTest(unittest.TestCase):
    def test_complete_cycle_requires_nsight_step_and_uses_its_wall_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            output = root / "benchmark"
            manifest = benchmark.run_benchmark(config, output, execute=False,
                                               metadata_collector=lambda value: {})
            reports = {}
            for run in manifest["runs"]:
                run_dir = Path(run["run_dir"])
                run_dir.mkdir()
                with (run_dir / "metrics_camera_batches.tsv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["iteration", "camera_ids_json"], delimiter="\t")
                    writer.writeheader()
                    writer.writerows({"iteration": iteration, "camera_ids_json": "[0,1,2,3]"}
                                     for iteration in range(1, 6))
                rank_reports = []
                for rank in range(2):
                    metrics_path = run_dir / f"metrics_batch_rank{rank}.tsv"
                    with metrics_path.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=[
                            "iteration", "local_cameras", "batch_total_ms",
                        ], delimiter="\t")
                        writer.writeheader()
                        writer.writerows({"iteration": iteration, "local_cameras": 2,
                                          "batch_total_ms": 10} for iteration in range(1, 6))
                    base = 1000.0 if run["mode"] == "off" else 900.0
                    rank_reports.append({
                        "summary": {"rank": rank, "mode": run["mode"],
                                    "analysis_schema_version": 2, "sqlite_path": "rank.sqlite",
                                    "iteration_mapping": "chronological_ordinal_equal_total_counts",
                                    "run_sources": {"metrics_tsv": str(metrics_path)}},
                        "batches": [{"iteration": iteration, "step_wall_ms": base * (1 + rank * 0.1)}
                                    for iteration in range(1, 6)],
                    })
                report = run_dir / "analysis.json"
                report.write_text(json.dumps({"method": "actual CUPTI kernel intervals",
                                               "ranks": rank_reports}))
                reports[run["run_id"]] = str(report)
            without = benchmark.summarize(output)
            self.assertEqual(without["performance_status"], "unverified")
            self.assertFalse(without["paired_comparisons"])
            actual = benchmark.summarize(output, reports)
            self.assertEqual(actual["performance_status"], "complete_cycle_evidence")
            self.assertEqual(len(actual["paired_comparisons"]), 2)
            self.assertAlmostEqual(actual["mean_paired_speedup_ratio"], 1000 / 900)
            first = actual["runs"][0]
            self.assertEqual(first["metrics"]["critical_engine_elapsed_ms"]["mean"], 10.0)
            self.assertEqual(first["complete_batch_cycle"]["critical_batch_cycle_ms"]["mean"], 1100.0)

    def test_external_timeout_kills_the_process_group_and_stops_later_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            output = root / "benchmark"
            process = mock.Mock(pid=123, returncode=-9)
            process.wait.side_effect = [benchmark.subprocess.TimeoutExpired("training", 1), -9]
            with mock.patch.object(benchmark.subprocess, "Popen", return_value=process) as popen, \
                    mock.patch.object(benchmark.os, "killpg") as killpg:
                with self.assertRaisesRegex(RuntimeError, "round-00-off timeout"):
                    benchmark.run_benchmark(config, output, metadata_collector=lambda value: {})
            self.assertEqual(popen.call_count, 1)
            killpg.assert_called_once_with(123, benchmark.signal.SIGKILL)
            execution = json.loads((output / "round-00-off/execution.json").read_text())
            self.assertEqual(execution["status"], "timeout")

    def make_config(self, root):
        checkpoint, cameras = root / "checkpoint.pt", root / "cameras.json"
        checkpoint.write_bytes(b"fixed checkpoint")
        cameras.write_text("[1,2,3,4]")
        path = root / "config.json"
        path.write_text(json.dumps({
            "command": ["torchrun", "--nproc_per_node=2", "train.py",
                        "--checkpoint", "{checkpoint}", "--camera_schedule", "{camera_schedule}",
                        "--tide_forward_pipeline", "{mode}", "--log_folder", "{run_dir}"],
            "checkpoint": str(checkpoint), "camera_schedule": str(cameras),
            "working_directory": str(root), "rounds": 2,
            "warmup_batches": 1, "exclude_tail_batches": 1, "expected_world_size": 2,
        }))
        return benchmark.load_config(path)

    def test_alternating_order_changes_only_mode_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            plan = benchmark.build_plan(config, root / "new-result")
            self.assertEqual([(run["round"], run["mode"]) for run in plan],
                             [(0, "off"), (0, "pc1"), (1, "pc1"), (1, "off")])
            self.assertEqual(len({run["run_dir"] for run in plan}), 4)
            for run in plan:
                self.assertEqual(run["command"][4], config["checkpoint"])
                self.assertEqual(run["command"][6], config["camera_schedule"])
                self.assertEqual(run["command"][-3], run["mode"])
                self.assertEqual(run["command"][-1], run["run_dir"])
            with self.assertRaisesRegex(ValueError, "two rounds"):
                benchmark.paired_order(1)

    def test_existing_output_is_rejected_and_prepare_does_not_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            output = root / "benchmark"
            collected = []
            metadata = lambda value: collected.append(value) or {"gpu_identity": "fake"}
            manifest = benchmark.run_benchmark(config, output, execute=False,
                                               metadata_collector=metadata)
            self.assertEqual(manifest["status"], "prepared")
            self.assertEqual(len(collected), 1)
            self.assertFalse(any(output.glob("round-*")))
            original = (output / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                benchmark.run_benchmark(config, output, execute=False,
                                        metadata_collector=metadata)
            self.assertEqual((output / "manifest.json").read_bytes(), original)

    def test_statistics_trim_ranks_and_keep_actual_overlap_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            output = root / "benchmark"
            manifest = benchmark.run_benchmark(config, output, execute=False,
                                               metadata_collector=lambda value: {})
            run = manifest["runs"][0]
            run_dir = Path(run["run_dir"])
            run_dir.mkdir()
            for rank in range(2):
                path = run_dir / f"metrics_batch_rank{rank}.tsv"
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=[
                        "iteration", "local_cameras", "batch_total_ms", "projection_c1_overlap_ms",
                    ], delimiter="\t")
                    writer.writeheader()
                    for index in range(1, 6):
                        writer.writerow({"iteration": index, "local_cameras": 2,
                                         "batch_total_ms": 100 * index * (rank + 1),
                                         "projection_c1_overlap_ms": 99.0})
                events = [
                    {"iteration": 2, "lane": "cpu", "name": "Tide:S1:MB0:C1:FutureWait",
                     "start_ns": 0, "end_ns": 2_000_000},
                    {"iteration": 2, "lane": "cpu", "name": "Tide:S1:MB0:C2:FutureWait",
                     "start_ns": 1_000_000, "end_ns": 3_000_000},
                    {"iteration": 3, "lane": "cpu", "name": "C1:FutureWait",
                     "start_ns": 0, "end_ns": 1_000_000},
                    {"iteration": 3, "cuda_peak_allocated_bytes": 1200},
                    {"iteration": 1, "cuda_peak_allocated_bytes": 9000},
                ]
                (run_dir / f"timeline_events_rank{rank}.jsonl").write_text(
                    "\n".join(json.dumps(event) for event in events)
                )
            result = benchmark.aggregate_run(run_dir, config)
            self.assertEqual(result["critical_engine_elapsed_ms"]["mean"], 600.0)
            self.assertAlmostEqual(result["engine_global_throughput_cameras_per_second"], 12 / 1.8)
            rank0 = result["ranks"][0]
            self.assertEqual(rank0["iterations"], [2, 3, 4])
            self.assertEqual(rank0["engine_batch_elapsed_ms"],
                             {"samples": 3, "mean": 300.0, "median": 300.0, "p95": 390.0})
            self.assertEqual(rank0["peak_allocated_bytes"], 1200)
            self.assertAlmostEqual(rank0["future_wait_ms"]["mean"], 4.0 / 3.0)
            summary = benchmark.summarize(output)
            self.assertTrue(all(run["actual_overlap"]["status"] == "unverified"
                                for run in summary["runs"]))
            self.assertIsNone(summary["mean_paired_speedup_ratio"])
            self.assertEqual(summary["performance_status"], "unverified")

    def write_rank_metrics(self, run_dir):
        for rank in range(2):
            path = run_dir / f"metrics_batch_rank{rank}.tsv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "iteration", "local_cameras", "batch_total_ms",
                ], delimiter="\t")
                writer.writeheader()
                writer.writerows({"iteration": iteration, "local_cameras": 2,
                                  "batch_total_ms": 10} for iteration in range(1, 6))

    def test_overlap_report_requires_unique_ranks_and_matching_existing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            run_dir = root / "run"
            run_dir.mkdir()
            self.write_rank_metrics(run_dir)
            metrics = benchmark.aggregate_run(run_dir, config)
            run = {"run_dir": str(run_dir), "mode": "pc1", "expected_world_size": 2}
            path = root / "analysis.json"
            report = {"method": "actual CUPTI kernel intervals", "ranks": [{
                "summary": {
                    "rank": rank, "analysis_schema_version": 2, "mode": "pc1",
                    "sqlite_path": f"rank{rank}.sqlite",
                    "iteration_mapping": "chronological_ordinal_equal_total_counts",
                    "run_sources": {"metrics_tsv": str(run_dir / f"metrics_batch_rank{rank}.tsv")},
                },
                "batches": [{"iteration": iteration, "step_wall_ms": 100}
                            for iteration in range(1, 6)],
            } for rank in range(2)]}
            path.write_text(json.dumps(report))
            evidence = benchmark.read_actual_overlap(path, run, metrics)
            self.assertEqual(evidence["status"], "independent_kernel_analysis")
            self.assertEqual(benchmark.resolve_complete_cycles(evidence, metrics)["status"],
                             "evidence_available")
            self.assertEqual(benchmark.read_actual_overlap(path, run)["status"], "unverified")

            unrelated = run_dir / "unrelated.tsv"
            unrelated.write_text("existing but not aggregated")
            outside = root / "metrics_batch_rank0.tsv"
            outside.write_text("existing but outside run")
            for source in (run_dir / "missing.tsv", unrelated,
                           run_dir / "metrics_batch_rank1.tsv", outside):
                with self.subTest(source=source):
                    changed = json.loads(json.dumps(report))
                    changed["ranks"][0]["summary"]["run_sources"]["metrics_tsv"] = str(source)
                    path.write_text(json.dumps(changed))
                    evidence = benchmark.read_actual_overlap(path, run, metrics)
                    self.assertEqual(evidence["status"], "unverified")
                    self.assertEqual(benchmark.resolve_complete_cycles(evidence, metrics)["status"],
                                     "unverified")
            report["ranks"].append(report["ranks"][0])
            path.write_text(json.dumps(report))
            self.assertEqual(benchmark.read_actual_overlap(path, run, metrics)["status"], "unverified")
            report["ranks"].pop()
            path.write_text(json.dumps(report))
            (run_dir / "metrics_batch_rank0.tsv").unlink()
            self.assertEqual(benchmark.read_actual_overlap(path, run, metrics)["status"], "unverified")

    def test_duplicate_camera_iterations_are_rejected_and_remain_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            output = root / "benchmark"
            manifest = benchmark.run_benchmark(config, output, execute=False,
                                               metadata_collector=lambda value: {})
            run_dir = Path(manifest["runs"][0]["run_dir"])
            run_dir.mkdir()
            self.write_rank_metrics(run_dir)
            for duplicate_ids in ("[0,1]", "[2,3]"):
                with self.subTest(duplicate_ids=duplicate_ids):
                    with (run_dir / "metrics_camera_batches.tsv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=[
                            "iteration", "camera_ids_json",
                        ], delimiter="\t")
                        writer.writeheader()
                        writer.writerows({"iteration": iteration, "camera_ids_json": "[0,1]"}
                                         for iteration in range(1, 6))
                        writer.writerow({"iteration": 2, "camera_ids_json": duplicate_ids})
                    with self.assertRaisesRegex(ValueError, "duplicate camera iteration 2"):
                        benchmark.aggregate_run(run_dir, config)
                    summary = benchmark.summarize(output)
                    self.assertEqual(summary["runs"][0]["status"], "unverified")
                    self.assertEqual(summary["performance_status"], "unverified")
                    self.assertFalse(summary["paired_comparisons"])


if __name__ == "__main__":
    unittest.main()
