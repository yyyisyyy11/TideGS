#!/usr/bin/env python3
"""Reproducible alternating off/Micro Pipeline short runs and rank-metric summaries.

A JSON config supplies one argv command with {mode}, {run_dir}, {checkpoint}
and optional {camera_schedule}/{seed} placeholders. Every invocation creates a new output
root; no Slurm submission or shell evaluation is performed. See --help.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import string
import subprocess
import sys
import time


PRODUCER = "tide-micro_pipeline-paired-benchmark-v1"
LEGACY_PRODUCERS = {"tide-pc1-paired-benchmark-v1"}
RANK_PATTERN = re.compile(r"metrics_batch_rank(\d+)\.tsv$")


def normalize_mode(mode):
    """A PC1 manifest/report names the same mode as the current micro CLI."""
    return "micro" if mode == "pc1" else mode


def normalize_overlap_summary(summary):
    """Keep historical analyzer fields readable without emitting old names."""
    summary = dict(summary)
    summary["mode"] = normalize_mode(summary.get("mode"))
    for old, new in (("observed_pc1_pairs", "observed_micro_pipeline_pairs"),
                     ("assessable_pc1_pairs", "assessable_micro_pipeline_pairs")):
        if old in summary:
            summary.setdefault(new, summary.pop(old))
    return summary


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    if path.is_file():
        return {"path": str(path), "kind": "file_sha256", "sha256": file_sha256(path)}
    # A sharded checkpoint can be enormous. This explicitly records a metadata
    # fingerprint, not a false content hash of an unread checkpoint tree.
    entries = [(str(item.relative_to(path)), item.stat().st_size, item.stat().st_mtime_ns)
               for item in sorted(path.rglob("*")) if item.is_file()]
    if not entries:
        raise ValueError(f"input directory is empty: {path}")
    return {"path": str(path), "kind": "directory_metadata", "files": entries,
            "metadata_sha256": hashlib.sha256(json.dumps(entries).encode()).hexdigest()}


def load_config(path):
    path = Path(path).resolve(strict=True)
    config = json.loads(path.read_text(encoding="utf-8"))
    command = config.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command):
        raise ValueError("config.command must be a nonempty argv list")
    if any(Path(arg).name == "sbatch" for arg in command):
        raise ValueError("automatic sbatch submission is not supported")
    if Path(command[0]).name in {"sh", "bash", "zsh", "fish"}:
        raise ValueError("provide the executable argv directly; shell commands are not supported")
    fields = [field for arg in command for _, field, _, _ in string.Formatter().parse(arg)
              if field is not None]
    required = {"mode", "run_dir", "checkpoint"}
    if not required.issubset(fields):
        raise ValueError(f"command must reference placeholders {sorted(required)}")
    if set(fields) - required - {"seed", "camera_schedule"}:
        raise ValueError("command has unsupported placeholders")
    if fields.count("mode") != 1:
        raise ValueError("command must contain exactly one {mode} placeholder")
    config["rounds"] = int(config.get("rounds", 2))
    if config["rounds"] < 2:
        raise ValueError("at least two alternating paired rounds are required")
    config["timeout_seconds"] = float(config.get("timeout_seconds", 600))
    if not math.isfinite(config["timeout_seconds"]) or config["timeout_seconds"] <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    for name, default in (("warmup_batches", 2), ("exclude_tail_batches", 1)):
        config[name] = int(config.get(name, default))
        if config[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    config["expected_world_size"] = int(config.get("expected_world_size", 2))
    if config["expected_world_size"] < 1:
        raise ValueError("expected_world_size must be positive")
    for name in ("checkpoint", "camera_schedule", "working_directory"):
        raw = config.get(name, str(Path(__file__).resolve().parents[1])
                         if name == "working_directory" else None)
        if raw is None:
            raise ValueError(f"config.{name} is required")
        candidate = Path(raw).expanduser()
        config[name] = str((candidate if candidate.is_absolute() else path.parent / candidate).resolve(strict=True))
    if not Path(config["working_directory"]).is_dir():
        raise ValueError("working_directory must be a directory")
    environment = config.get("environment", {})
    if not isinstance(environment, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                     for k, v in environment.items()):
        raise ValueError("environment must map strings to strings")
    config["environment"] = environment
    config["seed"] = int(config.get("seed", 1))
    config["source_config"] = str(path)
    return config


def paired_order(rounds):
    if int(rounds) < 2:
        raise ValueError("at least two rounds are required")
    return [(round_index, mode) for round_index in range(int(rounds))
            for mode in (("off", "micro") if round_index % 2 == 0 else ("micro", "off"))]


def build_plan(config, output_dir):
    output_dir = Path(output_dir).resolve()
    plan = []
    for ordinal, (round_index, mode) in enumerate(paired_order(config["rounds"])):
        run_id = f"round-{round_index:02d}-{mode}"
        run_dir = output_dir / run_id
        fields = dict(mode=mode, run_dir=str(run_dir), checkpoint=config["checkpoint"],
                      camera_schedule=config["camera_schedule"], seed=config["seed"],
                      round=round_index, run_id=run_id)
        plan.append({"ordinal": ordinal, "round": round_index, "mode": mode,
                     "run_id": run_id, "run_dir": str(run_dir),
                     "expected_world_size": config["expected_world_size"],
                     "command": [arg.format_map(fields) for arg in config["command"]]})
    return plan


def _probe(command, cwd):
    try:
        result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, timeout=15, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "unavailable": str(error)}


def runtime_metadata(config):
    cwd = config["working_directory"]
    selected_environment = {key: value for key, value in os.environ.items()
                            if key.startswith(("NCCL_", "TORCH_NCCL_", "CUDA_"))
                            or key in {"OMP_NUM_THREADS", "PYTHONHASHSEED"}}
    selected_environment.update(config["environment"])
    python = config.get("python_executable", sys.executable)
    versions = _probe([python, "-c", (
        "import json,platform,importlib.util; "
        "t=__import__('torch') if importlib.util.find_spec('torch') else None; "
        "print(json.dumps({'python':platform.python_version(),"
        "'torch':getattr(t,'__version__',None),'cuda':getattr(getattr(t,'version',None),'cuda',None)}))"
    )], cwd)
    return {"host": platform.node(), "platform": platform.platform(),
            "environment": selected_environment, "versions": versions,
            "git_commit": _probe(["git", "rev-parse", "HEAD"], cwd),
            "git_status": _probe(["git", "status", "--short"], cwd),
            "gpu_identity": _probe([
                "nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total",
                "--format=csv,noheader",
            ], cwd)}


def statistics_summary(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"samples": 0, "mean": None, "median": None, "p95": None}
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("metrics contain a nonfinite value")
    position = 0.95 * (len(ordered) - 1)
    left, right = math.floor(position), math.ceil(position)
    p95 = ordered[left] + (ordered[right] - ordered[left]) * (position - left)
    return {"samples": len(ordered), "mean": statistics.mean(ordered),
            "median": statistics.median(ordered), "p95": p95}


def _number(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {field}")
    return value


def _timeline_metrics(path, selected_iterations):
    if not path.exists():
        return {"peak_allocated_bytes": None, "future_wait_ms": None,
                "availability": "missing timeline JSONL"}
    peaks = []
    waits = {iteration: [] for iteration in selected_iterations}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            iteration = int(event["iteration"]) if event.get("iteration") is not None else None
            if iteration not in selected_iterations:
                continue
            if event.get("cuda_peak_allocated_bytes") is not None:
                peaks.append(int(event["cuda_peak_allocated_bytes"]))
            if event.get("lane") == "cpu" and "FutureWait" in str(event.get("name", "")):
                waits[iteration].append((int(event["start_ns"]), int(event["end_ns"])))
    totals = []
    for intervals in waits.values():
        merged = []
        for start, end in sorted(intervals):
            if end < start:
                raise ValueError("CPU Future wait ends before it starts")
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        totals.append(sum(end - start for start, end in merged) / 1e6)
    return {"peak_allocated_bytes": max(peaks) if peaks else None,
            "future_wait_ms": statistics_summary(totals),
            "availability": "timeline JSONL; Future waits use interval union"}


def aggregate_run(run_dir, config):
    run_dir = Path(run_dir)
    ranked = {}
    for path in sorted(run_dir.rglob("metrics_batch_rank*.tsv")):
        match = RANK_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        rank = int(match.group(1))
        if rank in ranked:
            raise ValueError(f"multiple metric files for rank {rank}")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        first, last = config["warmup_batches"], config["exclude_tail_batches"]
        if len(rows) <= first + last:
            raise ValueError(f"rank {rank} has no batches after warmup/tail exclusions")
        selected = rows[first:len(rows) - last if last else None]
        iterations = [int(row["iteration"]) for row in selected]
        if iterations != sorted(set(iterations)):
            raise ValueError(f"rank {rank} iterations must be strictly increasing")
        cycles = [_number(row, "batch_total_ms") for row in selected]
        if any(value <= 0 for value in cycles):
            raise ValueError("batch_total_ms must be positive")
        cameras = [_number(row, "local_cameras") for row in selected]
        timelines = list(run_dir.rglob(f"timeline_events_rank{rank}.jsonl"))
        if len(timelines) > 1:
            raise ValueError(f"multiple timeline files for rank {rank}")
        timeline = _timeline_metrics(
            timelines[0] if timelines else run_dir / f"timeline_events_rank{rank}.jsonl",
            set(iterations),
        )
        ranked[rank] = {
            "rank": rank, "metrics_tsv": str(path.resolve()), "iterations": iterations,
            "engine_batch_elapsed_ms": statistics_summary(cycles),
            "engine_throughput_cameras_per_second": 1000.0 * sum(cameras) / sum(cycles),
            "cameras_by_iteration": dict(zip(map(str, iterations), cameras)),
            "prefetch_cpu_wait_ms": statistics_summary([
                float(row.get("prefetch_inflight_wait_ms") or 0.0)
                + float(row.get("prefetch_buffer_wait_ms") or 0.0) for row in selected
            ]),
            **timeline, "_cycles": cycles, "_cameras": cameras,
        }
    if set(ranked) != set(range(config["expected_world_size"])):
        raise ValueError("metrics do not cover every expected rank exactly once")
    iterations = ranked[0]["iterations"]
    if any(rank["iterations"] != iterations for rank in ranked.values()):
        raise ValueError("ranks have different measured iterations")
    critical = [max(rank["_cycles"][index] for rank in ranked.values())
                for index in range(len(iterations))]
    camera_total = sum(sum(rank["_cameras"]) for rank in ranked.values())
    for rank in ranked.values():
        rank.pop("_cycles")
        rank.pop("_cameras")
    camera_paths = list(run_dir.rglob("metrics_camera_batches.tsv"))
    camera_order = {"status": "unverified", "reason": "unique camera batch TSV missing"}
    if len(camera_paths) == 1:
        with camera_paths[0].open(encoding="utf-8", newline="") as handle:
            camera_rows = list(csv.DictReader(handle, delimiter="\t"))
        by_iteration = {}
        for row in camera_rows:
            iteration = int(row["iteration"])
            if iteration in by_iteration:
                raise ValueError(f"duplicate camera iteration {iteration}")
            by_iteration[iteration] = json.loads(row["camera_ids_json"])
        if all(iteration in by_iteration for iteration in iterations):
            payload = [(iteration, by_iteration[iteration]) for iteration in iterations]
            camera_order = {"status": "recorded", "source": str(camera_paths[0].resolve()),
                            "sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest()}
    return {"diagnostic_source": "metrics_batch_rank*.tsv:batch_total_ms",
            "scope": "distributed_engine function only; excludes outer camera scheduling/loop cleanup",
            "camera_order": camera_order,
            "excluded_warmup_batches": config["warmup_batches"],
            "excluded_tail_batches": config["exclude_tail_batches"],
            "ranks": list(ranked.values()), "critical_engine_elapsed_ms": statistics_summary(critical),
            "engine_global_throughput_cameras_per_second": 1000.0 * camera_total / sum(critical)}


def read_actual_overlap(path, run, metrics=None):
    if path is None:
        return {"status": "unverified", "reason": "no independent Nsight analyzer JSON supplied"}
    path = Path(path).resolve()
    if not path.is_file():
        return {"status": "unverified", "reason": "analyzer JSON missing", "path": str(path)}
    report = json.loads(path.read_text(encoding="utf-8"))
    ranks = [normalize_overlap_summary(item.get("summary", {}))
             for item in report.get("ranks", [])]
    try:
        measured_sources = {item["rank"]: Path(item["metrics_tsv"]).resolve(strict=True)
                            for item in metrics["ranks"]}
        rank_ids = [item.get("rank") for item in ranks]
        valid = (bool(ranks) and len(rank_ids) == len(set(rank_ids))
                 and set(rank_ids) == set(measured_sources) and all(
                     int(item.get("analysis_schema_version", 0)) >= 2
                     and item.get("mode") == normalize_mode(run["mode"]) and item.get("sqlite_path")
                     for item in ranks
                 ))
        expected_world_size = run.get("expected_world_size")
        if expected_world_size is not None:
            valid = valid and set(rank_ids) == set(range(expected_world_size))
        # Bind each rank to the exact existing file aggregated above. Merely
        # residing in run_dir cannot establish which metrics the capture used.
        for rank in ranks:
            source = rank.get("run_sources", {}).get("metrics_tsv")
            if not source:
                valid = False
                break
            source = Path(source).resolve(strict=True)
            valid = (valid and source.is_file()
                     and source == measured_sources.get(rank.get("rank"))
                     and source.is_relative_to(Path(run["run_dir"]).resolve()))
    except (OSError, KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        return {"status": "unverified", "reason": "report schema/mode/run source mismatch",
                "path": str(path)}
    return {"status": "independent_kernel_analysis", "path": str(path),
            "sha256": file_sha256(path), "method": report.get("method"), "ranks": ranks,
            "limitations": report.get("limitations", [])}


def resolve_complete_cycles(evidence, metrics):
    unavailable = {"status": "unverified", "reason": "independent complete tide.batch.step evidence missing"}
    if evidence["status"] != "independent_kernel_analysis" or metrics is None:
        return unavailable
    try:
        report = json.loads(Path(evidence["path"]).read_text(encoding="utf-8"))
        by_rank = {int(item["summary"]["rank"]): item for item in report["ranks"]}
        cycles, camera_total = [], 0.0
        iterations = metrics["ranks"][0]["iterations"]
        rank_summaries = []
        for measured in metrics["ranks"]:
            captured = by_rank[measured["rank"]]
            if captured["summary"].get("iteration_mapping") != "chronological_ordinal_equal_total_counts":
                raise ValueError("Nsight batches are not aligned with metric iterations")
            wall_times = {}
            for batch in captured["batches"]:
                iteration = int(batch["iteration"])
                if iteration in wall_times:
                    raise ValueError("duplicate Nsight batch iteration")
                value = float(batch["step_wall_ms"])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("Nsight step_wall_ms must be positive and finite")
                wall_times[iteration] = value
            if not set(iterations).issubset(wall_times):
                raise ValueError("Nsight does not cover every selected metric iteration")
            measured_cycles = [wall_times[iteration] for iteration in iterations]
            cameras = sum(measured["cameras_by_iteration"][str(iteration)] for iteration in iterations)
            camera_total += cameras
            cycles.append(measured_cycles)
            rank_summaries.append({
                "rank": measured["rank"], "batch_cycle_ms": statistics_summary(measured_cycles),
                "throughput_cameras_per_second": 1000.0 * cameras / sum(measured_cycles),
            })
        critical = [max(values[index] for values in cycles) for index in range(len(iterations))]
        return {"status": "evidence_available", "source": evidence["path"],
                "source_sha256": evidence["sha256"], "scope": "Nsight tide.batch.step step_wall_ms",
                "iterations": iterations, "ranks": rank_summaries,
                "critical_batch_cycle_ms": statistics_summary(critical),
                "global_throughput_cameras_per_second": 1000.0 * camera_total / sum(critical)}
    except (KeyError, TypeError, ValueError) as error:
        return {"status": "unverified", "reason": str(error)}


def summarize(output_dir, overlap_reports=None):
    output_dir = Path(output_dir).resolve()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("producer") not in {PRODUCER, *LEGACY_PRODUCERS}:
        raise ValueError("output directory was not created by this paired benchmark tool")
    config, runs = manifest["config"], []
    overlap_reports = overlap_reports or {}
    unknown = set(overlap_reports) - {run["run_id"] for run in manifest["runs"]}
    if unknown:
        raise ValueError(f"unknown run ids for overlap reports: {sorted(unknown)}")
    for run in manifest["runs"]:
        run = dict(run, mode=normalize_mode(run["mode"]))
        record = {key: run[key] for key in ("run_id", "round", "mode", "ordinal")}
        try:
            record["metrics"] = aggregate_run(run["run_dir"], config)
            record["status"] = "metrics_available"
        except (OSError, KeyError, ValueError) as error:
            record.update(status="unverified", reason=str(error))
        record["actual_overlap"] = read_actual_overlap(
            overlap_reports.get(run["run_id"]), run, record.get("metrics"),
        )
        record["complete_batch_cycle"] = resolve_complete_cycles(
            record["actual_overlap"], record.get("metrics"),
        )
        runs.append(record)
    pairs = []
    for round_index in range(config["rounds"]):
        modes = {run["mode"]: run for run in runs if run["round"] == round_index}
        cycles = {mode: modes[mode]["complete_batch_cycle"] for mode in ("off", "micro")}
        if (all(cycles[mode]["status"] == "evidence_available" for mode in cycles)
                and cycles["off"]["iterations"] == cycles["micro"]["iterations"]
                and all(modes[mode]["metrics"]["camera_order"]["status"] == "recorded" for mode in modes)
                and modes["off"]["metrics"]["camera_order"]["sha256"]
                == modes["micro"]["metrics"]["camera_order"]["sha256"]):
            off = cycles["off"]["critical_batch_cycle_ms"]["mean"]
            micro = cycles["micro"]["critical_batch_cycle_ms"]["mean"]
            pairs.append({"round": round_index, "status": "complete_cycle_evidence",
                          "off_mean_cycle_ms": off,
                          "micro_pipeline_mean_cycle_ms": micro, "micro_pipeline_speedup_ratio": off / micro,
                          "micro_pipeline_cycle_change_percent": (micro / off - 1.0) * 100.0})
    result = {"producer": PRODUCER, "runs": runs, "paired_comparisons": pairs,
              "mean_paired_speedup_ratio": statistics.mean(pair["micro_pipeline_speedup_ratio"] for pair in pairs)
              if pairs else None,
              "performance_status": "complete_cycle_evidence" if len(pairs) == config["rounds"] else "unverified",
              "interpretation": "batch_total_ms is an engine-only diagnostic. Performance ratios require "
              "independent Nsight tide.batch.step step_wall_ms for matching measured iterations on every rank. "
              "Actual overlap is established only by the separately referenced kernel analyzer. "
              "Missing reports or metrics remain unverified; event estimates are never substituted."}
    _write_json(output_dir / "summary.json", result)
    return result


def run_benchmark(config, output_dir, *, execute=True, metadata_collector=runtime_metadata):
    output_dir = Path(output_dir).resolve()
    for name in ("checkpoint", "camera_schedule"):
        source = Path(config[name]).resolve()
        if source.is_dir() and output_dir.is_relative_to(source):
            raise ValueError("output directory must be outside immutable benchmark inputs")
    # Always fail on an existing directory, including a previous prepared plan.
    output_dir.mkdir(parents=True, exist_ok=False)
    plan = build_plan(config, output_dir)
    inputs = {name: fingerprint(config[name]) for name in
              ("checkpoint", "camera_schedule", "source_config")}
    metadata = metadata_collector(config)
    manifest = {"producer": PRODUCER, "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": config, "inputs": inputs, "runtime": metadata, "runs": plan,
                "status": "prepared"}
    _write_json(output_dir / "manifest.json", manifest)
    if not execute:
        return manifest
    environment = dict(os.environ, **config["environment"])
    for run in plan:
        for name, initial in inputs.items():
            if fingerprint(config[name]) != initial:
                raise RuntimeError(f"shared benchmark input changed: {name}")
        current = metadata_collector(config)
        for field in ("gpu_identity", "host", "versions", "git_commit", "environment"):
            if current.get(field) != metadata.get(field):
                raise RuntimeError(f"benchmark runtime changed between paired runs: {field}")
        run_dir = Path(run["run_dir"])
        run_dir.mkdir(exist_ok=False)
        started = time.monotonic()
        evidence = {"command": run["command"], "runtime": current, "status": "running"}
        _write_json(run_dir / "execution.json", evidence)
        with (run_dir / "console.log").open("w", encoding="utf-8") as console:
            try:
                process = subprocess.Popen(run["command"], cwd=config["working_directory"],
                                           env=environment, stdout=console, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            except OSError as error:
                evidence.update(status="failed_to_start", error=str(error))
            else:
                try:
                    returncode = process.wait(timeout=config["timeout_seconds"])
                    evidence.update(status="complete" if returncode == 0 else "failed", returncode=returncode)
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=10)
                    evidence.update(status="timeout" if isinstance(error, subprocess.TimeoutExpired)
                                    else "interrupted", returncode=process.returncode)
            evidence["elapsed_seconds"] = time.monotonic() - started
            _write_json(run_dir / "execution.json", evidence)
        if evidence["status"] != "complete":
            manifest["status"] = evidence["status"]
            _write_json(output_dir / "manifest.json", manifest)
            summarize(output_dir)
            raise RuntimeError(f"{run['run_id']} {evidence['status']}; see {run_dir / 'console.log'}")
    # Detect a training command that accidentally writes into its source inputs.
    for name, initial in inputs.items():
        if fingerprint(config[name]) != initial:
            raise RuntimeError(f"shared benchmark input changed: {name}")
    manifest["status"] = "complete"
    _write_json(output_dir / "manifest.json", manifest)
    summarize(output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run", help="run one common config in alternating paired order")
    run.add_argument("config", type=Path)
    run.add_argument("--output-dir", type=Path, required=True, help="new directory; existing paths are rejected")
    run.add_argument("--prepare-only", action="store_true", help="write the plan/provenance without launching training")
    summary = commands.add_parser("summarize", help="resolve metrics and attach independent analyzer reports")
    summary.add_argument("output_dir", type=Path)
    summary.add_argument("--overlap-report", action="append", default=[], metavar="RUN_ID=JSON")
    args = parser.parse_args()
    try:
        if args.action == "run":
            result = run_benchmark(load_config(args.config), args.output_dir, execute=not args.prepare_only)
            print(json.dumps({"output_dir": str(args.output_dir.resolve()), "status": result["status"]}))
        else:
            reports = {}
            for value in args.overlap_report:
                run_id, separator, path = value.partition("=")
                if not separator or run_id in reports:
                    raise ValueError("overlap reports require unique RUN_ID=JSON entries")
                reports[run_id] = path
            result = summarize(args.output_dir, reports)
            print(json.dumps({"summary": str(args.output_dir.resolve() / "summary.json"),
                              "paired_rounds": len(result["paired_comparisons"])}))
        return 0
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"Micro Pipeline paired benchmark: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
