#!/usr/bin/env python3
"""Measure actual CUDA-kernel overlap in off/PC1 Nsight Systems exports.

Uses only the Python standard library. CPU NVTX ranges assign ownership through
runtime API correlation IDs; they are never used as GPU execution intervals.
For PC1, report the four forward/backward targets (and the IntermediateGrad
image-side subset) using all process threads, the same device and different
CUDA streams. Nested ranges own each kernel once; uncorrelated work is unknown.

Example:
  python analyze_pc1_overlap.py --run-dir /path/to/run \
    --sqlite /path/to/rank0.sqlite /path/to/rank1.sqlite \
      /path/to/rank2.sqlite /path/to/rank3.sqlite \
    --output /path/to/verify/pc1-overlap --emit-kernels

SQLite arguments are ordered by rank, starting at rank 0. Legacy
--rank0-sqlite/--rank1-sqlite and --pid-rank0/1 remain available separately.
This script excludes the first and last captured complete batch by default.
CUDA-event spans in JSONL/TSV are diagnostic only and are not used for overlap.
"""

import argparse
import bisect
import csv
import json
import statistics
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path


PC1_PREFIX = "Tide:S1:MB"
PHASES = {"Projection", "C1:Metadata", "C1:Payload", "C1:Wait", "Render", "Loss", "Backward",
          "IntermediateGrad", "C2", "C2:Wait", "C2:Payload", "FinalGrad", "AccumulateGrad"}
KERNEL_AUDIT_FIELDS = (
    "rank", "trace_batch", "optimizer_step", "iteration", "microbatch",
    "phase", "kernel_rowid", "kernel_name", "category", "device_id",
    "stream_id", "correlation_id", "runtime_global_tid", "kernel_global_pid",
    "runtime_api", "runtime_start_ns", "runtime_end_ns", "nvtx_rowid",
    "nvtx_start_ns", "nvtx_end_ns", "kernel_start_ns", "kernel_end_ns",
    "graph_id", "graph_node_id",
)


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def intersection(first, second):
    first, second = union(first), union(second)
    result, left, right = [], 0, 0
    while left < len(first) and right < len(second):
        start = max(first[left][0], second[right][0])
        end = min(first[left][1], second[right][1])
        if end > start:
            result.append((start, end))
        if first[left][1] <= second[right][1]:
            left += 1
        else:
            right += 1
    return result


def duration_ms(intervals):
    return sum(end - start for start, end in union(intervals)) / 1e6


def intervals(records):
    return [(record["start"], record["end"]) for record in records]


def clipped(records, start, end):
    return [(max(start, item["start"]), min(end, item["end"]))
            for item in records if item["start"] < end and item["end"] > start]


def columns(conn, table):
    return {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}


def table_exists(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def require_columns(conn, table, required):
    missing = set(required) - columns(conn, table)
    if missing:
        raise ValueError(f"{table}: missing columns {sorted(missing)}")


def nvtx_ranges(conn):
    require_columns(conn, "NVTX_EVENTS", {"start", "end", "globalTid"})
    available = columns(conn, "NVTX_EVENTS")
    direct = "n.text" if "text" in available else "NULL"
    joined = "s.value" if "textId" in available else "NULL"
    join_sql = "LEFT JOIN StringIds s ON s.id=n.textId" if "textId" in available else ""
    query = (
        "SELECT n.rowid AS rowid,n.start,n.end,n.globalTid,"
        f"COALESCE(NULLIF({direct},''),{joined},'') AS name "
        f"FROM NVTX_EVENTS n {join_sql} "
        "WHERE n.end>n.start ORDER BY n.start,n.end DESC"
    )
    return [dict(row) for row in conn.execute(query)]


class RangeIndex:
    """Index non-overlapping same-thread ranges, refusing ambiguous ownership."""

    def __init__(self, ranges):
        self.ranges = sorted(ranges, key=lambda item: item["start"])
        self.starts = [item["start"] for item in self.ranges]
        if any(left["end"] > right["start"] for left, right in zip(self.ranges, self.ranges[1:])):
            raise ValueError("Overlapping same-level NVTX ranges; cannot assign ownership safely")

    def containing(self, start, end):
        position = bisect.bisect_right(self.starts, start) - 1
        if position >= 0:
            item = self.ranges[position]
            if item["start"] <= start and end <= item["end"]:
                return item
        return None


class ThreadRangeIndex:
    """Use the deepest containing range on the actual launching thread.

    Parent ranges organize nested nodes. A runtime/kernel is counted once;
    crossing ranges are rejected rather than attributed by time proximity.
    """

    def __init__(self, ranges):
        self.roots = {}
        self.children = {}
        grouped = defaultdict(list)
        for item in ranges:
            grouped[item["globalTid"]].append(item)
        for tid, items in grouped.items():
            roots, stack = [], []
            for item in sorted(items, key=lambda row: (row["start"], -row["end"], row["rowid"])):
                while stack and item["start"] >= stack[-1]["end"]:
                    stack.pop()
                if stack:
                    if item["end"] > stack[-1]["end"]:
                        raise ValueError("Crossing NVTX ranges on one thread cannot be assigned safely")
                    self.children.setdefault(stack[-1]["rowid"], []).append(item)
                else:
                    roots.append(item)
                stack.append(item)
            self.roots[tid] = RangeIndex(roots)
        self.children = {key: RangeIndex(value) for key, value in self.children.items()}

    def containing(self, tid, start, end):
        index = self.roots.get(tid)
        item = index.containing(start, end) if index is not None else None
        while item is not None:
            children = self.children.get(item["rowid"])
            child = children.containing(start, end) if children is not None else None
            if child is None:
                break
            item = child
        return item


def in_process(global_tid, global_pid):
    # Nsight's globalTid packs the native thread ID in the low 24 bits.
    return (int(global_tid) & ~((1 << 24) - 1)) == int(global_pid)


def process_identity(conn, ranges, pid):
    require_columns(conn, "PROCESSES", {"globalPid", "pid"})
    processes = list(conn.execute("SELECT DISTINCT globalPid,pid FROM PROCESSES"))
    candidates = []
    tids = {item["globalTid"] for item in ranges if item["name"] == "tide.batch.step"}
    for global_pid, process_pid in processes:
        if pid is not None and int(process_pid) != pid:
            continue
        main_tid = int(global_pid) | int(process_pid)
        if main_tid in tids:
            candidates.append((int(global_pid), int(process_pid), main_tid))
    if len(candidates) != 1:
        raise ValueError(f"Expected one rank main thread with tide.batch.step; candidates={candidates}; supply --pid-rank0/1")
    return candidates[0]


def read_run_metadata(run_dir, rank):
    result = {"pid": None, "metrics": [], "event_count": 0, "sources": {}, "warnings": []}
    if run_dir is None:
        return result
    event_paths = sorted(run_dir.rglob(f"timeline_events_rank{rank}.jsonl"))
    if len(event_paths) == 1:
        result["sources"]["timeline_jsonl"] = str(event_paths[0])
        pids = set()
        with event_paths[0].open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                result["event_count"] += 1
                if item.get("rank") == rank and item.get("pid") is not None:
                    pids.add(int(item["pid"]))
        if len(pids) == 1:
            result["pid"] = pids.pop()
        elif pids:
            result["warnings"].append(f"JSONL has multiple PIDs: {sorted(pids)}")
    elif len(event_paths) > 1:
        result["warnings"].append("Multiple matching JSONL files; refusing to choose one")
    metric_paths = sorted(run_dir.rglob(f"metrics_batch_rank{rank}.tsv"))
    if len(metric_paths) == 1:
        result["sources"]["metrics_tsv"] = str(metric_paths[0])
        with metric_paths[0].open(encoding="utf-8", newline="") as handle:
            result["metrics"] = list(csv.DictReader(handle, delimiter="\t"))
    elif len(metric_paths) > 1:
        result["warnings"].append("Multiple matching metrics files; refusing ordinal alignment")
    return result


def kernel_records(conn, global_pid):
    required = {"start", "end", "globalPid", "correlationId", "deviceId", "streamId"}
    require_columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL", required)
    available = columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
    names, joins = [], []
    for index, field in enumerate(("demangledName", "shortName", "mangledName")):
        if field in available:
            joins.append(f"LEFT JOIN StringIds s{index} ON s{index}.id=k.{field}")
            names.append(f"NULLIF(s{index}.value,'')")
    name_sql = "COALESCE(" + ",".join(names + ["'kernel-name-unavailable'"]) + ")" if names else "'kernel-name-unavailable'"
    optional = [f"k.{field}" if field in available else f"0 AS {field}" for field in ("graphId", "graphNodeId")]
    query = (
        "SELECT k.rowid AS rowid,k.start,k.end,k.globalPid,k.correlationId,k.deviceId,k.streamId,"
        f"{name_sql} AS name,{','.join(optional)} FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        f"{' '.join(joins)} WHERE k.globalPid=? AND k.end>k.start ORDER BY k.start"
    )
    result = [dict(row) for row in conn.execute(query, (global_pid,))]
    for item in result:
        item["category"] = "nccl" if "nccl" in item["name"].lower() else "compute"
    return result


def runtime_records(conn, global_pid, start, end):
    require_columns(conn, "CUPTI_ACTIVITY_KIND_RUNTIME", {"start", "end", "globalTid", "correlationId", "nameId"})
    query = (
        "SELECT r.rowid AS rowid,r.start,r.end,r.globalTid,r.correlationId,s.value AS name "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r LEFT JOIN StringIds s ON s.id=r.nameId "
        "WHERE (r.globalTid & -16777216)=? AND r.start>=? AND r.end<=? ORDER BY r.start"
    )
    return [dict(row) for row in conn.execute(query, (global_pid, start, end))]


def copy_records(conn, global_pid):
    result = []
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if not table_exists(conn, table):
            continue
        require_columns(conn, table, {"start", "end", "globalPid", "deviceId", "streamId"})
        for row in conn.execute(f"SELECT start,end,deviceId,streamId FROM {table} WHERE globalPid=? AND end>start", (global_pid,)):
            item = dict(row)
            item["category"] = "memcpy" if table.endswith("MEMCPY") else "memset"
            result.append(item)
    return result


def stage_name(name):
    if not name.startswith(PC1_PREFIX):
        return None
    microbatch, separator, remainder = name[len(PC1_PREFIX):].partition(":")
    if separator and microbatch.isdigit():
        for phase in sorted(PHASES, key=len, reverse=True):
            if remainder == phase or remainder.startswith(phase + ":") or remainder.startswith(phase + "|"):
                return int(microbatch), phase
    return None


def scope_metadata(name):
    _, separator, suffix = name.partition("|")
    if not separator:
        return {}
    return dict(field.split("=", 1) for field in suffix.split(",") if "=" in field)


def launches_kernel(name):
    name = (name or "").lower()
    return ("launch" in name and "kernel" in name) or "graphlaunch" in name


def statistics_summary(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values), "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90_nearest_rank": ordered[max(0, (9 * len(ordered) + 9) // 10 - 1)],
        "min": ordered[0], "max": ordered[-1],
    }


def cross_stream_overlap(projection, communication):
    by_p, by_c = defaultdict(list), defaultdict(list)
    for record in projection:
        by_p[(record["deviceId"], record["streamId"])].append((record["start"], record["end"]))
    for record in communication:
        by_c[(record["deviceId"], record["streamId"])].append((record["start"], record["end"]))
    result = []
    for (device, p_stream), p_intervals in by_p.items():
        for (c_device, c_stream), c_intervals in by_c.items():
            if device == c_device and p_stream != c_stream:
                result.extend(intersection(p_intervals, c_intervals))
    return union(result)


def target_pair(identity, name, compute_mb, comm_mb, compute_ranges,
                comm_ranges, by_stage, stage_runtimes, device):
    """Measure a target using correlated kernels, including nested hook nodes."""
    compute_all = [kernel for stage in compute_ranges for kernel in by_stage[stage["rowid"]]]
    comm_all = [kernel for stage in comm_ranges for kernel in by_stage[stage["rowid"]]]
    compute = [item for item in compute_all if item["category"] == "compute"]
    comm = [item for item in comm_all if item["category"] == "nccl"]
    reasons = []
    if not compute_ranges or not comm_ranges:
        reasons.append("missing_stage_range")
    # A surviving child phase must not turn a partially captured image stage
    # into a complete target. Kernel-free scopes are legitimate: require the
    # scope itself and audit any actual launch APIs below, rather than inventing
    # activity from nearby unknown kernels or requiring a kernel per scope.
    required_compute, required_comm_groups = {
        "image_c1": (("Render", "Loss", "IntermediateGrad"),
                     (("C1:Metadata",), ("C1:Payload",))),
        "image_intermediate_c1": (("IntermediateGrad",),
                                  (("C1:Metadata",), ("C1:Payload",))),
        "intermediate_c2": (("IntermediateGrad",), (("C2", "C2:Payload"),)),
        "final_c2": (("FinalGrad",), (("C2", "C2:Payload"),)),
    }[name]
    observed_compute = {stage["phase"] for stage in compute_ranges}
    observed_comm = {stage["phase"] for stage in comm_ranges}
    for phase in required_compute:
        if phase not in observed_compute:
            reasons.append(f"missing_compute_phase:{phase}")
    for alternatives in required_comm_groups:
        if not observed_comm.intersection(alternatives):
            reasons.append("missing_communication_phase:" + "|".join(alternatives))
    if not compute:
        reasons.append("no_correlated_compute_kernels")
    if not comm:
        reasons.append("no_correlated_nccl_kernels")
    if any(item["name"] == "kernel-name-unavailable" for item in compute_all + comm_all):
        reasons.append("missing_kernel_names")
    if any("GraphLaunch" in item["runtime_name"] for item in compute_all + comm_all):
        reasons.append("cuda_graph_phase_assignment_requires_validation")
    for stage in compute_ranges + comm_ranges:
        associated = {item["correlationId"] for item in by_stage[stage["rowid"]]}
        if any(launches_kernel(api["name"]) and api["correlationId"] not in associated
               for api in stage_runtimes[stage["rowid"]]):
            reasons.append("unmatched_phase_kernel_launch_api")
            break
    overlap = cross_stream_overlap(compute, comm)
    return {
        **identity, "target_type": name,
        "pair": f"{name}:compute({compute_mb})/comm({comm_mb})",
        "compute_microbatch": compute_mb, "communication_microbatch": comm_mb,
        "status": "assessable" if not reasons else ";".join(reasons),
        "compute_kernel_count": len(compute), "communication_kernel_count": len(comm),
        "compute_active_ms": duration_ms(intervals(compute)),
        "communication_active_ms": duration_ms(intervals(comm)),
        # Legacy field aliases keep existing report readers usable.
        "projection_active_ms": duration_ms(intervals(compute)),
        "c1_nccl_active_ms": duration_ms(intervals(comm)),
        "cross_stream_overlap_ms": duration_ms(overlap) if not reasons else None,
        "overlap_interval_count": len(overlap) if not reasons else None,
        "compute_stream_ids": sorted({item["streamId"] for item in compute}),
        "communication_stream_ids": sorted({item["streamId"] for item in comm}),
        "device_id": device,
    }, overlap if not reasons else []


def analyze_connection(conn, *, rank, pid=None, trim=1, mode="auto", metadata=None):
    conn.row_factory = sqlite3.Row
    metadata = metadata or {"metrics": [], "warnings": [], "sources": {}}
    all_ranges = nvtx_ranges(conn)
    global_pid, pid, main_tid = process_identity(conn, all_ranges, pid)
    ranges = [item for item in all_ranges if in_process(item["globalTid"], global_pid)]
    steps = [item for item in ranges if item["globalTid"] == main_tid and item["name"] == "tide.batch.step"]
    if not steps:
        raise ValueError("No complete tide.batch.step ranges")
    step_index = RangeIndex(steps)
    compute_index = RangeIndex([item for item in ranges if item["globalTid"] == main_tid and item["name"] == "tide.batch.compute"])
    stages = []
    for item in ranges:
        parsed = stage_name(item["name"])
        if parsed is None:
            continue
        parent = step_index.containing(item["start"], item["end"])
        compute = compute_index.containing(item["start"], item["end"])
        if parent is None or compute is None:
            continue
        item.update(microbatch=parsed[0], phase=parsed[1], batch_id=parent["rowid"],
                    scope_metadata=scope_metadata(item["name"]))
        stages.append(item)
    detected_mode = "pc1" if any(item["phase"] == "Projection" for item in stages) else "off"
    if mode != "auto" and mode != detected_mode:
        raise ValueError(f"Requested {mode}, but observed NVTX mode is {detected_mode}")
    mode = detected_mode
    stage_index = ThreadRangeIndex(stages)
    kernels = kernel_records(conn, global_pid)
    runtimes = runtime_records(conn, global_pid, steps[0]["start"], steps[-1]["end"])
    runtime_map = defaultdict(list)
    for item in runtimes:
        runtime_map[item["correlationId"]].append(item)
    devices = {item["deviceId"] for item in kernels if item["end"] > steps[0]["start"] and item["start"] < steps[-1]["end"]}
    if len(devices) != 1:
        raise ValueError(f"Expected one CUDA device per rank during training, observed {sorted(devices)}")
    device = next(iter(devices))
    kernels = [item for item in kernels if item["deviceId"] == device]
    copies = [item for item in copy_records(conn, global_pid) if item["deviceId"] == device]
    by_stage = defaultdict(list)
    stage_runtimes = defaultdict(list)
    for runtime in runtimes:
        stage = stage_index.containing(runtime["globalTid"], runtime["start"], runtime["end"])
        if stage is not None:
            stage_runtimes[stage["rowid"]].append(runtime)
    audit = []
    diagnostics = Counter()
    for kernel in kernels:
        matching = runtime_map.get(kernel["correlationId"], [])
        runtime = matching[0] if len(matching) == 1 else None
        stage = None
        if len(matching) != 1:
            diagnostics["no_unique_process_runtime_for_kernel"] += 1
        elif kernel["start"] < runtime["start"]:
            diagnostics["kernel_precedes_matched_runtime"] += 1
        else:
            stage = stage_index.containing(runtime["globalTid"], runtime["start"], runtime["end"])
        parent = step_index.containing(kernel["start"], kernel["end"])
        if stage is None and parent is None:
            continue
        if stage is not None:
            kernel["runtime_name"] = runtime["name"] or "runtime-name-unavailable"
            by_stage[stage["rowid"]].append(kernel)
        else:
            diagnostics["unknown_phase_kernel_rows"] += 1
        runtime = runtime or {}
        stage = stage or {"microbatch": None, "phase": "unknown", "batch_id": parent["rowid"]}
        audit.append({
            "rank": rank, "trace_batch": None, "optimizer_step": None, "iteration": None,
            "microbatch": stage["microbatch"], "phase": stage["phase"],
            "batch_id": stage["batch_id"], "kernel_rowid": kernel["rowid"],
            "kernel_name": kernel["name"], "category": kernel["category"],
            "device_id": kernel["deviceId"], "stream_id": kernel["streamId"],
            "correlation_id": kernel["correlationId"], "runtime_global_tid": runtime.get("globalTid"),
            "kernel_global_pid": kernel["globalPid"], "runtime_api": runtime.get("name"),
            "runtime_start_ns": runtime.get("start"), "runtime_end_ns": runtime.get("end"),
            "nvtx_rowid": stage.get("rowid"), "nvtx_start_ns": stage.get("start"), "nvtx_end_ns": stage.get("end"),
            "kernel_start_ns": kernel["start"], "kernel_end_ns": kernel["end"],
            "graph_id": kernel["graphId"], "graph_node_id": kernel["graphNodeId"],
        })
    selected = steps[trim:len(steps) - trim] if trim else steps
    selected_ids = {item["rowid"] for item in selected}
    metrics = metadata.get("metrics", [])
    aligned_metrics = len(metrics) == len(steps)
    warnings = list(metadata.get("warnings", []))
    if metrics and not aligned_metrics:
        warnings.append(f"Captured batches={len(steps)}, metric rows={len(metrics)}; numeric iteration/optimizer IDs left unset")
    elif aligned_metrics:
        warnings.append("Iteration/optimizer IDs aligned by chronological ordinal and equal total counts; fixed NVTX labels do not encode these IDs")
    if not selected:
        warnings.append("No batches remain after trimming")
    pairs, batches, stage_statistics = [], [], []
    batch_identity = {}
    for trace_batch, step in enumerate(steps, 1):
        metric = metrics[trace_batch - 1] if aligned_metrics else {}
        identity = {"rank": rank, "trace_batch": trace_batch,
                    "optimizer_step": metric.get("optimizer_step"), "iteration": metric.get("iteration")}
        batch_identity[step["rowid"]] = identity
        if step["rowid"] not in selected_ids:
            continue
        compute = [item for item in kernels if item["category"] == "compute"]
        communication = [item for item in kernels if item["category"] == "nccl"]
        p_intervals = clipped(compute, step["start"], step["end"])
        c_intervals = clipped(communication, step["start"], step["end"])
        gpu_intervals = p_intervals + c_intervals + clipped(copies, step["start"], step["end"])
        host_sync = [item for item in runtimes if item["globalTid"] == main_tid and "Synchronize" in (item["name"] or "")]
        cpu_plan = [item for item in ranges if item["name"].startswith("tide.cpu.plan.")]
        wall_ms = (step["end"] - step["start"]) / 1e6
        batch = {
            **identity, "mode": mode, "device_id": device,
            "start_ns": step["start"], "end_ns": step["end"], "step_wall_ms": wall_ms,
            "non_nccl_kernel_active_ms": duration_ms(p_intervals),
            "nccl_kernel_active_ms": duration_ms(c_intervals),
            "any_kernel_active_ms": duration_ms(p_intervals + c_intervals),
            "compute_nccl_cross_stream_overlap_ms": duration_ms(cross_stream_overlap(
                [dict(item, start=max(step["start"], item["start"]), end=min(step["end"], item["end"])) for item in compute if item["end"] > step["start"] and item["start"] < step["end"]],
                [dict(item, start=max(step["start"], item["start"]), end=min(step["end"], item["end"])) for item in communication if item["end"] > step["start"] and item["start"] < step["end"]],
            )),
            "memcpy_active_ms": duration_ms(clipped([item for item in copies if item["category"] == "memcpy"], step["start"], step["end"])),
            "memset_active_ms": duration_ms(clipped([item for item in copies if item["category"] == "memset"], step["start"], step["end"])),
            "any_cuda_activity_ms": duration_ms(gpu_intervals),
            "no_recorded_cuda_activity_ms": max(0.0, wall_ms - duration_ms(gpu_intervals)),
            "main_thread_cuda_sync_api_ms": duration_ms(clipped(host_sync, step["start"], step["end"])),
            "worker_thread_cuda_sync_api_ms": duration_ms(clipped([
                api for api in runtimes if api["globalTid"] != main_tid and "Synchronize" in (api["name"] or "")
            ], step["start"], step["end"])),
            "main_thread_cpu_plan_nvtx_ms": duration_ms(clipped(cpu_plan, step["start"], step["end"])),
            "graph_launch_api_count": sum("GraphLaunch" in (item["name"] or "") for item in runtimes if step["start"] <= item["start"] and item["end"] <= step["end"]),
        }
        batches.append(batch)
        for stage in stages:
            if stage["batch_id"] != step["rowid"]:
                continue
            stage_kernels = by_stage[stage["rowid"]]
            stage_apis = stage_runtimes[stage["rowid"]]
            launch_apis = [item for item in stage_apis if launches_kernel(item["name"])]
            associated = {item["correlationId"] for item in stage_kernels}
            stage_statistics.append({
                **identity, "microbatch": stage["microbatch"], "phase": stage["phase"],
                "global_tid": stage["globalTid"], "thread_role": "main" if stage["globalTid"] == main_tid else "worker",
                "nvtx_rowid": stage["rowid"], "nvtx_start_ns": stage["start"], "nvtx_end_ns": stage["end"],
                "cpu_nvtx_span_ms": (stage["end"] - stage["start"]) / 1e6,
                "kernel_launch_api_count": len(launch_apis),
                "unmatched_kernel_launch_api_count": sum(item["correlationId"] not in associated for item in launch_apis),
                "correlated_kernel_count": len(stage_kernels),
                "correlated_kernel_active_ms": duration_ms(intervals(stage_kernels)),
                "correlated_nccl_kernel_active_ms": duration_ms(intervals([item for item in stage_kernels if item["category"] == "nccl"])),
                "host_cuda_sync_api_ms": duration_ms(intervals([item for item in stage_apis if "Synchronize" in (item["name"] or "")])),
                "stream_ids": sorted({item["streamId"] for item in stage_kernels}),
            })
        if mode != "pc1":
            continue
        batch_stages = [item for item in stages if item["batch_id"] == step["rowid"]]
        stage_lookup = defaultdict(list)
        for item in batch_stages:
            stage_lookup[(item["microbatch"], item["phase"])].append(item)
        mb_indices = sorted({item["microbatch"] for item in batch_stages})
        for current, following in zip(mb_indices, mb_indices[1:]):
            if following != current + 1:
                continue
            p_ranges = stage_lookup[(following, "Projection")]
            previous_p = stage_lookup[(current, "Projection")]
            if len(p_ranges) == len(previous_p) == 1:
                previous_slot = previous_p[0]["scope_metadata"].get("slot")
                next_slot = p_ranges[0]["scope_metadata"].get("slot")
                if previous_slot is not None and next_slot is not None and int(next_slot) <= int(previous_slot):
                    continue  # A new fixed window has no cross-window overlap target.
            c_ranges = stage_lookup[(current, "C1:Metadata")] + stage_lookup[(current, "C1:Payload")]
            p_all = [kernel for item in p_ranges for kernel in by_stage[item["rowid"]]]
            c_all = [kernel for item in c_ranges for kernel in by_stage[item["rowid"]]]
            p_kernels = [item for item in p_all if item["category"] == "compute"]
            c_kernels = [item for item in c_all if item["category"] == "nccl"]
            reasons = []
            if len(p_ranges) != 1 or len(c_ranges) != 2:
                reasons.append("missing_or_duplicate_stage_range")
            if not p_kernels:
                reasons.append("no_correlated_projection_compute_kernels")
            if not c_kernels:
                reasons.append("no_correlated_c1_nccl_kernels")
            if any(item["name"] == "kernel-name-unavailable" for item in p_all + c_all):
                reasons.append("missing_kernel_names")
            if any("GraphLaunch" in item["runtime_name"] for item in p_all + c_all):
                reasons.append("cuda_graph_phase_assignment_requires_validation")
            pair_stages = p_ranges + c_ranges
            for stage in pair_stages:
                associated = {item["correlationId"] for item in by_stage[stage["rowid"]]}
                if any(launches_kernel(item["name"]) and item["correlationId"] not in associated for item in stage_runtimes[stage["rowid"]]):
                    reasons.append("unmatched_phase_kernel_launch_api")
                    break
            overlap = cross_stream_overlap(p_kernels, c_kernels)
            p_ms, c_ms, overlap_ms = duration_ms(intervals(p_kernels)), duration_ms(intervals(c_kernels)), duration_ms(overlap)
            pairs.append({
                **identity, "target_type": "projection_c1", "pair": f"C1({current})/P({following})", "c1_microbatch": current,
                "projection_microbatch": following, "status": "assessable" if not reasons else ";".join(reasons),
                "projection_kernel_count": len(p_kernels), "c1_nccl_kernel_count": len(c_kernels),
                "projection_active_ms": p_ms, "c1_nccl_active_ms": c_ms,
                "projection_c1_activity_union_ms": duration_ms(intervals(p_kernels) + intervals(c_kernels)),
                "cross_stream_overlap_ms": overlap_ms if not reasons else None,
                "overlap_fraction_of_projection": overlap_ms / p_ms if not reasons and p_ms else None,
                "overlap_fraction_of_c1_nccl": overlap_ms / c_ms if not reasons and c_ms else None,
                "overlap_interval_count": len(overlap) if not reasons else None,
                "projection_stream_ids": sorted({item["streamId"] for item in p_kernels}),
                "c1_nccl_stream_ids": sorted({item["streamId"] for item in c_kernels}),
                "device_id": device,
            })
            if not reasons:
                batch.setdefault("_target_intervals", defaultdict(list))["projection_c1"].extend(overlap)
            if not any(item["phase"] in {"IntermediateGrad", "FinalGrad", "C2:Payload"}
                       for item in batch_stages):
                continue  # Legacy P/C1 reports do not claim split-backward targets.

            def phases(microbatch, names):
                return [stage for phase in names for stage in stage_lookup[(microbatch, phase)]]

            targets = (
                ("image_c1", current, following, ("Render", "Loss", "IntermediateGrad"), ("C1:Metadata", "C1:Payload")),
                ("image_intermediate_c1", current, following, ("IntermediateGrad",), ("C1:Metadata", "C1:Payload")),
                ("intermediate_c2", following, current, ("IntermediateGrad",), ("C2", "C2:Payload")),
                ("final_c2", current, following, ("FinalGrad", "AccumulateGrad"), ("C2", "C2:Payload")),
            )
            for name, compute_mb, comm_mb, compute_phases, comm_phases in targets:
                row, common = target_pair(
                    identity, name, compute_mb, comm_mb,
                    phases(compute_mb, compute_phases), phases(comm_mb, comm_phases),
                    by_stage, stage_runtimes, device,
                )
                pairs.append(row)
                batch.setdefault("_target_intervals", defaultdict(list))[name].extend(common)
        target_intervals = batch.pop("_target_intervals", {})
        batch["target_pair_samples"] = dict(Counter(row["target_type"] for row in pairs
            if row["trace_batch"] == trace_batch))
        batch["target_overlap_ms"] = {
            name: duration_ms(target_intervals.get(name, [])) if all(
                row["status"] == "assessable" for row in pairs
                if row["trace_batch"] == trace_batch and row["target_type"] == name
            ) else None for name in batch["target_pair_samples"]
        }
    for item in audit:
        item.update(batch_identity[item.pop("batch_id")])
    audit = [item for item in audit if item["trace_batch"] in {row["trace_batch"] for row in batches}]
    assessable = [item for item in pairs if item["status"] == "assessable"]
    positive = [item for item in assessable if item["cross_stream_overlap_ms"] > 0]
    pair_summary = {}
    for name in sorted({item["pair"] for item in pairs}):
        selected_pairs = [item for item in assessable if item["pair"] == name]
        pair_summary[name] = {
            "observed_pairs": sum(item["pair"] == name for item in pairs),
            "assessable_pairs": len(selected_pairs),
            "overlap_positive_pairs": sum(item["cross_stream_overlap_ms"] > 0 for item in selected_pairs),
            "overlap_occurrence_fraction": sum(item["cross_stream_overlap_ms"] > 0 for item in selected_pairs) / len(selected_pairs) if selected_pairs else None,
            "overlap_ms": statistics_summary([item["cross_stream_overlap_ms"] for item in selected_pairs]),
            "projection_active_ms": statistics_summary([item["projection_active_ms"] for item in selected_pairs]),
            "c1_nccl_active_ms": statistics_summary([item["c1_nccl_active_ms"] for item in selected_pairs]),
        }
    target_summary = {}
    for name in sorted({item["target_type"] for item in pairs}):
        candidates = [item for item in pairs if item["target_type"] == name]
        valid = [item for item in candidates if item["status"] == "assessable"]
        target_summary[name] = {
            "samples": len(candidates), "assessable_samples": len(valid),
            "positive_samples": sum(item["cross_stream_overlap_ms"] > 0 for item in valid),
            "per_pair_overlap_ms": statistics_summary([item["cross_stream_overlap_ms"] for item in valid]),
            "per_batch_union_overlap_ms": statistics_summary([
                batch.get("target_overlap_ms", {}).get(name, 0.0) for batch in batches
                if batch.get("target_pair_samples", {}).get(name)
                and batch.get("target_overlap_ms", {}).get(name) is not None
            ]),
        }
    diagnostics["main_thread_graph_launch_apis"] = sum(item["globalTid"] == main_tid and "GraphLaunch" in (item["name"] or "") for item in runtimes)
    diagnostics["process_graph_launch_apis"] = sum("GraphLaunch" in (item["name"] or "") for item in runtimes)
    diagnostics["kernel_rows_with_graph_identity"] = sum(bool(item["graphId"] or item["graphNodeId"]) for item in kernels)
    diagnostics["stages_without_correlated_kernels"] = sum(not by_stage[item["rowid"]] for item in stages)
    diagnostics["selected_stage_unmatched_kernel_launch_apis"] = sum(item["unmatched_kernel_launch_api_count"] for item in stage_statistics)
    training_kernels = [item for item in kernels if item["start"] < steps[-1]["end"] and item["end"] > steps[0]["start"]]
    diagnostics["training_kernel_rows"] = len(training_kernels)
    diagnostics["training_kernel_rows_with_unique_process_runtime"] = sum(
        len(runtime_map.get(item["correlationId"], [])) == 1
        for item in training_kernels
    )
    if mode == "pc1" and len(assessable) != len(pairs):
        warnings.append("Unassessable pairs are missing association/phase evidence; their overlap is null, not zero")
    summary = {
        "rank": rank, "pid": pid, "global_pid": global_pid, "main_global_tid": main_tid, "related_global_tids": sorted({item["globalTid"] for item in runtimes}),
        "analysis_schema_version": 2,
        "mode": mode, "device_id": device, "captured_complete_batches": len(steps),
        "trimmed_each_end": trim, "analyzed_batches": len(batches),
        "observed_pc1_pairs": len(pairs), "assessable_pc1_pairs": len(assessable),
        "overlap_positive_pairs": len(positive),
        "overlap_occurrence_fraction": len(positive) / len(assessable) if assessable else None,
        "pair_summary": pair_summary, "target_summary": target_summary,
        "batch_statistics_ms": {field: statistics_summary([item[field] for item in batches]) for field in (
            "step_wall_ms", "non_nccl_kernel_active_ms", "nccl_kernel_active_ms", "any_kernel_active_ms",
            "compute_nccl_cross_stream_overlap_ms", "memcpy_active_ms", "any_cuda_activity_ms",
            "no_recorded_cuda_activity_ms", "main_thread_cuda_sync_api_ms", "worker_thread_cuda_sync_api_ms", "main_thread_cpu_plan_nvtx_ms",
        )},
        "correlation_diagnostics": dict(diagnostics), "warnings": warnings,
        "run_sources": metadata.get("sources", {}),
        "iteration_mapping": "chronological_ordinal_equal_total_counts" if aligned_metrics else "trace_batch_ordinal_only",
    }
    return {"summary": summary, "batches": batches, "pairs": pairs, "stages": stage_statistics}, audit


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["rank", "trace_batch"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def self_test():
    """Prove queued kernel overlap and process isolation with an in-memory export."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
      CREATE TABLE StringIds(id INTEGER, value TEXT);
      CREATE TABLE PROCESSES(globalPid INTEGER, pid INTEGER, name TEXT);
      CREATE TABLE NVTX_EVENTS(eventId INTEGER PRIMARY KEY,start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
      CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(runtimeEventId INTEGER PRIMARY KEY,start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
      CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(kernelEventId INTEGER PRIMARY KEY,start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,deviceId INTEGER,streamId INTEGER,shortName INTEGER,graphId INTEGER,graphNodeId INTEGER);
    """)
    pid, global_pid = 123, 123 << 24
    main_tid = global_pid | pid
    other_pid = 456 << 24
    conn.executemany("INSERT INTO PROCESSES VALUES(?,?,?)", [(global_pid, pid, "python"), (other_pid, 456, "python")])
    conn.executemany("INSERT INTO StringIds VALUES(?,?)", [(1, "projection_kernel"), (2, "ncclKernel_SendRecv"), (3, "cudaLaunchKernel")])
    spans = [(0, 1000, "tide.batch.step"), (0, 900, "tide.batch.compute"),
             (0, 20, "Tide:S1:MB0:Projection"), (30, 35, "Tide:S1:MB0:C1:Metadata"),
             (36, 45, "Tide:S1:MB0:C1:Payload"), (46, 60, "Tide:S1:MB1:Projection"),
             (71, 77, "Tide:S1:MB1:C1:Metadata"), (78, 85, "Tide:S1:MB1:C1:Payload"),
             (61, 70, "Tide:S1:MB2:Projection")]
    conn.executemany("INSERT INTO NVTX_EVENTS(start,end,globalTid,text,textId) VALUES(?,?,?,?,NULL)", [(start, end, main_tid, name) for start, end, name in spans])
    launches = [(10, 11, 1), (32, 33, 2), (38, 39, 3), (50, 51, 4), (65, 66, 5), (75, 76, 6), (80, 81, 7)]
    conn.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(start,end,globalTid,correlationId,nameId) VALUES(?,?,?,?,3)", [(start, end, main_tid, correlation) for start, end, correlation in launches])
    kernels = [(20, 40, 1, 7, 1), (100, 150, 2, 24, 2), (150, 350, 3, 24, 2),
               (200, 400, 4, 7, 1), (500, 600, 5, 7, 1), (450, 550, 6, 24, 2), (650, 700, 7, 24, 2)]
    conn.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start,end,globalPid,correlationId,deviceId,streamId,shortName,graphId,graphNodeId) VALUES(?,?,?, ?,0,?,?,0,0)", [(start, end, global_pid, correlation, stream, name) for start, end, correlation, stream, name in kernels])
    conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start,end,globalPid,correlationId,deviceId,streamId,shortName,graphId,graphNodeId) VALUES(0,1000,?,4,0,99,2,0,0)", (other_pid,))
    result, audit = analyze_connection(conn, rank=0, pid=pid, trim=0)
    assert result["summary"]["assessable_pc1_pairs"] == 2
    assert abs(result["pairs"][0]["cross_stream_overlap_ms"] - 150 / 1e6) < 1e-15
    assert abs(result["pairs"][1]["cross_stream_overlap_ms"] - 50 / 1e6) < 1e-15
    assert all(item["kernel_global_pid"] == global_pid for item in audit)
    conn.execute("INSERT INTO StringIds VALUES(4,'cudaGraphLaunch')")
    conn.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET nameId=4 WHERE correlationId=4")
    graph, _ = analyze_connection(conn, rank=0, pid=pid, trim=0)
    assert graph["pairs"][0]["cross_stream_overlap_ms"] is None
    assert "cuda_graph_phase_assignment_requires_validation" in graph["pairs"][0]["status"]
    conn.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET nameId=3 WHERE correlationId=4")
    conn.execute("UPDATE NVTX_EVENTS SET text='Off:Stage' WHERE text LIKE 'Tide:S1:%'")
    off, _ = analyze_connection(conn, rank=0, pid=pid, trim=0, mode="off")
    assert not off["pairs"]
    assert off["batches"][0]["compute_nccl_cross_stream_overlap_ms"] > 0
    conn.execute("DELETE FROM NVTX_EVENTS WHERE text='Off:Stage'")
    conn.executemany("INSERT INTO NVTX_EVENTS(start,end,globalTid,text,textId) VALUES(?,?,?,?,NULL)", [(start, end, main_tid, name) for start, end, name in spans if name.startswith(PC1_PREFIX)])
    conn.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET streamId=7 WHERE globalPid=?", (global_pid,))
    same_stream, _ = analyze_connection(conn, rank=0, pid=pid, trim=0)
    assert all(item["cross_stream_overlap_ms"] == 0 for item in same_stream["pairs"])
    conn.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=4")
    missing, _ = analyze_connection(conn, rank=0, pid=pid, trim=0)
    assert missing["pairs"][0]["cross_stream_overlap_ms"] is None
    assert "no_correlated_projection_compute_kernels" in missing["pairs"][0]["status"]
    parser = argument_parser()
    paths = [f"rank{rank}.sqlite" for rank in range(4)]
    ordered = rank_inputs(parser.parse_args(["--sqlite", *paths]), parser)
    assert ordered == [(rank, Path(path), None) for rank, path in enumerate(paths)]
    legacy = rank_inputs(parser.parse_args(["--rank1-sqlite", "rank1.sqlite", "--pid-rank1", "456"]), parser)
    assert legacy == [(1, Path("rank1.sqlite"), 456)]
    import contextlib
    import io
    for conflicting in (
        ["--sqlite", "rank0.sqlite", "--rank0-sqlite", "legacy.sqlite"],
        ["--sqlite", "rank0.sqlite", "--pid-rank1", "456"],
        ["--sqlite", "rank0.sqlite", "rank0.sqlite"],
    ):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                rank_inputs(parser.parse_args(conflicting), parser)
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError(f"Invalid rank-input selection accepted: {conflicting}")
    conn.close()
    print("Self-test OK: actual overlap/correlation checks plus ordered four-rank CLI, legacy rank/PID flags, mixed flags and duplicate reports rejected")


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--sqlite", type=Path, nargs="+", help="One report per rank in explicit rank0, rank1, ... order")
    parser.add_argument("--rank0-sqlite", type=Path)
    parser.add_argument("--rank1-sqlite", type=Path)
    parser.add_argument("--pid-rank0", type=int)
    parser.add_argument("--pid-rank1", type=int)
    parser.add_argument("--mode", choices=("auto", "pc1", "off"), default="auto")
    parser.add_argument("--trim-batches", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Output prefix for .json, .pairs.csv, .batches.csv and .stages.csv")
    parser.add_argument("--emit-kernels", action="store_true", help="Also save individual correlated kernels with launch and NVTX IDs")
    parser.add_argument("--self-test", action="store_true")
    return parser


def rank_inputs(args, parser):
    if args.sqlite is not None:
        if any(value is not None for value in (
            args.rank0_sqlite, args.rank1_sqlite, args.pid_rank0, args.pid_rank1,
        )):
            parser.error("--sqlite cannot be combined with legacy --rank0-sqlite/--rank1-sqlite or --pid-rank0/--pid-rank1")
        resolved = [path.resolve() for path in args.sqlite]
        if len(set(resolved)) != len(resolved):
            parser.error("--sqlite must supply a different report for each rank")
        return [(rank, path, None) for rank, path in enumerate(args.sqlite)]
    result = [(rank, path, pid) for rank, path, pid in (
        (0, args.rank0_sqlite, args.pid_rank0),
        (1, args.rank1_sqlite, args.pid_rank1),
    ) if path is not None]
    if not result:
        parser.error("Supply --sqlite or legacy --rank0-sqlite/--rank1-sqlite")
    return result


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.trim_batches < 0:
        parser.error("--trim-batches must be nonnegative")
    inputs = rank_inputs(args, parser)
    results, audits = [], []
    for rank, path, requested_pid in inputs:
        metadata = read_run_metadata(args.run_dir, rank)
        pid = requested_pid if requested_pid is not None else metadata.get("pid")
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            result, audit = analyze_connection(conn, rank=rank, pid=pid, trim=args.trim_batches, mode=args.mode, metadata=metadata)
        result["summary"]["sqlite_path"] = str(path.resolve())
        results.append(result)
        audits.extend(audit)
    output = {
        "method": "deepest same-thread NVTX ownership -> all same-process CUDA runtime correlationId -> actual CUPTI kernel intervals",
        "overlap": "union(target compute execution) intersect union(target NCCL execution), same device, different streams; four target classes plus IntermediateGrad subset",
        "limitations": [
            "NCCL kernel duration includes GPU protocol/peer waits; it is not wire transfer time",
            "GPU-active time, CPU plan and host waits overlap and must not be added",
            "No-recorded-CUDA-activity means no captured kernel/memcpy/memset on this device; it is not a causal bottleneck attribution",
            "Uncorrelated launches are unknown; no stage assignment by temporal proximity",
            "Off reports generic within-step compute/NCCL overlap only; no synthetic PC1 microbatch pairs",
            "Raw CUDA-event snapshots and event-based overlap metrics are not used to establish actual kernel overlap",
        ],
        "ranks": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_csv(args.output.with_suffix(".pairs.csv"), [item for result in results for item in result["pairs"]])
        write_csv(args.output.with_suffix(".batches.csv"), [item for result in results for item in result["batches"]])
        write_csv(args.output.with_suffix(".stages.csv"), [item for result in results for item in result["stages"]])
        if args.emit_kernels:
            write_csv(args.output.with_suffix(".kernels.csv"), audits, KERNEL_AUDIT_FIELDS)
        print(json.dumps({"output_prefix": str(args.output), "rank_summaries": [result["summary"] for result in results]}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, sqlite3.Error, OSError) as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        sys.exit(1)
