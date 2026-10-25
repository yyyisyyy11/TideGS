#!/usr/bin/env python3
"""Extract one Micro Pipeline/off SuperPOD run using identical, auditable conventions.

Stdlib only; run next to exported reports on Linux, then use render_timeline.py
locally. Each invocation handles all ranks of ONE mode. Use the same
--batch-start for the Micro Pipeline and off invocations, rather than selecting different
fast windows. All timestamps used for geometry come from Nsight itself.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import sys


SCHEDULE = "tide.cpu.camera.schedule"
STEP = "tide.batch.step"
READ = "tide.io.ssd.read.service"
READ_BATCH = "tide.cpu.resident.read_batch"
NS = 1_000_000_000
BIN_NS = 20_000_000
MIN_WAIT_NS = 100_000
RETAINED = (2, 93)
WARM = (11, 93)
EXPECTED_STEPS = 94
CPU_NAMES = {
    SCHEDULE: "Camera scheduling",
    "tide.wait.camera_prefetch": "Camera load / wait",
    "tide.cpu.plan.initial": "Initial plan",
    "tide.cpu.plan.preview": "Plan preview",
    "tide.cpu.plan.preview.total": "Plan preview (full host call)",
    "tide.cpu.plan.finalize": "Plan finalize",
    "tide.cpu.plan.active_refresh": "Plan active refresh (host call)",
    "tide.cpu.plan.finalize.total": "Plan finalize (full host call)",
    "tide.comm.plan.broadcast": "Plan broadcast (host call)",
    "tide.io.ssd.prefetch.submit": "SSD prefetch submission",
    "tide.cpu.camera.prefetch.submit": "Next-camera prefetch submission",
    "tide.cpu.camera.h2d.submit": "Camera H2D submission",
    "tide.cpu.plan.consume": "Plan consume",
    "tide.gpu.transfer.h2d": "H2D submission",
    "tide.cpu.resident.evict.stage": "Dirty-block D2H staging",
    "tide.cpu.resident.evict.submit": "Writeback submission",
    "tide.cpu.resident.evict": "Resident eviction (full host call)",
    "tide.cpu.resident.load": "Resident load (full host call)",
    "tide.cpu.resident.select": "Resident block selection",
    "tide.cpu.resident.slot_map": "Resident GPU slot mapping",
    READ_BATCH: "Gaussian cache fetch / pack",
    "tide.cpu.resident.cache_fetch": "Gaussian cache fetch / wait",
    "tide.cpu.resident.pack": "Gaussian CPU packing",
    "tide.cpu.resident.h2d.submit": "Gaussian H2D submission",
    "tide.cpu.resident.unpack.prepare": "GPU unpack preparation",
    "tide.cpu.resident.unpack.submit": "GPU unpack submission",
    "tide.cpu.resident.bind": "Resident tensor binding",
    "tide.cpu.resident.active_leaves": "Active Gaussian leaf construction",
    "tide.cpu.projection.submit": "Projection submission (host call)",
    "tide.comm.resident.status": "Resident status exchange",
    "tide.cpu.micro.worker.prepare": "Micro communication worker setup",
    "tide.cpu.micro.layout.local": "Micro camera layout preparation",
    "tide.comm.micro.layout": "Micro camera layout exchange",
    "tide.comm.micro.layout.counts": "Micro Gaussian / camera count exchange",
    "tide.comm.micro.layout.cameras": "Micro camera parameter exchange",
    "tide.comm.micro.layout_status": "Micro layout status exchange",
    "tide.cpu.materialize": "CPU materialize",
    READ: "Storage read service (host call)",
    "tide.gpu.materialize": "GPU materialize submission",
    "tide.batch.compute": "Compute (host call)",
    "tide.gpu.cull.block": "Block cull submission",
    "tide.gpu.optimizer": "Optimizer submission",
    "tide.gpu.optimizer.update": "Optimizer update submission",
    "tide.gpu.writeback.ram": "Writeback to RAM submission",
    "tide.gpu.writeback.d2h": "D2H writeback submission",
    "tide.cpu.loss_sync": "Loss synchronization",
    "tide.comm.loss.gather": "Loss gather (host call)",
    "tide.cpu.logging": "Logging",
    "tide.cpu.gradient.sparse_prepare": "Sparse gradient preparation",
    "tide.cpu.gradient.sparsity_sample": "Gradient sparsity sampling",
    "tide.cpu.gradient.sample_emit": "Gradient sample logging",
    "tide.cpu.writeback.mark_dirty": "Mark updated blocks dirty",
    "tide.cpu.bounds.gpu_submit": "GPU bounds submission",
    "tide.wait.bounds.d2h": "Wait for bounds D2H",
    "tide.cpu.bounds.serialize": "Bounds serialization",
    "tide.comm.bounds.prepare_status": "Bounds preparation status exchange",
    "tide.comm.bounds.gather": "Bounds gather (host call)",
    "tide.cpu.bounds.apply": "Bounds index update",
    "tide.cpu.bounds.submit": "Bounds refresh submission",
    "tide.comm.bounds.apply_status": "Bounds application status exchange",
    "tide.cpu.retention.prepare": "Resident retention preparation",
    "tide.comm.retention.status": "Retention status exchange",
    "tide.cpu.metrics.cuda_timing": "CUDA timing collection",
    "tide.cpu.metrics.timeline_emit": "Timeline event logging",
    "tide.cpu.metrics.io_emit": "I/O metrics logging",
    "tide.cpu.metrics.batch_emit": "Batch metrics logging",
    "tide.comm.metrics.gather": "Batch metrics gather",
    "tide.cpu.camera.cleanup": "Camera cleanup",
    "tide.cpu.memory.cleanup": "Memory cleanup",
    "tide.cpu.cache.hint": "Future cache hint",
    "tide.cpu.writeback.plan": "Writeback planning",
    "tide.cpu.writeback.orchestrate": "Writeback orchestration",
    "tide.cpu.writeback.pack": "Writeback packing",
    "tide.cpu.writeback.commit": "Writeback commit",
    "tide.io.ssd.read.submit": "SSD read submission",
    "tide.cpu.order.bitmap": "Camera order bitmap",
    "tide.cpu.order.camera": "Camera order selection",
    "tide.cpu.order.counts": "Camera order counts",
    "tide.cpu.order.distance_matrix": "Camera order distance matrix",
    "tide.cpu.order.tsp": "Camera TSP order",
    "tide.comm.batch.barrier": "Batch rank barrier",
    "tide.sync.barrier": "Barrier (host call)",
    "tide.wait.inflight_read": "Wait for inflight read",
    "tide.wait.prefetch": "Wait for prefetch",
    "tide.wait.plan": "Wait for plan",
    "tide.wait.bounds_refresh": "Wait for bounds refresh",
    "tide.wait.h2d_staging": "Wait for H2D staging",
    "tide.wait.gpu_writeback": "Wait for GPU writeback",
    "tide.wait.gpu_gradient": "Wait for GPU gradient",
    "tide.wait.gpu_transfer": "Wait for GPU transfer",
    "tide.wait.ssd_writeback": "Wait for SSD writeback",
    "tide.comm.object_collective": "Object collective (host call)",
}
RUNTIME_WAIT = re.compile(r"Synchronize|^cudaMemcpy(?:_v\d+)?$|^cudaMemset(?:_v\d+)?$", re.I)
OSRT_WAIT = re.compile(r"wait|poll|select|sleep|join|sem_timed|futex", re.I)
MICRO_PIPELINE_PHASE = re.compile(r"^Tide:S[12]:MB\d+:(Projection|C1(?::Metadata|:Payload|:Wait)?|C2(?::Payload|:Wait)?|Render|Loss|Backward|IntermediateGrad|FinalGrad|AccumulateGrad|Enqueue|FutureWait|Worker)(?::[^|]+)?(?:\|.*)?$")
OPTIMIZER_HOST_PHASE = re.compile(r"^Tide:S[12]:Optimizer:(?:adam|3dgs2_tr)(?::(?:Prepare|Commit))?(?:\|.*)?$")
PROJECTION_HOST_PHASE = re.compile(r"^Tide:S1:MB0:Projection(?:\|.*)?$")
HOST_GAP_STAGES = (
    "plan_prefetch", "camera_fetch_wait", "camera_transfer", "eviction_load", "cache_to_gpu",
    "compute_preparation", "update_tail", "bounds", "rank_retention",
    "metrics_record", "main_loop_return", "storage_read", "other_wait",
)
# A caller-side enqueue/FutureWait or a legacy C2 autograd node is not
# evidence of a communication worker. Bare backward stages are callers too;
# only node-qualified scopes (plus the explicit AccumulateGrad node) certify
# an autograd thread. Both identity-suffixed and older bare labels are valid.
COMM_THREAD_NVTX = re.compile(
    r"^Tide:S[12]:MB\d+:(?:C[12]:(?:Metadata|Payload|Worker)|Worker(?::C[12])?)(?:\|.*)?$")
AUTOGRAD_THREAD_NVTX = re.compile(
    r"^Tide:S[12]:MB\d+:(?:(?:Backward|IntermediateGrad|FinalGrad|AccumulateGrad):[^|]+"
    r"|AccumulateGrad|C2:[^|]*(?:Backward|AccumulateGrad)[^|]*)(?:\|.*)?$")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(path):
    return {"path": str(path.resolve()), "name": path.name,
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def covered(intervals, begin, finish):
    return sum(max(0, min(end, finish) - max(start, begin))
               for start, end in intervals if end > begin and start < finish)


def intersection(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            result.append([start, end])
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * fraction
    low, high = math.floor(pos), math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)


def columns(connection, table):
    return [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]


def require_columns(schema, table, needed):
    missing = set(needed) - set(schema.get(table, []))
    if missing:
        raise ValueError(f"{table}: missing required columns {sorted(missing)}")


def read_jsonl(path, rank):
    events = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if int(event["rank"]) != rank:
                raise ValueError(f"{path}:{number}: rank identity mismatch")
            if event.get("duration_ns") is not None:
                if int(event["end_ns"]) - int(event["start_ns"]) != int(event["duration_ns"]):
                    raise ValueError(f"{path}:{number}: inconsistent JSONL duration")
            events.append(event)
    pids = {int(event["pid"]) for event in events}
    if len(pids) != 1:
        raise ValueError(f"{path}: expected one worker PID, got {sorted(pids)}")
    return events, pids.pop()


def normalize_mode(mode):
    """Read historical PC1 captures while emitting the current mode name."""
    return "micro" if mode == "pc1" else mode


def recording_mode(config):
    if config is None:
        return "unknown"
    light = bool(config.get("tide_light_timeline", False))
    detailed = bool(config.get("tide_detailed_metrics", False))
    if light and detailed:
        raise ValueError("run args.json enables both light and detailed timeline modes")
    return "light" if light else "detailed" if detailed else "disabled"


def read_metrics(path, rank, mode, global_bsz=32):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != EXPECTED_STEPS:
        raise ValueError(f"{path}: expected {EXPECTED_STEPS} optimizer records, got {len(rows)}")
    for batch, row in enumerate(rows, 1):
        if int(row["rank"]) != rank or int(row["optimizer_step"]) != batch:
            raise ValueError(f"{path}: rank/optimizer clock mismatch at batch {batch}")
        if int(row["iteration"]) != 1 + (batch - 1) * global_bsz:
            raise ValueError(f"{path}: not a from-scratch global-bsz{global_bsz} run")
        if int(row["local_cameras"]) != 16 or normalize_mode(row.get("pipeline_mode")) != normalize_mode(mode):
            raise ValueError(f"{path}: local camera count or pipeline mode mismatch")
    return rows


def event_row(start, end, lane, category, name, *, tid=None, device=None,
              stream=None, size=None, correlation=None, rowid=None):
    return {"start_ns": int(start), "end_ns": int(end), "lane": lane,
            "category": category, "name": name, "global_tid": tid,
            "device_id": device, "stream_id": stream, "bytes": size,
            "correlation_id": correlation, "source_rowid": rowid}


def phase_name(label):
    if label in CPU_NAMES:
        return CPU_NAMES[label]
    if MICRO_PIPELINE_PHASE.match(label) or OPTIMIZER_HOST_PHASE.match(label):
        return label.removeprefix("Tide:") + " (host call)"
    if label.startswith("tide.gpu.forward.") or label.startswith("tide.gpu.backward"):
        return label.removeprefix("tide.gpu.").replace(".", " ") + " submission"
    return None


def host_gap_stage(label):
    """Classify specific main-thread host spans; generic parents stay unclassified."""
    if label in {SCHEDULE, "tide.cpu.plan.initial", "tide.cpu.plan.preview.total",
                 "tide.cpu.plan.preview", "tide.cpu.plan.finalize.total",
                 "tide.cpu.plan.finalize", "tide.cpu.plan.active_refresh", "tide.comm.plan.broadcast",
                 "tide.io.ssd.prefetch.submit", "tide.cpu.camera.prefetch.submit",
                 "tide.cpu.plan.consume", "tide.cpu.cache.hint"} or label.startswith("tide.cpu.order."):
        return "plan_prefetch"
    if label == "tide.cpu.camera.h2d.submit":
        return "camera_transfer"
    if label == "tide.wait.camera_prefetch":
        return "camera_fetch_wait"
    if label in {"tide.cpu.resident.evict.stage", "tide.cpu.resident.evict.submit",
                 "tide.cpu.resident.evict", "tide.cpu.resident.slot_map",
                 "tide.cpu.resident.select",
                 "tide.cpu.writeback.plan", "tide.cpu.writeback.orchestrate",
                 "tide.cpu.writeback.pack", "tide.cpu.writeback.commit",
                 "tide.gpu.writeback.d2h", "tide.gpu.writeback.ram"}:
        return "eviction_load"
    if label in {READ_BATCH, "tide.cpu.resident.cache_fetch", "tide.cpu.resident.pack",
                 "tide.cpu.resident.h2d.submit", "tide.cpu.resident.unpack.prepare",
                 "tide.cpu.resident.unpack.submit", "tide.gpu.materialize",
                 "tide.io.ssd.read.submit"}:
        return "cache_to_gpu"
    if label == READ:
        return "storage_read"
    if label in {"tide.cpu.resident.bind", "tide.cpu.resident.active_leaves",
                 "tide.cpu.micro.worker.prepare", "tide.cpu.micro.layout.local",
                 "tide.comm.micro.layout", "tide.comm.micro.layout.counts",
                 "tide.comm.micro.layout.cameras", "tide.comm.micro.layout_status",
                 "tide.comm.resident.status", "tide.cpu.materialize"}:
        return "compute_preparation"
    if label in {"tide.cpu.gradient.sparse_prepare", "tide.cpu.gradient.sparsity_sample",
                 "tide.cpu.gradient.sample_emit", "tide.cpu.writeback.mark_dirty",
                 "tide.gpu.optimizer", "tide.gpu.optimizer.update"} or OPTIMIZER_HOST_PHASE.match(label):
        return "update_tail"
    if label in {"tide.cpu.bounds.gpu_submit", "tide.cpu.bounds.submit",
                 "tide.wait.bounds.d2h", "tide.cpu.bounds.serialize",
                 "tide.comm.bounds.prepare_status", "tide.comm.bounds.gather",
                 "tide.cpu.bounds.apply", "tide.comm.bounds.apply_status"}:
        return "bounds"
    if label in {"tide.comm.batch.barrier", "tide.sync.barrier",
                 "tide.cpu.retention.prepare", "tide.comm.retention.status"}:
        return "rank_retention"
    if label.startswith("tide.cpu.metrics.") or label == "tide.comm.metrics.gather":
        return "metrics_record"
    if label in {"tide.cpu.loss_sync", "tide.comm.loss.gather", "tide.cpu.logging",
                 "tide.cpu.camera.cleanup", "tide.cpu.memory.cleanup"}:
        return "main_loop_return"
    if label.startswith("tide.wait."):
        return "other_wait"
    return None


def host_h2d_contexts(main_nvtx):
    """Index reused H2D scopes by their same-thread NVTX ancestors/children."""
    contexts, stack = {}, []
    ordered = sorted(main_nvtx, key=lambda row: (row["start_ns"], -row["end_ns"],
                                                row["source_rowid"]))
    resident_names = {"tide.cpu.resident.h2d.submit", "tide.cpu.resident.load"}
    for row in ordered:
        while stack and not (stack[-1]["start_ns"] <= row["start_ns"]
                             and row["end_ns"] <= stack[-1]["end_ns"]):
            stack.pop()
        ancestors = {parent["name"] for parent in stack}
        if row["name"] == "tide.gpu.transfer.h2d":
            contexts[row["source_rowid"]] = (
                "cache_to_gpu" if ancestors & resident_names else
                "camera_transfer" if "tide.cpu.camera.h2d.submit" in ancestors else None)
        elif row["name"] == "tide.cpu.camera.h2d.submit":
            for parent in stack:
                if parent["name"] != "tide.gpu.transfer.h2d":
                    continue
                key = parent["source_rowid"]
                if contexts[key] == "cache_to_gpu":
                    raise ValueError("transfer.h2d has both resident and camera contexts")
                contexts[key] = "camera_transfer"
        stack.append(row)
    return contexts


def exclusive_cpu_phases(events):
    """Subtract recognized nested children from parents to prevent overpainting.

    Original inclusive events remain in raw CSV and per-label summaries. These
    fragments represent NVTX call scopes, not measured CPU execution/utilization.
    """
    by_thread = defaultdict(list)
    for row in events:
        by_thread[row.get("global_tid")].append(row)
    if len(by_thread) > 1:
        return [fragment for rows in by_thread.values() for fragment in exclusive_cpu_phases(rows)]
    chosen = sorted((row for row in events if phase_name(row["name"])),
                    key=lambda row: (row["start_ns"], -row["end_ns"], row["source_rowid"]))
    children, stack = defaultdict(list), []
    for index, row in enumerate(chosen):
        while stack and not (chosen[stack[-1]]["start_ns"] <= row["start_ns"]
                             and chosen[stack[-1]]["end_ns"] >= row["end_ns"]):
            stack.pop()
        if stack:
            children[stack[-1]].append((row["start_ns"], row["end_ns"]))
        stack.append(index)
    result = []
    for index, row in enumerate(chosen):
        cursor = row["start_ns"]
        for start, end in union(children[index]):
            if start > cursor:
                result.append({**row, "start_ns": cursor, "end_ns": start,
                               "original_start_ns": row["start_ns"],
                               "original_end_ns": row["end_ns"]})
            cursor = max(cursor, end)
        if cursor < row["end_ns"]:
            result.append({**row, "start_ns": cursor,
                           "original_start_ns": row["start_ns"],
                           "original_end_ns": row["end_ns"]})
    return result


def select_ssd_services(nvtx, main_tid):
    """Discard foreground wrappers, keeping actual storage call leaves.

    Historical reports reuse READ for both the foreground read_batch wrapper
    and the nested storage call. New reports use READ_BATCH for the wrapper.
    A main-thread READ is certified only when contained by either wrapper;
    background READ leaves remain storage calls. Request identity stays unknown
    until separately paired.
    """
    by_thread = defaultdict(list)
    for row in nvtx:
        if row["name"] in {READ, READ_BATCH}:
            by_thread[row["global_tid"]].append(row)
    kept, rejected = [], []
    for tid, rows in by_thread.items():
        rows.sort(key=lambda row: (row["start_ns"], -row["end_ns"], row["source_rowid"]))
        parent, has_service_child, stack = {}, set(), []
        for index, row in enumerate(rows):
            while stack and not (rows[stack[-1]]["start_ns"] <= row["start_ns"]
                                 and rows[stack[-1]]["end_ns"] >= row["end_ns"]):
                stack.pop()
            if stack:
                parent[index] = stack[-1]
                if row["name"] == READ:
                    has_service_child.add(stack[-1])
            stack.append(index)
        for index, row in enumerate(rows):
            if row["name"] == READ_BATCH:
                rejected.append({**row, "reason": "explicit_read_batch_wrapper"})
            elif index in has_service_child:
                rejected.append({**row, "reason": "enclosing_read_service_wrapper"})
            elif tid == main_tid and index not in parent:
                rejected.append({**row, "reason": "unverified_main_thread_read_batch_wrapper"})
            else:
                kept.append({**row, "request_kind": "unknown", "target_iteration": None,
                             "thread_kind": "main_nested_storage" if tid == main_tid
                             else "background_storage"})
    return kept, rejected


def annotate_ssd_requests(reads, jsonl, main_tid, rank):
    """Pair worker storage calls to JSONL identities with proven clock bounds.

    In _read_current_storage_blocks the perf_counter start/end enclose the NVTX
    read.service range. Thus end_Nsight-end_host <= offset <= start_Nsight-start_host.
    Pair only when the SAME worker thread has equal counts and the bound
    intersection is nonempty. Request intervals are separate from services.
    """
    storage_by_thread, host_by_thread = defaultdict(list), defaultdict(list)
    for row in reads:
        row["read_kind"] = "urgent" if row["global_tid"] == main_tid else "unknown"
        row["request_kind"] = row["read_kind"]
        row["detail"] = ("Urgent nested storage call; request enqueue timestamp unavailable"
                         if row["read_kind"] == "urgent" else "Storage call; request identity not established")
        if row["global_tid"] != main_tid:
            storage_by_thread[row["global_tid"] & 0xFFFFFF].append(row)
    for event in jsonl:
        if event["name"] == "ssd_read_future" and event.get("timing_source") == "host_monotonic" and event.get("thread_id") is not None:
            host_by_thread[int(event["thread_id"])].append(event)
    paired, skipped = [], []
    for tid, services in storage_by_thread.items():
        hosts = sorted(host_by_thread.get(tid, []), key=lambda event: event["start_ns"])
        services = sorted(services, key=lambda row: row["start_ns"])
        if len(hosts) != len(services):
            skipped.append({"thread_id": tid, "nsight_services": len(services),
                            "jsonl_services": len(hosts), "reason": "unequal counts; identity not inferred"})
            continue
        paired.extend(zip(services, hosts))
    calibration = {"method": "JSONL perf_counter service scope encloses same-worker NVTX storage call",
                   "pair_count": len(paired), "skipped_threads": skipped,
                   "offset_ns": None, "offset_lower_ns": None, "offset_upper_ns": None}
    if paired:
        low = max(row["end_ns"] - int(event["end_ns"]) for row, event in paired)
        high = min(row["start_ns"] - int(event["start_ns"]) for row, event in paired)
        if low <= high:
            calibration.update(offset_ns=(low + high) // 2, offset_lower_ns=low, offset_upper_ns=high,
                               error_bound_ms=(high - low) / 2e6)
            for row, event in paired:
                row.update(read_kind="prefetch", request_kind="prefetch",
                           target_iteration=event.get("target_iteration"), bytes=event.get("bytes"),
                           blocks=event.get("blocks"), jsonl_start_ns=event["start_ns"],
                           jsonl_end_ns=event["end_ns"], detail="Prefetch storage call; paired by worker thread and enclosing clock bounds")
        else:
            calibration["failure"] = "No common clock offset; request identities/geometry are not inferred"
    requests = []
    for event in jsonl:
        if event["name"] != "prefetch_end_to_end" or event.get("timing_source") != "host_monotonic":
            continue
        request = {"rank": rank, "pid": event["pid"], "thread_id": event.get("thread_id"),
                   "request_id": [rank, event["start_ns"], event.get("target_iteration")],
                   "name": "Prefetch request to cache ready", "target_iteration": event.get("target_iteration"),
                   "status": event.get("status"), "blocks": event.get("blocks"),
                   "start_host_ns": event["start_ns"], "end_host_ns": event["end_ns"],
                   "request_latency_seconds": (event["end_ns"] - event["start_ns"]) / NS,
                   "start_nsys_ns": None, "end_nsys_ns": None, "paired_storage_calls": []}
        if calibration["offset_ns"] is not None:
            request["start_nsys_ns"] = event["start_ns"] + calibration["offset_ns"]
            request["end_nsys_ns"] = event["end_ns"] + calibration["offset_ns"]
        for row in reads:
            if row["read_kind"] == "prefetch" and row.get("target_iteration") == event.get("target_iteration") and (
                    row["global_tid"] & 0xFFFFFF) == event.get("thread_id") and (
                    event["start_ns"] <= row["jsonl_start_ns"] <= row["jsonl_end_ns"] <= event["end_ns"]):
                request["paired_storage_calls"].append({"source_rowid": row["source_rowid"],
                                                       "start_ns": row["start_ns"], "end_ns": row["end_ns"],
                                                       "bytes": row.get("bytes"), "blocks": row.get("blocks")})
        requests.append(request)
    return requests, calibration


def extract_rank(sqlite_path, report_path, metrics_path, jsonl_path, rank, mode, global_bsz=32):
    metrics = read_metrics(metrics_path, rank, mode, global_bsz)
    jsonl, pid = read_jsonl(jsonl_path, rank)
    connection = sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    checks = [row[0] for row in connection.execute("PRAGMA quick_check")]
    if checks != ["ok"]:
        raise ValueError(f"{sqlite_path}: quick_check failed: {checks}")
    tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    relevant = [name for name in tables if name.startswith(("CUPTI_", "META_DATA_", "DIAGNOSTIC", "ENUM_DIAGNOSTIC"))
                or name in {"PROCESSES", "NVTX_EVENTS", "StringIds", "OSRT_API"}]
    schema = {name: columns(connection, name) for name in relevant}
    require_columns(schema, "PROCESSES", ("globalPid", "pid"))
    processes = [dict(row) for row in connection.execute("SELECT * FROM PROCESSES")]
    identities = {int(row["globalPid"]) for row in processes if int(row["pid"]) == pid}
    if len(identities) != 1:
        raise ValueError(f"{sqlite_path}: JSONL PID {pid} does not resolve uniquely in PROCESSES")
    global_pid = identities.pop()
    require_columns(schema, "NVTX_EVENTS", ("start", "end", "globalTid"))
    require_columns(schema, "StringIds", ("id", "value"))
    strings = dict(connection.execute("SELECT id,value FROM StringIds"))
    nvtx = []
    # Nsight global IDs pack the process ID above the low 24 thread-ID bits.
    # Resolve globalPid from PROCESSES, then verify the schedule thread below.
    process_bits = global_pid >> 24
    for row in connection.execute(
            "SELECT rowid AS source_rowid,* FROM NVTX_EVENTS "
            "WHERE (globalTid >> 24)=? AND end>start ORDER BY start,end DESC", (process_bits,)):
        label = row["text"] if "text" in row.keys() else None
        if label is None and "textId" in row.keys():
            label = strings.get(row["textId"])
        if not label:
            continue
        nvtx.append(event_row(row["start"], row["end"], "CPU", "nvtx", label,
                              tid=int(row["globalTid"]), rowid=row["source_rowid"]))
    schedules = sorted((row for row in nvtx if row["name"] == SCHEDULE),
                       key=lambda row: row["start_ns"])
    if len(schedules) != EXPECTED_STEPS:
        raise ValueError(f"{sqlite_path}: expected {EXPECTED_STEPS} camera.schedule ranges, got {len(schedules)}")
    tids = {row["global_tid"] for row in schedules}
    if len(tids) != 1:
        raise ValueError(f"{sqlite_path}: schedule spans multiple threads")
    main_tid = tids.pop()
    if (main_tid & 0xFFFFFF) != pid:
        raise ValueError(f"{sqlite_path}: scheduling is not on verified process main thread")
    main_nvtx = [row for row in nvtx if row["global_tid"] == main_tid]
    steps = sorted((row for row in main_nvtx if row["name"] == STEP),
                   key=lambda row: row["start_ns"])
    if len(steps) != EXPECTED_STEPS:
        raise ValueError(f"{sqlite_path}: expected {EXPECTED_STEPS} batch.step ranges, got {len(steps)}")
    for batch, (schedule, step) in enumerate(zip(schedules, steps), 1):
        if not step["start_ns"] <= schedule["start_ns"] < schedule["end_ns"] <= step["end_ns"]:
            raise ValueError(f"{sqlite_path}: schedule/step containment fails at B{batch}")
    warnings, gpu, waits = [], [], []
    available = {}
    for table, category in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
                            ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
                            ("CUPTI_ACTIVITY_KIND_MEMCPY", "copy")):
        available[category] = table in schema
        if table not in schema:
            if category == "kernel":
                raise ValueError(f"{sqlite_path}: missing actual CUDA kernel capture")
            warnings.append(f"{table} absent: this activity class is unavailable, not certified zero")
            continue
        require_columns(schema, table, ("start", "end", "globalPid", "deviceId", "streamId"))
        for row in connection.execute(f'SELECT rowid AS source_rowid,* FROM "{table}" WHERE globalPid=? AND end>start',
                                      (global_pid,)):
            if category == "kernel":
                name = next((strings.get(row[key]) for key in ("shortName", "demangledName", "mangledName")
                             if key in row.keys() and strings.get(row[key])), "CUDA kernel (unnamed)")
                kind = "nccl" if "nccl" in name.lower() else "compute"
            elif category == "memset":
                name, kind = "CUDA memset", "compute"
            else:
                name, kind = "CUDA memcpy kind=" + str(row["copyKind"]), "copy"
            gpu.append(event_row(row["start"], row["end"], "GPU", kind, name,
                                 device=int(row["deviceId"]), stream=int(row["streamId"]),
                                 size=row["bytes"] if "bytes" in row.keys() else None,
                                 correlation=row["correlationId"] if "correlationId" in row.keys() else None,
                                 rowid=row["source_rowid"]))
    if not gpu:
        raise ValueError(f"{sqlite_path}: selected process has no CUPTI execution events")
    for table, kind, predicate in (("CUPTI_ACTIVITY_KIND_RUNTIME", "cuda_wait", RUNTIME_WAIT),
                                   ("OSRT_API", "os_wait", OSRT_WAIT)):
        if table not in schema:
            warnings.append(f"{table} absent: process-thread blocking API capture unavailable")
            continue
        require_columns(schema, table, ("start", "end", "globalTid", "nameId"))
        for row in connection.execute(f'SELECT rowid AS source_rowid,* FROM "{table}" WHERE (globalTid & -16777216)=? AND end-start>=?',
                                      (global_pid, MIN_WAIT_NS)):
            name = strings.get(row["nameId"], "")
            if predicate.search(name):
                waits.append(event_row(row["start"], row["end"], "CPU", kind, name,
                                       tid=int(row["globalTid"]), rowid=row["source_rowid"],
                                       correlation=row["correlationId"] if "correlationId" in row.keys() else None))
    diagnostics = []
    if "DIAGNOSTIC_EVENT" in schema:
        for row in connection.execute("SELECT * FROM DIAGNOSTIC_EVENT"):
            value = dict(row)
            if value.get("severity", 0) >= 2:
                diagnostic_pid = value.get("globalPid")
                value["selected_process"] = diagnostic_pid == global_pid
                diagnostics.append(value)
    else:
        warnings.append("DIAGNOSTIC_EVENT absent: capture diagnostics unavailable")
    capture_metadata = {}
    for table in relevant:
        if table.startswith("META_DATA_"):
            capture_metadata[table] = [dict(row) for row in connection.execute(f'SELECT * FROM "{table}" LIMIT 100')]
    connection.close()
    reads, rejected_reads = select_ssd_services(nvtx, main_tid)
    requests, calibration = annotate_ssd_requests(reads, jsonl, main_tid, rank)
    if calibration.get("failure"):
        warnings.append(calibration["failure"])
    if not reads:
        warnings.append("No certified storage read services recorded; missing I/O is not zero I/O")
    return {"rank": rank, "pid": pid, "global_pid": global_pid, "main_tid": main_tid,
            "metrics": metrics, "jsonl": jsonl, "schedules": schedules, "steps": steps,
            "gpu": gpu, "waits": waits, "nvtx": nvtx, "main_nvtx": main_nvtx,
            "cpu_fragments": exclusive_cpu_phases(nvtx), "reads": reads,
            "rejected_reads": rejected_reads, "available": available,
            "requests": requests, "ssd_clock_calibration": calibration,
            "provenance": {"sqlite": provenance(sqlite_path), "nsys_report": provenance(report_path),
                           "metrics": provenance(metrics_path), "jsonl": provenance(jsonl_path)},
            "schema": schema, "processes": processes, "diagnostics": diagnostics,
            "capture_metadata": capture_metadata, "warnings": warnings,
            "device_ids": sorted({row["device_id"] for row in gpu}), "quick_check": checks}


def batch_records(rank_data, mode):
    mode = normalize_mode(mode)
    gpu_unions = {kind: union((row["start_ns"], row["end_ns"]) for row in rank_data["gpu"]
                             if row["category"] == kind) for kind in ("compute", "nccl", "copy")}
    all_gpu = union(pair for intervals in gpu_unions.values() for pair in intervals)
    reads = rank_data["reads"]
    read_union = union((row["start_ns"], row["end_ns"]) for row in reads)
    wait_union = union((row["start_ns"], row["end_ns"]) for row in rank_data["waits"])
    compute_nccl = intersection(gpu_unions["compute"], gpu_unions["nccl"])
    read_gpu = intersection(read_union, all_gpu)
    nvtx_by_label = defaultdict(list)
    for row in rank_data["main_nvtx"]:
        nvtx_by_label[row["name"]].append((row["start_ns"], row["end_ns"]))
    cpu_keys = {"plan_preview_seconds": "tide.cpu.plan.preview",
                "plan_finalize_seconds": "tide.cpu.plan.finalize",
                "camera_wait_seconds": "tide.wait.camera_prefetch",
                "host_compute_scope_seconds": "tide.batch.compute",
                "host_h2d_scope_seconds": "tide.gpu.transfer.h2d",
                "object_collective_scope_seconds": "tide.comm.object_collective"}
    ssd_waits = [(row["start_ns"], row["end_ns"]) for row in rank_data["main_nvtx"]
                 if row["name"] == "tide.wait.inflight_read" and row["end_ns"] - row["start_ns"] >= MIN_WAIT_NS]
    ssd_wait_union = union(ssd_waits)
    records = []
    for index, (schedule, step, metric) in enumerate(zip(rank_data["schedules"], rank_data["steps"], rank_data["metrics"])):
        begin = schedule["start_ns"]
        complete = index + 1 < EXPECTED_STEPS
        finish = rank_data["schedules"][index + 1]["start_ns"] if complete else step["end_ns"]
        row = {"mode": mode, "rank": rank_data["rank"], "pid": rank_data["pid"],
               "batch": index + 1, "iteration": int(metric["iteration"]),
               "optimizer_step": index + 1, "start_ns": begin, "end_ns": finish,
               "cycle_seconds": (finish - begin) / NS if complete else None,
               "boundary_method": "adjacent_camera_schedule_starts" if complete
               else "final_batch_step_end_only_no_next_schedule",
               "retained": RETAINED[0] <= index + 1 <= RETAINED[1],
               "warm": WARM[0] <= index + 1 <= WARM[1],
               "batch_step_start_ns": step["start_ns"], "batch_step_end_ns": step["end_ns"],
               "nvtx_batch_step_seconds": (step["end_ns"] - step["start_ns"]) / NS,
               "schedule_after_step_start_ms": (begin - step["start_ns"]) / 1e6,
               "cpu_blocking_api_union_seconds": covered(wait_union, begin, finish) / NS,
               "cpu_blocking_api_intersecting_count": sum(event["end_ns"] > begin and event["start_ns"] < finish
                                                          for event in rank_data["waits"]),
               "ssd_inflight_wait_union_seconds": covered(ssd_wait_union, begin, finish) / NS,
               "ssd_inflight_wait_intersecting_count": sum(end > begin and start < finish for start, end in ssd_waits),
               "ssd_read_service_union_seconds": covered(read_union, begin, finish) / NS if reads else None,
               "ssd_read_gpu_overlap_seconds": covered(read_gpu, begin, finish) / NS if reads else None,
               "ssd_read_started_count": sum(begin <= event["start_ns"] < finish for event in reads),
               "ssd_main_nested_started_count": sum(begin <= event["start_ns"] < finish
                                                     and event["global_tid"] == rank_data["main_tid"] for event in reads),
               "ssd_background_started_count": sum(begin <= event["start_ns"] < finish
                                                    and event["global_tid"] != rank_data["main_tid"] for event in reads),
               "compute_nccl_overlap_seconds": covered(compute_nccl, begin, finish) / NS,
               "gpu_active_union_seconds": covered(all_gpu, begin, finish) / NS}
        for kind, intervals in gpu_unions.items():
            row[f"gpu_{kind}_union_seconds"] = covered(intervals, begin, finish) / NS if kind != "copy" or rank_data["available"]["copy"] else None
        for key, label in cpu_keys.items():
            row[key] = covered(union(nvtx_by_label[label]), begin, finish) / NS
        if complete:
            row["gpu_no_activity_seconds"] = row["cycle_seconds"] - row["gpu_active_union_seconds"]
        else:
            row["gpu_no_activity_seconds"] = None
        # Preserve every original numeric stage/workload field with an explicit
        # prefix; batch_total_ms remains an engine metric, never the denominator.
        for key, value in metric.items():
            if value is None or not str(value).strip():
                row["metrics_" + key] = None
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                row["metrics_" + key] = value
            else:
                row["metrics_" + key] = numeric if math.isfinite(numeric) else None
        records.append(row)
    return records


def aggregate(records, first, last, rank_data, global_bsz=32):
    chosen = [row for row in records if first <= row["batch"] <= last]
    durations = [row["cycle_seconds"] for row in chosen]
    if any(value is None for value in durations):
        raise ValueError("Aggregate includes a batch without a next-schedule boundary")
    total = sum(durations)
    skip = {"rank", "pid", "batch", "iteration", "optimizer_step", "start_ns", "end_ns",
            "batch_step_start_ns", "batch_step_end_ns", "retained", "warm"}
    means = {}
    for key in chosen[0]:
        if key in skip:
            continue
        values = [row[key] for row in chosen]
        if all(isinstance(value, (float, int)) and not isinstance(value, bool) for value in values):
            means[key] = statistics.mean(values)
    begin, finish = chosen[0]["start_ns"], chosen[-1]["end_ns"]
    service_calls = [row for row in rank_data["reads"] if begin <= row["start_ns"] < finish]
    service_durations = [(row["end_ns"] - row["start_ns"]) / NS for row in service_calls]
    requests = [row for row in rank_data["requests"] if row["target_iteration"] is not None
                and chosen[0]["iteration"] <= row["target_iteration"] <= chosen[-1]["iteration"]]
    latencies = [row["request_latency_seconds"] for row in requests]
    return {"batches": [first, last], "batch_count": len(chosen), "total_cycle_seconds": total,
            "mean_cycle_seconds": statistics.mean(durations), "median_cycle_seconds": statistics.median(durations),
            "p90_cycle_seconds": quantile(durations, .9), "p95_cycle_seconds": quantile(durations, .95),
            "global_camera_observations": len(chosen) * global_bsz,
            "throughput_cameras_per_second": len(chosen) * global_bsz / total,
            "storage_service_calls": {
                "attribution": "NVTX call start lies within retained schedule period; duration is full call, not clipped",
                "count": len(service_calls), "kind_counts": dict(Counter(row["read_kind"] for row in service_calls)),
                "mean_call_seconds": statistics.mean(service_durations) if service_durations else None,
                "median_call_seconds": statistics.median(service_durations) if service_durations else None,
                "p95_call_seconds": quantile(service_durations, .95),
                "call_duration_sum_seconds": sum(service_durations) if service_durations else None},
            "prefetch_requests": {
                "attribution": "JSONL target_iteration belongs to retained batches; not service occurrence batch",
                "count": len(requests), "status_counts": dict(Counter(str(row["status"]) for row in requests)),
                "mean_request_latency_seconds": statistics.mean(latencies) if latencies else None,
                "median_request_latency_seconds": statistics.median(latencies) if latencies else None,
                "p95_request_latency_seconds": quantile(latencies, .95)},
            "batches_with_cpu_blocking_api": sum(row["cpu_blocking_api_intersecting_count"] > 0 for row in chosen),
            "batches_with_ssd_inflight_wait": sum(row["ssd_inflight_wait_intersecting_count"] > 0 for row in chosen),
            "batches_with_recorded_storage_service": sum(row["ssd_read_started_count"] > 0 for row in chosen),
            "mean_fields": means}


def host_gap_window(rank_data, batch, mode, kind, start_event, start_edge,
                    end_event, end_edge, gpu_union, h2d_contexts=None):
    """Account for one host-defined interval using exclusive main-thread NVTX.

    GPU coverage is independently sourced from CUPTI; host labels do not imply
    that the GPU was idle or busy throughout the interval.
    """
    result = {"rank": rank_data["rank"], "mode": mode, "batch": batch,
              "gap": kind, "status": "complete"}
    missing = [side for side, event in (("start", start_event), ("end", end_event))
               if event is None]
    if missing:
        return {**result, "status": "missing_boundary", "missing": missing}
    start, end = start_event[start_edge + "_ns"], end_event[end_edge + "_ns"]
    if end <= start:
        raise ValueError(f"rank {rank_data['rank']} B{batch} {kind}: reversed host boundaries")
    if h2d_contexts is None:
        h2d_contexts = host_h2d_contexts(rank_data["main_nvtx"])
    stage_intervals, label_intervals, unknown_intervals = defaultdict(list), defaultdict(list), []
    rejected_read_ids = {row["source_rowid"] for row in rank_data.get("rejected_reads", [])
                         if row["name"] == READ}
    for fragment in rank_data["cpu_fragments"]:
        if fragment["global_tid"] != rank_data["main_tid"]:
            continue
        left, right = max(start, fragment["start_ns"]), min(end, fragment["end_ns"])
        if right <= left:
            continue
        label = fragment["name"]
        if label == READ and fragment["source_rowid"] in rejected_read_ids:
            stage = "cache_to_gpu"
        elif label == "tide.gpu.transfer.h2d":
            stage = h2d_contexts.get(fragment["source_rowid"])
        else:
            stage = host_gap_stage(label)
        if stage is None:
            unknown_intervals.append((left, right))
        else:
            stage_intervals[stage].append((left, right))
            label_intervals[label].append((left, right))
    classified = union(pair for intervals in stage_intervals.values() for pair in intervals)
    classified_ns = covered(classified, start, end)
    stage_ns = {stage: covered(union(stage_intervals[stage]), start, end)
                for stage in HOST_GAP_STAGES}
    if sum(stage_ns.values()) != classified_ns:
        raise ValueError(f"rank {rank_data['rank']} B{batch} {kind}: overlapping exclusive CPU stages")
    unknown_ns = covered(union(unknown_intervals), start, end)
    duration_ns = end - start
    if classified_ns + unknown_ns > duration_ns:
        raise ValueError(f"rank {rank_data['rank']} B{batch} {kind}: CPU fragments overlap")
    return {**result,
            "start_ns": start, "end_ns": end, "duration_seconds": duration_ns / NS,
            "start_boundary": {"label": start_event["name"], "edge": start_edge,
                               "source_rowid": start_event["source_rowid"]},
            "end_boundary": {"label": end_event["name"], "edge": end_edge,
                             "source_rowid": end_event["source_rowid"]},
            "stage_exclusive_seconds": {stage: value / NS for stage, value in stage_ns.items()},
            "label_exclusive_seconds": {label: covered(union(intervals), start, end) / NS
                                        for label, intervals in sorted(label_intervals.items())},
            "classified_seconds": classified_ns / NS,
            "unclassified_seconds": (duration_ns - classified_ns) / NS,
            "unclassified_labeled_seconds": unknown_ns / NS,
            "untagged_seconds": (duration_ns - classified_ns - unknown_ns) / NS,
            "gpu_active_union_seconds": covered(gpu_union, start, end) / NS}


def host_gap_records(rank_data, mode):
    """Pair per-batch main-thread anchors without crossing adjacent schedules."""
    mode = normalize_mode(mode)
    schedules, steps = rank_data["schedules"], rank_data["steps"]
    main = rank_data["main_nvtx"]
    def is_projection(row):
        return (row["name"] == "tide.cpu.projection.submit" if mode == "off"
                else bool(PROJECTION_HOST_PHASE.match(row["name"])))
    projection_seen = any(is_projection(row) for row in main)
    sparse_seen = any(row["name"] == "tide.cpu.gradient.sparse_prepare" for row in main)
    finalize_seen = any(row["name"] == "tide.cpu.plan.finalize.total" for row in main)
    anchors = []
    for index, (schedule, step) in enumerate(zip(schedules, steps)):
        body = [row for row in main
                if step["start_ns"] <= row["start_ns"] < row["end_ns"] <= step["end_ns"]]
        projections = sorted((row for row in body if is_projection(row)),
                             key=lambda row: row["start_ns"])
        sparse = [row for row in body if row["name"] == "tide.cpu.gradient.sparse_prepare"]
        finalize = [row for row in body if row["name"] == "tide.cpu.plan.finalize.total"]
        batch = index + 1
        if projection_seen and not projections:
            raise ValueError(f"rank {rank_data['rank']} B{batch}: missing first Projection host scope")
        if mode == "micro" and len(projections) > 1:
            raise ValueError(f"rank {rank_data['rank']} B{batch}: multiple MB0 Projection scopes")
        if sparse_seen and len(sparse) != 1:
            raise ValueError(f"rank {rank_data['rank']} B{batch}: expected one sparse_prepare scope")
        if finalize_seen and index + 1 < len(schedules) and len(finalize) != 1:
            raise ValueError(f"rank {rank_data['rank']} B{batch}: expected one next-batch finalize scope")
        projection = projections[0] if projections else None
        gradient = sparse[0] if sparse else None
        next_finalize = finalize[0] if finalize else None
        if projection and not schedule["start_ns"] <= projection["start_ns"] < step["end_ns"]:
            raise ValueError(f"rank {rank_data['rank']} B{batch}: Projection outside scheduled batch")
        if projection and gradient and not projection["start_ns"] < gradient["start_ns"]:
            raise ValueError(f"rank {rank_data['rank']} B{batch}: sparse_prepare precedes Projection")
        if gradient and next_finalize and not gradient["start_ns"] < next_finalize["start_ns"]:
            raise ValueError(f"rank {rank_data['rank']} B{batch}: finalize precedes sparse_prepare")
        if next_finalize and index + 1 < len(schedules) and not next_finalize["end_ns"] <= schedules[index + 1]["start_ns"]:
            raise ValueError(f"rank {rank_data['rank']} B{batch}: finalize crosses next schedule")
        anchors.append({"schedule": schedule, "projection": projection,
                        "gradient": gradient, "finalize": next_finalize})
    gpu_union = union((row["start_ns"], row["end_ns"]) for row in rank_data["gpu"])
    h2d_contexts = host_h2d_contexts(main)
    records = []
    for index, current in enumerate(anchors):
        batch = index + 1
        previous = anchors[index - 1] if index else None
        following = anchors[index + 1] if index + 1 < len(anchors) else None
        specifications = (
            ("pre_projection", current["schedule"], "start", current["projection"], "start"),
            ("post_backward_host_tail", current["gradient"], "start",
             current["finalize"] if following else None, "start"),
            ("prior_finalize_to_schedule", previous["finalize"] if previous else None,
             "end", current["schedule"], "start"),
            ("finalize_to_next_projection", current["finalize"] if following else None,
             "start", following["projection"] if following else None, "start"),
        )
        for kind, start_event, start_edge, end_event, end_edge in specifications:
            records.append(host_gap_window(rank_data, batch, mode, kind, start_event,
                                           start_edge, end_event, end_edge, gpu_union,
                                           h2d_contexts))
    return records


def host_gap_aggregate(records, first, last):
    chosen = [row for row in records if first <= row["batch"] <= last]
    summary = {}
    for kind in ("pre_projection", "post_backward_host_tail",
                 "prior_finalize_to_schedule", "finalize_to_next_projection"):
        matching = [row for row in chosen if row["gap"] == kind]
        complete = [row for row in matching if row["status"] == "complete"]
        def stats(key):
            values = [row[key] for row in complete]
            return {"mean_seconds": statistics.mean(values) if values else None,
                    "p90_seconds": quantile(values, .9)}
        summary[kind] = {"expected_count": len(matching), "available_count": len(complete),
                         "missing_batches": [row["batch"] for row in matching
                                             if row["status"] != "complete"],
                         "duration": stats("duration_seconds"),
                         "unclassified": stats("unclassified_seconds"),
                         "gpu_active_union": stats("gpu_active_union_seconds"),
                         "stage_exclusive": {
                             stage: {"mean_seconds": statistics.mean(
                                         row["stage_exclusive_seconds"][stage] for row in complete)
                                         if complete else None,
                                     "p90_seconds": quantile(
                                         [row["stage_exclusive_seconds"][stage] for row in complete], .9)}
                             for stage in HOST_GAP_STAGES}}
    return summary


def gpu_bins(events, begin, finish):
    result = []
    for kind in ("compute", "nccl", "copy"):
        intervals = union((max(begin, row["start_ns"]), min(finish, row["end_ns"]))
                          for row in events if row["category"] == kind)
        occupancy = defaultdict(int)
        for start, end in intervals:
            if end <= start:
                continue
            for index in range((start - begin) // BIN_NS, (end - begin - 1) // BIN_NS + 1):
                occupancy[index] += max(0, min(end, begin + (index + 1) * BIN_NS, finish)
                                        - max(start, begin + index * BIN_NS))
        for index, active in sorted(occupancy.items()):
            bin_start, bin_end = begin + index * BIN_NS, min(begin + (index + 1) * BIN_NS, finish)
            width = bin_end - bin_start
            if not 0 < active <= width:
                raise ValueError("GPU-bin union invariant failed")
            result.append({"kind": kind, "start": (bin_start - begin) / NS,
                           "end": (bin_end - begin) / NS, "fraction": active / width})
    return result


def build_timeline(rank_data, records, mode, first, count, world_size=2, global_bsz=32):
    mode = normalize_mode(mode)
    selected = [row for row in records if first <= row["batch"] < first + count]
    if len(selected) != count or any(row["cycle_seconds"] is None for row in selected):
        raise ValueError("Timeline window must contain consecutive complete retained batch cycles")
    begin, finish = selected[0]["start_ns"], selected[-1]["end_ns"]
    def within(row):
        return row["end_ns"] > begin and row["start_ns"] < finish
    def relative(row):
        return {"start": (row["start_ns"] - begin) / NS,
                "end": (row["end_ns"] - begin) / NS}
    phases = []
    comm_tids = {row["global_tid"] for row in rank_data["nvtx"]
                 if COMM_THREAD_NVTX.fullmatch(row["name"])}
    autograd_tids = {row["global_tid"] for row in rank_data["nvtx"]
                     if AUTOGRAD_THREAD_NVTX.fullmatch(row["name"])}
    def thread_role(row):
        tid = row["global_tid"]
        if tid == rank_data["main_tid"]:
            return "main"
        if tid in comm_tids:
            return "comm"
        if tid in autograd_tids:
            return "autograd"
        return "unknown"
    certified_main_storage = {row["source_rowid"] for row in rank_data["reads"]
                              if row["global_tid"] == rank_data["main_tid"]}
    legacy_read_wrappers = {row["source_rowid"] for row in rank_data.get("rejected_reads", [])
                            if row["name"] == READ and row["reason"] in {
                                "enclosing_read_service_wrapper",
                                "unverified_main_thread_read_batch_wrapper"}}
    for row in rank_data["cpu_fragments"]:
        if within(row):
            if row["source_rowid"] in certified_main_storage:
                name = "Urgent storage read (host call)"
            elif row["source_rowid"] in legacy_read_wrappers:
                name = "Gaussian cache fetch / pack (legacy wrapper)"
            else:
                name = phase_name(row["name"])
            phases.append({**relative(row), "name": name, "source": "Nsight NVTX",
                           "detail": row["name"] + "; exclusive display fragment of inclusive host call",
                           "original_start_ns": row["original_start_ns"],
                           "original_end_ns": row["original_end_ns"],
                           "global_tid": row["global_tid"], "thread_id": row["global_tid"] & 0xFFFFFF,
                           "thread_role": thread_role(row), "source_rowid": row["source_rowid"]})
    waits = [{**relative(row), "name": row["name"], "kind": row["category"],
              "source": "Nsight CUDA runtime" if row["category"] == "cuda_wait" else "Nsight OSRT",
              "global_tid": row["global_tid"], "thread_id": row["global_tid"] & 0xFFFFFF,
              "thread_role": thread_role(row), "source_rowid": row["source_rowid"]}
             for row in rank_data["waits"] if within(row)]
    reads = [{**relative(row), "source": "Nsight NVTX nested storage read.service",
              "thread_id": row["global_tid"] & 0xFFFFFF, "global_tid": row["global_tid"],
              "thread_kind": row["thread_kind"], "request_kind": row["request_kind"],
              "read_kind": row["read_kind"], "detail": row["detail"],
              "target_iteration": row.get("target_iteration"), "bytes": row.get("bytes"), "blocks": row.get("blocks"),
              "source_rowid": row["source_rowid"]}
             for row in rank_data["reads"] if within(row)]
    metadata = {"world_size": world_size, "global_bsz": global_bsz, "mode": mode,
                "recording_mode": rank_data.get("recording_mode", "unknown"),
                "rank": rank_data["rank"], "pid": rank_data["pid"],
                "global_pid": rank_data["global_pid"], "main_global_tid": rank_data["main_tid"],
                "device_ids": rank_data["device_ids"], "window_start_ns": begin,
                "window_end_ns": finish, "duration_seconds": (finish - begin) / NS,
                "gpu_bin_ms": 20, "window_kind": "steady", "batch_window": [first, first + count - 1],
                "selected_rank_method": f"longest sum of B{RETAINED[0]}-B{RETAINED[1]} adjacent camera.schedule periods",
                "boundary_method": "same main thread adjacent tide.cpu.camera.schedule starts",
                "jsonl_clock_alignment": "CPU/GPU/storage geometry comes directly from Nsight; JSONL services independently bound offset for request identity",
                "ssd_clock_calibration": rank_data["ssd_clock_calibration"],
                "raw_sources": rank_data["provenance"], "capture_diagnostics": rank_data["diagnostics"],
                "gpu_capture_warning": " ".join(rank_data["warnings"]),
                "notes": [
                    "Actual CUPTI non-NCCL kernels and memset form compute; NCCL can include peer waits.",
                    "Per-category unions are independently binned at 20 ms; categories may overlap.",
                    "CPU phases are exclusive display fragments of NVTX call scopes, not CPU utilization.",
                    "CPU thread roles require verified main identity or explicit communication/backward-node NVTX evidence; other workers remain unknown.",
                    "SSD main-thread outer read_batch wrappers are excluded; only certified nested storage leaves remain.",
                    "Background read services are included only from this rank's process.",
                    "Request kind, bytes and target are populated only for proven same-worker JSONL/NVTX service pairs.",
                    "Read service is not physical SSD busy time or latency from prefetch submit.",
                    "SSD writes and writeback are excluded from the SSD read track.",
                    "Selected three batches illustrate timing; summary statistics use all retained batches."]}
    return {"metadata": metadata,
            "batches": [{"number": row["batch"], "iteration": row["iteration"],
                         "start": (row["start_ns"] - begin) / NS,
                         "end": (row["end_ns"] - begin) / NS,
                         "duration": row["cycle_seconds"]} for row in selected],
            "gpu_bins": gpu_bins(rank_data["gpu"], begin, finish),
            "cpu_phases": phases, "cpu_waits": waits, "ssd_reads": reads}


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_windows(ranks, all_records, mode, starts, count, selected_ranks,
                   output_dir, world_size, global_bsz, longest_rank):
    """Reuse verified events; keep each rank/report on its own local clock."""
    entries = []
    for rank in selected_ranks:
        data = ranks[rank]
        records = [row for row in all_records if row["rank"] == rank]
        for first in starts:
            last = first + count - 1
            folder = output_dir / "windows" / f"rank-{rank}" / f"B{first:02d}-{last:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            timeline = build_timeline(data, records, mode, first, count,
                                      world_size, global_bsz)
            meta = timeline["metadata"]
            meta["selected_rank_method"] = "explicit fixed rank requested for multi-rank inspection"
            meta["longest_summary_rank"] = longest_rank
            meta["cross_rank_clock_alignment"] = "not established; each chart zero is its own first schedule start"
            meta["notes"][-1] = "Selected consecutive batches illustrate timing; all retained batches determine summary statistics."
            begin, finish = meta["window_start_ns"], meta["window_end_ns"]
            originals = sorted(
                [row for collection in (data["gpu"], data["waits"], data["nvtx"])
                 for row in collection if row["end_ns"] > begin and row["start_ns"] < finish],
                key=lambda row: (row["start_ns"], row["lane"], row["source_rowid"]))
            source = folder / f"{mode}.timeline.data.json"
            source.write_text(json.dumps(timeline, ensure_ascii=False, indent=2,
                                         allow_nan=False) + "\n", encoding="utf-8")
            raw = folder / f"{mode}.window_events.csv"
            write_csv(raw, originals)
            entries.append({"rank": rank, "pid": data["pid"], "mode": mode,
                            "batch_start": first, "batch_end": last,
                            "timeline": provenance(source), "raw_events": provenance(raw),
                            "batch_durations_seconds": [row["duration"] for row in timeline["batches"]]})
    return entries


def main():
    global EXPECTED_STEPS, RETAINED, WARM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sqlite", nargs="+", type=Path, required=True, metavar="RANK_SQLITE")
    parser.add_argument("--reports", nargs="+", type=Path, metavar="RANK_REPORT")
    parser.add_argument("--metrics", nargs="+", type=Path, metavar="RANK_METRICS")
    parser.add_argument("--jsonl", nargs="+", type=Path, metavar="RANK_JSONL")
    parser.add_argument("--mode", choices=("micro", "pc1", "off"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--global-bsz", type=int, help="Default: world-size * 16 cameras")
    parser.add_argument("--expected-steps", type=int, default=94)
    parser.add_argument("--batch-start", type=int, default=45)
    parser.add_argument("--batch-count", type=int, default=3)
    parser.add_argument("--timeline-ranks", nargs="+", type=int,
                        help="Also export these fixed ranks into windows/ (default with --batch-starts: all ranks)")
    parser.add_argument("--batch-starts", nargs="+", type=int,
                        help="Also export consecutive windows beginning at these batch numbers")
    args = parser.parse_args()
    args.mode = normalize_mode(args.mode)
    if args.global_bsz is None:
        args.global_bsz = args.world_size * 16
    if args.world_size not in (2, 4) or args.global_bsz != args.world_size * 16:
        parser.error("this comparison supports 2 or 4 ranks and 16 cameras per rank")
    if args.expected_steps < 13:
        parser.error("--expected-steps must be at least 13 to retain the B11 warm summary")
    EXPECTED_STEPS = args.expected_steps
    RETAINED = (2, EXPECTED_STEPS - 1)
    WARM = (11, EXPECTED_STEPS - 1)
    retained_key = f"retained_B{RETAINED[0]}_B{RETAINED[1]}"
    warm_key = f"warm_B{WARM[0]}_B{WARM[1]}"
    for name in ("sqlite", "reports", "metrics", "jsonl"):
        value = getattr(args, name)
        if value is not None and len(value) != args.world_size:
            parser.error(f"--{name} needs {args.world_size} inputs ordered by rank")
    if not (RETAINED[0] <= args.batch_start and args.batch_start + args.batch_count - 1 <= RETAINED[1]) or not 2 <= args.batch_count <= 4:
        parser.error(f"choose 2-4 consecutive batches entirely within B{RETAINED[0]}-B{RETAINED[1]}")
    if args.timeline_ranks is not None and (len(set(args.timeline_ranks)) != len(args.timeline_ranks)
            or any(rank < 0 or rank >= args.world_size for rank in args.timeline_ranks)):
        parser.error("--timeline-ranks must contain distinct ranks within --world-size")
    if args.batch_starts is not None and (len(set(args.batch_starts)) != len(args.batch_starts)
            or any(first < RETAINED[0] or first + args.batch_count - 1 > RETAINED[1]
                   for first in args.batch_starts)):
        parser.error(f"--batch-starts must contain distinct complete windows within B{RETAINED[0]}-B{RETAINED[1]}")
    run_dir = args.run_dir.expanduser().resolve()
    reports = args.reports or [run_dir / "nsys" / f"rank{rank}.nsys-rep" for rank in range(args.world_size)]
    rank_dirs = [run_dir / "train" if rank == 0 else run_dir / "train" / f"rank_{rank}"
                 for rank in range(args.world_size)]
    metrics = args.metrics or [rank_dirs[rank] / f"metrics_batch_rank{rank}.tsv" for rank in range(args.world_size)]
    jsonl = args.jsonl or [rank_dirs[rank] / f"timeline_events_rank{rank}.jsonl" for rank in range(args.world_size)]
    inputs = [*args.sqlite, *reports, *metrics, *jsonl]
    for path in inputs:
        if not path.is_file():
            parser.error(f"missing required original input: {path}")
    output_dir = args.output_dir.expanduser().resolve()
    if any(path.resolve().parent == output_dir for path in inputs):
        parser.error("use a separate extraction output directory to preserve original inputs")
    config_path = run_dir / "train/args.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
    capture_mode = recording_mode(config)
    if config is not None:
        if normalize_mode(config.get("tide_forward_pipeline")) != args.mode or int(config.get("bsz", -1)) != args.global_bsz:
            parser.error("run args.json disagrees with requested mode/global-bsz")
        requested = int(config.get("iterations", -1))
        if requested <= 0 or (requested + args.global_bsz - 1) // args.global_bsz != EXPECTED_STEPS:
            parser.error(f"run args.json camera iterations do not imply {EXPECTED_STEPS} optimizer steps")
    ranks, all_records, all_host_gaps, aggregates = [], [], [], {}
    for rank in range(args.world_size):
        print(f"Reading rank {rank}: {args.sqlite[rank]}", file=sys.stderr, flush=True)
        data = extract_rank(args.sqlite[rank], reports[rank], metrics[rank], jsonl[rank], rank, args.mode, args.global_bsz)
        data["recording_mode"] = capture_mode
        records = batch_records(data, args.mode)
        host_gaps = host_gap_records(data, args.mode)
        ranks.append(data)
        all_records.extend(records)
        all_host_gaps.extend(host_gaps)
        aggregates[str(rank)] = {retained_key: aggregate(records, *RETAINED, data, args.global_bsz),
                                 warm_key: aggregate(records, *WARM, data, args.global_bsz)}
        aggregates[str(rank)][retained_key]["host_gaps"] = host_gap_aggregate(host_gaps, *RETAINED)
        aggregates[str(rank)][warm_key]["host_gaps"] = host_gap_aggregate(host_gaps, *WARM)
    selected_rank = max(range(args.world_size), key=lambda rank: aggregates[str(rank)][retained_key]["total_cycle_seconds"])
    selected_data = ranks[selected_rank]
    selected_records = [row for row in all_records if row["rank"] == selected_rank]
    timeline = build_timeline(selected_data, selected_records, args.mode, args.batch_start, args.batch_count,
                              args.world_size, args.global_bsz)
    begin, finish = timeline["metadata"]["window_start_ns"], timeline["metadata"]["window_end_ns"]
    originals = sorted([row for collection in (selected_data["gpu"], selected_data["waits"], selected_data["nvtx"])
                        for row in collection if row["end_ns"] > begin and row["start_ns"] < finish],
                       key=lambda row: (row["start_ns"], row["lane"], row["source_rowid"]))
    ssd_window = [row for row in selected_data["reads"] if row["end_ns"] > begin and row["start_ns"] < finish]
    run_files = {name: provenance(run_dir / name) for name in
                 ("run_info.txt", "command.sh", "status.txt", "nsys_version.txt") if (run_dir / name).is_file()}
    summary = {"schema_version": 1, "mode": args.mode, "recording_mode": capture_mode,
               "world_size": args.world_size, "global_bsz": args.global_bsz,
               "expected_optimizer_steps": EXPECTED_STEPS, "record_count": len(all_records),
               "selected_rank": selected_rank, "selected_pid": selected_data["pid"],
               "selected_window": [args.batch_start, args.batch_start + args.batch_count - 1],
               "rank_aggregates": aggregates, "selected_rank_aggregates": aggregates[str(selected_rank)],
               "batch_cycle_definition": "same main thread adjacent tide.cpu.camera.schedule starts",
               "final_batch_limitation": f"B{EXPECTED_STEPS} has no next schedule; cycle_seconds=null; batch.step_end is retained only for audit",
               "metric_limitations": ["Original engine batch_total_ms is preserved with metrics_ prefix and is never the throughput denominator.",
                                      "Union coverage is not summed kernel time or mean service request latency.",
                                      "NCCL includes possible peer waits; do not add plan and NCCL as disjoint costs.",
                                      "CPU scopes are inclusive in statistics; timeline fragments prevent nested overpainting.",
                                      "Host gaps use main-thread NVTX boundaries, not GPU idle boundaries; CUPTI GPU activity is reported separately.",
                                      "The post-backward host tail starts at sparse_prepare, excluding any preceding backward-to-gradient gap.",
                                      "No physical SSD bytes, bandwidth, or submit-to-ready latency is inferred from read.service."],
               "host_gap_definitions": {
                   "pre_projection": "current camera.schedule start to first S1 Projection host submission start",
                   "post_backward_host_tail": "current sparse_prepare start to next-batch finalize.total start",
                   "prior_finalize_to_schedule": "prior finalize.total end to current camera.schedule start",
                   "finalize_to_next_projection": "current next-batch finalize.total start to next batch first S1 Projection host submission start",
                   "exclusive_accounting": "main-thread NVTX nested parents minus children; generic parent residual and untagged time stay unclassified; windows overlap and must not be added",
                   "gpu_activity": "CUPTI kernel/memset/NCCL/memcpy union clipped to each host window; no host NVTX range is inferred to be GPU execution"},
               "run_config": config, "run_files": run_files,
               "ranks": [{key: data[key] for key in ("rank", "pid", "global_pid", "main_tid", "device_ids",
                          "quick_check", "schema", "processes", "diagnostics", "capture_metadata", "provenance", "warnings", "ssd_clock_calibration")}
                         | {"nvtx_label_counts": dict(Counter(row["name"] for row in data["nvtx"])),
                            "storage_service_count": len(data["reads"]),
                            "excluded_read_wrapper_count": len(data["rejected_reads"]),
                            "jsonl_event_count": len(data["jsonl"]),
                            "ssd_service_kind_counts": dict(Counter(row["read_kind"] for row in data["reads"])),
                            "prefetch_request_count": len(data["requests"]),
                            "mean_prefetch_request_latency_seconds": statistics.mean(
                                row["request_latency_seconds"] for row in data["requests"]) if data["requests"] else None}
                         for data in ranks],
               "extractor": provenance(Path(__file__))}
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.timeline_ranks is not None or args.batch_starts is not None:
        windows = export_windows(
            ranks, all_records, args.mode, args.batch_starts or [args.batch_start], args.batch_count,
            args.timeline_ranks if args.timeline_ranks is not None else list(range(args.world_size)),
            output_dir, args.world_size, args.global_bsz, selected_rank)
        summary["timeline_windows"] = windows
        (output_dir / "windows.json").write_text(
            json.dumps({"schema_version": 1, "mode": args.mode,
                        "recording_mode": capture_mode,
                        "world_size": args.world_size, "global_bsz": args.global_bsz,
                        "cross_rank_clock_alignment": "not established",
                        "windows": windows}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
    write_csv(output_dir / "batch_metrics.csv", all_records)
    gap_csv = []
    for gap in all_host_gaps:
        gap_csv.append({key: gap.get(key) for key in
                        ("mode", "rank", "batch", "gap", "status", "start_ns", "end_ns",
                         "duration_seconds", "classified_seconds", "unclassified_seconds",
                         "unclassified_labeled_seconds", "untagged_seconds",
                         "gpu_active_union_seconds")}
                       | {f"stage_{stage}_seconds": gap.get("stage_exclusive_seconds", {}).get(stage)
                          for stage in HOST_GAP_STAGES}
                       | {"start_source_rowid": gap.get("start_boundary", {}).get("source_rowid"),
                          "end_source_rowid": gap.get("end_boundary", {}).get("source_rowid"),
                          "missing": ",".join(gap.get("missing", []))})
    write_csv(output_dir / "host_gaps.csv", gap_csv)
    write_csv(output_dir / f"{args.mode}.window_events.csv", originals)
    write_csv(output_dir / f"{args.mode}.ssd_read_services.csv", selected_data["reads"])
    write_csv(output_dir / f"{args.mode}.excluded_read_wrappers.csv", selected_data["rejected_reads"])
    # Empty raw subsets still retain a header, rather than a misleading blank file.
    write_csv(output_dir / f"{args.mode}.window_ssd_services.csv", ssd_window,
              fields=list(dict.fromkeys(key for row in selected_data["reads"] for key in row))
              if selected_data["reads"] else
              ["start_ns", "end_ns", "global_tid", "name", "request_kind", "target_iteration"])
    for path, value in ((output_dir / "summary.json", summary),
                        (output_dir / f"{args.mode}.timeline.data.json", timeline),
                        (output_dir / "host_gaps.json", {
                            "definitions": summary["host_gap_definitions"],
                            "stage_names": HOST_GAP_STAGES, "records": all_host_gaps,
                            "extractor": summary["extractor"]}),
                        (output_dir / f"{args.mode}.ssd_requests.json", {
                            "definition": "enqueue to cache-ready request latency; independent of storage service call duration",
                            "note": "Request geometry requires proven per-report clock calibration. Do not sum request and service durations.",
                            "ranks": [{"rank": data["rank"], "clock_calibration": data["ssd_clock_calibration"],
                                       "requests": data["requests"]} for data in ranks]})):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    compact = {key: {field: value[field] for field in
                    ("batch_count", "total_cycle_seconds", "mean_cycle_seconds", "throughput_cameras_per_second")}
               for key, value in aggregates[str(selected_rank)].items()}
    print(json.dumps({"mode": args.mode, "selected_rank": selected_rank,
                      "aggregates": compact,
                      "timeline": str(output_dir / f"{args.mode}.timeline.data.json")}, indent=2))


if __name__ == "__main__":
    main()
