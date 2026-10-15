#!/usr/bin/env python3
"""Audit PC1 C1 host synchronization and the next projection's submit order.

Consumes analyze_pc1_overlap.py outputs; optionally uses window_events.csv for
individual CUDA synchronization API evidence. No inference from CUDA-event
snapshot spans or from GPU activity occurring during a CPU NVTX range.

Example:
  python analyze_pc1_submission_order.py --derived-dir /path/to/derived/pc1 \
    --prefix pc1-overlap --output /path/to/derived/pc1/pc1-submission-order
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_pc1_overlap import duration_ms, intersection, union


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stats(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values), "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90_nearest_rank": ordered[max(0, (9 * len(ordered) + 9) // 10 - 1)],
        "min": ordered[0], "max": ordered[-1],
    }


def write_csv(path, rows):
    fields = list(rows[0]) if rows else ["rank", "trace_batch"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="pc1-overlap")
    parser.add_argument("--window-events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_prefix = args.derived_dir / args.prefix
    stages = read_csv(source_prefix.with_suffix(".stages.csv"))
    batches = read_csv(source_prefix.with_suffix(".batches.csv"))
    target_pairs = read_csv(source_prefix.with_suffix(".pairs.csv"))
    kernels = read_csv(source_prefix.with_suffix(".kernels.csv"))
    lookup = {(item["rank"], item["trace_batch"], item["microbatch"], item["phase"]): item for item in stages}
    payloads = [item for item in stages if item["phase"] == "C1:Payload"]
    orders, summaries = [], []
    by_batch = defaultdict(list)
    for item in kernels:
        by_batch[(item["rank"], item["trace_batch"])].append(item)
    overlap_rows = []
    for (rank, batch), items in sorted(by_batch.items(), key=lambda pair: (int(pair[0][0]), int(pair[0][1]))):
        groups = defaultdict(list)
        for item in items:
            if item["phase"] == "unknown" or not item.get("microbatch"):
                continue
            key = (item["phase"], int(item["microbatch"]), item["category"], int(item["stream_id"]), int(item["device_id"]))
            groups[key].append((int(item["kernel_start_ns"]), int(item["kernel_end_ns"])))
        overlap, phase_overlaps = [], defaultdict(list)
        for (p_phase, p_mb, p_category, p_stream, device), p_intervals in groups.items():
            if p_category != "compute" or p_phase == "unknown":
                continue
            for (c_phase, c_mb, c_category, c_stream, c_device), c_intervals in groups.items():
                if c_category != "nccl" or c_phase == "unknown" or device != c_device or p_stream == c_stream:
                    continue
                common = intersection(p_intervals, c_intervals)
                overlap.extend(common)
                phase_overlaps[(p_phase, p_mb, c_phase, c_mb)].extend(common)
        for (p_phase, p_mb, c_phase, c_mb), common in phase_overlaps.items():
            if common:
                overlap_rows.append({
                    "rank": int(rank), "trace_batch": int(batch),
                    "compute_phase": p_phase, "compute_microbatch": p_mb,
                    "nccl_phase": c_phase, "nccl_microbatch": c_mb,
                    "actual_cross_stream_overlap_ms": duration_ms(common),
                })
        by_batch[(rank, batch)] = duration_ms(overlap)
    for rank in sorted({item["rank"] for item in batches}, key=int):
        rank_payloads = [item for item in payloads if item["rank"] == rank]
        sync = [float(item["host_cuda_sync_api_ms"]) for item in rank_payloads]
        main_sync = [float(item["host_cuda_sync_api_ms"]) for item in rank_payloads
                     if item.get("thread_role", "main") == "main"]
        spans = [float(item["cpu_nvtx_span_ms"]) for item in rank_payloads]
        rank_orders = []
        for item in rank_payloads:
            mb = int(item["microbatch"])
            following = lookup.get((rank, item["trace_batch"], str(mb + 1), "Projection"))
            if following is None:
                continue
            gap = (int(following["nvtx_start_ns"]) - int(item["nvtx_end_ns"])) / 1e6
            row = {
                "rank": int(rank), "trace_batch": int(item["trace_batch"]),
                "optimizer_step": item["optimizer_step"], "iteration": item["iteration"],
                "c1_microbatch": mb, "next_projection_microbatch": mb + 1,
                "c1_payload_nvtx_rowid": item["nvtx_rowid"],
                "c1_payload_nvtx_start_ns": int(item["nvtx_start_ns"]),
                "c1_payload_nvtx_end_ns": int(item["nvtx_end_ns"]),
                "c1_payload_cpu_span_ms": float(item["cpu_nvtx_span_ms"]),
                "c1_payload_main_thread_sync_api_ms": float(item["host_cuda_sync_api_ms"]) if item.get("thread_role", "main") == "main" else 0.0,
                "c1_payload_submitter_sync_api_ms": float(item["host_cuda_sync_api_ms"]),
                "c1_payload_global_tid": item.get("global_tid"),
                "next_projection_nvtx_rowid": following["nvtx_rowid"],
                "next_projection_nvtx_start_ns": int(following["nvtx_start_ns"]),
                "next_projection_begins_after_payload_cpu_range": gap >= 0,
                "cpu_gap_after_payload_ms": gap,
            }
            orders.append(row)
            rank_orders.append(row)
        rank_batches = [item for item in batches if item["rank"] == rank]
        count = len(rank_batches)
        generic = sum(float(item["compute_nccl_cross_stream_overlap_ms"]) for item in rank_batches) / count
        fine = sum(by_batch.get((rank, item["trace_batch"]), 0.0) for item in rank_batches) / count
        target = sum(float(item["cross_stream_overlap_ms"]) for item in target_pairs if item["rank"] == rank and item["status"] == "assessable" and item.get("target_type", "projection_c1") == "projection_c1") / count
        summaries.append({
            "rank": int(rank), "analyzed_batches": count,
            "payload_microbatches": len(rank_payloads), "payloads_with_sync_api": sum(value > 0 for value in sync),
            "payload_cpu_span_ms_per_microbatch": stats(spans),
            "payload_main_thread_sync_api_ms_per_microbatch": stats(main_sync),
            "payload_submitter_sync_api_ms_per_microbatch": stats(sync),
            "sync_api_fraction_of_payload_cpu_span": sum(sync) / sum(spans) if sum(spans) else None,
            "payload_main_thread_sync_api_ms_per_batch": sum(main_sync) / count,
            "payload_submitter_sync_api_ms_per_batch": sum(sync) / count,
            "ordered_next_projection_pairs": len(rank_orders),
            "next_projection_starts_after_payload_cpu_end": sum(item["next_projection_begins_after_payload_cpu_range"] for item in rank_orders),
            "cpu_gap_after_payload_ms": stats([item["cpu_gap_after_payload_ms"] for item in rank_orders]),
            "all_kernel_compute_nccl_overlap_ms_per_batch": generic,
            "fine_tag_owned_kernel_overlap_ms_per_batch": fine,
            "target_next_projection_c1_overlap_ms_per_batch": target,
            "overlap_not_explained_by_fine_tag_owned_kernels_ms_per_batch": generic - fine,
            "overlap_attribution_status": "Remaining activity lacks complete fine-tag kernel ownership; do not assign it to backward/plan/communication subphases",
        })
    examples = []
    events_path = args.window_events or args.derived_dir / "pc1.window_events.csv"
    if events_path.exists():
        for event in read_csv(events_path):
            if event["lane"] != "CPU" or "Synchronize" not in event["name"]:
                continue
            start, end = int(event["start_ns"]), int(event["end_ns"])
            for payload in payloads:
                if int(payload["nvtx_start_ns"]) <= start and end <= int(payload["nvtx_end_ns"]):
                    # Absolute timestamps from different reports are not enough
                    # to identify a rank; require the audited payload thread ID.
                    payload_tid = payload.get("global_tid")
                    owner = next((item for item in kernels if item["rank"] == payload["rank"]
                                  and item["nvtx_rowid"] == payload["nvtx_rowid"]), None)
                    payload_tid = payload_tid or (owner["runtime_global_tid"] if owner else None)
                    if payload_tid is None or event.get("global_tid") != payload_tid:
                        continue
                    examples.append({
                        "rank": int(payload["rank"]), "trace_batch": int(payload["trace_batch"]),
                        "microbatch": int(payload["microbatch"]), "runtime_api": event["name"],
                        "runtime_global_tid": event["global_tid"], "runtime_rowid": event["source_rowid"],
                        "correlation_id": event["correlation_id"], "start_ns": start, "end_ns": end,
                        "duration_ms": (end - start) / 1e6,
                        "payload_nvtx_rowid": payload["nvtx_rowid"],
                    })
                    break
    output = {
        "sources": {name: str(source_prefix.with_suffix(f".{name}.csv")) for name in ("stages", "batches", "pairs", "kernels")},
        "source_window_events": str(events_path) if events_path.exists() else None,
        "method": "Use the verified payload-thread runtime spans; report observed relative CPU submission order without imposing a dependency; intersect correlated CUDA execution independently",
        "facts": [
            "Payload synchronization blocks its submitting thread; another CPU thread can continue submitting Projection",
            "The sign of next-Projection minus Payload-end is an observation, not a required ordering; legacy reports remain readable",
        ],
        "limits": [
            "Sync API duration includes GPU waits, CUDA runtime/driver service, and possible host descheduling; it is not pure network transfer time",
            "The gap to Projection NVTX also includes untagged transformed-leaf preparation and Python work; it is not just idle time",
            "This is an observed submission dependency; it does not measure the counterfactual speedup from removing synchronization",
            "Fine-tag-owned phase overlaps can overlap each other; generic minus their union is left unattributed",
            "Kernel-only overall overlap can differ slightly from timeline categories that include CUDA memset as compute",
        ],
        "ranks": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output.with_suffix(".pairs.csv"), orders)
    write_csv(args.output.with_suffix(".overlap-attribution.csv"), overlap_rows)
    write_csv(args.output.with_suffix(".sync-examples.csv"), examples)
    print(json.dumps({"output_prefix": str(args.output), "rank_summaries": summaries, "individual_sync_examples": len(examples)}, indent=2))


if __name__ == "__main__":
    main()
