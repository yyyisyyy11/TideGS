"""Per-rank and global batch metrics for distributed TideGS."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List


BATCH_FIELDS = [
    "iteration",
    "rank",
    "local_cameras",
    "global_active_blocks",
    "global_resident_blocks",
    "rank_active_blocks",
    "rank_resident_blocks",
    "cold_blocks",
    "retained_blocks",
    "touched_gaussians",
    "block_cull_ms",
    "plan_ms",
    "writeback_submit_ms",
    "resident_load_ms",
    "ssd_foreground_wait_ms",
    "ssd_inflight_wait_ms",
    "cpu_materialize_ms",
    "ssd_urgent_read_blocks",
    "ssd_urgent_read_bytes",
    "ssd_future_read_blocks",
    "ssd_future_read_bytes",
    "prefetch_cpu_ms",
    "prefetch_ssd_ms",
    "h2d_ms",
    "gsplat_forward_ms",
    "gaussian_projection_cull_ms",
    "backward_ms",
    "optimizer_ms",
    "train_ms",
    "bounds_sync_ms",
    "barrier_ms",
    "gpu_d2h_ms",
    "cpu_cache_commit_ms",
    "ssd_write_blocks",
    "ssd_write_bytes",
    "ssd_write_service_ms",
    "batch_total_ms",
]

IO_FIELDS = [
    "operation",
    "tier",
    "rank",
    "origin_iteration",
    "target_iteration",
    "blocks",
    "bytes",
    "service_ms",
]

SUM_FIELDS = {
    "local_cameras",
    "rank_active_blocks",
    "rank_resident_blocks",
    "cold_blocks",
    "retained_blocks",
    "touched_gaussians",
    "ssd_urgent_read_blocks",
    "ssd_urgent_read_bytes",
    "ssd_future_read_blocks",
    "ssd_future_read_bytes",
    "ssd_write_blocks",
    "ssd_write_bytes",
}

GLOBAL_ONCE_FIELDS = {
    "global_active_blocks",
    "global_resident_blocks",
}

TIME_FIELDS = [field for field in BATCH_FIELDS if field.endswith("_ms")]
GLOBAL_FIELDS = (
    ["iteration", "world_size"]
    + sorted(SUM_FIELDS | GLOBAL_ONCE_FIELDS)
    + [f"{field}_max" for field in TIME_FIELDS]
    + [f"{field}_mean" for field in TIME_FIELDS]
)


def _append_tsv(path: Path, fields: Iterable[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields)
    needs_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, 0) for field in fields})


class DistributedMetricsWriter:
    def __init__(self, *, args, context):
        self.context = context
        self.enabled = bool(getattr(args, "tide_detailed_metrics", False))
        self.rank_path = Path(args.log_folder) / f"metrics_batch_rank{context.rank}.tsv"
        self.io_path = Path(args.log_folder) / f"metrics_io_rank{context.rank}.tsv"
        self.global_path = Path(args.log_folder) / "metrics_batch_global.tsv"
        self._reads_by_target: Dict[int, Dict[str, float]] = {}

    def write_io_events(self, events: Iterable[Dict[str, object]]) -> None:
        if not self.enabled:
            return
        for event in events:
            row = dict(event)
            row["rank"] = self.context.rank
            _append_tsv(self.io_path, IO_FIELDS, row)
            target = row.get("target_iteration")
            operation = str(row.get("operation", ""))
            if target is None or operation not in {
                "ssd_read_urgent",
                "ssd_read_future",
            }:
                continue
            target_values = self._reads_by_target.setdefault(int(target), {})
            prefix = (
                "ssd_urgent_read"
                if operation == "ssd_read_urgent"
                else "ssd_future_read"
            )
            target_values[f"{prefix}_blocks"] = (
                target_values.get(f"{prefix}_blocks", 0.0)
                + float(row.get("blocks", 0))
            )
            target_values[f"{prefix}_bytes"] = (
                target_values.get(f"{prefix}_bytes", 0.0)
                + float(row.get("bytes", 0))
            )

    def write_batch(self, row: Dict[str, object]) -> None:
        if not self.enabled:
            return
        rank_row = dict(row)
        target_reads = self._reads_by_target.pop(int(rank_row["iteration"]), {})
        for field in (
            "ssd_urgent_read_blocks",
            "ssd_urgent_read_bytes",
            "ssd_future_read_blocks",
            "ssd_future_read_bytes",
        ):
            rank_row[field] = target_reads.get(field, 0.0)
        rank_row["rank"] = self.context.rank
        _append_tsv(self.rank_path, BATCH_FIELDS, rank_row)
        rank_rows: List[Dict[str, object]] = self.context.all_gather_object(rank_row)
        if not self.context.is_rank0:
            return

        global_row: Dict[str, object] = {
            "iteration": int(rank_row["iteration"]),
            "world_size": int(self.context.world_size),
        }
        for field in SUM_FIELDS:
            global_row[field] = sum(float(value.get(field, 0)) for value in rank_rows)
        for field in GLOBAL_ONCE_FIELDS:
            values = [float(value.get(field, 0)) for value in rank_rows]
            global_row[field] = max(values, default=0.0)
        for field in TIME_FIELDS:
            values = [float(value.get(field, 0.0)) for value in rank_rows]
            global_row[f"{field}_max"] = max(values, default=0.0)
            global_row[f"{field}_mean"] = (
                sum(values) / len(values) if values else 0.0
            )
        _append_tsv(self.global_path, GLOBAL_FIELDS, global_row)


def get_distributed_metrics_writer(gaussians, context) -> DistributedMetricsWriter:
    writer = getattr(gaussians, "_tide_distributed_metrics_writer", None)
    if writer is None:
        writer = DistributedMetricsWriter(args=gaussians.args, context=context)
        gaussians._tide_distributed_metrics_writer = writer
    return writer
