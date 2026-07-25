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
    "gpu_slot_capacity_blocks",
    "gpu_slot_growth_blocks",
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
    "h2d_bytes",
    "h2d_ms",
    "gsplat_forward_ms",
    "gaussian_projection_cull_ms",
    "backward_ms",
    "optimizer_ms",
    "train_ms",
    "bounds_sync_ms",
    "barrier_ms",
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
    "gpu_slot_capacity_blocks",
    "gpu_slot_growth_blocks",
    "touched_gaussians",
    "ssd_urgent_read_blocks",
    "ssd_urgent_read_bytes",
    "h2d_bytes",
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

ASYNC_OPERATION_FIELDS = {
    "ssd_read_urgent": (
        "target_iteration",
        "ssd_urgent_read_blocks",
        "ssd_urgent_read_bytes",
        "ssd_urgent_read_service_ms",
    ),
    "ssd_read_future": (
        "target_iteration",
        "ssd_future_read_blocks",
        "ssd_future_read_bytes",
        "prefetch_ssd_ms",
    ),
    "cpu_materialize_future": (
        "target_iteration",
        "prefetch_cpu_blocks",
        "prefetch_cpu_bytes",
        "prefetch_cpu_ms",
    ),
    "gpu_d2h": (
        "origin_iteration",
        "gpu_d2h_blocks",
        "gpu_d2h_bytes",
        "gpu_d2h_ms",
    ),
    "cpu_cache_commit": (
        "origin_iteration",
        "cpu_cache_commit_blocks",
        "cpu_cache_commit_bytes",
        "cpu_cache_commit_ms",
    ),
    "ssd_write_async": (
        "origin_iteration",
        "ssd_write_blocks",
        "ssd_write_bytes",
        "ssd_write_service_ms",
    ),
    "ssd_write_sync": (
        "origin_iteration",
        "ssd_write_blocks",
        "ssd_write_bytes",
        "ssd_write_service_ms",
    ),
}
ASYNC_SUM_FIELDS = {
    values[index]
    for values in ASYNC_OPERATION_FIELDS.values()
    for index in (1, 2)
}
ASYNC_TIME_FIELDS = {
    values[3] for values in ASYNC_OPERATION_FIELDS.values()
}
ASYNC_FIELDS = (
    ["iteration", "rank"]
    + sorted(ASYNC_SUM_FIELDS)
    + sorted(ASYNC_TIME_FIELDS)
)
ASYNC_GLOBAL_FIELDS = (
    ["iteration", "world_size"]
    + sorted(ASYNC_SUM_FIELDS)
    + [f"{field}_max" for field in sorted(ASYNC_TIME_FIELDS)]
    + [f"{field}_mean" for field in sorted(ASYNC_TIME_FIELDS)]
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


def _write_tsv(
    path: Path,
    fields: Iterable[str],
    rows: Iterable[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields)
    temp_path = path.with_name(f"{path.name}.tmp")
    with open(temp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, 0) for field in fields})
    temp_path.replace(path)


class DistributedMetricsWriter:
    def __init__(self, *, args, context):
        self.context = context
        self.enabled = bool(getattr(args, "tide_detailed_metrics", False))
        self.rank_path = Path(args.log_folder) / f"metrics_batch_rank{context.rank}.tsv"
        self.io_path = Path(args.log_folder) / f"metrics_io_rank{context.rank}.tsv"
        self.global_path = Path(args.log_folder) / "metrics_batch_global.tsv"
        self.async_rank_path = (
            Path(args.log_folder)
            / f"metrics_async_iteration_rank{context.rank}.tsv"
        )
        self.async_global_path = (
            Path(args.log_folder) / "metrics_async_iteration_global.tsv"
        )
        self._async_by_iteration: Dict[int, Dict[str, float]] = {}
        self._async_finalized = False

    def write_io_events(self, events: Iterable[Dict[str, object]]) -> None:
        if not self.enabled:
            return
        if self._async_finalized:
            raise RuntimeError("Cannot record I/O events after metrics finalization")
        for event in events:
            row = dict(event)
            row["rank"] = self.context.rank
            _append_tsv(self.io_path, IO_FIELDS, row)
            operation = str(row.get("operation", ""))
            field_spec = ASYNC_OPERATION_FIELDS.get(operation)
            if field_spec is None:
                continue
            attribution_field, blocks_field, bytes_field, time_field = field_spec
            iteration = row.get(attribution_field)
            if iteration is None:
                raise ValueError(
                    f"{operation} is missing required {attribution_field}"
                )
            values = self._async_by_iteration.setdefault(int(iteration), {})
            values[blocks_field] = (
                values.get(blocks_field, 0.0) + float(row.get("blocks", 0))
            )
            values[bytes_field] = (
                values.get(bytes_field, 0.0) + float(row.get("bytes", 0))
            )
            values[time_field] = (
                values.get(time_field, 0.0) + float(row.get("service_ms", 0.0))
            )

    def write_batch(self, row: Dict[str, object]) -> None:
        if not self.enabled:
            return
        rank_row = dict(row)
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

    def finalize_async_metrics(self) -> None:
        if not self.enabled or self._async_finalized:
            return
        self._async_finalized = True

        rank_payload = {
            int(iteration): dict(values)
            for iteration, values in self._async_by_iteration.items()
        }
        rank_rows = []
        for iteration in sorted(rank_payload):
            rank_rows.append(
                {
                    "iteration": iteration,
                    "rank": self.context.rank,
                    **rank_payload[iteration],
                }
            )
        _write_tsv(self.async_rank_path, ASYNC_FIELDS, rank_rows)

        gathered = self.context.all_gather_object(rank_payload)
        if not self.context.is_rank0:
            return

        iterations = sorted(
            {
                int(iteration)
                for payload in gathered
                for iteration in payload
            }
        )
        global_rows = []
        for iteration in iterations:
            per_rank = [
                payload.get(iteration, payload.get(str(iteration), {}))
                for payload in gathered
            ]
            global_row: Dict[str, object] = {
                "iteration": iteration,
                "world_size": int(self.context.world_size),
            }
            for field in ASYNC_SUM_FIELDS:
                global_row[field] = sum(
                    float(values.get(field, 0.0)) for values in per_rank
                )
            for field in ASYNC_TIME_FIELDS:
                values = [
                    float(rank_values.get(field, 0.0))
                    for rank_values in per_rank
                ]
                global_row[f"{field}_max"] = max(values, default=0.0)
                global_row[f"{field}_mean"] = (
                    sum(values) / len(values) if values else 0.0
                )
            global_rows.append(global_row)
        _write_tsv(
            self.async_global_path,
            ASYNC_GLOBAL_FIELDS,
            global_rows,
        )


def get_distributed_metrics_writer(gaussians, context) -> DistributedMetricsWriter:
    writer = getattr(gaussians, "_tide_distributed_metrics_writer", None)
    if writer is None:
        writer = DistributedMetricsWriter(args=gaussians.args, context=context)
        gaussians._tide_distributed_metrics_writer = writer
    return writer
