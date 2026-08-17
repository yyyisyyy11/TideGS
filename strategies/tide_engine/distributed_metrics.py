"""Per-rank and global batch metrics for distributed TideGS."""

from __future__ import annotations

import csv
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List


BATCH_FIELDS = [
    "iteration",
    "rank",
    "optimizer_algorithm",
    "optimizer_step",
    "curvature_due",
    "local_cameras",
    "global_active_blocks",
    "global_resident_blocks",
    "predicted_stream_in_blocks",
    "prediction_missing_blocks",
    "prediction_extra_blocks",
    "prediction_replanned",
    "prediction_repair_ms",
    "prediction_exact_plan_ms",
    "rank_active_blocks",
    "rank_gradient_active_blocks",
    "rank_curvature_active_blocks",
    "rank_resident_blocks",
    "cold_blocks",
    "retained_blocks",
    "gpu_slot_capacity_blocks",
    "gpu_slot_growth_blocks",
    "owner_active_gaussians",
    "collective_participation_gaussians",
    "uses_zero_opacity_sentinel",
    "touched_gaussians",
    "gradient_participation_rows",
    "curvature_participation_rows",
    "updated_blocks",
    "optimizer_touched_rows",
    "optimizer_cold_rows",
    "curvature_blocks",
    "curvature_rows",
    "clipped_values",
    "rows_skipped_without_curvature",
    "trust_region_epsilon",
    "block_cull_ms",
    "block_cull_backend",
    "block_cull_gpu_kernel_ms",
    "block_cull_gpu_d2h_ms",
    "block_cull_cache_hit_cameras",
    "block_cull_gpu_cameras",
    "block_cull_output_blocks",
    "plan_ms",
    "writeback_submit_ms",
    "resident_load_ms",
    "block_reader_foreground_ms",
    "ssd_urgent_read_ms",
    "prefetch_inflight_wait_ms",
    "cpu_materialize_ms",
    "ssd_urgent_read_blocks",
    "ssd_urgent_read_bytes",
    "h2d_bytes",
    "h2d_ms",
    "gsplat_forward_ms",
    "curvature_forward_ms",
    "curvature_vjp_ms",
    "gaussian_projection_cull_ms",
    "backward_ms",
    "optimizer_ms",
    "train_ms",
    "bounds_sync_ms",
    "barrier_ms",
    "batch_total_ms",
    # Legacy single-rank pipeline details.  These are intentionally separate
    # from the common fields above: the N+1 planner is asynchronous and must
    # not be added to the foreground critical path.
    "legacy_stage4_wall_ms",
    "bounds_refresh_submit_ms",
    "prefetch_buffer_wait_ms",
    "next_plan_cull_ms",
    "next_plan_select_ms",
    "next_plan_hint_submit_ms",
    "next_plan_buffer_submit_ms",
    "next_plan_service_ms",
    "next_plan_finalize_wait_ms",
    "next_plan_target_iteration",
]

GRAD_ZERO_COMPONENTS = (
    "xyz",
    "opacity",
    "scaling",
    "rotation",
    "features_dc",
    "features_rest",
)
GRAD_ZERO_HISTOGRAM_FIELDS = [
    f"projection_cull_zero_gradient_elements_{count}_gaussians"
    for count in range(60)
]
GRAD_ZERO_VALUE_FIELDS = [
    "projection_cull_unique_gaussians",
    "projection_cull_parameter_elements",
    "projection_cull_zero_gradient_elements",
    "projection_cull_all_zero_gradient_gaussians",
] + [
    f"projection_cull_{component}_{suffix}"
    for component in GRAD_ZERO_COMPONENTS
    for suffix in ("parameter_elements", "zero_gradient_elements")
] + GRAD_ZERO_HISTOGRAM_FIELDS
GRAD_ZERO_FIELDS = ["iteration", "rank"] + GRAD_ZERO_VALUE_FIELDS
GRAD_ZERO_GLOBAL_FIELDS = ["iteration", "world_size"] + GRAD_ZERO_VALUE_FIELDS

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
    "rank_gradient_active_blocks",
    "rank_curvature_active_blocks",
    "rank_resident_blocks",
    "cold_blocks",
    "retained_blocks",
    "gpu_slot_capacity_blocks",
    "gpu_slot_growth_blocks",
    "owner_active_gaussians",
    "collective_participation_gaussians",
    "uses_zero_opacity_sentinel",
    "touched_gaussians",
    "gradient_participation_rows",
    "curvature_participation_rows",
    "updated_blocks",
    "optimizer_touched_rows",
    "optimizer_cold_rows",
    "curvature_blocks",
    "curvature_rows",
    "clipped_values",
    "rows_skipped_without_curvature",
    "ssd_urgent_read_blocks",
    "ssd_urgent_read_bytes",
    "h2d_bytes",
    "block_cull_cache_hit_cameras",
    "block_cull_gpu_cameras",
    "block_cull_output_blocks",
}

GLOBAL_ONCE_FIELDS = {
    "optimizer_step",
    "curvature_due",
    "trust_region_epsilon",
    "global_active_blocks",
    "global_resident_blocks",
    "predicted_stream_in_blocks",
    "prediction_missing_blocks",
    "prediction_extra_blocks",
    "prediction_replanned",
    "next_plan_target_iteration",
}
GLOBAL_TEXT_ONCE_FIELDS = {"optimizer_algorithm", "block_cull_backend"}

TIME_FIELDS = [field for field in BATCH_FIELDS if field.endswith("_ms")]
GLOBAL_FIELDS = (
    ["iteration", "world_size"]
    + sorted(SUM_FIELDS | GLOBAL_ONCE_FIELDS | GLOBAL_TEXT_ONCE_FIELDS)
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
COMPACTION_FIELDS = [
    "trigger",
    "iteration",
    "rank",
    "rounds",
    "before_patches",
    "after_patches",
    "input_bytes",
    "output_bytes",
    "reclaimed_bytes",
    "duration_ms",
    "actual_concurrency",
    "free_space_gb_before",
    "free_space_gb_after",
]
COMPACTION_GLOBAL_FIELDS = [
    "trigger",
    "iteration",
    "world_size",
    "rounds",
    "before_patches",
    "after_patches",
    "input_bytes",
    "output_bytes",
    "reclaimed_bytes",
    "duration_ms_max",
    "duration_ms_mean",
    "actual_concurrency",
    "free_space_gb_before",
    "free_space_gb_after",
]


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
        self.grad_zero_enabled = bool(
            getattr(args, "tide_grad_zero_metrics", False)
        )
        self.grad_zero_interval = max(
            1, int(getattr(args, "tide_grad_zero_metrics_interval", 1))
        )
        self.rank_path = Path(args.log_folder) / f"metrics_batch_rank{context.rank}.tsv"
        self.io_path = Path(args.log_folder) / f"metrics_io_rank{context.rank}.tsv"
        self.timeline_path = (
            Path(args.log_folder) / f"timeline_events_rank{context.rank}.jsonl"
        )
        self.global_path = Path(args.log_folder) / "metrics_batch_global.tsv"
        self.grad_zero_rank_path = (
            Path(args.log_folder) / f"metrics_grad_zero_rank{context.rank}.tsv"
        )
        self.grad_zero_global_path = (
            Path(args.log_folder) / "metrics_grad_zero_global.tsv"
        )
        self.async_rank_path = (
            Path(args.log_folder)
            / f"metrics_async_iteration_rank{context.rank}.tsv"
        )
        self.async_global_path = (
            Path(args.log_folder) / "metrics_async_iteration_global.tsv"
        )
        self.compaction_rank_path = (
            Path(args.log_folder)
            / f"metrics_compaction_rank{context.rank}.tsv"
        )
        self.compaction_global_path = (
            Path(args.log_folder) / "metrics_compaction_global.tsv"
        )
        self._async_by_iteration: Dict[int, Dict[str, float]] = {}
        self._async_finalized = False
        self._timeline_lock = threading.Lock()
        self._timeline_host = socket.gethostname()
        self._timeline_pid = os.getpid()

    def should_write_grad_zero(self, iteration: int) -> bool:
        return self.grad_zero_enabled and int(iteration) % self.grad_zero_interval == 0

    def write_grad_zero(self, row: Dict[str, object]) -> None:
        """Write exact zero-gradient counts for projection-cull survivors.

        Gaussian ownership is disjoint across ranks, so summing the per-rank
        unique-row counts and 0..59 histogram yields the global distribution.
        """

        if not self.grad_zero_enabled:
            return
        rank_row = dict(row)
        rank_row["rank"] = self.context.rank
        _append_tsv(self.grad_zero_rank_path, GRAD_ZERO_FIELDS, rank_row)
        rank_rows: List[Dict[str, object]] = self.context.all_gather_object(rank_row)
        if not self.context.is_rank0:
            return
        global_row: Dict[str, object] = {
            "iteration": int(rank_row["iteration"]),
            "world_size": int(self.context.world_size),
        }
        for field in GRAD_ZERO_VALUE_FIELDS:
            global_row[field] = sum(
                int(value.get(field, 0)) for value in rank_rows
            )
        _append_tsv(
            self.grad_zero_global_path,
            GRAD_ZERO_GLOBAL_FIELDS,
            global_row,
        )

    def write_timeline_event(
        self,
        *,
        name: str,
        lane: str,
        start_ns: int,
        end_ns: int | None = None,
        iteration: int | None = None,
        target_iteration: int | None = None,
        timing_source: str = "host_monotonic",
        **fields,
    ) -> None:
        """Append one interval to this rank's JSONL timeline.

        Background I/O is drained in batches, so JSONL append order is not a
        temporal guarantee; consumers must sort events by ``start_ns``.

        ``host_monotonic`` events use ``time.perf_counter_ns()`` from the
        shared host clock.  GPU events deliberately use
        ``host_submit_plus_cuda_event`` instead: their location is an
        approximate host submission point while their duration comes from a
        CUDA event.
        """
        if not self.enabled:
            return
        start_ns = int(start_ns)
        end_ns = start_ns if end_ns is None else int(end_ns)
        if end_ns < start_ns:
            raise ValueError(
                f"Timeline event {name!r} ends before it starts: "
                f"{end_ns} < {start_ns}"
            )
        event = {
            "clock": "perf_counter_ns",
            "host": self._timeline_host,
            "pid": self._timeline_pid,
            "rank": int(self.context.rank),
            "name": str(name),
            "lane": str(lane),
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": end_ns - start_ns,
            "iteration": None if iteration is None else int(iteration),
            "target_iteration": (
                None if target_iteration is None else int(target_iteration)
            ),
            "timing_source": str(timing_source),
            **fields,
        }
        with self._timeline_lock:
            self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.timeline_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

    def write_memory_point(
        self,
        *,
        iteration: int,
        phase: str,
        microbatch_index: int | None = None,
        microbatch_count: int | None = None,
        camera_uids: Iterable[int] | None = None,
        **fields,
    ) -> None:
        """Append a zero-duration CUDA allocator snapshot to the rank timeline.

        Memory points intentionally remain rank-local JSONL events: they are
        useful when a rank OOMs before the collective batch metric can be
        emitted, and do not belong on the global critical-path timeline.
        """
        if not self.enabled:
            return
        self.write_timeline_event(
            name="cuda_memory_point",
            lane="gpu",
            start_ns=time.perf_counter_ns(),
            iteration=int(iteration),
            timing_source="cuda_allocator_snapshot",
            phase=str(phase),
            microbatch_index=(
                None if microbatch_index is None else int(microbatch_index)
            ),
            microbatch_count=(
                None if microbatch_count is None else int(microbatch_count)
            ),
            camera_uids=(
                [] if camera_uids is None else [int(value) for value in camera_uids]
            ),
            **fields,
        )

    def write_io_events(self, events: Iterable[Dict[str, object]]) -> None:
        if not self.enabled:
            return
        if self._async_finalized:
            raise RuntimeError("Cannot record I/O events after metrics finalization")
        for event in events:
            row = dict(event)
            row["rank"] = self.context.rank
            _append_tsv(self.io_path, IO_FIELDS, row)
            start_ns = row.get("start_ns")
            if start_ns is not None:
                self.write_timeline_event(
                    name=str(row.get("operation", "io")),
                    lane=str(row.get("lane", row.get("tier", "io"))),
                    start_ns=int(start_ns),
                    end_ns=(
                        None
                        if row.get("end_ns") is None
                        else int(row["end_ns"])
                    ),
                    iteration=row.get("origin_iteration"),
                    target_iteration=row.get("target_iteration"),
                    timing_source=str(
                        row.get("timing_source", "host_monotonic")
                    ),
                    thread_id=row.get("thread_id"),
                    blocks=row.get("blocks"),
                    bytes=row.get("bytes"),
                    tier=row.get("tier"),
                    status=row.get("status"),
                )
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
        for field in GLOBAL_TEXT_ONCE_FIELDS:
            values = {str(value.get(field, "")) for value in rank_rows}
            global_row[field] = values.pop() if len(values) == 1 else "mixed"
        for field in TIME_FIELDS:
            values = [float(value.get(field, 0.0)) for value in rank_rows]
            global_row[f"{field}_max"] = max(values, default=0.0)
            global_row[f"{field}_mean"] = (
                sum(values) / len(values) if values else 0.0
            )
        _append_tsv(self.global_path, GLOBAL_FIELDS, global_row)

    def write_compaction(self, row: Dict[str, object]) -> None:
        if not self.enabled:
            return
        rank_row = dict(row)
        rank_row["rank"] = self.context.rank
        _append_tsv(
            self.compaction_rank_path,
            COMPACTION_FIELDS,
            rank_row,
        )
        rank_rows: List[Dict[str, object]] = self.context.all_gather_object(
            rank_row
        )
        if not self.context.is_rank0:
            return
        durations = [
            float(value.get("duration_ms", 0.0)) for value in rank_rows
        ]
        global_row = {
            "trigger": rank_row.get("trigger", ""),
            "iteration": int(rank_row["iteration"]),
            "world_size": int(self.context.world_size),
            "rounds": sum(
                int(value.get("rounds", 0)) for value in rank_rows
            ),
            "before_patches": sum(
                int(value.get("before_patches", 0)) for value in rank_rows
            ),
            "after_patches": sum(
                int(value.get("after_patches", 0)) for value in rank_rows
            ),
            "input_bytes": sum(
                int(value.get("input_bytes", 0)) for value in rank_rows
            ),
            "output_bytes": sum(
                int(value.get("output_bytes", 0)) for value in rank_rows
            ),
            "reclaimed_bytes": sum(
                int(value.get("reclaimed_bytes", 0)) for value in rank_rows
            ),
            "duration_ms_max": max(durations, default=0.0),
            "duration_ms_mean": (
                sum(durations) / len(durations) if durations else 0.0
            ),
            "actual_concurrency": max(
                int(value.get("actual_concurrency", 0))
                for value in rank_rows
            ),
            "free_space_gb_before": min(
                float(value.get("free_space_gb_before", 0.0))
                for value in rank_rows
            ),
            "free_space_gb_after": min(
                float(value.get("free_space_gb_after", 0.0))
                for value in rank_rows
            ),
        }
        _append_tsv(
            self.compaction_global_path,
            COMPACTION_GLOBAL_FIELDS,
            global_row,
        )

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


class LegacyCudaMetricsCollector:
    """Defer legacy CUDA-event reads without adding a per-batch GPU sync.

    ``torch.cuda.Event.elapsed_time`` is only valid after its end event has
    completed.  The legacy engine records event pairs while submitting work and
    queues the row here.  Later batches opportunistically flush completed rows;
    shutdown performs the only blocking flush.  This keeps the measured
    ``batch_total_ms`` on the original foreground path.
    """

    def __init__(self, writer: DistributedMetricsWriter):
        self.writer = writer
        self._pending: List[tuple[Dict[str, object], Dict[str, List[Dict[str, object]]]]] = []

    def enqueue(
        self,
        row: Dict[str, object],
        cuda_ranges: Dict[str, List[Dict[str, object]]],
    ) -> None:
        if not self.writer.enabled:
            return
        self._pending.append((dict(row), dict(cuda_ranges)))
        self.flush_ready()

    def flush_ready(self) -> None:
        while self._pending:
            row, cuda_ranges = self._pending[0]
            if any(
                not spec["end"].query()
                for specs in cuda_ranges.values()
                for spec in specs
            ):
                return
            self._pending.pop(0)
            self._write_resolved(row, cuda_ranges)

    def finalize(self) -> None:
        """Resolve the remaining rows during shutdown only."""
        for _, cuda_ranges in self._pending:
            for specs in cuda_ranges.values():
                for spec in specs:
                    spec["end"].synchronize()
        self.flush_ready()

    def _write_resolved(
        self,
        row: Dict[str, object],
        cuda_ranges: Dict[str, List[Dict[str, object]]],
    ) -> None:
        for field, specs in cuda_ranges.items():
            total_ms = 0.0
            for spec in specs:
                duration_ms = float(spec["start"].elapsed_time(spec["end"]))
                total_ms += duration_ms
                start_ns = spec.get("start_ns")
                if start_ns is not None:
                    self.writer.write_timeline_event(
                        name=str(spec.get("name", field)),
                        lane=str(spec.get("lane", "gpu")),
                        start_ns=int(start_ns),
                        end_ns=int(start_ns) + round(duration_ms * 1e6),
                        iteration=row.get("iteration"),
                        timing_source="host_submit_plus_cuda_event",
                        detail=spec.get("detail"),
                    )
            row[field] = total_ms

        row["train_ms"] = (
            float(row.get("gsplat_forward_ms", 0.0))
            + float(row.get("backward_ms", 0.0))
            + float(row.get("optimizer_ms", 0.0))
        )
        self.writer.write_batch(row)


def get_distributed_metrics_writer(gaussians, context) -> DistributedMetricsWriter:
    writer = getattr(gaussians, "_tide_distributed_metrics_writer", None)
    if writer is None:
        writer = DistributedMetricsWriter(args=gaussians.args, context=context)
        gaussians._tide_distributed_metrics_writer = writer
    return writer


def get_legacy_cuda_metrics_collector(gaussians, context) -> LegacyCudaMetricsCollector:
    collector = getattr(gaussians, "_tide_legacy_cuda_metrics_collector", None)
    if collector is None:
        collector = LegacyCudaMetricsCollector(
            get_distributed_metrics_writer(gaussians, context)
        )
        gaussians._tide_legacy_cuda_metrics_collector = collector
    return collector
