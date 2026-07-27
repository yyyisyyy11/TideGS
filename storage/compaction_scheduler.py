"""Iteration-based compaction scheduling for TideGS SSD training."""

from __future__ import annotations

import time
from typing import Dict, List, Optional


def crossed_periodic_iteration(
    *,
    iteration: int,
    batch_size: int,
    interval_iterations: int,
) -> Optional[int]:
    """Return the periodic target crossed by ``[iteration, iteration + batch_size)``."""
    interval_iterations = int(interval_iterations)
    if interval_iterations <= 0:
        return None
    iteration = int(iteration)
    batch_size = int(batch_size)
    target = ((iteration + interval_iterations - 1) // interval_iterations)
    target *= interval_iterations
    if target <= 0:
        target = interval_iterations
    return target if iteration <= target < iteration + batch_size else None


def resolve_emergency_free_gb(*, configured_gb: float, min_free_gb: float) -> float:
    configured_gb = float(configured_gb)
    return 2.0 * float(min_free_gb) if configured_gb < 0 else configured_gb


def _rank_groups(
    rank_states: List[Dict[str, float]],
    *,
    requested_concurrency: int,
    target_patch_files: int,
) -> List[List[int]]:
    world_size = len(rank_states)
    concurrency = max(1, min(int(requested_concurrency), world_size))
    active_estimates = sorted(
        (
            int(state["estimated_output_bytes"])
            for state in rank_states
            if int(state["num_patches"]) > int(target_patch_files)
        ),
        reverse=True,
    )
    shared_free_bytes = min(
        int(state["free_bytes"]) for state in rank_states
    )
    shared_reserve_bytes = max(
        int(state["min_free_bytes"]) for state in rank_states
    )
    if (
        concurrency > 1
        and active_estimates
        and shared_free_bytes
        < shared_reserve_bytes + sum(active_estimates[:concurrency])
    ):
        concurrency = 1
    return [
        list(range(start, min(start + concurrency, world_size)))
        for start in range(0, world_size, concurrency)
    ]


def run_compaction_maintenance(
    *,
    context,
    storage_adapter,
    iteration: int,
    periodic_iteration: Optional[int],
    target_patch_files: int,
    rank_concurrency: int,
    emergency_free_gb: float,
    forced_trigger: Optional[str] = None,
    flush_dirty_cache: bool = False,
) -> Optional[Dict[str, object]]:
    """Run one globally coordinated periodic or emergency maintenance window."""
    storage = storage_adapter.storage
    target_patch_files = max(1, int(target_patch_files))

    def snapshot_state() -> Dict[str, int]:
        stats = storage.get_stats()
        return {
            "rank": int(context.rank),
            "free_bytes": int(
                float(stats["free_space_gb"]) * (1024 ** 3)
            ),
            "min_free_bytes": int(storage.min_free_bytes),
            "num_patches": int(stats["num_patches"]),
            "estimated_output_bytes": int(
                storage.estimate_next_compaction_output_bytes()
            ),
        }

    rank_states = context.all_gather_object(snapshot_state())
    minimum_free_gb = min(
        float(state["free_bytes"]) / (1024 ** 3) for state in rank_states
    )
    emergency = minimum_free_gb < float(emergency_free_gb)
    if forced_trigger is None and periodic_iteration is None and not emergency:
        return None

    trigger = (
        str(forced_trigger)
        if forced_trigger is not None
        else ("periodic" if periodic_iteration is not None else "emergency")
    )
    trigger_iteration = (
        int(periodic_iteration)
        if periodic_iteration is not None
        else int(iteration)
    )
    preparation_error = None
    try:
        if flush_dirty_cache:
            flush_dirty = getattr(
                storage_adapter,
                "flush_dirty_cache_to_storage",
                None,
            )
            if callable(flush_dirty):
                flush_dirty()
            else:
                storage_adapter.drain_cache_writebacks()
        else:
            drain_storage_writebacks = getattr(
                storage_adapter,
                "drain_storage_writebacks",
                None,
            )
            if callable(drain_storage_writebacks):
                drain_storage_writebacks()
            else:
                storage_adapter.drain_cache_writebacks()
        wait_for_reads = getattr(
            storage_adapter,
            "wait_for_storage_reads",
            None,
        )
        if callable(wait_for_reads):
            wait_for_reads()
    except Exception as exc:
        preparation_error = (
            f"rank {context.rank}: {type(exc).__name__}: {exc}"
        )
    preparation_errors = context.all_gather_object(preparation_error)
    failures = [error for error in preparation_errors if error]
    if failures:
        raise RuntimeError(
            "SSD compaction preparation failed: " + "; ".join(failures)
        )
    context.barrier()

    local_state = snapshot_state()
    rank_states = context.all_gather_object(local_state)
    minimum_free_gb = min(
        float(state["free_bytes"]) / (1024 ** 3) for state in rank_states
    )
    if (
        emergency
        and all(
            int(state["num_patches"]) <= target_patch_files
            for state in rank_states
        )
    ):
        raise RuntimeError(
            "SSD free space is below the emergency compaction threshold, "
            "but every rank is already at the patch low watermark"
        )

    groups = _rank_groups(
        rank_states,
        requested_concurrency=rank_concurrency,
        target_patch_files=target_patch_files,
    )
    local_result = {
        "trigger": trigger,
        "iteration": trigger_iteration,
        "rank": int(context.rank),
        "rounds": 0,
        "before_patches": int(local_state["num_patches"]),
        "after_patches": int(local_state["num_patches"]),
        "input_bytes": 0,
        "output_bytes": 0,
        "reclaimed_bytes": 0,
        "duration_ms": 0.0,
        "actual_concurrency": 0,
        "free_space_gb_before": minimum_free_gb,
        "free_space_gb_after": minimum_free_gb,
    }

    for group in groups:
        local_error = None
        if int(context.rank) in group:
            active_ranks = [
                rank
                for rank in group
                if int(rank_states[rank]["num_patches"]) > target_patch_files
            ]
            local_result["actual_concurrency"] = len(active_ranks)
            started = time.perf_counter()
            try:
                result = storage.compact_to_patch_count(
                    target_patch_files=target_patch_files,
                )
                local_result.update(result)
            except Exception as exc:
                local_error = f"rank {context.rank}: {type(exc).__name__}: {exc}"
            finally:
                local_result["duration_ms"] = (
                    time.perf_counter() - started
                ) * 1000.0
        errors = context.all_gather_object(local_error)
        failures = [error for error in errors if error]
        if failures:
            raise RuntimeError(
                "Coordinated SSD compaction failed: " + "; ".join(failures)
            )
        context.barrier()

    final_stats = storage.get_stats()
    final_free = context.all_gather_object(float(final_stats["free_space_gb"]))
    final_minimum_free_gb = min(float(value) for value in final_free)
    local_result["after_patches"] = int(final_stats["num_patches"])
    local_result["free_space_gb_after"] = final_minimum_free_gb
    if emergency and final_minimum_free_gb < float(emergency_free_gb):
        raise RuntimeError(
            "SSD free space remains below the emergency threshold after "
            f"compaction: free={final_minimum_free_gb:.2f} GiB "
            f"threshold={float(emergency_free_gb):.2f} GiB"
        )
    return local_result
