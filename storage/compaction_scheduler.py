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


def _rank_groups(
    world_size: int,
    *,
    requested_concurrency: int,
) -> List[List[int]]:
    """Split ranks into waves of at most ``requested_concurrency``."""
    concurrency = max(1, min(int(requested_concurrency), int(world_size)))
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
    forced_trigger: Optional[str] = None,
    flush_dirty_cache: bool = False,
) -> Optional[Dict[str, object]]:
    """Run one globally coordinated periodic maintenance window."""
    storage = storage_adapter.storage
    target_patch_files = max(1, int(target_patch_files))

    def snapshot_state() -> Dict[str, int]:
        stats = storage.get_stats()
        return {
            "rank": int(context.rank),
            "num_patches": int(stats["num_patches"]),
            "estimated_output_bytes": int(
                storage.estimate_next_compaction_output_bytes()
            ),
        }

    rank_states = context.all_gather_object(snapshot_state())
    if forced_trigger is None and periodic_iteration is None:
        return None

    trigger = (
        str(forced_trigger)
        if forced_trigger is not None
        else "periodic"
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

    groups = _rank_groups(
        len(rank_states),
        requested_concurrency=rank_concurrency,
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
    local_result["after_patches"] = int(final_stats["num_patches"])
    return local_result
