import os
import sys
import json
import csv
import gc
import hashlib
import time
import psutil
import numpy as np
from collections import Counter
from pathlib import Path

# import faulthandler
# faulthandler_log = open("fault.log", "w")
# faulthandler.enable(file=faulthandler_log, all_threads=True)

import torch
import torch.multiprocessing
from torch.cuda import nvtx
from tqdm import tqdm

from utils.mem_monitor import MemMonitor
from utils.camera_batch_prefetcher import CameraBatchPrefetcher
from argparse import ArgumentParser
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    BenchmarkParams,
    DebugParams,
    print_all_args,
    init_args,
)

from scene import Scene, OffloadSceneDataset
from strategies.tide_engine.gaussian_model import TideGaussianModel
from strategies.tide_engine.runtime import (
    train_tide_batch,
    validate_tide_runtime_args,
)

from utils.general_utils import safe_state, prepare_output_and_logger
import utils.general_utils as utils
from utils.timer import Timer, End2endTimer

from storage.tide_storage_adapter import TideStorageAdapter
from storage.compaction_scheduler import (
    crossed_periodic_iteration,
    resolve_emergency_free_gb,
    run_compaction_maintenance,
)
from storage.schedule_utils import (
    get_camera_batch_schedule,
    rotate_training_schedule,
    validate_trajectory_start_offset,
)
from storage.pure_ssd_checkpoint import (
    is_pure_ssd_checkpoint,
    load_pure_ssd_checkpoint_manifest,
    prune_checkpoint_history,
    write_pure_ssd_incremental_checkpoint,
    write_pure_ssd_snapshot_checkpoint,
)
from storage.distributed_checkpoint import (
    is_distributed_checkpoint,
    load_distributed_checkpoint_manifest,
    write_distributed_incremental_checkpoint,
)
from strategies.tide_engine.distributed_plan import (
    DistributedBatchPlan,
    DistributedBatchPlanner,
    DistributedPlannerState,
    build_stable_block_owner,
    validate_block_owner,
)
from strategies.tide_engine.checkpoint_validation import (
    validate_pure_ssd_checkpoint_optimizer,
)
from strategies.tide_engine.sophia_tr_math import (
    optimizer_step_from_iteration,
    should_update_curvature,
)
from strategies.tide_engine.sophia_tr_sampling import sample_s2_camera_ids
from strategies.tide_engine.gsplat_backend import prepare_distributed_gsplat
from strategies.tide_engine.distributed_metrics import (
    get_distributed_metrics_writer,
)
from utils.distributed import DistributedContext, get_distributed_context


CULL_METRIC_FIELDS = (
    "block_cull_backend",
    "block_cull_gpu_kernel_ms",
    "block_cull_gpu_d2h_ms",
    "block_cull_cache_hit_cameras",
    "block_cull_gpu_cameras",
    "block_cull_output_blocks",
)

CAMERA_BATCH_FIELDS = (
    "iteration",
    "optimizer_step",
    "trajectory_start_offset",
    "batch_start_position",
    "camera_count",
    "camera_ids_json",
)


def _schedule_sha256(schedule):
    payload = ",".join(str(int(camera_id)) for camera_id in schedule)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _camera_window_geometry(storage_adapter, camera_ids):
    camera_ids = [int(camera_id) for camera_id in camera_ids]
    positions = np.asarray(storage_adapter.camera_positions, dtype=np.float64)[
        camera_ids
    ]
    directions = np.asarray(storage_adapter.camera_directions, dtype=np.float64)[
        camera_ids
    ]
    clusters = getattr(storage_adapter, "camera_clusters", None)
    cluster_histogram = {}
    if clusters is not None and len(clusters) == len(storage_adapter.cameras):
        selected_clusters = [int(clusters[camera_id]) for camera_id in camera_ids]
        cluster_histogram = {
            str(cluster_id): int(count)
            for cluster_id, count in sorted(Counter(selected_clusters).items())
        }
    return {
        "position_centroid": positions.mean(axis=0).tolist(),
        "position_min": positions.min(axis=0).tolist(),
        "position_max": positions.max(axis=0).tolist(),
        "mean_view_direction": directions.mean(axis=0).tolist(),
        "cluster_count": len(cluster_histogram),
        "cluster_histogram": cluster_histogram,
    }


def _write_schedule_metadata(
    *,
    log_folder,
    storage_adapter,
    canonical_schedule,
    rotated_schedule,
    requested_offset,
    effective_offset,
    window_camera_count,
):
    window_camera_ids = [
        int(camera_id) for camera_id in rotated_schedule[:window_camera_count]
    ]
    metadata = {
        "num_cameras": len(canonical_schedule),
        "requested_trajectory_start_offset": int(requested_offset),
        "effective_trajectory_start_offset": int(effective_offset),
        "canonical_schedule_sha256": _schedule_sha256(canonical_schedule),
        "rotated_schedule_sha256": _schedule_sha256(rotated_schedule),
        "canonical_first_camera_id": int(canonical_schedule[0]),
        "canonical_last_camera_id": int(canonical_schedule[-1]),
        "rotated_first_camera_id": int(rotated_schedule[0]),
        "rotated_last_camera_id": int(rotated_schedule[-1]),
        "analysis_window_camera_count": len(window_camera_ids),
        "analysis_window_camera_ids": window_camera_ids,
        "analysis_window_geometry": _camera_window_geometry(
            storage_adapter,
            window_camera_ids,
        ),
    }
    path = Path(log_folder) / "schedule_metadata.json"
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)
    return metadata


def _append_camera_batch_metrics(log_folder, row):
    path = Path(log_folder) / "metrics_camera_batches.tsv"
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CAMERA_BATCH_FIELDS,
            delimiter="\t",
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerow({field: row[field] for field in CAMERA_BATCH_FIELDS})


def _optimizer_algorithm(args):
    return str(getattr(args, "paper_optimizer_algorithm", "adam")).lower()


def _curvature_camera_ids_for_step(
    *,
    args,
    training_schedule,
    s1_camera_ids,
    optimizer_step,
):
    if _optimizer_algorithm(args) != "3dgs2_tr":
        return []
    interval = int(getattr(args, "paper_sophia_curvature_interval", 10))
    if not should_update_curvature(int(optimizer_step), interval):
        return []
    return sample_s2_camera_ids(
        population_camera_ids=training_schedule,
        s1_camera_ids=s1_camera_ids,
        seed=int(getattr(args, "paper_sophia_curvature_seed", 1)),
        optimizer_step=int(optimizer_step),
    )


def _union_camera_ids(s1_camera_ids, s2_camera_ids):
    s1_ids = [int(value) for value in s1_camera_ids]
    seen = set(s1_ids)
    return s1_ids + [
        int(value) for value in s2_camera_ids if int(value) not in seen
    ]


def _split_camera_blocks(camera_blocks, camera_ids):
    return {
        int(camera_id): camera_blocks[int(camera_id)]
        for camera_id in camera_ids
    }


def _prepare_camera_batch_on_gpu(cameras, camera_ids):
    if len(cameras) != len(camera_ids):
        raise RuntimeError(
            f"Camera batch size mismatch: cameras={len(cameras)} ids={len(camera_ids)}"
        )
    for uid, (camera, global_idx) in enumerate(zip(cameras, camera_ids)):
        camera.uid = uid
        camera.global_idx = int(global_idx)
        camera.world_view_transform = camera.world_view_transform.cuda()
        camera.full_proj_transform = camera.full_proj_transform.cuda()

    if not cameras:
        return
    world_view_transforms = torch.stack(
        [camera.world_view_transform.transpose(0, 1) for camera in cameras]
    )
    camera_to_world = torch.unbind(torch.inverse(world_view_transforms), dim=0)
    for camera, inverse in zip(cameras, camera_to_world):
        camera.K = camera.create_k_on_gpu()
        camera.camtoworlds = inverse.unsqueeze(0)
        camera.original_image = camera.original_image_backup.cuda()


def _planner_state_from_manifest(value):
    if value is None:
        return None
    return DistributedPlannerState(
        resident=tuple(int(block_id) for block_id in value["resident"]),
        active=tuple(int(block_id) for block_id in value["active"]),
        recency={
            int(block_id): float(score)
            for block_id, score in value["recency"].items()
        },
    )


def _prepare_distributed_block_owner(
    *,
    args,
    context,
    storage_adapter,
    training_schedule,
    resume_manifest,
):
    def prepare_rank0_owner():
        owner_file = None if resume_manifest is None else resume_manifest.get("_tide_block_owner_file")
        if owner_file:
            owner = np.load(owner_file).astype(np.int32, copy=False)
            owner_policy = str(
                resume_manifest.get("_tide_owner_policy", "checkpoint")
            )
        else:
            owner = build_stable_block_owner(
                num_blocks=int(storage_adapter.num_blocks),
                world_size=int(context.world_size),
            )
            owner_policy = "stable_round_robin"
        return {
            "owner": owner.tolist(),
            "owner_policy": owner_policy,
        }

    owner_payload = _run_synchronized_training_phase(
        context,
        phase="block owner preparation",
        operation=prepare_rank0_owner if context.is_rank0 else None,
    )
    owner_payload = context.broadcast_object(owner_payload)
    owner = np.asarray(owner_payload["owner"], dtype=np.int32)
    args._tide_owner_policy = str(owner_payload["owner_policy"])

    def configure_local_owner():
        validate_block_owner(
            owner,
            num_blocks=int(storage_adapter.num_blocks),
            world_size=int(context.world_size),
        )
        storage_adapter.configure_block_ownership(owner, context.rank)

    _run_synchronized_training_phase(
        context,
        phase="block owner configuration",
        operation=configure_local_owner,
    )
    args._tide_block_owner = owner
    if context.is_rank0:
        counts = np.bincount(owner, minlength=context.world_size).tolist()
        utils.print_rank_0(
            f"[DISTRIBUTED] Static block ownership ({args._tide_owner_policy}) "
            f"per rank: {counts}"
        )
    return owner


def _build_distributed_plan(
    *,
    context,
    planner,
    storage_adapter,
    training_schedule,
    iteration,
    batch_size,
    schedule_ordering,
    curvature_camera_ids=None,
):
    schedule_info = get_camera_batch_schedule(
        training_schedule=training_schedule,
        iteration=iteration,
        batch_size=batch_size,
        schedule_ordering=schedule_ordering,
    )
    def build_rank0_payload():
        rank0_curvature_camera_ids = [
            int(value) for value in (curvature_camera_ids or [])
        ]
        union_camera_ids = _union_camera_ids(
            schedule_info.batch_indices,
            rank0_curvature_camera_ids,
        )
        cull_start = time.perf_counter()
        _, union_camera_blocks = storage_adapter.get_visible_blocks_batch(
            union_camera_ids
        )
        block_cull_ms = (time.perf_counter() - cull_start) * 1000.0
        camera_blocks = _split_camera_blocks(
            union_camera_blocks,
            schedule_info.batch_indices,
        )
        curvature_camera_blocks = (
            _split_camera_blocks(union_camera_blocks, rank0_curvature_camera_ids)
            if rank0_curvature_camera_ids
            else None
        )
        plan_start = time.perf_counter()
        plan = planner.plan(
            iteration=iteration,
            epoch=schedule_info.epoch,
            camera_ids=schedule_info.batch_indices,
            camera_blocks=camera_blocks,
            curvature_camera_ids=(rank0_curvature_camera_ids or None),
            curvature_camera_blocks=curvature_camera_blocks,
        )
        payload = plan.to_dict()
        payload["block_cull_ms"] = block_cull_ms
        payload["plan_ms"] = (time.perf_counter() - plan_start) * 1000.0
        cull_metrics = getattr(storage_adapter, "_batch_cull_metrics", None)
        if cull_metrics:
            payload.update(cull_metrics)
        return payload

    payload = _run_synchronized_training_phase(
        context,
        phase="initial distributed planning",
        operation=build_rank0_payload if context.is_rank0 else None,
    )
    return DistributedBatchPlan.from_dict(context.broadcast_object(payload)), schedule_info


def _normalize_camera_blocks(camera_blocks):
    return {
        int(camera_id): tuple(int(block_id) for block_id in blocks)
        for camera_id, blocks in camera_blocks.items()
    }


def _preview_distributed_plan(
    *,
    context,
    planner,
    storage_adapter,
    training_schedule,
    iteration,
    batch_size,
    schedule_ordering,
    curvature_camera_ids=None,
):
    schedule_info = get_camera_batch_schedule(
        training_schedule=training_schedule,
        iteration=iteration,
        batch_size=batch_size,
        schedule_ordering=schedule_ordering,
    )
    rank0_preview = None
    def build_rank0_preview():
        rank0_curvature_camera_ids = [
            int(value) for value in (curvature_camera_ids or [])
        ]
        union_camera_ids = _union_camera_ids(
            schedule_info.batch_indices,
            rank0_curvature_camera_ids,
        )
        cull_start = time.perf_counter()
        bounds_generation, union_camera_blocks = (
            storage_adapter.get_visible_blocks_batch(union_camera_ids)
        )
        block_cull_ms = (time.perf_counter() - cull_start) * 1000.0
        camera_blocks = _split_camera_blocks(
            union_camera_blocks,
            schedule_info.batch_indices,
        )
        curvature_camera_blocks = (
            _split_camera_blocks(union_camera_blocks, rank0_curvature_camera_ids)
            if rank0_curvature_camera_ids
            else None
        )
        plan_start = time.perf_counter()
        plan, predicted_state = planner.preview(
            iteration=iteration,
            epoch=schedule_info.epoch,
            camera_ids=schedule_info.batch_indices,
            camera_blocks=camera_blocks,
            curvature_camera_ids=(rank0_curvature_camera_ids or None),
            curvature_camera_blocks=curvature_camera_blocks,
        )
        plan_ms = (time.perf_counter() - plan_start) * 1000.0
        payload = plan.to_dict()
        payload["block_cull_ms"] = block_cull_ms
        payload["plan_ms"] = plan_ms
        cull_metrics = getattr(storage_adapter, "_batch_cull_metrics", None)
        if cull_metrics:
            payload.update(cull_metrics)
        preview = {
            "bounds_generation": int(bounds_generation),
            "union_camera_blocks": _normalize_camera_blocks(
                union_camera_blocks
            ),
            "predicted_state": predicted_state,
        }
        return payload, preview

    preview_result = _run_synchronized_training_phase(
        context,
        phase="predictive distributed planning",
        operation=build_rank0_preview if context.is_rank0 else None,
    )
    payload = None
    if context.is_rank0:
        payload, rank0_preview = preview_result
    plan = DistributedBatchPlan.from_dict(context.broadcast_object(payload))
    return plan, schedule_info, rank0_preview


def _finalize_distributed_plan(
    *,
    context,
    planner,
    storage_adapter,
    training_schedule,
    iteration,
    batch_size,
    schedule_ordering,
    predicted_plan,
    rank0_preview,
    curvature_camera_ids=None,
):
    schedule_info = get_camera_batch_schedule(
        training_schedule=training_schedule,
        iteration=iteration,
        batch_size=batch_size,
        schedule_ordering=schedule_ordering,
    )
    def finalize_rank0_payload():
        if rank0_preview is None:
            raise RuntimeError("Rank 0 is missing distributed plan preview state")
        if predicted_plan.global_camera_ids != schedule_info.batch_indices:
            raise RuntimeError(
                f"Distributed prediction mismatch at iteration {iteration}"
            )
        rank0_curvature_camera_ids = [
            int(value) for value in (curvature_camera_ids or [])
        ]
        if predicted_plan.global_s2_camera_ids != rank0_curvature_camera_ids:
            raise RuntimeError(
                f"Distributed S2 prediction mismatch at iteration {iteration}"
            )
        union_camera_ids = _union_camera_ids(
            schedule_info.batch_indices,
            rank0_curvature_camera_ids,
        )

        repair_start = time.perf_counter()
        current_generation = storage_adapter.get_bounds_generation()
        repair_cull_metrics = None
        if current_generation == rank0_preview["bounds_generation"]:
            union_camera_blocks = rank0_preview["union_camera_blocks"]
        else:
            _, union_camera_blocks = storage_adapter.get_visible_blocks_batch(
                union_camera_ids
            )
            union_camera_blocks = _normalize_camera_blocks(
                union_camera_blocks
            )
            repair_cull_metrics = getattr(storage_adapter, "_batch_cull_metrics", None)
        repair_ms = (time.perf_counter() - repair_start) * 1000.0

        prediction_changed = (
            union_camera_blocks != rank0_preview["union_camera_blocks"]
        )
        exact_plan_ms = 0.0
        if prediction_changed:
            camera_blocks = _split_camera_blocks(
                union_camera_blocks,
                schedule_info.batch_indices,
            )
            curvature_camera_blocks = (
                _split_camera_blocks(
                    union_camera_blocks,
                    rank0_curvature_camera_ids,
                )
                if rank0_curvature_camera_ids
                else None
            )
            plan_start = time.perf_counter()
            exact_plan = planner.plan(
                iteration=iteration,
                epoch=schedule_info.epoch,
                camera_ids=schedule_info.batch_indices,
                camera_blocks=camera_blocks,
                curvature_camera_ids=(rank0_curvature_camera_ids or None),
                curvature_camera_blocks=curvature_camera_blocks,
            )
            exact_plan_ms = (time.perf_counter() - plan_start) * 1000.0
        else:
            planner.restore_state(rank0_preview["predicted_state"])
            exact_plan = predicted_plan

        predicted_stream_in = set(predicted_plan.stream_in_blocks)
        exact_stream_in = set(exact_plan.stream_in_blocks)
        payload = exact_plan.to_dict()
        payload["block_cull_ms"] = float(predicted_plan.block_cull_ms) + repair_ms
        payload["plan_ms"] = float(predicted_plan.plan_ms) + exact_plan_ms
        payload.update(
            {
                field: getattr(predicted_plan, field)
                for field in CULL_METRIC_FIELDS
            }
        )
        if repair_cull_metrics:
            repair_backend = str(repair_cull_metrics["block_cull_backend"])
            payload["block_cull_backend"] = (
                predicted_plan.block_cull_backend
                if repair_backend == predicted_plan.block_cull_backend
                else "mixed"
            )
            for field in (
                "block_cull_gpu_kernel_ms",
                "block_cull_gpu_d2h_ms",
                "block_cull_cache_hit_cameras",
                "block_cull_gpu_cameras",
                "block_cull_output_blocks",
            ):
                payload[field] = (
                    getattr(predicted_plan, field) + repair_cull_metrics[field]
                )
        payload["predicted_stream_in_blocks"] = len(predicted_stream_in)
        payload["prediction_missing_blocks"] = len(
            exact_stream_in - predicted_stream_in
        )
        payload["prediction_extra_blocks"] = len(
            predicted_stream_in - exact_stream_in
        )
        payload["prediction_replanned"] = int(prediction_changed)
        payload["prediction_repair_ms"] = repair_ms
        payload["prediction_exact_plan_ms"] = exact_plan_ms
        return payload

    payload = _run_synchronized_training_phase(
        context,
        phase="final distributed planning",
        operation=finalize_rank0_payload if context.is_rank0 else None,
    )
    plan = DistributedBatchPlan.from_dict(context.broadcast_object(payload))
    return plan, schedule_info


def _load_pure_ssd_prebuilt_manifest(args):
    manifest_path = getattr(args, "pure_ssd_prebuilt_manifest", "")
    if not manifest_path:
        return None

    manifest_path = Path(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_dir = manifest_path.parent
    base_file = Path(getattr(args, "pure_ssd_prebuilt_base_file", "") or manifest["base_file"])
    block_bounds = Path(getattr(args, "pure_ssd_prebuilt_block_bounds", "") or manifest["block_bounds"])
    if not base_file.is_absolute():
        base_file = manifest_dir / base_file
    if not block_bounds.is_absolute():
        block_bounds = manifest_dir / block_bounds
    base_file = base_file.resolve()
    block_bounds = block_bounds.resolve()
    if not base_file.is_file():
        raise FileNotFoundError(f"Pure SSD prebuilt base_file not found: {base_file}")
    if not block_bounds.is_file():
        raise FileNotFoundError(f"Pure SSD prebuilt block_bounds not found: {block_bounds}")

    total_points = int(manifest["total_points"])
    param_dim = int(manifest.get("param_dim", 59))
    expected_size = total_points * param_dim * 4
    actual_size = base_file.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Pure SSD prebuilt base_file size mismatch: got {actual_size}, expected {expected_size}"
        )

    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["base_file"] = str(base_file)
    manifest["block_bounds"] = str(block_bounds)
    manifest.setdefault("param_dim", param_dim)
    manifest.setdefault("next_iteration", 1)
    manifest["_prebuilt_base_reuse"] = True
    return manifest


def _validate_pure_ssd_runtime(args, gaussians, storage_adapter, resolved_backend, log_file):
    """Fail early if TideGS silently falls back to a RAM-resident state."""
    if not getattr(args, "pure_ssd_offload", False):
        return

    if storage_adapter is None:
        raise RuntimeError("[PURE SSD CHECK] storage_adapter is required")
    if getattr(storage_adapter, "execution_mode", "fast_ram") != "paper":
        raise RuntimeError("[PURE SSD CHECK] storage_adapter.execution_mode must be 'paper'")
    if resolved_backend != "tiered_cache":
        raise RuntimeError(
            f"[PURE SSD CHECK] BlockReader backend must be tiered_cache, got {resolved_backend!r}"
        )
    block_reader = getattr(gaussians, "_block_reader", None)
    if block_reader is None or block_reader.__class__.__name__ != "TieredCacheBlockReader":
        raise RuntimeError(
            "[PURE SSD CHECK] gaussians._block_reader must be a TieredCacheBlockReader"
        )
    if getattr(gaussians, "_unified_params", None) is not None:
        raise RuntimeError("[PURE SSD CHECK] _unified_params must be released")
    if not getattr(gaussians, "_paper_unified_params_freed", False):
        raise RuntimeError("[PURE SSD CHECK] _paper_unified_params_freed marker is missing")
    if getattr(args, "paper_optimizer_backend", "cpu") != "gpu_resident":
        raise RuntimeError("[PURE SSD CHECK] optimizer backend must be gpu_resident")
    if getattr(args, "paper_optimizer_state_mode", "full_cpu") != "resident_blocks":
        raise RuntimeError("[PURE SSD CHECK] optimizer state mode must be resident_blocks")

    total_gaussians = getattr(gaussians, "_paper_unified_params_num_total", None)
    total_desc = f"{int(total_gaussians):,}" if total_gaussians is not None else "unknown"
    num_blocks = getattr(storage_adapter, "num_blocks", None)
    cache = getattr(storage_adapter, "cache", None)
    cache_limit_gb = getattr(cache, "max_ram_bytes", 0) / (1024 ** 3) if cache is not None else 0.0
    init_state = (
        "_unified_params never allocated"
        if getattr(gaussians, "_paper_unified_params_never_allocated", False)
        else "_unified_params released"
    )
    optimizer_label = (
        "GPUResident3DGS2-TR"
        if _optimizer_algorithm(args) == "3dgs2_tr"
        else "GPUResidentAdam"
    )
    message = (
        "[PURE SSD CHECK] Runtime path verified: "
        f"gaussians={total_desc} blocks={num_blocks} "
        f"block_reader=TieredCacheBlockReader optimizer={optimizer_label} "
        f"state=resident_blocks init={init_state} ram_cache_limit={cache_limit_gb:.2f}GB\n"
    )
    log_file.write(message)
    utils.print_rank_0(message.strip())


def _run_synchronized_training_phase(context, *, phase, operation=None):
    """Run a local storage phase and publish failures before later collectives."""

    result = None
    local_error = None
    try:
        if operation is not None:
            result = operation()
    except BaseException as error:
        local_error = error

    if not context.enabled:
        if local_error is not None:
            raise local_error
        return result

    statuses = context.all_gather_object(
        {
            "rank": int(context.rank),
            "error": (
                None
                if local_error is None
                else f"{type(local_error).__name__}: {local_error}"
            ),
        }
    )
    failures = [
        f"rank={status.get('rank', rank)} {status['error']}"
        for rank, status in enumerate(statuses)
        if status.get("error")
    ]
    if failures:
        synchronized_error = RuntimeError(
            f"Distributed training {phase} failed: " + "; ".join(failures)
        )
        if local_error is not None:
            raise synchronized_error from local_error
        raise synchronized_error
    return result


def training(dataset_args, opt_args, pipe_args, args, log_file):
    """Main training loop for the pure SSD/Tide release path."""

    distributed_context = get_distributed_context(args)
    distributed_enabled = distributed_context.enabled

    # ============================================================================
    # STAGE 1: INITIALIZATION
    # ============================================================================

    assert args.dataset_cache_and_stream_mode in [
        "load_from_source_on_demand",
        "load_from_disk_on_demand",
    ], f"Unsupported dataset_cache_and_stream_mode: {args.dataset_cache_and_stream_mode}"
    try:
        validate_tide_runtime_args(args)
    except RuntimeError as exc:
        raise ValueError(
            "train_tidegs.py is the pure SSD/Tide release entry. "
            "Use scripts/train_matrixcity_1b.sh or enable the TideGS SSD flags."
        ) from exc
    if distributed_enabled:
        prepare_distributed_gsplat(
            distributed_context,
            enable_timing=bool(args.tide_detailed_metrics),
            enable_grad_zero_metrics=bool(args.tide_grad_zero_metrics),
        )
    # ------------------------------------------------------------------------
    # 1.1: Setup auxiliary tools and GPU configuration
    # ------------------------------------------------------------------------
    gc.set_threshold(700, 10, 500)  # gen0, gen1, gen2

    torch.cuda.set_device(args.gpu)
    timers = Timer(args)
    utils.set_timers(timers)
    _run_synchronized_training_phase(
        distributed_context,
        phase="output initialization",
        operation=(
            lambda: prepare_output_and_logger(dataset_args)
            if not distributed_enabled or distributed_context.is_rank0
            else None
        ),
    )
    distributed_context.barrier()
    utils.log_cpu_memory_usage("at the beginning of training")
    start_from_this_iteration = 1
    completed_optimizer_steps = 0
    pure_ssd_resume_manifest = None
    pure_ssd_prebuilt_manifest = _run_synchronized_training_phase(
        distributed_context,
        phase="prebuilt manifest validation",
        operation=lambda: _load_pure_ssd_prebuilt_manifest(args),
    )
    if pure_ssd_prebuilt_manifest is not None and args.start_checkpoint != "":
        raise ValueError(
            "--pure_ssd_prebuilt_manifest is for fresh runs from an existing SSD base; "
            "use --start_checkpoint alone for checkpoint resume."
        )
    if pure_ssd_prebuilt_manifest is not None:
        args._pure_ssd_prebuilt_manifest = pure_ssd_prebuilt_manifest
        args.gaussian_block_size = int(pure_ssd_prebuilt_manifest["block_size"])
        prebuilt_msg = (
            "[PURE SSD PREBUILT] loaded manifest: "
            f"{args.pure_ssd_prebuilt_manifest} "
            f"points={int(pure_ssd_prebuilt_manifest['total_points']):,} "
            f"base={pure_ssd_prebuilt_manifest.get('base_file')}"
        )
        utils.print_rank_0(prebuilt_msg)
        log_file.write(prebuilt_msg + "\n")
    if args.start_checkpoint != "":
        if distributed_enabled:
            def load_distributed_resume_manifest():
                if not is_distributed_checkpoint(args.start_checkpoint):
                    raise ValueError(
                        "gaussian_sharded mode only resumes distributed Pure SSD "
                        "checkpoints."
                    )
                return load_distributed_checkpoint_manifest(
                    args.start_checkpoint,
                    rank=distributed_context.rank,
                    world_size=distributed_context.world_size,
                    global_bsz=args.bsz,
                    args=args,
                )

            pure_ssd_resume_manifest = _run_synchronized_training_phase(
                distributed_context,
                phase="distributed checkpoint resume validation",
                operation=load_distributed_resume_manifest,
            )
        else:
            if not is_pure_ssd_checkpoint(args.start_checkpoint):
                raise ValueError(
                    "train_tidegs.py only resumes pure SSD checkpoints."
                )
            pure_ssd_resume_manifest = load_pure_ssd_checkpoint_manifest(
                args.start_checkpoint
            )
            validate_pure_ssd_checkpoint_optimizer(
                args,
                pure_ssd_resume_manifest,
            )
        args._pure_ssd_resume_manifest = pure_ssd_resume_manifest
        start_from_this_iteration = int(pure_ssd_resume_manifest["next_iteration"])
        derived_completed_steps = max(
            0,
            (start_from_this_iteration - 1) // int(args.bsz),
        )
        completed_optimizer_steps = int(
            pure_ssd_resume_manifest.get(
                "_tide_global_optimizer_step",
                derived_completed_steps,
            )
        )
        if completed_optimizer_steps != derived_completed_steps:
            raise ValueError(
                "Checkpoint optimizer step does not match next_iteration/global batch: "
                f"step={completed_optimizer_steps} expected={derived_completed_steps}"
            )
        args.gaussian_block_size = int(pure_ssd_resume_manifest["block_size"])
        resume_msg = (
            "[PURE SSD RESUME] loaded checkpoint: "
            f"{args.start_checkpoint} next_iteration={start_from_this_iteration} "
            f"base={pure_ssd_resume_manifest.get('base_file')}"
        )
        utils.print_rank_0(resume_msg)
        log_file.write(resume_msg + "\n")

    # Configure multiprocessing sharing strategy if needed
    if args.sharing_strategy != "default":
        torch.multiprocessing.set_sharing_strategy(args.sharing_strategy)

    # ------------------------------------------------------------------------
    # 1.2: Initialize pure SSD Gaussian shell
    # ------------------------------------------------------------------------
    gaussians = TideGaussianModel(sh_degree=dataset_args.sh_degree)
    utils.print_rank_0("Using TideGaussianModel for TideGS/SSD")
    log_file.write("Using TideGaussianModel for TideGS/SSD\n")

    storage_adapter = None
    ssd_training_schedule = None

    with torch.no_grad():
        scene = _run_synchronized_training_phase(
            distributed_context,
            phase="scene initialization",
            operation=lambda: Scene(args, gaussians),
        )
        utils.print_rank_0("[SSD] Initializing Tide storage engine...")
        storage_dir = (
            os.path.join(args.ssd_cache_dir, f"rank_{distributed_context.rank}")
            if distributed_enabled
            else args.ssd_cache_dir
        )
        def create_storage_adapter():
            return TideStorageAdapter(
                gaussians=gaussians,
                cameras=scene.getTrainCamerasInfo(),
                storage_dir=storage_dir,
                block_size=args.gaussian_block_size,
                max_ram_gb=args.max_ram_gb,
                num_clusters=args.num_clusters,
                use_6plane=args.use_6plane,
                execution_mode=args.ssd_execution_mode,
                max_patch_files=args.tide_storage_max_patch_files,
                max_patch_gb=args.tide_storage_max_patch_gb,
                min_free_gb=args.tide_storage_min_free_gb,
                compaction_batch_files=args.tide_storage_compaction_batch_files,
                idle_compaction_seconds=args.tide_storage_idle_compaction_seconds,
                block_cull_backend=getattr(args, "tide_block_cull_backend", "cpu") or "cpu",
                block_cull_camera_chunk=int(getattr(args, "tide_block_cull_camera_chunk", 8) or 8),
            )

        ssd_schedule_ordering = getattr(args, "ssd_schedule_ordering", "trajectory")
        ssd_schedule_shuffle = ssd_schedule_ordering == "shuffle"
        if distributed_enabled:
            def create_rank0_storage_and_schedule():
                rank0_adapter = create_storage_adapter()
                rank0_schedule = rank0_adapter.get_training_schedule(
                    shuffle=ssd_schedule_shuffle
                )
                return rank0_adapter, rank0_schedule

            rank0_result = _run_synchronized_training_phase(
                distributed_context,
                phase="rank 0 storage and schedule initialization",
                operation=(
                    create_rank0_storage_and_schedule
                    if distributed_context.is_rank0
                    else None
                ),
            )
            if distributed_context.is_rank0:
                storage_adapter, ssd_training_schedule = rank0_result
            ssd_training_schedule = distributed_context.broadcast_object(
                ssd_training_schedule
            )
            worker_adapter = _run_synchronized_training_phase(
                distributed_context,
                phase="worker storage initialization",
                operation=(
                    create_storage_adapter
                    if not distributed_context.is_rank0
                    else None
                ),
            )
            if not distributed_context.is_rank0:
                storage_adapter = worker_adapter
            distributed_context.barrier()
        else:
            storage_adapter = create_storage_adapter()
            ssd_training_schedule = storage_adapter.get_training_schedule(
                shuffle=ssd_schedule_shuffle
            )

        canonical_ssd_training_schedule = [
            int(camera_id) for camera_id in ssd_training_schedule
        ]
        requested_trajectory_offset = validate_trajectory_start_offset(
            ssd_schedule_ordering,
            getattr(args, "tide_trajectory_start_offset", 0),
        )
        ssd_training_schedule, effective_trajectory_offset = (
            rotate_training_schedule(
                canonical_ssd_training_schedule,
                requested_trajectory_offset,
            )
        )
        args._tide_trajectory_start_offset_effective = int(
            effective_trajectory_offset
        )
        planned_batch_count = len(
            range(
                start_from_this_iteration,
                opt_args.iterations + 1,
                int(args.bsz),
            )
        )
        if distributed_context.is_rank0:
            schedule_metadata = _write_schedule_metadata(
                log_folder=args.log_folder,
                storage_adapter=storage_adapter,
                canonical_schedule=canonical_ssd_training_schedule,
                rotated_schedule=ssd_training_schedule,
                requested_offset=requested_trajectory_offset,
                effective_offset=effective_trajectory_offset,
                window_camera_count=min(
                    len(ssd_training_schedule),
                    planned_batch_count * int(args.bsz),
                ),
            )
            schedule_message = (
                "[SSD Schedule] canonical_sha256="
                f"{schedule_metadata['canonical_schedule_sha256']} "
                f"rotated_sha256={schedule_metadata['rotated_schedule_sha256']} "
                f"requested_offset={requested_trajectory_offset} "
                f"effective_offset={effective_trajectory_offset} "
                f"first_camera={ssd_training_schedule[0]} "
                f"last_camera={ssd_training_schedule[-1]}"
            )
            utils.print_rank_0(schedule_message)
            log_file.write(schedule_message + "\n")

        log_file.write(f"[SSD] Execution mode: {args.ssd_execution_mode}\n")

        utils.print_rank_0(f"[SSD] Schedule ordering: {ssd_schedule_ordering}")
        log_file.write(f"[SSD] Schedule ordering: {ssd_schedule_ordering}\n")

        gaussians.offload_params_to_ssd_storage()
        utils.print_rank_0("[PURE SSD] Training will use the SSD → RAM → GPU pipeline")


        gaussians.training_setup(opt_args)

        from storage.block_reader import TieredCacheBlockReader, resolve_block_reader_backend

        requested_backend = args.paper_block_reader_backend
        resolved_backend = resolve_block_reader_backend(
            requested_backend,
            getattr(storage_adapter, 'execution_mode', 'paper'),
        )
        if resolved_backend != 'tiered_cache':
            raise RuntimeError(
                "train_tidegs.py requires paper_block_reader_backend=tiered_cache "
                f"(resolved {resolved_backend!r})"
            )

        total_gaussians = getattr(gaussians, '_paper_unified_params_num_total', None)
        if total_gaussians is None:
            total_gaussians = getattr(storage_adapter, 'num_points', None)
        gaussians._block_reader = TieredCacheBlockReader(
            cache_manager=storage_adapter.cache,
            total_gaussians=int(total_gaussians),
            block_size=int(args.gaussian_block_size),
            before_read=storage_adapter.wait_for_cache_blocks,
            filter_hint=storage_adapter.filter_cache_prefetch_candidates,
        )

        utils.print_rank_0(
            f"[SSD] BlockReader backend = {resolved_backend} "
            f"(requested={requested_backend}, ssd_execution_mode={storage_adapter.execution_mode})"
        )
        log_file.write(
            f"[SSD] BlockReader backend = {resolved_backend} "
            f"(requested={requested_backend}, ssd_execution_mode={storage_adapter.execution_mode})\n"
        )

        gaussians.free_unified_params()
        utils.print_rank_0(
            "[SSD] ✓ Paper mode: _unified_params released; "
            "reads served by TieredCacheBlockReader, writeback via direct GPU→cache"
        )

        _validate_pure_ssd_runtime(args, gaussians, storage_adapter, resolved_backend, log_file)

        distributed_planner = None
        if distributed_enabled:
            block_owner = _prepare_distributed_block_owner(
                args=args,
                context=distributed_context,
                storage_adapter=storage_adapter,
                training_schedule=ssd_training_schedule,
                resume_manifest=pure_ssd_resume_manifest,
            )
            if distributed_context.is_rank0:
                distributed_planner = DistributedBatchPlanner(
                    block_owner=block_owner,
                    total_points=int(storage_adapter.num_points),
                    block_size=int(storage_adapter.block_size),
                    world_size=int(distributed_context.world_size),
                    resident_capacity_blocks=int(args.paper_resident_capacity_blocks),
                    resident_lambda=float(args.paper_resident_lambda),
                    resident_recency_decay=float(args.paper_resident_recency_decay),
                    balanced_seed_fraction=float(args.paper_balanced_seed_fraction),
                    camera_assignment=str(args.tide_camera_assignment),
                )
                resume_planner_state = _planner_state_from_manifest(
                    None
                    if pure_ssd_resume_manifest is None
                    else pure_ssd_resume_manifest.get("_tide_planner_state")
                )
                if resume_planner_state is not None:
                    distributed_planner.restore_state(resume_planner_state)
            log_file.write(
                f"[DISTRIBUTED] rank={distributed_context.rank}/{distributed_context.world_size} "
                f"local_gpu={distributed_context.local_rank} "
                f"global_cap={args.paper_resident_capacity_blocks}\n"
            )

        if pure_ssd_resume_manifest is not None:
            optimizer_label = (
                "3DGS2-TR"
                if _optimizer_algorithm(args) == "3dgs2_tr"
                else "Adam"
            )
            msg = (
                f"[PURE SSD RESUME] GPUResident{optimizer_label} cold-started; "
                "optimizer EMA state is not restored"
            )
            utils.print_rank_0(msg)
            log_file.write(msg + "\n")

        gaussians._tide_global_optimizer_step = int(completed_optimizer_steps)
        args._tide_global_optimizer_step = int(completed_optimizer_steps)

        scene.log_scene_info_to_file(log_file, "Scene Info Before Training")
    emergency_compaction_free_gb = resolve_emergency_free_gb(
        configured_gb=args.tide_storage_compaction_emergency_free_gb,
        min_free_gb=args.tide_storage_min_free_gb,
    )
    final_storage_maintenance_complete = False
    utils.check_initial_gpu_memory_usage("after init and before training loop")

    # ------------------------------------------------------------------------
    # 1.3: Initialize data loader
    # ------------------------------------------------------------------------
    train_dataset = OffloadSceneDataset(scene.getTrainCamerasInfo())

    # ------------------------------------------------------------------------
    # 1.4: Initialize background and CUDA streams
    # ------------------------------------------------------------------------
    background = None
    bg_color = [1, 1, 1] if dataset_args.white_background else None

    if bg_color is not None:
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Dedicated stream for CPU↔GPU communication (overlapped with compute)
    comm_stream = torch.cuda.Stream(device=args.gpu, priority=args.comm_stream_priority)

    # ------------------------------------------------------------------------
    # 1.5: Initialize training loop state
    # ------------------------------------------------------------------------
    end2end_timers = End2endTimer(args)
    end2end_timers.start()
    progress_bar = tqdm(
        range(1, opt_args.iterations + 1),
        desc="Training progress",
        disable=distributed_enabled and not distributed_context.is_rank0,
    )
    progress_bar.update(start_from_this_iteration - 1)
    num_trained_batches = 0

    mem_mon = MemMonitor(log_dir=args.log_folder, warn_avail_gb=15.0)

    # Random number generator for camera ordering in retention-based offloading
    perm_generator = torch.Generator(device="cuda")
    perm_generator.manual_seed(1)

    # Training state variables
    ema_loss_for_log = 0
    last_iteration = None

    optimizer_churn_tsv = None
    optimizer_churn_state = {
        'current_epoch': None,
        'last_optimizer_rows_total': 0,
        'last_cold_rows_total': 0,
        'last_cold_restarts_total': 0,
        'last_state_evictions_total': 0,
    }

    def _snapshot_optimizer_churn_counters():
        gpu_resident_optimizer = getattr(gaussians, '_paper_gpu_resident_optimizer', None)
        if gpu_resident_optimizer is None:
            return None
        stats = gpu_resident_optimizer.get_stats()
        if stats.get('state_mode', 'full_cpu') != 'resident_blocks':
            return None
        return {
            'optimizer_rows_touched_total': int(stats.get('optimizer_rows_touched_total', 0)),
            'cold_restarted_rows_touched_total': int(stats.get('cold_restarted_rows_touched_total', 0)),
            'cold_restarts_total': int(stats.get('cold_restarts', 0)),
            'state_evictions_total': int(stats.get('state_evictions', 0)),
            'mean_resident_streak': float(stats.get('mean_resident_streak', 0.0)),
        }

    def _write_optimizer_churn_epoch(epoch_zero_based: int, iteration_end: int):
        if optimizer_churn_tsv is None:
            return
        stats = _snapshot_optimizer_churn_counters()
        if stats is None:
            return

        epoch_optimizer_rows = max(0, stats['optimizer_rows_touched_total'] - optimizer_churn_state['last_optimizer_rows_total'])
        epoch_cold_rows = max(0, stats['cold_restarted_rows_touched_total'] - optimizer_churn_state['last_cold_rows_total'])
        epoch_cold_restarts = max(0, stats['cold_restarts_total'] - optimizer_churn_state['last_cold_restarts_total'])
        epoch_state_evictions = max(0, stats['state_evictions_total'] - optimizer_churn_state['last_state_evictions_total'])
        epoch_cold_ratio_pct = 100.0 * epoch_cold_rows / max(1, epoch_optimizer_rows)

        optimizer_churn_tsv.write(
            f"{epoch_zero_based + 1}\t{iteration_end}\t{epoch_optimizer_rows}\t{epoch_cold_rows}\t"
            f"{epoch_cold_ratio_pct:.6f}\t{epoch_cold_restarts}\t{epoch_state_evictions}\t"
            f"{stats['mean_resident_streak']:.6f}\n"
        )
        log_file.write(
            f"[OPTIMIZER CHURN] Epoch {epoch_zero_based + 1}: rows={epoch_optimizer_rows}, "
            f"cold_rows={epoch_cold_rows}, cold_ratio={epoch_cold_ratio_pct:.4f}%, "
            f"cold_restarts={epoch_cold_restarts}, state_evictions={epoch_state_evictions}, "
            f"mean_streak={stats['mean_resident_streak']:.4f}\n"
        )

        optimizer_churn_state['last_optimizer_rows_total'] = stats['optimizer_rows_touched_total']
        optimizer_churn_state['last_cold_rows_total'] = stats['cold_restarted_rows_touched_total']
        optimizer_churn_state['last_cold_restarts_total'] = stats['cold_restarts_total']
        optimizer_churn_state['last_state_evictions_total'] = stats['state_evictions_total']

    def _enable_optimizer_churn_logging(reason: str):
        nonlocal optimizer_churn_tsv
        if optimizer_churn_tsv is not None:
            return
        churn_tsv_path = os.path.join(args.log_folder, 'optimizer_state_churn_by_epoch.tsv')
        optimizer_churn_tsv = open(churn_tsv_path, 'w', buffering=1)
        optimizer_churn_tsv.write(
            'epoch\titeration_end\toptimizer_rows_updated\tcold_restarted_rows_updated\t'
            'cold_restarted_row_ratio_pct\tcold_restarts\tstate_evictions\tmean_resident_streak\n'
        )
        log_file.write(f"[OPTIMIZER CHURN] Per-epoch churn logging enabled: {churn_tsv_path} ({reason})\n")
        log_file.flush()

    gaussians._paper_optimizer_state_mode = str(getattr(args, 'paper_optimizer_state_mode', 'resident_blocks')).lower()
    gaussians._paper_optimizer_block_size = int(getattr(args, 'gaussian_block_size', 4096))
    gaussians._paper_optimizer_backend = 'gpu_resident'
    _enable_optimizer_churn_logging("pure_ssd_gpu_resident")
    camera_batch_prefetcher = CameraBatchPrefetcher(train_dataset)
    curvature_camera_batch_prefetcher = (
        CameraBatchPrefetcher(train_dataset)
        if _optimizer_algorithm(args) == "3dgs2_tr"
        else None
    )
    timeline_writer = (
        get_distributed_metrics_writer(gaussians, distributed_context)
        if args.tide_detailed_metrics
        else None
    )
    if timeline_writer is not None:
        set_timeline_enabled = getattr(
            storage_adapter.cache, "set_timeline_enabled", None
        )
        if callable(set_timeline_enabled):
            set_timeline_enabled(True)
    current_distributed_plan = None
    initial_optimizer_step = optimizer_step_from_iteration(
        start_from_this_iteration,
        int(args.bsz),
    )
    if initial_optimizer_step != completed_optimizer_steps + 1:
        raise ValueError(
            "Resume optimizer clock is inconsistent with start iteration: "
            f"next_step={completed_optimizer_steps + 1} "
            f"iteration_step={initial_optimizer_step}"
        )
    if distributed_enabled:
        initial_schedule = get_camera_batch_schedule(
            training_schedule=ssd_training_schedule,
            iteration=start_from_this_iteration,
            batch_size=args.bsz,
            schedule_ordering=getattr(args, "ssd_schedule_ordering", "trajectory"),
        )
        initial_s2_camera_ids = _curvature_camera_ids_for_step(
            args=args,
            training_schedule=ssd_training_schedule,
            s1_camera_ids=initial_schedule.batch_indices,
            optimizer_step=initial_optimizer_step,
        )
        current_distributed_plan, _ = _build_distributed_plan(
            context=distributed_context,
            planner=distributed_planner,
            storage_adapter=storage_adapter,
            training_schedule=ssd_training_schedule,
            iteration=start_from_this_iteration,
            batch_size=args.bsz,
            schedule_ordering=getattr(args, "ssd_schedule_ordering", "trajectory"),
            curvature_camera_ids=initial_s2_camera_ids,
        )
        initial_camera_ids = current_distributed_plan.rank_camera_ids[
            distributed_context.rank
        ]
        initial_resident = current_distributed_plan.rank_resident_blocks[
            distributed_context.rank
        ]
        initial_curvature_camera_ids = current_distributed_plan.rank_s2_camera_ids[
            distributed_context.rank
        ]
        gaussians._block_reader.hint_future(
            initial_resident,
            target_iteration=start_from_this_iteration,
        )
    else:
        initial_camera_schedule = get_camera_batch_schedule(
            training_schedule=ssd_training_schedule,
            iteration=start_from_this_iteration,
            batch_size=args.bsz,
            schedule_ordering=getattr(args, "ssd_schedule_ordering", "trajectory"),
        )
        initial_camera_ids = initial_camera_schedule.batch_indices
        initial_curvature_camera_ids = _curvature_camera_ids_for_step(
            args=args,
            training_schedule=ssd_training_schedule,
            s1_camera_ids=initial_camera_ids,
            optimizer_step=initial_optimizer_step,
        )
    camera_batch_prefetcher.submit(initial_camera_ids)
    if initial_curvature_camera_ids:
        if curvature_camera_batch_prefetcher is None:
            raise RuntimeError("3DGS2-TR S2 prefetcher is not initialized")
        curvature_camera_batch_prefetcher.submit(initial_curvature_camera_ids)
    # ============================================================================
    # STAGE 2: MAIN TRAINING LOOP
    # ============================================================================

    forced_active_sh_degree = int(
        getattr(args, "tide_force_active_sh_degree", -1)
    )
    if not -1 <= forced_active_sh_degree <= int(gaussians.max_sh_degree):
        raise ValueError(
            "tide_force_active_sh_degree must be -1 or an integer in "
            f"[0, {int(gaussians.max_sh_degree)}], got "
            f"{forced_active_sh_degree}"
        )
    if forced_active_sh_degree >= 0:
        gaussians.active_sh_degree = forced_active_sh_degree
        log_file.write(
            "[SH DIAGNOSTIC] forcing active_sh_degree="
            f"{forced_active_sh_degree} for every batch\n"
        )

    for iteration in range(
        start_from_this_iteration, opt_args.iterations + 1, args.bsz
    ):
        # # rewrite the checking iterations
        # ------------------------------------------------------------------------
        # 2.1: Iteration setup and profiling
        # ------------------------------------------------------------------------
        # Optional: trace CUDA memory usage for debugging
        if args.trace_cuda_mem:
            if (iteration % args.log_interval) == 1 or (
                iteration % args.densification_interval
            ) == 0:
                torch.cuda.memory._record_memory_history()
                log_file.write(
                    "[ITER {}] Tracing cuda memory usage.\n".format(iteration)
                )

        # Update progress bar and iteration state
        if iteration // args.bsz % 30 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
        progress_bar.update(args.bsz)
        utils.set_cur_iter(iteration)
        optimizer_step = optimizer_step_from_iteration(iteration, int(args.bsz))
        if optimizer_step != completed_optimizer_steps + 1:
            raise RuntimeError(
                "Global optimizer clock diverged from the training iteration: "
                f"iteration={iteration} step={optimizer_step} "
                f"completed={completed_optimizer_steps}"
            )
        gaussians.update_learning_rate(iteration)  # Learning rate scheduling
        num_trained_batches += 1
        last_iteration = iteration

        # Optional: reset memory tracking for per-iteration profiling
        if args.reset_each_iter:
            torch.cuda.reset_max_memory_cached()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_max_memory_allocated()

        # Start timing this iteration
        timers.clear()
        timers.start("[iteration end2end]")

        # Optional: NSight Systems profiling
        if args.nsys_profile:
            if iteration == args.nsys_profile_start_iter:
                torch.cuda.cudart().cudaProfilerStart()
            if (
                iteration == args.nsys_profile_end_iter
                or iteration == opt_args.iterations
            ):
                torch.cuda.cudart().cudaProfilerStop()
            if (
                iteration >= args.nsys_profile_start_iter
                and iteration < args.nsys_profile_end_iter
            ):
                nvtx.range_push(f"iteration[{iteration},{iteration+args.bsz})")

        # Gradually increase spherical harmonics degree (every 1000 iterations)
        if forced_active_sh_degree >= 0:
            gaussians.active_sh_degree = forced_active_sh_degree
        elif utils.check_update_at_this_iter(iteration, args.bsz, 1000, 0):
            gaussians.oneupSHdegree()

        # ------------------------------------------------------------------------
        # 2.2: Load training data (camera images)
        # ------------------------------------------------------------------------
        timers.start("dataloader: load the next image from disk and decode")

        if (
            optimizer_churn_tsv is None
            and getattr(args, 'pure_ssd_offload', False)
            and str(getattr(args, 'paper_optimizer_backend', '')).lower() == 'gpu_resident'
            and str(getattr(args, 'paper_optimizer_state_mode', '')).lower() == 'resident_blocks'
        ):
            _enable_optimizer_churn_logging("pure_ssd_gpu_resident")

        nvtx.range_push("Outer: next_camera_load")
        schedule_info = get_camera_batch_schedule(
            training_schedule=ssd_training_schedule,
            iteration=iteration,
            batch_size=args.bsz,
            schedule_ordering=getattr(args, "ssd_schedule_ordering", "trajectory"),
        )
        global_batch_indices = schedule_info.batch_indices
        if distributed_context.is_rank0:
            canonical_batch_start_position = (
                effective_trajectory_offset + int(schedule_info.batch_start_cam)
            ) % len(ssd_training_schedule)
            _append_camera_batch_metrics(
                args.log_folder,
                {
                    "iteration": int(iteration),
                    "optimizer_step": int(optimizer_step),
                    "trajectory_start_offset": int(effective_trajectory_offset),
                    "batch_start_position": int(canonical_batch_start_position),
                    "camera_count": len(global_batch_indices),
                    "camera_ids_json": json.dumps(
                        [int(camera_id) for camera_id in global_batch_indices],
                        separators=(",", ":"),
                    ),
                },
            )
        global_curvature_batch_indices = _curvature_camera_ids_for_step(
            args=args,
            training_schedule=ssd_training_schedule,
            s1_camera_ids=global_batch_indices,
            optimizer_step=optimizer_step,
        )
        if distributed_enabled:
            if (
                current_distributed_plan is None
                or current_distributed_plan.iteration != iteration
                or current_distributed_plan.global_camera_ids != global_batch_indices
            ):
                raise RuntimeError(
                    f"Distributed plan mismatch at iteration {iteration}"
                )
            if (
                current_distributed_plan.global_s2_camera_ids
                != global_curvature_batch_indices
            ):
                raise RuntimeError(
                    f"Distributed S2 plan mismatch at iteration {iteration}"
                )
            batch_indices = current_distributed_plan.rank_camera_ids[
                distributed_context.rank
            ]
            curvature_batch_indices = (
                current_distributed_plan.rank_s2_camera_ids[
                    distributed_context.rank
                ]
            )
        else:
            batch_indices = global_batch_indices
            curvature_batch_indices = global_curvature_batch_indices
        checkpoint_planner_state = (
            distributed_planner.snapshot_state()
            if distributed_enabled and distributed_context.is_rank0
            else None
        )
        epoch = schedule_info.epoch
        within_epoch_idx = schedule_info.within_epoch_idx
        n_batches = schedule_info.num_batches
        epoch_camera_offset = schedule_info.epoch_camera_offset

        if optimizer_churn_tsv is not None:
            if optimizer_churn_state['current_epoch'] is None:
                optimizer_churn_state['current_epoch'] = epoch
            elif epoch != optimizer_churn_state['current_epoch']:
                _write_optimizer_churn_epoch(
                    epoch_zero_based=optimizer_churn_state['current_epoch'],
                    iteration_end=max(start_from_this_iteration, iteration - args.bsz),
                )
                optimizer_churn_state['current_epoch'] = epoch

        batched_cameras = camera_batch_prefetcher.get(batch_indices)
        curvature_cameras = []
        if curvature_batch_indices:
            if curvature_camera_batch_prefetcher is None:
                raise RuntimeError("3DGS2-TR S2 prefetcher is not initialized")
            curvature_cameras = curvature_camera_batch_prefetcher.get(
                curvature_batch_indices
            )
        nvtx.range_pop()

        next_iteration = iteration + args.bsz
        next_optimizer_step = optimizer_step + 1
        next_curvature_camera_ids = []
        if next_iteration <= opt_args.iterations:
            next_schedule = get_camera_batch_schedule(
                training_schedule=ssd_training_schedule,
                iteration=next_iteration,
                batch_size=args.bsz,
                schedule_ordering=getattr(
                    args, "ssd_schedule_ordering", "trajectory"
                ),
            )
            next_curvature_camera_ids = _curvature_camera_ids_for_step(
                args=args,
                training_schedule=ssd_training_schedule,
                s1_camera_ids=next_schedule.batch_indices,
                optimizer_step=next_optimizer_step,
            )
        next_distributed_prediction = None
        next_distributed_rank0_preview = None
        if distributed_enabled and next_iteration <= opt_args.iterations:
            nvtx.range_push("Outer: distributed_predictive_prefetch")
            preview_start_ns = time.perf_counter_ns()
            (
                next_distributed_prediction,
                _,
                next_distributed_rank0_preview,
            ) = _preview_distributed_plan(
                context=distributed_context,
                planner=distributed_planner,
                storage_adapter=storage_adapter,
                training_schedule=ssd_training_schedule,
                iteration=next_iteration,
                batch_size=args.bsz,
                schedule_ordering=getattr(
                    args, "ssd_schedule_ordering", "trajectory"
                ),
                curvature_camera_ids=next_curvature_camera_ids,
            )
            preview_end_ns = time.perf_counter_ns()
            if timeline_writer is not None:
                timeline_writer.write_timeline_event(
                    name="preview_plan",
                    lane="cpu",
                    start_ns=preview_start_ns,
                    end_ns=preview_end_ns,
                    iteration=iteration,
                    target_iteration=next_iteration,
                )
            rank = distributed_context.rank
            current_resident = set(
                current_distributed_plan.rank_resident_blocks[rank]
            )
            predicted_resident = set(
                next_distributed_prediction.rank_resident_blocks[rank]
            )
            gaussians._block_reader.hint_future(
                sorted(predicted_resident - current_resident),
                target_iteration=next_iteration,
            )
            nvtx.range_pop()

        if not distributed_enabled and next_iteration <= opt_args.iterations:
            nvtx.range_push("Outer: next_camera_prefetch_submit")
            camera_batch_prefetcher.submit(next_schedule.batch_indices)
            if next_curvature_camera_ids:
                if curvature_camera_batch_prefetcher is None:
                    raise RuntimeError("3DGS2-TR S2 prefetcher is not initialized")
                curvature_camera_batch_prefetcher.submit(
                    next_curvature_camera_ids
                )
            nvtx.range_pop()

        if iteration % 100 == 0:
            log_file.write(
                f"[SSD Schedule] Iter {iteration}: epoch={epoch}, "
                f"within_epoch_batch={within_epoch_idx}/{n_batches}, "
                f"camera_offset={epoch_camera_offset}, "
                f"Global camera indices: {global_batch_indices[:3]}..."
                f"{global_batch_indices[-1]}; local={batch_indices}\n"
            )

        timers.stop("dataloader: load the next image from disk and decode")

        # ------------------------------------------------------------------------
        # 2.3: Transfer camera matrices to GPU
        # ------------------------------------------------------------------------
        nvtx.range_push("Outer: camera_h2d")
        timers.start("send cam matrices to gpu")
        with torch.no_grad():
            _prepare_camera_batch_on_gpu(batched_cameras, batch_indices)
            _prepare_camera_batch_on_gpu(
                curvature_cameras,
                curvature_batch_indices,
            )
        timers.stop("send cam matrices to gpu")
        nvtx.range_pop()
        assert args.bsz > 1, "Pipelined offload requires batch size > 1"
        losses, ordered_cams, sparsity = train_tide_batch(
            gaussians=gaussians,
            scene=scene,
            batched_cameras=batched_cameras,
            parameters_grad_buffer=gaussians.parameters_grad_buffer,
            background=background,
            pipe_args=pipe_args,
            comm_stream=comm_stream,
            perm_generator=perm_generator,
            storage_adapter=storage_adapter,
            training_schedule=ssd_training_schedule,
            distributed_plan=current_distributed_plan,
            distributed_context=distributed_context,
            curvature_cameras=curvature_cameras,
            optimizer_step=optimizer_step,
        )

        completed_optimizer_steps = optimizer_step
        gaussians._tide_global_optimizer_step = int(completed_optimizer_steps)
        args._tide_global_optimizer_step = int(completed_optimizer_steps)

        mem_mon.tick(iteration)

        if len(losses) == 0:
            if distributed_enabled:
                raise RuntimeError(
                    f"Distributed rank {distributed_context.rank} produced no loss at "
                    f"iteration {iteration}"
                )
            log_file.write(
                f"[WARNING] Iteration {iteration}: All {len(batched_cameras)} cameras see no Gaussians; "
                "skipping optimizer step.\n"
            )
            for camera in batched_cameras + curvature_cameras:
                camera.original_image = None
            continue

        batched_cameras = [batched_cameras[i] for i in ordered_cams]

        nvtx.range_push("Outer: loss_sync")
        timers.start("sync_loss_and_log")
        batched_losses = torch.stack(losses)
        local_loss_cpu = batched_losses.cpu().numpy()
        if distributed_enabled:
            local_payload = {
                "records": [
                    (
                        int(camera.global_idx),
                        float(loss),
                        str(camera.image_name),
                    )
                    for camera, loss in zip(batched_cameras, local_loss_cpu)
                ],
                "sparsity": float(sparsity),
            }
            rank_payloads = distributed_context.all_gather_object(local_payload)
            records = [
                record
                for payload in rank_payloads
                for record in payload["records"]
            ]
            if len(records) != len(global_batch_indices):
                raise RuntimeError(
                    f"Distributed loss count mismatch: got={len(records)} "
                    f"expected={len(global_batch_indices)}"
                )
            by_camera = {camera_id: (loss, image_name) for camera_id, loss, image_name in records}
            if len(by_camera) != len(global_batch_indices):
                raise RuntimeError("Distributed camera batch contains duplicate camera IDs")
            try:
                ordered_records = [by_camera[camera_id] for camera_id in global_batch_indices]
            except KeyError as exc:
                raise RuntimeError(
                    f"Distributed loss is missing camera {int(exc.args[0])}"
                ) from exc
            batched_loss_cpu = np.asarray(
                [record[0] for record in ordered_records], dtype=np.float32
            )
            logged_image_names = [record[1] for record in ordered_records]
            logged_sparsity = [payload["sparsity"] for payload in rank_payloads]
        else:
            batched_loss_cpu = local_loss_cpu
            logged_image_names = [camera.image_name for camera in batched_cameras]
            logged_sparsity = sparsity
        nvtx.range_pop()

        nvtx.range_push("Outer: batch_logging")
        ema_loss_for_log = (
            batched_loss_cpu.mean()
            if ema_loss_for_log is None
            else 0.6 * ema_loss_for_log + 0.4 * batched_loss_cpu.mean()
        )

        if not distributed_enabled or distributed_context.is_rank0:
            train_dataset.update_losses(batched_loss_cpu)

        batched_loss_cpu = [round(loss, 6) for loss in batched_loss_cpu]
        log_file.write(
            "iteration[{},{}), loss: {} sparsity: {} image: {}\n".format(
                iteration,
                iteration + args.bsz,
                batched_loss_cpu,
                logged_sparsity,
                logged_image_names,
            )
        )
        timers.stop("sync_loss_and_log")
        nvtx.range_pop()

        if distributed_enabled:
            next_distributed_plan = None
            if next_iteration <= opt_args.iterations:
                if next_distributed_prediction is None:
                    raise RuntimeError(
                        f"Missing distributed prediction for iteration {next_iteration}"
                    )
                finalize_start_ns = time.perf_counter_ns()
                next_distributed_plan, _ = _finalize_distributed_plan(
                    context=distributed_context,
                    planner=distributed_planner,
                    storage_adapter=storage_adapter,
                    training_schedule=ssd_training_schedule,
                    iteration=next_iteration,
                    batch_size=args.bsz,
                    schedule_ordering=getattr(args, "ssd_schedule_ordering", "trajectory"),
                    predicted_plan=next_distributed_prediction,
                    rank0_preview=next_distributed_rank0_preview,
                    curvature_camera_ids=next_curvature_camera_ids,
                )
                finalize_end_ns = time.perf_counter_ns()
                if timeline_writer is not None:
                    timeline_writer.write_timeline_event(
                        name="finalize_plan",
                        lane="cpu",
                        start_ns=finalize_start_ns,
                        end_ns=finalize_end_ns,
                        iteration=iteration,
                        target_iteration=next_iteration,
                    )
                rank = distributed_context.rank
                current_resident = set(
                    current_distributed_plan.rank_resident_blocks[rank]
                )
                next_resident = set(
                    next_distributed_plan.rank_resident_blocks[rank]
                )
                predicted_resident = set(
                    next_distributed_prediction.rank_resident_blocks[rank]
                )
                gaussians._block_reader.hint_future(
                    sorted(
                        (next_resident - current_resident)
                        - (predicted_resident - current_resident)
                    ),
                    target_iteration=next_iteration,
                )
                camera_batch_prefetcher.submit(
                    next_distributed_plan.rank_camera_ids[rank]
                )
                next_rank_s2 = next_distributed_plan.rank_s2_camera_ids[rank]
                if next_rank_s2:
                    if curvature_camera_batch_prefetcher is None:
                        raise RuntimeError(
                            "3DGS2-TR S2 prefetcher is not initialized"
                        )
                    curvature_camera_batch_prefetcher.submit(next_rank_s2)
            current_distributed_plan = next_distributed_plan

        matched_checkpoint_iterations = [
            checkpoint_iteration
            for checkpoint_iteration in args.checkpoint_iterations
            if iteration <= checkpoint_iteration < iteration + args.bsz
        ]
        periodic_compaction_iteration = crossed_periodic_iteration(
            iteration=iteration,
            batch_size=args.bsz,
            interval_iterations=(
                args.tide_storage_compaction_interval_iterations
            ),
        )
        is_final_batch = next_iteration > opt_args.iterations
        if matched_checkpoint_iterations or is_final_batch:
            _run_synchronized_training_phase(
                distributed_context,
                phase="resident dirty flush",
                operation=storage_adapter.flush_resident_dirty,
            )
        compaction_result = run_compaction_maintenance(
            context=distributed_context,
            storage_adapter=storage_adapter,
            iteration=iteration,
            periodic_iteration=periodic_compaction_iteration,
            target_patch_files=(
                args.tide_storage_compaction_target_patch_files
            ),
            rank_concurrency=args.tide_storage_compaction_rank_concurrency,
            emergency_free_gb=emergency_compaction_free_gb,
            flush_dirty_cache=bool(
                matched_checkpoint_iterations or is_final_batch
            ),
        )
        if compaction_result is not None:
            get_distributed_metrics_writer(
                gaussians,
                distributed_context,
            ).write_compaction(compaction_result)
            utils.print_rank_0(
                "[SSD COMPACTION] "
                f"trigger={compaction_result['trigger']} "
                f"iteration={compaction_result['iteration']} "
                f"target_patches="
                f"{args.tide_storage_compaction_target_patch_files}"
            )
            if is_final_batch:
                final_storage_maintenance_complete = True

        with torch.no_grad():
            if any(
                [
                    iteration <= save_iteration < iteration + args.bsz
                    for save_iteration in args.save_iterations
                ]
            ):
                utils.print_rank_0("\n[ITER {}] Saving End2end".format(iteration))
                end2end_timers.stop()
                end2end_timers.print_time(log_file, iteration + args.bsz)

                skip_msg = (
                    f"[ITER {iteration}] SKIP model save: pure SSD release "
                    "keeps the full parameter table out-of-core. Use "
                    "--checkpoint_iterations for resumable state, or "
                    "tools/export_pure_ssd_checkpoint_to_ply.py for PLY preview/export.\n"
                )
                utils.print_rank_0(skip_msg.rstrip())
                log_file.write(skip_msg)

                end2end_timers.start()

            # ------------------------------------------------------------------------
            # 2.10: Save training checkpoint (for resuming)
            # ------------------------------------------------------------------------
            if matched_checkpoint_iterations:
                end2end_timers.stop()
                checkpoint_iteration = matched_checkpoint_iterations[-1]
                save_folder = os.path.join(
                    scene.model_path,
                    "checkpoints",
                    str(checkpoint_iteration),
                )
                utils.print_rank_0(
                    f"\n[ITER {iteration}] Saving Pure SSD Checkpoint {checkpoint_iteration}"
                )
                log_file.write(
                    f"[ITER {iteration}] Saving Pure SSD Checkpoint {checkpoint_iteration}\n"
                )
                _run_synchronized_training_phase(
                    distributed_context,
                    phase="checkpoint writeback drain",
                    operation=storage_adapter.drain_cache_writebacks,
                )
                pure_ssd_checkpoint_mode = str(
                    getattr(args, "pure_ssd_checkpoint_mode", "incremental")
                ).lower()
                if distributed_enabled:
                    write_distributed_incremental_checkpoint(
                        context=distributed_context,
                        storage_adapter=storage_adapter,
                        gaussians=gaussians,
                        checkpoint_dir=save_folder,
                        iteration=checkpoint_iteration,
                        next_iteration=iteration + args.bsz,
                        args=args,
                        block_owner=args._tide_block_owner,
                        planner_state=checkpoint_planner_state,
                        global_optimizer_step=completed_optimizer_steps,
                        log_file=log_file,
                    )
                elif pure_ssd_checkpoint_mode == "snapshot":
                    write_pure_ssd_snapshot_checkpoint(
                        storage_adapter=storage_adapter,
                        gaussians=gaussians,
                        checkpoint_dir=save_folder,
                        iteration=checkpoint_iteration,
                        next_iteration=iteration + args.bsz,
                        args=args,
                        chunk_blocks=getattr(args, "pure_ssd_checkpoint_chunk_blocks", 256),
                        log_file=log_file,
                    )
                else:
                    write_pure_ssd_incremental_checkpoint(
                        storage_adapter=storage_adapter,
                        gaussians=gaussians,
                        checkpoint_dir=save_folder,
                        iteration=checkpoint_iteration,
                        next_iteration=iteration + args.bsz,
                        args=args,
                        log_file=log_file,
                    )
                prune_operation = None
                if not distributed_enabled or distributed_context.is_rank0:
                    prune_operation = lambda: prune_checkpoint_history(
                        scene.model_path,
                        keep_last=getattr(args, "pure_ssd_checkpoint_keep_last", 2),
                        log_file=log_file,
                    )
                _run_synchronized_training_phase(
                    distributed_context,
                    phase="checkpoint history prune",
                    operation=prune_operation,
                )
                distributed_context.barrier()
                end2end_timers.start()

        # ------------------------------------------------------------------------
        # 2.12: Iteration cleanup
        # ------------------------------------------------------------------------
        if storage_adapter is None:
            torch.cuda.synchronize()

        # Release camera image memory
        nvtx.range_push("Outer: camera_cleanup")
        for viewpoint_cam in batched_cameras:
            viewpoint_cam.original_image = None
        for viewpoint_cam in curvature_cameras:
            viewpoint_cam.original_image = None
        nvtx.range_pop()

        # End profiling range if active
        if args.nsys_profile:
            if (
                iteration >= args.nsys_profile_start_iter
                and iteration < args.nsys_profile_end_iter
            ):
                nvtx.range_pop()

        # Print timing statistics
        if utils.check_enable_python_timer():
            timers.stop("[iteration end2end]")
            timers.printTimers(iteration, mode="sum")

        # Dump CUDA memory trace if enabled
        if args.trace_cuda_mem:
            if (iteration % args.log_interval) == 1 or (
                iteration % args.densification_interval
            ) == 0:
                dump_name = args.log_folder + f"/trace_dump/iter={iteration}"
                torch.cuda.memory._dump_snapshot(filename=dump_name)
                torch.cuda.memory._record_memory_history(enabled=None)

        utils.memory_report("at the end of the iteration")
        log_file.flush()

    if optimizer_churn_tsv is not None and optimizer_churn_state['current_epoch'] is not None:
        _write_optimizer_churn_epoch(
            epoch_zero_based=optimizer_churn_state['current_epoch'],
            iteration_end=last_iteration if last_iteration is not None else opt_args.iterations,
        )
        optimizer_churn_tsv.close()
        optimizer_churn_tsv = None

    camera_prefetch_stats = camera_batch_prefetcher.get_stats()
    camera_batch_prefetcher.close()
    log_file.write(
        "[CAMERA PREFETCH] "
        f"batches={camera_prefetch_stats['batches']} "
        f"ready_hits={camera_prefetch_stats['ready_hits']} "
        f"wait_seconds={camera_prefetch_stats['wait_seconds']:.6f}\n"
    )
    if curvature_camera_batch_prefetcher is not None:
        curvature_prefetch_stats = curvature_camera_batch_prefetcher.get_stats()
        curvature_camera_batch_prefetcher.close()
        log_file.write(
            "[S2 CAMERA PREFETCH] "
            f"batches={curvature_prefetch_stats['batches']} "
            f"ready_hits={curvature_prefetch_stats['ready_hits']} "
            f"wait_seconds={curvature_prefetch_stats['wait_seconds']:.6f}\n"
        )

    # ============================================================================
    # STAGE 3: POST-TRAINING CLEANUP AND REPORTING
    # ============================================================================

    # Clean up CUDA resources
    del comm_stream

    # Print final timing statistics
    if opt_args.iterations not in args.save_iterations:
        end2end_timers.print_time(log_file, opt_args.iterations)

    # Log peak memory usage
    log_file.write(
        "Max Memory usage: {} GB.\n".format(
            torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        )
    )

    # Close progress bar and clean up scene
    progress_bar.close()
    mem_mon.close()

    if storage_adapter is not None:
        from strategies.tide_engine.engine import shutdown_double_buffer_gpu

        distributed_context.barrier()
        if args.tide_detailed_metrics:
            legacy_collector = getattr(
                gaussians, "_tide_legacy_cuda_metrics_collector", None
            )
            _run_synchronized_training_phase(
                distributed_context,
                phase="legacy metrics finalize",
                operation=(
                    None
                    if legacy_collector is None
                    else legacy_collector.finalize
                ),
            )
        if not final_storage_maintenance_complete:
            _run_synchronized_training_phase(
                distributed_context,
                phase="shutdown resident dirty flush",
                operation=storage_adapter.flush_resident_dirty,
            )
            compaction_result = run_compaction_maintenance(
                context=distributed_context,
                storage_adapter=storage_adapter,
                iteration=opt_args.iterations,
                periodic_iteration=None,
                target_patch_files=(
                    args.tide_storage_compaction_target_patch_files
                ),
                rank_concurrency=(
                    args.tide_storage_compaction_rank_concurrency
                ),
                emergency_free_gb=emergency_compaction_free_gb,
                forced_trigger="shutdown",
                flush_dirty_cache=True,
            )
            if compaction_result is not None:
                _run_synchronized_training_phase(
                    distributed_context,
                    phase="compaction metrics write",
                    operation=lambda: get_distributed_metrics_writer(
                        gaussians,
                        distributed_context,
                    ).write_compaction(compaction_result),
                )
        _run_synchronized_training_phase(
            distributed_context,
            phase="storage shutdown",
            operation=lambda: storage_adapter.shutdown(
                compact_storage=False
            ),
        )
        if args.tide_detailed_metrics:
            def finalize_distributed_metrics():
                metrics_writer = get_distributed_metrics_writer(
                    gaussians,
                    distributed_context,
                )
                metrics_writer.write_io_events(
                    storage_adapter.cache.drain_io_events()
                )
                metrics_writer.finalize_async_metrics()

            _run_synchronized_training_phase(
                distributed_context,
                phase="distributed metrics finalize",
                operation=finalize_distributed_metrics,
            )
        _run_synchronized_training_phase(
            distributed_context,
            phase="double buffer shutdown",
            operation=shutdown_double_buffer_gpu,
        )
        distributed_context.barrier()

    scene.clean_up()

    # Stop profiler if active
    if args.nsys_profile:
        torch.cuda.cudart().cudaProfilerStop()



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    args = parser.parse_args(sys.argv[1:])

    init_args(args)

    args = utils.get_args()
    distributed_context = DistributedContext.initialize(args)
    root_log_folder = args.log_folder

    if distributed_context.is_rank0:
        os.makedirs(root_log_folder, exist_ok=True)
        os.makedirs(args.model_path, exist_ok=True)
        serializable_args = {
            key: value for key, value in vars(args).items() if not key.startswith("_")
        }
        with open(os.path.join(root_log_folder, "args.json"), "w") as f:
            json.dump(serializable_args, f, default=str)
    distributed_context.barrier()

    if distributed_context.enabled and not distributed_context.is_rank0:
        args.log_folder = os.path.join(
            root_log_folder, f"rank_{distributed_context.rank}"
        )
    os.makedirs(args.log_folder, exist_ok=True)
    if args.trace_cuda_mem:
        os.makedirs(os.path.join(args.log_folder, "trace_dump"), exist_ok=True)

    log_file = open(
        os.path.join(args.log_folder, "python.log"),
        "a" if args.auto_start_checkpoint else "w",
    )
    utils.set_log_file(log_file)

    try:
        worker_quiet = args.quiet or (
            distributed_context.enabled and not distributed_context.is_rank0
        )
        safe_state(worker_quiet, log_file=log_file)
        print_all_args(args, log_file)

        p = psutil.Process()
        log_file.write(
            f"Initial pinned memory: {p.memory_info().shared / 1024 / 1024 / 1024} GB\n"
        )

        training(lp.extract(args), op.extract(args), pp.extract(args), args, log_file)
        utils.print_rank_0("\nTraining complete.")
    finally:
        log_file.flush()
        log_file.close()
        distributed_context.close()
