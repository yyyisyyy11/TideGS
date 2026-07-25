"""Gaussian-sharded, synchronously stepped TideGS batch execution."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np
import torch
import torch.nn as nn

import utils.general_utils as utils
from strategies.base_engine import torch_compiled_loss
from utils.distributed import DistributedContext

from .distributed_plan import DistributedBatchPlan
from .distributed_metrics import get_distributed_metrics_writer
from .engine import get_gpu_resident_optimizer
from .gsplat_backend import distributed_rasterize, projection_elapsed_ms


@dataclass
class DistributedResidentState:
    resident: Set[int] = field(default_factory=set)
    block_versions: Dict[int, int] = field(default_factory=dict)
    gpu_dirty_versions: Dict[int, int] = field(default_factory=dict)
    pending_commit_versions: Dict[int, int] = field(default_factory=dict)
    dirty_origin_iterations: Dict[int, int] = field(default_factory=dict)
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    def dirty_blocks(self) -> List[int]:
        with self._lock:
            return sorted(self.gpu_dirty_versions)

    def mark_dirty_blocks(
        self,
        block_ids: Iterable[int],
        *,
        iteration: int,
    ) -> None:
        with self._lock:
            for raw_block_id in block_ids:
                block_id = int(raw_block_id)
                version = int(self.block_versions.get(block_id, 0)) + 1
                self.block_versions[block_id] = version
                self.gpu_dirty_versions[block_id] = version
                self.dirty_origin_iterations[block_id] = int(iteration)

    def mark_blocks_written_back(self, block_ids: Iterable[int]) -> None:
        with self._lock:
            for raw_block_id in block_ids:
                block_id = int(raw_block_id)
                self.gpu_dirty_versions.pop(block_id, None)
                self.pending_commit_versions.pop(block_id, None)
                self.dirty_origin_iterations.pop(block_id, None)

    def set_resident_versions(
        self,
        block_ids: Iterable[int],
        versions: Dict[int, int],
    ) -> None:
        with self._lock:
            for raw_block_id in block_ids:
                block_id = int(raw_block_id)
                if (
                    block_id not in self.gpu_dirty_versions
                    and block_id not in self.pending_commit_versions
                ):
                    self.block_versions[block_id] = int(versions.get(block_id, 0))

    def versions_for_blocks(self, block_ids: Iterable[int]) -> Dict[int, int]:
        with self._lock:
            return {
                int(block_id): int(self.block_versions.get(int(block_id), 0))
                for block_id in block_ids
            }

    def latest_dirty_iteration(self, block_ids: Iterable[int]):
        with self._lock:
            origins = {
                self.dirty_origin_iterations[int(block_id)]
                for block_id in block_ids
                if int(block_id) in self.dirty_origin_iterations
            }
        return next(iter(origins)) if len(origins) == 1 else None

    def begin_writeback(self, block_ids: Iterable[int]) -> Dict[int, int]:
        with self._lock:
            versions = {}
            for raw_block_id in block_ids:
                block_id = int(raw_block_id)
                if block_id not in self.gpu_dirty_versions:
                    continue
                version = int(self.gpu_dirty_versions.pop(block_id))
                self.pending_commit_versions[block_id] = version
                versions[block_id] = version
            return versions

    def cancel_writeback(self, block_versions: Dict[int, int]) -> None:
        with self._lock:
            for block_id, version in block_versions.items():
                if self.pending_commit_versions.get(int(block_id)) == int(version):
                    self.pending_commit_versions.pop(int(block_id), None)
                    self.gpu_dirty_versions[int(block_id)] = int(version)

    def complete_writeback(
        self,
        block_ids: Iterable[int],
        block_versions: Dict[int, int],
    ) -> None:
        with self._lock:
            for raw_block_id in block_ids:
                block_id = int(raw_block_id)
                version = int(
                    block_versions.get(
                        block_id,
                        self.pending_commit_versions.get(block_id, -1),
                    )
                )
                if self.pending_commit_versions.get(block_id) == version:
                    self.pending_commit_versions.pop(block_id, None)
                    if block_id not in self.gpu_dirty_versions:
                        self.dirty_origin_iterations.pop(block_id, None)


def _resident_state(gaussians) -> DistributedResidentState:
    state = getattr(gaussians, "_tide_distributed_resident_state", None)
    if state is None:
        state = DistributedResidentState()
        gaussians._tide_distributed_resident_state = state
    return state


def _flush_blocks(storage_adapter, manager, state, block_ids: Sequence[int]) -> int:
    selected = sorted(
        set(int(block_id) for block_id in block_ids).intersection(
            state.dirty_blocks()
        )
    )
    if not selected:
        return 0
    block_versions = state.versions_for_blocks(selected)
    payload = manager.stage_updated_blocks(
        selected,
        block_versions=block_versions,
        origin_iteration=state.latest_dirty_iteration(selected),
    )
    if payload is None or set(payload.block_ids) != set(selected):
        staged = [] if payload is None else list(payload.block_ids)
        raise RuntimeError(f"Distributed writeback incomplete: expected={selected[:8]} staged={staged[:8]}")
    pending_versions = state.begin_writeback(selected)
    try:
        staged = storage_adapter.submit_cache_writeback(
            payload,
            bounds_managed_externally=True,
            on_complete=state.complete_writeback,
            on_error=state.cancel_writeback,
        )
    except Exception:
        state.cancel_writeback(pending_versions)
        raise
    if staged != len(selected):
        raise RuntimeError(
            f"Distributed writeback staged {staged}/{len(selected)} blocks"
        )
    return len(selected)


def _bind_working_set(gaussians, gpu_tensors: Dict[str, torch.Tensor]) -> None:
    gaussians._xyz = nn.Parameter(gpu_tensors["xyz"].requires_grad_(True))
    gaussians._scaling = nn.Parameter(gpu_tensors["scaling"].requires_grad_(True))
    gaussians._rotation = nn.Parameter(gpu_tensors["rotation"].requires_grad_(True))
    gaussians._opacity = nn.Parameter(gpu_tensors["opacity"].requires_grad_(True))
    gaussians._features_dc = nn.Parameter(gpu_tensors["features_dc"].requires_grad_(True))
    gaussians._features_rest = nn.Parameter(gpu_tensors["features_rest"].requires_grad_(True))
    manager = gaussians.gpu_working_set_manager
    manager.gpu_xyz = gaussians._xyz
    manager.gpu_scaling = gaussians._scaling
    manager.gpu_rotation = gaussians._rotation
    manager.gpu_opacity = gaussians._opacity
    manager.gpu_features_dc = gaussians._features_dc
    manager.gpu_features_rest = gaussians._features_rest


def _local_ids_for_blocks(manager, block_ids: Sequence[int]) -> torch.Tensor:
    ranges = []
    for block_id in sorted(set(int(value) for value in block_ids)):
        local_slice = manager.block_to_gpu_slice.get(block_id)
        if local_slice is None:
            continue
        ranges.append(
            torch.arange(
                int(local_slice.start),
                int(local_slice.stop),
                dtype=torch.long,
                device=manager.device,
            )
        )
    if not ranges:
        return torch.empty((0,), dtype=torch.long, device=manager.device)
    return torch.cat(ranges, dim=0)


def _active_leaves(gaussians, active_local_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    specs = {
        "xyz": gaussians._xyz,
        "scaling": gaussians._scaling,
        "rotation": gaussians._rotation,
        "opacity": gaussians._opacity,
        "features_dc": gaussians._features_dc,
        "features_rest": gaussians._features_rest,
    }
    return {
        name: tensor.detach().index_select(0, active_local_ids).contiguous().requires_grad_(True)
        for name, tensor in specs.items()
    }


def _touched_active_ids(meta_values: Sequence[Dict], active_count: int, device) -> torch.Tensor:
    ids = []
    for meta in meta_values:
        gaussian_ids = meta.get("gaussian_ids")
        if gaussian_ids is not None and gaussian_ids.numel() > 0:
            ids.append(gaussian_ids.detach().to(device=device, dtype=torch.long))
    if not ids:
        return torch.empty((0,), dtype=torch.long, device=device)
    touched = torch.unique(torch.cat(ids, dim=0), sorted=True)
    if int(touched.min()) < 0 or int(touched.max()) >= int(active_count):
        raise RuntimeError("gsplat returned a Gaussian id outside the local active shard")
    return touched


def _sync_bounds(
    *,
    context: DistributedContext,
    storage_adapter,
    manager,
    updated_blocks: Sequence[int],
) -> None:
    pending = manager.stage_block_bounds(list(updated_blocks))
    if pending is None:
        local_payload = ([], [])
    else:
        block_ids, bounds = pending.wait()
        if torch.is_tensor(bounds):
            bounds = bounds.detach().cpu().numpy()
        local_payload = (list(block_ids), np.asarray(bounds, dtype=np.float32).tolist())
    for block_ids, bounds in context.all_gather_object(local_payload):
        if block_ids:
            storage_adapter.update_block_bounds(block_ids, np.asarray(bounds, dtype=np.float32))


def train_distributed_tide_batch(
    *,
    gaussians,
    scene,
    batched_cameras,
    background,
    storage_adapter,
    plan: DistributedBatchPlan,
    context: DistributedContext,
):
    args = gaussians.args
    iteration = int(plan.iteration)
    batch_start = time.perf_counter()
    metrics_writer = get_distributed_metrics_writer(gaussians, context)
    local_counts = context.all_gather_object(len(batched_cameras))
    if len(set(int(value) for value in local_counts)) != 1:
        raise RuntimeError(f"Distributed gsplat requires equal camera counts, got {local_counts}")

    manager = gaussians.gpu_working_set_manager
    cache = storage_adapter.cache
    cache.set_foreground_iteration(iteration)
    cache_before = cache.get_stats()
    execution_before = dict(storage_adapter.execution_metrics)
    state = _resident_state(gaussians)
    target_resident = set(int(value) for value in plan.rank_resident_blocks[context.rank])
    evicted = state.resident - target_resident
    writeback_start = time.perf_counter()
    _flush_blocks(storage_adapter, manager, state, sorted(evicted))
    writeback_submit_ms = (time.perf_counter() - writeback_start) * 1000.0

    resident_load_start = time.perf_counter()
    gpu_tensors, retention_stats = manager.load_visible_blocks_with_retention(
        visible_block_ids=sorted(target_resident),
        active_blocks_ram=None,
        enable_retention=bool(state.resident),
        unified_params=None,
        block_reader=gaussians._block_reader,
        allow_gpu_hotspots=True,
    )
    resident_load_ms = (time.perf_counter() - resident_load_start) * 1000.0
    _bind_working_set(gaussians, gpu_tensors)
    cache_versions = storage_adapter.cache.get_block_versions(
        sorted(target_resident)
    )
    state.set_resident_versions(target_resident, cache_versions)
    state.resident = target_resident
    storage_adapter.bind_resident_writeback(state, manager)

    optimizer = get_gpu_resident_optimizer(gaussians, int(args.bsz))
    optimizer.set_resident_blocks(sorted(target_resident))

    active_blocks = sorted(
        target_resident.intersection(plan.rank_active_blocks[context.rank])
    )
    active_local_ids = _local_ids_for_blocks(manager, active_blocks)
    active_counts = context.all_gather_object(int(active_local_ids.numel()))
    if any(count == 0 for count in active_counts):
        raise RuntimeError(
            f"Every rank needs active owner Gaussians for distributed gsplat; "
            f"iteration={iteration} active_counts={active_counts}"
        )
    leaves = _active_leaves(gaussians, active_local_ids)
    scales = gaussians.scaling_activation(leaves["scaling"])
    rotations = gaussians.rotation_activation(leaves["rotation"])
    opacities = gaussians.opacity_activation(leaves["opacity"]).squeeze(-1)
    sh_coefficients = torch.cat(
        [leaves["features_dc"], leaves["features_rest"]], dim=1
    ).reshape(-1, 16, 3)

    image_width = int(utils.get_img_width())
    image_height = int(utils.get_img_height())
    microbatch = int(getattr(args, "tide_camera_microbatch", len(batched_cameras)))
    losses = []
    metas = []
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    forward_start.record()
    for start in range(0, len(batched_cameras), microbatch):
        cameras = batched_cameras[start:start + microbatch]
        rendered, _, meta = distributed_rasterize(
            means=leaves["xyz"],
            quats=rotations,
            scales=scales,
            opacities=opacities,
            sh_coefficients=sh_coefficients,
            cameras=cameras,
            width=image_width,
            height=image_height,
            sh_degree=gaussians.active_sh_degree,
            background=background,
            radius_clip=float(getattr(args, "radius_clip", 0.0)),
        )
        metas.append(meta)
        for local_index, camera in enumerate(cameras):
            image = rendered[local_index].permute(2, 0, 1).contiguous()
            losses.append(torch_compiled_loss(image, camera.original_image))
    forward_end.record()

    if not losses:
        raise RuntimeError("Distributed TideGS produced no local losses")
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    backward_start.record()
    torch.stack(losses).sum().backward()
    backward_end.record()

    touched_active = _touched_active_ids(metas, active_local_ids.numel(), active_local_ids.device)
    touched_local = active_local_ids.index_select(0, touched_active)
    sparse_grad_components = {}
    for name, leaf in leaves.items():
        if leaf.grad is None:
            grad = torch.zeros_like(leaf)
        else:
            grad = leaf.grad
        sparse_grad_components[name] = grad.index_select(0, touched_active).contiguous()
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    optimizer_start.record()
    step_stats = optimizer.step(
        iteration=iteration,
        gaussians=gaussians,
        sparse_grad_local_ids=touched_local,
        sparse_grad_components=sparse_grad_components,
    )
    optimizer_end.record()
    updated_blocks = list(step_stats.get("updated_block_ids", []))
    state.mark_dirty_blocks(updated_blocks, iteration=iteration)
    bounds_start = time.perf_counter()
    _sync_bounds(
        context=context,
        storage_adapter=storage_adapter,
        manager=manager,
        updated_blocks=updated_blocks,
    )
    bounds_sync_ms = (time.perf_counter() - bounds_start) * 1000.0
    barrier_start = time.perf_counter()
    context.barrier()
    barrier_ms = (time.perf_counter() - barrier_start) * 1000.0
    manager.prepare_for_retention()

    h2d_events = retention_stats.get("h2d_events")
    h2d_ms = (
        float(h2d_events[0].elapsed_time(h2d_events[1]))
        if h2d_events is not None
        else 0.0
    )
    cache_after = cache.get_stats()
    execution_after = dict(storage_adapter.execution_metrics)

    def cache_delta(name):
        return float(cache_after.get(name, 0)) - float(cache_before.get(name, 0))

    def execution_delta(name):
        return float(execution_after.get(name, 0)) - float(
            execution_before.get(name, 0)
        )

    projection_ms = sum(projection_elapsed_ms(meta) for meta in metas)
    forward_ms = float(forward_start.elapsed_time(forward_end))
    backward_ms = float(backward_start.elapsed_time(backward_end))
    optimizer_ms = float(optimizer_start.elapsed_time(optimizer_end))
    metrics_row = {
        "iteration": iteration,
        "local_cameras": len(batched_cameras),
        "global_active_blocks": len(plan.global_active_blocks),
        "global_resident_blocks": len(plan.global_resident_blocks),
        "rank_active_blocks": len(active_blocks),
        "rank_resident_blocks": len(target_resident),
        "cold_blocks": int(retention_stats.get("cold_count", 0)),
        "retained_blocks": int(retention_stats.get("hotspot_count", 0)),
        "gpu_slot_capacity_blocks": int(
            retention_stats.get("gpu_slot_capacity_blocks", 0)
        ),
        "gpu_slot_growth_blocks": int(
            retention_stats.get("gpu_slot_growth_blocks", 0)
        ),
        "touched_gaussians": int(touched_active.numel()),
        "block_cull_ms": float(plan.block_cull_ms),
        "plan_ms": float(plan.plan_ms),
        "writeback_submit_ms": writeback_submit_ms,
        "resident_load_ms": resident_load_ms,
        "ssd_foreground_wait_ms": float(
            retention_stats.get("foreground_read_ms", 0.0)
        ),
        "ssd_inflight_wait_ms": cache_delta("inflight_wait_time") * 1000.0,
        "cpu_materialize_ms": float(
            retention_stats.get("cpu_materialize_ms", 0.0)
        ),
        "ssd_urgent_read_blocks": cache_delta("urgent_storage_read_blocks"),
        "ssd_urgent_read_bytes": cache_delta("ssd_bytes_read_urgent"),
        "ssd_future_read_blocks": cache_delta("future_storage_read_blocks"),
        "ssd_future_read_bytes": cache_delta("ssd_bytes_read_future"),
        "prefetch_cpu_ms": max(
            0.0,
            cache_delta("future_materialize_time") * 1000.0,
        ),
        "prefetch_ssd_ms": cache_delta("future_storage_read_time") * 1000.0,
        "h2d_bytes": int(retention_stats.get("h2d_bytes", 0)),
        "h2d_ms": h2d_ms,
        "gsplat_forward_ms": forward_ms,
        "gaussian_projection_cull_ms": projection_ms,
        "backward_ms": backward_ms,
        "optimizer_ms": optimizer_ms,
        "train_ms": forward_ms + backward_ms + optimizer_ms,
        "bounds_sync_ms": bounds_sync_ms,
        "barrier_ms": barrier_ms,
        "gpu_d2h_ms": execution_delta("background_gpu_d2h_time_ms"),
        "cpu_cache_commit_ms": execution_delta(
            "background_cache_commit_time_ms"
        ),
        "ssd_write_blocks": cache_delta("async_flush_blocks")
        + cache_delta("sync_flush_blocks"),
        "ssd_write_bytes": cache_delta("ssd_bytes_written_async")
        + cache_delta("ssd_bytes_written_sync"),
        "ssd_write_service_ms": (
            cache_delta("async_flush_time") + cache_delta("sync_flush_time")
        )
        * 1000.0,
        "batch_total_ms": (time.perf_counter() - batch_start) * 1000.0,
    }
    metrics_writer.write_io_events(cache.drain_io_events())
    metrics_writer.write_batch(metrics_row)

    gaussians._tide_distributed_last_step = {
        **step_stats,
        "rank": context.rank,
        "local_cameras": len(batched_cameras),
        "resident_blocks": len(target_resident),
        "active_blocks": len(active_blocks),
    }
    detached_losses = [loss.detach() for loss in losses]
    return detached_losses, list(range(len(batched_cameras))), float(active_local_ids.numel())
