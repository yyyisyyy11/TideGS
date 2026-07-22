"""Gaussian-sharded, synchronously stepped TideGS batch execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np
import torch
import torch.nn as nn

import utils.general_utils as utils
from strategies.base_engine import torch_compiled_loss
from utils.distributed import DistributedContext

from .distributed_plan import DistributedBatchPlan
from .engine import get_gpu_resident_optimizer
from .gsplat_backend import distributed_rasterize


@dataclass
class DistributedResidentState:
    resident: Set[int] = field(default_factory=set)
    dirty: Set[int] = field(default_factory=set)

    def dirty_blocks(self) -> List[int]:
        return sorted(self.dirty)

    def mark_dirty_blocks(self, block_ids: Iterable[int]) -> None:
        self.dirty.update(int(block_id) for block_id in block_ids)

    def mark_blocks_written_back(self, block_ids: Iterable[int]) -> None:
        self.dirty.difference_update(int(block_id) for block_id in block_ids)


def _resident_state(gaussians) -> DistributedResidentState:
    state = getattr(gaussians, "_tide_distributed_resident_state", None)
    if state is None:
        state = DistributedResidentState()
        gaussians._tide_distributed_resident_state = state
    return state


def _flush_blocks(storage_adapter, manager, state, block_ids: Sequence[int]) -> int:
    selected = sorted(set(int(block_id) for block_id in block_ids).intersection(state.dirty))
    if not selected:
        return 0
    payload = manager.stage_updated_blocks(selected)
    if payload is None or set(payload.block_ids) != set(selected):
        staged = [] if payload is None else list(payload.block_ids)
        raise RuntimeError(f"Distributed writeback incomplete: expected={selected[:8]} staged={staged[:8]}")
    staged = storage_adapter.submit_cache_writeback(
        payload, bounds_managed_externally=True
    )
    if staged != len(selected):
        raise RuntimeError(
            f"Distributed writeback staged {staged}/{len(selected)} blocks"
        )
    state.mark_blocks_written_back(selected)
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
    local_counts = context.all_gather_object(len(batched_cameras))
    if len(set(int(value) for value in local_counts)) != 1:
        raise RuntimeError(f"Distributed gsplat requires equal camera counts, got {local_counts}")

    manager = gaussians.gpu_working_set_manager
    state = _resident_state(gaussians)
    target_resident = set(int(value) for value in plan.rank_resident_blocks[context.rank])
    evicted = state.resident - target_resident
    _flush_blocks(storage_adapter, manager, state, sorted(evicted))

    gpu_tensors, _ = manager.load_visible_blocks_with_retention(
        visible_block_ids=sorted(target_resident),
        active_blocks_ram=None,
        enable_retention=bool(state.resident),
        unified_params=None,
        block_reader=gaussians._block_reader,
        allow_gpu_hotspots=True,
    )
    _bind_working_set(gaussians, gpu_tensors)
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

    if not losses:
        raise RuntimeError("Distributed TideGS produced no local losses")
    torch.stack(losses).sum().backward()

    touched_active = _touched_active_ids(metas, active_local_ids.numel(), active_local_ids.device)
    touched_local = active_local_ids.index_select(0, touched_active)
    sparse_grad_components = {}
    for name, leaf in leaves.items():
        if leaf.grad is None:
            grad = torch.zeros_like(leaf)
        else:
            grad = leaf.grad
        sparse_grad_components[name] = grad.index_select(0, touched_active).contiguous()
    step_stats = optimizer.step(
        iteration=iteration,
        gaussians=gaussians,
        sparse_grad_local_ids=touched_local,
        sparse_grad_components=sparse_grad_components,
    )
    updated_blocks = list(step_stats.get("updated_block_ids", []))
    state.mark_dirty_blocks(updated_blocks)
    _sync_bounds(
        context=context,
        storage_adapter=storage_adapter,
        manager=manager,
        updated_blocks=updated_blocks,
    )
    context.barrier()
    manager.prepare_for_retention()

    gaussians._tide_distributed_last_step = {
        **step_stats,
        "rank": context.rank,
        "local_cameras": len(batched_cameras),
        "resident_blocks": len(target_resident),
        "active_blocks": len(active_blocks),
    }
    detached_losses = [loss.detach() for loss in losses]
    return detached_losses, list(range(len(batched_cameras))), float(active_local_ids.numel())
