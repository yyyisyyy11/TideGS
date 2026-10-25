"""Gaussian-sharded, synchronously stepped TideGS batch execution."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

import utils.general_utils as utils
from strategies.base_engine import torch_compiled_loss
from utils.distributed import DistributedContext
from utils.pipeline_mode import normalize_pipeline_mode
from utils.tide_trace import tide_range

from .distributed_plan import DistributedBatchPlan
from .micro_pipeline_schedule import MICRO_PIPELINE_SCHEDULE_VERSION, run_micro_pipeline_windows
from .distributed_metrics import (
    GRAD_METRIC_VALUE_FIELDS,
    cuda_interval_union_ms,
    get_distributed_metrics_writer,
)
from .gradient_sparsity import (
    build_gradient_sparsity_records,
    collect_gradient_sparsity,
    touched_component_rows as _touched_component_rows,
)
from .gradient_schema import GRADIENT_COMPONENTS as _LEAF_NAMES
from .engine import get_gpu_resident_optimizer
from .gsplat_backend import (
    PROJECTION_TIMING_FIX,
    distributed_rasterize,
    distributed_rasterization_pipeline,
    prepare_distributed_camera_layout,
    projection_elapsed_ms,
)
from .sophia_tr_curvature import (
    build_fused_3dgs2_curvature_residuals,
    estimate_seeded_curvature_sample,
)
from .sophia_tr_math import (
    resolve_curvature_schedule,
)


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

    def origins_for_blocks(self, block_ids: Iterable[int]) -> Dict[int, int]:
        with self._lock:
            return {
                int(block_id): int(self.dirty_origin_iterations[int(block_id)])
                for block_id in block_ids
                if int(block_id) in self.dirty_origin_iterations
            }

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
                    current_dirty = self.gpu_dirty_versions.get(int(block_id))
                    if current_dirty is None or int(current_dirty) < int(version):
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
    with tide_range("tide.cpu.resident.evict.stage"):
        payload = manager.stage_updated_blocks(
            selected,
            block_versions=block_versions,
            block_origin_iterations=state.origins_for_blocks(selected),
        )
    if payload is None or set(payload.block_ids) != set(selected):
        staged = [] if payload is None else list(payload.block_ids)
        raise RuntimeError(f"Distributed writeback incomplete: expected={selected[:8]} staged={staged[:8]}")
    pending_versions = state.begin_writeback(selected)
    try:
        with tide_range("tide.cpu.resident.evict.submit"):
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
    block_slices = []
    for block_id in set(int(value) for value in block_ids):
        local_slice = manager.block_to_gpu_slice.get(block_id)
        if local_slice is not None:
            block_slices.append((int(local_slice.start), local_slice))
    block_slices.sort(key=lambda item: item[0])

    ranges = []
    for _, local_slice in block_slices:
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


def _zero_opacity_sentinel_leaves(gaussians) -> Dict[str, torch.Tensor]:
    """Return one collective-only Gaussian that contributes exactly zero opacity."""

    specs = {
        "xyz": gaussians._xyz,
        "opacity": gaussians._opacity,
        "scaling": gaussians._scaling,
        "rotation": gaussians._rotation,
        "features_dc": gaussians._features_dc,
        "features_rest": gaussians._features_rest,
    }
    leaves = {}
    for name, tensor in specs.items():
        value = tensor.detach().new_zeros((1, *tensor.shape[1:]))
        if name == "opacity":
            value.fill_(-torch.inf)
        elif name == "rotation":
            value.reshape(1, -1)[:, 0] = 1.0
        leaves[name] = value.contiguous().requires_grad_(True)
    return leaves


def _transformed_leaves(gaussians, leaves: Dict[str, torch.Tensor]):
    scales = gaussians.scaling_activation(leaves["scaling"])
    rotations = gaussians.rotation_activation(leaves["rotation"])
    opacities = gaussians.opacity_activation(leaves["opacity"]).squeeze(-1)
    sh_coefficients = torch.cat(
        [leaves["features_dc"], leaves["features_rest"]], dim=1
    ).reshape(-1, 16, 3)
    return scales, rotations, opacities, sh_coefficients


def _camera_global_ids(cameras) -> List[int]:
    result = []
    for camera in cameras or []:
        if not hasattr(camera, "global_idx"):
            raise ValueError("Distributed cameras must expose global_idx")
        result.append(int(camera.global_idx))
    return result


def _error_description(error: Optional[BaseException]) -> Optional[str]:
    if error is None:
        return None
    return f"{type(error).__name__}: {error}"


def _synchronize_rank_errors(
    context: DistributedContext,
    *,
    phase: str,
    error: Optional[BaseException],
    group=None,
) -> None:
    # A peer may be stuck in a CUDA-to-CPU split conversion inside C1 and
    # never reach this Gloo checkpoint. Publish failure independently first.
    controller = getattr(context, "micro_pipeline_error_controller", None) if group is not None else None
    if controller is not None and error is not None:
        controller.fail(phase, error)
    payload = {
        "rank": int(context.rank),
        "error": _error_description(error),
    }
    if controller is not None:
        with controller.operation(phase):
            statuses = context.all_gather_object(payload, group=group)
    elif group is None:
        statuses = context.all_gather_object(payload)
    else:
        statuses = context.all_gather_object(payload, group=group)
    failures = [
        f"rank={status.get('rank', index)} {status['error']}"
        for index, status in enumerate(statuses)
        if status.get("error")
    ]
    if not failures:
        return
    synchronized = RuntimeError(
        f"Distributed TideGS {phase} failed: " + "; ".join(failures)
    )
    if error is not None:
        raise synchronized from error
    raise synchronized


def _validate_distributed_batch_contract(
    *,
    args,
    batched_cameras,
    curvature_cameras,
    plan: DistributedBatchPlan,
    context: DistributedContext,
    optimizer_step: Optional[int],
) -> Tuple[str, int, bool, int, int]:
    local_error = None
    contract = None
    try:
        algorithm = str(
            getattr(args, "paper_optimizer_algorithm", "adam")
        ).lower()
        if algorithm not in {"adam", "3dgs2_tr"}:
            raise ValueError(f"Unsupported resident optimizer: {algorithm!r}")
        interval = int(getattr(args, "paper_sophia_curvature_interval", 10))
        optimizer_step, scheduled_curvature = resolve_curvature_schedule(
            iteration=int(plan.iteration),
            batch_size=int(args.bsz),
            interval=interval,
            optimizer_step=optimizer_step,
        )
        curvature_due = algorithm == "3dgs2_tr" and scheduled_curvature
        s1_ids = _camera_global_ids(batched_cameras)
        s2_ids = _camera_global_ids(curvature_cameras)
        expected_s1 = [int(value) for value in plan.rank_s1_camera_ids[context.rank]]
        expected_s2 = [int(value) for value in plan.rank_s2_camera_ids[context.rank]]
        if s1_ids != expected_s1:
            raise ValueError(
                f"rank {context.rank} S1 camera schedule mismatch: "
                f"actual={s1_ids}, expected={expected_s1}"
            )
        if s2_ids != expected_s2:
            raise ValueError(
                f"rank {context.rank} S2 camera schedule mismatch: "
                f"actual={s2_ids}, expected={expected_s2}"
            )
        if curvature_due:
            if not s2_ids or len(s2_ids) != len(s1_ids):
                raise ValueError("A due curvature step requires S2 to match S1 size")
        elif s2_ids or plan.global_s2_camera_ids:
            raise ValueError("S2 cameras are only valid on a due 3DGS2-TR step")
        microbatch = int(
            getattr(args, "tide_camera_microbatch", len(batched_cameras))
        )
        if microbatch <= 0 or not s1_ids or len(s1_ids) % microbatch:
            raise ValueError(
                "Distributed local S1 count must be positive and divisible by "
                "tide_camera_microbatch"
            )
        if s2_ids and len(s2_ids) % microbatch:
            raise ValueError(
                "Distributed local S2 count must be divisible by "
                "tide_camera_microbatch"
            )
        sample_count = int(
            getattr(args, "paper_sophia_hutchinson_samples", 1)
        )
        if sample_count <= 0:
            raise ValueError("paper_sophia_hutchinson_samples must be positive")
        pipeline_mode = normalize_pipeline_mode(getattr(args, "tide_forward_pipeline", "off"))
        pipeline_buffers = int(getattr(args, "tide_pipeline_buffer_count", 2))
        if pipeline_mode == "micro" and pipeline_buffers < 2:
            raise ValueError("Micro Pipeline buffer_count must be at least 2")
        contract = {
            "algorithm": algorithm,
            "iteration": int(plan.iteration),
            "optimizer_step": optimizer_step,
            "curvature_due": curvature_due,
            "s1_count": len(s1_ids),
            "s2_count": len(s2_ids),
            "microbatch": microbatch,
            "sample_count": sample_count,
            "pipeline_mode": pipeline_mode,
            "pipeline_buffer_count": pipeline_buffers if pipeline_mode == "micro" else None,
            "global_s1": tuple(int(value) for value in plan.global_s1_camera_ids),
            "global_s2": tuple(int(value) for value in plan.global_s2_camera_ids),
        }
    except BaseException as error:
        local_error = error

    statuses = context.all_gather_object(
        {
            "rank": int(context.rank),
            "error": _error_description(local_error),
            "contract": contract,
        }
    )
    failures = [
        f"rank={status.get('rank', index)} {status['error']}"
        for index, status in enumerate(statuses)
        if status.get("error")
    ]
    if failures:
        synchronized = RuntimeError(
            "Distributed TideGS batch contract failed: " + "; ".join(failures)
        )
        if local_error is not None:
            raise synchronized from local_error
        raise synchronized
    reference = statuses[0]["contract"]
    if any(status["contract"] != reference for status in statuses[1:]):
        raise RuntimeError(
            "Distributed ranks disagree on S1/S2 counts, optimizer or pipeline schedule: "
            f"{[status['contract'] for status in statuses]}"
        )
    return (
        str(contract["algorithm"]),
        int(contract["optimizer_step"]),
        bool(contract["curvature_due"]),
        int(contract["microbatch"]),
        int(contract["sample_count"]),
    )


def _mark_projection_cull_survivors(
    survivor_mask: torch.Tensor,
    meta: Dict[str, object],
    active_count: int,
) -> None:
    """Union pre-all-to-all projection survivors into owner-local rows."""

    if int(active_count) == 0:
        return
    owner_mask = meta.get("_tide_owner_projection_survivor_mask")
    if not torch.is_tensor(owner_mask):
        raise RuntimeError(
            "Gradient-zero profiling requires gsplat owner projection survivor mask"
        )
    owner_mask = owner_mask.reshape(-1)
    if owner_mask.dtype != torch.bool or owner_mask.numel() != int(active_count):
        raise RuntimeError(
            "gsplat owner projection survivor mask has invalid shape or dtype: "
            f"expected=({int(active_count)},) bool, "
            f"actual={tuple(owner_mask.shape)} {owner_mask.dtype}"
        )
    survivor_mask |= owner_mask


def _merge_owner_tile_survivors(
    survivor_mask: torch.Tensor,
    tile_counts: torch.Tensor,
    meta: Dict[str, object],
    *,
    owner_active_rows: int,
    participation_rows: int,
    uses_sentinel: bool,
) -> None:
    """Union real owner rows while excluding collective-only sentinels."""

    owner_mask = meta.get("_tide_owner_tile_survivor_mask")
    expected_rows = participation_rows if uses_sentinel else owner_active_rows
    if (
        not torch.is_tensor(owner_mask)
        or owner_mask.dtype != torch.bool
        or owner_mask.numel() != expected_rows
    ):
        raise RuntimeError(
            "gsplat owner tile-survivor mask has invalid shape or dtype"
        )
    counts = meta.get("_tide_tile_mask_counts")
    if (
        not torch.is_tensor(counts)
        or counts.dtype != torch.long
        or counts.numel() != 5
    ):
        raise RuntimeError("gsplat tile-mask counters have invalid shape or dtype")
    if uses_sentinel:
        if owner_active_rows != 0:
            raise RuntimeError("tile-mask sentinel requires an empty owner rank")
        return
    survivor_mask |= owner_mask.reshape(-1)
    tile_counts.add_(counts.reshape(-1))


def _sync_bounds(
    *,
    context: DistributedContext,
    storage_adapter,
    manager,
    updated_blocks: Sequence[int],
) -> None:
    local_error = None
    local_payload = ([], [])
    try:
        with tide_range("tide.cpu.bounds.gpu_submit"):
            pending = manager.stage_block_bounds(list(updated_blocks))
        if pending is not None:
            with tide_range("tide.wait.bounds.d2h"):
                block_ids, bounds = pending.wait()
            with tide_range("tide.cpu.bounds.serialize"):
                if torch.is_tensor(bounds):
                    bounds = bounds.detach().cpu().numpy()
                local_payload = (
                    list(block_ids),
                    np.asarray(bounds, dtype=np.float32).tolist(),
                )
    except BaseException as error:
        local_error = error
    with tide_range("tide.comm.bounds.prepare_status"):
        _synchronize_rank_errors(
            context,
            phase="bounds preparation",
            error=local_error,
        )
    with tide_range("tide.comm.bounds.gather"):
        gathered_payloads = context.all_gather_object(local_payload)
    local_error = None
    try:
        with tide_range("tide.cpu.bounds.apply"):
            for block_ids, bounds in gathered_payloads:
                if block_ids:
                    storage_adapter.update_block_bounds(
                        block_ids,
                        np.asarray(bounds, dtype=np.float32),
                    )
    except BaseException as error:
        local_error = error
    with tide_range("tide.comm.bounds.apply_status"):
        _synchronize_rank_errors(
            context,
            phase="bounds application",
            error=local_error,
        )


def train_distributed_tide_batch(
    *,
    gaussians,
    scene,
    batched_cameras,
    curvature_cameras=None,
    optimizer_step=None,
    background,
    storage_adapter,
    plan: DistributedBatchPlan,
    context: DistributedContext,
):
    del scene
    args = gaussians.args
    iteration = int(plan.iteration)
    batch_start = time.perf_counter()
    metrics_writer = get_distributed_metrics_writer(gaussians, context)
    timeline_enabled = bool(metrics_writer.detailed_timeline_enabled)
    (
        optimizer_algorithm,
        optimizer_step,
        curvature_due,
        microbatch,
        hutchinson_samples,
    ) = _validate_distributed_batch_contract(
        args=args,
        batched_cameras=batched_cameras,
        curvature_cameras=curvature_cameras,
        plan=plan,
        context=context,
        optimizer_step=optimizer_step,
    )
    if timeline_enabled:
        torch.cuda.reset_peak_memory_stats()

    manager = gaussians.gpu_working_set_manager
    cache = storage_adapter.cache
    setup_error = None
    try:
        cache.set_foreground_iteration(iteration)
        cache_before = cache.get_stats()
        state = _resident_state(gaussians)
        target_resident = set(
            int(value) for value in plan.rank_resident_blocks[context.rank]
        )
        evicted = state.resident - target_resident
        writeback_start = time.perf_counter()
        with tide_range("tide.cpu.resident.evict"):
            _flush_blocks(storage_adapter, manager, state, sorted(evicted))
        writeback_submit_ms = (time.perf_counter() - writeback_start) * 1000.0

        resident_load_start = time.perf_counter()
        with tide_range("tide.cpu.resident.load"):
            gpu_tensors, retention_stats = manager.load_visible_blocks_with_retention(
                visible_block_ids=sorted(target_resident),
                active_blocks_ram=None,
                enable_retention=bool(state.resident),
                unified_params=None,
                block_reader=gaussians._block_reader,
                allow_gpu_hotspots=True,
                capture_timeline=timeline_enabled,
            )
        resident_load_ms = (time.perf_counter() - resident_load_start) * 1000.0
        with tide_range("tide.cpu.resident.bind"):
            _bind_working_set(gaussians, gpu_tensors)
            cache_versions = storage_adapter.cache.get_block_versions(
                sorted(target_resident)
            )
            state.set_resident_versions(target_resident, cache_versions)
            state.resident = target_resident
            storage_adapter.bind_resident_writeback(state, manager)

        with tide_range("tide.cpu.resident.active_leaves"):
            optimizer = get_gpu_resident_optimizer(gaussians, int(args.bsz))
            optimizer.set_resident_blocks(sorted(target_resident))
            active_blocks = sorted(
                target_resident.intersection(
                    plan.rank_union_active_blocks[context.rank]
                )
            )
            gradient_active_blocks = sorted(
                target_resident.intersection(
                    plan.rank_gradient_active_blocks[context.rank]
                )
            )
            curvature_active_blocks = sorted(
                target_resident.intersection(
                    plan.rank_curvature_active_blocks[context.rank]
                )
            )
            active_local_ids = _local_ids_for_blocks(manager, active_blocks)
            owner_active_rows = int(active_local_ids.numel())
            owner_rows_plan = getattr(plan, "rank_owner_active_rows", None)
            if owner_rows_plan is not None and owner_active_rows != int(
                owner_rows_plan[context.rank]
            ):
                raise RuntimeError(
                    f"rank {context.rank} owner active row mismatch: "
                    f"actual={owner_active_rows}, planned={owner_rows_plan[context.rank]}"
                )
            uses_sentinel = owner_active_rows == 0
            leaves = (
                _zero_opacity_sentinel_leaves(gaussians)
                if uses_sentinel
                else _active_leaves(gaussians, active_local_ids)
            )
            participation_rows = int(leaves["xyz"].shape[0])
            participation_plan = getattr(plan, "rank_participation_rows", None)
            if participation_plan is not None and participation_rows != int(
                participation_plan[context.rank]
            ):
                raise RuntimeError(
                    f"rank {context.rank} collective participation row mismatch: "
                    f"actual={participation_rows}, "
                    f"planned={participation_plan[context.rank]}"
                )
    except BaseException as error:
        setup_error = error
    with tide_range("tide.comm.resident.status"):
        _synchronize_rank_errors(
            context,
            phase="resident preparation",
            error=setup_error,
        )

    image_width = int(utils.get_img_width())
    image_height = int(utils.get_img_height())
    microbatch_count = len(batched_cameras) // microbatch
    detached_losses = []
    projection_event_pairs = []
    forward_ranges = []
    loss_ranges = []
    backward_ranges = []
    curvature_forward_ranges = []
    curvature_vjp_ranges = []
    probe_device = leaves["xyz"].device
    pipeline_mode = normalize_pipeline_mode(getattr(args, "tide_forward_pipeline", "off"))
    args.tide_forward_pipeline = pipeline_mode
    micro_pipeline = None
    micro_pipeline_nccl_stream = None
    micro_pipeline_layout = None
    micro_pipeline_error_group = getattr(context, "error_group", None)
    micro_pipeline_error_controller = getattr(context, "micro_pipeline_error_controller", None)
    micro_pipeline_records = []

    def _micro_pipeline_run(phase, function, *positional, **keywords):
        if micro_pipeline_error_controller is None:
            raise RuntimeError("Micro Pipeline requires an initialized CPU error controller")
        with micro_pipeline_error_controller.operation(phase):
            return function(*positional, **keywords)

    if pipeline_mode == "micro":
        with tide_range("tide.cpu.micro.worker.prepare"):
            existing_worker = getattr(context, "pipeline_comm_worker", None)
            if existing_worker is None:
                micro_pipeline_nccl_stream = torch.cuda.Stream(
                    device=probe_device,
                    priority=int(getattr(args, "tide_nccl_stream_priority", -1)),
                )
            else:
                micro_pipeline_nccl_stream = existing_worker.comm_stream
            micro_pipeline_worker = context.ensure_pipeline_comm_worker(
                buffer_count=int(getattr(args, "tide_pipeline_buffer_count", 2)),
                comm_stream=micro_pipeline_nccl_stream,
                require_tile_mask=(
                    str(getattr(args, "tide_tile_contribution_mode", "off")).lower() != "off"
                ),
            )
            micro_pipeline_nccl_stream = micro_pipeline_worker.comm_stream
            micro_pipeline = distributed_rasterization_pipeline(
                compute_stream=torch.cuda.current_stream(leaves["xyz"].device),
                comm_stream=micro_pipeline_nccl_stream,
                buffer_count=int(getattr(args, "tide_pipeline_buffer_count", 2)),
                capture_timeline=timeline_enabled,
                comm_worker=micro_pipeline_worker,
                batch_index=iteration,
            )
        layout_error = None
        batch_viewmats = None
        batch_Ks = None
        try:
            with tide_range("tide.cpu.micro.layout.local"):
                batch_viewmats = torch.stack(
                    [camera.world_view_transform.transpose(0, 1) for camera in batched_cameras],
                    dim=0,
                )
                batch_Ks = torch.stack([camera.K for camera in batched_cameras], dim=0)
                if batch_viewmats.ndim != 3 or batch_viewmats.shape[-2:] != (4, 4):
                    raise ValueError("Micro Pipeline camera viewmats must have shape [C,4,4]")
                if batch_Ks.ndim != 3 or batch_Ks.shape[-2:] != (3, 3):
                    raise ValueError("Micro Pipeline camera intrinsics must have shape [C,3,3]")
                if (
                    batch_viewmats.device != leaves["xyz"].device
                    or batch_Ks.device != leaves["xyz"].device
                ):
                    raise ValueError("Micro Pipeline camera tensors must be on the Gaussian CUDA device")
        except BaseException as error:
            layout_error = error
            micro_pipeline.abort()
        with tide_range("tide.comm.micro.layout_status"):
            _synchronize_rank_errors(
                context,
                phase="S1 Micro Pipeline local camera layout preparation",
                error=layout_error,
                group=getattr(context, "error_group", None),
            )
        layout_error = None
        try:
            with tide_range("tide.comm.micro.layout"):
                micro_pipeline_layout = _micro_pipeline_run(
                    "Micro Pipeline batch camera layout", prepare_distributed_camera_layout,
                    means=leaves["xyz"],
                    viewmats=batch_viewmats,
                    Ks=batch_Ks,
                    microbatch=microbatch,
                )
        except BaseException as error:
            layout_error = error
            micro_pipeline.abort()
        with tide_range("tide.comm.micro.layout_status"):
            _synchronize_rank_errors(
                context,
                phase="S1 Micro Pipeline batch camera layout collective",
                error=layout_error,
                group=getattr(context, "error_group", None),
            )
    grad_zero_sample_due = metrics_writer.should_write_grad_zero(optimizer_step)
    tile_contribution_mode = str(
        getattr(args, "tide_tile_contribution_mode", "off")
    ).lower()
    tile_alpha_threshold = float(
        getattr(args, "tide_tile_alpha_threshold", 1.0 / 255.0)
    )
    projection_cull_survivor_mask = (
        torch.zeros((owner_active_rows,), dtype=torch.bool, device=probe_device)
        if grad_zero_sample_due
        else None
    )
    tile_survivor_mask = (
        torch.zeros((owner_active_rows,), dtype=torch.bool, device=probe_device)
        if grad_zero_sample_due
        else None
    )
    tile_mask_counts = (
        torch.zeros((5,), dtype=torch.long, device=probe_device)
        if grad_zero_sample_due
        else None
    )
    slot_capacity_blocks = int(
        retention_stats.get("gpu_slot_capacity_blocks", 0)
    )

    def camera_uids(cameras) -> List[int]:
        if cameras is None:
            return []
        return [
            int(getattr(camera, "uid", getattr(camera, "global_idx", -1)))
            for camera in cameras
        ]

    def debug_projection_input(
        *,
        phase: str,
        microbatch_index: int,
        cameras,
        scales: torch.Tensor,
        opacities: torch.Tensor,
    ) -> None:
        local_cameras = int(len(cameras))
        global_cameras = local_cameras * int(context.world_size)
        pair_upper_bound = participation_rows * global_cameras
        if pair_upper_bound > np.iinfo(np.int32).max:
            raise RuntimeError(
                "gsplat packed projection int32 capacity exceeded: "
                f"rank={context.rank} phase={phase} "
                f"gaussians={participation_rows} "
                f"global_cameras={global_cameras} pairs={pair_upper_bound}"
            )
        if int(optimizer_step) > 1 or microbatch_index != 0:
            return
        free_bytes, total_bytes = torch.cuda.mem_get_info(probe_device)
        with torch.no_grad():
            finite_xyz = bool(torch.isfinite(leaves["xyz"]).all().item())
            finite_scales = bool(torch.isfinite(scales).all().item())
            finite_opacities = bool(torch.isfinite(opacities).all().item())
            scale_min = float(scales.min().item())
            scale_max = float(scales.max().item())
            opacity_min = float(opacities.min().item())
            opacity_max = float(opacities.max().item())
        print(
            "[GSPROJ INPUT] "
            f"rank={context.rank} step={optimizer_step} phase={phase} "
            f"microbatch={microbatch_index} local_cameras={local_cameras} "
            f"global_cameras={global_cameras} active_blocks={len(active_blocks)} "
            f"resident_blocks={len(target_resident)} gaussians={participation_rows} "
            f"pair_upper_bound={pair_upper_bound} "
            f"finite_xyz={int(finite_xyz)} finite_scales={int(finite_scales)} "
            f"finite_opacities={int(finite_opacities)} "
            f"scale_range=[{scale_min:.9g},{scale_max:.9g}] "
            f"opacity_range=[{opacity_min:.9g},{opacity_max:.9g}] "
            f"allocated_gib={torch.cuda.memory_allocated(probe_device) / (1 << 30):.3f} "
            f"reserved_gib={torch.cuda.memory_reserved(probe_device) / (1 << 30):.3f} "
            f"free_gib={free_bytes / (1 << 30):.3f} "
            f"total_gib={total_bytes / (1 << 30):.3f} "
            f"camera_uids={camera_uids(cameras)}",
            flush=True,
        )

    def write_memory_point(
        *,
        phase: str,
        microbatch_index: int | None = None,
        cameras=None,
        error: BaseException | None = None,
    ) -> None:
        if not timeline_enabled:
            return
        point = {
            "active_gaussians": owner_active_rows,
            "collective_participation_gaussians": participation_rows,
            "uses_zero_opacity_sentinel": int(uses_sentinel),
            "rank_active_blocks": int(len(active_blocks)),
            "rank_resident_blocks": int(len(target_resident)),
            "gpu_slot_capacity_blocks": slot_capacity_blocks,
        }
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(probe_device)
            point.update(
                {
                    "cuda_allocated_bytes": int(
                        torch.cuda.memory_allocated(probe_device)
                    ),
                    "cuda_reserved_bytes": int(
                        torch.cuda.memory_reserved(probe_device)
                    ),
                    "cuda_peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(probe_device)
                    ),
                    "cuda_peak_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(probe_device)
                    ),
                    "cuda_free_bytes": int(free_bytes),
                    "cuda_total_bytes": int(total_bytes),
                }
            )
        except Exception as snapshot_error:
            # Never hide the original CUDA OOM because a diagnostic query
            # itself failed after the allocator entered an error state.
            point["snapshot_error"] = (
                f"{type(snapshot_error).__name__}: {snapshot_error}"
            )
        if error is not None:
            point["status"] = "oom"
            point["error_type"] = type(error).__name__
            point["error_message"] = str(error)
        metrics_writer.write_memory_point(
            iteration=iteration,
            phase=phase,
            microbatch_index=microbatch_index,
            microbatch_count=microbatch_count,
            camera_uids=camera_uids(cameras),
            **point,
        )

    write_memory_point(phase="batch_start")
    curvature_accumulators = {
        name: torch.zeros_like(leaves[name]) for name in _LEAF_NAMES
    }
    if curvature_due:
        for microbatch_index, start in enumerate(
            range(0, len(curvature_cameras), microbatch)
        ):
            cameras = curvature_cameras[start:start + microbatch]
            render_inputs = None
            local_error = None
            try:
                render_inputs = _transformed_leaves(gaussians, leaves)
            except BaseException as error:
                local_error = error
            _synchronize_rank_errors(
                context,
                phase=f"S2 microbatch {microbatch_index} render preparation",
                error=local_error,
            )
            scales, rotations, opacities, sh_coefficients = render_inputs

            local_error = None
            try:
                write_memory_point(
                    phase="before_curvature_forward",
                    microbatch_index=microbatch_index,
                    cameras=cameras,
                )
                debug_projection_input(
                    phase="curvature",
                    microbatch_index=microbatch_index,
                    cameras=cameras,
                    scales=scales,
                    opacities=opacities,
                )
                submit_ns = time.perf_counter_ns() if timeline_enabled else None
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                event_start.record()
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
                rendered_batch = rendered.permute(0, 3, 1, 2).contiguous()
                target_batch = torch.stack(
                    [
                        torch.clamp(camera.original_image / 255.0, 0.0, 1.0)
                        for camera in cameras
                    ],
                    dim=0,
                )
                residuals = build_fused_3dgs2_curvature_residuals(
                    rendered_batch,
                    target_batch,
                    lambda_dssim=float(getattr(args, "lambda_dssim", 0.2)),
                )
                if not residuals:
                    raise RuntimeError("Distributed S2 produced no residuals")
                event_end.record()
                curvature_forward_ranges.append(
                    (event_start, event_end, submit_ns, cameras)
                )
                projection_events = meta.get(PROJECTION_TIMING_FIX)
                if projection_events is not None:
                    projection_event_pairs.append(projection_events)
                write_memory_point(
                    phase="after_curvature_forward",
                    microbatch_index=microbatch_index,
                    cameras=cameras,
                )
            except BaseException as error:
                local_error = error
                if isinstance(error, torch.OutOfMemoryError):
                    write_memory_point(
                        phase="oom",
                        microbatch_index=microbatch_index,
                        cameras=cameras,
                        error=error,
                    )
            _synchronize_rank_errors(
                context,
                phase=f"S2 microbatch {microbatch_index} forward",
                error=local_error,
            )

            curvature_inputs = tuple(leaves[name] for name in _LEAF_NAMES)
            for sample_index in range(hutchinson_samples):
                local_error = None
                try:
                    submit_ns = time.perf_counter_ns() if timeline_enabled else None
                    event_start = torch.cuda.Event(enable_timing=True)
                    event_end = torch.cuda.Event(enable_timing=True)
                    event_start.record()
                    curvature_values = estimate_seeded_curvature_sample(
                        residuals,
                        curvature_inputs,
                        base_seed=int(
                            getattr(args, "paper_sophia_curvature_seed", 1)
                        ),
                        optimizer_step=optimizer_step,
                        microbatch_index=microbatch_index,
                        sample_index=sample_index,
                        sample_count=hutchinson_samples,
                        rank=int(context.rank),
                    )
                    event_end.record()
                    curvature_vjp_ranges.append(
                        (
                            event_start,
                            event_end,
                            submit_ns,
                            cameras,
                            microbatch_index,
                            sample_index,
                        )
                    )
                    for name, value in zip(_LEAF_NAMES, curvature_values):
                        curvature_accumulators[name].add_(value)
                except BaseException as error:
                    local_error = error
                _synchronize_rank_errors(
                    context,
                    phase=(
                        f"S2 microbatch {microbatch_index} Hutchinson "
                        f"sample {sample_index}"
                    ),
                    error=local_error,
                )
            del (
                rendered,
                rendered_batch,
                target_batch,
                meta,
                residuals,
                scales,
                rotations,
                opacities,
                sh_coefficients,
            )

    if pipeline_mode == "micro":
        def _micro_pipeline_submit(microbatch_index: int, cameras, *, deadline):
            if micro_pipeline_layout is None:
                raise RuntimeError("Micro Pipeline batch camera layout is not initialized")
            render_inputs = _transformed_leaves(gaussians, leaves)
            scales, rotations, opacities, sh_coefficients = render_inputs
            debug_projection_input(
                phase="gradient",
                microbatch_index=microbatch_index,
                cameras=cameras,
                scales=scales,
                opacities=opacities,
            )
            backgrounds = None
            if background is not None:
                backgrounds = background.reshape(1, 3).expand(len(cameras), 3).contiguous()
            global_viewmats, global_Ks = micro_pipeline_layout.microbatch_cameras(
                microbatch_index
            )
            handle = micro_pipeline.submit_projection(
                microbatch_index=microbatch_index,
                deadline=deadline,
                means=leaves["xyz"],
                quats=rotations,
                scales=scales,
                opacities=opacities,
                colors=sh_coefficients,
                viewmats=torch.stack(
                    [camera.world_view_transform.transpose(0, 1) for camera in cameras],
                    dim=0,
                ),
                Ks=torch.stack([camera.K for camera in cameras], dim=0),
                viewmats_global=global_viewmats,
                Ks_global=global_Ks,
                width=image_width,
                height=image_height,
                sh_degree=int(gaussians.active_sh_degree),
                backgrounds=backgrounds,
                near_plane=0.01,
                far_plane=1e10,
                radius_clip=float(getattr(args, "radius_clip", 0.0)),
                eps2d=0.3,
                tile_size=16,
                rasterize_mode="classic",
                N_world=micro_pipeline_layout.N_world,
                C_world=micro_pipeline_layout.microbatch_C_world,
            )
            return handle

        # Compile the actual Loss forward/backward for each camera shape on
        # independent leaves before measured Micro Pipeline stages. Gaussian .grad and
        # optimizer state are untouched. Subsequent batches reuse this cache.
        warmup_keys = getattr(gaussians, "_tide_micro_pipeline_loss_warmup_keys", set())
        with micro_pipeline_error_controller.operation("Micro Pipeline Loss shape warmup"):
            for camera in batched_cameras:
                target = camera.original_image
                key = (
                    image_height, image_width, str(probe_device),
                    leaves["xyz"].dtype, tuple(target.shape), target.dtype,
                    tuple(target.stride()),
                )
                if key in warmup_keys:
                    continue
                warmup_image = torch.zeros(
                    (3, image_height, image_width), device=probe_device,
                    dtype=leaves["xyz"].dtype, requires_grad=True,
                )
                warmup_loss = torch_compiled_loss(warmup_image, target)
                torch.autograd.grad(warmup_loss, warmup_image)
                warmup_done = torch.cuda.Event()
                warmup_done.record(micro_pipeline.compute_stream)
                warmup_done.synchronize()
                warmup_keys.add(key)
                del warmup_image, warmup_loss, warmup_done
        gaussians._tide_micro_pipeline_loss_warmup_keys = warmup_keys
        if timeline_enabled:
            torch.cuda.reset_peak_memory_stats()

        def _project_window_item(item, *, deadline):
            microbatch_index, cameras = item
            handle = _micro_pipeline_submit(microbatch_index, cameras, deadline=deadline)
            handle.start_forward_exchange(deadline=deadline)
            return handle

        def _image_window_item(item, handle, *, deadline):
            microbatch_index, cameras = item
            write_memory_point(
                phase="before_forward", microbatch_index=microbatch_index,
                cameras=cameras,
            )
            forward_submit_ns = time.perf_counter_ns() if timeline_enabled else None
            forward_start = handle.state["projection_start"]
            rendered, _, meta = handle.wait_and_render(deadline=deadline)
            forward_ranges.append(
                (forward_start, handle.render_done, forward_submit_ns, cameras)
            )
            projection_events = meta.get(PROJECTION_TIMING_FIX)
            if projection_events is not None:
                projection_event_pairs.append(projection_events)
            if grad_zero_sample_due:
                _mark_projection_cull_survivors(
                    projection_cull_survivor_mask, meta, owner_active_rows,
                )
                if tile_contribution_mode == "off":
                    tile_survivor_mask.copy_(projection_cull_survivor_mask)
                else:
                    _merge_owner_tile_survivors(
                        tile_survivor_mask, tile_mask_counts, meta,
                        owner_active_rows=owner_active_rows,
                        participation_rows=participation_rows,
                        uses_sentinel=uses_sentinel,
                    )
            loss_start = torch.cuda.Event(enable_timing=True)
            loss_end = torch.cuda.Event(enable_timing=True)
            loss_submit_ns = time.perf_counter_ns() if timeline_enabled else None
            loss_start.record(micro_pipeline.compute_stream)
            with tide_range(
                f"Tide:S1:MB{microbatch_index}:Loss"
                f"|rank={context.rank},batch={iteration},slot={handle.slot},"
                f"tid={threading.get_native_id()},schedule={MICRO_PIPELINE_SCHEDULE_VERSION}"
            ):
                micro_losses = [
                    torch_compiled_loss(
                        rendered[local_index].permute(2, 0, 1).contiguous(),
                        camera.original_image,
                    )
                    for local_index, camera in enumerate(cameras)
                ]
            loss_end.record(micro_pipeline.compute_stream)
            loss_ranges.append((loss_start, loss_end, loss_submit_ns, cameras))
            detached_losses.extend(loss.detach() for loss in micro_losses)
            write_memory_point(
                phase="after_forward", microbatch_index=microbatch_index,
                cameras=cameras,
            )
            # This differentiation stops at independent C1 image leaves. Only
            # G below can accumulate into the shared Gaussian parameter leaves.
            handle.intermediate_backward(
                torch.stack(micro_losses).sum(), deadline=deadline,
            )

        def _final_window_item(item, handle, *, deadline):
            microbatch_index, cameras = item
            backward_end = handle.final_backward(deadline=deadline)
            record = handle.snapshot()
            backward_ranges.append((
                record["intermediate_start"], backward_end,
                record.get("intermediate_submit_ns"), cameras,
            ))
            write_memory_point(
                phase="after_backward", microbatch_index=microbatch_index,
                cameras=cameras,
            )
            # Timing snapshots contain CUDA events and scalars, never tensors
            # from the completed projection or image computation graphs.
            micro_pipeline_records.append((record, cameras, microbatch_index))

        try:
            with torch.cuda.stream(micro_pipeline.compute_stream):
                run_micro_pipeline_windows(
                    [
                        (index, batched_cameras[start:start + microbatch])
                        for index, start in enumerate(
                            range(0, len(batched_cameras), microbatch)
                        )
                    ],
                    buffer_count=micro_pipeline.buffer_count,
                    controller=micro_pipeline_error_controller,
                    project=_project_window_item,
                    image_backward=_image_window_item,
                    final_backward=_final_window_item,
                )
        except BaseException as error:
            micro_pipeline.abort()
            micro_pipeline_error_controller.fail("Micro Pipeline fixed-window execution", error)
            raise

        drain_error = None
        try:
            _micro_pipeline_run("Micro Pipeline drain", micro_pipeline.drain)
        except BaseException as error:
            drain_error = error
            micro_pipeline.abort()
        _synchronize_rank_errors(
            context,
            phase="S1 Micro Pipeline drain",
            error=drain_error,
            group=micro_pipeline_error_group,
        )
        micro_pipeline.close()

    for microbatch_index, start in enumerate(
        range(0, len(batched_cameras), microbatch)
    ):
        if pipeline_mode == "micro":
            continue
        cameras = batched_cameras[start:start + microbatch]
        render_inputs = None
        local_error = None
        try:
            render_inputs = _transformed_leaves(gaussians, leaves)
        except BaseException as error:
            local_error = error
        _synchronize_rank_errors(
            context,
            phase=f"S1 microbatch {microbatch_index} render preparation",
            error=local_error,
        )
        scales, rotations, opacities, sh_coefficients = render_inputs

        local_error = None
        try:
            write_memory_point(
                phase="before_forward",
                microbatch_index=microbatch_index,
                cameras=cameras,
            )
            debug_projection_input(
                phase="gradient",
                microbatch_index=microbatch_index,
                cameras=cameras,
                scales=scales,
                opacities=opacities,
            )
            forward_submit_ns = time.perf_counter_ns() if timeline_enabled else None
            forward_start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)
            forward_start.record()
            with tide_range("tide.cpu.projection.submit"):
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
            forward_end.record()
            forward_ranges.append(
                (forward_start, forward_end, forward_submit_ns, cameras)
            )
            loss_start = torch.cuda.Event(enable_timing=True)
            loss_end = torch.cuda.Event(enable_timing=True)
            loss_start.record()
            micro_losses = []
            for local_index, camera in enumerate(cameras):
                image = rendered[local_index].permute(2, 0, 1).contiguous()
                micro_losses.append(torch_compiled_loss(image, camera.original_image))
            loss_end.record()
            loss_ranges.append(
                (loss_start, loss_end, forward_submit_ns, cameras)
            )

            projection_events = meta.get(PROJECTION_TIMING_FIX)
            if projection_events is not None:
                projection_event_pairs.append(projection_events)
            if grad_zero_sample_due:
                _mark_projection_cull_survivors(
                    projection_cull_survivor_mask,
                    meta,
                    owner_active_rows,
                )
                if tile_contribution_mode == "off":
                    tile_survivor_mask.copy_(projection_cull_survivor_mask)
                else:
                    _merge_owner_tile_survivors(
                        tile_survivor_mask,
                        tile_mask_counts,
                        meta,
                        owner_active_rows=owner_active_rows,
                        participation_rows=participation_rows,
                        uses_sentinel=uses_sentinel,
                    )
            detached_losses.extend(loss.detach() for loss in micro_losses)
            write_memory_point(
                phase="after_forward",
                microbatch_index=microbatch_index,
                cameras=cameras,
            )
        except BaseException as error:
            local_error = error
            if isinstance(error, torch.OutOfMemoryError):
                write_memory_point(
                    phase="oom",
                    microbatch_index=microbatch_index,
                    cameras=cameras,
                    error=error,
                )
        _synchronize_rank_errors(
            context,
            phase=f"S1 microbatch {microbatch_index} forward",
            error=local_error,
        )

        local_error = None
        try:
            backward_submit_ns = time.perf_counter_ns() if timeline_enabled else None
            backward_start = torch.cuda.Event(enable_timing=True)
            backward_end = torch.cuda.Event(enable_timing=True)
            backward_start.record()
            torch.stack(micro_losses).sum().backward()
            backward_end.record()
            backward_ranges.append(
                (backward_start, backward_end, backward_submit_ns, cameras)
            )
            write_memory_point(
                phase="after_backward",
                microbatch_index=microbatch_index,
                cameras=cameras,
            )
        except BaseException as error:
            local_error = error
            if isinstance(error, torch.OutOfMemoryError):
                write_memory_point(
                    phase="oom",
                    microbatch_index=microbatch_index,
                    cameras=cameras,
                    error=error,
                )
        _synchronize_rank_errors(
            context,
            phase=f"S1 microbatch {microbatch_index} backward",
            error=local_error,
        )
        del (
            rendered,
            meta,
            image,
            micro_losses,
            scales,
            rotations,
            opacities,
            sh_coefficients,
        )

    if not detached_losses:
        raise RuntimeError("Distributed TideGS produced no local losses")

    sparse_error = None
    grad_zero_record = None
    grad_sample_payload = None
    try:
        with tide_range("tide.cpu.gradient.sparse_prepare"):
            owner_grad_components = {
                name: (
                    torch.zeros_like(leaves[name])
                    if leaves[name].grad is None
                    else leaves[name].grad
                )
                for name in _LEAF_NAMES
            }
            if grad_zero_sample_due:
                with tide_range("tide.cpu.gradient.sparsity_sample"):
                    grad_profile = collect_gradient_sparsity(
                        owner_grad_components,
                        projection_cull_survivor_mask,
                        tile_survivor_mask,
                        active_sh_degree=int(gaussians.active_sh_degree),
                        tile_mode=tile_contribution_mode,
                        tile_counts=tile_mask_counts,
                        near_zero_threshold=metrics_writer.grad_near_zero_threshold,
                        sample_rows=metrics_writer.grad_sample_rows,
                        chunk_rows=metrics_writer.grad_stats_chunk_rows,
                    )
                    grad_zero_record, grad_sample_payload = build_gradient_sparsity_records(
                        grad_profile,
                        GRAD_METRIC_VALUE_FIELDS,
                        active_local_ids,
                        tile_survivor_mask,
                        iteration=iteration,
                        optimizer_step=optimizer_step,
                        active_sh_degree=int(gaussians.active_sh_degree),
                        near_zero_threshold=metrics_writer.grad_near_zero_threshold,
                        tile_mode=tile_contribution_mode,
                        tile_alpha_threshold=tile_alpha_threshold,
                    )
            gradient_active = (
                grad_profile["touched_rows"]
                if grad_zero_sample_due
                else _touched_component_rows(
                    owner_grad_components,
                    owner_active_rows,
                    active_local_ids.device,
                )
            )
            curvature_active = _touched_component_rows(
                curvature_accumulators,
                owner_active_rows,
                active_local_ids.device,
            )
            gradient_local = active_local_ids.index_select(0, gradient_active)
            curvature_local = active_local_ids.index_select(0, curvature_active)
            sparse_grad_components = {}
            sparse_curvature_components = {}
            for name in _LEAF_NAMES:
                sparse_grad_components[name] = owner_grad_components[name].index_select(
                    0, gradient_active
                ).contiguous()
                sparse_curvature_components[name] = curvature_accumulators[
                    name
                ].index_select(0, curvature_active).contiguous()
            del owner_grad_components
    except BaseException as error:
        sparse_error = error
    _synchronize_rank_errors(
        context,
        phase="sparse gradient/curvature preparation",
        error=sparse_error,
    )
    if grad_zero_sample_due:
        with tide_range("tide.cpu.gradient.sample_emit"):
            metrics_writer.write_grad_zero(grad_zero_record)
            metrics_writer.write_grad_sample(grad_sample_payload)

    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    optimizer_start.record()
    if optimizer_algorithm == "3dgs2_tr":
        prepared = None
        prepare_error = None
        try:
            with tide_range(f"Tide:S1:Optimizer:{optimizer_algorithm}:Prepare"):
                prepared = optimizer.prepare_step(
                    iteration=iteration,
                    gaussians=gaussians,
                    sparse_grad_local_ids=gradient_local,
                    sparse_grad_components=sparse_grad_components,
                    sparse_curvature_local_ids=curvature_local,
                    sparse_curvature_components=(
                        sparse_curvature_components if curvature_due else None
                    ),
                    curvature_due=curvature_due,
                    optimizer_step=optimizer_step,
                )
        except BaseException as error:
            prepare_error = error
        try:
            _synchronize_rank_errors(
                context,
                phase="3DGS2-TR optimizer prepare",
                error=prepare_error,
            )
        except BaseException:
            if prepared is not None:
                optimizer.abort_step(prepared)
            raise

        step_error = None
        try:
            with tide_range(f"Tide:S1:Optimizer:{optimizer_algorithm}:Commit"):
                step_stats = optimizer.commit_step(prepared)
        except BaseException as error:
            step_error = error
        _synchronize_rank_errors(
            context,
            phase="3DGS2-TR optimizer commit",
            error=step_error,
        )
    else:
        step_error = None
        try:
            with tide_range(f"Tide:S1:Optimizer:{optimizer_algorithm}"):
                step_stats = optimizer.step(
                    iteration=iteration,
                    gaussians=gaussians,
                    sparse_grad_local_ids=gradient_local,
                    sparse_grad_components=sparse_grad_components,
                )
        except BaseException as error:
            step_error = error
        _synchronize_rank_errors(
            context,
            phase="Adam optimizer step",
            error=step_error,
        )
    optimizer_end.record()
    gaussians._tide_global_optimizer_step = optimizer_step
    updated_blocks = list(step_stats.get("updated_block_ids", []))
    with tide_range("tide.cpu.writeback.mark_dirty"):
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
    with tide_range("tide.comm.batch.barrier"):
        context.barrier()
    barrier_ms = (time.perf_counter() - barrier_start) * 1000.0
    retention_error = None
    try:
        with tide_range("tide.cpu.retention.prepare"):
            manager.prepare_for_retention()
    except BaseException as error:
        retention_error = error
    with tide_range("tide.comm.retention.status"):
        _synchronize_rank_errors(
            context,
            phase="retention preparation",
            error=retention_error,
        )

    h2d_events = retention_stats.get("h2d_events")
    h2d_ms = (
        float(h2d_events[0].elapsed_time(h2d_events[1]))
        if h2d_events is not None
        else 0.0
    )
    cache_after = cache.get_stats()

    def cache_delta(name):
        return float(cache_after.get(name, 0)) - float(cache_before.get(name, 0))

    # ``block_reader.read_batch`` covers RAM-cache lookup and CPU packing in
    # addition to any storage wait.  Record that full foreground path
    # separately from direct SSD misses and waits for an N+1 prefetch to
    # become cache-ready.
    ssd_urgent_read_ms = cache_delta("urgent_storage_read_time") * 1000.0
    prefetch_inflight_wait_ms = cache_delta("inflight_wait_time") * 1000.0

    with tide_range("tide.cpu.metrics.cuda_timing"):
        projection_ms = sum(
            projection_elapsed_ms({PROJECTION_TIMING_FIX: events})
            for events in projection_event_pairs
        )
        forward_range_ms = [
            float(start.elapsed_time(end)) for start, end, _, _ in forward_ranges
        ]
        loss_range_ms = [
            float(start.elapsed_time(end)) for start, end, _, _ in loss_ranges
        ]
        backward_range_ms = [
            float(start.elapsed_time(end)) for start, end, _, _ in backward_ranges
        ]
        curvature_forward_range_ms = [
            float(start.elapsed_time(end))
            for start, end, _, _ in curvature_forward_ranges
        ]
        curvature_vjp_range_ms = [
            float(start.elapsed_time(end))
            for start, end, _, _, _, _ in curvature_vjp_ranges
        ]
        forward_ms = sum(forward_range_ms)
        loss_ms = sum(loss_range_ms)
        backward_ms = sum(backward_range_ms)
        curvature_forward_ms = sum(curvature_forward_range_ms)
        curvature_vjp_ms = sum(curvature_vjp_range_ms)
        optimizer_ms = float(optimizer_start.elapsed_time(optimizer_end))
        micro_pipeline_overlap_ms = 0.0
        micro_pipeline_wait_ms = 0.0
        micro_pipeline_metadata_ms = 0.0
        micro_pipeline_payload_ms = 0.0
        micro_pipeline_render_ms = 0.0
        backward_stage_ms = dict.fromkeys(
            ("IntermediateGrad", "C2", "FinalGrad", "AccumulateGrad"), 0.0
        )
        train_ms = (
            curvature_forward_ms + curvature_vjp_ms + forward_ms
            + loss_ms + backward_ms + optimizer_ms
        )
        if pipeline_mode == "micro":
            for record_index, (record, _, _) in enumerate(micro_pipeline_records):
                timing = micro_pipeline.timing_from_snapshot(record)
                micro_pipeline_wait_ms += timing.get("c1_wait_ms", 0.0)
                micro_pipeline_metadata_ms += timing.get("c1_metadata_ms", 0.0)
                micro_pipeline_payload_ms += timing.get("c1_payload_ms", 0.0)
                micro_pipeline_render_ms += timing.get("render_ms", 0.0)
                if record_index + 1 < len(micro_pipeline_records):
                    next_record = micro_pipeline_records[record_index + 1][0]
                    c1_start = record.get("metadata_start")
                    c1_end = record.get("payload_end")
                    p_start = next_record.get("projection_start")
                    p_end = next_record.get("projection_end")
                    if None not in (c1_start, c1_end, p_start, p_end):
                        c1_ms = float(c1_start.elapsed_time(c1_end))
                        p_ms = float(p_start.elapsed_time(p_end))
                        p_offset = float(c1_start.elapsed_time(p_start))
                        micro_pipeline_overlap_ms += max(
                            0.0,
                            min(c1_ms, p_offset + p_ms) - max(0.0, p_offset),
                        )
            gpu_anchor = micro_pipeline_records[0][0]["projection_start"]
            explicit_stages = {
                "IntermediateGrad": ("intermediate_start", "intermediate_done"),
                "C2": ("c2_payload_start", "c2_payload_end"),
                "FinalGrad": ("final_start", "backward_done"),
            }
            for stage in backward_stage_ms:
                if stage == "AccumulateGrad" and not timeline_enabled:
                    # This stage has no standalone event outside MicrobatchTrace.
                    backward_stage_ms[stage] = None
                    continue
                explicit = explicit_stages.get(stage)
                backward_stage_ms[stage] = cuda_interval_union_ms(
                    gpu_anchor,
                    [
                        (span["start_event"], span["end_event"])
                        for record, _, _ in micro_pipeline_records
                        for span in record.get("backward_trace", [])
                        if span["stage"] == stage and span["complete"]
                    ] + [
                        (record[explicit[0]], record[explicit[1]])
                        for record, _, _ in micro_pipeline_records
                        if explicit is not None
                        and record.get(explicit[0]) is not None
                        and record.get(explicit[1]) is not None
                    ],
                )
            # Forward request intervals overlap in Micro Pipeline. Adding them (and then
            # adding backward again) overstates covered training time.
            train_ms = cuda_interval_union_ms(
                gpu_anchor,
                [(entry[0], entry[1]) for ranges in (
                    curvature_forward_ranges, curvature_vjp_ranges,
                    forward_ranges, loss_ranges, backward_ranges,
                ) for entry in ranges] + [(optimizer_start, optimizer_end)],
            )
    if timeline_enabled:
        with tide_range("tide.cpu.metrics.timeline_emit"):
            anchor_event = None
            anchor_host_ns = None
            if pipeline_mode == "micro" and micro_pipeline_records:
                anchor_record = micro_pipeline_records[0][0]
                anchor_event = anchor_record.get("projection_start")
                anchor_host_ns = int(anchor_record.get("projection_submit_ns", 0))
            elif forward_ranges:
                anchor_event = forward_ranges[0][0]
                anchor_host_ns = int(forward_ranges[0][2] or 0)

            def _event_span(start_event, end_event):
                if anchor_event is None or start_event is None or end_event is None:
                    return None
                try:
                    start_ns = anchor_host_ns + round(
                        anchor_event.elapsed_time(start_event) * 1e6
                    )
                    end_ns = anchor_host_ns + round(
                        anchor_event.elapsed_time(end_event) * 1e6
                    )
                    return start_ns, end_ns
                except RuntimeError:
                    return None

            def _write_cuda_ranges(name, lane, ranges, detail=None):
                for microbatch_index, (start_event, end_event, _, cameras) in enumerate(
                    ranges
                ):
                    span = _event_span(start_event, end_event)
                    if span is None:
                        continue
                    metrics_writer.write_timeline_event(
                        name=(
                            name(microbatch_index)
                            if callable(name)
                            else name
                        ),
                        lane=lane,
                        start_ns=span[0],
                        end_ns=span[1],
                        iteration=iteration,
                        timing_source="cuda_event_relative_to_batch_anchor",
                        detail=detail,
                        microbatch_index=microbatch_index,
                        microbatch_count=microbatch_count,
                        camera_uids=camera_uids(cameras),
                        thread_id=threading.get_native_id(),
                        thread_role="main",
                        pipeline_mode=pipeline_mode,
                        batch_index=iteration,
                        schedule_version=(MICRO_PIPELINE_SCHEDULE_VERSION if pipeline_mode == "micro" else "off"),
                        pipeline_buffer_slot=(
                            microbatch_index % micro_pipeline.buffer_count
                            if pipeline_mode == "micro" else None
                        ),
                        slot=(
                            microbatch_index % micro_pipeline.buffer_count
                            if pipeline_mode == "micro" else None
                        ),
                        communication_sequence=-1,
                    )

            def _write_cuda_event(
                *,
                name,
                lane,
                start_event,
                end_event,
                detail=None,
                **fields,
            ):
                span = _event_span(start_event, end_event)
                if span is None:
                    return
                metrics_writer.write_timeline_event(
                    name=name,
                    lane=lane,
                    start_ns=span[0],
                    end_ns=span[1],
                    iteration=iteration,
                    timing_source="cuda_event_relative_to_batch_anchor",
                    detail=detail,
                    **fields,
                )

            if pipeline_mode == "micro":
                for record, cameras, microbatch_index in micro_pipeline_records:
                    spans = (
                        ("Projection", "gpu", record.get("projection_start"), record.get("projection_end")),
                        ("C1:Wait", "nccl", record.get("c1_wait_start"), record.get("c1_wait_end")),
                        ("C1:Metadata", "nccl", record.get("metadata_start"), record.get("metadata_end")),
                        ("C1:Payload", "nccl", record.get("payload_start"), record.get("payload_end")),
                        ("Render", "gpu", record.get("render_start"), record.get("render_done")),
                        ("IntermediateGrad", "gpu", record.get("intermediate_start"), record.get("intermediate_done")),
                        ("C2:Wait", "nccl", record.get("c2_wait_start"), record.get("c2_wait_end")),
                        ("C2:Payload", "nccl", record.get("c2_payload_start"), record.get("c2_payload_end")),
                        ("FinalGrad", "gpu", record.get("final_start"), record.get("backward_done")),
                    )
                    for suffix, lane, start_event, end_event in spans:
                        span = _event_span(start_event, end_event)
                        if span is None:
                            continue
                        comm_stage = suffix.split(":", 1)[0]
                        worker_span = next((
                            value for value in record.get("cpu_trace", [])
                            if value.get("name") == f"{comm_stage}:Worker"
                        ), {})
                        metrics_writer.write_timeline_event(
                            name=f"Tide:S1:MB{microbatch_index}:{suffix}",
                            lane=lane,
                            start_ns=span[0],
                            end_ns=span[1],
                            iteration=iteration,
                            timing_source="cuda_event_relative_to_batch_anchor",
                            microbatch_index=microbatch_index,
                            microbatch_count=microbatch_count,
                            pipeline_buffer_slot=int(record["pipeline_buffer_slot"]),
                            pipeline_mode=pipeline_mode,
                            batch_index=iteration,
                            slot=int(record["pipeline_buffer_slot"]),
                            schedule_version=MICRO_PIPELINE_SCHEDULE_VERSION,
                            thread_id=worker_span.get("thread_id", threading.get_native_id()),
                            thread_role="comm" if worker_span else "main",
                            stream_id=int((micro_pipeline_nccl_stream if worker_span
                                           else micro_pipeline.compute_stream).cuda_stream),
                            communication_sequence=worker_span.get("communication_sequence", -1),
                            camera_uids=camera_uids(cameras),
                            stage=suffix,
                            scope="stage",
                        )
                    for cpu_span in record.get("cpu_trace", []):
                        fields = dict(cpu_span)
                        name = fields.pop("name")
                        start_ns = fields.pop("start_ns")
                        end_ns = fields.pop("end_ns")
                        fields.update(
                            microbatch_index=microbatch_index,
                            pipeline_buffer_slot=int(record["pipeline_buffer_slot"]),
                            schedule_version=MICRO_PIPELINE_SCHEDULE_VERSION,
                        )
                        metrics_writer.write_timeline_event(
                            name=f"Tide:S1:MB{microbatch_index}:{name}", lane="cpu",
                            start_ns=start_ns, end_ns=end_ns, iteration=iteration, **fields,
                        )
                    for node_index, node_span in enumerate(record.get("backward_trace", [])):
                        if not node_span["complete"]:
                            continue
                        _write_cuda_event(
                            name=f"Tide:S1:MB{microbatch_index}:{node_span['stage']}",
                            lane="nccl" if node_span["stage"] == "C2" else "gpu",
                            start_event=node_span["start_event"],
                            end_event=node_span["end_event"],
                            microbatch_index=microbatch_index,
                            microbatch_count=microbatch_count,
                            camera_uids=camera_uids(cameras),
                            stage=node_span["stage"],
                            scope="autograd_node",
                            node_index=node_index,
                            node_name=node_span["node_name"],
                            stream_id=node_span["stream_id"],
                            timing_kind=node_span["timing_kind"],
                            thread_id=node_span.get("thread_id"),
                            thread_role=node_span.get("thread_role", "autograd"),
                            pipeline_mode=pipeline_mode,
                            batch_index=iteration,
                            slot=int(record["pipeline_buffer_slot"]),
                            pipeline_buffer_slot=int(record["pipeline_buffer_slot"]),
                            communication_sequence=-1,
                            schedule_version=MICRO_PIPELINE_SCHEDULE_VERSION,
                            detail=("includes_collective_readiness_wait"
                                    if node_span["stage"] == "C2" else None),
                        )
                _write_cuda_ranges(
                    lambda index: f"Tide:S1:MB{index}:Loss",
                    "gpu",
                    loss_ranges,
                )
            else:
                _write_cuda_ranges(
                    "gaussian_cull_forward",
                    "gpu",
                    forward_ranges,
                    detail="includes_projection_cull",
                )
                _write_cuda_ranges("Loss", "gpu", loss_ranges)
            if pipeline_mode != "micro":
                _write_cuda_ranges("Backward", "gpu", backward_ranges)
            _write_cuda_ranges("curvature_forward", "gpu", curvature_forward_ranges)
            for (
                start_event,
                end_event,
                _,
                cameras,
                microbatch_index,
                sample_index,
            ) in curvature_vjp_ranges:
                _write_cuda_event(
                    name="curvature_vjp",
                    lane="gpu",
                    start_event=start_event,
                    end_event=end_event,
                    microbatch_index=microbatch_index,
                    microbatch_count=microbatch_count,
                    sample_index=sample_index,
                    camera_uids=camera_uids(cameras),
                )
            _write_cuda_event(
                name=(f"Tide:S1:Optimizer:{optimizer_algorithm}"
                      if pipeline_mode == "micro" else optimizer_algorithm),
                lane="gpu",
                start_event=optimizer_start,
                end_event=optimizer_end,
                stage="Optimizer",
                scope="batch",
            )
            h2d_events = retention_stats.get("h2d_events")
            if h2d_events is not None:
                _write_cuda_event(
                    name="gaussian_materialize_h2d",
                    lane="gpu",
                    start_event=h2d_events[0],
                    end_event=h2d_events[1],
                    bytes=int(retention_stats.get("h2d_bytes", 0)),
                    detail="h2d_plus_gpu_unpack",
                )
    metrics_row = {
        "iteration": iteration,
        "optimizer_algorithm": optimizer_algorithm,
        "optimizer_step": optimizer_step,
        "curvature_due": int(curvature_due),
        "local_cameras": len(batched_cameras),
        "global_active_blocks": len(plan.global_active_blocks),
        "global_resident_blocks": len(plan.global_resident_blocks),
        "global_active_resident_blocks": len(
            set(plan.global_active_blocks).intersection(plan.global_resident_blocks)
        ),
        "predicted_stream_in_blocks": int(plan.predicted_stream_in_blocks),
        "prediction_missing_blocks": plan.prediction_missing_blocks,
        "prediction_extra_blocks": plan.prediction_extra_blocks,
        "prediction_replanned": int(plan.prediction_replanned),
        "plan_resolution": plan.plan_resolution,
        "prediction_repair_ms": float(plan.prediction_repair_ms),
        "prediction_exact_plan_ms": float(plan.prediction_exact_plan_ms),
        "rank_active_blocks": len(active_blocks),
        "rank_gradient_active_blocks": len(gradient_active_blocks),
        "rank_curvature_active_blocks": len(curvature_active_blocks),
        "rank_resident_blocks": len(target_resident),
        "cold_blocks": int(retention_stats.get("cold_count", 0)),
        "retained_blocks": int(retention_stats.get("hotspot_count", 0)),
        "gpu_slot_capacity_blocks": int(
            retention_stats.get("gpu_slot_capacity_blocks", 0)
        ),
        "gpu_slot_growth_blocks": int(
            retention_stats.get("gpu_slot_growth_blocks", 0)
        ),
        "owner_active_gaussians": owner_active_rows,
        "collective_participation_gaussians": participation_rows,
        "uses_zero_opacity_sentinel": int(uses_sentinel),
        "touched_gaussians": int(gradient_active.numel()),
        "gradient_participation_rows": int(gradient_active.numel()),
        "curvature_participation_rows": int(curvature_active.numel()),
        "updated_blocks": int(step_stats.get("updated_blocks", 0)),
        "optimizer_touched_rows": int(step_stats.get("touched_rows", 0)),
        "optimizer_cold_rows": int(step_stats.get("cold_rows", 0)),
        "curvature_blocks": len(step_stats.get("curvature_block_ids", [])),
        "curvature_rows": int(step_stats.get("curvature_rows", 0)),
        "clipped_values": int(step_stats.get("clipped_values", 0)),
        "rows_skipped_without_curvature": int(
            step_stats.get("rows_skipped_without_curvature", 0)
        ),
        "trust_region_epsilon": float(
            step_stats.get("trust_region_epsilon", 0.0)
        ),
        "block_cull_ms": float(plan.block_cull_ms),
        "block_cull_backend": str(plan.block_cull_backend),
        "block_cull_gpu_kernel_ms": float(plan.block_cull_gpu_kernel_ms),
        "block_cull_gpu_d2h_ms": float(plan.block_cull_gpu_d2h_ms),
        "block_cull_cache_hit_cameras": int(
            plan.block_cull_cache_hit_cameras
        ),
        "block_cull_gpu_cameras": int(plan.block_cull_gpu_cameras),
        "block_cull_output_blocks": int(plan.block_cull_output_blocks),
        "plan_ms": float(plan.plan_ms),
        "writeback_submit_ms": writeback_submit_ms,
        "resident_load_ms": resident_load_ms,
        "block_reader_foreground_ms": float(
            retention_stats.get("foreground_read_ms", 0.0)
        ),
        "ssd_urgent_read_ms": ssd_urgent_read_ms,
        "prefetch_inflight_wait_ms": prefetch_inflight_wait_ms,
        "cpu_materialize_ms": float(
            retention_stats.get("cpu_materialize_ms", 0.0)
        ),
        "ssd_urgent_read_blocks": cache_delta("urgent_storage_read_blocks"),
        "ssd_urgent_read_bytes": cache_delta("ssd_bytes_read_urgent"),
        "h2d_bytes": int(retention_stats.get("h2d_bytes", 0)),
        "h2d_ms": h2d_ms,
        "gsplat_forward_ms": forward_ms,
        "loss_ms": loss_ms,
        "pipeline_mode": pipeline_mode,
        "projection_c1_overlap_ms": micro_pipeline_overlap_ms,
        "c1_wait_ms": micro_pipeline_wait_ms,
        "c1_metadata_ms": micro_pipeline_metadata_ms,
        "c1_payload_ms": micro_pipeline_payload_ms,
        "pipeline_render_ms": micro_pipeline_render_ms,
        "intermediate_grad_ms": backward_stage_ms["IntermediateGrad"],
        "c2_ms": backward_stage_ms["C2"],
        "final_grad_ms": backward_stage_ms["FinalGrad"],
        "grad_accumulate_ms": backward_stage_ms["AccumulateGrad"],
        "curvature_forward_ms": curvature_forward_ms,
        "curvature_vjp_ms": curvature_vjp_ms,
        "gaussian_projection_cull_ms": projection_ms,
        "backward_ms": backward_ms,
        "optimizer_ms": optimizer_ms,
        "train_ms": train_ms,
        "bounds_sync_ms": bounds_sync_ms,
        "barrier_ms": barrier_ms,
        "batch_total_ms": (time.perf_counter() - batch_start) * 1000.0,
    }
    with tide_range("tide.cpu.metrics.io_emit"):
        metrics_writer.write_io_events(cache.drain_io_events())
    with tide_range("tide.cpu.metrics.batch_emit"):
        metrics_writer.write_batch(metrics_row)

    gaussians._tide_distributed_last_step = {
        **step_stats,
        "rank": context.rank,
        "local_cameras": len(batched_cameras),
        "local_curvature_cameras": len(curvature_cameras or []),
        "resident_blocks": len(target_resident),
        "active_blocks": len(active_blocks),
        "gradient_participation_rows": int(gradient_active.numel()),
        "curvature_participation_rows": int(curvature_active.numel()),
        "collective_participation_rows": participation_rows,
        "uses_zero_opacity_sentinel": uses_sentinel,
    }
    return detached_losses, list(range(len(batched_cameras))), float(owner_active_rows)
