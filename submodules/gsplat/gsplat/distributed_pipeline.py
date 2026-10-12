"""Two-stage distributed rasterization for the TideGS P/C1 prototype.

The public :func:`gsplat.rasterization` entry point remains synchronous.  This
module exposes a deliberately small packed, distributed path used by TideGS:
projection and feature preparation run on the compute stream, while the first
packed Gaussian exchange runs on a separate NCCL stream.  The implementation
shares the same projection, all-to-all layout, and rasterization kernels as
gsplat 1.5.3 so that the returned tensors remain connected to autograd.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, Optional

import torch
import torch.distributed
from torch import Tensor

from .cuda._wrapper import fully_fused_projection, isect_offset_encode, isect_tiles, rasterize_to_pixels, spherical_harmonics
from .distributed import all_gather_int32, all_gather_tensor_list, all_to_all_int32, all_to_all_tensor_list


def _nvtx_push(name: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def _nvtx_pop() -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def _cuda_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _projection_prepare(
    *,
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    eps2d: float,
    sh_degree: int,
    backgrounds: Optional[Tensor],
    tile_size: int,
    rasterize_mode: str,
) -> Dict[str, object]:
    """Run P(i) and return tensors immediately before C1(i)."""

    if means.ndim != 2 or quats.ndim != 2 or scales.ndim != 2:
        raise ValueError("pc1 requires unbatched Gaussian tensors")
    if not torch.is_tensor(viewmats) or viewmats.ndim != 3:
        raise ValueError("pc1 requires [C,4,4] viewmats")
    if not torch.is_tensor(Ks) or Ks.ndim != 3:
        raise ValueError("pc1 requires [C,3,3] intrinsics")
    if means.device.type != "cuda":
        raise ValueError("pc1 requires CUDA tensors")
    if colors.ndim != 3 or colors.shape[0] != means.shape[0] or colors.shape[-1] != 3:
        raise ValueError("pc1 requires per-Gaussian SH colors with shape [N,K,3]")
    if sh_degree is None:
        raise ValueError("pc1 requires an active SH degree")
    if rasterize_mode != "classic":
        raise ValueError("pc1 currently supports rasterize_mode='classic' only")

    device = means.device
    world_size = torch.distributed.get_world_size()
    N = int(means.shape[0])
    C_local = int(viewmats.shape[0])
    if C_local <= 0:
        raise ValueError("pc1 requires at least one local camera")
    N_world = all_gather_int32(world_size, N, device=device)
    # The packed exchange assumes a rectangular camera layout.  Collect the
    # counts before the tensor gather so every rank makes the same decision and
    # an invalid job cannot leave the next collective waiting forever.
    C_world = all_gather_int32(world_size, C_local, device=device)
    if any(int(count) != C_local for count in C_world):
        raise ValueError(
            "pc1 requires the same number of local cameras on every rank; "
            f"got {C_world}"
        )
    viewmats_global, Ks_global = all_gather_tensor_list(world_size, [viewmats, Ks])
    C_global = int(viewmats_global.shape[0])

    projection_start = _cuda_event()
    projection_end = _cuda_event()
    projection_start.record(torch.cuda.current_stream(device))
    proj_results = fully_fused_projection(
        means,
        None,
        quats,
        scales,
        viewmats_global,
        Ks_global,
        width,
        height,
        eps2d=eps2d,
        packed=True,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=False,
        calc_compensations=(rasterize_mode == "antialiased"),
        camera_model="pinhole",
        opacities=opacities,
    )
    (
        batch_ids,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        compensations,
    ) = proj_results
    # Inputs are explicitly unbatched above, so the projection kernel returns
    # zero batch ids.  Avoid converting a CUDA predicate to a Python bool here:
    # that would synchronize the compute stream and defeat P(i+1)/C1(i).
    opacities_packed = opacities[gaussian_ids]
    if compensations is not None:
        opacities_packed = opacities_packed * compensations

    original_radii = radii
    meta: Dict[str, object] = {}
    tile_mode = getattr(torch, "_tide_tile_contribution_mode", "off")
    if tile_mode != "off":
        tile_mask_fn = getattr(torch, "_tide_tile_contribution_mask", None)
        if tile_mask_fn is None:
            raise RuntimeError("TideGS tile contribution callback is not installed")
        keep, candidate_counts, contributing_counts = tile_mask_fn(
            means2d,
            conics,
            opacities_packed,
            radii,
            width,
            height,
            tile_size,
            getattr(torch, "_tide_tile_alpha_threshold", 1.0 / 255.0),
        )
        owner_tile = torch.zeros((N,), dtype=torch.bool, device=device)
        owner_tile[gaussian_ids[keep]] = True
        meta["_tide_owner_tile_survivor_mask"] = owner_tile
        meta["_tide_tile_mask_counts"] = torch.stack(
            [
                keep.new_tensor(keep.numel(), dtype=torch.long),
                keep.sum(dtype=torch.long),
                candidate_counts.sum(dtype=torch.long),
                candidate_counts[keep].sum(dtype=torch.long),
                contributing_counts.sum(dtype=torch.long),
            ]
        )
        if tile_mode == "apply":
            radii = torch.where(keep, radii, torch.zeros_like(radii))

    if getattr(torch, "_tide_gsplat_grad_zero_metrics", False):
        owner_projection = torch.zeros((N,), dtype=torch.bool, device=device)
        owner_projection[gaussian_ids[(original_radii > 0).all(dim=-1)]] = True
        meta["_tide_owner_projection_survivor_mask"] = owner_projection

    campos = torch.inverse(viewmats_global)[..., :3, 3]
    dirs = means[gaussian_ids] - campos[camera_ids]
    shs = colors[gaussian_ids]
    colors_packed = spherical_harmonics(
        sh_degree,
        dirs,
        shs,
        masks=(radii > 0).all(dim=-1),
    )
    colors_packed = torch.clamp_min(colors_packed + 0.5, 0.0)
    projection_end.record(torch.cuda.current_stream(device))
    meta["_tide_projection_events"] = (projection_start, projection_end)

    meta.update(
        {
            "batch_ids": batch_ids,
            "camera_ids": camera_ids,
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities_packed,
        }
    )
    cnts = torch.bincount(camera_ids, minlength=C_global)
    cnts = cnts.split(C_world, dim=0)
    cnts = [cuts.sum() for cuts in cnts]
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "viewmats": viewmats,
        "Ks": Ks,
        "backgrounds": backgrounds,
        "width": int(width),
        "height": int(height),
        "tile_size": int(tile_size),
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": opacities_packed,
        "colors": colors_packed,
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "cnts": cnts,
        "N_world": N_world,
        "C_world": C_world,
        "C_global": C_global,
        "meta": meta,
        "projection_start": projection_start,
        "projection_end": projection_end,
    }


def _forward_exchange(
    state: Dict[str, object], stream: torch.cuda.Stream, microbatch_index: int
) -> None:
    """Run C1 on ``stream`` and mutate the prepared state in place."""

    world_size = torch.distributed.get_world_size()
    world_rank = torch.distributed.get_rank()
    device = state["means"].device
    cnts = state["cnts"]
    camera_ids = state["camera_ids"]
    gaussian_ids = state["gaussian_ids"]
    radii = state["radii"]
    means2d = state["means2d"]
    depths = state["depths"]
    conics = state["conics"]
    opacities = state["opacities"]
    colors = state["colors"]
    C_world = state["C_world"]
    N_world = state["N_world"]

    metadata_start = _cuda_event()
    metadata_end = _cuda_event()
    payload_start = _cuda_event()
    payload_end = _cuda_event()
    _nvtx_push(f"Tide:S1:MB{microbatch_index}:C1:Metadata")
    metadata_start.record(stream)
    collected_splits = all_to_all_int32(world_size, cnts, device=device)
    metadata_end.record(stream)
    _nvtx_pop()
    _nvtx_push(f"Tide:S1:MB{microbatch_index}:C1:Payload")
    payload_start.record(stream)
    (radii,) = all_to_all_tensor_list(
        world_size, [radii], cnts, output_splits=collected_splits
    )
    means2d, depths, conics, opacities, colors = all_to_all_tensor_list(
        world_size,
        [means2d, depths, conics, opacities, colors],
        cnts,
        output_splits=collected_splits,
    )
    offsets = torch.tensor(
        [0] + C_world[:-1], device=camera_ids.device, dtype=camera_ids.dtype
    )
    offsets = torch.cumsum(offsets, dim=0)
    offsets = offsets.repeat_interleave(torch.stack(cnts))
    camera_ids = camera_ids - offsets
    offsets = torch.tensor(
        [0] + N_world[:-1], device=gaussian_ids.device, dtype=gaussian_ids.dtype
    )
    offsets = torch.cumsum(offsets, dim=0)
    offsets = offsets.repeat_interleave(torch.stack(cnts))
    gaussian_ids = gaussian_ids + offsets
    camera_ids, gaussian_ids = all_to_all_tensor_list(
        world_size,
        [camera_ids, gaussian_ids],
        cnts,
        output_splits=collected_splits,
    )
    payload_end.record(stream)
    _nvtx_pop()
    state.update(
        {
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
            "colors": colors,
            "camera_ids": camera_ids,
            "gaussian_ids": gaussian_ids,
            "metadata_start": metadata_start,
            "metadata_end": metadata_end,
            "payload_start": payload_start,
            "payload_end": payload_end,
        }
    )
    C_local = int(C_world[world_rank])
    state["C_local"] = C_local
    state["meta"]["camera_ids"] = camera_ids
    state["meta"]["gaussian_ids"] = gaussian_ids
    state["meta"]["radii"] = radii
    state["meta"]["means2d"] = means2d
    state["meta"]["depths"] = depths
    state["meta"]["conics"] = conics
    state["meta"]["opacities"] = opacities
    state["meta"]["n_cameras"] = C_local
    if getattr(torch, "_tide_partial_grad_metrics", False) and getattr(
        torch, "_tide_partial_grad_metrics_sample", True
    ):
        state["meta"]["_tide_partial_gradient_context"] = torch._tide_begin_partial_gradient_metrics(
            means2d=means2d,
            depths=depths,
            conics=conics,
            opacity=opacities,
            features=colors,
            valid_mask=(radii > 0),
        )


def _render_prepared(state: Dict[str, object]) -> tuple[Tensor, Tensor, Dict[str, object]]:
    """Run the post-C1 packed rasterization stage on the compute stream."""

    means = state["means"]
    radii = state["radii"]
    means2d = state["means2d"]
    depths = state["depths"]
    conics = state["conics"]
    opacities = state["opacities"]
    colors = state["colors"]
    camera_ids = state["camera_ids"]
    gaussian_ids = state["gaussian_ids"]
    width = state["width"]
    height = state["height"]
    tile_size = state["tile_size"]
    C = int(state["C_local"])
    backgrounds = state["backgrounds"]
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=True,
        n_images=C,
        image_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(C, tile_height, tile_width)
    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=True,
        absgrad=False,
    )
    meta = state["meta"]
    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_batches": 1,
            "n_cameras": C,
        }
    )
    return render_colors, render_alphas, meta


@dataclass
class DistributedRasterizationHandle:
    pipeline: "DistributedRasterizationPipeline"
    slot: int
    microbatch_index: int
    state: Optional[Dict[str, object]] = None
    projection_done: Optional[torch.cuda.Event] = None
    exchange_done: Optional[torch.cuda.Event] = None
    render_done: Optional[torch.cuda.Event] = None
    c1_wait_start: Optional[torch.cuda.Event] = None
    c1_wait_end: Optional[torch.cuda.Event] = None
    rendered: Optional[Tensor] = None
    alphas: Optional[Tensor] = None
    meta: Optional[Dict[str, object]] = None
    _started: bool = False
    _rendered: bool = False

    def start_forward_exchange(self) -> "DistributedRasterizationHandle":
        if self.state is None or self._started:
            raise RuntimeError("projection handle is not ready or C1 was already started")
        stream = self.pipeline.comm_stream
        name = f"Tide:S1:MB{self.microbatch_index}:C1"
        self.state["c1_submit_ns"] = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            _nvtx_push(name + ":Wait")
            self.c1_wait_start = _cuda_event()
            self.c1_wait_end = _cuda_event()
            self.c1_wait_start.record(stream)
            stream.wait_event(self.projection_done)
            self.c1_wait_end.record(stream)
            _nvtx_pop()
            _forward_exchange(self.state, stream, self.microbatch_index)
            self.exchange_done = _cuda_event()
            self.exchange_done.record(stream)
        self._started = True
        return self

    def wait_and_render(self):
        if not self._started or self.state is None:
            raise RuntimeError("C1 must be started before render")
        if self._rendered:
            raise RuntimeError("render for this pipeline handle was already submitted")
        compute_stream = self.pipeline.compute_stream
        self.state["render_submit_ns"] = time.perf_counter_ns()
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(self.exchange_done)
            _nvtx_push(f"Tide:S1:MB{self.microbatch_index}:Render")
            self.rendered, self.alphas, self.meta = _render_prepared(self.state)
            _nvtx_pop()
            self.render_done = _cuda_event()
            self.render_done.record(compute_stream)
        self._rendered = True
        return self.rendered, self.alphas, self.meta

    def snapshot(self) -> Dict[str, object]:
        """Return timing-only state so completed tensors can be released."""

        if self.state is None:
            return {}
        state = self.state
        return {
            "microbatch_index": self.microbatch_index,
            "pipeline_buffer_slot": self.slot,
            "projection_start": state.get("projection_start"),
            "projection_end": state.get("projection_end"),
            "projection_submit_ns": state.get("projection_submit_ns", 0),
            "c1_submit_ns": state.get("c1_submit_ns", 0),
            "render_submit_ns": state.get("render_submit_ns", 0),
            "c1_wait_start": self.c1_wait_start,
            "c1_wait_end": self.c1_wait_end,
            "metadata_start": state.get("metadata_start"),
            "metadata_end": state.get("metadata_end"),
            "payload_start": state.get("payload_start"),
            "payload_end": state.get("payload_end"),
            "exchange_done": self.exchange_done,
            "render_done": self.render_done,
        }

    @staticmethod
    def timing_from_snapshot(snapshot: Dict[str, object]) -> Dict[str, float]:
        def elapsed(start, end):
            return float(start.elapsed_time(end)) if start is not None and end is not None else 0.0

        return {
            "projection_ms": elapsed(
                snapshot.get("projection_start"), snapshot.get("projection_end")
            ),
            "c1_wait_ms": elapsed(
                snapshot.get("c1_wait_start"), snapshot.get("c1_wait_end")
            ),
            "c1_metadata_ms": elapsed(
                snapshot.get("metadata_start"), snapshot.get("metadata_end")
            ),
            "c1_payload_ms": elapsed(
                snapshot.get("payload_start"), snapshot.get("payload_end")
            ),
            "c1_ms": elapsed(
                snapshot.get("metadata_start"), snapshot.get("payload_end")
            ),
            "render_ms": elapsed(
                snapshot.get("exchange_done"), snapshot.get("render_done")
            ),
        }

    def timing(self) -> Dict[str, float]:
        return self.timing_from_snapshot(self.snapshot())

    def overlap_with(self, next_handle: "DistributedRasterizationHandle") -> float:
        """Return GPU-time overlap between this handle's C1 and the next P."""

        if self.state is None or next_handle.state is None:
            return 0.0
        c1_start = self.state.get("metadata_start")
        c1_end = self.state.get("payload_end")
        p_start = next_handle.state.get("projection_start")
        p_end = next_handle.state.get("projection_end")
        if None in (c1_start, c1_end, p_start, p_end):
            return 0.0
        try:
            c1_ms = float(c1_start.elapsed_time(c1_end))
            p_ms = float(p_start.elapsed_time(p_end))
            p_start_offset = float(c1_start.elapsed_time(p_start))
            left = max(0.0, p_start_offset)
            right = min(c1_ms, p_start_offset + p_ms)
            return max(0.0, right - left)
        except RuntimeError:
            return 0.0


class DistributedRasterizationPipeline:
    """Double-buffered P/C1 pipeline for packed distributed rendering."""

    def __init__(self, *, compute_stream=None, comm_stream=None, buffer_count: int = 2):
        if not torch.cuda.is_available():
            raise RuntimeError("pc1 requires CUDA")
        self.compute_stream = compute_stream or torch.cuda.current_stream()
        self.comm_stream = comm_stream or torch.cuda.Stream(device=self.compute_stream.device)
        self.buffer_count = max(2, int(buffer_count))
        self._active = {}

    def submit_projection(self, *, microbatch_index: int = 0, **kwargs) -> DistributedRasterizationHandle:
        slot = int(microbatch_index) % self.buffer_count
        if slot in self._active:
            previous = self._active[slot]
            if previous.render_done is None:
                raise RuntimeError(f"pipeline buffer slot {slot} is still in use")
            # The render was queued on this same compute stream.  Enqueueing
            # the dependency before reusing the slot preserves the lifetime
            # rule without a host-side synchronize.
            self.compute_stream.wait_event(previous.render_done)
            self._active.pop(slot, None)
        handle = DistributedRasterizationHandle(self, slot, int(microbatch_index))
        with torch.cuda.stream(self.compute_stream):
            _nvtx_push(f"Tide:S1:MB{microbatch_index}:Projection")
            submit_ns = time.perf_counter_ns()
            handle.state = _projection_prepare(**kwargs)
            handle.state["projection_submit_ns"] = submit_ns
            handle.projection_done = _cuda_event()
            handle.projection_done.record(self.compute_stream)
            _nvtx_pop()
        self._active[slot] = handle
        return handle

    def close(self) -> None:
        """Drop Python references after all queued work has been submitted."""

        self._active.clear()

    def abort(self) -> None:
        """Release all Python-side handles after a stage raises.

        CUDA/NCCL work already submitted to a stream cannot be cancelled here;
        clearing the handles prevents stale state from being reused after the
        caller has synchronized the rank error.
        """

        self._active.clear()

    @staticmethod
    def timing_from_snapshot(snapshot: Dict[str, object]) -> Dict[str, float]:
        return DistributedRasterizationHandle.timing_from_snapshot(snapshot)


def distributed_rasterization_pipeline(**kwargs) -> DistributedRasterizationPipeline:
    """Construct the TideGS P/C1 pipeline."""

    return DistributedRasterizationPipeline(**kwargs)
