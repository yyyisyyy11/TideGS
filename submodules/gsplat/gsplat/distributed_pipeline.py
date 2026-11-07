"""Explicit C1/C2 transport and segmented backward for the Micro Pipeline.

Only the main thread owns graphs, handles and slots.  The persistent worker
receives detached immutable packets and publishes results plus GPU events.
The public synchronous rasterization/off path is independent of this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Dict, Optional

import torch
import torch.distributed
from torch import Tensor

from .cuda._wrapper import fully_fused_projection, isect_offset_encode, isect_tiles, rasterize_to_pixels, spherical_harmonics
from .distributed import all_gather_int32, all_gather_tensor_list
from .distributed_trace import MicrobatchTrace
from ._pipeline_worker import (
    CommunicationIdentity, FIFOCommWorker, remaining, wait_event, wait_future,
)


def _nvtx_push(name: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def _nvtx_pop() -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


@contextmanager
def _nvtx_range(name):
    _nvtx_push(name)
    try:
        yield
    finally:
        _nvtx_pop()


def _record_stream_recursive(value, stream, seen=None) -> None:
    """Tell the CUDA allocator that tensors cross into ``stream``."""

    if seen is None:
        seen = set()
    if torch.is_tensor(value):
        identity = id(value)
        if identity not in seen:
            value.record_stream(stream)
            seen.add(identity)
        return
    if isinstance(value, dict):
        for item in value.values():
            _record_stream_recursive(item, stream, seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _record_stream_recursive(item, stream, seen)


def _cuda_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _mask_radii(radii, keep):
    """Apply a packed row mask to scalar or modern per-axis radii."""
    mask = keep.unsqueeze(-1) if radii.ndim == keep.ndim + 1 else keep
    return torch.where(mask, radii, torch.zeros_like(radii))


@dataclass
class BatchCameraLayout:
    """Rank-major camera metadata gathered once for one training batch."""

    N_world: list[int]
    C_world: list[int]
    viewmats_global: Tensor
    Ks_global: Tensor
    local_camera_count: int
    microbatch: int

    @property
    def world_size(self) -> int:
        return len(self.N_world)

    @property
    def global_camera_count(self) -> int:
        return int(sum(self.C_world))

    @property
    def microbatch_C_world(self) -> list[int]:
        """Camera counts matching one rank-major microbatch camera slice."""

        return [int(self.microbatch)] * self.world_size

    def microbatch_cameras(self, microbatch_index: int) -> tuple[Tensor, Tensor]:
        start = int(microbatch_index) * int(self.microbatch)
        end = start + int(self.microbatch)
        if start < 0 or end > self.local_camera_count:
            raise IndexError(
                f"microbatch {microbatch_index} is outside local camera layout"
            )
        # all_gather_tensor_list concatenates rank-major.  Reshaping before
        # slicing preserves the same local microbatch on every rank.
        viewmats = self.viewmats_global.reshape(
            self.world_size, self.local_camera_count, 4, 4
        )[:, start:end].reshape(-1, 4, 4)
        Ks = self.Ks_global.reshape(
            self.world_size, self.local_camera_count, 3, 3
        )[:, start:end].reshape(-1, 3, 3)
        return viewmats, Ks


def prepare_batch_camera_layout(
    *, means: Tensor, viewmats: Tensor, Ks: Tensor, microbatch: int
) -> BatchCameraLayout:
    """Gather all static projection metadata before the Micro Pipeline loop starts."""

    if means.device.type != "cuda":
        raise ValueError("Micro Pipeline requires CUDA tensors")
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "Micro Pipeline requires an initialized torch.distributed process group"
        )
    if viewmats.ndim != 3 or viewmats.shape[-2:] != (4, 4):
        raise ValueError("Micro Pipeline requires batch viewmats with shape [C,4,4]")
    if Ks.ndim != 3 or Ks.shape[-2:] != (3, 3):
        raise ValueError("Micro Pipeline requires batch intrinsics with shape [C,3,3]")
    local_camera_count = int(viewmats.shape[0])
    microbatch = int(microbatch)
    if local_camera_count <= 0 or microbatch <= 0:
        raise ValueError("Micro Pipeline requires positive camera and microbatch counts")
    if local_camera_count % microbatch:
        raise ValueError("Micro Pipeline local camera count must be divisible by microbatch")

    world_size = torch.distributed.get_world_size()
    device = means.device
    with _nvtx_range("tide.comm.micro.layout.counts"):
        N_world = [int(value) for value in all_gather_int32(
            world_size, int(means.shape[0]), device=device
        )]
        C_world = [int(value) for value in all_gather_int32(
            world_size, local_camera_count, device=device
        )]
    if any(count != local_camera_count for count in C_world):
        raise ValueError(
            "Micro Pipeline requires the same number of local cameras on every rank; "
            f"got {C_world}"
        )
    with _nvtx_range("tide.comm.micro.layout.cameras"):
        viewmats_global, Ks_global = all_gather_tensor_list(
            world_size, [viewmats, Ks]
        )
    return BatchCameraLayout(
        N_world=N_world,
        C_world=C_world,
        viewmats_global=viewmats_global,
        Ks_global=Ks_global,
        local_camera_count=local_camera_count,
        microbatch=microbatch,
    )


def _projection_prepare(
    *,
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    viewmats_global: Tensor,
    Ks_global: Tensor,
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
    N_world: list[int],
    C_world: list[int],
) -> Dict[str, object]:
    """Run P(i) and return tensors immediately before C1(i)."""

    if means.ndim != 2 or quats.ndim != 2 or scales.ndim != 2:
        raise ValueError("Micro Pipeline requires unbatched Gaussian tensors")
    if not torch.is_tensor(viewmats) or viewmats.ndim != 3:
        raise ValueError("Micro Pipeline requires [C,4,4] viewmats")
    if not torch.is_tensor(Ks) or Ks.ndim != 3:
        raise ValueError("Micro Pipeline requires [C,3,3] intrinsics")
    if means.device.type != "cuda":
        raise ValueError("Micro Pipeline requires CUDA tensors")
    if colors.ndim != 3 or colors.shape[0] != means.shape[0] or colors.shape[-1] != 3:
        raise ValueError("Micro Pipeline requires per-Gaussian SH colors with shape [N,K,3]")
    if sh_degree is None:
        raise ValueError("Micro Pipeline requires an active SH degree")
    if rasterize_mode != "classic":
        raise ValueError("Micro Pipeline currently supports rasterize_mode='classic' only")

    device = means.device
    N = int(means.shape[0])
    C_local = int(viewmats.shape[0])
    if C_local <= 0:
        raise ValueError("Micro Pipeline requires at least one local camera")
    world_size = len(N_world)
    if len(C_world) != world_size or any(int(count) <= 0 for count in C_world):
        raise ValueError("Micro Pipeline received an invalid microbatch camera layout")
    if world_size != torch.distributed.get_world_size():
        raise ValueError("Micro Pipeline batch camera layout world size changed during the batch")
    world_rank = torch.distributed.get_rank()
    if int(N_world[world_rank]) != N:
        raise ValueError(
            "Micro Pipeline local Gaussian count changed after batch layout preparation"
        )
    if int(C_world[world_rank]) != C_local:
        raise ValueError(
            "Micro Pipeline local camera count does not match microbatch camera layout"
        )
    if int(sum(C_world)) != int(viewmats_global.shape[0]):
        raise ValueError("Micro Pipeline microbatch camera layout does not match gathered tensors")
    if int(sum(C_world)) != int(Ks_global.shape[0]):
        raise ValueError("Micro Pipeline microbatch intrinsics layout does not match gathered tensors")
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
            radii = _mask_radii(radii, keep)

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


FLOAT_ATTRIBUTES = ("means2d", "depths", "conics", "opacities", "colors")


@dataclass(frozen=True)
class FloatColumn:
    name: str
    shape: tuple[int, ...]
    width: int


@dataclass(frozen=True)
class ExchangeLayout:
    send_splits: tuple[int, ...]
    recv_splits: tuple[int, ...]
    columns: tuple[FloatColumn, ...]


@dataclass(frozen=True)
class C1Request:
    identity: CommunicationIdentity
    deadline: float
    projection_done: object
    floats: tuple[Tensor, ...]
    radii: Tensor
    camera_ids: Tensor
    gaussian_ids: Tensor
    counts: tuple[Tensor, ...]
    camera_counts: tuple[int, ...]
    gaussian_counts: tuple[int, ...]


@dataclass(frozen=True)
class C1Result:
    identity: CommunicationIdentity
    layout: ExchangeLayout
    floats: tuple[Tensor, ...]
    radii: Tensor
    camera_ids: Tensor
    gaussian_ids: Tensor
    done_event: object
    retained: tuple[object, ...]
    events: tuple[tuple[str, object], ...]
    worker_begin_ns: int
    worker_end_ns: int
    thread_id: int
    native_thread_id: int


@dataclass(frozen=True)
class C2Request:
    identity: CommunicationIdentity
    deadline: float
    intermediate_done: object
    gradients: tuple[Tensor, ...]
    layout: ExchangeLayout


@dataclass(frozen=True)
class C2Result:
    identity: CommunicationIdentity
    layout: ExchangeLayout
    gradients: tuple[Tensor, ...]
    done_event: object
    retained: tuple[object, ...]
    events: tuple[tuple[str, object], ...]
    worker_begin_ns: int
    worker_end_ns: int
    thread_id: int
    native_thread_id: int


def _columns(tensors) -> tuple[FloatColumn, ...]:
    return tuple(FloatColumn(name, tuple(tensor.shape[1:]),
                             math.prod(tensor.shape[1:]))
                 for name, tensor in zip(FLOAT_ATTRIBUTES, tensors))


def _pack_floats(tensors, columns):
    rows = int(tensors[0].shape[0])
    dtype = tensors[0].dtype
    for tensor in tensors:
        if int(tensor.shape[0]) != rows or tensor.dtype != dtype:
            raise ValueError("Micro Pipeline float payload must have a common row count and dtype")
    # Explicit widths also support packed zero-row projections.
    return torch.cat([tensor.reshape(rows, column.width)
                      for tensor, column in zip(tensors, columns)], dim=1).contiguous()


def _unpack_floats(packed, columns):
    rows = int(packed.shape[0])
    return tuple(piece.reshape(rows, *column.shape)
                 for piece, column in zip(packed.split(
                     [column.width for column in columns], dim=1), columns))


def _raw_exchange(data, send_splits, recv_splits, retained):
    """Submit non-autograd NCCL, establish readiness on the caller stream."""
    output = data.new_empty((sum(recv_splits), *data.shape[1:]))
    retained.extend((data, output))
    if len(send_splits) == 1:
        output.copy_(data)
    else:
        work = torch.distributed.all_to_all_single(
            output, data, output_split_sizes=list(recv_splits),
            input_split_sizes=list(send_splits), async_op=True)
        retained.append(work)
        # NCCL Work.wait() inserts the dependency on the *current comm stream*.
        # Future completion below is submission completion, not device completion.
        work.wait()
    return output


def _identity_range(identity, suffix=""):
    return (f"Tide:S1:MB{identity.microbatch_index}:{identity.stage}{suffix}"
            f"|rank={identity.rank},batch={identity.batch_index},slot={identity.slot},"
            f"seq={identity.comm_sequence},tid={threading.get_native_id()},"
            f"schedule={identity.schedule_version}")


class PipelineCommWorker(FIFOCommWorker):
    """CUDA backend for the context-owned, persistent FIFO communication worker."""

    def __init__(self, *, device, comm_stream=None, buffer_count=2,
                 on_failure=None, default_timeout_s=120.0):
        self.device = torch.device(device)
        self.comm_stream = comm_stream or torch.cuda.Stream(device=self.device)
        self._transport_inflight = []
        super().__init__(execute=self._execute_request,
                         initialize=lambda: torch.cuda.set_device(self.device),
                         buffer_count=buffer_count, on_failure=on_failure,
                         default_timeout_s=default_timeout_s,
                         name=f"micro-pipeline-comm-{self.device.index}")

    def _execute_request(self, stage, request):
        remaining(request.deadline, f"{stage} worker submission")
        with self._lock:
            self._ensure_healthy()
            self._transport_inflight = [request]
        with torch.cuda.device(self.device), torch.cuda.stream(self.comm_stream), torch.no_grad():
            with _nvtx_range(_identity_range(request.identity)):
                result = self._c1(request) if stage == "C1" else self._c2(request)
        with self._lock:
            self._transport_inflight = []
        return result

    def abort(self, error):
        # Keep the *mutable list* before super.abort(): a native submission may
        # still be running and append another NCCL buffer after main-thread
        # abort. Never clear this lease on an exception or synchronize here.
        with self._lock:
            if self._transport_inflight and not any(
                    item is self._transport_inflight for item in self._quarantine):
                self._quarantine.append(self._transport_inflight)
        super().abort(error)

    def _c1(self, request):
        stream = self.comm_stream
        begin = time.perf_counter_ns()
        retained = self._transport_inflight
        events = {}
        for name in ("c1_wait_start", "c1_wait_end", "metadata_start", "metadata_end",
                     "payload_start", "payload_end"):
            events[name] = _cuda_event()
        with _nvtx_range(_identity_range(request.identity, ":Wait")):
            events["c1_wait_start"].record(stream)
            stream.wait_event(request.projection_done)
            events["c1_wait_end"].record(stream)
        _record_stream_recursive((request.floats, request.radii, request.camera_ids,
                                  request.gaussian_ids, request.counts), stream)
        with _nvtx_range(_identity_range(request.identity, ":Metadata")):
            events["metadata_start"].record(stream)
            counts = torch.stack(request.counts).to(dtype=torch.int64).contiguous()
            received = counts.new_empty(counts.shape)
            retained.extend((counts, received))
            if len(request.counts) == 1:
                received.copy_(counts)
            else:
                work = torch.distributed.all_to_all_single(received, counts, async_op=True)
                retained.append(work)
                work.wait()
            # One host conversion for all send and receive sizes, cached in the
            # result and reused unchanged by C2. No per-payload scalar .item().
            split_values = torch.cat((counts, received)).cpu().tolist()
            world_size = len(request.counts)
            send_splits = tuple(int(value) for value in split_values[:world_size])
            recv_splits = tuple(int(value) for value in split_values[world_size:])
            columns = _columns(request.floats)
            layout = ExchangeLayout(send_splits, recv_splits, columns)
            events["metadata_end"].record(stream)
        remaining(request.deadline, "C1 metadata")
        with _nvtx_range(_identity_range(request.identity, ":Payload")):
            events["payload_start"].record(stream)
            radii = _raw_exchange(request.radii.contiguous(), send_splits, recv_splits, retained)
            packed = _pack_floats(request.floats, columns)
            packed_received = _raw_exchange(packed, send_splits, recv_splits, retained)
            floats = _unpack_floats(packed_received, columns)
            rows = int(request.camera_ids.shape[0])
            repeats = counts
            camera_offsets = torch.tensor(
                [0] + list(request.camera_counts[:-1]), device=self.device,
                dtype=request.camera_ids.dtype).cumsum(0)
            gaussian_offsets = torch.tensor(
                [0] + list(request.gaussian_counts[:-1]), device=self.device,
                dtype=request.gaussian_ids.dtype).cumsum(0)
            camera_ids = request.camera_ids - camera_offsets.repeat_interleave(
                repeats, output_size=rows)
            gaussian_ids = request.gaussian_ids + gaussian_offsets.repeat_interleave(
                repeats, output_size=rows)
            # IDs keep their integer dtype; radii likewise remain a separate payload.
            ids = torch.stack((camera_ids, gaussian_ids), dim=1).contiguous()
            received_ids = _raw_exchange(ids, send_splits, recv_splits, retained)
            events["payload_end"].record(stream)
        done = _cuda_event()
        done.record(stream)
        _record_stream_recursive((floats, radii, received_ids), stream)
        return C1Result(request.identity, layout, floats, radii,
                        received_ids[:, 0], received_ids[:, 1], done,
                        tuple(retained), tuple(events.items()), begin,
                        time.perf_counter_ns(), threading.get_ident(),
                        threading.get_native_id())

    def _c2(self, request):
        stream = self.comm_stream
        begin = time.perf_counter_ns()
        retained = self._transport_inflight
        events = {name: _cuda_event() for name in
                  ("c2_wait_start", "c2_wait_end", "c2_payload_start", "c2_payload_end")}
        with _nvtx_range(_identity_range(request.identity, ":Wait")):
            events["c2_wait_start"].record(stream)
            stream.wait_event(request.intermediate_done)
            events["c2_wait_end"].record(stream)
        _record_stream_recursive(request.gradients, stream)
        with _nvtx_range(_identity_range(request.identity, ":Payload")):
            events["c2_payload_start"].record(stream)
            packed = _pack_floats(request.gradients, request.layout.columns)
            received = _raw_exchange(packed, request.layout.recv_splits,
                                     request.layout.send_splits, retained)
            gradients = _unpack_floats(received, request.layout.columns)
            events["c2_payload_end"].record(stream)
        done = _cuda_event()
        done.record(stream)
        _record_stream_recursive(gradients, stream)
        return C2Result(request.identity, request.layout, gradients, done,
                        tuple(retained), tuple(events.items()), begin,
                        time.perf_counter_ns(), threading.get_ident(),
                        threading.get_native_id())


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
    projection_outputs: Dict[str, Tensor] = field(default_factory=dict)
    image_leaves: Dict[str, Tensor] = field(default_factory=dict)
    projection_done: object = None
    exchange_done: object = None
    intermediate_start: object = None
    intermediate_done: object = None
    final_start: object = None
    render_start: object = None
    render_done: object = None
    backward_done: object = None
    c1_wait_start: object = None
    c1_wait_end: object = None
    c1_future: object = None
    c2_future: object = None
    c1_result: Optional[C1Result] = None
    c2_result: Optional[C2Result] = None
    rendered: Optional[Tensor] = None
    alphas: Optional[Tensor] = None
    meta: Optional[Dict[str, object]] = None
    trace: Optional[MicrobatchTrace] = None
    phase: str = "PROJECTED"
    cpu_trace: list = field(default_factory=list)
    _events: dict = field(default_factory=dict)
    _identities: dict = field(default_factory=dict)

    def _check(self, expected):
        self.pipeline._ensure_main_thread()
        self.pipeline._ensure_not_aborted()
        if self.phase != expected:
            raise RuntimeError(f"Micro Pipeline MB{self.microbatch_index} requires {expected}; got {self.phase}")

    def _identity(self, stage):
        identity = CommunicationIdentity(
            self.pipeline.rank, self.pipeline.batch_index, self.microbatch_index,
            self.slot, stage, self.pipeline.comm_worker.next_sequence())
        self._identities[stage] = identity
        return identity

    @contextmanager
    def _cpu_span(self, name, identity=None):
        start = time.perf_counter_ns()
        sequence = identity.comm_sequence if identity else -1
        range_name = (f"Tide:S1:MB{self.microbatch_index}:{name}"
                      f"|rank={self.pipeline.rank},batch={self.pipeline.batch_index},"
                      f"slot={self.slot},seq={sequence},tid={threading.get_native_id()},"
                      "schedule=micro-pipeline-window-v2")
        with _nvtx_range(range_name):
            try:
                yield
            finally:
                self.cpu_trace.append({
                    "name": name, "start_ns": start, "end_ns": time.perf_counter_ns(),
                    "thread_id": threading.get_native_id(),
                    "python_thread_id": threading.get_ident(), "thread_role": "main",
                    "rank": self.pipeline.rank, "batch_index": self.pipeline.batch_index,
                    "microbatch_index": self.microbatch_index, "slot": self.slot,
                    "communication_sequence": identity.comm_sequence if identity else -1,
                    "schedule_version": "micro-pipeline-window-v2",
                })

    def start_forward_exchange(self, deadline=None):
        self._check("PROJECTED")
        deadline = self.pipeline._deadline(deadline)
        identity = self._identity("C1")
        request = C1Request(
            identity, deadline, self.projection_done,
            tuple(self.projection_outputs[name].detach() for name in FLOAT_ATTRIBUTES),
            self.state["radii"].detach(), self.state["camera_ids"].detach(),
            self.state["gaussian_ids"].detach(), tuple(self.state["cnts"]),
            tuple(self.state["C_world"]), tuple(self.state["N_world"]))
        self.state["c1_submit_ns"] = time.perf_counter_ns()
        with self._cpu_span("C1:Enqueue", identity):
            self.c1_future = self.pipeline.comm_worker.enqueue_c1(request)
        self.phase = "C1_QUEUED"
        return self

    def _result(self, future, deadline, stage):
        try:
            with self._cpu_span(f"{stage}:FutureWait", self._identities.get(stage)):
                result = wait_future(future, self.pipeline._deadline(deadline))
        except BaseException as error:
            self.pipeline.abort(error)
            raise
        identity = result.identity
        if (identity.batch_index, identity.microbatch_index, identity.slot) != (
                self.pipeline.batch_index, self.microbatch_index, self.slot):
            raise RuntimeError("Micro Pipeline received a communication result for a different slot")
        self.cpu_trace.append({
            "name": f"{stage}:Worker", "start_ns": result.worker_begin_ns,
            "end_ns": result.worker_end_ns, "thread_id": result.native_thread_id,
            "python_thread_id": result.thread_id, "thread_role": "comm",
            "rank": identity.rank, "batch_index": identity.batch_index,
            "microbatch_index": identity.microbatch_index, "slot": identity.slot,
            "communication_sequence": identity.comm_sequence,
            "schedule_version": identity.schedule_version,
        })
        return result

    def attach_c1_result(self, result):
        self._check("C1_QUEUED")
        self.c1_result = result
        self.exchange_done = result.done_event
        self._events.update(result.events)
        self.c1_wait_start = self._events["c1_wait_start"]
        self.c1_wait_end = self._events["c1_wait_end"]
        compute_stream = self.pipeline.compute_stream
        # Future only means CPU publication. Always establish the GPU dependency,
        # even when Future.done() already returned True.
        compute_stream.wait_event(result.done_event)
        _record_stream_recursive((result.floats, result.radii, result.camera_ids,
                                  result.gaussian_ids), compute_stream)
        with torch.cuda.stream(compute_stream):
            self.image_leaves = {name: tensor.detach().requires_grad_(True)
                                 for name, tensor in zip(FLOAT_ATTRIBUTES, result.floats)}
        self.state.update(self.image_leaves)
        self.state.update(radii=result.radii, camera_ids=result.camera_ids,
                          gaussian_ids=result.gaussian_ids,
                          C_local=int(self.state["C_world"][self.pipeline.rank]))
        self.state["meta"].update({name: self.state[name] for name in
            ("camera_ids", "gaussian_ids", "radii", "means2d", "depths", "conics", "opacities")})
        self.state["meta"]["n_cameras"] = self.state["C_local"]
        if getattr(torch, "_tide_partial_grad_metrics", False) and getattr(
                torch, "_tide_partial_grad_metrics_sample", True):
            with torch.cuda.stream(compute_stream):
                self.state["meta"]["_tide_partial_gradient_context"] = torch._tide_begin_partial_gradient_metrics(
                    means2d=self.image_leaves["means2d"], depths=self.image_leaves["depths"],
                    conics=self.image_leaves["conics"], opacity=self.image_leaves["opacities"],
                    features=self.image_leaves["colors"], valid_mask=(result.radii > 0))
        self.phase = "C1_ATTACHED"

    def wait_and_render(self, deadline=None):
        self._check("C1_QUEUED")
        result = self._result(self.c1_future, deadline, "C1")
        self.attach_c1_result(result)
        compute_stream = self.pipeline.compute_stream
        self.state["render_submit_ns"] = time.perf_counter_ns()
        with torch.cuda.stream(compute_stream), self._cpu_span("Render"):
            self.render_start = _cuda_event()
            self.render_start.record(compute_stream)
            self.rendered, self.alphas, self.meta = _render_prepared(self.state)
            self.render_done = _cuda_event()
            self.render_done.record(compute_stream)
        self.phase = "RENDERED"
        return self.rendered, self.alphas, self.meta

    def trace_backward(self, loss):
        self._check("RENDERED")
        if self.trace is not None:
            self.trace.watch_backward(loss)

    def intermediate_backward(self, loss, deadline=None):
        self._check("RENDERED")
        deadline = self.pipeline._deadline(deadline)
        self.state["intermediate_submit_ns"] = time.perf_counter_ns()
        stream = self.pipeline.compute_stream
        with torch.cuda.stream(stream), self._cpu_span("IntermediateGrad"):
            self.intermediate_start = _cuda_event()
            self.intermediate_start.record(stream)
            if self.trace is not None:
                self.trace.watch_backward(loss)
            leaves = tuple(self.image_leaves[name] for name in FLOAT_ATTRIBUTES)
            if loss.requires_grad:
                gradients = torch.autograd.grad(loss, leaves, allow_unused=True,
                                                create_graph=False, retain_graph=False)
            else:
                gradients = (None,) * len(leaves)
            gradients = tuple(torch.zeros_like(leaf) if gradient is None else gradient.detach()
                              for leaf, gradient in zip(leaves, gradients))
            self.intermediate_done = _cuda_event()
            self.intermediate_done.record(stream)
            # Hook observers must not retain the completed image graph until G.
            if self.trace is not None:
                self.trace.remove_hooks()
        remaining(deadline, "intermediate gradient")
        identity = self._identity("C2")
        request = C2Request(identity, deadline, self.intermediate_done,
                            gradients, self.c1_result.layout)
        self.state["c2_submit_ns"] = time.perf_counter_ns()
        with self._cpu_span("C2:Enqueue", identity):
            self.c2_future = self.pipeline.comm_worker.enqueue_c2(request)
        self.rendered = self.alphas = None
        self.phase = "C2_QUEUED"
        return self.c2_future

    def final_backward(self, deadline=None):
        self._check("C2_QUEUED")
        deadline = self.pipeline._deadline(deadline)
        result = self._result(self.c2_future, deadline, "C2")
        self.c2_result = result
        self._events.update(result.events)
        stream = self.pipeline.compute_stream
        with torch.cuda.stream(stream):
            stream.wait_event(result.done_event)
            _record_stream_recursive(result.gradients, stream)
            self.final_start = _cuda_event()
            self.final_start.record(stream)
            with self._cpu_span("FinalGrad"):
                pairs = [(self.projection_outputs[name], gradient)
                         for name, gradient in zip(FLOAT_ATTRIBUTES, result.gradients)
                         if self.projection_outputs[name].requires_grad]
                if self.trace is not None:
                    self.trace.watch_backward([pair[0] for pair in pairs], default_stage="FinalGrad")
                if pairs:
                    torch.autograd.backward([pair[0] for pair in pairs],
                                            grad_tensors=[pair[1] for pair in pairs])
            done = _cuda_event()
            done.record(stream)
        remaining(deadline, "final gradient")
        self.mark_backward_done(done)
        return done

    def mark_backward_done(self, event):
        self._check("C2_QUEUED")
        if self.c2_result is None:
            raise RuntimeError("Micro Pipeline final gradient must consume explicit C2 before slot release")
        self.backward_done = event
        self.phase = "BACKWARD_SUBMITTED"
        if self.trace is not None:
            self.trace.close()

    def snapshot(self):
        if self.state is None:
            return {}
        snapshot = {
            "rank": self.pipeline.rank, "batch_index": self.pipeline.batch_index,
            "microbatch_index": self.microbatch_index,
            "pipeline_buffer_slot": self.slot, "schedule_version": "micro-pipeline-window-v2",
            "projection_start": self.state.get("projection_start"),
            "projection_end": self.state.get("projection_end"),
            "projection_done": self.projection_done,
            "projection_submit_ns": self.state.get("projection_submit_ns", 0),
            "c1_submit_ns": self.state.get("c1_submit_ns", 0),
            "render_submit_ns": self.state.get("render_submit_ns", 0),
            "intermediate_submit_ns": self.state.get("intermediate_submit_ns", 0),
            "c2_submit_ns": self.state.get("c2_submit_ns", 0),
            "c1_wait_start": self.c1_wait_start, "c1_wait_end": self.c1_wait_end,
            "exchange_done": self.exchange_done,
            "render_start": self.render_start, "render_done": self.render_done,
            "intermediate_start": self.intermediate_start,
            "intermediate_done": self.intermediate_done, "final_start": self.final_start,
            "c2_done": self.c2_result.done_event if self.c2_result is not None else None,
            "backward_done": self.backward_done,
            "backward_trace": self.trace.snapshot() if self.trace is not None else [],
            "cpu_trace": [dict(record) for record in self.cpu_trace],
        }
        snapshot.update(self._events)
        return snapshot

    @staticmethod
    def timing_from_snapshot(snapshot):
        def elapsed(start, end):
            return float(start.elapsed_time(end)) if start is not None and end is not None else 0.0
        return {name: elapsed(snapshot.get(start), snapshot.get(end))
                for name, start, end in (
                    ("projection_ms", "projection_start", "projection_end"),
                    ("c1_wait_ms", "c1_wait_start", "c1_wait_end"),
                    ("c1_metadata_ms", "metadata_start", "metadata_end"),
                    ("c1_payload_ms", "payload_start", "payload_end"),
                    ("c1_ms", "metadata_start", "payload_end"),
                    ("render_ms", "render_start", "render_done"),
                    ("intermediate_grad_ms", "intermediate_start", "intermediate_done"),
                    ("c2_wait_ms", "c2_wait_start", "c2_wait_end"),
                    ("c2_payload_ms", "c2_payload_start", "c2_payload_end"),
                    ("final_grad_ms", "final_start", "backward_done"))}

    def timing(self):
        return self.timing_from_snapshot(self.snapshot())

    def overlap_with(self, next_handle):
        left, right = self.snapshot(), next_handle.snapshot()
        anchor, end = left.get("metadata_start"), left.get("payload_end")
        start, finish = right.get("projection_start"), right.get("projection_end")
        if any(event is None for event in (anchor, end, start, finish)):
            return 0.0
        offset = float(anchor.elapsed_time(start))
        return max(0.0, min(float(anchor.elapsed_time(end)),
                            offset + float(start.elapsed_time(finish))) - max(0.0, offset))

    def _buffer_lease(self):
        # Results/requests contain detached transport tensors only. Recording them
        # on the compute stream protects the allocator; explicit leases additionally
        # retain NCCL Work and buffers until final consumption completes.
        return (self.c1_result, self.c2_result,
                tuple(tensor.detach() for tensor in self.projection_outputs.values()),
                tuple(self.image_leaves.values()))


    def _release_graph(self):
        self.projection_outputs.clear()
        self.image_leaves.clear()
        self.state = self.meta = self.rendered = self.alphas = None
        self.c1_result = self.c2_result = self.c1_future = self.c2_future = None



class DistributedRasterizationPipeline:
    """Main-thread-owned per-batch slots using one context-owned comm worker."""

    def __init__(self, *, compute_stream=None, comm_stream=None, buffer_count=2,
                 capture_timeline=False, comm_worker=None, batch_index=0, deadline=None):
        if not torch.cuda.is_available():
            raise RuntimeError("Micro Pipeline requires CUDA")
        if int(buffer_count) <= 0:
            raise ValueError("Micro Pipeline buffer_count must be positive")
        self.compute_stream = compute_stream or torch.cuda.current_stream()
        self.buffer_count = int(buffer_count)
        self.capture_timeline = bool(capture_timeline)
        self.batch_index = int(batch_index)
        self.rank = torch.distributed.get_rank()
        self.deadline = deadline
        self._owner_thread = threading.get_ident()
        self._owns_worker = comm_worker is None
        self.comm_worker = comm_worker or PipelineCommWorker(
            device=self.compute_stream.device, comm_stream=comm_stream,
            buffer_count=self.buffer_count)
        self.comm_stream = self.comm_worker.comm_stream
        if self.comm_worker.buffer_count != self.buffer_count:
            raise ValueError("Micro Pipeline context worker buffer count does not match pipeline")
        self._active = {}
        self._retired = []
        self._drained = self._closed = self._aborted = False
        self._quarantine = []

    def _deadline(self, deadline=None):
        return self.comm_worker.resolve_deadline(self.deadline if deadline is None else deadline)

    def _ensure_main_thread(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Micro Pipeline graph and slot state belongs to the main thread")

    def _ensure_not_aborted(self):
        if self._aborted:
            raise RuntimeError("Micro Pipeline aborted; worker must exit, NCCL batch cannot be resumed")

    def submit_projection(self, *, microbatch_index=0, deadline=None, **kwargs):
        self._ensure_main_thread()
        self._ensure_not_aborted()
        if self._closed or self._drained:
            raise RuntimeError("cannot submit projection after pipeline shutdown")
        if deadline is not None:
            remaining(deadline, "projection begin")
        slot = int(microbatch_index) % self.buffer_count
        if slot in self._active:
            previous = self._active[slot]
            if previous.phase != "BACKWARD_SUBMITTED" or previous.backward_done is None:
                raise RuntimeError(f"pipeline buffer slot {slot} is still in use; final backward is required")
            self.compute_stream.wait_event(previous.backward_done)
            # Saved tensors have been consumed by backward. Keep only buffer
            # leases until its event completes, never retain an old training graph.
            self._retired.append((previous.backward_done, previous._buffer_lease()))
            previous._release_graph()
            self._active.pop(slot)
        self._retired = [(event, lease) for event, lease in self._retired if not event.query()]
        handle = DistributedRasterizationHandle(self, slot, int(microbatch_index))
        if self.capture_timeline:
            handle.trace = MicrobatchTrace(kwargs["means"].device, microbatch_index,
                rank=self.rank, batch_index=self.batch_index, slot=slot)
        with torch.cuda.stream(self.compute_stream), handle._cpu_span("Projection"):
            submit_ns = time.perf_counter_ns()
            handle.state = _projection_prepare(**kwargs)
            handle.projection_outputs = {name: handle.state[name] for name in FLOAT_ATTRIBUTES}
            handle.state["projection_submit_ns"] = submit_ns
            handle.projection_done = _cuda_event()
            handle.projection_done.record(self.compute_stream)
        self._active[slot] = handle
        return handle

    def drain(self, deadline=None):
        self._ensure_main_thread()
        self._ensure_not_aborted()
        if self._drained:
            return
        deadline = self._deadline(deadline)
        try:
            for handle in self._active.values():
                if handle.phase != "BACKWARD_SUBMITTED":
                    raise RuntimeError("Micro Pipeline drain requires final backward for every active slot")
            self.comm_worker.drain(deadline)
            for handle in self._active.values():
                wait_event(handle.backward_done, deadline, "pipeline final backward drain")
            for event, _ in self._retired:
                wait_event(event, deadline, "pipeline retired buffer drain")
            self._retired.clear()
            self._drained = True
        except BaseException as error:
            self.abort(error)
            raise

    def close(self, deadline=None):
        self._ensure_main_thread()
        self._ensure_not_aborted()
        if self._closed:
            return
        if not self._drained:
            raise RuntimeError("pipeline.close() requires pipeline.drain() first")
        for handle in self._active.values():
            handle._release_graph()
        self._active.clear()
        if self._owns_worker:
            self.comm_worker.close(self._deadline(deadline))
        self._closed = True

    def abort(self, error=None):
        if self._aborted:
            return
        self._aborted = True
        for handle in self._active.values():
            if handle.trace is not None:
                handle.trace.abort()
        self._quarantine.extend(self._active.values())
        self._quarantine.extend(self._retired)
        self._active.clear()
        self._retired.clear()
        self.comm_worker.abort(error)

    @staticmethod
    def timing_from_snapshot(snapshot):
        return DistributedRasterizationHandle.timing_from_snapshot(snapshot)


def distributed_rasterization_pipeline(**kwargs):
    return DistributedRasterizationPipeline(**kwargs)
