"""Version-gated gsplat entry points used by the distributed TideGS path."""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Iterable, Tuple

import torch


MIN_DISTRIBUTED_GSPLAT = (1, 5, 3)
PACKED_DISTRIBUTED_FIX = "image_ids = camera_ids"
PROJECTION_TIMING_FIX = "_tide_projection_events"
OWNER_PROJECTION_SURVIVOR_FIX = "_tide_owner_projection_survivor_mask"


def _version_tuple(value: str) -> Tuple[int, ...]:
    parts = []
    for token in str(value).split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _require_packed_distributed_fix(gsplat, version: Tuple[int, ...]) -> None:
    if version[:3] != (1, 5, 3):
        return
    try:
        source = inspect.getsource(gsplat.rasterization)
    except (OSError, TypeError):
        return
    if PACKED_DISTRIBUTED_FIX not in source:
        raise RuntimeError(
            "gsplat 1.5.3 has an unsafe packed distributed rasterization path. "
            "Run `python tools/patch_gsplat_distributed_packed.py` in the TideGS "
            "environment before launching distributed training."
        )


def _require_projection_timing_fix(gsplat) -> None:
    try:
        source = inspect.getsource(gsplat.rasterization)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "Detailed TideGS metrics require inspectable gsplat rasterization source"
        ) from exc
    if PROJECTION_TIMING_FIX not in source:
        raise RuntimeError(
            "Detailed TideGS metrics require the gsplat projection timing patch. "
            "Run `python tools/patch_gsplat_distributed_packed.py` in the TideGS "
            "environment before launching training."
        )


def _require_owner_projection_survivor_fix(gsplat) -> None:
    try:
        source = inspect.getsource(gsplat.rasterization)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "Gradient-zero profiling requires inspectable gsplat rasterization source"
        ) from exc
    if OWNER_PROJECTION_SURVIVOR_FIX not in source:
        raise RuntimeError(
            "Gradient-zero profiling requires the gsplat owner-survivor patch. "
            "Run `python tools/patch_gsplat_distributed_packed.py` in the TideGS "
            "environment before launching training."
        )


@lru_cache(maxsize=1)
def require_distributed_gsplat():
    import gsplat

    version = _version_tuple(getattr(gsplat, "__version__", "0"))
    if version < MIN_DISTRIBUTED_GSPLAT:
        expected = ".".join(str(value) for value in MIN_DISTRIBUTED_GSPLAT)
        raise RuntimeError(
            f"Distributed TideGS requires gsplat>={expected}; found "
            f"{getattr(gsplat, '__version__', 'unknown')} at {getattr(gsplat, '__file__', 'unknown')}"
        )
    signature = inspect.signature(gsplat.rasterization)
    if "distributed" not in signature.parameters:
        raise RuntimeError("Installed gsplat.rasterization has no distributed parameter")
    _require_packed_distributed_fix(gsplat, version)
    return gsplat


@lru_cache(maxsize=1)
def require_gsplat_cuda_backend():
    """Load or JIT-build gsplat's CUDA extension in the current process."""
    require_distributed_gsplat()
    from gsplat.cuda._backend import _C

    if _C is None:
        raise RuntimeError("gsplat CUDA backend is unavailable")
    return _C


def prepare_distributed_gsplat(
    context,
    *,
    enable_timing: bool = False,
    enable_grad_zero_metrics: bool = False,
) -> None:
    """Serialize first-time CUDA extension setup across local ranks."""
    gsplat = require_distributed_gsplat()
    torch._tide_gsplat_detailed_metrics = bool(enable_timing)
    torch._tide_gsplat_grad_zero_metrics = bool(enable_grad_zero_metrics)
    if enable_timing:
        _require_projection_timing_fix(gsplat)
    if enable_grad_zero_metrics:
        _require_owner_projection_survivor_fix(gsplat)
    for loader_rank in range(context.world_size):
        error = None
        if context.rank == loader_rank:
            try:
                require_gsplat_cuda_backend()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = context.broadcast_object(error, src=loader_rank)
        if error is not None:
            raise RuntimeError(
                f"gsplat CUDA backend setup failed on rank {loader_rank}: {error}"
            )


def projection_elapsed_ms(meta) -> float:
    events = meta.get(PROJECTION_TIMING_FIX) if isinstance(meta, dict) else None
    if events is None:
        return 0.0
    if not isinstance(events, (tuple, list)) or len(events) != 2:
        raise RuntimeError("gsplat returned an invalid TideGS projection event pair")
    start_event, end_event = events
    end_event.synchronize()
    elapsed_ms = float(start_event.elapsed_time(end_event))
    if elapsed_ms < 0.0:
        raise RuntimeError(f"gsplat returned negative projection time: {elapsed_ms}")
    return elapsed_ms


def _stack_cameras(cameras: Iterable[object]):
    cameras = list(cameras)
    if not cameras:
        raise ValueError("Distributed rasterization requires at least one local camera")
    viewmats = torch.stack(
        [camera.world_view_transform.transpose(0, 1) for camera in cameras], dim=0
    )
    intrinsics = torch.stack([camera.K for camera in cameras], dim=0)
    return cameras, viewmats, intrinsics


def distributed_rasterize(
    *,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients: torch.Tensor,
    cameras,
    width: int,
    height: int,
    sh_degree: int,
    background: torch.Tensor | None,
    radius_clip: float = 0.0,
):
    gsplat = require_distributed_gsplat()
    cameras, viewmats, intrinsics = _stack_cameras(cameras)
    backgrounds = None
    if background is not None:
        backgrounds = background.reshape(1, 3).expand(len(cameras), 3).contiguous()
    return gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh_coefficients,
        viewmats=viewmats,
        Ks=intrinsics,
        width=int(width),
        height=int(height),
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=float(radius_clip),
        sh_degree=int(sh_degree),
        packed=True,
        backgrounds=backgrounds,
        sparse_grad=False,
        distributed=True,
    )
