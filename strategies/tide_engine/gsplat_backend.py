"""Version-gated gsplat entry points used by the distributed TideGS path."""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Iterable, Tuple

import torch


MIN_DISTRIBUTED_GSPLAT = (1, 5, 3)


def _version_tuple(value: str) -> Tuple[int, ...]:
    parts = []
    for token in str(value).split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


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
    return gsplat


@lru_cache(maxsize=1)
def require_gsplat_cuda_backend():
    """Load or JIT-build gsplat's CUDA extension in the current process."""
    require_distributed_gsplat()
    from gsplat.cuda._backend import _C

    if _C is None:
        raise RuntimeError("gsplat CUDA backend is unavailable")
    return _C


def prepare_distributed_gsplat(context) -> None:
    """Serialize first-time CUDA extension setup across local ranks."""
    require_distributed_gsplat()
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
