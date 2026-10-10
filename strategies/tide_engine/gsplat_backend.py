"""Version-gated gsplat entry points used by the distributed TideGS path."""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Dict, Iterable, Tuple

import torch


MIN_DISTRIBUTED_GSPLAT = (1, 5, 3)
PACKED_DISTRIBUTED_FIX = "image_ids = camera_ids"
PROJECTION_TIMING_FIX = "_tide_projection_events"
OWNER_PROJECTION_SURVIVOR_FIX = "_tide_owner_projection_survivor_mask"
TILE_CONTRIBUTION_FIXES = (
    "_tide_owner_tile_survivor_mask",
    "_tide_tile_contribution_mask",
    "_tide_original_projection_radii",
    "_tide_radius_mask",
)
PARTIAL_GRADIENT_FIX = "_tide_begin_partial_gradient_metrics"
PARTIAL_GRADIENT_CONTEXT = "_tide_partial_gradient_context"


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


def _require_tile_contribution_fix(gsplat) -> None:
    try:
        source = inspect.getsource(gsplat.rasterization)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "Tile-contribution masking requires inspectable gsplat rasterization source"
        ) from exc
    if any(marker not in source for marker in TILE_CONTRIBUTION_FIXES):
        raise RuntimeError(
            "Tile-contribution masking requires the gsplat tile-mask patch. "
            "Run `python tools/patch_gsplat_distributed_packed.py` in the TideGS "
            "environment before launching training."
        )


def _require_partial_gradient_fix(gsplat) -> None:
    try:
        source = inspect.getsource(gsplat.rasterization)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "Partial-gradient metrics require inspectable gsplat rasterization source"
        ) from exc
    if PARTIAL_GRADIENT_FIX not in source or PARTIAL_GRADIENT_CONTEXT not in source:
        raise RuntimeError(
            "Partial-gradient metrics require the gsplat receive-side hook patch. "
            "Run `python tools/patch_gsplat_distributed_packed.py` in the TideGS "
            "environment before launching training."
        )


def begin_partial_gradient_metrics(
    *,
    means2d: torch.Tensor,
    depths: torch.Tensor,
    conics: torch.Tensor,
    opacity: torch.Tensor,
    features: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> Dict[str, object]:
    """Register receive-side row hooks without retaining gradient tensors.

    The tensors passed by the patched gsplat function are the five projected
    record buffers exchanged by distributed packed rendering.  A single
    boolean vector records whether any receive-side field has a non-zero row
    gradient.  ``valid_mask`` is derived from post-projection radii and removes
    padding/sentinel records from the denominator.
    """

    tensors = {
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacity": opacity,
        "features": features,
    }
    first = next((value for value in tensors.values() if torch.is_tensor(value)), None)
    if first is None or first.ndim == 0:
        raise RuntimeError("gsplat receive-side partial-gradient tensors are invalid")
    rows = int(first.shape[0])
    for name, value in tensors.items():
        if not torch.is_tensor(value) or value.ndim == 0 or int(value.shape[0]) != rows:
            raise RuntimeError(
                "gsplat receive-side partial-gradient tensor shape mismatch: "
                f"{name}={getattr(value, 'shape', None)}, rows={rows}"
            )

    valid = torch.ones((rows,), dtype=torch.bool, device=first.device)
    for value in tensors.values():
        valid &= torch.isfinite(value.reshape(rows, -1)).all(dim=1)
    if valid_mask is None:
        # Keep the fallback useful for direct callers that do not have radii.
        # The patched gsplat path always supplies a radii-derived mask.
        opacity_flat = opacity.reshape(rows, -1)
        valid &= opacity_flat.abs().sum(dim=1) != 0
    else:
        if (
            not torch.is_tensor(valid_mask)
            or valid_mask.ndim == 0
            or int(valid_mask.shape[0]) != rows
        ):
            raise RuntimeError(
                "gsplat receive-side partial-gradient validity mask shape mismatch: "
                f"mask={getattr(valid_mask, 'shape', None)}, rows={rows}"
            )
        valid &= valid_mask.to(device=first.device, dtype=torch.bool).reshape(
            rows, -1
        ).all(dim=1)
    valid = valid.detach()
    nonzero_rows = torch.zeros((rows,), dtype=torch.bool, device=first.device)
    registered = 0
    context: Dict[str, object] = {
        "valid_rows": valid,
        "nonzero_rows": nonzero_rows,
        "registered": 0,
        "consumed": False,
    }

    def hook(gradient):
        if gradient is None:
            return gradient
        dense = gradient.to_dense() if gradient.is_sparse else gradient
        if dense.ndim == 0 or int(dense.shape[0]) != rows:
            raise RuntimeError(
                "gsplat receive-side partial-gradient hook returned an invalid "
                f"shape: {tuple(dense.shape)} for rows={rows}"
            )
        row_nonzero = torch.any(dense.reshape(rows, -1) != 0, dim=1)
        nonzero_rows.logical_or_(row_nonzero)
        return gradient

    for value in tensors.values():
        if value.requires_grad:
            value.register_hook(hook)
            registered += 1
    if registered == 0:
        raise RuntimeError(
            "gsplat receive-side partial-gradient tensors do not require gradients"
        )
    context["registered"] = registered
    return context


def consume_partial_gradient_metrics(meta) -> Tuple[int, int]:
    """Read and release one rasterization's receive-side row counters."""

    if not isinstance(meta, dict):
        raise RuntimeError("gsplat did not return rasterization metadata")
    context = meta.get(PARTIAL_GRADIENT_CONTEXT)
    if not isinstance(context, dict):
        raise RuntimeError(
            "gsplat did not return the receive-side partial-gradient context"
        )
    if bool(context.get("consumed", False)):
        raise RuntimeError("receive-side partial-gradient context was consumed twice")
    valid = context.get("valid_rows")
    nonzero = context.get("nonzero_rows")
    if (
        not torch.is_tensor(valid)
        or valid.dtype != torch.bool
        or not torch.is_tensor(nonzero)
        or nonzero.dtype != torch.bool
        or valid.shape != nonzero.shape
    ):
        raise RuntimeError("invalid gsplat partial-gradient counter tensors")
    # .item() synchronizes the stream after all backward hooks have run.
    total = int(valid.sum().item())
    all_zero = int((valid & ~nonzero).sum().item())
    if all_zero > total:
        raise RuntimeError(
            f"invalid partial-gradient counters: all_zero={all_zero} total={total}"
        )
    context["consumed"] = True
    return total, all_zero


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
    enable_partial_grad_metrics: bool = False,
    tile_contribution_mode: str = "off",
    tile_alpha_threshold: float = 1.0 / 255.0,
) -> None:
    """Serialize first-time CUDA extension setup across local ranks."""
    gsplat = require_distributed_gsplat()
    torch._tide_gsplat_detailed_metrics = bool(enable_timing)
    torch._tide_gsplat_grad_zero_metrics = bool(enable_grad_zero_metrics)
    torch._tide_partial_grad_metrics = bool(enable_partial_grad_metrics)
    torch._tide_partial_grad_metrics_sample = False
    torch._tide_begin_partial_gradient_metrics = begin_partial_gradient_metrics
    tile_contribution_mode = str(tile_contribution_mode).lower()
    torch._tide_tile_contribution_mode = tile_contribution_mode
    torch._tide_tile_alpha_threshold = float(tile_alpha_threshold)
    if tile_contribution_mode != "off":
        from clm_kernels import tile_contribution_mask

        torch._tide_tile_contribution_mask = tile_contribution_mask
    if enable_timing:
        _require_projection_timing_fix(gsplat)
    if enable_grad_zero_metrics:
        _require_owner_projection_survivor_fix(gsplat)
    if enable_partial_grad_metrics:
        _require_partial_gradient_fix(gsplat)
    if tile_contribution_mode != "off":
        _require_tile_contribution_fix(gsplat)
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
