"""Shared residual and probe helpers for single- and multi-rank 3DGS2-TR."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from .sophia_tr_math import (
    build_3dgs2_residual_vector_from_ssim_map,
    estimate_residual_vjp_curvature,
)
from .sophia_tr_sampling import make_curvature_probe_generator


def build_fused_3dgs2_curvature_residuals(
    image: torch.Tensor,
    target: torch.Tensor,
    lambda_dssim: float = 0.2,
    *,
    ssim_map_builder: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
) -> Tuple[torch.Tensor, ...]:
    """Build one independently normalized residual vector per camera.

    The per-camera split is intentional: changing a distributed camera chunk
    must not add another batch-size factor to either the gradient or the
    Gauss-Newton diagonal. The optimizer applies the single global ``1/|S|``
    normalization for both paths.
    """

    if not torch.is_tensor(image) or not torch.is_tensor(target):
        raise TypeError("image and target must be tensors")
    if image.ndim == 3:
        image_batch = image.unsqueeze(0)
        target_batch = target.unsqueeze(0) if target.ndim == 3 else target
    elif image.ndim == 4:
        image_batch = image
        target_batch = target
    else:
        raise ValueError("image and target must use CHW or NCHW layout")
    if target_batch.shape != image_batch.shape:
        raise ValueError(
            "image and target shapes differ: "
            f"{tuple(image_batch.shape)} vs {tuple(target_batch.shape)}"
        )
    if int(image_batch.shape[0]) < 1:
        raise ValueError("curvature residual batches must contain a camera")

    if ssim_map_builder is None:
        from clm_kernels import FusedSSIMMap

        ssim_map = FusedSSIMMap.apply(
            0.01**2,
            0.03**2,
            image_batch,
            target_batch,
            "same",
            True,
        )
    else:
        ssim_map = ssim_map_builder(image_batch, target_batch)
    if not torch.is_tensor(ssim_map) or ssim_map.shape != image_batch.shape:
        raise ValueError(
            "SSIM map must match the rendered image batch: "
            f"expected={tuple(image_batch.shape)}, "
            f"actual={getattr(ssim_map, 'shape', None)}"
        )

    return tuple(
        build_3dgs2_residual_vector_from_ssim_map(
            image_batch[camera_index],
            target_batch[camera_index],
            ssim_map[camera_index],
            lambda_dssim=lambda_dssim,
        )
        for camera_index in range(int(image_batch.shape[0]))
    )


def estimate_seeded_curvature_sample(
    residuals: Tuple[torch.Tensor, ...],
    inputs: Tuple[torch.Tensor, ...],
    *,
    base_seed: int,
    optimizer_step: int,
    microbatch_index: int,
    sample_index: int,
    sample_count: int,
    rank: int,
) -> Tuple[torch.Tensor, ...]:
    """Estimate one deterministic probe and apply the shared ``1/nu`` scale."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer, not bool")
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")
    if not inputs:
        raise ValueError("inputs must be non-empty")
    generator = make_curvature_probe_generator(
        base_seed=base_seed,
        optimizer_step=optimizer_step,
        microbatch_index=microbatch_index,
        sample_index=sample_index,
        rank=rank,
        device=inputs[0].device,
    )
    values = estimate_residual_vjp_curvature(
        residuals,
        inputs,
        1,
        generator=generator,
    )
    inverse_sample_count = 1.0 / float(sample_count)
    return tuple(value.mul(inverse_sample_count) for value in values)
