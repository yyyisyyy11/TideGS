"""Pure-Torch math helpers for the 3DGS2-TR optimizer."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


__all__ = [
    "build_3dgs2_residual_tuple_from_ssim_map",
    "build_3dgs2_residual_vector",
    "build_3dgs2_residual_vector_from_ssim_map",
    "clip_hellinger_step",
    "estimate_residual_vjp_curvature",
    "exponential_schedule",
    "optimizer_step_from_iteration",
    "rademacher_like",
    "resolve_curvature_schedule",
    "should_update_curvature",
]


_COMPONENT_WIDTHS = {
    "xyz": 3,
    "opacity": 1,
    "scaling": 3,
    "rotation": 4,
    "features_dc": 3,
    "features_rest": 45,
}
_SH0 = 0.28209479177387814
_SH_DEGREE_WIDTHS = (1, 3, 5, 7)
_SH_DEGREE_BOUNDS = tuple(
    math.sqrt((2 * degree + 1) / (4.0 * math.pi)) for degree in range(4)
)


def _positive_index(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be >= 1")
    return int(result)


def optimizer_step_from_iteration(iteration: int, batch_size: int) -> int:
    """Map a 1-based image iteration to the paper's 1-based optimizer step."""

    iteration = _positive_index(iteration, "iteration")
    batch_size = _positive_index(batch_size, "batch_size")
    return (iteration - 1) // batch_size + 1


def should_update_curvature(step: int, interval: int) -> bool:
    """Use the paper schedule ``t mod l = 1`` for 1-based steps."""

    step = _positive_index(step, "step")
    interval = _positive_index(interval, "interval")
    return (step - 1) % interval == 0


def resolve_curvature_schedule(
    *,
    iteration: int,
    batch_size: int,
    interval: int,
    optimizer_step: Optional[int] = None,
) -> Tuple[int, bool]:
    """Resolve and validate the shared 1-based 3DGS2-TR curvature clock."""

    expected_step = optimizer_step_from_iteration(iteration, batch_size)
    if optimizer_step is None:
        resolved_step = expected_step
    else:
        resolved_step = _positive_index(optimizer_step, "optimizer_step")
        if resolved_step != expected_step:
            raise ValueError(
                "optimizer_step disagrees with the iteration-derived step: "
                f"provided={resolved_step}, expected={expected_step}"
            )
    return resolved_step, should_update_curvature(resolved_step, interval)


def exponential_schedule(
    initial_value: float,
    final_value: float,
    step: float,
    max_steps: int,
) -> float:
    """Exponentially interpolate positive values and clamp outside the range."""

    if isinstance(step, bool) or not isinstance(step, (int, float)):
        raise TypeError("step must be a real scalar")
    step = float(step)
    initial_value = float(initial_value)
    final_value = float(final_value)
    max_steps = _positive_index(max_steps, "max_steps")
    if not math.isfinite(step):
        raise ValueError("step must be finite")
    if not math.isfinite(initial_value) or initial_value <= 0.0:
        raise ValueError("initial_value must be finite and > 0")
    if not math.isfinite(final_value) or final_value <= 0.0:
        raise ValueError("final_value must be finite and > 0")
    if step <= 0.0:
        return initial_value
    if step >= float(max_steps):
        return final_value

    fraction = step / float(max_steps)
    return initial_value * math.exp(
        math.log(final_value / initial_value) * fraction
    )


class _SafeSqrt(torch.autograd.Function):
    """Exact square root values with a finite derivative at zero."""

    @staticmethod
    def forward(input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(input_tensor)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        del output
        (input_tensor,) = inputs
        ctx.save_for_backward(input_tensor)
        ctx.save_for_forward(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input_tensor,) = ctx.saved_tensors
        denominator = (2.0 * torch.sqrt(input_tensor)).clamp_min(
            torch.finfo(input_tensor.dtype).eps
        )
        return grad_output / denominator

    @staticmethod
    def jvp(ctx, grad_input: Optional[torch.Tensor]):
        if grad_input is None:
            return None
        (input_tensor,) = ctx.saved_tensors
        denominator = (2.0 * torch.sqrt(input_tensor)).clamp_min(
            torch.finfo(input_tensor.dtype).eps
        )
        return grad_input / denominator


def _safe_sqrt(input_tensor: torch.Tensor) -> torch.Tensor:
    return _SafeSqrt.apply(input_tensor)


def _ssim_map(
    image: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    channel = image.shape[1]
    center = window_size // 2
    gaussian_values = [
        math.exp(-((x - center) ** 2) / (2.0 * 1.5**2))
        for x in range(window_size)
    ]
    gaussian_1d = torch.tensor(
        gaussian_values,
        dtype=torch.float32,
        device=image.device,
    ).to(dtype=image.dtype)
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    window_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
    window = window_2d.reshape(1, 1, window_size, window_size).expand(
        channel, 1, window_size, window_size
    ).contiguous()

    padding = window_size // 2
    mu_image = F.conv2d(image, window, padding=padding, groups=channel)
    mu_target = F.conv2d(target, window, padding=padding, groups=channel)
    mu_image_sq = mu_image.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_image * mu_target
    sigma_image_sq = (
        F.conv2d(image.square(), window, padding=padding, groups=channel)
        - mu_image_sq
    )
    sigma_target_sq = (
        F.conv2d(target.square(), window, padding=padding, groups=channel)
        - mu_target_sq
    )
    sigma_product = (
        F.conv2d(image * target, window, padding=padding, groups=channel)
        - mu_product
    )
    c1 = 0.01**2
    c2 = 0.03**2
    return ((2.0 * mu_product + c1) * (2.0 * sigma_product + c2)) / (
        (mu_image_sq + mu_target_sq + c1)
        * (sigma_image_sq + sigma_target_sq + c2)
    )


def _validate_residual_inputs(
    image: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if not torch.is_tensor(image) or not torch.is_tensor(target):
        raise TypeError("image and target must be tensors")
    if image.shape != target.shape:
        raise ValueError(
            f"image and target shapes differ: {tuple(image.shape)} vs {tuple(target.shape)}"
        )
    if image.ndim not in (3, 4):
        raise ValueError("image and target must use CHW or NCHW layout")
    if not image.is_floating_point() or not target.is_floating_point():
        raise TypeError("image and target must be floating-point tensors")
    if image.dtype != target.dtype or image.device != target.device:
        raise ValueError("image and target must have the same dtype and device")


def _validate_lambda_dssim(lambda_dssim: float) -> float:
    lambda_dssim = float(lambda_dssim)
    if not math.isfinite(lambda_dssim) or not 0.0 <= lambda_dssim <= 1.0:
        raise ValueError("lambda_dssim must be finite and in [0, 1]")
    return lambda_dssim


def build_3dgs2_residual_tuple_from_ssim_map(
    image: torch.Tensor,
    target: torch.Tensor,
    ssim_map: torch.Tensor,
    lambda_dssim: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build scaled L1/DSSIM residual tensors from a precomputed SSIM map.

    ``ssim_map`` is the similarity map (one means identical), not ``1-SSIM``.
    The tuple preserves the input image layout for fused JVP/VJP paths. Its
    squared norm uses the same mean reduction as the standard 3DGS loss:
    ``0.5 * sum(r.square().sum() for r in residuals) == combined_loss``.
    """

    _validate_residual_inputs(image, target)
    if not torch.is_tensor(ssim_map):
        raise TypeError("ssim_map must be a tensor")
    if ssim_map.shape != image.shape:
        raise ValueError(
            f"ssim_map must have shape {tuple(image.shape)}, got {tuple(ssim_map.shape)}"
        )
    if ssim_map.dtype != image.dtype or ssim_map.device != image.device:
        raise ValueError("ssim_map must have the same dtype and device as image")
    lambda_dssim = _validate_lambda_dssim(lambda_dssim)

    num_values = image.numel()
    if num_values == 0:
        return image.clone(), image.clone()
    l1_scale = math.sqrt(2.0 * (1.0 - lambda_dssim) / num_values)
    dssim_scale = math.sqrt(2.0 * lambda_dssim / num_values)
    l1_residual = l1_scale * _safe_sqrt((image - target).abs())
    dssim_residual = dssim_scale * _safe_sqrt((1.0 - ssim_map).clamp_min(0.0))
    return l1_residual, dssim_residual


def build_3dgs2_residual_vector_from_ssim_map(
    image: torch.Tensor,
    target: torch.Tensor,
    ssim_map: torch.Tensor,
    lambda_dssim: float = 0.2,
) -> torch.Tensor:
    """Flatten the precomputed-map residual tuple into one vector."""

    residuals = build_3dgs2_residual_tuple_from_ssim_map(
        image, target, ssim_map, lambda_dssim
    )
    return torch.cat(tuple(residual.reshape(-1) for residual in residuals), dim=0)


def build_3dgs2_residual_vector(
    image: torch.Tensor,
    target: torch.Tensor,
    lambda_dssim: float = 0.2,
    *,
    window_size: int = 11,
) -> torch.Tensor:
    """Build the 3DGS2-TR residual vector, including SSIM-map computation."""

    _validate_residual_inputs(image, target)
    lambda_dssim = _validate_lambda_dssim(lambda_dssim)
    window_size = _positive_index(window_size, "window_size")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    image_batch = image.unsqueeze(0) if image.ndim == 3 else image
    target_batch = target.unsqueeze(0) if target.ndim == 3 else target
    ssim_values = _ssim_map(image_batch, target_batch, window_size)
    if image.ndim == 3:
        ssim_values = ssim_values.squeeze(0)
    return build_3dgs2_residual_vector_from_ssim_map(
        image, target, ssim_values, lambda_dssim
    )


def rademacher_like(reference: Any, *, generator=None):
    """Return independent {-1, +1} samples with the reference structure."""

    if torch.is_tensor(reference):
        if not reference.is_floating_point():
            raise TypeError("Rademacher reference tensors must be floating point")
        result = torch.empty_like(reference, requires_grad=False)
        result.bernoulli_(0.5, generator=generator)
        return result.mul_(2.0).sub_(1.0)
    if isinstance(reference, Mapping):
        return {
            key: rademacher_like(value, generator=generator)
            for key, value in reference.items()
        }
    if isinstance(reference, tuple):
        return tuple(rademacher_like(value, generator=generator) for value in reference)
    if isinstance(reference, list):
        return [rademacher_like(value, generator=generator) for value in reference]
    raise TypeError(f"Unsupported Rademacher reference type: {type(reference).__name__}")


def estimate_residual_vjp_curvature(
    residuals: Tuple[torch.Tensor, ...],
    inputs: Tuple[torch.Tensor, ...],
    sample_count: int,
    *,
    generator=None,
) -> Tuple[torch.Tensor, ...]:
    """Estimate ``diag(J.T @ J)`` with output-space Rademacher probes."""

    sample_count = _positive_index(sample_count, "sample_count")
    if not residuals or not inputs:
        raise ValueError("residuals and inputs must be non-empty tuples")
    if not all(torch.is_tensor(value) for value in residuals + inputs):
        raise TypeError("residuals and inputs must contain only tensors")

    accumulators = [torch.zeros_like(value) for value in inputs]
    for _ in range(sample_count):
        probes = rademacher_like(residuals, generator=generator)
        vjps = torch.autograd.grad(
            outputs=residuals,
            inputs=inputs,
            grad_outputs=probes,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        for index, vjp in enumerate(vjps):
            if vjp is not None:
                accumulators[index].add_(vjp.detach().to(torch.float32).square())
    inverse_samples = 1.0 / float(sample_count)
    return tuple(
        value.mul_(inverse_samples).to(dtype=input_value.dtype).contiguous()
        for value, input_value in zip(accumulators, inputs)
    )


def _working_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _finite_tensor(value: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(value.dtype)
    return torch.nan_to_num(
        value,
        nan=0.0,
        posinf=finfo.max,
        neginf=-finfo.max,
    )


def _quat_to_rot_unnormalized(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    q2 = quat.square().sum(dim=-1)
    return torch.stack(
        (
            torch.stack((q2 - 2.0 * (y.square() + z.square()), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)), dim=-1),
            torch.stack((2.0 * (x * y + z * w), q2 - 2.0 * (x.square() + z.square()), 2.0 * (y * z - x * w)), dim=-1),
            torch.stack((2.0 * (x * z - y * w), 2.0 * (y * z + x * w), q2 - 2.0 * (x.square() + y.square())), dim=-1),
        ),
        dim=-2,
    )


def _quat_first_derivative(quat: torch.Tensor, index: int) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    if index == 0:
        rows = ((2 * w, -2 * z, 2 * y), (2 * z, 2 * w, -2 * x), (-2 * y, 2 * x, 2 * w))
    elif index == 1:
        rows = ((2 * x, 2 * y, 2 * z), (2 * y, -2 * x, -2 * w), (2 * z, 2 * w, -2 * x))
    elif index == 2:
        rows = ((-2 * y, 2 * x, 2 * w), (2 * x, 2 * y, 2 * z), (-2 * w, 2 * z, -2 * y))
    else:
        rows = ((-2 * z, -2 * w, 2 * x), (2 * w, -2 * z, 2 * y), (2 * x, 2 * y, 2 * z))
    return torch.stack(
        tuple(torch.stack(row, dim=-1) for row in rows),
        dim=-2,
    )


def _quat_second_derivative(quat: torch.Tensor, index: int) -> torch.Tensor:
    one = torch.ones_like(quat[:, 0])
    zero = torch.zeros_like(one)
    diagonals = (
        (2 * one, 2 * one, 2 * one),
        (2 * one, -2 * one, -2 * one),
        (-2 * one, 2 * one, -2 * one),
        (-2 * one, -2 * one, 2 * one),
    )[index]
    return torch.stack(
        (
            torch.stack((diagonals[0], zero, zero), dim=-1),
            torch.stack((zero, diagonals[1], zero), dim=-1),
            torch.stack((zero, zero, diagonals[2]), dim=-1),
        ),
        dim=-2,
    )


def _stable_quaternion(raw_quat: torch.Tensor):
    finfo = torch.finfo(raw_quat.dtype)
    component_limit = math.sqrt(finfo.max) / 4.0
    identity = torch.zeros_like(raw_quat)
    identity[:, 0] = 1.0
    eligible = torch.isfinite(raw_quat).all(dim=-1) & (
        raw_quat.abs().amax(dim=-1) <= component_limit
    )
    quat = torch.where(eligible[:, None], raw_quat, identity)
    q2 = quat.square().sum(dim=-1)
    minimum_q2 = max(finfo.tiny, finfo.eps**2, 1e-24)
    valid = eligible & (q2 >= minimum_q2)
    quat_for_rotation = torch.where(valid[:, None], quat, identity)
    safe_q2 = quat_for_rotation.square().sum(dim=-1).clamp_min(minimum_q2)
    raw_norm = torch.where(valid, torch.sqrt(q2.clamp_min(0.0)), 0.0)
    return quat_for_rotation, safe_q2, raw_norm, valid


def _quat_trace_coefficients(
    raw_quat: torch.Tensor,
    scaling: torch.Tensor,
) -> torch.Tensor:
    quat, q2, _, _ = _stable_quaternion(raw_quat)
    q4 = q2.square()
    q6 = q4 * q2
    r_tilde = _quat_to_rot_unnormalized(quat).transpose(-1, -2)
    rotation = r_tilde / q2[:, None, None]
    coefficients = []
    for index in range(4):
        t = quat[:, index, None, None]
        dr_tilde = _quat_first_derivative(quat, index).transpose(-1, -2)
        d2r_tilde = _quat_second_derivative(quat, index).transpose(-1, -2)
        dg = rotation.transpose(-1, -2) @ (
            dr_tilde / q2[:, None, None]
            - 2.0 * t * r_tilde / q4[:, None, None]
        )
        d2g = rotation.transpose(-1, -2) @ (
            -4.0 * t * dr_tilde / q4[:, None, None]
            + d2r_tilde / q2[:, None, None]
            + 8.0 * t.square() * r_tilde / q6[:, None, None]
            - 2.0 * r_tilde / q4[:, None, None]
        )
        scaled_dg = scaling[:, :, None] * dg / scaling[:, None, :]
        coefficient = 2.0 * (
            scaled_dg.square().sum(dim=(-2, -1))
            + torch.diagonal(d2g, dim1=-2, dim2=-1).sum(dim=-1)
        )
        coefficients.append(coefficient)
    result = torch.stack(coefficients, dim=-1)
    valid = torch.isfinite(result) & (result >= 0.0)
    return torch.where(valid, result, torch.full_like(result, float("inf")))


def _validated_components(raw_params, steps):
    if not isinstance(raw_params, Mapping) or not isinstance(steps, Mapping):
        raise TypeError("raw_params and steps must be mappings")
    missing_raw = [name for name in _COMPONENT_WIDTHS if name not in raw_params]
    missing_steps = [name for name in _COMPONENT_WIDTHS if name not in steps]
    if missing_raw or missing_steps:
        raise KeyError(
            f"missing raw components={missing_raw}, missing step components={missing_steps}"
        )

    first = raw_params["xyz"]
    if not torch.is_tensor(first) or first.ndim != 2:
        raise ValueError("raw_params['xyz'] must have shape [N, 3]")
    rows = first.shape[0]
    dtype = first.dtype
    device = first.device
    if not first.is_floating_point():
        raise TypeError("3DGS2-TR parameters must be floating point")
    for name, width in _COMPONENT_WIDTHS.items():
        raw_value = raw_params[name]
        step_value = steps[name]
        if not torch.is_tensor(raw_value) or not torch.is_tensor(step_value):
            raise TypeError(f"{name} parameter and step must be tensors")
        expected = (rows, width)
        if tuple(raw_value.shape) != expected or tuple(step_value.shape) != expected:
            raise ValueError(
                f"{name} parameter and step must both have shape {expected}"
            )
        if raw_value.dtype != dtype or step_value.dtype != dtype:
            raise ValueError("all parameters and steps must share one dtype")
        if raw_value.device != device or step_value.device != device:
            raise ValueError("all parameters and steps must share one device")
    return rows, dtype, device


def _negative_log_survival_ratio(
    epsilon: float, alpha: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    ratio = _finite_tensor(alpha.new_tensor(epsilon) / alpha)
    finite_ratio = torch.where(ratio < 1.0, ratio, torch.zeros_like(ratio))
    return ratio, -torch.log1p(-finite_ratio.clamp_min(0.0))


def clip_hellinger_step(
    raw_params: Mapping[str, torch.Tensor],
    steps: Mapping[str, torch.Tensor],
    epsilon: float,
    *,
    opacity_epsilon: Optional[float] = None,
    min_opacity_scaling: Optional[float] = None,
    max_opacity_scaling: Optional[float] = None,
    scale_modifier: float = 1.0,
    quat_norm_tr: Optional[float] = None,
    active_sh_degree: int = 3,
) -> Dict[str, torch.Tensor]:
    """Clip latent updates using the paper's parameter-wise Hellinger bounds.

    ``epsilon`` is the full per-parameter radius from paper pages 6-7 and is
    never split across parameter groups. ``min_opacity_scaling`` and
    ``max_opacity_scaling`` are optional numerical safety overrides; leaving
    them unset uses the physical opacity exactly. ``quat_norm_tr`` is an
    optional extra raw-quaternion cap and is disabled by default.
    """

    rows, output_dtype, _ = _validated_components(raw_params, steps)
    epsilon = float(epsilon)
    opacity_epsilon = epsilon if opacity_epsilon is None else float(opacity_epsilon)
    min_opacity_scaling = (
        None if min_opacity_scaling is None else float(min_opacity_scaling)
    )
    max_opacity_scaling = (
        None if max_opacity_scaling is None else float(max_opacity_scaling)
    )
    scalar_values = {
        "epsilon": epsilon,
        "opacity_epsilon": opacity_epsilon,
        "scale_modifier": float(scale_modifier),
    }
    if any(not math.isfinite(value) for value in scalar_values.values()):
        raise ValueError("Hellinger clip scalars must be finite")
    if epsilon < 0.0 or opacity_epsilon < 0.0:
        raise ValueError("epsilon and opacity_epsilon must be >= 0")
    if scale_modifier <= 0.0:
        raise ValueError("scale_modifier must be > 0")
    scale_modifier = float(scale_modifier)
    if min_opacity_scaling is not None and (
        not math.isfinite(min_opacity_scaling) or min_opacity_scaling <= 0.0
    ):
        raise ValueError("min_opacity_scaling must be finite and > 0")
    if max_opacity_scaling is not None and (
        not math.isfinite(max_opacity_scaling) or max_opacity_scaling <= 0.0
    ):
        raise ValueError("max_opacity_scaling must be finite and > 0")
    if (
        min_opacity_scaling is not None
        and max_opacity_scaling is not None
        and max_opacity_scaling < min_opacity_scaling
    ):
        raise ValueError("opacity scaling bounds are invalid")
    if quat_norm_tr is not None:
        quat_norm_tr = float(quat_norm_tr)
        if math.isnan(quat_norm_tr) or quat_norm_tr < 0.0:
            raise ValueError("quat_norm_tr must be None or >= 0")
    if isinstance(active_sh_degree, bool):
        raise TypeError("active_sh_degree must be an integer")
    try:
        active_sh_degree = operator.index(active_sh_degree)
    except TypeError as exc:
        raise TypeError("active_sh_degree must be an integer") from exc
    if not 0 <= active_sh_degree <= 3:
        raise ValueError("active_sh_degree must be in [0, 3]")

    work_dtype = _working_dtype(output_dtype)
    raw = {name: raw_params[name].to(dtype=work_dtype) for name in _COMPONENT_WIDTHS}
    delta = {name: steps[name].to(dtype=work_dtype) for name in _COMPONENT_WIDTHS}
    opacity = torch.sigmoid(_finite_tensor(raw["opacity"]))
    opacity_scaling = opacity.clamp_min(torch.finfo(work_dtype).tiny)
    if min_opacity_scaling is not None:
        opacity_scaling = opacity_scaling.clamp_min(min_opacity_scaling)
    if max_opacity_scaling is not None:
        opacity_scaling = opacity_scaling.clamp_max(max_opacity_scaling)
    minimum_safe_scaling = 2.0 * torch.finfo(work_dtype).tiny
    maximum_safe_scaling = 0.5 * math.sqrt(torch.finfo(work_dtype).max)
    minimum_target_scaling = 2.0 * minimum_safe_scaling
    maximum_target_scaling = 0.5 * maximum_safe_scaling
    scaling = torch.exp(raw["scaling"]) * scale_modifier
    scale_valid = (
        torch.isfinite(raw["scaling"])
        & torch.isfinite(scaling)
        & (scaling >= minimum_safe_scaling)
        & (scaling <= maximum_safe_scaling)
    )
    safe_scaling = torch.where(scale_valid, scaling, torch.ones_like(scaling))
    quat, quat_q2, quat_raw_norm, quaternion_valid = _stable_quaternion(
        raw["rotation"]
    )
    geometry_valid = quaternion_valid & scale_valid.all(dim=-1)
    rotation = _quat_to_rot_unnormalized(quat) / quat_q2[:, None, None]

    hellinger_ratio, negative_log_survival = _negative_log_survival_ratio(
        epsilon, opacity_scaling
    )
    covariance_diag = _finite_tensor(
        (rotation.square() * safe_scaling.square()[:, None, :]).sum(dim=-1)
    ).clamp_min(0.0)
    position_threshold = torch.sqrt(
        _finite_tensor(8.0 * covariance_diag * negative_log_survival).clamp_min(0.0)
    )
    position_threshold = torch.where(
        hellinger_ratio >= 1.0,
        torch.full_like(position_threshold, torch.finfo(work_dtype).max),
        position_threshold,
    )
    xyz_step = torch.minimum(
        torch.maximum(delta["xyz"], -position_threshold), position_threshold
    )
    xyz_step = torch.where(geometry_valid[:, None], xyz_step, 0.0)

    # The paper's physical-scale radius is S*sqrt(2*epsilon/alpha). In log
    # coordinates this becomes a relative bound and can be applied without
    # exponentiating a proposed raw scale, avoiding overflow and underflow.
    relative_scale_radius = torch.sqrt(
        _finite_tensor(2.0 * (epsilon / opacity_scaling)).clamp_min(0.0)
    )
    upper_scaling_step = torch.log1p(relative_scale_radius)
    lower_scaling_candidate = torch.log1p(
        -relative_scale_radius.clamp(max=1.0 - torch.finfo(work_dtype).eps)
    )
    minimum_raw_scaling = (
        math.log(minimum_target_scaling) - math.log(scale_modifier)
    )
    maximum_raw_scaling = (
        math.log(maximum_target_scaling) - math.log(scale_modifier)
    )
    # Keep an interior factor-of-two margin for float32 subtraction, addition,
    # exp, and scale-modifier rounding. Near an edge, freeze only updates that
    # would move farther outward instead of reversing their direction.
    numerical_lower_scaling_step = torch.minimum(
        minimum_raw_scaling - raw["scaling"],
        torch.zeros_like(raw["scaling"]),
    )
    paper_lower_scaling_step = torch.where(
        relative_scale_radius < 1.0,
        lower_scaling_candidate,
        numerical_lower_scaling_step,
    )
    lower_scaling_step = torch.maximum(
        paper_lower_scaling_step, numerical_lower_scaling_step
    )
    numerical_upper_scaling_step = torch.maximum(
        maximum_raw_scaling - raw["scaling"],
        torch.zeros_like(raw["scaling"]),
    )
    upper_scaling_step = torch.minimum(
        upper_scaling_step, numerical_upper_scaling_step
    )
    scaling_step = torch.minimum(
        torch.maximum(delta["scaling"], lower_scaling_step), upper_scaling_step
    )
    scaling_step = torch.where(geometry_valid[:, None], scaling_step, 0.0)

    quat_coefficients = _quat_trace_coefficients(
        raw["rotation"], safe_scaling
    )
    quat_hellinger_threshold = torch.sqrt(
        _finite_tensor(
            8.0 * negative_log_survival / quat_coefficients
        ).clamp_min(0.0)
    )
    quat_hellinger_threshold = torch.where(
        hellinger_ratio >= 1.0,
        torch.full_like(quat_hellinger_threshold, torch.finfo(work_dtype).max),
        quat_hellinger_threshold,
    )
    if quat_norm_tr is None or math.isinf(quat_norm_tr):
        quat_threshold = quat_hellinger_threshold
    else:
        quat_norm_threshold = _finite_tensor(quat_raw_norm[:, None] * quat_norm_tr)
        quat_threshold = torch.minimum(
            quat_hellinger_threshold, quat_norm_threshold
        )
    rotation_step = torch.minimum(
        torch.maximum(delta["rotation"], -quat_threshold), quat_threshold
    )
    rotation_step = torch.where(geometry_valid[:, None], rotation_step, 0.0)

    opacity_step_threshold = torch.sqrt(
        _finite_tensor(4.0 * opacity * opacity_epsilon).clamp_min(0.0)
    )
    proposed_opacity = torch.sigmoid(
        _finite_tensor(raw["opacity"] + delta["opacity"])
    )
    clipped_opacity = torch.minimum(
        torch.maximum(proposed_opacity, opacity - opacity_step_threshold),
        opacity + opacity_step_threshold,
    )
    clipped_opacity = clipped_opacity.clamp(
        torch.finfo(work_dtype).tiny, 1.0 - torch.finfo(work_dtype).eps
    )
    opacity_step = (
        torch.log(clipped_opacity) - torch.log1p(-clipped_opacity) - raw["opacity"]
    )
    opacity_is_interior = (opacity > 0.0) & (opacity < 1.0)
    opacity_step = torch.where(opacity_is_interior, opacity_step, 0.0)

    # The renderer evaluates C(d)=clamp_min(0.5+sum_l Y_l(d)^T c_l, 0).
    # The addition theorem gives ||Y_l(d)||_2=sqrt((2l+1)/(4pi)), so the
    # expression below is a lower bound on C(d) for every viewing direction.
    sh_raw = torch.cat(
        (raw["features_dc"].reshape(rows, 1, 3), raw["features_rest"].reshape(rows, 15, 3)),
        dim=1,
    )
    sh_delta = torch.cat(
        (
            delta["features_dc"].reshape(rows, 1, 3),
            delta["features_rest"].reshape(rows, 15, 3),
        ),
        dim=1,
    )
    color_lower_bound = _finite_tensor(0.5 + _SH0 * sh_raw[:, 0, :])
    degree_start = 1
    for degree in range(1, active_sh_degree + 1):
        degree_width = _SH_DEGREE_WIDTHS[degree]
        degree_end = degree_start + degree_width
        coefficient_norm = torch.linalg.vector_norm(
            sh_raw[:, degree_start:degree_end, :], dim=1
        )
        color_lower_bound = color_lower_bound - (
            _SH_DEGREE_BOUNDS[degree] * coefficient_norm
        )
        degree_start = degree_end
    color_lower_bound = _finite_tensor(color_lower_bound).clamp_min(0.0)
    physical_color_threshold = torch.sqrt(
        _finite_tensor(
            4.0 * color_lower_bound * (epsilon / opacity_scaling)
        ).clamp_min(0.0)
    )

    # Bound the combined color change, rather than clipping 16 coefficients
    # independently. One scale per Gaussian/channel preserves the proposal
    # direction and guarantees |Delta C_c(d)| <= physical_color_threshold.
    color_change_bound = _SH_DEGREE_BOUNDS[0] * sh_delta[:, 0, :].abs()
    degree_start = 1
    for degree in range(1, active_sh_degree + 1):
        degree_width = _SH_DEGREE_WIDTHS[degree]
        degree_end = degree_start + degree_width
        update_norm = torch.linalg.vector_norm(
            sh_delta[:, degree_start:degree_end, :], dim=1
        )
        color_change_bound = color_change_bound + (
            _SH_DEGREE_BOUNDS[degree] * update_norm
        )
        degree_start = degree_end
    color_change_bound = _finite_tensor(color_change_bound).clamp_min(0.0)
    color_scale = torch.where(
        color_change_bound > 0.0,
        (
            physical_color_threshold
            / color_change_bound.clamp_min(torch.finfo(work_dtype).tiny)
        ).clamp(max=1.0),
        torch.ones_like(color_change_bound),
    )
    sh_step = sh_delta * color_scale[:, None, :]
    active_coefficients = (active_sh_degree + 1) ** 2
    if active_coefficients < sh_step.shape[1]:
        sh_step[:, active_coefficients:, :].zero_()
    features_dc_step = sh_step[:, 0, :]
    features_rest_step = sh_step[:, 1:, :].reshape(rows, 45)

    clipped = {
        "xyz": xyz_step,
        "opacity": opacity_step,
        "scaling": scaling_step,
        "rotation": rotation_step,
        "features_dc": features_dc_step,
        "features_rest": features_rest_step,
    }
    output_limit = torch.finfo(output_dtype).max
    result = {}
    for name, value in clipped.items():
        value = _finite_tensor(value).clamp(-output_limit, output_limit)
        value = torch.where(delta[name] == 0.0, torch.zeros_like(value), value)
        result[name] = value.to(dtype=output_dtype)
    return result
