"""Shared row-wise gradient sparsity profiling for TideGS engines."""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from .gradient_schema import (
    GRADIENT_COMPONENTS,
    GRADIENT_WIDTHS,
    GRAD_NEAR_ZERO_THRESHOLDS,
    PARAMETERS_PER_GAUSSIAN,
    active_parameter_width,
    gradient_stat_fields,
)


def touched_component_rows(
    components: Dict[str, torch.Tensor],
    active_count: int,
    device: torch.device,
) -> torch.Tensor:
    active_count = int(active_count)
    if active_count < 0:
        raise ValueError("active_count must be non-negative")
    if active_count == 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    touched = torch.zeros((active_count,), dtype=torch.bool, device=device)
    for name, value in components.items():
        if not torch.is_tensor(value) or int(value.shape[0]) != int(active_count):
            raise RuntimeError(
                f"Gradient component {name!r} has invalid leading rows"
            )
        touched |= torch.any(value.detach().reshape(active_count, -1) != 0, dim=1)
    return torch.nonzero(touched, as_tuple=False).reshape(-1)


def profile_gradient_sparsity(
    components: Dict[str, torch.Tensor],
    row_mask: torch.Tensor,
    *,
    prefix: str,
    active_sh_degree: int,
    near_zero_threshold: float = 1e-8,
    sample_rows: int = 0,
    chunk_rows: int = 262144,
    include_sample_count: bool = False,
    include_active: bool = True,
) -> Dict[str, object]:
    """Profile selected Gaussian rows using bounded temporary tensors."""

    fields = gradient_stat_fields(
        prefix,
        include_sample_count=include_sample_count,
        include_active=include_active,
    )
    active_count = int(row_mask.numel())
    device = row_mask.device
    active_width = active_parameter_width(active_sh_degree)
    if active_count == 0:
        return {
            "stats": {
                field: torch.zeros((), dtype=torch.long, device=device)
                for field in fields
            },
            "sample_owner_rows": torch.empty(
                (0,), dtype=torch.long, device=device
            ),
            "sample_gradients": torch.empty(
                (0, PARAMETERS_PER_GAUSSIAN),
                dtype=torch.float32,
                device=device,
            ),
            "active_parameter_width": active_width,
        }
    if row_mask.dtype != torch.bool:
        raise ValueError("row_mask must be boolean")
    near_zero_threshold = float(near_zero_threshold)
    if near_zero_threshold < 0.0:
        raise ValueError("near_zero_threshold must be non-negative")
    chunk_rows = max(1, int(chunk_rows))
    for name, width in GRADIENT_WIDTHS.items():
        value = components.get(name)
        if not torch.is_tensor(value) or int(value.shape[0]) != active_count:
            raise RuntimeError(
                f"Gradient profiling received invalid {name!r} rows: "
                f"expected={active_count}, actual={getattr(value, 'shape', None)}"
            )
        if int(value.flatten(start_dim=1).shape[1]) != width:
            raise RuntimeError(
                f"Gradient profiling expected {width} values per Gaussian for {name!r}"
            )

    selected_rows = torch.nonzero(row_mask, as_tuple=False).reshape(-1)
    selected_count = int(selected_rows.numel())
    row_zero_counts = torch.zeros(
        (selected_count,), dtype=torch.int16, device=device
    )
    row_near_counts = torch.zeros_like(row_zero_counts)
    active_row_zero_counts = torch.zeros_like(row_zero_counts)
    active_row_near_counts = torch.zeros_like(row_zero_counts)
    requested_samples = int(sample_rows)
    sample_count = (
        selected_count
        if requested_samples == -1
        else min(max(0, requested_samples), selected_count)
    )
    if sample_count:
        sample_positions = torch.div(
            torch.arange(sample_count, dtype=torch.long, device=device)
            * selected_count,
            sample_count,
            rounding_mode="floor",
        )
        sample_owner_rows = selected_rows.index_select(0, sample_positions)
    else:
        sample_owner_rows = torch.empty((0,), dtype=torch.long, device=device)

    stats: Dict[str, torch.Tensor] = {
        f"{prefix}_unique_gaussians": torch.tensor(
            selected_count, dtype=torch.long, device=device
        ),
        f"{prefix}_parameter_elements": torch.tensor(
            selected_count * PARAMETERS_PER_GAUSSIAN,
            dtype=torch.long,
            device=device,
        ),
    }
    total_nonfinite = torch.zeros((), dtype=torch.long, device=device)
    total_near = [
        torch.zeros((), dtype=torch.long, device=device)
        for _ in GRAD_NEAR_ZERO_THRESHOLDS
    ]
    active_nonfinite = torch.zeros((), dtype=torch.long, device=device)
    active_near = [
        torch.zeros((), dtype=torch.long, device=device)
        for _ in GRAD_NEAR_ZERO_THRESHOLDS
    ]
    sample_components = []
    active_remaining = active_width

    for name in GRADIENT_COMPONENTS:
        value = components[name].detach()
        width = GRADIENT_WIDTHS[name]
        component_active_width = min(width, max(0, active_remaining))
        active_remaining -= component_active_width
        component_zero = torch.zeros((), dtype=torch.long, device=device)
        component_nonfinite = torch.zeros((), dtype=torch.long, device=device)
        component_near = [
            torch.zeros((), dtype=torch.long, device=device)
            for _ in GRAD_NEAR_ZERO_THRESHOLDS
        ]
        for start in range(0, selected_count, chunk_rows):
            end = min(start + chunk_rows, selected_count)
            owner_rows = selected_rows[start:end]
            flat = value.index_select(0, owner_rows).flatten(start_dim=1)
            absolute = flat.abs()
            exact_by_row = (flat == 0).sum(dim=1)
            primary_near_by_row = (absolute <= near_zero_threshold).sum(dim=1)
            row_zero_counts[start:end].add_(exact_by_row.to(torch.int16))
            row_near_counts[start:end].add_(primary_near_by_row.to(torch.int16))
            component_zero.add_(exact_by_row.sum(dtype=torch.long))
            component_nonfinite.add_((~torch.isfinite(flat)).sum(dtype=torch.long))
            for index, (_, threshold) in enumerate(GRAD_NEAR_ZERO_THRESHOLDS):
                component_near[index].add_(
                    (absolute <= threshold).sum(dtype=torch.long)
                )
            if include_active and component_active_width:
                active_flat = flat[:, :component_active_width]
                active_absolute = active_flat.abs()
                active_row_zero_counts[start:end].add_(
                    (active_flat == 0).sum(dim=1).to(torch.int16)
                )
                active_row_near_counts[start:end].add_(
                    (active_absolute <= near_zero_threshold)
                    .sum(dim=1)
                    .to(torch.int16)
                )
                active_nonfinite.add_(
                    (~torch.isfinite(active_flat)).sum(dtype=torch.long)
                )
                for index, (_, threshold) in enumerate(
                    GRAD_NEAR_ZERO_THRESHOLDS
                ):
                    active_near[index].add_(
                        (active_absolute <= threshold).sum(dtype=torch.long)
                    )
        stats[f"{prefix}_{name}_parameter_elements"] = torch.tensor(
            selected_count * width, dtype=torch.long, device=device
        )
        stats[f"{prefix}_{name}_zero_gradient_elements"] = component_zero
        stats[f"{prefix}_{name}_nonfinite_gradient_elements"] = component_nonfinite
        for (token, _), count in zip(GRAD_NEAR_ZERO_THRESHOLDS, component_near):
            stats[f"{prefix}_{name}_abs_le_{token}_gradient_elements"] = count
        total_nonfinite.add_(component_nonfinite)
        for total, count in zip(total_near, component_near):
            total.add_(count)
        if sample_count:
            sample_components.append(
                value.index_select(0, sample_owner_rows).flatten(start_dim=1)
            )

    zero_histogram = torch.bincount(
        row_zero_counts.to(torch.long), minlength=PARAMETERS_PER_GAUSSIAN + 1
    )
    near_histogram = torch.bincount(
        row_near_counts.to(torch.long), minlength=PARAMETERS_PER_GAUSSIAN + 1
    )
    stats[f"{prefix}_zero_gradient_elements"] = row_zero_counts.sum(dtype=torch.long)
    stats[f"{prefix}_nonfinite_gradient_elements"] = total_nonfinite
    stats[f"{prefix}_all_zero_gradient_gaussians"] = zero_histogram[
        PARAMETERS_PER_GAUSSIAN
    ]
    if include_sample_count:
        stats["raw_sampled_gaussians"] = torch.tensor(
            sample_count, dtype=torch.long, device=device
        )
    for (token, _), count in zip(GRAD_NEAR_ZERO_THRESHOLDS, total_near):
        stats[f"{prefix}_abs_le_{token}_gradient_elements"] = count
    for count, value in enumerate(zero_histogram):
        stats[f"{prefix}_zero_gradient_elements_{count}_gaussians"] = value
    for count, value in enumerate(near_histogram):
        stats[f"{prefix}_near_zero_elements_{count}_gaussians"] = value

    if include_active:
        active_zero_histogram = torch.bincount(
            active_row_zero_counts.to(torch.long),
            minlength=PARAMETERS_PER_GAUSSIAN + 1,
        )
        active_near_histogram = torch.bincount(
            active_row_near_counts.to(torch.long),
            minlength=PARAMETERS_PER_GAUSSIAN + 1,
        )
        stats[f"{prefix}_active_parameter_elements"] = torch.tensor(
            selected_count * active_width, dtype=torch.long, device=device
        )
        stats[f"{prefix}_active_zero_gradient_elements"] = (
            active_row_zero_counts.sum(dtype=torch.long)
        )
        stats[f"{prefix}_active_nonfinite_gradient_elements"] = active_nonfinite
        stats[f"{prefix}_active_all_zero_gradient_gaussians"] = (
            active_zero_histogram[active_width]
        )
        for (token, _), count in zip(GRAD_NEAR_ZERO_THRESHOLDS, active_near):
            stats[f"{prefix}_active_abs_le_{token}_gradient_elements"] = count
        for count, value in enumerate(active_zero_histogram):
            stats[f"{prefix}_active_zero_gradient_elements_{count}_gaussians"] = value
        for count, value in enumerate(active_near_histogram):
            stats[f"{prefix}_active_near_zero_elements_{count}_gaussians"] = value

    missing = set(fields).difference(stats)
    if missing:
        raise RuntimeError(f"Gradient sparsity stats are missing fields: {sorted(missing)}")
    sample_gradients = (
        torch.cat(sample_components, dim=1).contiguous()
        if sample_components
        else torch.empty(
            (0, PARAMETERS_PER_GAUSSIAN), dtype=torch.float32, device=device
        )
    )
    return {
        "stats": stats,
        "sample_owner_rows": sample_owner_rows,
        "sample_gradients": sample_gradients,
        "active_parameter_width": active_width,
    }


def collect_gradient_sparsity(
    components: Dict[str, torch.Tensor],
    projection_mask: torch.Tensor,
    tile_mask: torch.Tensor,
    *,
    active_sh_degree: int,
    tile_mode: str,
    tile_counts: Sequence[torch.Tensor],
    near_zero_threshold: float,
    sample_rows: int,
    chunk_rows: int,
) -> Dict[str, object]:
    """Collect projection, tile, and final nonzero-row statistics."""

    projection = profile_gradient_sparsity(
        components,
        projection_mask,
        prefix="projection_cull",
        active_sh_degree=active_sh_degree,
        near_zero_threshold=near_zero_threshold,
        sample_rows=sample_rows,
        chunk_rows=chunk_rows,
        include_sample_count=True,
        include_active=True,
    )
    tile = profile_gradient_sparsity(
        components,
        tile_mask,
        prefix="tile_mask_keep",
        active_sh_degree=active_sh_degree,
        near_zero_threshold=near_zero_threshold,
        sample_rows=0,
        chunk_rows=chunk_rows,
        include_sample_count=False,
        include_active=True,
    )
    touched_rows = touched_component_rows(
        components, projection_mask.numel(), projection_mask.device
    )
    nonzero_mask = torch.zeros_like(projection_mask)
    nonzero_mask[touched_rows] = True
    rejected_mask = projection_mask & ~tile_mask
    rejected = rejected_mask.sum(dtype=torch.long)
    rejected_nonzero = (rejected_mask & nonzero_mask).sum(dtype=torch.long)
    if len(tile_counts) != 5:
        raise ValueError("tile_counts must contain five counters")

    stats = {
        **projection["stats"],
        **tile["stats"],
        "projection_cull_nonzero_gradient_gaussians": (
            projection_mask & nonzero_mask
        ).sum(dtype=torch.long),
        "tile_mask_keep_nonzero_gradient_gaussians": (
            tile_mask & nonzero_mask
        ).sum(dtype=torch.long),
        "tile_mask_rejected_gaussians": rejected,
        "tile_mask_rejected_all_zero_gradient_gaussians": (
            rejected - rejected_nonzero
        ),
        "tile_mask_rejected_nonzero_gradient_gaussians": rejected_nonzero,
        "would_drop_nonzero_gradient_gaussians": (
            rejected_nonzero
            if tile_mode == "profile"
            else torch.zeros_like(rejected_nonzero)
        ),
        **dict(
            zip(
                (
                    "tile_mask_projection_pairs",
                    "tile_mask_kept_projection_pairs",
                    "tile_mask_candidate_tile_pairs_before",
                    "tile_mask_candidate_tile_pairs_after",
                    "tile_mask_contributing_tile_pairs",
                ),
                tile_counts,
            )
        ),
    }
    return {
        "stats": stats,
        "sample_owner_rows": projection["sample_owner_rows"],
        "sample_gradients": projection["sample_gradients"],
        "active_parameter_width": projection["active_parameter_width"],
        "touched_rows": touched_rows,
    }


def build_gradient_sparsity_records(
    profile: Dict[str, object],
    metric_fields: Sequence[str],
    local_ids: torch.Tensor,
    tile_mask: torch.Tensor,
    *,
    iteration: int,
    optimizer_step: int,
    active_sh_degree: int,
    near_zero_threshold: float,
    tile_mode: str,
    tile_alpha_threshold: float,
) -> tuple[Dict[str, object], Dict[str, object]]:
    """Move one profile to CPU and build the stable TSV/sample payloads."""

    stats = profile["stats"]
    values = dict(
        zip(
            metric_fields,
            torch.stack([stats[field] for field in metric_fields])
            .detach()
            .cpu()
            .tolist(),
        )
    )
    sample_rows = profile["sample_owner_rows"]
    sample_local_ids = local_ids.index_select(0, sample_rows)
    common = {
        "iteration": iteration,
        "optimizer_step": optimizer_step,
        "active_sh_degree": active_sh_degree,
        "active_parameter_width": profile["active_parameter_width"],
        "near_zero_threshold": near_zero_threshold,
        "tile_contribution_mode": tile_mode,
        "tile_alpha_threshold": tile_alpha_threshold,
    }
    return (
        {
            **common,
            "tile_drop_gradients_observed": int(tile_mode == "profile"),
            **values,
        },
        {
            "format_version": 2,
            **common,
            "parameter_names": GRADIENT_COMPONENTS,
            "parameter_widths": tuple(
                GRADIENT_WIDTHS[name] for name in GRADIENT_COMPONENTS
            ),
            "owner_active_rows": int(tile_mask.numel()),
            "projection_cull_unique_gaussians": int(
                stats["projection_cull_unique_gaussians"].item()
            ),
            "owner_active_row_indices": sample_rows.detach().cpu(),
            "resident_local_ids": sample_local_ids.detach().cpu(),
            "tile_keep": tile_mask.index_select(0, sample_rows).detach().cpu(),
            "gradients": profile["sample_gradients"].detach().cpu(),
        },
    )
