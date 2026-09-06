"""Dependency-free schema helpers for TideGS gradient metrics."""

GRADIENT_COMPONENTS = (
    "xyz",
    "opacity",
    "scaling",
    "rotation",
    "features_dc",
    "features_rest",
)
GRADIENT_WIDTHS = {
    "xyz": 3,
    "opacity": 1,
    "scaling": 3,
    "rotation": 4,
    "features_dc": 3,
    "features_rest": 45,
}
PARAMETERS_PER_GAUSSIAN = sum(GRADIENT_WIDTHS.values())
GRAD_NEAR_ZERO_THRESHOLDS = (
    ("1em16", 1e-16),
    ("1em14", 1e-14),
    ("1em12", 1e-12),
    ("1em10", 1e-10),
    ("1em8", 1e-8),
    ("1em6", 1e-6),
    ("1em4", 1e-4),
)


def active_parameter_width(active_sh_degree: int) -> int:
    degree = int(active_sh_degree)
    if degree < 0 or degree > 3:
        raise ValueError("active_sh_degree must be in [0, 3]")
    return 11 + 3 * (degree + 1) ** 2


def gradient_stat_fields(
    prefix: str,
    *,
    include_sample_count: bool,
    include_active: bool,
) -> list[str]:
    fields = [
        f"{prefix}_unique_gaussians",
        f"{prefix}_parameter_elements",
        f"{prefix}_zero_gradient_elements",
        f"{prefix}_nonfinite_gradient_elements",
        f"{prefix}_all_zero_gradient_gaussians",
    ]
    if include_sample_count:
        fields.append("raw_sampled_gaussians")
    fields.extend(
        f"{prefix}_{component}_{suffix}"
        for component in GRADIENT_COMPONENTS
        for suffix in (
            "parameter_elements",
            "zero_gradient_elements",
            "nonfinite_gradient_elements",
        )
    )
    fields.extend(
        f"{prefix}_abs_le_{token}_gradient_elements"
        for token, _ in GRAD_NEAR_ZERO_THRESHOLDS
    )
    fields.extend(
        f"{prefix}_{component}_abs_le_{token}_gradient_elements"
        for component in GRADIENT_COMPONENTS
        for token, _ in GRAD_NEAR_ZERO_THRESHOLDS
    )
    fields.extend(
        f"{prefix}_zero_gradient_elements_{count}_gaussians"
        for count in range(PARAMETERS_PER_GAUSSIAN + 1)
    )
    fields.extend(
        f"{prefix}_near_zero_elements_{count}_gaussians"
        for count in range(PARAMETERS_PER_GAUSSIAN + 1)
    )
    if include_active:
        fields.extend(
            [
                f"{prefix}_active_parameter_elements",
                f"{prefix}_active_zero_gradient_elements",
                f"{prefix}_active_nonfinite_gradient_elements",
                f"{prefix}_active_all_zero_gradient_gaussians",
            ]
        )
        fields.extend(
            f"{prefix}_active_abs_le_{token}_gradient_elements"
            for token, _ in GRAD_NEAR_ZERO_THRESHOLDS
        )
        fields.extend(
            f"{prefix}_active_zero_gradient_elements_{count}_gaussians"
            for count in range(PARAMETERS_PER_GAUSSIAN + 1)
        )
        fields.extend(
            f"{prefix}_active_near_zero_elements_{count}_gaussians"
            for count in range(PARAMETERS_PER_GAUSSIAN + 1)
        )
    return fields
