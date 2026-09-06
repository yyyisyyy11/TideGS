#!/usr/bin/env python3
"""Summarize projection-cull gradient-zero metrics emitted by TideGS."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COMPONENTS = (
    "xyz",
    "opacity",
    "scaling",
    "rotation",
    "features_dc",
    "features_rest",
)
THRESHOLDS = (
    ("1em16", 1e-16),
    ("1em14", 1e-14),
    ("1em12", 1e-12),
    ("1em10", 1e-10),
    ("1em8", 1e-8),
    ("1em6", 1e-6),
    ("1em4", 1e-4),
)
TOTAL_FIELD = "projection_cull_parameter_elements"
ZERO_FIELD = "projection_cull_zero_gradient_elements"
ROWS_FIELD = "projection_cull_unique_gaussians"
ALL_ZERO_ROWS_FIELD = "projection_cull_all_zero_gradient_gaussians"


def _quantile_from_histogram(histogram, quantile: float) -> int:
    total = sum(histogram)
    if total == 0:
        return 0
    threshold = total * quantile
    cumulative = 0
    for zero_count, rows in enumerate(histogram):
        cumulative += rows
        if cumulative >= threshold:
            return zero_count
    return len(histogram) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metrics_tsv",
        type=Path,
        help="metrics_grad_zero_global.tsv (recommended) or one rank TSV",
    )
    parser.add_argument(
        "--per-batch",
        action="store_true",
        help="Print one compact exact/near-zero row per profiled batch",
    )
    args = parser.parse_args()

    with args.metrics_tsv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise SystemExit(f"No metric rows in {args.metrics_tsv}")

    if args.per_batch:
        fields = [
            "iteration",
            "optimizer_step",
            "tile_contribution_mode",
            "projection_cull_unique_gaussians",
            "tile_mask_keep_unique_gaussians",
            "tile_mask_rejected_gaussians",
            "tile_mask_rejected_nonzero_gradient_gaussians",
            "projection_exact_zero_ratio",
            "tile_keep_exact_zero_ratio",
            *[f"abs_le_{token}_ratio" for token, _ in THRESHOLDS],
        ]
        print("\t".join(fields))
        for row in rows:
            denominator = int(row.get(TOTAL_FIELD, 0) or 0)
            exact = int(row.get(ZERO_FIELD, 0) or 0)
            tile_rows = int(
                row.get("tile_mask_keep_unique_gaussians", 0) or 0
            )
            tile_elements = int(
                row.get("tile_mask_keep_parameter_elements", 0) or 0
            )
            tile_zero = int(
                row.get("tile_mask_keep_zero_gradient_elements", 0) or 0
            )
            ratios = [
                int(
                    row.get(
                        f"projection_cull_abs_le_{token}_gradient_elements",
                        0,
                    )
                    or 0
                )
                / denominator
                if denominator
                else 0.0
                for token, _ in THRESHOLDS
            ]
            values = [
                row.get("iteration", ""),
                row.get("optimizer_step", ""),
                row.get("tile_contribution_mode", "legacy"),
                row.get(ROWS_FIELD, "0"),
                str(tile_rows),
                row.get("tile_mask_rejected_gaussians", "0"),
                row.get("tile_mask_rejected_nonzero_gradient_gaussians", "0"),
                f"{exact / denominator:.6%}" if denominator else "N/A",
                f"{tile_zero / tile_elements:.6%}" if tile_elements else "N/A",
                *[f"{ratio:.6%}" for ratio in ratios],
            ]
            print("\t".join(values))
        return

    def total(field: str) -> int:
        return sum(int(row.get(field, 0) or 0) for row in rows)

    gaussian_count = total(ROWS_FIELD)
    parameter_count = total(TOTAL_FIELD)
    zero_count = total(ZERO_FIELD)
    histogram = [
        total(f"projection_cull_zero_gradient_elements_{count}_gaussians")
        for count in range(60)
    ]
    print(f"sampled_iterations\t{len(rows)}")
    modes = sorted({row.get("tile_contribution_mode", "legacy") for row in rows})
    print(f"tile_contribution_modes\t{','.join(modes)}")
    print(f"projection_cull_unique_gaussians\t{gaussian_count}")
    print(f"gradient_elements\t{parameter_count}")
    print(f"exact_zero_gradient_elements\t{zero_count}")
    print(
        "exact_zero_gradient_ratio\t"
        f"{zero_count / parameter_count:.6%}" if parameter_count else "exact_zero_gradient_ratio\tN/A"
    )
    print("absolute_threshold\tnear_zero_elements\tnear_zero_ratio")
    for token, threshold in THRESHOLDS:
        near_count = total(
            f"projection_cull_abs_le_{token}_gradient_elements"
        )
        ratio = near_count / parameter_count if parameter_count else 0.0
        print(f"{threshold:.0e}\t{near_count}\t{ratio:.6%}")
    print(
        "all_zero_gradient_gaussian_ratio\t"
        f"{total(ALL_ZERO_ROWS_FIELD) / gaussian_count:.6%}"
        if gaussian_count
        else "all_zero_gradient_gaussian_ratio\tN/A"
    )
    if "tile_mask_keep_unique_gaussians" in rows[0]:
        tile_rows = total("tile_mask_keep_unique_gaussians")
        tile_rejected = total("tile_mask_rejected_gaussians")
        tile_nonzero = total("tile_mask_keep_nonzero_gradient_gaussians")
        tile_all_zero = total("tile_mask_keep_all_zero_gradient_gaussians")
        print(f"tile_mask_keep_unique_gaussians\t{tile_rows}")
        print(f"tile_mask_rejected_gaussians\t{tile_rejected}")
        print(
            "tile_mask_rejected_ratio\t"
            f"{tile_rejected / gaussian_count:.6%}"
            if gaussian_count
            else "tile_mask_rejected_ratio\tN/A"
        )
        print(
            "tile_mask_keep_all_zero_gradient_ratio\t"
            f"{tile_all_zero / tile_rows:.6%}"
            if tile_rows
            else "tile_mask_keep_all_zero_gradient_ratio\tN/A"
        )
        print(f"tile_mask_keep_nonzero_gradient_gaussians\t{tile_nonzero}")
        print(
            "would_drop_nonzero_gradient_gaussians\t"
            f"{total('would_drop_nonzero_gradient_gaussians')}"
        )
        print(
            "tile_pairs\tcandidate_before\tcandidate_after\tcontributing"
        )
        print(
            "tile_pairs\t"
            f"{total('tile_mask_candidate_tile_pairs_before')}\t"
            f"{total('tile_mask_candidate_tile_pairs_after')}\t"
            f"{total('tile_mask_contributing_tile_pairs')}"
        )
        for prefix in ("projection_cull", "tile_mask_keep"):
            active_total = total(f"{prefix}_active_parameter_elements")
            active_zero = total(f"{prefix}_active_zero_gradient_elements")
            print(
                f"{prefix}_active_exact_zero_ratio\t"
                f"{active_zero / active_total:.6%}"
                if active_total
                else f"{prefix}_active_exact_zero_ratio\tN/A"
            )
    print(
        "cohort\tcomponent\ttotal_elements\texact_zero_ratio\t"
        + "\t".join(f"abs_le_{token}_ratio" for token, _ in THRESHOLDS)
    )
    cohorts = ["projection_cull"]
    if "tile_mask_keep_unique_gaussians" in rows[0]:
        cohorts.append("tile_mask_keep")
    for cohort in cohorts:
        for component in COMPONENTS:
            component_total = total(
                f"{cohort}_{component}_parameter_elements"
            )
            component_zero = total(
                f"{cohort}_{component}_zero_gradient_elements"
            )
            exact_ratio = component_zero / component_total if component_total else 0.0
            near_ratios = []
            for token, _ in THRESHOLDS:
                near_count = total(
                    f"{cohort}_{component}_abs_le_{token}_gradient_elements"
                )
                near_ratios.append(
                    near_count / component_total if component_total else 0.0
                )
            print(
                f"{cohort}\t{component}\t{component_total}\t{exact_ratio:.6%}\t"
                + "\t".join(f"{ratio:.6%}" for ratio in near_ratios)
            )
    print("row_zero_elements_quantiles\tp10\tp50\tp90\tp99\tmean")
    mean = zero_count / gaussian_count if gaussian_count else 0.0
    print(
        "row_zero_elements_quantiles\t"
        f"{_quantile_from_histogram(histogram, 0.10)}\t"
        f"{_quantile_from_histogram(histogram, 0.50)}\t"
        f"{_quantile_from_histogram(histogram, 0.90)}\t"
        f"{_quantile_from_histogram(histogram, 0.99)}\t{mean:.3f}"
    )
    near_histogram = [
        total(f"projection_cull_near_zero_elements_{count}_gaussians")
        for count in range(60)
    ]
    near_total = sum(count * rows for count, rows in enumerate(near_histogram))
    near_mean = near_total / gaussian_count if gaussian_count else 0.0
    threshold_values = {
        row.get("near_zero_threshold", "") for row in rows
    }
    threshold_label = (
        threshold_values.pop() if len(threshold_values) == 1 else "mixed"
    )
    print(
        "row_near_zero_elements_quantiles\tthreshold\tp10\tp50\tp90\tp99\tmean"
    )
    print(
        "row_near_zero_elements_quantiles\t"
        f"{threshold_label}\t"
        f"{_quantile_from_histogram(near_histogram, 0.10)}\t"
        f"{_quantile_from_histogram(near_histogram, 0.50)}\t"
        f"{_quantile_from_histogram(near_histogram, 0.90)}\t"
        f"{_quantile_from_histogram(near_histogram, 0.99)}\t{near_mean:.3f}"
    )


if __name__ == "__main__":
    main()
