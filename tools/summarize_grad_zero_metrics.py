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
    args = parser.parse_args()

    with args.metrics_tsv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise SystemExit(f"No metric rows in {args.metrics_tsv}")

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
    print(f"projection_cull_unique_gaussians\t{gaussian_count}")
    print(f"gradient_elements\t{parameter_count}")
    print(f"exact_zero_gradient_elements\t{zero_count}")
    print(
        "exact_zero_gradient_ratio\t"
        f"{zero_count / parameter_count:.6%}" if parameter_count else "exact_zero_gradient_ratio\tN/A"
    )
    print(
        "all_zero_gradient_gaussian_ratio\t"
        f"{total(ALL_ZERO_ROWS_FIELD) / gaussian_count:.6%}"
        if gaussian_count
        else "all_zero_gradient_gaussian_ratio\tN/A"
    )
    print("component\tzero_elements\ttotal_elements\tzero_ratio")
    for component in COMPONENTS:
        component_total = total(
            f"projection_cull_{component}_parameter_elements"
        )
        component_zero = total(
            f"projection_cull_{component}_zero_gradient_elements"
        )
        ratio = component_zero / component_total if component_total else 0.0
        print(f"{component}\t{component_zero}\t{component_total}\t{ratio:.6%}")
    print("row_zero_elements_quantiles\tp10\tp50\tp90\tp99\tmean")
    mean = zero_count / gaussian_count if gaussian_count else 0.0
    print(
        "row_zero_elements_quantiles\t"
        f"{_quantile_from_histogram(histogram, 0.10)}\t"
        f"{_quantile_from_histogram(histogram, 0.50)}\t"
        f"{_quantile_from_histogram(histogram, 0.90)}\t"
        f"{_quantile_from_histogram(histogram, 0.99)}\t{mean:.3f}"
    )


if __name__ == "__main__":
    main()
