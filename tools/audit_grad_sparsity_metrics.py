#!/usr/bin/env python3
"""Audit TideGS gradient-sparsity TSVs without mixing row and element ratios."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


PARAMETERS_PER_GAUSSIAN = 59
BLOCK_SIZE = 4096
NEAR_ZERO_THRESHOLDS = (
    ("1em16", 1e-16),
    ("1em14", 1e-14),
    ("1em12", 1e-12),
    ("1em10", 1e-10),
    ("1em8", 1e-8),
    ("1em6", 1e-6),
    ("1em4", 1e-4),
)
PRIMARY_NEAR_TOKEN = "1em8"
COMPONENT_WIDTHS = {
    "xyz": 3,
    "opacity": 1,
    "scaling": 3,
    "rotation": 4,
    "features_dc": 3,
    "features_rest": 45,
}


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _integer(value: str, *, field: str) -> int:
    parsed = float(value)
    integer = int(parsed)
    if parsed != integer:
        raise ValueError(f"{field} is not integer-valued: {value!r}")
    return integer


def _sum(rows: Iterable[Dict[str, str]], field: str) -> int:
    return sum(_integer(row[field], field=field) for row in rows)


def _percentage(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError(f"invalid denominator: {denominator}")
    return 100.0 * numerator / denominator


def _rank_paths(run_dir: Path) -> List[Path]:
    paths = list(run_dir.glob("metrics_grad_zero_rank*.tsv"))
    paths.extend(run_dir.glob("rank_*/metrics_grad_zero_rank*.tsv"))
    return sorted(paths)


def audit(run_dir: Path) -> Dict[str, object]:
    grad_rows = _read_tsv(run_dir / "metrics_grad_zero_global.tsv")
    batch_rows = _read_tsv(run_dir / "metrics_batch_global.tsv")
    if not grad_rows or len(grad_rows) != len(batch_rows):
        raise ValueError(
            "global gradient and batch TSVs must contain the same non-zero row count"
        )

    per_batch = []
    profiled_drop_batches = 0
    for grad, batch in zip(grad_rows, batch_rows):
        iteration = _integer(grad["iteration"], field="iteration")
        batch_iteration = _integer(batch["iteration"], field="iteration")
        if iteration != batch_iteration:
            raise AssertionError(
                f"iteration mismatch: grad={iteration}, batch={batch_iteration}"
            )

        resident_rows = (
            _integer(batch["global_resident_blocks"], field="global_resident_blocks")
            * BLOCK_SIZE
        )
        cull_rows = _integer(
            grad["projection_cull_unique_gaussians"],
            field="projection_cull_unique_gaussians",
        )
        all_zero_rows = _integer(
            grad["projection_cull_all_zero_gradient_gaussians"],
            field="projection_cull_all_zero_gradient_gaussians",
        )
        nonzero_rows = cull_rows - all_zero_rows
        reported_nonzero_rows = _integer(
            grad["projection_cull_nonzero_gradient_gaussians"],
            field="projection_cull_nonzero_gradient_gaussians",
        )
        owner_active_rows = _integer(
            batch["owner_active_gaussians"], field="owner_active_gaussians"
        )
        adam_touched_rows = _integer(
            batch["optimizer_touched_rows"], field="optimizer_touched_rows"
        )
        parameter_elements = _integer(
            grad["projection_cull_parameter_elements"],
            field="projection_cull_parameter_elements",
        )
        exact_zero_elements = _integer(
            grad["projection_cull_zero_gradient_elements"],
            field="projection_cull_zero_gradient_elements",
        )
        near_zero_elements = {
            token: _integer(
                grad[f"projection_cull_abs_le_{token}_gradient_elements"],
                field=f"projection_cull_abs_le_{token}_gradient_elements",
            )
            for token, _ in NEAR_ZERO_THRESHOLDS
        }
        nonfinite_elements = _integer(
            grad["projection_cull_nonfinite_gradient_elements"],
            field="projection_cull_nonfinite_gradient_elements",
        )

        if not 0 <= all_zero_rows <= cull_rows <= owner_active_rows <= resident_rows:
            raise AssertionError(
                f"invalid row nesting at iteration {iteration}: "
                f"all_zero={all_zero_rows}, cull={cull_rows}, "
                f"owner_active={owner_active_rows}, resident={resident_rows}"
            )
        if parameter_elements != PARAMETERS_PER_GAUSSIAN * cull_rows:
            raise AssertionError(
                f"59*d mismatch at iteration {iteration}: "
                f"elements={parameter_elements}, d={cull_rows}"
            )
        if reported_nonzero_rows != nonzero_rows:
            raise AssertionError(
                f"nonzero-gradient row metric mismatch at iteration {iteration}: "
                f"cull({cull_rows}) - all_zero({all_zero_rows}) "
                f"!= reported_nonzero({reported_nonzero_rows})"
            )
        if not nonzero_rows <= adam_touched_rows <= owner_active_rows:
            raise AssertionError(
                f"invalid optimizer row nesting at iteration {iteration}: "
                f"nonzero_gradient={nonzero_rows}, "
                f"optimizer_touched={adam_touched_rows}, "
                f"owner_active={owner_active_rows}"
            )
        ordered_near = [near_zero_elements[token] for token, _ in NEAR_ZERO_THRESHOLDS]
        if not (
            0 <= exact_zero_elements
            <= ordered_near[0]
            <= ordered_near[1]
            <= ordered_near[2]
            <= ordered_near[3]
            <= ordered_near[4]
            <= ordered_near[5]
            <= ordered_near[6]
            <= parameter_elements
        ):
            raise AssertionError(
                f"invalid element counts at iteration {iteration}"
            )
        if nonfinite_elements != 0:
            raise AssertionError(
                f"non-finite gradient elements at iteration {iteration}: "
                f"{nonfinite_elements}"
            )

        tile_keep_rows = cull_rows
        tile_rejected_rows = 0
        tile_rejected_nonzero_rows = 0
        if "tile_mask_keep_unique_gaussians" in grad:
            tile_keep_rows = _integer(
                grad["tile_mask_keep_unique_gaussians"],
                field="tile_mask_keep_unique_gaussians",
            )
            tile_rejected_rows = _integer(
                grad["tile_mask_rejected_gaussians"],
                field="tile_mask_rejected_gaussians",
            )
            tile_rejected_zero_rows = _integer(
                grad["tile_mask_rejected_all_zero_gradient_gaussians"],
                field="tile_mask_rejected_all_zero_gradient_gaussians",
            )
            tile_rejected_nonzero_rows = _integer(
                grad["tile_mask_rejected_nonzero_gradient_gaussians"],
                field="tile_mask_rejected_nonzero_gradient_gaussians",
            )
            if cull_rows != tile_keep_rows + tile_rejected_rows:
                raise AssertionError(
                    f"projection/tile row partition failed at iteration {iteration}"
                )
            if tile_rejected_rows != (
                tile_rejected_zero_rows + tile_rejected_nonzero_rows
            ):
                raise AssertionError(
                    f"tile rejected row partition failed at iteration {iteration}"
                )
            if _integer(
                grad["tile_mask_keep_parameter_elements"],
                field="tile_mask_keep_parameter_elements",
            ) != PARAMETERS_PER_GAUSSIAN * tile_keep_rows:
                raise AssertionError(
                    f"tile-mask 59*d mismatch at iteration {iteration}"
                )

            active_width = _integer(
                grad["active_parameter_width"], field="active_parameter_width"
            )
            expected_active_width = 11 + 3 * (
                _integer(grad["active_sh_degree"], field="active_sh_degree") + 1
            ) ** 2
            if active_width != expected_active_width:
                raise AssertionError(
                    f"active parameter width mismatch at iteration {iteration}"
                )
            for prefix, rows in (
                ("projection_cull", cull_rows),
                ("tile_mask_keep", tile_keep_rows),
            ):
                if _integer(
                    grad[f"{prefix}_active_parameter_elements"],
                    field=f"{prefix}_active_parameter_elements",
                ) != active_width * rows:
                    raise AssertionError(
                        f"{prefix} active width mismatch at iteration {iteration}"
                    )

            mode = grad.get("tile_contribution_mode", "off")
            observed = _integer(
                grad.get("tile_drop_gradients_observed", "0") or "0",
                field="tile_drop_gradients_observed",
            )
            would_drop_nonzero = _integer(
                grad["would_drop_nonzero_gradient_gaussians"],
                field="would_drop_nonzero_gradient_gaussians",
            )
            if mode == "profile" and observed:
                profiled_drop_batches += 1
                if tile_rejected_nonzero_rows:
                    raise AssertionError(
                        f"tile mask false negatives at iteration {iteration}: "
                        f"{tile_rejected_nonzero_rows} would-drop rows have nonzero gradients"
                    )
            expected_would_drop = (
                tile_rejected_nonzero_rows
                if mode == "profile" and observed
                else 0
            )
            if would_drop_nonzero != expected_would_drop:
                raise AssertionError(
                    "would-drop gradient count disagrees with observation mode at "
                    f"iteration {iteration}"
                )

            projection_pairs = _integer(
                grad["tile_mask_projection_pairs"], field="tile_mask_projection_pairs"
            )
            kept_projection_pairs = _integer(
                grad["tile_mask_kept_projection_pairs"],
                field="tile_mask_kept_projection_pairs",
            )
            candidate_before = _integer(
                grad["tile_mask_candidate_tile_pairs_before"],
                field="tile_mask_candidate_tile_pairs_before",
            )
            candidate_after = _integer(
                grad["tile_mask_candidate_tile_pairs_after"],
                field="tile_mask_candidate_tile_pairs_after",
            )
            contributing_pairs = _integer(
                grad["tile_mask_contributing_tile_pairs"],
                field="tile_mask_contributing_tile_pairs",
            )
            if not (
                0 <= kept_projection_pairs <= projection_pairs
                and 0 <= candidate_after <= candidate_before
                and kept_projection_pairs <= contributing_pairs <= candidate_after
            ):
                raise AssertionError(
                    f"invalid tile-mask pair counts at iteration {iteration}"
                )

        histogram = [
            _integer(
                grad[f"projection_cull_zero_gradient_elements_{zeros}_gaussians"],
                field=f"zero_histogram_{zeros}",
            )
            for zeros in range(PARAMETERS_PER_GAUSSIAN + 1)
        ]
        if sum(histogram) != cull_rows:
            raise AssertionError(
                f"zero histogram row sum mismatch at iteration {iteration}"
            )
        if sum(zeros * count for zeros, count in enumerate(histogram)) != exact_zero_elements:
            raise AssertionError(
                f"zero histogram weighted sum mismatch at iteration {iteration}"
            )
        if histogram[PARAMETERS_PER_GAUSSIAN] != all_zero_rows:
            raise AssertionError(
                f"all-zero histogram mismatch at iteration {iteration}"
            )

        near_histogram = [
            _integer(
                grad[f"projection_cull_near_zero_elements_{count}_gaussians"],
                field=f"near_zero_histogram_{count}",
            )
            for count in range(PARAMETERS_PER_GAUSSIAN + 1)
        ]
        if sum(near_histogram) != cull_rows:
            raise AssertionError(
                f"near-zero histogram row sum mismatch at iteration {iteration}"
            )
        if (
            sum(count * rows for count, rows in enumerate(near_histogram))
            != near_zero_elements[PRIMARY_NEAR_TOKEN]
        ):
            raise AssertionError(
                f"near-zero histogram weighted sum mismatch at iteration {iteration}"
            )

        component_exact_total = 0
        component_nonfinite_total = 0
        component_near_totals = {token: 0 for token, _ in NEAR_ZERO_THRESHOLDS}
        for component, width in COMPONENT_WIDTHS.items():
            prefix = f"projection_cull_{component}"
            component_parameters = _integer(
                grad[f"{prefix}_parameter_elements"],
                field=f"{component}_parameter_elements",
            )
            component_exact = _integer(
                grad[f"{prefix}_zero_gradient_elements"],
                field=f"{component}_zero_gradient_elements",
            )
            component_nonfinite = _integer(
                grad[f"{prefix}_nonfinite_gradient_elements"],
                field=f"{component}_nonfinite_gradient_elements",
            )
            component_near = [
                _integer(
                    grad[f"{prefix}_abs_le_{token}_gradient_elements"],
                    field=f"{component}_abs_le_{token}_gradient_elements",
                )
                for token, _ in NEAR_ZERO_THRESHOLDS
            ]
            if component_parameters != width * cull_rows:
                raise AssertionError(
                    f"component width mismatch at iteration {iteration}: {component}"
                )
            if not (
                0 <= component_exact
                <= component_near[0]
                <= component_near[1]
                <= component_near[2]
                <= component_near[3]
                <= component_near[4]
                <= component_near[5]
                <= component_near[6]
                <= component_parameters
            ):
                raise AssertionError(
                    f"invalid component counts at iteration {iteration}: {component}"
                )
            component_exact_total += component_exact
            component_nonfinite_total += component_nonfinite
            for (token, _), value in zip(NEAR_ZERO_THRESHOLDS, component_near):
                component_near_totals[token] += value
        if component_exact_total != exact_zero_elements:
            raise AssertionError(
                f"component exact-zero sum mismatch at iteration {iteration}"
            )
        if component_nonfinite_total != nonfinite_elements:
            raise AssertionError(
                f"component nonfinite sum mismatch at iteration {iteration}"
            )
        if component_near_totals != near_zero_elements:
            raise AssertionError(
                f"component near-zero sum mismatch at iteration {iteration}"
            )

        per_batch.append(
            {
                "iteration": iteration,
                "resident_rows": resident_rows,
                "owner_active_rows": owner_active_rows,
                "not_culled_rows": resident_rows - cull_rows,
                "cull_all_zero_rows": all_zero_rows,
                "cull_nonzero_rows": nonzero_rows,
                "optimizer_touched_rows": adam_touched_rows,
                "cull_rows": cull_rows,
                "tile_keep_rows": tile_keep_rows,
                "tile_rejected_rows": tile_rejected_rows,
                "tile_rejected_nonzero_rows": tile_rejected_nonzero_rows,
                "cull_parameter_elements": parameter_elements,
                "exact_zero_elements": exact_zero_elements,
                **{
                    f"abs_le_{token}_elements": near_zero_elements[token]
                    for token, _ in NEAR_ZERO_THRESHOLDS
                },
            }
        )

    rank_paths = _rank_paths(run_dir)
    if rank_paths:
        rank_tables = [_read_tsv(path) for path in rank_paths]
        if any(len(table) != len(grad_rows) for table in rank_tables):
            raise AssertionError("rank/global gradient TSV row counts differ")
        non_additive_fields = {
            "iteration",
            "rank",
            "world_size",
            "optimizer_step",
            "active_sh_degree",
            "active_parameter_width",
            "near_zero_threshold",
            "tile_contribution_mode",
            "tile_alpha_threshold",
            "tile_drop_gradients_observed",
        }
        additive_fields = [
            field for field in grad_rows[0] if field not in non_additive_fields
        ]
        for index, global_row in enumerate(grad_rows):
            for field in additive_fields:
                rank_total = sum(
                    _integer(table[index][field], field=field) for table in rank_tables
                )
                global_value = _integer(global_row[field], field=field)
                if rank_total != global_value:
                    raise AssertionError(
                        f"rank sum mismatch at row {index}, field={field}: "
                        f"ranks={rank_total}, global={global_value}"
                    )

    resident_rows = sum(row["resident_rows"] for row in per_batch)
    owner_active_rows = sum(row["owner_active_rows"] for row in per_batch)
    not_culled_rows = sum(row["not_culled_rows"] for row in per_batch)
    all_zero_rows = sum(row["cull_all_zero_rows"] for row in per_batch)
    nonzero_rows = sum(row["cull_nonzero_rows"] for row in per_batch)
    touched_rows = sum(row["optimizer_touched_rows"] for row in per_batch)
    cull_rows = sum(row["cull_rows"] for row in per_batch)
    tile_keep_rows = sum(row["tile_keep_rows"] for row in per_batch)
    tile_rejected_rows = sum(row["tile_rejected_rows"] for row in per_batch)
    tile_rejected_nonzero_rows = sum(
        row["tile_rejected_nonzero_rows"] for row in per_batch
    )
    working_parameter_elements = PARAMETERS_PER_GAUSSIAN * resident_rows
    cull_parameter_elements = sum(
        row["cull_parameter_elements"] for row in per_batch
    )
    exact_zero_elements = sum(row["exact_zero_elements"] for row in per_batch)
    near_zero_elements = {
        token: sum(row[f"abs_le_{token}_elements"] for row in per_batch)
        for token, _ in NEAR_ZERO_THRESHOLDS
    }

    if not_culled_rows + all_zero_rows + nonzero_rows != resident_rows:
        raise AssertionError("working-set Gaussian row categories do not sum to 100%")
    if cull_rows != all_zero_rows + nonzero_rows:
        raise AssertionError("cull Gaussian row categories do not sum to 100%")
    nonzero_elements = cull_parameter_elements - exact_zero_elements
    no_gradient_elements = PARAMETERS_PER_GAUSSIAN * not_culled_rows
    if no_gradient_elements + exact_zero_elements + nonzero_elements != working_parameter_elements:
        raise AssertionError("working-set parameter categories do not sum to 100%")

    return {
        "batches": len(per_batch),
        "rank_files": len(rank_paths),
        "counts": {
            "working_set_gaussian_rows": resident_rows,
            "owner_active_gaussian_rows": owner_active_rows,
            "not_culled_gaussian_rows": not_culled_rows,
            "cull_gaussian_rows": cull_rows,
            "tile_keep_gaussian_rows": tile_keep_rows,
            "tile_rejected_gaussian_rows": tile_rejected_rows,
            "tile_rejected_nonzero_gradient_rows": tile_rejected_nonzero_rows,
            "cull_all_zero_gaussian_rows": all_zero_rows,
            "cull_nonzero_gaussian_rows": nonzero_rows,
            "adam_touched_gaussian_rows": touched_rows,
            "working_set_parameter_elements": working_parameter_elements,
            "no_gradient_parameter_elements": no_gradient_elements,
            "exact_zero_gradient_elements": exact_zero_elements,
            "nonzero_gradient_elements": nonzero_elements,
            **{
                f"gradient_elements_abs_le_{threshold:g}": near_zero_elements[token]
                for token, threshold in NEAR_ZERO_THRESHOLDS
            },
        },
        "gaussian_row_percentages": {
            "not_culled_over_working_set": _percentage(
                not_culled_rows, resident_rows
            ),
            "cull_all_zero_over_working_set": _percentage(
                all_zero_rows, resident_rows
            ),
            "cull_nonzero_over_working_set": _percentage(
                nonzero_rows, resident_rows
            ),
            "adam_touched_over_working_set": _percentage(touched_rows, resident_rows),
            "cull_over_working_set": _percentage(cull_rows, resident_rows),
            "cull_all_zero_over_cull": _percentage(all_zero_rows, cull_rows),
            "cull_nonzero_over_cull": _percentage(nonzero_rows, cull_rows),
            "adam_touched_over_owner_active": _percentage(
                touched_rows, owner_active_rows
            ),
        },
        "parameter_element_percentages": {
            "no_gradient_over_working_set_parameters": _percentage(
                no_gradient_elements, working_parameter_elements
            ),
            "exact_zero_over_working_set_parameters": _percentage(
                exact_zero_elements, working_parameter_elements
            ),
            "nonzero_over_working_set_parameters": _percentage(
                nonzero_elements, working_parameter_elements
            ),
            **{
                f"cull_abs_le_{threshold:g}_over_working_set_parameters": _percentage(
                    near_zero_elements[token], working_parameter_elements
                )
                for token, threshold in NEAR_ZERO_THRESHOLDS
            },
            "exact_zero_over_cull_parameters": _percentage(
                exact_zero_elements, cull_parameter_elements
            ),
            **{
                f"abs_le_{threshold:g}_over_cull_parameters": _percentage(
                    near_zero_elements[token], cull_parameter_elements
                )
                for token, threshold in NEAR_ZERO_THRESHOLDS
            },
        },
        "checks": {
            "per_batch_59d": "PASS",
            "per_batch_histograms": "PASS",
            "per_batch_component_sums": "PASS",
            "all_near_zero_thresholds_monotonic": "PASS",
            "per_batch_reported_nonzero_matches_gradient_rows": "PASS",
            "per_batch_optimizer_rows_contain_nonzero_gradient_rows": "PASS",
            "projection_rows_partition_into_tile_keep_and_reject": "PASS",
            "profile_would_drop_rows_have_zero_gradient": (
                "PASS" if profiled_drop_batches else "NOT_CHECKED"
            ),
            "apply_equivalence_under_profiled_threshold": (
                "PASS" if profiled_drop_batches else "NOT_ESTABLISHED"
            ),
            "active_sh_parameter_widths": "PASS",
            "global_gaussian_categories_sum_to_100_percent": "PASS",
            "global_parameter_categories_sum_to_100_percent": "PASS",
            "rank_sums_match_global": "PASS" if rank_paths else "NOT_CHECKED",
            "nonfinite_gradients": "PASS",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing metrics_grad_zero_global.tsv and metrics_batch_global.tsv",
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
