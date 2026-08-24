#!/usr/bin/env python3
"""Aggregate and compare TideGS gradient sparsity across TSP windows."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


BLOCK_SIZE = 4096
PARAMETERS_PER_GAUSSIAN = 59
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
REPORT_METRIC_KEYS = {
    "cull_over_working_set_pct",
    "adam_over_cull_pct",
    "all_zero_rows_over_cull_pct",
    "exact_zero_over_cull_59d_pct",
    "abs_le_1em8_over_cull_59d_pct",
    "significant_gt_1e8_over_cull_59d_pct",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    numerator: Callable[[Dict[str, int]], int]
    denominator: Callable[[Dict[str, int]], int]


METRICS = (
    MetricSpec(
        "cull_over_working_set_pct",
        "Cull / working set",
        lambda counts: counts["cull_rows"],
        lambda counts: counts["resident_rows"],
    ),
    MetricSpec(
        "adam_over_cull_pct",
        "Adam touched / Cull",
        lambda counts: counts["touched_rows"],
        lambda counts: counts["cull_rows"],
    ),
    MetricSpec(
        "all_zero_rows_over_cull_pct",
        "All-zero rows / Cull",
        lambda counts: counts["all_zero_rows"],
        lambda counts: counts["cull_rows"],
    ),
    MetricSpec(
        "exact_zero_over_cull_59d_pct",
        "Exact-zero / Cull 59d",
        lambda counts: counts["exact_zero_elements"],
        lambda counts: counts["cull_parameter_elements"],
    ),
) + tuple(
    MetricSpec(
        f"abs_le_{token}_over_cull_59d_pct",
        f"abs(g)<={threshold:g} / Cull 59d",
        lambda counts, token=token: counts[f"abs_le_{token}_elements"],
        lambda counts: counts["cull_parameter_elements"],
    )
    for token, threshold in NEAR_ZERO_THRESHOLDS
) + (
    MetricSpec(
        "significant_gt_1e8_over_cull_59d_pct",
        "abs(g)>1e-8 / Cull 59d",
        lambda counts: (
            counts["cull_parameter_elements"]
            - counts[f"abs_le_{PRIMARY_NEAR_TOKEN}_elements"]
        ),
        lambda counts: counts["cull_parameter_elements"],
    ),
)


def read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_int(value: str, field: str) -> int:
    parsed = float(value)
    integer = int(parsed)
    if parsed != integer:
        raise ValueError(f"{field} is not integer-valued: {value!r}")
    return integer


def percentage(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError(f"invalid denominator: {denominator}")
    return 100.0 * numerator / denominator


def counts_from_rows(
    grad_rows: Sequence[Dict[str, str]],
    batch_rows: Sequence[Dict[str, str]],
    *,
    resident_block_field: str = "global_resident_blocks",
) -> Dict[str, int]:
    if len(grad_rows) != len(batch_rows) or not grad_rows:
        raise ValueError("gradient and batch rows must have equal non-zero length")
    counts = {
        "batches": len(grad_rows),
        "resident_rows": 0,
        "cull_rows": 0,
        "touched_rows": 0,
        "all_zero_rows": 0,
        "cull_parameter_elements": 0,
        "exact_zero_elements": 0,
        "nonfinite_elements": 0,
    }
    for token, _ in NEAR_ZERO_THRESHOLDS:
        counts[f"abs_le_{token}_elements"] = 0
    for component in COMPONENT_WIDTHS:
        counts[f"{component}_parameter_elements"] = 0
        counts[f"{component}_exact_zero_elements"] = 0
        for token, _ in NEAR_ZERO_THRESHOLDS:
            counts[f"{component}_abs_le_{token}_elements"] = 0

    for grad, batch in zip(grad_rows, batch_rows):
        grad_iteration = as_int(grad["iteration"], "iteration")
        batch_iteration = as_int(batch["iteration"], "iteration")
        if grad_iteration != batch_iteration:
            raise AssertionError(
                f"iteration mismatch: grad={grad_iteration}, batch={batch_iteration}"
            )
        resident = as_int(batch[resident_block_field], resident_block_field)
        resident *= BLOCK_SIZE
        cull = as_int(
            grad["projection_cull_unique_gaussians"],
            "projection_cull_unique_gaussians",
        )
        parameter_elements = as_int(
            grad["projection_cull_parameter_elements"],
            "projection_cull_parameter_elements",
        )
        if parameter_elements != PARAMETERS_PER_GAUSSIAN * cull:
            raise AssertionError(f"59d mismatch at iteration {grad_iteration}")
        counts["resident_rows"] += resident
        counts["cull_rows"] += cull
        counts["touched_rows"] += as_int(
            batch["optimizer_touched_rows"], "optimizer_touched_rows"
        )
        counts["all_zero_rows"] += as_int(
            grad["projection_cull_all_zero_gradient_gaussians"],
            "projection_cull_all_zero_gradient_gaussians",
        )
        counts["cull_parameter_elements"] += parameter_elements
        counts["exact_zero_elements"] += as_int(
            grad["projection_cull_zero_gradient_elements"],
            "projection_cull_zero_gradient_elements",
        )
        for token, _ in NEAR_ZERO_THRESHOLDS:
            field = f"projection_cull_abs_le_{token}_gradient_elements"
            counts[f"abs_le_{token}_elements"] += as_int(grad[field], field)
        counts["nonfinite_elements"] += as_int(
            grad["projection_cull_nonfinite_gradient_elements"],
            "projection_cull_nonfinite_gradient_elements",
        )
        for component in COMPONENT_WIDTHS:
            counts[f"{component}_parameter_elements"] += as_int(
                grad[f"projection_cull_{component}_parameter_elements"],
                f"{component}_parameter_elements",
            )
            counts[f"{component}_exact_zero_elements"] += as_int(
                grad[f"projection_cull_{component}_zero_gradient_elements"],
                f"{component}_zero_gradient_elements",
            )
            for token, _ in NEAR_ZERO_THRESHOLDS:
                field = (
                    f"projection_cull_{component}_abs_le_{token}_gradient_elements"
                )
                counts[f"{component}_abs_le_{token}_elements"] += as_int(
                    grad[field], field
                )

    if counts["nonfinite_elements"] != 0:
        raise AssertionError(f"non-finite gradients: {counts['nonfinite_elements']}")
    if counts["cull_rows"] - counts["all_zero_rows"] != counts["touched_rows"]:
        raise AssertionError("Cull - all-zero rows != Adam touched rows")
    return counts


def metric_values(counts: Dict[str, int]) -> Dict[str, float]:
    values = {
        spec.key: percentage(spec.numerator(counts), spec.denominator(counts))
        for spec in METRICS
    }
    working_elements = counts["resident_rows"] * PARAMETERS_PER_GAUSSIAN
    values["significant_gt_1e8_over_working_59n_pct"] = percentage(
        counts["cull_parameter_elements"]
        - counts[f"abs_le_{PRIMARY_NEAR_TOKEN}_elements"],
        working_elements,
    )
    return values


def batch_metrics(
    grad_rows: Sequence[Dict[str, str]],
    batch_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    result = []
    for grad, batch in zip(grad_rows, batch_rows):
        counts = counts_from_rows([grad], [batch])
        row = {
            "iteration": as_int(grad["iteration"], "iteration"),
            **metric_values(counts),
        }
        result.append(row)
    return result


def component_metrics(counts: Dict[str, int]) -> List[Dict[str, object]]:
    rows = []
    for component, width in COMPONENT_WIDTHS.items():
        total = counts[f"{component}_parameter_elements"]
        exact = counts[f"{component}_exact_zero_elements"]
        row = {
            "component": component,
            "width": width,
            "exact_zero_pct": percentage(exact, total),
        }
        for token, _ in NEAR_ZERO_THRESHOLDS:
            near = counts[f"{component}_abs_le_{token}_elements"]
            row[f"abs_le_{token}_pct"] = percentage(near, total)
        primary_near = counts[
            f"{component}_abs_le_{PRIMARY_NEAR_TOKEN}_elements"
        ]
        row["significant_gt_1e8_pct"] = percentage(total - primary_near, total)
        rows.append(row)
    return rows


def load_camera_trace(model_path: Path) -> Tuple[List[int], List[Dict[str, str]]]:
    rows = read_tsv(model_path / "metrics_camera_batches.tsv")
    camera_ids: List[int] = []
    for row in rows:
        ids = [int(value) for value in json.loads(row["camera_ids_json"])]
        if len(ids) != as_int(row["camera_count"], "camera_count"):
            raise AssertionError("camera trace count mismatch")
        camera_ids.extend(ids)
    return camera_ids, rows


def load_rank_metrics(model_path: Path) -> List[Dict[str, object]]:
    grad_paths = list(model_path.glob("metrics_grad_zero_rank*.tsv"))
    grad_paths.extend(model_path.glob("rank_*/metrics_grad_zero_rank*.tsv"))
    results = []
    for grad_path in sorted(grad_paths):
        grad_rows = read_tsv(grad_path)
        if not grad_rows:
            raise ValueError(f"empty rank gradient metrics: {grad_path}")
        rank = as_int(grad_rows[0]["rank"], "rank")
        batch_path = grad_path.parent / f"metrics_batch_rank{rank}.tsv"
        batch_rows = read_tsv(batch_path)
        counts = counts_from_rows(
            grad_rows,
            batch_rows,
            resident_block_field="rank_resident_blocks",
        )
        results.append(
            {
                "rank": rank,
                "counts": counts,
                "metrics": metric_values(counts),
            }
        )
    if [result["rank"] for result in results] != [0, 1, 2, 3]:
        raise AssertionError(f"unexpected rank metrics for {model_path}")
    return results


def aggregate_run(model_path: Path, *, include_trace: bool = True) -> Dict[str, object]:
    grad_rows = read_tsv(model_path / "metrics_grad_zero_global.tsv")
    batch_rows = read_tsv(model_path / "metrics_batch_global.tsv")
    counts = counts_from_rows(grad_rows, batch_rows)
    warm_counts = counts_from_rows(grad_rows[1:], batch_rows[1:])
    result: Dict[str, object] = {
        "model_path": str(model_path),
        "counts": counts,
        "exclude_first_batch_counts": warm_counts,
        "metrics": metric_values(counts),
        "exclude_first_batch_metrics": metric_values(warm_counts),
        "batch_metrics": batch_metrics(grad_rows, batch_rows),
        "component_metrics": component_metrics(counts),
        "rank_metrics": load_rank_metrics(model_path),
    }
    if include_trace:
        camera_ids, camera_rows = load_camera_trace(model_path)
        with (model_path / "schedule_metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata["analysis_window_camera_ids"] != camera_ids:
            raise AssertionError("camera trace differs from schedule metadata")
        result["camera_ids"] = camera_ids
        result["camera_rows"] = camera_rows
        result["schedule_metadata"] = metadata
    return result


def pooled_counts(
    runs: Iterable[Dict[str, object]],
    *,
    counts_key: str = "counts",
) -> Dict[str, int]:
    total: Dict[str, int] = {}
    for run in runs:
        for key, value in run[counts_key].items():
            total[key] = total.get(key, 0) + int(value)
    return total


def write_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def format_pct(value: float) -> str:
    return f"{value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-default", type=Path, nargs=2, required=True)
    parser.add_argument("--historical-forced-sh3", type=Path, nargs=1, required=True)
    args = parser.parse_args()

    manifest_rows = read_tsv(args.manifest)
    if len(manifest_rows) != 16:
        raise ValueError(f"expected 16 matrix runs, found {len(manifest_rows)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, object]] = []
    for manifest_row in manifest_rows:
        run = aggregate_run(Path(manifest_row["model_path"]))
        run["run_tag"] = manifest_row["run_tag"]
        run["offset"] = int(manifest_row["offset"])
        run["sh_mode"] = manifest_row["sh_mode"]
        if len(run["batch_metrics"]) != 16:
            raise AssertionError(f"run does not contain 16 batches: {run['run_tag']}")
        runs.append(run)

    by_key = {(run["sh_mode"], run["offset"]): run for run in runs}
    offsets = sorted({int(run["offset"]) for run in runs})
    if len(offsets) != 8:
        raise AssertionError(f"expected 8 offsets, found {offsets}")
    for offset in offsets:
        default_ids = by_key[("default", offset)]["camera_ids"]
        forced_ids = by_key[("forced_sh3", offset)]["camera_ids"]
        if default_ids != forced_ids:
            raise AssertionError(f"SH camera mismatch at offset {offset}")
    for mode in ("default", "forced_sh3"):
        windows = [(offset, set(by_key[(mode, offset)]["camera_ids"])) for offset in offsets]
        for index, (offset, camera_ids) in enumerate(windows):
            for previous_offset, previous_ids in windows[:index]:
                if camera_ids & previous_ids:
                    raise AssertionError(
                        f"camera overlap for {mode}: {offset} vs {previous_offset}"
                    )

    historical_default = [
        aggregate_run(path, include_trace=False) for path in args.historical_default
    ]
    historical_forced = aggregate_run(
        args.historical_forced_sh3[0], include_trace=False
    )
    current_default_zero = by_key[("default", 0)]
    current_forced_zero = by_key[("forced_sh3", 0)]

    noise_by_mode: Dict[str, Dict[str, float]] = {"default": {}, "forced_sh3": {}}
    for spec in METRICS:
        default_values = [run["metrics"][spec.key] for run in historical_default]
        default_values.append(current_default_zero["metrics"][spec.key])
        noise_by_mode["default"][spec.key] = max(default_values) - min(default_values)
        noise_by_mode["forced_sh3"][spec.key] = abs(
            current_forced_zero["metrics"][spec.key]
            - historical_forced["metrics"][spec.key]
        )

    run_rows = []
    batch_rows = []
    component_rows = []
    rank_rows = []
    for run in sorted(runs, key=lambda value: (value["sh_mode"], value["offset"])):
        metadata = run["schedule_metadata"]
        geometry = metadata["analysis_window_geometry"]
        row = {
            "sh_mode": run["sh_mode"],
            "offset": run["offset"],
            "run_tag": run["run_tag"],
            **run["metrics"],
            **{
                f"exclude_first_{key}": value
                for key, value in run["exclude_first_batch_metrics"].items()
            },
            "cluster_count": geometry["cluster_count"],
            "position_centroid_json": json.dumps(geometry["position_centroid"]),
            "position_min_json": json.dumps(geometry["position_min"]),
            "position_max_json": json.dumps(geometry["position_max"]),
            "mean_view_direction_json": json.dumps(geometry["mean_view_direction"]),
            "cluster_histogram_json": json.dumps(geometry["cluster_histogram"]),
        }
        run_rows.append(row)
        for batch in run["batch_metrics"]:
            batch_rows.append(
                {
                    "sh_mode": run["sh_mode"],
                    "offset": run["offset"],
                    **batch,
                }
            )
        for component in run["component_metrics"]:
            component_rows.append(
                {
                    "sh_mode": run["sh_mode"],
                    "offset": run["offset"],
                    **component,
                }
            )
        for rank_result in run["rank_metrics"]:
            rank_rows.append(
                {
                    "sh_mode": run["sh_mode"],
                    "offset": run["offset"],
                    "rank": rank_result["rank"],
                    **rank_result["metrics"],
                }
            )

    summary_rows = []
    exclude_first_summary_rows = []
    locality_by_mode: Dict[str, bool] = {}
    for mode in ("default", "forced_sh3"):
        mode_runs = [run for run in runs if run["sh_mode"] == mode]
        pooled = metric_values(pooled_counts(mode_runs))
        locality_by_mode[mode] = False
        for spec in METRICS:
            values = [float(run["metrics"][spec.key]) for run in mode_runs]
            range_pp = max(values) - min(values)
            noise_pp = noise_by_mode[mode][spec.key]
            meaningful = range_pp > 1.0 and range_pp > 5.0 * noise_pp
            locality_by_mode[mode] = locality_by_mode[mode] or meaningful
            summary_rows.append(
                {
                    "sh_mode": mode,
                    "metric": spec.key,
                    "label": spec.label,
                    "mean_pct": statistics.mean(values),
                    "sample_std_pct": statistics.stdev(values),
                    "min_pct": min(values),
                    "max_pct": max(values),
                    "range_pp": range_pp,
                    "pooled_pct": pooled[spec.key],
                    "repeat_noise_pp": noise_pp,
                    "meaningful_locality": int(meaningful),
                }
            )
        warm_pooled = metric_values(
            pooled_counts(mode_runs, counts_key="exclude_first_batch_counts")
        )
        for spec in METRICS:
            values = [
                float(run["exclude_first_batch_metrics"][spec.key])
                for run in mode_runs
            ]
            exclude_first_summary_rows.append(
                {
                    "sh_mode": mode,
                    "metric": spec.key,
                    "label": spec.label,
                    "mean_pct": statistics.mean(values),
                    "sample_std_pct": statistics.stdev(values),
                    "min_pct": min(values),
                    "max_pct": max(values),
                    "range_pp": max(values) - min(values),
                    "pooled_pct": warm_pooled[spec.key],
                }
            )

    paired_rows = []
    for offset in offsets:
        default = by_key[("default", offset)]
        forced = by_key[("forced_sh3", offset)]
        for spec in METRICS:
            paired_rows.append(
                {
                    "offset": offset,
                    "metric": spec.key,
                    "default_pct": default["metrics"][spec.key],
                    "forced_sh3_pct": forced["metrics"][spec.key],
                    "forced_minus_default_pp": (
                        forced["metrics"][spec.key] - default["metrics"][spec.key]
                    ),
                }
            )

    component_summary_rows = []
    for mode in ("default", "forced_sh3"):
        mode_runs = [run for run in runs if run["sh_mode"] == mode]
        pooled = pooled_counts(mode_runs)
        for component, width in COMPONENT_WIDTHS.items():
            total = pooled[f"{component}_parameter_elements"]
            exact = pooled[f"{component}_exact_zero_elements"]
            primary_near = pooled[
                f"{component}_abs_le_{PRIMARY_NEAR_TOKEN}_elements"
            ]
            row = {
                "sh_mode": mode,
                "component": component,
                "width": width,
                "exact_zero_pct": percentage(exact, total),
                "significant_gt_1e8_pct": percentage(total - primary_near, total),
            }
            for token, _ in NEAR_ZERO_THRESHOLDS:
                row[f"abs_le_{token}_pct"] = percentage(
                    pooled[f"{component}_abs_le_{token}_elements"], total
                )
            component_summary_rows.append(row)

    rank_spread_rows = []
    for run in runs:
        for spec in METRICS:
            values = [
                float(rank_result["metrics"][spec.key])
                for rank_result in run["rank_metrics"]
            ]
            rank_spread_rows.append(
                {
                    "sh_mode": run["sh_mode"],
                    "offset": run["offset"],
                    "metric": spec.key,
                    "label": spec.label,
                    "rank_min_pct": min(values),
                    "rank_max_pct": max(values),
                    "rank_range_pp": max(values) - min(values),
                }
            )

    write_tsv(args.output_dir / "scene_locality_runs.tsv", run_rows)
    write_tsv(args.output_dir / "scene_locality_batches.tsv", batch_rows)
    write_tsv(args.output_dir / "scene_locality_components.tsv", component_rows)
    write_tsv(
        args.output_dir / "scene_locality_component_summary.tsv",
        component_summary_rows,
    )
    write_tsv(args.output_dir / "scene_locality_ranks.tsv", rank_rows)
    write_tsv(args.output_dir / "scene_locality_rank_spread.tsv", rank_spread_rows)
    write_tsv(args.output_dir / "scene_locality_summary.tsv", summary_rows)
    write_tsv(
        args.output_dir / "scene_locality_summary_exclude_first.tsv",
        exclude_first_summary_rows,
    )
    write_tsv(args.output_dir / "scene_locality_paired_sh.tsv", paired_rows)

    serializable_runs = []
    for run in runs:
        serializable_runs.append(
            {
                key: value
                for key, value in run.items()
                if key not in {"camera_rows"}
            }
        )
    analysis = {
        "manifest": str(args.manifest),
        "offsets": offsets,
        "window_camera_count": 1024,
        "total_unique_camera_count": 8 * 1024,
        "dataset_camera_count": runs[0]["schedule_metadata"]["num_cameras"],
        "camera_windows_disjoint": True,
        "camera_order_matches_between_sh_modes": True,
        "decision_rule": {
            "minimum_range_pp": 1.0,
            "minimum_repeat_noise_multiple": 5.0,
        },
        "meaningful_locality_by_mode": locality_by_mode,
        "repeat_noise_by_mode": noise_by_mode,
        "summary": summary_rows,
        "exclude_first_batch_summary": exclude_first_summary_rows,
        "paired_sh": paired_rows,
        "component_summary": component_summary_rows,
        "rank_spread": rank_spread_rows,
        "runs": serializable_runs,
    }
    with (args.output_dir / "scene_locality_analysis.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(analysis, handle, indent=2, sort_keys=True)
        handle.write("\n")

    report_lines = [
        "# TideGS TSP场景局部性实验",
        "",
        "## 结论",
        "",
    ]
    for mode, label in (("default", "默认SH"), ("forced_sh3", "强制SH3")):
        conclusion = "存在有实际意义的场景局部性" if locality_by_mode[mode] else "在8个采样片段内保持稳定"
        report_lines.append(f"- **{label}：{conclusion}。**")
    report_lines.extend(["", "## 跨片段汇总", ""])
    for mode, label in (("default", "默认SH"), ("forced_sh3", "强制SH3")):
        report_lines.extend(
            [
                f"### {label}",
                "",
                "| 指标 | 均值±标准差 | 最小–最大 | 极差 | 局部性 |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in summary_rows:
            if row["sh_mode"] != mode or row["metric"] not in REPORT_METRIC_KEYS:
                continue
            report_lines.append(
                f"| {row['label']} | {row['mean_pct']:.2f}% ± {row['sample_std_pct']:.2f} | "
                f"{row['min_pct']:.2f}%–{row['max_pct']:.2f}% | "
                f"{row['range_pp']:.2f} pp | "
                f"{'**是**' if row['meaningful_locality'] else '否'} |"
            )
        report_lines.append("")
    report_lines.extend(
        [
            "## 逐片段结果",
            "",
            "| SH | Offset | Cull / N | 全零行 / Cull | Exact-zero / 59d | abs(g)>1e-8 / 59d |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for run in sorted(runs, key=lambda value: (value["offset"], value["sh_mode"])):
        metrics = run["metrics"]
        report_lines.append(
            f"| {run['sh_mode']} | {run['offset']} | "
            f"{metrics['cull_over_working_set_pct']:.2f}% | "
            f"{metrics['all_zero_rows_over_cull_pct']:.2f}% | "
            f"{metrics['exact_zero_over_cull_59d_pct']:.2f}% | "
            f"{metrics['significant_gt_1e8_over_cull_59d_pct']:.2f}% |"
        )
    report_lines.extend(
        [
            "",
            "## 参数类型（8个片段 pooled）",
            "",
            "| SH | 参数 | 维度 | Exact-zero | abs(g)>1e-8 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in component_summary_rows:
        report_lines.append(
            f"| {row['sh_mode']} | {row['component']} | {row['width']} | "
            f"{row['exact_zero_pct']:.2f}% | "
            f"{row['significant_gt_1e8_pct']:.2f}% |"
        )
    report_lines.extend(
        [
            "",
            "## Rank差异",
            "",
            "| SH | 指标 | 16组中最大rank极差 |",
            "|---|---|---:|",
        ]
    )
    for mode in ("default", "forced_sh3"):
        for spec in METRICS:
            if spec.key not in REPORT_METRIC_KEYS:
                continue
            max_spread = max(
                row["rank_range_pp"]
                for row in rank_spread_rows
                if row["sh_mode"] == mode and row["metric"] == spec.key
            )
            report_lines.append(
                f"| {mode} | {spec.label} | {max_spread:.2f} pp |"
            )
    report_lines.extend(
        [
            "",
            "## 排除首个冷启动batch",
            "",
            "| SH | 指标 | 均值±标准差 | 最小–最大 | 极差 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    cold_report_keys = {
        "all_zero_rows_over_cull_pct",
        "exact_zero_over_cull_59d_pct",
        "significant_gt_1e8_over_cull_59d_pct",
    }
    for row in exclude_first_summary_rows:
        if row["metric"] not in cold_report_keys:
            continue
        report_lines.append(
            f"| {row['sh_mode']} | {row['label']} | "
            f"{row['mean_pct']:.2f}% ± {row['sample_std_pct']:.2f} | "
            f"{row['min_pct']:.2f}%–{row['max_pct']:.2f}% | "
            f"{row['range_pp']:.2f} pp |"
        )
    report_lines.extend(
        [
            "",
            "## 覆盖与核验",
            "",
            f"- TSP offsets：`{', '.join(str(value) for value in offsets)}`",
            f"- 覆盖camera：**{8 * 1024}/{runs[0]['schedule_metadata']['num_cameras']}**",
            "- 8个窗口无重叠，默认SH与强制SH3的camera对象及顺序完全一致。",
            "- 比例均由原始计数累计得到，不是逐batch百分比的简单平均。",
            "- 7档near-zero阈值见`scene_locality_summary.tsv`和参数类型TSV。",
            "- 冷启动batch排除口径见`scene_locality_summary_exclude_first.tsv`。",
            "",
        ]
    )
    (args.output_dir / "scene_locality_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
