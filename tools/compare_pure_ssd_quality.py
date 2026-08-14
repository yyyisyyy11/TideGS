#!/usr/bin/env python3
"""Paired per-camera comparison of Adam and a candidate optimizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.pure_ssd_quality_utils import read_tsv, summarize_values, write_tsv  # noqa: E402


COMPARE_METRICS = ("psnr", "ssim", "lpips_alex", "render_ms")


def resolve_metrics_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "per_camera_metrics.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"per-camera metrics not found: {path}")
    return path


def camera_key(row: Mapping[str, str]) -> Tuple[int, str]:
    return int(row["camera_index"]), str(row["image_name"])


def validate_camera_alignment(
    adam_rows: Sequence[Mapping[str, str]],
    candidate_rows: Sequence[Mapping[str, str]],
) -> None:
    adam_keys = [camera_key(row) for row in adam_rows]
    candidate_keys = [camera_key(row) for row in candidate_rows]
    if len(adam_keys) != len(set(adam_keys)):
        raise ValueError("Adam metrics contain duplicate camera index/name pairs")
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("Candidate metrics contain duplicate camera index/name pairs")
    if adam_keys != candidate_keys:
        adam_set = set(adam_keys)
        candidate_set = set(candidate_keys)
        missing = sorted(adam_set - candidate_set)[:5]
        extra = sorted(candidate_set - adam_set)[:5]
        first_mismatch = next(
            (
                index
                for index, pair in enumerate(zip(adam_keys, candidate_keys))
                if pair[0] != pair[1]
            ),
            None,
        )
        raise ValueError(
            "A/B camera sets or order do not match: "
            f"adam={len(adam_keys)} candidate={len(candidate_keys)} "
            f"first_mismatch={first_mismatch} missing={missing} extra={extra}"
        )


def build_paired_rows(
    adam_rows: Sequence[Mapping[str, str]],
    candidate_rows: Sequence[Mapping[str, str]],
) -> List[Dict[str, object]]:
    validate_camera_alignment(adam_rows, candidate_rows)
    paired: List[Dict[str, object]] = []
    for adam_row, candidate_row in zip(adam_rows, candidate_rows):
        row: Dict[str, object] = {
            "camera_index": int(adam_row["camera_index"]),
            "image_name": adam_row["image_name"],
        }
        for metric in COMPARE_METRICS:
            adam_value = float(adam_row[metric])
            candidate_value = float(candidate_row[metric])
            row[f"adam_{metric}"] = adam_value
            row[f"candidate_{metric}"] = candidate_value
            row[f"delta_{metric}"] = candidate_value - adam_value
        paired.append(row)
    return paired


def compare_quality(adam: str | Path, candidate: str | Path, output_dir: str | Path) -> Dict[str, object]:
    adam_path = resolve_metrics_path(adam)
    candidate_path = resolve_metrics_path(candidate)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adam_rows = read_tsv(adam_path)
    candidate_rows = read_tsv(candidate_path)
    paired_rows = build_paired_rows(adam_rows, candidate_rows)
    if not paired_rows:
        raise ValueError("cannot compare empty metric files")

    fields = ["camera_index", "image_name"]
    for metric in COMPARE_METRICS:
        fields.extend((f"adam_{metric}", f"candidate_{metric}", f"delta_{metric}"))
    write_tsv(output_dir / "paired_quality_comparison.tsv", paired_rows, fields)

    metrics_summary: Dict[str, object] = {}
    for metric in COMPARE_METRICS:
        adam_values = [float(row[f"adam_{metric}"]) for row in paired_rows]
        candidate_values = [float(row[f"candidate_{metric}"]) for row in paired_rows]
        deltas = [float(row[f"delta_{metric}"]) for row in paired_rows]
        metrics_summary[metric] = {
            "adam": summarize_values(adam_values),
            "candidate": summarize_values(candidate_values),
            "candidate_minus_adam": summarize_values(deltas),
            "delta_positive_fraction": sum(delta > 0.0 for delta in deltas) / len(deltas),
            "delta_negative_fraction": sum(delta < 0.0 for delta in deltas) / len(deltas),
            "delta_zero_fraction": sum(delta == 0.0 for delta in deltas) / len(deltas),
        }

    summary = {
        "camera_count": len(paired_rows),
        "camera_alignment_verified": True,
        "delta_definition": "Candidate - Adam",
        "adam_metrics": str(adam_path.resolve()),
        "candidate_metrics": str(candidate_path.resolve()),
        "metrics": metrics_summary,
    }
    with open(output_dir / "paired_quality_comparison.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Adam and candidate quality by camera.")
    parser.add_argument("--adam", required=True, help="Adam quality directory or per_camera_metrics.tsv.")
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate quality directory or per_camera_metrics.tsv.",
    )
    parser.add_argument("--output-dir", required=True, help="Comparison output directory.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = compare_quality(args.adam, args.candidate, args.output_dir)
    print(
        f"[QUALITY COMPARE] complete: cameras={summary['camera_count']} "
        f"output={Path(args.output_dir).resolve()}"
    )


if __name__ == "__main__":
    main()
