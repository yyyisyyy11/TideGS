#!/usr/bin/env python3
"""Shared helpers for deterministic Pure SSD checkpoint quality evaluation."""

from __future__ import annotations

import csv
import hashlib
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


METRIC_NAMES = ("psnr", "ssim", "lpips_alex", "render_ms")


def select_preview_indices(camera_count: int, preview_count: int) -> List[int]:
    """Return deterministic, unique camera indices spanning the full split."""
    camera_count = int(camera_count)
    preview_count = int(preview_count)
    if camera_count <= 0 or preview_count <= 0:
        return []
    count = min(camera_count, preview_count)
    if count == camera_count:
        return list(range(camera_count))
    if count == 1:
        return [0]
    step = float(camera_count - 1) / float(count - 1)
    result = [int(math.floor(index * step + 0.5)) for index in range(count)]
    if len(result) != len(set(result)):
        raise AssertionError("uniform preview selection produced duplicate indices")
    return result


def compute_psnr(render, target, max_value: float = 1.0) -> float:
    """Compute RGB PSNR from same-shaped Torch tensors."""
    import torch

    if tuple(render.shape) != tuple(target.shape):
        raise ValueError(
            f"PSNR inputs must have the same shape, got {tuple(render.shape)} and {tuple(target.shape)}"
        )
    mse = torch.mean((render.float() - target.float()) ** 2)
    mse_value = float(mse.item())
    if mse_value == 0.0:
        return float("inf")
    return float(10.0 * math.log10((float(max_value) ** 2) / mse_value))


def summarize_values(values: Iterable[float]) -> Dict[str, float]:
    array = [float(value) for value in values]
    if not array:
        raise ValueError("cannot summarize an empty metric")
    if not all(math.isfinite(value) for value in array):
        raise ValueError("metric contains NaN or infinite values")
    ordered = sorted(array)

    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * float(percent) / 100.0
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean": float(statistics.fmean(array)),
        "median": float(statistics.median(array)),
        "p5": float(percentile(5)),
        "p95": float(percentile(95)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def summarize_metric_rows(
    rows: Sequence[Mapping[str, object]],
    metric_names: Sequence[str] = METRIC_NAMES,
) -> Dict[str, Dict[str, float]]:
    return {
        metric: summarize_values(float(row[metric]) for row in rows)
        for metric in metric_names
    }


def write_tsv(path: str | Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_manifest_fingerprint(path: str | Path) -> Dict[str, object]:
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def validate_resident_configuration(
    policy: str,
    capacity: int,
    resident_lambda: float,
    recency_decay: float,
    balanced_seed_fraction: float,
) -> None:
    if str(policy).lower() != "topc_balanced":
        raise ValueError(f"quality evaluation requires topc_balanced, got {policy!r}")
    if int(capacity) <= 0:
        raise ValueError(f"resident capacity must be positive, got {capacity}")
    for name, value in (
        ("resident_lambda", resident_lambda),
        ("recency_decay", recency_decay),
        ("balanced_seed_fraction", balanced_seed_fraction),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")


def select_initial_resident_blocks(
    current_blocks: Sequence[int],
    current_camera_blocks: Mapping[int, Sequence[int]],
    *,
    capacity: int,
    resident_lambda: float,
    recency_decay: float,
    balanced_seed_fraction: float,
) -> List[int]:
    from strategies.tide_engine.resident_policy import compute_topc_resident_transition

    transition = compute_topc_resident_transition(
        current_active_blocks=list(current_blocks),
        next_active_blocks=list(current_blocks),
        current_resident_blocks=[],
        next_camera_blocks={int(key): list(value) for key, value in current_camera_blocks.items()},
        previous_recency_scores={},
        lambda_weight=float(resident_lambda),
        recency_decay=float(recency_decay),
        resident_capacity_blocks=int(capacity),
        balanced_camera_seeds=True,
        balanced_seed_fraction=float(balanced_seed_fraction),
    )
    resident = sorted(int(block_id) for block_id in transition.next_resident_blocks)
    if len(resident) > int(capacity):
        raise AssertionError("resident selection exceeded its configured capacity")
    return resident


def compute_next_resident_transition(
    current_blocks: Sequence[int],
    next_blocks: Sequence[int],
    current_resident_blocks: Sequence[int],
    next_camera_ids: Sequence[int],
    next_camera_blocks: Mapping[int, Sequence[int]],
    previous_recency_scores: Mapping[int, float],
    *,
    capacity: int,
    resident_lambda: float,
    recency_decay: float,
    balanced_seed_fraction: float,
):
    from strategies.tide_engine.resident_policy import compute_topc_resident_transition

    transition = compute_topc_resident_transition(
        current_active_blocks=list(current_blocks),
        next_active_blocks=list(next_blocks),
        current_resident_blocks=list(current_resident_blocks),
        next_camera_ids=list(next_camera_ids),
        next_camera_blocks={int(key): list(value) for key, value in next_camera_blocks.items()},
        previous_recency_scores={int(key): float(value) for key, value in previous_recency_scores.items()},
        lambda_weight=float(resident_lambda),
        recency_decay=float(recency_decay),
        resident_capacity_blocks=int(capacity),
        balanced_camera_seeds=True,
        balanced_seed_fraction=float(balanced_seed_fraction),
    )
    if len(transition.next_resident_blocks) > int(capacity):
        raise AssertionError("resident transition exceeded its configured capacity")
    return transition


__all__ = [
    "METRIC_NAMES",
    "checkpoint_manifest_fingerprint",
    "compute_next_resident_transition",
    "compute_psnr",
    "read_tsv",
    "select_initial_resident_blocks",
    "select_preview_indices",
    "sha256_file",
    "summarize_metric_rows",
    "summarize_values",
    "validate_resident_configuration",
    "write_tsv",
]
