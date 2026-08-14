#!/usr/bin/env python3
"""Shared helpers for deterministic Pure SSD checkpoint quality evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import stat
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


METRIC_NAMES = ("psnr", "ssim", "lpips_alex", "render_ms")


def fingerprint_camera_schedule(schedule: Sequence[int]) -> str:
    """Return a stable fingerprint for an ordered camera schedule."""
    payload = ",".join(str(int(camera_id)) for camera_id in schedule)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


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


def checkpoint_tree_fingerprint(path: str | Path) -> Dict[str, object]:
    """Hash every directory, file, and symlink in a checkpoint tree.

    The aggregate includes relative names, file metadata, symlink targets, and
    regular-file contents.  It intentionally does not follow symlinks outside
    the checkpoint tree.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"checkpoint directory not found: {root}")

    digest = hashlib.sha256()
    directory_count = 0
    file_count = 0
    symlink_count = 0
    total_bytes = 0
    entries = [root, *root.rglob("*")]
    for entry in sorted(
        entries,
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = entry.relative_to(root).as_posix()
        metadata = entry.lstat()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.S_IMODE(metadata.st_mode)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(int(metadata.st_size)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(int(metadata.st_mtime_ns)).encode("ascii"))
        digest.update(b"\0")
        if stat.S_ISDIR(metadata.st_mode):
            directory_count += 1
            digest.update(b"D")
        elif stat.S_ISLNK(metadata.st_mode):
            symlink_count += 1
            digest.update(b"L")
            digest.update(os.readlink(entry).encode("utf-8"))
        elif stat.S_ISREG(metadata.st_mode):
            file_count += 1
            total_bytes += int(metadata.st_size)
            digest.update(b"F")
            with open(entry, "rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
        else:
            digest.update(b"O")
        digest.update(b"\n")
    return {
        "path": str(root),
        "directory_count": directory_count,
        "file_count": file_count,
        "symlink_count": symlink_count,
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


class CheckpointShardBlockReader:
    """Read one Pure SSD checkpoint shard without creating writable storage."""

    def __init__(self, manifest: Mapping[str, object]):
        from storage.block_reader import BlockLayout

        self.layout = BlockLayout.CACHE
        self.total_gaussians = int(manifest["total_points"])
        self.block_size = int(manifest["block_size"])
        self.num_blocks = int(manifest["num_blocks"])
        self.point_dim = int(manifest.get("param_dim", 59))
        if self.total_gaussians <= 0 or self.block_size <= 0:
            raise ValueError("checkpoint shard dimensions must be positive")
        expected_blocks = (
            self.total_gaussians + self.block_size - 1
        ) // self.block_size
        if self.num_blocks != expected_blocks:
            raise ValueError(
                f"checkpoint shard num_blocks mismatch: "
                f"manifest={self.num_blocks} expected={expected_blocks}"
            )
        if self.point_dim != 59:
            raise ValueError(f"quality reader requires param_dim=59, got {self.point_dim}")
        self._bytes_per_row = self.point_dim * 4
        self._locations: Dict[int, tuple[Path, int, int]] = {}

        checkpoint_type = str(manifest.get("checkpoint_type", "pure_ssd_snapshot"))
        if checkpoint_type == "pure_ssd_incremental":
            self._load_incremental_index(Path(str(manifest["storage_index"])))
        else:
            base_file = Path(str(manifest["base_file"])).resolve()
            if not base_file.is_file():
                raise FileNotFoundError(f"checkpoint shard base file not found: {base_file}")
            required_size = self.total_gaussians * self._bytes_per_row
            if base_file.stat().st_size < required_size:
                raise RuntimeError(
                    f"checkpoint shard base file is truncated: "
                    f"size={base_file.stat().st_size} expected_at_least={required_size}"
                )
            for block_id in range(self.num_blocks):
                rows = self._rows_for_block(block_id)
                self._locations[block_id] = (
                    base_file,
                    block_id * self.block_size * self._bytes_per_row,
                    rows * self._bytes_per_row,
                )

    def _rows_for_block(self, block_id: int) -> int:
        start = int(block_id) * self.block_size
        return max(0, min(self.block_size, self.total_gaussians - start))

    def _load_incremental_index(self, index_path: Path) -> None:
        index_path = index_path.resolve()
        with open(index_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if str(payload.get("dtype", "float32")) != "float32":
            raise ValueError(
                f"checkpoint shard index dtype must be float32, got {payload.get('dtype')!r}"
            )
        expected = {
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "point_dim": self.point_dim,
        }
        for field, value in expected.items():
            if int(payload.get(field, -1)) != value:
                raise ValueError(
                    f"checkpoint shard index {field} mismatch: "
                    f"manifest={payload.get(field)!r} expected={value}"
                )
        files: Dict[int, Path] = {}
        for file_id_raw, file_info in payload.get("files", {}).items():
            file_id = int(file_id_raw)
            if file_id in files:
                raise ValueError(
                    f"checkpoint shard index contains duplicate file_id={file_id}"
                )
            file_path = Path(str(file_info["path"]))
            if not file_path.is_absolute():
                file_path = index_path.parent / file_path
            file_path = file_path.resolve()
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"checkpoint shard file_id={file_id_raw} not found: {file_path}"
                )
            files[file_id] = file_path
        for block_id_raw, location in payload.get("index", {}).items():
            block_id = int(block_id_raw)
            if block_id in self._locations:
                raise ValueError(
                    f"checkpoint shard index contains duplicate block_id={block_id}"
                )
            if block_id < 0 or block_id >= self.num_blocks:
                raise ValueError(
                    f"checkpoint shard index block_id={block_id} is out of range"
                )
            file_id = int(location["file_id"])
            if file_id not in files:
                raise KeyError(f"block {block_id} references missing file_id={file_id}")
            offset = int(location["offset"])
            stored_size = int(location["size"])
            expected_size = self._rows_for_block(block_id) * self._bytes_per_row
            if offset < 0 or stored_size < expected_size:
                raise ValueError(
                    f"checkpoint shard index has an invalid location for block {block_id}: "
                    f"offset={offset} size={stored_size} expected_at_least={expected_size}"
                )
            if offset + expected_size > files[file_id].stat().st_size:
                raise RuntimeError(
                    f"checkpoint shard file is truncated for block {block_id}: "
                    f"path={files[file_id]} offset={offset} bytes={expected_size}"
                )
            self._locations[block_id] = (
                files[file_id],
                offset,
                stored_size,
            )
        if set(self._locations) != set(range(self.num_blocks)):
            raise ValueError(
                f"checkpoint shard index must cover all {self.num_blocks} blocks; "
                f"got {len(self._locations)}"
            )

    def read_blocks(self, block_ids: Sequence[int]):
        import numpy as np
        import torch

        result = {}
        for raw_block_id in dict.fromkeys(int(value) for value in block_ids):
            if not self.contains_block(raw_block_id):
                continue
            file_path, offset, stored_size = self._locations[raw_block_id]
            expected_size = self._rows_for_block(raw_block_id) * self._bytes_per_row
            if stored_size < expected_size:
                raise RuntimeError(
                    f"block {raw_block_id} stores {stored_size} bytes, expected at least {expected_size}"
                )
            with open(file_path, "rb") as handle:
                handle.seek(offset)
                raw = handle.read(expected_size)
            if len(raw) != expected_size:
                raise RuntimeError(
                    f"short read for block {raw_block_id}: got {len(raw)}, expected {expected_size}"
                )
            array = np.frombuffer(raw, dtype=np.float32).copy().reshape(-1, self.point_dim)
            result[raw_block_id] = torch.from_numpy(array)
        return result

    def read_batch(self, block_ids: Sequence[int], out=None):
        from storage.block_reader import _pack_block_batch

        valid_ids = [int(value) for value in block_ids if self.contains_block(value)]
        return _pack_block_batch(self.read_blocks(valid_ids), valid_ids, self.layout, out)

    def hint_future(self, block_ids: Sequence[int], *, target_iteration=None) -> int:
        return 0

    def contains_block(self, block_id: int) -> bool:
        return 0 <= int(block_id) < self.num_blocks


class OwnerRoutedBlockReader:
    """Route every distributed checkpoint block to its authoritative rank."""

    def __init__(self, readers: Sequence[object], block_owner: Sequence[int]):
        import operator

        if not readers:
            raise ValueError("at least one shard reader is required")
        self._readers = tuple(readers)
        try:
            self._owner = tuple(operator.index(value) for value in block_owner)
        except TypeError as error:
            raise ValueError("block_owner must be a flat sequence of integers") from error
        self.num_blocks = len(self._owner)
        first = readers[0]
        self.layout = first.layout
        self.total_gaussians = int(first.total_gaussians)
        self.block_size = int(first.block_size)
        for rank, reader in enumerate(readers):
            if int(reader.total_gaussians) != self.total_gaussians:
                raise ValueError(f"rank {rank} total_gaussians mismatch")
            if int(reader.block_size) != self.block_size:
                raise ValueError(f"rank {rank} block_size mismatch")
            if int(reader.num_blocks) != self.num_blocks:
                raise ValueError(f"rank {rank} num_blocks mismatch")
            if reader.layout != self.layout:
                raise ValueError(f"rank {rank} block layout mismatch")
        invalid = [
            block_id
            for block_id, rank in enumerate(self._owner)
            if rank < 0 or rank >= len(readers)
        ]
        if invalid:
            block_id = invalid[0]
            raise ValueError(
                f"block_owner[{block_id}]={int(self._owner[block_id])} "
                f"outside [0, {len(readers)})"
            )

    def read_blocks(self, block_ids: Sequence[int]):
        grouped: Dict[int, List[int]] = {}
        for block_id in dict.fromkeys(int(value) for value in block_ids):
            if self.contains_block(block_id):
                grouped.setdefault(int(self._owner[block_id]), []).append(block_id)
        result = {}
        for rank, owned_ids in grouped.items():
            loaded = self._readers[rank].read_blocks(owned_ids)
            missing = sorted(set(owned_ids) - set(loaded))
            if missing:
                raise RuntimeError(f"rank {rank} did not return owned blocks {missing[:8]}")
            unexpected = sorted(set(loaded) - set(owned_ids))
            if unexpected:
                raise RuntimeError(
                    f"rank {rank} returned unrequested blocks {unexpected[:8]}"
                )
            result.update(loaded)
        return result

    def read_batch(self, block_ids: Sequence[int], out=None):
        from storage.block_reader import _pack_block_batch

        valid_ids = [int(value) for value in block_ids if self.contains_block(value)]
        return _pack_block_batch(self.read_blocks(valid_ids), valid_ids, self.layout, out)

    def hint_future(self, block_ids: Sequence[int], *, target_iteration=None) -> int:
        grouped: Dict[int, List[int]] = {}
        for block_id in dict.fromkeys(int(value) for value in block_ids):
            if self.contains_block(block_id):
                grouped.setdefault(int(self._owner[block_id]), []).append(block_id)
        submitted = 0
        for rank, owned_ids in grouped.items():
            hint = getattr(self._readers[rank], "hint_future", None)
            if callable(hint):
                submitted += int(hint(owned_ids, target_iteration=target_iteration) or 0)
        return submitted

    def contains_block(self, block_id: int) -> bool:
        return 0 <= int(block_id) < self.num_blocks


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
    "CheckpointShardBlockReader",
    "OwnerRoutedBlockReader",
    "checkpoint_manifest_fingerprint",
    "checkpoint_tree_fingerprint",
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
