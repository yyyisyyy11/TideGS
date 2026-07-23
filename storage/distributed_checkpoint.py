"""Root manifest for rank-sharded TideGS incremental checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from storage.pure_ssd_checkpoint import (
    _atomic_json_dump,
    load_pure_ssd_checkpoint_manifest,
    write_pure_ssd_incremental_checkpoint,
)


DISTRIBUTED_CHECKPOINT_MANIFEST = "pure_ssd_distributed_checkpoint.json"


def is_distributed_checkpoint(path: str | Path) -> bool:
    return bool(path) and (Path(path) / DISTRIBUTED_CHECKPOINT_MANIFEST).is_file()


def load_distributed_checkpoint_manifest(
    path: str | Path,
    *,
    rank: int,
    world_size: int,
    global_bsz: int | None = None,
) -> Dict[str, Any]:
    checkpoint_dir = Path(path).resolve()
    root_file = checkpoint_dir / DISTRIBUTED_CHECKPOINT_MANIFEST
    if not root_file.is_file():
        raise FileNotFoundError(f"Distributed checkpoint manifest not found: {root_file}")
    with open(root_file, "r", encoding="utf-8") as handle:
        root = json.load(handle)

    saved_world_size = int(root["world_size"])
    if saved_world_size != int(world_size):
        raise RuntimeError(
            f"Distributed checkpoint WORLD_SIZE mismatch: saved={saved_world_size}, "
            f"current={world_size}"
        )
    if global_bsz is not None and int(root.get("global_bsz", -1)) != int(global_bsz):
        raise RuntimeError(
            f"Distributed checkpoint global batch mismatch: "
            f"saved={root.get('global_bsz')} current={global_bsz}"
        )
    rank_manifests = list(root["rank_manifests"])
    if len(rank_manifests) != saved_world_size:
        raise RuntimeError("Distributed checkpoint has an invalid rank_manifests list")
    rank_dir = Path(rank_manifests[int(rank)])
    if not rank_dir.is_absolute():
        rank_dir = checkpoint_dir / rank_dir
    manifest = load_pure_ssd_checkpoint_manifest(rank_dir)

    owner_file = Path(root["block_owner"])
    bounds_file = Path(root["block_bounds"])
    if not owner_file.is_absolute():
        owner_file = checkpoint_dir / owner_file
    if not bounds_file.is_absolute():
        bounds_file = checkpoint_dir / bounds_file
    manifest["block_bounds"] = str(bounds_file.resolve())
    manifest["next_iteration"] = int(root["next_iteration"])
    manifest["_tide_distributed_root"] = root
    manifest["_tide_distributed_checkpoint_dir"] = str(checkpoint_dir)
    manifest["_tide_block_owner_file"] = str(owner_file.resolve())
    checkpoint_version = int(root.get("checkpoint_version", 1))
    if checkpoint_version >= 2:
        global_capacity = int(root["global_capacity_blocks"])
    else:
        global_capacity = int(root["per_rank_capacity_blocks"]) * saved_world_size
    manifest["_tide_global_capacity_blocks"] = global_capacity
    manifest["_tide_owner_policy"] = str(
        root.get("owner_policy", "visibility_weighted_lpt")
    )
    return manifest


def _atomic_numpy_save(array: np.ndarray, path: Path) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    with open(temp, "wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def write_distributed_incremental_checkpoint(
    *,
    context,
    storage_adapter,
    gaussians,
    checkpoint_dir: str | Path,
    iteration: int,
    next_iteration: int,
    args,
    block_owner,
    log_file=None,
) -> Dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    if context.is_rank0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    context.barrier()

    rank_dir = checkpoint_dir / f"rank_{context.rank}"
    rank_manifest = write_pure_ssd_incremental_checkpoint(
        storage_adapter=storage_adapter,
        gaussians=gaussians,
        checkpoint_dir=rank_dir,
        iteration=iteration,
        next_iteration=next_iteration,
        args=args,
        log_file=log_file,
    )
    rank_result = {
        "rank": int(context.rank),
        "path": f"rank_{context.rank}",
        "patch_files": int(rank_manifest.get("patch_files", 0)),
        "patch_bytes": int(rank_manifest.get("patch_bytes", 0)),
    }
    rank_results = context.all_gather_object(rank_result)
    context.barrier()

    root_manifest = None
    if context.is_rank0:
        owner_file = checkpoint_dir / "block_owner.npy"
        bounds_file = checkpoint_dir / "block_bounds.npy"
        _atomic_numpy_save(np.asarray(block_owner, dtype=np.int32), owner_file)
        _atomic_numpy_save(
            np.asarray(storage_adapter.block_bounds, dtype=np.float32), bounds_file
        )
        ordered = sorted(rank_results, key=lambda value: int(value["rank"]))
        root_manifest = {
            "checkpoint_version": 2,
            "checkpoint_type": "pure_ssd_distributed_incremental",
            "iteration": int(iteration),
            "next_iteration": int(next_iteration),
            "world_size": int(context.world_size),
            "global_bsz": int(args.bsz),
            "global_capacity_blocks": int(args.paper_resident_capacity_blocks),
            "capacity_semantics": "global",
            "owner_policy": str(
                getattr(args, "_tide_owner_policy", "stable_round_robin")
            ),
            "block_version_semantics": "monotonic_gpu_cpu_ssd",
            "camera_assignment": str(args.tide_camera_assignment),
            "camera_microbatch": int(args.tide_camera_microbatch),
            "block_owner": owner_file.name,
            "block_bounds": bounds_file.name,
            "rank_manifests": [value["path"] for value in ordered],
            "rank_patch_files": [value["patch_files"] for value in ordered],
            "rank_patch_bytes": [value["patch_bytes"] for value in ordered],
            "total_points": int(storage_adapter.num_points),
            "num_blocks": int(storage_adapter.num_blocks),
            "block_size": int(storage_adapter.block_size),
            "param_dim": 59,
        }
        _atomic_json_dump(root_manifest, checkpoint_dir / DISTRIBUTED_CHECKPOINT_MANIFEST)
    root_manifest = context.broadcast_object(root_manifest)
    context.barrier()
    return root_manifest
