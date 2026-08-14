"""Root manifest for rank-sharded TideGS incremental checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from storage.pure_ssd_checkpoint import (
    CHECKPOINT_MANIFEST,
    _atomic_json_dump,
    load_pure_ssd_checkpoint_manifest,
    write_pure_ssd_incremental_checkpoint,
)
from strategies.tide_engine.checkpoint_validation import (
    build_optimizer_provenance,
    validate_distributed_checkpoint_resume,
)


DISTRIBUTED_CHECKPOINT_MANIFEST = "pure_ssd_distributed_checkpoint.json"
DISTRIBUTED_CHECKPOINT_VERSION = 3


def is_distributed_checkpoint(path: str | Path) -> bool:
    return bool(path) and (Path(path) / DISTRIBUTED_CHECKPOINT_MANIFEST).is_file()


def _resolve_checkpoint_member(
    checkpoint_dir: Path,
    path_value: str | Path,
    *,
    label: str = "member",
    require_relative: bool = False,
) -> Path:
    path = Path(path_value)
    if require_relative and path.is_absolute():
        raise RuntimeError(
            f"Distributed checkpoint v3 {label} must be a relative path: {path}"
        )
    if not path.is_absolute():
        path = checkpoint_dir / path
    resolved = path.resolve()
    if require_relative:
        try:
            resolved.relative_to(checkpoint_dir.resolve())
        except ValueError as error:
            raise RuntimeError(
                f"Distributed checkpoint v3 {label} escapes the checkpoint tree: "
                f"{path_value}"
            ) from error
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected: Any, *, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"Distributed checkpoint {label} not found: {path}")
    actual = _sha256_file(path)
    if actual != str(expected).lower():
        raise RuntimeError(
            f"Distributed checkpoint {label} SHA-256 mismatch: "
            f"expected={expected} actual={actual} path={path}"
        )


def _normalize_planner_state(value: Any) -> Dict[str, Any]:
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise ValueError("Distributed checkpoint v3 requires a planner_state mapping")
    required = {"resident", "active", "recency"}
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(
            "Distributed checkpoint planner_state is missing: " + ", ".join(missing)
        )

    resident = [int(block_id) for block_id in value["resident"]]
    active = [int(block_id) for block_id in value["active"]]
    if any(block_id < 0 for block_id in resident + active):
        raise ValueError("Distributed checkpoint planner block IDs must be non-negative")
    if len(resident) != len(set(resident)):
        raise ValueError("Distributed checkpoint planner resident blocks must be unique")
    if len(active) != len(set(active)):
        raise ValueError("Distributed checkpoint planner active blocks must be unique")
    recency = {}
    for raw_block_id, raw_score in dict(value["recency"]).items():
        block_id = int(raw_block_id)
        if block_id < 0 or str(block_id) in recency:
            raise ValueError(
                "Distributed checkpoint planner recency block IDs must be unique "
                "non-negative integers"
            )
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(
                f"Distributed checkpoint planner recency is non-finite for block {block_id}"
            )
        recency[str(block_id)] = score
    return {
        "resident": resident,
        "active": active,
        "recency": recency,
    }


def planner_state_from_distributed_manifest(
    root: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return planner state with integer recency keys for planner restoration."""

    if int(root.get("checkpoint_version", 1)) < 3:
        raise ValueError("Distributed checkpoint v1/v2 has no resumable planner state")
    normalized = _normalize_planner_state(root.get("planner_state"))
    normalized["recency"] = {
        int(block_id): float(score)
        for block_id, score in normalized["recency"].items()
    }
    return normalized


def load_distributed_checkpoint_root(
    path: str | Path,
    *,
    verify_integrity: bool = True,
) -> Dict[str, Any]:
    """Load the root manifest and verify v3 rank/owner provenance hashes."""

    checkpoint_dir = Path(path).resolve()
    root_file = checkpoint_dir / DISTRIBUTED_CHECKPOINT_MANIFEST
    if not root_file.is_file():
        raise FileNotFoundError(f"Distributed checkpoint manifest not found: {root_file}")
    with open(root_file, "r", encoding="utf-8") as handle:
        root = json.load(handle)

    version = int(root.get("checkpoint_version", 1))
    if version not in {1, 2, DISTRIBUTED_CHECKPOINT_VERSION}:
        raise RuntimeError(
            "Unsupported distributed checkpoint version: "
            f"{version}; supported versions are 1, 2, and "
            f"{DISTRIBUTED_CHECKPOINT_VERSION}"
        )
    for field in (
        "world_size",
        "global_bsz",
        "next_iteration",
        "rank_manifests",
        "block_owner",
        "block_bounds",
    ):
        if field not in root:
            raise KeyError(f"Distributed checkpoint manifest missing required key: {field}")

    world_size = int(root["world_size"])
    rank_manifests = list(root["rank_manifests"])
    if world_size <= 0 or len(rank_manifests) != world_size:
        raise RuntimeError("Distributed checkpoint has an invalid rank_manifests list")

    if version == 1 and "per_rank_capacity_blocks" not in root:
        raise KeyError(
            "Distributed checkpoint v1 manifest missing required key: "
            "per_rank_capacity_blocks"
        )
    if version == 2 and "global_capacity_blocks" not in root:
        raise KeyError(
            "Distributed checkpoint v2 manifest missing required key: "
            "global_capacity_blocks"
        )
    if version == DISTRIBUTED_CHECKPOINT_VERSION:
        required_v3 = (
            "global_capacity_blocks",
            "capacity_semantics",
            "owner_policy",
            "block_version_semantics",
            "camera_assignment",
            "camera_microbatch",
            "optimizer_provenance",
            "global_optimizer_step",
            "optimizer_state_mode",
            "optimizer_ema_saved",
            "optimizer_curvature_saved",
            "planner_state",
            "block_owner_sha256",
            "rank_manifest_sha256",
            "total_points",
            "num_blocks",
            "block_size",
            "param_dim",
        )
        missing = [field for field in required_v3 if field not in root]
        if missing:
            raise KeyError(
                "Distributed checkpoint v3 manifest missing required keys: "
                + ", ".join(missing)
            )
        if not isinstance(root["optimizer_provenance"], Mapping):
            raise RuntimeError(
                "Distributed checkpoint v3 optimizer_provenance must be a mapping"
            )
        if str(root["capacity_semantics"]) != "global":
            raise RuntimeError(
                "Distributed checkpoint v3 capacity_semantics must be 'global'"
            )
        if str(root["block_version_semantics"]) != "monotonic_gpu_cpu_ssd":
            raise RuntimeError(
                "Distributed checkpoint v3 block_version_semantics must be "
                "'monotonic_gpu_cpu_ssd'"
            )
        global_bsz = int(root["global_bsz"])
        global_capacity = int(root["global_capacity_blocks"])
        next_iteration = int(root["next_iteration"])
        total_points = int(root["total_points"])
        planner_state = _normalize_planner_state(root["planner_state"])
        num_blocks = int(root["num_blocks"])
        block_size = int(root["block_size"])
        param_dim = int(root["param_dim"])
        if global_bsz <= 0 or global_capacity <= 0:
            raise RuntimeError(
                "Distributed checkpoint v3 batch size and global capacity must be positive"
            )
        if next_iteration < 1:
            raise RuntimeError(
                "Distributed checkpoint v3 next_iteration must be positive"
            )
        expected_num_blocks = (
            (total_points + block_size - 1) // block_size
            if total_points > 0 and block_size > 0
            else -1
        )
        if (
            total_points <= 0
            or block_size <= 0
            or num_blocks != expected_num_blocks
            or param_dim != 59
        ):
            raise RuntimeError(
                "Distributed checkpoint v3 has inconsistent Gaussian topology: "
                f"total_points={total_points} num_blocks={num_blocks} "
                f"block_size={block_size} param_dim={param_dim}"
            )
        planner_block_ids = (
            planner_state["resident"]
            + planner_state["active"]
            + [int(value) for value in planner_state["recency"]]
        )
        if num_blocks < 0 or any(block_id >= num_blocks for block_id in planner_block_ids):
            raise RuntimeError(
                "Distributed checkpoint planner_state contains an out-of-range block ID"
            )
        if len(planner_state["resident"]) > global_capacity:
            raise RuntimeError(
                "Distributed checkpoint planner_state exceeds global capacity: "
                f"resident={len(planner_state['resident'])} capacity={global_capacity}"
            )
        global_optimizer_step = int(root["global_optimizer_step"])
        expected_optimizer_step = max(0, (next_iteration - 1) // global_bsz)
        if global_optimizer_step != expected_optimizer_step:
            raise RuntimeError(
                "Distributed checkpoint global_optimizer_step mismatch: "
                f"checkpoint={global_optimizer_step} expected={expected_optimizer_step}"
            )
        if str(root["optimizer_state_mode"]) != "cold_start_per_rank":
            raise RuntimeError(
                "Distributed checkpoint v3 optimizer_state_mode must be "
                "'cold_start_per_rank'"
            )
        if bool(root["optimizer_ema_saved"]) or bool(
            root["optimizer_curvature_saved"]
        ):
            raise RuntimeError(
                "Distributed checkpoint v3 must not claim persisted optimizer EMA state"
            )
        rank_hashes = list(root["rank_manifest_sha256"])
        if len(rank_hashes) != world_size:
            raise RuntimeError(
                "Distributed checkpoint has an invalid rank_manifest_sha256 list"
            )
        owner_file = _resolve_checkpoint_member(
            checkpoint_dir,
            root["block_owner"],
            label="block_owner",
            require_relative=True,
        )
        bounds_file = _resolve_checkpoint_member(
            checkpoint_dir,
            root["block_bounds"],
            label="block_bounds",
            require_relative=True,
        )
        rank_dirs = [
            _resolve_checkpoint_member(
                checkpoint_dir,
                rank_dir_value,
                label=f"rank {rank} manifest directory",
                require_relative=True,
            )
            for rank, rank_dir_value in enumerate(rank_manifests)
        ]
        if len(rank_dirs) != len(set(rank_dirs)):
            raise RuntimeError(
                "Distributed checkpoint v3 rank_manifests must reference distinct directories"
            )
        if verify_integrity:
            _verify_sha256(
                owner_file,
                root["block_owner_sha256"],
                label="block_owner",
            )
            for rank, (rank_dir, expected_hash) in enumerate(
                zip(rank_dirs, rank_hashes)
            ):
                _verify_sha256(
                    rank_dir / CHECKPOINT_MANIFEST,
                    expected_hash,
                    label=f"rank {rank} manifest",
                )

        if not owner_file.is_file():
            raise RuntimeError(
                f"Distributed checkpoint block_owner not found: {owner_file}"
            )
        block_owner = np.load(owner_file, allow_pickle=False)
        if block_owner.shape != (num_blocks,):
            raise RuntimeError(
                "Distributed checkpoint block_owner shape mismatch: "
                f"expected={(num_blocks,)} actual={block_owner.shape}"
            )
        if not np.issubdtype(block_owner.dtype, np.integer):
            raise RuntimeError(
                "Distributed checkpoint block_owner must have an integer dtype, "
                f"got {block_owner.dtype}"
            )
        invalid_owner = np.flatnonzero(
            (block_owner < 0) | (block_owner >= world_size)
        )
        if invalid_owner.size:
            raise RuntimeError(
                "Distributed checkpoint block_owner contains invalid ranks at "
                f"blocks={invalid_owner[:8].tolist()}"
            )

        if not bounds_file.is_file():
            raise RuntimeError(
                f"Distributed checkpoint block_bounds not found: {bounds_file}"
            )
        block_bounds = np.load(bounds_file, allow_pickle=False)
        if block_bounds.shape != (num_blocks, 6):
            raise RuntimeError(
                "Distributed checkpoint block_bounds shape mismatch: "
                f"expected={(num_blocks, 6)} actual={block_bounds.shape}"
            )
        if not np.issubdtype(block_bounds.dtype, np.floating) or not np.all(
            np.isfinite(block_bounds)
        ):
            raise RuntimeError(
                "Distributed checkpoint block_bounds must contain finite floating-point values"
            )
        if np.any(block_bounds[:, :3] > block_bounds[:, 3:]):
            raise RuntimeError(
                "Distributed checkpoint block_bounds contains min/max inversions"
            )

        rank_metadata_fields = (
            "total_points",
            "num_blocks",
            "block_size",
            "param_dim",
            "next_iteration",
        )
        root_metadata = {
            field: int(root[field]) for field in rank_metadata_fields
        }
        active_sh_degree = None
        for rank, rank_dir in enumerate(rank_dirs):
            rank_manifest = load_pure_ssd_checkpoint_manifest(rank_dir)
            if str(rank_manifest.get("checkpoint_type")) != "pure_ssd_incremental":
                raise RuntimeError(
                    f"Distributed checkpoint rank {rank} must be incremental"
                )
            for field, expected in root_metadata.items():
                actual = int(rank_manifest[field])
                if actual != expected:
                    raise RuntimeError(
                        f"Distributed checkpoint rank {rank} {field} mismatch: "
                        f"root={expected} rank={actual}"
                    )
            rank_sh_degree = int(rank_manifest.get("active_sh_degree", 0))
            if active_sh_degree is None:
                active_sh_degree = rank_sh_degree
            elif rank_sh_degree != active_sh_degree:
                raise RuntimeError(
                    f"Distributed checkpoint rank {rank} active_sh_degree mismatch"
                )

    root["_tide_distributed_checkpoint_dir"] = str(checkpoint_dir)
    root["_tide_distributed_root_file"] = str(root_file)
    return root


def load_distributed_checkpoint_manifest(
    path: str | Path,
    *,
    rank: int,
    world_size: int,
    global_bsz: int | None = None,
    args=None,
    verify_integrity: bool = True,
) -> Dict[str, Any]:
    checkpoint_dir = Path(path).resolve()
    root = load_distributed_checkpoint_root(
        checkpoint_dir,
        verify_integrity=verify_integrity,
    )

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
    if int(rank) < 0 or int(rank) >= saved_world_size:
        raise ValueError(
            f"Distributed checkpoint rank out of range: rank={rank}, "
            f"world_size={saved_world_size}"
        )
    rank_dir = _resolve_checkpoint_member(checkpoint_dir, rank_manifests[int(rank)])
    manifest = load_pure_ssd_checkpoint_manifest(rank_dir)

    owner_file = _resolve_checkpoint_member(checkpoint_dir, root["block_owner"])
    bounds_file = _resolve_checkpoint_member(checkpoint_dir, root["block_bounds"])
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
    if checkpoint_version >= 3:
        manifest["_tide_optimizer_provenance"] = dict(
            root["optimizer_provenance"]
        )
        manifest["_tide_global_optimizer_step"] = int(
            root["global_optimizer_step"]
        )
        manifest["_tide_optimizer_state_mode"] = str(
            root["optimizer_state_mode"]
        )
        manifest["_tide_planner_state"] = planner_state_from_distributed_manifest(root)
    if args is not None:
        validate_distributed_checkpoint_resume(
            args,
            root,
            world_size=world_size,
            global_bsz=global_bsz,
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
    planner_state=None,
    global_optimizer_step: int | None = None,
    log_file=None,
) -> Dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    setup_result = {
        "ok": True,
        "rank": int(context.rank),
    }
    if context.is_rank0:
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except BaseException as error:
            setup_result = {
                "ok": False,
                "rank": int(context.rank),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
    setup_results = context.all_gather_object(setup_result)
    setup_errors = [
        value for value in setup_results if not bool(value.get("ok"))
    ]
    if setup_errors:
        details = "; ".join(
            f"rank {value['rank']}: {value['error_type']}: "
            f"{value['error_message']}"
            for value in setup_errors
        )
        raise RuntimeError(
            f"Distributed checkpoint directory setup failed: {details}"
        )
    context.barrier()

    rank_dir = checkpoint_dir / f"rank_{context.rank}"
    try:
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
            "ok": True,
            "rank": int(context.rank),
            "path": f"rank_{context.rank}",
            "manifest_sha256": _sha256_file(rank_dir / CHECKPOINT_MANIFEST),
            "patch_files": int(rank_manifest.get("patch_files", 0)),
            "patch_bytes": int(rank_manifest.get("patch_bytes", 0)),
        }
    except Exception as error:
        rank_result = {
            "ok": False,
            "rank": int(context.rank),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    rank_results = context.all_gather_object(rank_result)
    rank_errors = [value for value in rank_results if not bool(value.get("ok"))]
    if rank_errors:
        details = "; ".join(
            f"rank {value['rank']}: {value['error_type']}: {value['error_message']}"
            for value in rank_errors
        )
        raise RuntimeError(f"Distributed checkpoint rank write failed: {details}")
    context.barrier()

    root_manifest = None
    root_error = None
    if context.is_rank0:
        try:
            if planner_state is None:
                planner_state = getattr(args, "_tide_distributed_planner_state", None)
            normalized_planner_state = _normalize_planner_state(planner_state)
            if global_optimizer_step is None:
                global_optimizer_step = getattr(
                    gaussians,
                    "_tide_global_optimizer_step",
                    None,
                )
            if global_optimizer_step is None:
                global_optimizer_step = getattr(
                    args,
                    "_tide_global_optimizer_step",
                    None,
                )
            if global_optimizer_step is None:
                global_bsz = int(args.bsz)
                if global_bsz <= 0:
                    raise ValueError(
                        "Distributed checkpoint global batch size must be positive"
                    )
                global_optimizer_step = max(
                    0,
                    (int(next_iteration) - 1) // global_bsz,
                )
            global_optimizer_step = int(global_optimizer_step)
            if global_optimizer_step < 0:
                raise ValueError("global_optimizer_step must be non-negative")

            owner_file = checkpoint_dir / "block_owner.npy"
            bounds_file = checkpoint_dir / "block_bounds.npy"
            _atomic_numpy_save(np.asarray(block_owner, dtype=np.int32), owner_file)
            _atomic_numpy_save(
                np.asarray(storage_adapter.block_bounds, dtype=np.float32), bounds_file
            )
            ordered = sorted(rank_results, key=lambda value: int(value["rank"]))
            if [int(value["rank"]) for value in ordered] != list(
                range(int(context.world_size))
            ):
                raise RuntimeError(
                    "Distributed checkpoint rank results are incomplete or duplicated"
                )
            root_manifest = {
                "checkpoint_version": DISTRIBUTED_CHECKPOINT_VERSION,
                "checkpoint_type": "pure_ssd_distributed_incremental",
                "iteration": int(iteration),
                "next_iteration": int(next_iteration),
                "world_size": int(context.world_size),
                "global_bsz": int(args.bsz),
                "global_capacity_blocks": int(
                    args.paper_resident_capacity_blocks
                ),
                "capacity_semantics": "global",
                "owner_policy": str(
                    getattr(args, "_tide_owner_policy", "stable_round_robin")
                ),
                "block_version_semantics": "monotonic_gpu_cpu_ssd",
                "camera_assignment": str(args.tide_camera_assignment),
                "camera_microbatch": int(args.tide_camera_microbatch),
                "block_owner": owner_file.name,
                "block_owner_sha256": _sha256_file(owner_file),
                "block_bounds": bounds_file.name,
                "rank_manifests": [value["path"] for value in ordered],
                "rank_manifest_sha256": [
                    value["manifest_sha256"] for value in ordered
                ],
                "rank_patch_files": [value["patch_files"] for value in ordered],
                "rank_patch_bytes": [value["patch_bytes"] for value in ordered],
                "optimizer_provenance": build_optimizer_provenance(args),
                "global_optimizer_step": global_optimizer_step,
                "optimizer_state_mode": "cold_start_per_rank",
                "optimizer_ema_saved": False,
                "optimizer_curvature_saved": False,
                "planner_state": normalized_planner_state,
                "total_points": int(storage_adapter.num_points),
                "num_blocks": int(storage_adapter.num_blocks),
                "block_size": int(storage_adapter.block_size),
                "param_dim": 59,
            }
            _atomic_json_dump(
                root_manifest,
                checkpoint_dir / DISTRIBUTED_CHECKPOINT_MANIFEST,
            )
        except Exception as error:
            root_error = {
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
    root_result = context.broadcast_object(
        {"manifest": root_manifest, "error": root_error}
        if context.is_rank0
        else None
    )
    if root_result["error"] is not None:
        error = root_result["error"]
        raise RuntimeError(
            "Distributed checkpoint root manifest write failed: "
            f"{error['error_type']}: {error['error_message']}"
        )
    root_manifest = root_result["manifest"]
    context.barrier()
    return root_manifest
