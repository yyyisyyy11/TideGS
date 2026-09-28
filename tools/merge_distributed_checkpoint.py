#!/usr/bin/env python3
"""Adapt a gaussian_sharded (multi-GPU) Pure SSD checkpoint to the single-process layout.

A distributed checkpoint written by ``storage/distributed_checkpoint.py`` is

    <ckpt>/pure_ssd_distributed_checkpoint.json   root manifest (world_size, block_owner, rank_manifests, ...)
    <ckpt>/block_owner.npy                         owner rank of every block (int array, shape [num_blocks])
    <ckpt>/block_bounds.npy                        global block bounds
    <ckpt>/rank_<r>/pure_ssd_checkpoint.json       ordinary single-process incremental manifest of rank r
    <ckpt>/rank_<r>/ssd_delta/storage_index.json   rank r's index over ALL blocks
    <ckpt>/rank_<r>/ssd_delta/patches/*.bin        rank r's patch files (hard links)

Every rank's index lists every block, but a rank only trains and writes back the blocks it
owns; blocks owned by another rank stay at the immutable base (file_id 0, version 0). The
authoritative version of block b is therefore rank ``block_owner[b]``'s index entry. This
tool builds one single-process checkpoint whose index takes, for every block, exactly the
owner's entry (same bytes: file, offset, size, version), and links the owners' patch files
into the new checkpoint. No tensor data is read, modified or copied.

Any block that a non-owner rank holds in a patch, or with a non-zero version, makes the
ownership ambiguous and aborts the merge.

Usage:
    python tools/merge_distributed_checkpoint.py --input <dist_ckpt> --output <merged_ckpt>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

DIST_MANIFEST = "pure_ssd_distributed_checkpoint.json"
CKPT_MANIFEST = "pure_ssd_checkpoint.json"
TOOL_VERSION = 1


class MergeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(value: str, base: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _rows_in_block(block_id: int, total_points: int, block_size: int) -> int:
    start = block_id * block_size
    return max(0, min(block_size, total_points - start))


def load_distributed(ckpt: Path) -> dict:
    root_path = ckpt / DIST_MANIFEST
    if not root_path.is_file():
        raise MergeError(f"not a distributed checkpoint (missing {DIST_MANIFEST}): {ckpt}")
    root = _load_json(root_path)
    world = int(root["world_size"])
    rank_dirs = [_resolve(v, ckpt) for v in root["rank_manifests"]]
    if len(rank_dirs) != world or world < 1:
        raise MergeError(f"rank_manifests has {len(rank_dirs)} entries for world_size={world}")

    owner_path = _resolve(root["block_owner"], ckpt)
    if root.get("block_owner_sha256") and _sha256(owner_path) != root["block_owner_sha256"]:
        raise MergeError("block_owner.npy sha256 does not match the root manifest")
    for r, d in enumerate(rank_dirs):
        expected = (root.get("rank_manifest_sha256") or [None] * world)[r]
        if expected and _sha256(d / CKPT_MANIFEST) != expected:
            raise MergeError(f"rank {r} manifest sha256 does not match the root manifest")

    owner = np.load(owner_path, allow_pickle=False)
    num_blocks = int(root["num_blocks"])
    if owner.shape != (num_blocks,) or not np.issubdtype(owner.dtype, np.integer):
        raise MergeError(f"block_owner has shape {owner.shape} dtype {owner.dtype}; expected ({num_blocks},) int")
    if owner.min() < 0 or owner.max() >= world:
        raise MergeError("block_owner contains ranks outside [0, world_size)")

    ranks = []
    for r, d in enumerate(rank_dirs):
        m = _load_json(d / CKPT_MANIFEST)
        index_path = _resolve(m["storage_index"], d)
        idx = _load_json(index_path)
        if int(idx["num_blocks"]) != num_blocks or len(idx["index"]) != num_blocks:
            raise MergeError(f"rank {r} index covers {len(idx['index'])} blocks, expected {num_blocks}")
        ranks.append({"dir": d, "manifest": m, "index": idx, "bounds": _resolve(m["block_bounds"], d)})
    return {"root": root, "owner": owner, "ranks": ranks, "ckpt": ckpt,
            "bounds": _resolve(root["block_bounds"], ckpt)}


def plan_merge(dist: dict) -> dict:
    root, owner, ranks = dist["root"], dist["owner"], dist["ranks"]
    num_blocks = int(root["num_blocks"])
    block_size = int(root["block_size"])
    total_points = int(root["total_points"])
    param_dim = int(root.get("param_dim", 59))

    # every rank must share the same base file (file id 0)
    base_paths = {str(Path(r["index"]["files"]["0"]["path"]).resolve()) for r in ranks}
    if len(base_paths) != 1:
        raise MergeError(f"ranks reference different base files: {sorted(base_paths)}")
    for key in ("block_size", "point_dim", "dtype", "bytes_per_block"):
        values = {str(r["index"].get(key)) for r in ranks}
        if len(values) != 1:
            raise MergeError(f"ranks disagree on index field {key}: {values}")

    conflicts: List[str] = []
    per_rank_owned = Counter()
    per_rank_owned_at_base = Counter()
    chosen: Dict[int, dict] = {}
    for b in range(num_blocks):
        key = str(b)
        o = int(owner[b])
        for r, rank in enumerate(ranks):
            entry = rank["index"]["index"][key]
            if r == o:
                continue
            if int(entry["file_id"]) != 0 or int(entry["version"]) != 0:
                conflicts.append(f"block {b}: non-owner rank {r} holds file_id={entry['file_id']} "
                                 f"version={entry['version']} (owner rank {o})")
        entry = dict(ranks[o]["index"]["index"][key])
        rows = _rows_in_block(b, total_points, block_size)
        if int(entry["size"]) != rows * param_dim * 4:
            conflicts.append(f"block {b}: owner entry size {entry['size']} != {rows * param_dim * 4}")
        per_rank_owned[o] += 1
        if int(entry["file_id"]) == 0:
            per_rank_owned_at_base[o] += 1
        chosen[b] = {"rank": o, **entry}
    if conflicts:
        raise MergeError(f"{len(conflicts)} ambiguous/conflicting blocks; first: {conflicts[:5]}")

    # patch files actually referenced by owner entries, with collision-free new ids and names
    used = sorted({(c["rank"], int(c["file_id"])) for c in chosen.values() if int(c["file_id"]) != 0})
    new_ids: Dict[tuple, int] = {}
    names: Dict[tuple, str] = {}
    taken_ids, taken_names = {0}, set()
    for rank_id, fid in used:
        info = ranks[rank_id]["index"]["files"][str(fid)]
        src = Path(info["path"])
        new_id = fid
        while new_id in taken_ids:
            new_id += 1_000_000
        name = src.name
        if name in taken_names:
            name = f"patch_r{rank_id}_{src.name[len('patch_'):]}" if src.name.startswith("patch_") else f"patch_r{rank_id}_{src.name}"
        if not name.startswith("patch_") or not name.endswith(".bin"):
            raise MergeError(f"unexpected patch file name {src.name}")
        taken_ids.add(new_id)
        taken_names.add(name)
        new_ids[(rank_id, fid)] = new_id
        names[(rank_id, fid)] = name
    return {"chosen": chosen, "new_ids": new_ids, "names": names,
            "per_rank_owned": dict(per_rank_owned), "per_rank_owned_at_base": dict(per_rank_owned_at_base),
            "base_path": base_paths.pop()}


def write_merged(dist: dict, plan: dict, out: Path, link_mode: str) -> dict:
    if out.exists():
        raise MergeError(f"output already exists, refusing to overwrite: {out}")
    root, ranks = dist["root"], dist["ranks"]
    rank0 = ranks[0]
    delta = out / "ssd_delta"
    patches = delta / "patches"
    patches.mkdir(parents=True)

    def link(src: Path, dst: Path) -> None:
        if link_mode == "hardlink":
            os.link(src, dst)
        else:
            os.symlink(src.resolve(), dst)

    # patch files
    files = {"0": dict(rank0["index"]["files"]["0"])}
    patch_bytes = 0
    for (rank_id, fid), new_id in sorted(plan["new_ids"].items(), key=lambda kv: kv[1]):
        info = ranks[rank_id]["index"]["files"][str(fid)]
        src = Path(info["path"])
        dst = patches / plan["names"][(rank_id, fid)]
        link(src, dst)
        size = dst.stat().st_size
        if "size" in info and int(info["size"]) != size:
            raise MergeError(f"patch {src} has size {size}, index says {info['size']}")
        files[str(new_id)] = {"path": str(dst.resolve() if link_mode == "hardlink" else dst.absolute()),
                              "role": "patch", "size": size, "linked": True, "copied": False,
                              "source_rank": rank_id, "source_file_id": fid, "source_path": str(src)}
        patch_bytes += size

    index = {}
    for b, c in plan["chosen"].items():
        fid = int(c["file_id"])
        index[str(b)] = {"file_id": 0 if fid == 0 else plan["new_ids"][(c["rank"], fid)],
                         "offset": int(c["offset"]), "size": int(c["size"]), "version": int(c["version"])}

    idx0 = rank0["index"]
    merged_index = {k: idx0[k] for k in ("block_size", "bytes_per_block", "dtype", "num_blocks", "point_dim",
                                         "storage_type", "version") if k in idx0}
    merged_index.update({
        "files": files, "index": index,
        "next_patch_id": max(int(k) for k in files) + 1,
        "patch_files": len(files) - 1, "patch_bytes": patch_bytes,
        "linked_patch_files": len(files) - 1, "linked_patch_bytes": patch_bytes,
        "copied_patch_files": 0, "copied_patch_bytes": 0, "patch_file_mode": link_mode,
    })
    _write_json(delta / "storage_index.json", merged_index)

    # bounds: the root copy is authoritative; it must equal the owner-selected rank bounds
    root_bounds = np.load(dist["bounds"])
    owner = dist["owner"]
    selected = np.load(ranks[0]["bounds"]).copy()
    for r in range(1, len(ranks)):
        rb = np.load(ranks[r]["bounds"])
        selected[owner == r] = rb[owner == r]
    if not np.array_equal(root_bounds, selected):
        raise MergeError("root block_bounds differ from the owner-selected rank bounds")
    link(dist["bounds"], delta / "block_bounds.npy")

    # training state (the evaluator only needs it to exist; rank 0's matches the root iteration)
    ts_src = _resolve(rank0["manifest"]["training_state"], rank0["dir"])
    link(ts_src, out / "training_state.pth")

    manifest = dict(rank0["manifest"])
    manifest.update({
        "delta_dir": str(delta.absolute()),
        "storage_index": str((delta / "storage_index.json").absolute()),
        "block_bounds": str((delta / "block_bounds.npy").absolute()),
        "patches_dir": str(patches.absolute()),
        "training_state": str((out / "training_state.pth").absolute()),
        "patch_files": len(files) - 1, "patch_bytes": patch_bytes,
        "patch_files_linked": len(files) - 1, "patch_bytes_linked": patch_bytes,
        "patch_files_copied": 0, "patch_bytes_copied": 0, "patch_file_mode": link_mode,
        "iteration": int(root["iteration"]), "checkpoint_iter": int(root["iteration"]),
        "next_iteration": int(root["next_iteration"]),
        "merged_from_distributed": {
            "tool": "tools/merge_distributed_checkpoint.py", "tool_version": TOOL_VERSION,
            "source_checkpoint": str(dist["ckpt"].resolve()), "world_size": int(root["world_size"]),
            "owner_policy": root.get("owner_policy"), "block_owner_sha256": root.get("block_owner_sha256"),
            "blocks_per_rank": {str(k): v for k, v in sorted(plan["per_rank_owned"].items())},
            "owned_blocks_at_base_per_rank": {str(k): v for k, v in sorted(plan["per_rank_owned_at_base"].items())},
            "global_bsz": root.get("global_bsz"), "global_capacity_blocks": root.get("global_capacity_blocks"),
            "resident_selection_policy": (root.get("optimizer_provenance") or {}).get("paper_resident_selection_policy"),
        },
    })
    manifest.pop("checkpoint_dir", None)
    _write_json(out / CKPT_MANIFEST, manifest)  # written last: marks the merged checkpoint complete
    return manifest


def verify_merged(dist: dict, out: Path) -> dict:
    """Re-read the merged checkpoint and check every block against its owner's original entry."""
    owner, ranks = dist["owner"], dist["ranks"]
    m = _load_json(out / CKPT_MANIFEST)
    idx = _load_json(Path(m["storage_index"]))
    files = {int(k): v for k, v in idx["files"].items()}
    inode_of = {k: os.stat(v["path"]).st_ino for k, v in files.items()}
    src_inode = {}
    for r, rank in enumerate(ranks):
        for k, v in rank["index"]["files"].items():
            src_inode[(r, int(k))] = os.stat(v["path"]).st_ino
    counts, mismatches = Counter(), []
    for b in range(len(owner)):
        o = int(owner[b])
        want = ranks[o]["index"]["index"][str(b)]
        got = idx["index"][str(b)]
        same_file = inode_of[int(got["file_id"])] == src_inode[(o, int(want["file_id"]))]
        if not (same_file and int(got["offset"]) == int(want["offset"]) and int(got["size"]) == int(want["size"])
                and int(got["version"]) == int(want["version"])):
            mismatches.append(b)
        size = os.stat(files[int(got["file_id"])]["path"]).st_size
        if int(got["offset"]) + int(got["size"]) > size:
            mismatches.append(b)
        counts[o] += 1
    if len(idx["index"]) != len(owner):
        raise MergeError(f"merged index has {len(idx['index'])} blocks, expected {len(owner)}")
    if mismatches:
        raise MergeError(f"{len(mismatches)} merged blocks do not match their owner entry; first {mismatches[:5]}")
    return {"blocks_total": len(owner), "blocks_per_rank": {str(k): v for k, v in sorted(counts.items())},
            "missing_blocks": 0, "conflicting_blocks": 0, "mismatched_blocks": 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="distributed checkpoint dir (contains pure_ssd_distributed_checkpoint.json)")
    ap.add_argument("--output", required=True, help="new single-process checkpoint dir (must not exist)")
    ap.add_argument("--link-mode", choices=["hardlink", "symlink"], default="hardlink")
    ns = ap.parse_args(argv)
    src, out = Path(ns.input).resolve(), Path(ns.output).absolute()
    try:
        dist = load_distributed(src)
        plan = plan_merge(dist)
        write_merged(dist, plan, out, ns.link_mode)
        report = verify_merged(dist, out)
    except MergeError as exc:
        print(f"[MERGE] FAILED: {exc}", file=sys.stderr)
        return 2
    report.update({"source": str(src), "output": str(out), "link_mode": ns.link_mode,
                   "owned_blocks_at_base_per_rank": {str(k): v for k, v in sorted(plan["per_rank_owned_at_base"].items())},
                   "patch_files": {f"{k[0]}:{k[1]}": plan["names"][k] for k in sorted(plan["names"])},
                   "iteration": int(dist["root"]["iteration"])})
    _write_json(out / "merge_report.json", report)
    print(json.dumps({k: report[k] for k in ("iteration", "blocks_total", "blocks_per_rank", "missing_blocks",
                                             "conflicting_blocks", "mismatched_blocks", "owned_blocks_at_base_per_rank")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
