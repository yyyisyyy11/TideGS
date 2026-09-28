"""CPU tests for tools/merge_distributed_checkpoint.py on synthetic distributed checkpoints."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import merge_distributed_checkpoint as mdc

BLOCK = 4
DIM = 59
BYTES = BLOCK * DIM * 4


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _make(root: Path, *, num_blocks=6, total_points=22, world=2, conflict_block=None, same_ids=False,
          same_names=False):
    """Two-rank checkpoint in the layout written by storage/distributed_checkpoint.py."""
    ckpt = root / "ckpt"
    ckpt.mkdir(parents=True)
    base = root / "base_file.bin"
    base.write_bytes(np.zeros((total_points, DIM), np.float32).tobytes())
    owner = np.array([b % world for b in range(num_blocks)], dtype=np.int32)
    np.save(ckpt / "block_owner.npy", owner)
    bounds = np.arange(num_blocks * 6, dtype=np.float32).reshape(num_blocks, 6)
    np.save(ckpt / "block_bounds.npy", bounds)

    def rows(b):
        return max(0, min(BLOCK, total_points - b * BLOCK))

    rank_hashes = []
    for r in range(world):
        rd = ckpt / f"rank_{r}"
        pd = rd / "ssd_delta" / "patches"
        pd.mkdir(parents=True)
        fid = 7 if same_ids else 7 + 10 * r
        name = "patch_000007_1_compact.bin" if same_names else f"patch_{fid:06d}_{r}_compact.bin"
        patch = pd / name
        owned = [b for b in range(num_blocks) if owner[b] == r]
        payload, index, off = b"", {}, 0
        for b in range(num_blocks):
            if owner[b] == r:
                data = np.full((rows(b), DIM), 100 * r + b, np.float32).tobytes()
                index[str(b)] = {"file_id": fid, "offset": off, "size": len(data), "version": 5}
                payload += data
                off += len(data)
            else:
                index[str(b)] = {"file_id": 0, "offset": b * BYTES, "size": rows(b) * DIM * 4, "version": 0}
        if conflict_block is not None and r != owner[conflict_block]:
            index[str(conflict_block)] = {"file_id": fid, "offset": 0, "size": rows(conflict_block) * DIM * 4,
                                          "version": 3}
        patch.write_bytes(payload)
        np.save(rd / "ssd_delta" / "block_bounds.npy", bounds)
        (rd / "training_state.pth").write_bytes(b"state")
        idx = {"block_size": BLOCK, "bytes_per_block": BYTES, "dtype": "float32", "num_blocks": num_blocks,
               "point_dim": DIM, "storage_type": "log_structured", "version": 1, "next_patch_id": fid + 1,
               "files": {"0": {"path": str(base), "role": "base", "size": base.stat().st_size},
                         str(fid): {"path": str(patch), "role": "patch", "size": len(payload)}},
               "index": index}
        (rd / "ssd_delta" / "storage_index.json").write_text(json.dumps(idx))
        man = {"checkpoint_type": "pure_ssd_incremental", "checkpoint_iter": 100, "iteration": 100,
               "next_iteration": 101, "total_points": total_points, "num_blocks": num_blocks, "block_size": BLOCK,
               "param_dim": DIM, "base_file": str(base), "active_sh_degree": 3, "_prebuilt_base_reuse": True,
               "storage_index": str(rd / "ssd_delta" / "storage_index.json"),
               "block_bounds": str(rd / "ssd_delta" / "block_bounds.npy"),
               "training_state": str(rd / "training_state.pth"), "patches_dir": str(pd),
               "patch_files": 1, "patch_bytes": len(payload), "checkpoint_dir": str(rd)}
        (rd / "pure_ssd_checkpoint.json").write_text(json.dumps(man))
        rank_hashes.append(_sha(rd / "pure_ssd_checkpoint.json"))
    root_manifest = {"world_size": world, "rank_manifests": [f"rank_{r}" for r in range(world)],
                     "block_owner": "block_owner.npy", "block_bounds": "block_bounds.npy",
                     "block_owner_sha256": _sha(ckpt / "block_owner.npy"), "rank_manifest_sha256": rank_hashes,
                     "num_blocks": num_blocks, "block_size": BLOCK, "total_points": total_points, "param_dim": DIM,
                     "iteration": 100, "next_iteration": 101, "global_bsz": 32, "owner_policy": "stable_round_robin",
                     "optimizer_provenance": {"paper_resident_selection_policy": "topc_balanced_active_first"}}
    (ckpt / "pure_ssd_distributed_checkpoint.json").write_text(json.dumps(root_manifest))
    return ckpt, owner, base


def _snapshot(ckpt: Path):
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns, _sha(p)) for p in sorted(ckpt.rglob("*")) if p.is_file()}


def _read_block(ckpt_manifest: Path, b: int):
    m = json.loads(ckpt_manifest.read_text())
    idx = json.loads(Path(m["storage_index"]).read_text())
    e = idx["index"][str(b)]
    with open(idx["files"][str(e["file_id"])]["path"], "rb") as f:
        f.seek(e["offset"])
        return np.frombuffer(f.read(e["size"]), np.float32)


class MergeDistributedCheckpointTest(unittest.TestCase):
    def test_merge_takes_owner_entries_and_keeps_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, owner, base = _make(Path(d))
            before = _snapshot(ckpt)
            out = Path(d) / "merged"
            self.assertEqual(mdc.main(["--input", str(ckpt), "--output", str(out)]), 0)
            self.assertEqual(before, _snapshot(ckpt))  # source untouched
            report = json.loads((out / "merge_report.json").read_text())
            self.assertEqual(report["blocks_total"], 6)
            self.assertEqual(report["blocks_per_rank"], {"0": 3, "1": 3})
            m = json.loads((out / "pure_ssd_checkpoint.json").read_text())
            self.assertEqual(m["patch_files"], 2)
            self.assertEqual(m["next_iteration"], 101)
            self.assertNotIn("checkpoint_dir", m)
            for b in range(6):
                vals = _read_block(out / "pure_ssd_checkpoint.json", b)
                self.assertTrue(np.all(vals == 100 * owner[b] + b), b)
            # patch files are hard links of the originals (no copy)
            for p in (out / "ssd_delta" / "patches").iterdir():
                self.assertEqual(p.stat().st_nlink, 2)

    def test_ambiguous_block_aborts_without_output(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, _, _ = _make(Path(d), conflict_block=2)
            out = Path(d) / "merged"
            self.assertEqual(mdc.main(["--input", str(ckpt), "--output", str(out)]), 2)
            self.assertFalse(out.exists())

    def test_colliding_file_ids_and_names_are_remapped(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, owner, _ = _make(Path(d), same_ids=True, same_names=True)
            out = Path(d) / "merged"
            self.assertEqual(mdc.main(["--input", str(ckpt), "--output", str(out)]), 0)
            names = sorted(p.name for p in (out / "ssd_delta" / "patches").iterdir())
            self.assertEqual(len(names), 2)
            self.assertTrue(all(n.startswith("patch_") and n.endswith(".bin") for n in names))
            for b in range(6):
                self.assertTrue(np.all(_read_block(out / "pure_ssd_checkpoint.json", b) == 100 * owner[b] + b))

    def test_refuses_existing_output_and_tampered_owner(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, _, _ = _make(Path(d))
            out = Path(d) / "merged"
            out.mkdir()
            self.assertEqual(mdc.main(["--input", str(ckpt), "--output", str(out)]), 2)
            np.save(ckpt / "block_owner.npy", np.zeros(6, np.int32))
            self.assertEqual(mdc.main(["--input", str(ckpt), "--output", str(Path(d) / "m2")]), 2)

    def test_merged_checkpoint_passes_upstream_completeness_rule(self):
        from tools import eval_protocol
        with tempfile.TemporaryDirectory() as d:
            ckpt, _, _ = _make(Path(d))
            out = Path(d) / "run" / "checkpoints" / "100"
            out.parent.mkdir(parents=True)
            self.assertEqual(mdc.main(["--input", str(ckpt), "--output", str(out)]), 0)
            st = eval_protocol.checkpoint_status(out)
            self.assertTrue(st["complete"], st["reasons"])
            self.assertEqual(st["iteration"], 100)


if __name__ == "__main__":
    unittest.main()
