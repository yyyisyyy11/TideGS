#!/usr/bin/env python3
"""Patch gsplat 1.5.3 for TideGS distributed rendering and profiling."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
from pathlib import Path


ORIGINAL = "            C = C_world[world_rank]\n\n        else:\n"
BUGGY_IMAGE_IDS = "image_ids = batch_ids * C + camera_ids"
PROJECTION_TIMING_MARKER = "_tide_projection_events"
OWNER_SURVIVOR_MARKER = "_tide_owner_projection_survivor_mask"
PROJECTION_START_ANCHOR = "    if with_ut:\n"
PROJECTION_END_ANCHOR = (
    "    if packed:\n"
    "        # The results are packed into shape [nnz, ...]. All elements are valid.\n"
)
PATCHED = (
    "            C = C_world[world_rank]\n"
    "            image_ids = camera_ids\n"
    "\n"
    "        else:\n"
)
PROJECTION_START_PATCHED = (
    "    _tide_projection_start = None\n"
    "    _tide_projection_end = None\n"
    '    if getattr(torch, "_tide_gsplat_detailed_metrics", False):\n'
    "        _tide_projection_start = torch.cuda.Event(enable_timing=True)\n"
    "        _tide_projection_end = torch.cuda.Event(enable_timing=True)\n"
    "        _tide_projection_start.record()\n"
    "\n"
    "    if with_ut:\n"
)
PROJECTION_END_PATCHED = (
    "    if _tide_projection_start is not None:\n"
    "        _tide_projection_end.record()\n"
    f'        meta["{PROJECTION_TIMING_MARKER}"] = (\n'
    "            _tide_projection_start,\n"
    "            _tide_projection_end,\n"
    "        )\n"
    "\n"
    f"{PROJECTION_END_ANCHOR}"
)
OWNER_SURVIVOR_ANCHOR = (
    "    meta.update(\n"
    "        {\n"
    "            # global batch and camera ids\n"
)
OWNER_SURVIVOR_PATCHED = (
    '    if distributed and packed and getattr(torch, "_tide_gsplat_grad_zero_metrics", False):\n'
    "        _tide_owner_projection_survivor_mask = torch.zeros(\n"
    "            (N,), dtype=torch.bool, device=device\n"
    "        )\n"
    "        _tide_owner_projection_survivor_mask[\n"
    "            gaussian_ids[(radii > 0).all(dim=-1)]\n"
    "        ] = True\n"
    f'        meta["{OWNER_SURVIVOR_MARKER}"] = (\n'
    "            _tide_owner_projection_survivor_mask\n"
    "        )\n"
    "\n"
    f"{OWNER_SURVIVOR_ANCHOR}"
)


def patch_rendering_source(source: str) -> tuple[str, bool]:
    if PATCHED in source:
        return source, False
    if BUGGY_IMAGE_IDS not in source:
        raise RuntimeError(
            "This is not the affected gsplat batched rendering layout. "
            "Refusing to modify an unknown source layout."
        )
    occurrences = source.count(ORIGINAL)
    if occurrences != 1:
        raise RuntimeError(
            "Expected exactly one gsplat 1.5.3 distributed packed insertion point; "
            f"found {occurrences}. Refusing to modify an unknown source layout."
        )
    return source.replace(ORIGINAL, PATCHED, 1), True


def patch_projection_timing_source(source: str) -> tuple[str, bool]:
    if PROJECTION_TIMING_MARKER in source:
        return source, False
    start_occurrences = source.count(PROJECTION_START_ANCHOR)
    end_occurrences = source.count(PROJECTION_END_ANCHOR)
    if start_occurrences != 1 or end_occurrences != 1:
        raise RuntimeError(
            "Expected exactly one gsplat 1.5.3 projection timing region; "
            f"found start={start_occurrences}, end={end_occurrences}. "
            "Refusing to modify an unknown source layout."
        )
    patched = source.replace(
        PROJECTION_START_ANCHOR,
        PROJECTION_START_PATCHED,
        1,
    )
    patched = patched.replace(
        PROJECTION_END_ANCHOR,
        PROJECTION_END_PATCHED,
        1,
    )
    return patched, True


def patch_owner_survivor_source(source: str) -> tuple[str, bool]:
    if OWNER_SURVIVOR_MARKER in source:
        return source, False
    occurrences = source.count(OWNER_SURVIVOR_ANCHOR)
    if occurrences != 1:
        raise RuntimeError(
            "Expected exactly one gsplat 1.5.3 owner-survivor insertion point; "
            f"found {occurrences}. Refusing to modify an unknown source layout."
        )
    return source.replace(
        OWNER_SURVIVOR_ANCHOR,
        OWNER_SURVIVOR_PATCHED,
        1,
    ), True


def find_rendering_path() -> Path:
    spec = importlib.util.find_spec("gsplat")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("gsplat is not installed in the active Python environment")
    package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    rendering_path = package_dir / "rendering.py"
    if not rendering_path.is_file():
        raise RuntimeError(f"gsplat rendering.py was not found at {rendering_path}")
    return rendering_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        help="Optional explicit path to gsplat/rendering.py",
    )
    args = parser.parse_args()

    rendering_path = (args.path.resolve() if args.path else find_rendering_path())
    source = rendering_path.read_text(encoding="utf-8")
    patched_source, camera_changed = patch_rendering_source(source)
    patched_source, timing_changed = patch_projection_timing_source(patched_source)
    patched_source, survivor_changed = patch_owner_survivor_source(patched_source)
    if not camera_changed and not timing_changed and not survivor_changed:
        print(f"TideGS gsplat fixes already present: {rendering_path}")
        return

    backup_path = rendering_path.with_name("rendering.py.tidegs-original")
    if not backup_path.exists():
        shutil.copy2(rendering_path, backup_path)

    temporary_path = rendering_path.with_name("rendering.py.tidegs-tmp")
    temporary_path.write_text(patched_source, encoding="utf-8")
    compile(patched_source, str(rendering_path), "exec")
    os.replace(temporary_path, rendering_path)
    if camera_changed:
        print(f"Patched gsplat packed-distributed camera IDs: {rendering_path}")
    if timing_changed:
        print(f"Patched gsplat projection timing events: {rendering_path}")
    if survivor_changed:
        print(f"Patched gsplat owner projection survivor mask: {rendering_path}")
    print(f"Original source backup: {backup_path}")


if __name__ == "__main__":
    main()
