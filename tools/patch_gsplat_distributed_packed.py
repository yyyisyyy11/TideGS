#!/usr/bin/env python3
"""Patch the gsplat 1.5.3 packed-distributed camera-index bug."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
from pathlib import Path


ORIGINAL = "            C = C_world[world_rank]\n\n        else:\n"
BUGGY_IMAGE_IDS = "image_ids = batch_ids * C + camera_ids"
PATCHED = (
    "            C = C_world[world_rank]\n"
    "            image_ids = camera_ids\n"
    "\n"
    "        else:\n"
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
    patched_source, changed = patch_rendering_source(source)
    if not changed:
        print(f"gsplat packed-distributed fix already present: {rendering_path}")
        return

    backup_path = rendering_path.with_name("rendering.py.tidegs-original")
    if not backup_path.exists():
        shutil.copy2(rendering_path, backup_path)

    temporary_path = rendering_path.with_name("rendering.py.tidegs-tmp")
    temporary_path.write_text(patched_source, encoding="utf-8")
    compile(patched_source, str(rendering_path), "exec")
    os.replace(temporary_path, rendering_path)
    print(f"Patched gsplat packed-distributed camera IDs: {rendering_path}")
    print(f"Original source backup: {backup_path}")


if __name__ == "__main__":
    main()
