#!/usr/bin/env python3
"""Explicitly predecode only the Test split into the shared raw image cache."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.pure_ssd_image_io import (  # noqa: E402
    load_test_scene_metadata,
    raw_cache_is_usable,
    raw_cache_path,
)


def _load_run_args(run_dir: Path) -> Namespace:
    args_path = run_dir / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError(f"training args.json not found: {args_path}")
    with open(args_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {args_path}")
    return Namespace(**payload)


def _attach_latest_checkpoint_manifest(args: Namespace, run_dir: Path) -> None:
    from storage.pure_ssd_checkpoint import load_pure_ssd_checkpoint_manifest

    checkpoints_dir = run_dir / "checkpoints"
    candidates = sorted(
        (
            path
            for path in checkpoints_dir.iterdir()
            if path.is_dir() and path.name.isdigit() and (path / "pure_ssd_checkpoint.json").is_file()
        ),
        key=lambda path: int(path.name),
    ) if checkpoints_dir.is_dir() else []
    args._pure_ssd_resume_manifest = (
        load_pure_ssd_checkpoint_manifest(candidates[-1]) if candidates else None
    )
    args._pure_ssd_prebuilt_manifest = None


def _decode_one(task: Tuple[str, str, int, int]) -> Dict[str, str]:
    from PIL import Image

    source_path_raw, target_path_raw, image_height, image_width = task
    source_path = Path(source_path_raw)
    target_path = Path(target_path_raw)
    if raw_cache_is_usable(target_path, image_height, image_width):
        return {"status": "existing", "target": str(target_path)}

    temp_path = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
    try:
        with Image.open(source_path) as image:
            image = image.convert("RGB").crop((0, 0, int(image_width), int(image_height)))
            raw_bytes = image.tobytes()
        expected_bytes = int(image_height) * int(image_width) * 3
        if len(raw_bytes) != expected_bytes:
            raise RuntimeError(
                f"decoded byte count mismatch: got {len(raw_bytes)}, expected {expected_bytes}"
            )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as handle:
            handle.write(raw_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target_path)
        return {"status": "decoded", "target": str(target_path)}
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        return {
            "status": "failed",
            "target": str(target_path),
            "source": str(source_path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def predecode_test_images(cli_args: argparse.Namespace) -> Dict[str, object]:
    import utils.general_utils as utils

    run_dir = Path(cli_args.run_dir).resolve()
    output_path = Path(cli_args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = _load_run_args(run_dir)
    args.eval = True
    args.debug = False
    args.debug_max_test_cameras = -1
    args.num_test_cameras = -1
    args.model_path = str(output_path.parent)
    args.log_folder = args.model_path
    _attach_latest_checkpoint_manifest(args, run_dir)

    log_path = output_path.with_suffix(".log")
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
        utils.set_args(args)
        utils.set_log_file(log_file)
        try:
            test_camera_infos, _, image_height, image_width = load_test_scene_metadata(args)
        finally:
            utils.set_log_file(None)

    expected_count = int(cli_args.expected_count)
    if expected_count >= 0 and len(test_camera_infos) != expected_count:
        raise RuntimeError(
            f"Test camera count mismatch: got {len(test_camera_infos)}, expected {expected_count}"
        )

    tasks = []
    for camera_info in test_camera_infos:
        tasks.append(
            (
                str(Path(camera_info.image_path)),
                str(raw_cache_path(args, camera_info)),
                int(image_height),
                int(image_width),
            )
        )

    existing_count = 0
    decoded_count = 0
    failures = []
    workers = max(1, int(cli_args.workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for completed, result in enumerate(executor.map(_decode_one, tasks), start=1):
            status = result["status"]
            if status == "existing":
                existing_count += 1
            elif status == "decoded":
                decoded_count += 1
            else:
                failures.append(result)
            if completed == 1 or completed % 100 == 0 or completed == len(tasks):
                print(
                    f"[TEST PREDECODE] {completed}/{len(tasks)} "
                    f"existing={existing_count} decoded={decoded_count} failed={len(failures)}"
                )

    valid_raw_count = sum(
        raw_cache_is_usable(raw_cache_path(args, camera_info), image_height, image_width)
        for camera_info in test_camera_infos
    )
    summary = {
        "run_dir": str(run_dir),
        "decode_dataset_path": str(Path(args.decode_dataset_path).resolve()),
        "raw_cache_dir": str((Path(args.decode_dataset_path) / "dataset_raw").resolve()),
        "test_camera_count": len(test_camera_infos),
        "image_height": int(image_height),
        "image_width": int(image_width),
        "workers": workers,
        "existing_valid_raw_count": existing_count,
        "decoded_raw_count": decoded_count,
        "valid_raw_count": int(valid_raw_count),
        "failed_count": len(failures),
        "failures": failures[:50],
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    if failures or valid_raw_count != len(test_camera_infos):
        raise RuntimeError(
            f"Test predecode incomplete: valid={valid_raw_count}/{len(test_camera_infos)} "
            f"failed={len(failures)}; see {output_path}"
        )
    print(
        f"[TEST PREDECODE] complete: valid={valid_raw_count} "
        f"existing={existing_count} decoded={decoded_count} summary={output_path}"
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predecode only the complete Test camera split.")
    parser.add_argument("--run-dir", required=True, help="Training run containing args.json.")
    parser.add_argument("--workers", type=int, default=16, help="Parallel image decoder processes.")
    parser.add_argument("--output", required=True, help="Output JSON summary path.")
    parser.add_argument(
        "--expected-count",
        type=int,
        default=2584,
        help="Expected Test split size; -1 disables the count check.",
    )
    return parser


def main() -> None:
    predecode_test_images(build_argparser().parse_args())


if __name__ == "__main__":
    main()
