#!/usr/bin/env python3
"""Read-only quality evaluation for a Pure SSD checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.pure_ssd_quality_utils import (  # noqa: E402
    checkpoint_manifest_fingerprint,
    compute_next_resident_transition,
    compute_psnr,
    fingerprint_camera_schedule,
    select_initial_resident_blocks,
    select_preview_indices,
    summarize_metric_rows,
    summarize_values,
    validate_resident_configuration,
    write_tsv,
)
from tools.pure_ssd_image_io import (  # noqa: E402
    load_evaluation_camera,
    load_test_scene_metadata,
    normalize_gt_image,
)


EVAL_BATCH_SIZE = 16
RESIDENT_POLICY = "topc_balanced"
RESIDENT_LAMBDA = 0.3
RESIDENT_RECENCY_DECAY = 0.95
BALANCED_SEED_FRACTION = 0.25
RESIDENT_CAPACITY = 2048


def shared_schedule_cache_dir(output_dir: Path) -> Path:
    """Place the A/B schedule cache beside both evaluator output directories."""
    return output_dir.resolve().parent / "camera_schedule_cache"


PER_CAMERA_FIELDS = (
    "camera_index",
    "image_name",
    "schedule_position",
    "batch_index",
    "camera_visible_blocks",
    "camera_resident_blocks",
    "batch_visible_blocks",
    "resident_blocks",
    "resident_coverage",
    "rendered_gaussians",
    "psnr",
    "ssim",
    "lpips_alex",
    "render_ms",
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


def _require_training_configuration(args: Namespace) -> None:
    expected = {
        "bsz": EVAL_BATCH_SIZE,
        "paper_resident_selection_policy": RESIDENT_POLICY,
        "paper_resident_capacity_blocks": RESIDENT_CAPACITY,
    }
    for name, value in expected.items():
        actual = getattr(args, name, None)
        if actual != value:
            raise ValueError(f"training configuration mismatch: {name}={actual!r}, expected {value!r}")

    float_expected = {
        "paper_resident_lambda": RESIDENT_LAMBDA,
        "paper_resident_recency_decay": RESIDENT_RECENCY_DECAY,
        "paper_balanced_seed_fraction": BALANCED_SEED_FRACTION,
    }
    for name, value in float_expected.items():
        actual = float(getattr(args, name, float("nan")))
        if not math.isclose(actual, value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"training configuration mismatch: {name}={actual!r}, expected {value!r}")


def _configure_eval_args(
    args: Namespace,
    output_dir: Path,
    checkpoint_dir: Path,
    manifest: Dict[str, object],
) -> Namespace:
    _require_training_configuration(args)
    args.eval = True
    args.model_path = str(output_dir.resolve())
    args.log_folder = args.model_path
    args.start_checkpoint = str(checkpoint_dir.resolve())
    args.auto_start_checkpoint = False
    args.num_train_cameras = -1
    args.num_test_cameras = -1
    args.debug_max_test_cameras = -1
    args.gpu = 0
    args.quiet = False
    args.debug = False
    args.debug_frustum = False
    args.paper_debug_logging = False
    args.tide_debug_logging = False
    args.pure_ssd_disable_schedule_cache = False
    args.pure_ssd_schedule_cache_dir = str(shared_schedule_cache_dir(output_dir))
    args.ssd_cache_dir = str(output_dir / "ssd_eval_cache")
    args.ssd_schedule_ordering = "trajectory"
    args._pure_ssd_resume_manifest = manifest
    args._pure_ssd_prebuilt_manifest = None
    return args


def _collect_camera_blocks(storage_adapter, camera_ids: Sequence[int]) -> Dict[int, List[int]]:
    return {
        int(camera_id): list(storage_adapter.get_visible_blocks(int(camera_id)))
        for camera_id in camera_ids
    }


def _union_blocks(camera_blocks: Dict[int, List[int]]) -> List[int]:
    return sorted({int(block_id) for blocks in camera_blocks.values() for block_id in blocks})


def _prepare_camera(camera, local_uid: int, global_idx: int):
    camera.uid = int(local_uid)
    camera.global_idx = int(global_idx)
    camera.world_view_transform = camera.world_view_transform.cuda(non_blocking=True)
    camera.full_proj_transform = camera.full_proj_transform.cuda(non_blocking=True)
    camera.K = camera.create_k_on_gpu()
    viewmat = camera.world_view_transform.transpose(0, 1)
    camera.camtoworlds = viewmat.inverse().unsqueeze(0)
    return camera


def _tensor_to_uint8_image(tensor):
    import torch
    from PIL import Image

    array = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _safe_preview_name(camera_index: int, image_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(image_name)).strip("._") or "camera"
    return f"{int(camera_index):04d}_{stem}.png"


def _save_preview(path: Path, target, render) -> None:
    from PIL import Image, ImageDraw

    gt_image = _tensor_to_uint8_image(target)
    render_image = _tensor_to_uint8_image(render)
    label_height = 28
    canvas = Image.new("RGB", (gt_image.width + render_image.width, gt_image.height + label_height), "white")
    canvas.paste(gt_image, (0, label_height))
    canvas.paste(render_image, (gt_image.width, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "GT", fill="black")
    draw.text((gt_image.width + 8, 7), "Render", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _empty_render_like(target, background):
    import torch

    if background is None:
        return torch.zeros_like(target)
    return background[:, None, None].expand_as(target).clone()


def _render_camera(
    *,
    camera,
    local_filter,
    gpu_tensors,
    gaussians,
    scene,
    background,
    pipe_args,
):
    import torch
    from strategies.base_engine import pipeline_forward_one_step

    target = normalize_gt_image(
        camera.original_image_backup.cuda(non_blocking=True)
    )
    if int(local_filter.numel()) == 0:
        return _empty_render_like(target, background), target, 0.0

    xyz = gpu_tensors["xyz"].index_select(0, local_filter)
    opacity = gaussians.opacity_activation(gpu_tensors["opacity"].index_select(0, local_filter))
    scaling = gaussians.scaling_activation(gpu_tensors["scaling"].index_select(0, local_filter))
    rotation = gaussians.rotation_activation(gpu_tensors["rotation"].index_select(0, local_filter))
    shs = torch.cat(
        (
            gpu_tensors["features_dc"].index_select(0, local_filter),
            gpu_tensors["features_rest"].index_select(0, local_filter),
        ),
        dim=1,
    )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    render, _, _ = pipeline_forward_one_step(
        opacity,
        scaling,
        rotation,
        xyz,
        shs,
        camera,
        scene,
        gaussians,
        background,
        pipe_args,
        eval=True,
    )
    end_event.record()
    end_event.synchronize()
    render_ms = float(start_event.elapsed_time(end_event))
    render = render.clamp(0.0, 1.0)
    return render, target, render_ms


def evaluate_checkpoint(cli_args: argparse.Namespace) -> Dict[str, object]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu)

    import torch
    import lpips
    import numpy as np

    import utils.general_utils as utils
    from storage.block_reader import TieredCacheBlockReader
    from storage.pure_ssd_checkpoint import load_pure_ssd_checkpoint_manifest
    from storage.tide_storage_adapter import TideStorageAdapter
    from strategies.base_engine import calculate_filters
    from strategies.tide_engine.gaussian_model import TideGaussianModel
    from utils.loss_utils import ssim

    if not torch.cuda.is_available():
        raise RuntimeError("Pure SSD quality evaluation requires CUDA")
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    np.random.seed(42)

    run_dir = Path(cli_args.run_dir).resolve()
    output_dir = Path(cli_args.output_dir).resolve()
    checkpoint_dir = run_dir / "checkpoints" / str(int(cli_args.iteration))
    checkpoint_manifest_path = checkpoint_dir / "pure_ssd_checkpoint.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(parents=True, exist_ok=True)

    manifest_before = checkpoint_manifest_fingerprint(checkpoint_manifest_path)
    manifest = load_pure_ssd_checkpoint_manifest(checkpoint_dir)
    training_args = _configure_eval_args(
        _load_run_args(run_dir),
        output_dir,
        checkpoint_dir,
        manifest,
    )
    validate_resident_configuration(
        RESIDENT_POLICY,
        RESIDENT_CAPACITY,
        RESIDENT_LAMBDA,
        RESIDENT_RECENCY_DECAY,
        BALANCED_SEED_FRACTION,
    )

    log_path = output_dir / "evaluation.log"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    utils.set_args(training_args)
    utils.set_log_file(log_file)

    storage_adapter = None
    gaussians = None
    started = time.perf_counter()
    rows: List[Dict[str, object]] = []
    preview_rows: List[Dict[str, object]] = []
    raw_cache_hits = 0
    source_image_fallbacks = 0

    try:
        gaussians = TideGaussianModel(
            sh_degree=int(training_args.sh_degree),
            only_for_rendering=True,
        )
        (
            test_camera_infos,
            cameras_extent,
            image_height,
            image_width,
        ) = load_test_scene_metadata(training_args)
        gaussians.prepare_pure_ssd_checkpoint_resume(
            manifest,
            cameras_extent,
        )
        scene = SimpleNamespace(
            model_path=str(output_dir),
            cameras_extent=cameras_extent,
        )

        storage_adapter = TideStorageAdapter(
            gaussians=gaussians,
            cameras=test_camera_infos,
            storage_dir=str(output_dir / "ssd_eval_cache"),
            num_clusters=int(getattr(training_args, "num_clusters", 64)),
            max_ram_gb=float(getattr(training_args, "max_ram_gb", 32.0)),
            block_size=int(manifest["block_size"]),
            skip_camera_clustering=False,
            use_6plane=bool(getattr(training_args, "use_6plane", True)),
            execution_mode="paper",
        )
        block_reader = TieredCacheBlockReader(
            cache_manager=storage_adapter.cache,
            total_gaussians=int(manifest["total_points"]),
            block_size=int(manifest["block_size"]),
        )
        gaussians._block_reader = block_reader

        schedule = [int(camera_id) for camera_id in storage_adapter.get_training_schedule(shuffle=False)]
        if sorted(schedule) != list(range(len(test_camera_infos))):
            raise RuntimeError("Test TSP schedule is not a permutation of the Test split")
        schedule_sha256 = fingerprint_camera_schedule(schedule)
        schedule_message = (
            f"[QUALITY SCHEDULE] sha256={schedule_sha256} "
            f"cache_dir={training_args.pure_ssd_schedule_cache_dir}"
        )
        print(schedule_message)
        log_file.write(schedule_message + "\n")
        if int(cli_args.camera_limit) >= 0:
            schedule = schedule[: int(cli_args.camera_limit)]
        if not schedule:
            raise ValueError("camera-limit selected zero cameras")

        evaluated_camera_ids = sorted(schedule)
        preview_positions = select_preview_indices(len(evaluated_camera_ids), int(cli_args.preview_count))
        preview_camera_ids = {evaluated_camera_ids[position] for position in preview_positions}
        schedule_positions = {camera_id: position for position, camera_id in enumerate(schedule)}

        batches = [schedule[start : start + EVAL_BATCH_SIZE] for start in range(0, len(schedule), EVAL_BATCH_SIZE)]
        background = None
        if bool(getattr(training_args, "white_background", False)):
            background = torch.ones(3, dtype=torch.float32, device="cuda")
        lpips_model = lpips.LPIPS(net="alex", version="0.1").cuda().eval()

        resident_blocks: List[int] = []
        recency_scores: Dict[int, float] = {}

        with torch.inference_mode():
            for batch_index, camera_ids in enumerate(batches):
                camera_blocks = _collect_camera_blocks(storage_adapter, camera_ids)
                batch_visible_blocks = _union_blocks(camera_blocks)
                if batch_index == 0:
                    resident_blocks = select_initial_resident_blocks(
                        batch_visible_blocks,
                        camera_blocks,
                        capacity=RESIDENT_CAPACITY,
                        resident_lambda=RESIDENT_LAMBDA,
                        recency_decay=RESIDENT_RECENCY_DECAY,
                        balanced_seed_fraction=BALANCED_SEED_FRACTION,
                    )
                if len(resident_blocks) > RESIDENT_CAPACITY:
                    raise AssertionError("resident working set exceeds cap2048")

                gpu_tensors, _ = gaussians.gpu_working_set_manager.load_visible_blocks_with_retention(
                    visible_block_ids=resident_blocks,
                    enable_retention=True,
                    block_reader=block_reader,
                )
                cameras = []
                for local_uid, camera_id in enumerate(camera_ids):
                    camera_load = load_evaluation_camera(
                        training_args,
                        local_uid,
                        test_camera_infos[camera_id],
                        image_height,
                        image_width,
                    )
                    if camera_load.source == "raw":
                        raw_cache_hits += 1
                    else:
                        source_image_fallbacks += 1
                        fallback_message = (
                            f"[QUALITY IMAGE] source fallback camera={camera_id} "
                            f"image={test_camera_infos[camera_id].image_name} "
                            f"reason={camera_load.fallback_reason} raw={camera_load.raw_path}"
                        )
                        print(fallback_message)
                        log_file.write(fallback_message + "\n")
                    cameras.append(
                        _prepare_camera(camera_load.camera, local_uid, camera_id)
                    )

                filters_local, _, _ = calculate_filters(
                    cameras,
                    gpu_tensors["xyz"],
                    gaussians.opacity_activation(gpu_tensors["opacity"]),
                    gaussians.scaling_activation(gpu_tensors["scaling"]),
                    gaussians.rotation_activation(gpu_tensors["rotation"]),
                )

                resident_set = set(resident_blocks)
                batch_visible_set = set(batch_visible_blocks)
                batch_coverage = (
                    len(batch_visible_set & resident_set) / len(batch_visible_set)
                    if batch_visible_set
                    else 1.0
                )

                for local_uid, (camera_id, camera, local_filter) in enumerate(
                    zip(camera_ids, cameras, filters_local)
                ):
                    render, target, render_ms = _render_camera(
                        camera=camera,
                        local_filter=local_filter,
                        gpu_tensors=gpu_tensors,
                        gaussians=gaussians,
                        scene=scene,
                        background=background,
                        pipe_args=training_args,
                    )
                    psnr_value = compute_psnr(render, target)
                    ssim_value = float(ssim(render.unsqueeze(0), target.unsqueeze(0)).item())
                    lpips_value = float(
                        lpips_model(render.unsqueeze(0), target.unsqueeze(0), normalize=True).item()
                    )
                    metrics = [psnr_value, ssim_value, lpips_value, render_ms]
                    if not all(math.isfinite(value) for value in metrics):
                        raise RuntimeError(
                            f"camera {camera_id} produced non-finite metrics: {metrics}"
                        )

                    camera_visible = set(camera_blocks[camera_id])
                    row = {
                        "camera_index": int(camera_id),
                        "image_name": str(test_camera_infos[camera_id].image_name),
                        "schedule_position": int(schedule_positions[camera_id]),
                        "batch_index": int(batch_index),
                        "camera_visible_blocks": len(camera_visible),
                        "camera_resident_blocks": len(camera_visible & resident_set),
                        "batch_visible_blocks": len(batch_visible_blocks),
                        "resident_blocks": len(resident_blocks),
                        "resident_coverage": float(batch_coverage),
                        "rendered_gaussians": int(local_filter.numel()),
                        "psnr": psnr_value,
                        "ssim": ssim_value,
                        "lpips_alex": lpips_value,
                        "render_ms": render_ms,
                    }
                    rows.append(row)

                    if camera_id in preview_camera_ids:
                        preview_name = _safe_preview_name(camera_id, row["image_name"])
                        _save_preview(output_dir / "previews" / preview_name, target, render)
                        preview_rows.append(
                            {
                                "camera_index": int(camera_id),
                                "image_name": row["image_name"],
                                "schedule_position": int(schedule_positions[camera_id]),
                                "preview_file": f"previews/{preview_name}",
                            }
                        )
                    camera.original_image_backup = None

                if batch_index + 1 < len(batches):
                    next_camera_ids = batches[batch_index + 1]
                    next_camera_blocks = _collect_camera_blocks(storage_adapter, next_camera_ids)
                    next_visible_blocks = _union_blocks(next_camera_blocks)
                    transition = compute_next_resident_transition(
                        batch_visible_blocks,
                        next_visible_blocks,
                        resident_blocks,
                        next_camera_ids,
                        next_camera_blocks,
                        recency_scores,
                        capacity=RESIDENT_CAPACITY,
                        resident_lambda=RESIDENT_LAMBDA,
                        recency_decay=RESIDENT_RECENCY_DECAY,
                        balanced_seed_fraction=BALANCED_SEED_FRACTION,
                    )
                    resident_blocks = sorted(int(block_id) for block_id in transition.next_resident_blocks)
                    recency_scores = dict(transition.updated_recency_scores)

                if batch_index == 0 or (batch_index + 1) % 10 == 0 or batch_index + 1 == len(batches):
                    message = (
                        f"[QUALITY] batch {batch_index + 1}/{len(batches)} "
                        f"cameras={len(rows)}/{len(schedule)} resident={len(resident_blocks)}"
                    )
                    print(message)
                    log_file.write(message + "\n")

        rows.sort(key=lambda row: int(row["camera_index"]))
        preview_rows.sort(key=lambda row: int(row["camera_index"]))
        if len(rows) != len(schedule):
            raise AssertionError(f"expected {len(schedule)} metric rows, got {len(rows)}")
        if len(preview_rows) != len(preview_camera_ids):
            raise AssertionError(
                f"expected {len(preview_camera_ids)} previews, got {len(preview_rows)}"
            )
        if raw_cache_hits + source_image_fallbacks != len(rows):
            raise AssertionError(
                "image source accounting mismatch: "
                f"raw={raw_cache_hits} fallback={source_image_fallbacks} rows={len(rows)}"
            )
        image_source_message = (
            f"[QUALITY IMAGE] raw_cache_hits={raw_cache_hits} "
            f"source_image_fallbacks={source_image_fallbacks}"
        )
        print(image_source_message)
        log_file.write(image_source_message + "\n")

        write_tsv(output_dir / "per_camera_metrics.tsv", rows, PER_CAMERA_FIELDS)
        write_tsv(
            output_dir / "preview_manifest.tsv",
            preview_rows,
            ("camera_index", "image_name", "schedule_position", "preview_file"),
        )

        manifest_after = checkpoint_manifest_fingerprint(checkpoint_manifest_path)
        read_only_verified = manifest_before == manifest_after
        if not read_only_verified:
            raise RuntimeError("checkpoint manifest changed during read-only evaluation")

        elapsed = time.perf_counter() - started
        summary = {
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "iteration": int(cli_args.iteration),
            "camera_count": len(rows),
            "test_split_camera_count": len(test_camera_infos),
            "camera_limit": int(cli_args.camera_limit),
            "preview_count": len(preview_rows),
            "raw_cache_hits": int(raw_cache_hits),
            "source_image_fallbacks": int(source_image_fallbacks),
            "batch_size": EVAL_BATCH_SIZE,
            "schedule_sha256": schedule_sha256,
            "schedule_cache_dir": str(training_args.pure_ssd_schedule_cache_dir),
            "resident": {
                "selection_policy": RESIDENT_POLICY,
                "capacity_blocks": RESIDENT_CAPACITY,
                "lambda": RESIDENT_LAMBDA,
                "recency_decay": RESIDENT_RECENCY_DECAY,
                "balanced_seed_fraction": BALANCED_SEED_FRACTION,
                "resident_blocks": summarize_values(row["resident_blocks"] for row in rows),
                "coverage": summarize_values(row["resident_coverage"] for row in rows),
            },
            "metrics": summarize_metric_rows(rows),
            "rendered_gaussians": summarize_values(row["rendered_gaussians"] for row in rows),
            "elapsed_seconds": float(elapsed),
            "checkpoint_type": manifest.get("checkpoint_type"),
            "storage_index": manifest.get("storage_index", ""),
            "checkpoint_manifest_fingerprint": manifest_after,
            "read_only_verified": True,
            "optimizer_created": False,
            "writeback_executed": False,
        }
        with open(output_dir / "quality_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(
            f"[QUALITY] complete: cameras={len(rows)} previews={len(preview_rows)} "
            f"elapsed={elapsed:.1f}s output={output_dir}"
        )
        return summary
    finally:
        if gaussians is not None and hasattr(gaussians, "gpu_working_set_manager"):
            gaussians.gpu_working_set_manager.clear()
        if storage_adapter is not None:
            storage_adapter.shutdown()
        utils.set_log_file(None)
        log_file.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Pure SSD checkpoint without mutation.")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing args.json.")
    parser.add_argument("--iteration", type=int, default=500000, help="Checkpoint iteration.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics and previews.")
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU id exposed to the evaluator.")
    parser.add_argument(
        "--camera-limit",
        type=int,
        default=-1,
        help="Debug limit in deterministic TSP order; -1 evaluates all Test cameras.",
    )
    parser.add_argument("--preview-count", type=int, default=64, help="Uniform preview count.")
    return parser


def main() -> None:
    evaluate_checkpoint(build_argparser().parse_args())


if __name__ == "__main__":
    main()
