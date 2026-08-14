#!/usr/bin/env python3
"""Read-only camera loading helpers for Pure SSD quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class EvaluationCameraLoad:
    camera: object
    source: str
    raw_path: Path
    fallback_reason: str = ""


def raw_cache_path(args, camera_info) -> Path:
    cache_key = getattr(camera_info, "image_cache_key", None) or camera_info.image_name
    relative_name = str(cache_key).lstrip("/") + ".raw"
    return Path(args.decode_dataset_path) / "dataset_raw" / relative_name


def raw_cache_channels(path: str | Path, image_height: int, image_width: int) -> int:
    path = Path(path)
    if not path.is_file():
        return 0
    pixels = int(image_height) * int(image_width)
    if pixels <= 0:
        return 0
    size = int(path.stat().st_size)
    if size % pixels != 0:
        return 0
    channels = size // pixels
    return channels if channels in (3, 4) else 0


def raw_cache_is_usable(path: str | Path, image_height: int, image_width: int) -> bool:
    return raw_cache_channels(path, image_height, image_width) > 0


def load_source_image_camera(args, camera_id: int, camera_info, image_height: int, image_width: int):
    """Load the source image without writing or populating the decoded cache."""
    import torch
    from PIL import Image

    from scene.cameras import Camera

    with Image.open(camera_info.image_path) as image:
        image = image.convert("RGB").crop((0, 0, int(image_width), int(image_height)))
        image_bytes = bytearray(image.tobytes())
    image_tensor = (
        torch.frombuffer(image_bytes, dtype=torch.uint8)
        .view(int(image_height), int(image_width), 3)
        .permute(2, 0, 1)
        .contiguous()
    )
    return Camera(
        colmap_id=camera_info.uid,
        R=camera_info.R,
        T=camera_info.T,
        FoVx=camera_info.FovX,
        FoVy=camera_info.FovY,
        image=image_tensor,
        gt_alpha_mask=None,
        image_name=camera_info.image_name,
        uid=int(camera_id),
        offload=True,
    )


def load_evaluation_camera(
    args,
    camera_id: int,
    camera_info,
    image_height: int,
    image_width: int,
    *,
    use_raw_cache: bool = True,
    raw_loader: Optional[Callable] = None,
    source_loader: Optional[Callable] = None,
) -> EvaluationCameraLoad:
    path = raw_cache_path(args, camera_info)
    if not use_raw_cache:
        if source_loader is None:
            source_loader = load_source_image_camera
        camera = source_loader(args, camera_id, camera_info, image_height, image_width)
        return EvaluationCameraLoad(
            camera=camera,
            source="source_image",
            raw_path=path,
            fallback_reason="raw_disabled",
        )

    raw_usable = raw_cache_is_usable(path, image_height, image_width)
    if raw_usable:
        if raw_loader is None:
            from utils.camera_utils import loadCam_raw_from_disk

            raw_loader = loadCam_raw_from_disk
        try:
            camera = raw_loader(args, camera_id, camera_info)
            return EvaluationCameraLoad(camera=camera, source="raw", raw_path=path)
        except Exception as exc:
            fallback_reason = f"raw_load_failed:{type(exc).__name__}"
    else:
        fallback_reason = "raw_missing" if not path.is_file() else "raw_invalid_size"

    if source_loader is None:
        source_loader = load_source_image_camera
    camera = source_loader(args, camera_id, camera_info, image_height, image_width)
    return EvaluationCameraLoad(
        camera=camera,
        source="source_image",
        raw_path=path,
        fallback_reason=fallback_reason,
    )


def normalize_gt_image(image):
    """Normalize uint8 or already-normalized floating RGB tensors to float32 [0, 1]."""
    import torch

    if image.dtype == torch.uint8:
        return image.float().div(255.0)
    if image.is_floating_point():
        normalized = image.float()
        if normalized.numel() > 0:
            minimum = float(normalized.min().item())
            maximum = float(normalized.max().item())
            if minimum < -1e-6 or maximum > 1.0 + 1e-6:
                raise ValueError(
                    f"floating GT image must already be in [0, 1], got range [{minimum}, {maximum}]"
                )
        return normalized.clamp(0.0, 1.0)
    raise TypeError(f"unsupported GT image dtype: {image.dtype}")


def load_scene_camera_metadata(args, split: str):
    """Load one camera split and image dimensions without constructing Scene."""
    import utils.general_utils as utils
    from scene import load_scene_info_for_rendering

    scene_info, cameras_extent = load_scene_info_for_rendering(args)
    train_camera_infos = list(scene_info.train_cameras or [])
    test_camera_infos = list(scene_info.test_cameras or [])
    all_camera_infos = train_camera_infos + test_camera_infos
    if not all_camera_infos:
        raise RuntimeError("camera metadata is empty")
    split = str(split).lower()
    if split == "train":
        camera_infos = train_camera_infos
    elif split == "test":
        camera_infos = test_camera_infos
    else:
        raise ValueError(f"unsupported camera split: {split!r}")
    if not camera_infos:
        raise RuntimeError(f"{split.capitalize()} split is empty")
    image_width = min(int(camera.width) for camera in all_camera_infos)
    image_height = min(int(camera.height) for camera in all_camera_infos)
    utils.set_img_size(image_height, image_width)
    return camera_infos, float(cameras_extent), image_height, image_width


def load_test_scene_metadata(args):
    """Load Test camera metadata without constructing Scene or decoding images."""
    return load_scene_camera_metadata(args, "test")


__all__ = [
    "EvaluationCameraLoad",
    "load_evaluation_camera",
    "load_scene_camera_metadata",
    "load_source_image_camera",
    "load_test_scene_metadata",
    "normalize_gt_image",
    "raw_cache_channels",
    "raw_cache_is_usable",
    "raw_cache_path",
]
