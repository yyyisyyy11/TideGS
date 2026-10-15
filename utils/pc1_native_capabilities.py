"""Fail-fast PC1 native ABI checks, executed on the main thread before workers.

Importing this module does not import torch or compile extensions. The explicit
``check_pc1_native_capabilities`` entry point loads both actual CUDA modules;
the pure validator is also usable by CPU-only tests.
"""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
from pathlib import Path


PC1_NATIVE_ABI_VERSION = 1
GSPLAT_GIL_OPS = (
    "projection_ewa_3dgs_packed_fwd", "projection_ewa_3dgs_packed_bwd",
    "spherical_harmonics_fwd", "spherical_harmonics_bwd",
    "intersect_tile", "intersect_offset",
    "rasterize_to_pixels_3dgs_fwd", "rasterize_to_pixels_3dgs_bwd",
)
CLM_GIL_OPS = ("fusedssim", "fusedssim_backward")


def _rebuild_hint():
    return (
        "Rebuild BOTH extensions from the PC1 checkout, then restart Python: "
        "TIDE_CODE=<checkout> python ssh/superpod/pc1_pipeline/build_gsplat.py "
        "build_ext --inplace --force (use the project's modern binding adapter, "
        "not unadapted gsplat/setup.py); in submodules/clm_kernels run "
        "python setup.py build_ext --inplace --force."
    )


def _inspect_extension(name, module, required_ops, binding):
    path = str(getattr(module, "__file__", "<unavailable>"))
    problems = []
    abi = getattr(module, "PC1_NATIVE_ABI_VERSION", None)
    if type(abi) is not int or abi != PC1_NATIVE_ABI_VERSION:
        problems.append(f"PC1_NATIVE_ABI_VERSION={abi!r}, expected {PC1_NATIVE_ABI_VERSION}")
    actual_binding = getattr(module, "PC1_NATIVE_BINDING", None)
    if actual_binding != binding:
        problems.append(f"PC1_NATIVE_BINDING={actual_binding!r}, expected {binding!r}")
    released_ops = getattr(module, "PC1_GIL_RELEASED_OPS", ())
    if not isinstance(released_ops, (list, tuple)) or any(
        not isinstance(op, str) for op in released_ops
    ):
        problems.append("PC1_GIL_RELEASED_OPS must be a sequence of operation names")
        released_ops = ()
    missing = sorted(set(required_ops) - set(released_ops))
    if missing:
        problems.append("missing GIL release: " + ", ".join(missing))
    missing_exports = [op for op in required_ops if not callable(getattr(module, op, None))]
    if missing_exports:
        problems.append("missing native entry points: " + ", ".join(missing_exports))
    info = {
        "path": path, "abi_version": abi, "binding": actual_binding,
        "gil_released_ops": list(released_ops),
    }
    return info, [f"{name} loaded from {path}: {problem}" for problem in problems]


def validate_pc1_native_capabilities(
    gsplat_module, clm_module, *, require_tile_mask=False, runtime=None,
):
    """Validate already loaded native modules; never imports torch or uses CUDA."""
    gsplat_info, problems = _inspect_extension(
        "gsplat", gsplat_module, GSPLAT_GIL_OPS, "gsplat/cuda/ext.cpp",
    )
    clm_ops = CLM_GIL_OPS + (("tile_contribution_mask",) if require_tile_mask else ())
    clm_info, clm_problems = _inspect_extension(
        "clm_kernels", clm_module, clm_ops, "clm_kernels/ext.cpp",
    )
    problems.extend(clm_problems)
    clm_info["ssim_current_stream"] = getattr(clm_module, "PC1_SSIM_CURRENT_STREAM", False)
    if clm_info["ssim_current_stream"] is not True:
        problems.append(
            f"clm_kernels loaded from {clm_info['path']}: "
            "missing PC1_SSIM_CURRENT_STREAM=True"
        )
    if problems:
        raise RuntimeError("PC1 native capability check failed:\n" + "\n".join(problems)
                           + "\n" + _rebuild_hint())
    return {
        "runtime": dict(runtime or {}), "gsplat": gsplat_info,
        "clm_kernels": clm_info, "require_tile_mask": bool(require_tile_mask),
    }


def check_pc1_native_capabilities(*, require_tile_mask=False, logger=None):
    """Load/check actual extensions before the first communication task.

    This deliberately has no successful-result cache: every startup reports the
    exact modules Python loaded, including a stale installed .so if present.
    Loss compilation/shape warmup must also finish before starting timed work.
    """
    try:
        torch = importlib.import_module("torch")
        gsplat = importlib.import_module("gsplat")
        backend = importlib.import_module("gsplat.cuda._backend")
        clm_native = importlib.import_module("clm_kernels._C")
    except Exception as error:
        raise RuntimeError(f"PC1 native extensions could not be loaded: {error}. "
                           + _rebuild_hint()) from error
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unknown"
    info = validate_pc1_native_capabilities(
        backend._C, clm_native, require_tile_mask=require_tile_mask,
        runtime={
            "code_commit": revision, "python": platform.python_version(),
            "pytorch": str(torch.__version__), "cuda": torch.version.cuda,
            "gsplat": str(getattr(gsplat, "__version__", "unknown")),
        },
    )
    message = "PC1 native capabilities: " + json.dumps(info, sort_keys=True)
    if logger is None:
        print(message, flush=True)
    elif callable(logger):
        logger(message)
    else:
        logger.info(message)
    return info
