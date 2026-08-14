"""Optimizer provenance and strict checkpoint-resume validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Iterable


_MISSING = object()


def normalize_optimizer_algorithm(value: Any) -> str:
    algorithm = str(value).strip().lower()
    if algorithm in {"3dgs2-tr", "sophia_tr", "sophia-tr"}:
        return "3dgs2_tr"
    return algorithm


def _normalize_resident_policy(value: Any) -> str:
    policy = str(value).strip().lower()
    return "topc_strict" if policy == "topc" else policy


def _normalize_quaternion_cap(value: Any) -> float:
    cap = float(value)
    return -1.0 if cap < 0.0 else cap


def _normalize_lower(value: Any) -> str:
    return str(value).strip().lower()


def _optimizer_step_count(iterations: Any, bsz: Any) -> int:
    iterations = int(iterations)
    bsz = int(bsz)
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if bsz <= 0:
        raise ValueError(f"bsz must be positive, got {bsz}")
    return (iterations - 1) // bsz + 1


_ALIASES = {
    "paper_optimizer_algorithm": (
        "paper_optimizer_algorithm",
        "optimizer_algorithm",
        "optimizer",
    ),
    "paper_sophia_beta1": ("paper_sophia_beta1", "sophia_beta1"),
    "paper_sophia_beta2": ("paper_sophia_beta2", "sophia_beta2"),
    "paper_sophia_curvature_interval": (
        "paper_sophia_curvature_interval",
        "sophia_curvature_interval",
        "curvature_interval",
    ),
    "paper_sophia_hutchinson_samples": (
        "paper_sophia_hutchinson_samples",
        "sophia_hutchinson_samples",
        "hutchinson_samples",
    ),
    "paper_sophia_curvature_estimator": (
        "paper_sophia_curvature_estimator",
        "sophia_curvature_estimator",
        "curvature_estimator",
    ),
    "paper_sophia_gamma": ("paper_sophia_gamma", "sophia_gamma"),
    "paper_sophia_epsilon": ("paper_sophia_epsilon", "sophia_epsilon"),
    "paper_sophia_curvature_seed": (
        "paper_sophia_curvature_seed",
        "sophia_curvature_seed",
        "curvature_seed",
    ),
    "paper_tr_epsilon_init": ("paper_tr_epsilon_init", "tr_epsilon_init"),
    "paper_tr_epsilon_final": ("paper_tr_epsilon_final", "tr_epsilon_final"),
    "paper_tr_quat_norm": ("paper_tr_quat_norm", "tr_quat_norm"),
}


_FIELD_SPECS = {
    "paper_optimizer_algorithm": (normalize_optimizer_algorithm, "adam"),
    "bsz": (int, _MISSING),
    "paper_optimizer_backend": (_normalize_lower, "gpu_resident"),
    "paper_optimizer_state_mode": (_normalize_lower, "resident_blocks"),
    "paper_resident_capacity_blocks": (int, 2048),
    "paper_resident_selection_policy": (_normalize_resident_policy, "topc_balanced"),
    "paper_resident_lambda": (float, 0.3),
    "paper_resident_recency_decay": (float, 0.95),
    "paper_balanced_seed_fraction": (float, 0.25),
    "lambda_dssim": (float, 0.2),
    "ssd_schedule_ordering": (_normalize_lower, "trajectory"),
    "paper_sophia_beta1": (float, 0.9),
    "paper_sophia_beta2": (float, 0.999),
    "paper_sophia_curvature_interval": (int, 10),
    "paper_sophia_hutchinson_samples": (int, 1),
    "paper_sophia_curvature_estimator": (_normalize_lower, "residual_vjp"),
    "paper_sophia_gamma": (float, 1.0),
    "paper_sophia_epsilon": (float, 1e-15),
    "paper_sophia_curvature_seed": (int, 1),
    "paper_tr_epsilon_init": (float, 1e-6),
    "paper_tr_epsilon_final": (float, 1e-8),
    "paper_tr_quat_norm": (_normalize_quaternion_cap, -1.0),
}

_COMMON_FIELDS = (
    "paper_optimizer_algorithm",
    "bsz",
    "paper_optimizer_backend",
    "paper_optimizer_state_mode",
    "paper_resident_capacity_blocks",
    "paper_resident_selection_policy",
    "paper_resident_lambda",
    "paper_resident_recency_decay",
    "paper_balanced_seed_fraction",
    "lambda_dssim",
    "ssd_schedule_ordering",
)

_SOPHIA_FIELDS = (
    "paper_sophia_beta1",
    "paper_sophia_beta2",
    "paper_sophia_curvature_interval",
    "paper_sophia_hutchinson_samples",
    "paper_sophia_curvature_estimator",
    "paper_sophia_gamma",
    "paper_sophia_epsilon",
    "paper_sophia_curvature_seed",
    "paper_tr_epsilon_init",
    "paper_tr_epsilon_final",
    "paper_tr_quat_norm",
)


def _get(source: Any, name: str, default: Any = _MISSING) -> Any:
    aliases = _ALIASES.get(name, (name,))
    if isinstance(source, Mapping):
        for alias in aliases:
            if alias in source:
                return source[alias]
    else:
        for alias in aliases:
            if hasattr(source, alias):
                return getattr(source, alias)
    if default is _MISSING:
        raise KeyError(name)
    return default


def _canonical_value(source: Any, name: str, *, use_default: bool) -> Any:
    normalize, default = _FIELD_SPECS[name]
    value = _get(source, name, default if use_default else _MISSING)
    if value is None:
        if default is _MISSING:
            raise KeyError(name)
        value = default
    return normalize(value)


def build_optimizer_provenance(args: Any) -> Dict[str, Any]:
    """Build the canonical optimizer configuration stored by v3 checkpoints."""

    provenance = {"provenance_version": 1}
    for name in _COMMON_FIELDS + _SOPHIA_FIELDS:
        provenance[name] = _canonical_value(args, name, use_default=True)

    iterations = _get(args, "iterations", None)
    if iterations is not None:
        provenance["iterations"] = int(iterations)
        provenance["optimizer_total_steps"] = _optimizer_step_count(
            iterations,
            provenance["bsz"],
        )
    return provenance


def _format_mismatches(prefix: str, mismatches: Iterable[str]) -> ValueError:
    return ValueError(f"{prefix}: " + "; ".join(mismatches))


def validate_optimizer_provenance(
    args: Any,
    provenance: Mapping[str, Any],
    *,
    strict: bool,
    context: str,
) -> None:
    """Validate an optimizer provenance mapping against requested arguments."""

    try:
        saved_algorithm = _canonical_value(
            provenance,
            "paper_optimizer_algorithm",
            use_default=False,
        )
    except KeyError:
        requested = _canonical_value(args, "paper_optimizer_algorithm", use_default=True)
        if requested != "adam":
            raise ValueError(
                f"{context} has no optimizer metadata and predates 3DGS2-TR; "
                "resume it with Adam"
            )
        return

    requested_algorithm = _canonical_value(
        args,
        "paper_optimizer_algorithm",
        use_default=True,
    )
    if saved_algorithm != requested_algorithm:
        raise ValueError(
            f"{context} optimizer mismatch: checkpoint={saved_algorithm!r}, "
            f"requested={requested_algorithm!r}"
        )

    fields = list(_COMMON_FIELDS)
    if saved_algorithm == "3dgs2_tr":
        fields.extend(_SOPHIA_FIELDS)
    mismatches = []
    for name in fields:
        if name == "paper_optimizer_algorithm":
            continue
        try:
            saved = _canonical_value(provenance, name, use_default=not strict)
        except KeyError:
            mismatches.append(f"{name}: missing from checkpoint")
            continue
        try:
            requested = _canonical_value(args, name, use_default=True)
        except KeyError:
            mismatches.append(f"{name}: missing from requested configuration")
            continue
        if saved != requested:
            mismatches.append(f"{name}: checkpoint={saved!r}, requested={requested!r}")

    saved_total_steps_field = provenance.get("optimizer_total_steps")
    saved_total_steps = saved_total_steps_field
    if (
        saved_total_steps is None
        and provenance.get("iterations") is not None
        and provenance.get("bsz") is not None
    ):
        saved_total_steps = _optimizer_step_count(
            provenance["iterations"],
            provenance["bsz"],
        )
    requested_iterations = _get(args, "iterations", None)
    requested_bsz = _get(args, "bsz", None)
    if saved_algorithm == "3dgs2_tr":
        if strict and saved_total_steps_field is None:
            mismatches.append("optimizer_total_steps: missing from checkpoint")
        elif (
            saved_total_steps is not None
            and requested_iterations is not None
            and requested_bsz is not None
        ):
            requested_total_steps = _optimizer_step_count(
                requested_iterations,
                requested_bsz,
            )
            if int(saved_total_steps) != requested_total_steps:
                mismatches.append(
                    "optimizer_total_steps: "
                    f"checkpoint={int(saved_total_steps)!r}, "
                    f"requested={requested_total_steps!r}"
                )

    if mismatches:
        raise _format_mismatches(f"{context} configuration mismatch", mismatches)


def validate_pure_ssd_checkpoint_optimizer(args: Any, manifest: Mapping[str, Any]) -> None:
    """Validate optimizer semantics for a single-rank or wrapped checkpoint."""

    distributed_root = manifest.get("_tide_distributed_root")
    if distributed_root is not None:
        validate_distributed_checkpoint_resume(args, distributed_root)
        return
    validate_optimizer_provenance(
        args,
        manifest.get("args") or {},
        strict=False,
        context="Pure SSD checkpoint",
    )


def validate_distributed_checkpoint_resume(
    args: Any,
    root: Mapping[str, Any],
    *,
    world_size: int | None = None,
    global_bsz: int | None = None,
) -> None:
    """Reject distributed resumes whose execution or optimizer semantics changed."""

    version = int(root.get("checkpoint_version", 1))
    if version not in {1, 2, 3}:
        raise ValueError(
            "Unsupported distributed checkpoint version: "
            f"{version}; supported versions are 1, 2, and 3"
        )
    requested_world_size = world_size
    if requested_world_size is None:
        requested_world_size = _get(args, "world_size", None)
    if requested_world_size is not None and int(root["world_size"]) != int(
        requested_world_size
    ):
        raise ValueError(
            "Distributed checkpoint world_size mismatch: "
            f"checkpoint={root['world_size']!r}, requested={requested_world_size!r}"
        )

    requested_bsz = global_bsz
    if requested_bsz is None:
        requested_bsz = _get(args, "bsz", None)
    if requested_bsz is not None and int(root["global_bsz"]) != int(requested_bsz):
        raise ValueError(
            "Distributed checkpoint global_bsz mismatch: "
            f"checkpoint={root['global_bsz']!r}, requested={requested_bsz!r}"
        )

    requested_algorithm = _canonical_value(
        args,
        "paper_optimizer_algorithm",
        use_default=True,
    )
    if version < 3:
        if requested_algorithm != "adam":
            raise ValueError(
                f"Distributed checkpoint v{version} predates optimizer provenance; "
                "legacy v1/v2 checkpoints can only resume with Adam"
            )
    else:
        provenance = root.get("optimizer_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("Distributed checkpoint v3 is missing optimizer_provenance")
        validate_optimizer_provenance(
            args,
            provenance,
            strict=True,
            context="Distributed checkpoint",
        )
        if str(root.get("optimizer_state_mode")) != "cold_start_per_rank":
            raise ValueError(
                "Distributed checkpoint v3 optimizer_state_mode must be "
                "'cold_start_per_rank'"
            )
        if bool(root.get("optimizer_ema_saved", True)):
            raise ValueError(
                "Distributed checkpoint v3 unexpectedly claims persisted optimizer EMA state"
            )
        if bool(root.get("optimizer_curvature_saved", True)):
            raise ValueError(
                "Distributed checkpoint v3 unexpectedly claims persisted curvature EMA state"
            )
        global_optimizer_step = int(root.get("global_optimizer_step", -1))
        if global_optimizer_step < 0:
            raise ValueError(
                "Distributed checkpoint v3 has an invalid global_optimizer_step"
            )
        expected_step = max(
            0,
            (int(root["next_iteration"]) - 1) // int(root["global_bsz"]),
        )
        if global_optimizer_step != expected_step:
            raise ValueError(
                "Distributed checkpoint global_optimizer_step mismatch: "
                f"checkpoint={global_optimizer_step!r}, expected={expected_step!r} "
                "from next_iteration/global_bsz"
            )
        saved_total_steps = int(provenance.get("optimizer_total_steps", 0))
        if saved_total_steps and global_optimizer_step > saved_total_steps:
            raise ValueError(
                "Distributed checkpoint global_optimizer_step exceeds "
                "optimizer_total_steps"
            )

    if version >= 2:
        saved_capacity = int(root["global_capacity_blocks"])
    else:
        saved_capacity = int(root["per_rank_capacity_blocks"]) * int(root["world_size"])
    requested_capacity = _get(args, "paper_resident_capacity_blocks", None)
    if requested_capacity is not None and saved_capacity != int(requested_capacity):
        raise ValueError(
            "Distributed checkpoint paper_resident_capacity_blocks mismatch: "
            f"checkpoint={saved_capacity!r}, requested={int(requested_capacity)!r}"
        )

    camera_fields = {
        "tide_camera_assignment": "camera_assignment",
        "tide_camera_microbatch": "camera_microbatch",
    }
    for argument_name, manifest_name in camera_fields.items():
        requested = _get(args, argument_name, None)
        if requested is None or manifest_name not in root:
            continue
        normalize = int if argument_name == "tide_camera_microbatch" else _normalize_lower
        saved_value = normalize(root[manifest_name])
        requested_value = normalize(requested)
        if saved_value != requested_value:
            raise ValueError(
                f"Distributed checkpoint {argument_name} mismatch: "
                f"checkpoint={saved_value!r}, requested={requested_value!r}"
            )
