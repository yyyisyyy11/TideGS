"""Deterministic camera sampling helpers for 3DGS2-TR curvature batches."""

from __future__ import annotations

import operator
from typing import List, Sequence

import numpy as np
import torch


__all__ = ["make_curvature_probe_generator", "sample_s2_camera_ids"]


def _nonnegative_index(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be >= 0")
    return int(result)


def _positive_index(value: int, name: str) -> int:
    result = _nonnegative_index(value, name)
    if result < 1:
        raise ValueError(f"{name} must be >= 1")
    return result


def make_curvature_probe_generator(
    *,
    base_seed: int,
    optimizer_step: int,
    microbatch_index: int,
    sample_index: int,
    rank: int,
    device: torch.device,
) -> torch.Generator:
    """Build a stateless generator for one distributed Hutchinson probe."""

    seed_values = (
        _nonnegative_index(base_seed, "base_seed"),
        _positive_index(optimizer_step, "optimizer_step"),
        _nonnegative_index(microbatch_index, "microbatch_index"),
        _nonnegative_index(sample_index, "sample_index"),
        _nonnegative_index(rank, "rank"),
    )
    seed_word = int(
        np.random.SeedSequence(seed_values).generate_state(1, dtype=np.uint64)[0]
    )
    # torch.Generator.manual_seed accepts signed 64-bit values. Preserve all
    # lower 63 bits while keeping the seed portable across CPU and CUDA.
    seed_word &= (1 << 63) - 1
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed_word)
    return generator


def sample_s2_camera_ids(
    *,
    population_camera_ids: Sequence[int],
    s1_camera_ids: Sequence[int],
    seed: int,
    optimizer_step: int,
) -> List[int]:
    """Sample an independent, stateless S2 batch without replacement.

    S2 has the same global size as S1. Sampling uses the complete training
    camera population, so an S2 camera may also occur in S1 while every camera
    remains unique within S2 itself.
    """

    population = [int(camera_id) for camera_id in population_camera_ids]
    s1 = [int(camera_id) for camera_id in s1_camera_ids]
    seed = _nonnegative_index(seed, "seed")
    optimizer_step = _positive_index(optimizer_step, "optimizer_step")

    if len(set(population)) != len(population):
        raise ValueError("population_camera_ids must contain unique camera IDs")
    if len(set(s1)) != len(s1):
        raise ValueError("s1_camera_ids must contain unique camera IDs")
    if len(s1) > len(population):
        raise ValueError("S1 batch size cannot exceed the camera population")
    missing = sorted(set(s1).difference(population))
    if missing:
        raise ValueError(
            "S1 camera IDs are missing from the camera population: "
            f"{missing[:8]}"
        )
    if not s1:
        return []

    # SeedSequence combines both values without mutating NumPy's global RNG.
    generator = np.random.default_rng(
        np.random.SeedSequence([seed, optimizer_step])
    )
    positions = generator.choice(len(population), size=len(s1), replace=False)
    return [population[int(position)] for position in positions]
