"""Deterministic CPU planning for Gaussian-sharded TideGS batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .resident_policy import compute_topc_resident_transition


def rows_in_block(block_id: int, total_points: int, block_size: int) -> int:
    start = int(block_id) * int(block_size)
    return max(0, min(int(block_size), int(total_points) - start))


def build_balanced_block_owner(
    *,
    num_blocks: int,
    total_points: int,
    block_size: int,
    world_size: int,
    visibility_counts: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Assign every block once using deterministic largest-processing-time scheduling."""
    if num_blocks < 0 or total_points < 0 or block_size <= 0 or world_size <= 0:
        raise ValueError("Invalid block-owner dimensions")
    if visibility_counts is not None and len(visibility_counts) != num_blocks:
        raise ValueError("visibility_counts must contain one value per block")

    weights = []
    for block_id in range(num_blocks):
        rows = rows_in_block(block_id, total_points, block_size)
        visibility = 0 if visibility_counts is None else max(0, int(visibility_counts[block_id]))
        weights.append(rows * (1 + visibility))

    owner = np.full((num_blocks,), -1, dtype=np.int32)
    rank_loads = [0 for _ in range(world_size)]
    rank_counts = [0 for _ in range(world_size)]
    for block_id in sorted(range(num_blocks), key=lambda value: (-weights[value], value)):
        rank = min(range(world_size), key=lambda value: (rank_loads[value], rank_counts[value], value))
        owner[block_id] = rank
        rank_loads[rank] += weights[block_id]
        rank_counts[rank] += 1
    validate_block_owner(owner, num_blocks=num_blocks, world_size=world_size)
    return owner


def build_stable_block_owner(*, num_blocks: int, world_size: int) -> np.ndarray:
    """Assign blocks once with deterministic round-robin ownership."""
    if num_blocks < 0 or world_size <= 0:
        raise ValueError("Invalid block-owner dimensions")
    owner = np.arange(int(num_blocks), dtype=np.int64) % int(world_size)
    owner = owner.astype(np.int32, copy=False)
    validate_block_owner(owner, num_blocks=num_blocks, world_size=world_size)
    return owner


def validate_block_owner(owner: Sequence[int], *, num_blocks: int, world_size: int) -> None:
    values = np.asarray(owner, dtype=np.int64)
    if values.shape != (int(num_blocks),):
        raise ValueError(f"Expected block_owner shape {(int(num_blocks),)}, got {values.shape}")
    invalid = np.flatnonzero((values < 0) | (values >= int(world_size)))
    if invalid.size:
        raise ValueError(f"Invalid block owners at indices {invalid[:8].tolist()}")


def assign_cameras_balanced(
    *,
    camera_ids: Sequence[int],
    camera_blocks: Mapping[int, Sequence[int]],
    block_rows: Sequence[int],
    world_size: int,
) -> List[List[int]]:
    if world_size <= 0 or len(camera_ids) % world_size != 0:
        raise ValueError("Global camera count must be divisible by world_size")
    slots_per_rank = len(camera_ids) // world_size
    assignments: List[List[int]] = [[] for _ in range(world_size)]
    loads = [0 for _ in range(world_size)]
    original_position = {int(camera_id): position for position, camera_id in enumerate(camera_ids)}

    def camera_cost(camera_id: int) -> int:
        return sum(
            int(block_rows[int(block_id)])
            for block_id in camera_blocks.get(int(camera_id), [])
            if 0 <= int(block_id) < len(block_rows)
        )

    ordered = sorted(
        (int(camera_id) for camera_id in camera_ids),
        key=lambda camera_id: (-camera_cost(camera_id), original_position[camera_id], camera_id),
    )
    for camera_id in ordered:
        eligible = [rank for rank in range(world_size) if len(assignments[rank]) < slots_per_rank]
        rank = min(eligible, key=lambda value: (loads[value], len(assignments[value]), value))
        assignments[rank].append(camera_id)
        loads[rank] += camera_cost(camera_id)

    for rank in range(world_size):
        assignments[rank].sort(key=lambda camera_id: original_position[camera_id])
    return assignments


@dataclass(frozen=True)
class DistributedBatchPlan:
    iteration: int
    epoch: int
    global_camera_ids: List[int]
    rank_camera_ids: List[List[int]]
    rank_resident_blocks: List[List[int]]
    rank_active_blocks: List[List[int]]
    global_active_blocks: List[int] = field(default_factory=list)
    global_resident_blocks: List[int] = field(default_factory=list)
    stream_in_blocks: List[int] = field(default_factory=list)
    evict_blocks: List[int] = field(default_factory=list)
    block_cull_ms: float = 0.0
    plan_ms: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DistributedBatchPlan":
        rank_active_blocks = [
            [int(v) for v in values] for values in value["rank_active_blocks"]
        ]
        rank_resident_blocks = [
            [int(v) for v in values] for values in value["rank_resident_blocks"]
        ]
        return cls(
            iteration=int(value["iteration"]),
            epoch=int(value["epoch"]),
            global_camera_ids=[int(v) for v in value["global_camera_ids"]],
            rank_camera_ids=[[int(v) for v in values] for values in value["rank_camera_ids"]],
            rank_resident_blocks=rank_resident_blocks,
            rank_active_blocks=rank_active_blocks,
            global_active_blocks=[
                int(v)
                for v in value.get(
                    "global_active_blocks",
                    sorted({block for values in rank_active_blocks for block in values}),
                )
            ],
            global_resident_blocks=[
                int(v)
                for v in value.get(
                    "global_resident_blocks",
                    sorted({block for values in rank_resident_blocks for block in values}),
                )
            ],
            stream_in_blocks=[int(v) for v in value.get("stream_in_blocks", [])],
            evict_blocks=[int(v) for v in value.get("evict_blocks", [])],
            block_cull_ms=float(value.get("block_cull_ms", 0.0)),
            plan_ms=float(value.get("plan_ms", 0.0)),
        )


class DistributedBatchPlanner:
    def __init__(
        self,
        *,
        block_owner: Sequence[int],
        total_points: int,
        block_size: int,
        world_size: int,
        resident_capacity_blocks: int,
        resident_lambda: float,
        resident_recency_decay: float,
        balanced_seed_fraction: float,
        camera_assignment: str = "gaussian_balanced",
    ):
        self.block_owner = np.asarray(block_owner, dtype=np.int32)
        self.total_points = int(total_points)
        self.block_size = int(block_size)
        self.world_size = int(world_size)
        self.capacity = int(resident_capacity_blocks)
        self.resident_lambda = float(resident_lambda)
        self.recency_decay = float(resident_recency_decay)
        self.balanced_seed_fraction = float(balanced_seed_fraction)
        self.camera_assignment = str(camera_assignment).lower()
        validate_block_owner(
            self.block_owner,
            num_blocks=len(self.block_owner),
            world_size=self.world_size,
        )
        self._block_rows = [
            rows_in_block(block_id, self.total_points, self.block_size)
            for block_id in range(len(self.block_owner))
        ]
        self._resident: List[int] = []
        self._active: List[int] = []
        self._recency: Dict[int, float] = {}

    def plan(
        self,
        *,
        iteration: int,
        epoch: int,
        camera_ids: Sequence[int],
        camera_blocks: Mapping[int, Sequence[int]],
    ) -> DistributedBatchPlan:
        global_camera_ids = [int(camera_id) for camera_id in camera_ids]
        if len(global_camera_ids) % self.world_size != 0:
            raise ValueError("Global camera count must be divisible by world_size")
        if len(set(global_camera_ids)) != len(global_camera_ids):
            raise ValueError("Distributed camera batches must contain unique camera IDs")
        if self.camera_assignment == "equal":
            local_count = len(global_camera_ids) // self.world_size
            rank_camera_ids = [
                global_camera_ids[rank * local_count:(rank + 1) * local_count]
                for rank in range(self.world_size)
            ]
        elif self.camera_assignment == "gaussian_balanced":
            rank_camera_ids = assign_cameras_balanced(
                camera_ids=global_camera_ids,
                camera_blocks=camera_blocks,
                block_rows=self._block_rows,
                world_size=self.world_size,
            )
        else:
            raise ValueError(f"Unsupported camera assignment: {self.camera_assignment}")

        normalized_camera_blocks = {
            int(camera_id): sorted(
                {
                    int(block_id)
                    for block_id in camera_blocks.get(int(camera_id), [])
                    if 0 <= int(block_id) < len(self.block_owner)
                }
            )
            for camera_id in global_camera_ids
        }
        global_active = sorted(
            {
                block_id
                for blocks in normalized_camera_blocks.values()
                for block_id in blocks
            }
        )
        transition = compute_topc_resident_transition(
            current_active_blocks=self._active,
            next_active_blocks=global_active,
            current_resident_blocks=self._resident,
            next_camera_ids=global_camera_ids,
            next_camera_blocks=normalized_camera_blocks,
            previous_recency_scores=self._recency,
            lambda_weight=self.resident_lambda,
            recency_decay=self.recency_decay,
            resident_capacity_blocks=self.capacity,
            balanced_camera_seeds=True,
            balanced_seed_fraction=self.balanced_seed_fraction,
        )
        global_resident = list(transition.next_resident_blocks)
        self._active = global_active
        self._resident = global_resident
        self._recency = dict(transition.updated_recency_scores or {})

        rank_active_blocks = [
            [block_id for block_id in global_active if self.block_owner[block_id] == rank]
            for rank in range(self.world_size)
        ]
        rank_resident_blocks = [
            [block_id for block_id in global_resident if self.block_owner[block_id] == rank]
            for rank in range(self.world_size)
        ]

        resident_sets = [set(values) for values in rank_resident_blocks]
        for left in range(self.world_size):
            for right in range(left + 1, self.world_size):
                if resident_sets[left].intersection(resident_sets[right]):
                    raise RuntimeError("Distributed resident sets must be pairwise disjoint")

        return DistributedBatchPlan(
            iteration=int(iteration),
            epoch=int(epoch),
            global_camera_ids=global_camera_ids,
            rank_camera_ids=rank_camera_ids,
            rank_resident_blocks=rank_resident_blocks,
            rank_active_blocks=rank_active_blocks,
            global_active_blocks=global_active,
            global_resident_blocks=global_resident,
            stream_in_blocks=list(transition.stream_in_blocks),
            evict_blocks=list(transition.evict_blocks),
        )
