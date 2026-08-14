"""Deterministic CPU planning for Gaussian-sharded TideGS batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
    block_cull_backend: str = "cpu"
    block_cull_gpu_kernel_ms: float = 0.0
    block_cull_gpu_d2h_ms: float = 0.0
    block_cull_cache_hit_cameras: int = 0
    block_cull_gpu_cameras: int = 0
    block_cull_output_blocks: int = 0
    plan_ms: float = 0.0
    predicted_stream_in_blocks: int = 0
    prediction_missing_blocks: int = 0
    prediction_extra_blocks: int = 0
    prediction_replanned: int = 0
    prediction_repair_ms: float = 0.0
    prediction_exact_plan_ms: float = 0.0
    global_s2_camera_ids: List[int] = field(default_factory=list)
    rank_s2_camera_ids: Optional[List[List[int]]] = None
    rank_gradient_active_blocks: Optional[List[List[int]]] = None
    rank_curvature_active_blocks: Optional[List[List[int]]] = None
    global_gradient_active_blocks: Optional[List[int]] = None
    global_curvature_active_blocks: List[int] = field(default_factory=list)
    rank_owner_active_rows: Optional[List[int]] = None
    rank_participation_rows: Optional[List[int]] = None

    def __post_init__(self) -> None:
        world_size = len(self.rank_camera_ids)
        if self.rank_s2_camera_ids is None:
            object.__setattr__(
                self, "rank_s2_camera_ids", [[] for _ in range(world_size)]
            )
        if self.rank_gradient_active_blocks is None:
            object.__setattr__(
                self,
                "rank_gradient_active_blocks",
                [list(values) for values in self.rank_active_blocks],
            )
        if self.rank_curvature_active_blocks is None:
            object.__setattr__(
                self,
                "rank_curvature_active_blocks",
                [[] for _ in range(world_size)],
            )
        if self.global_gradient_active_blocks is None:
            object.__setattr__(
                self,
                "global_gradient_active_blocks",
                list(self.global_active_blocks),
            )
        if self.rank_owner_active_rows is None:
            object.__setattr__(
                self,
                "rank_owner_active_rows",
                [0 for _ in range(world_size)],
            )
        if self.rank_participation_rows is None:
            object.__setattr__(
                self,
                "rank_participation_rows",
                [max(1, int(value)) for value in self.rank_owner_active_rows],
            )
        rank_fields = {
            "rank_resident_blocks": self.rank_resident_blocks,
            "rank_active_blocks": self.rank_active_blocks,
            "rank_s2_camera_ids": self.rank_s2_camera_ids,
            "rank_gradient_active_blocks": self.rank_gradient_active_blocks,
            "rank_curvature_active_blocks": self.rank_curvature_active_blocks,
            "rank_owner_active_rows": self.rank_owner_active_rows,
            "rank_participation_rows": self.rank_participation_rows,
        }
        for name, values in rank_fields.items():
            if values is None or len(values) != world_size:
                raise ValueError(f"{name} must contain one entry per rank")
        if any(int(value) < 0 for value in self.rank_owner_active_rows):
            raise ValueError("rank_owner_active_rows must be non-negative")
        if [int(value) for value in self.rank_participation_rows] != [
            max(1, int(value)) for value in self.rank_owner_active_rows
        ]:
            raise ValueError(
                "rank_participation_rows must equal max(1, rank_owner_active_rows)"
            )
        has_rank_s2 = any(self.rank_s2_camera_ids)
        if has_rank_s2 and not self.global_s2_camera_ids:
            raise ValueError("rank_s2_camera_ids requires global_s2_camera_ids")
        if self.global_s2_camera_ids:
            if len(self.global_s2_camera_ids) != len(self.global_camera_ids):
                raise ValueError("S2 global camera count must equal S1")
            if len(set(self.global_s2_camera_ids)) != len(self.global_s2_camera_ids):
                raise ValueError("global_s2_camera_ids must contain unique camera IDs")
            if [len(values) for values in self.rank_s2_camera_ids] != [
                len(values) for values in self.rank_camera_ids
            ]:
                raise ValueError("S2 local camera counts must equal S1 on every rank")
            assigned_s2 = [
                camera_id
                for rank_camera_ids in self.rank_s2_camera_ids
                for camera_id in rank_camera_ids
            ]
            if sorted(assigned_s2) != sorted(self.global_s2_camera_ids):
                raise ValueError(
                    "rank_s2_camera_ids must assign every global S2 camera once"
                )

    @property
    def global_s1_camera_ids(self) -> List[int]:
        return self.global_camera_ids

    @property
    def rank_s1_camera_ids(self) -> List[List[int]]:
        return self.rank_camera_ids

    @property
    def global_union_active_blocks(self) -> List[int]:
        return self.global_active_blocks

    @property
    def rank_union_active_blocks(self) -> List[List[int]]:
        return self.rank_active_blocks

    def to_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["global_s1_camera_ids"] = list(self.global_s1_camera_ids)
        value["rank_s1_camera_ids"] = [
            list(values) for values in self.rank_s1_camera_ids
        ]
        value["global_union_active_blocks"] = list(
            self.global_union_active_blocks
        )
        value["rank_union_active_blocks"] = [
            list(values) for values in self.rank_union_active_blocks
        ]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DistributedBatchPlan":
        global_camera_source = (
            value["global_camera_ids"]
            if "global_camera_ids" in value
            else value["global_s1_camera_ids"]
        )
        rank_camera_source = (
            value["rank_camera_ids"]
            if "rank_camera_ids" in value
            else value["rank_s1_camera_ids"]
        )
        global_camera_ids = [
            int(v) for v in global_camera_source
        ]
        rank_camera_ids = [
            [int(v) for v in values]
            for values in rank_camera_source
        ]
        if "global_s1_camera_ids" in value and global_camera_ids != [
            int(v) for v in value["global_s1_camera_ids"]
        ]:
            raise ValueError("global_camera_ids and global_s1_camera_ids differ")
        if "rank_s1_camera_ids" in value and rank_camera_ids != [
            [int(v) for v in values] for values in value["rank_s1_camera_ids"]
        ]:
            raise ValueError("rank_camera_ids and rank_s1_camera_ids differ")

        rank_active_source = (
            value["rank_active_blocks"]
            if "rank_active_blocks" in value
            else value["rank_union_active_blocks"]
        )
        rank_active_blocks = [
            [int(v) for v in values] for values in rank_active_source
        ]
        rank_resident_blocks = [
            [int(v) for v in values] for values in value["rank_resident_blocks"]
        ]
        global_active_blocks = [
            int(v)
            for v in value.get(
                "global_active_blocks",
                value.get(
                    "global_union_active_blocks",
                    sorted({block for values in rank_active_blocks for block in values}),
                ),
            )
        ]
        world_size = len(rank_camera_ids)
        return cls(
            iteration=int(value["iteration"]),
            epoch=int(value["epoch"]),
            global_camera_ids=global_camera_ids,
            rank_camera_ids=rank_camera_ids,
            rank_resident_blocks=rank_resident_blocks,
            rank_active_blocks=rank_active_blocks,
            global_active_blocks=global_active_blocks,
            global_s2_camera_ids=[
                int(v) for v in value.get("global_s2_camera_ids", [])
            ],
            rank_s2_camera_ids=[
                [int(v) for v in values]
                for values in value.get(
                    "rank_s2_camera_ids", [[] for _ in range(world_size)]
                )
            ],
            rank_gradient_active_blocks=[
                [int(v) for v in values]
                for values in value.get(
                    "rank_gradient_active_blocks", rank_active_blocks
                )
            ],
            rank_curvature_active_blocks=[
                [int(v) for v in values]
                for values in value.get(
                    "rank_curvature_active_blocks", [[] for _ in range(world_size)]
                )
            ],
            global_gradient_active_blocks=[
                int(v)
                for v in value.get(
                    "global_gradient_active_blocks", global_active_blocks
                )
            ],
            global_curvature_active_blocks=[
                int(v) for v in value.get("global_curvature_active_blocks", [])
            ],
            rank_owner_active_rows=[
                int(v)
                for v in value.get(
                    "rank_owner_active_rows",
                    [0 for _ in range(world_size)],
                )
            ],
            rank_participation_rows=[
                int(v)
                for v in value.get(
                    "rank_participation_rows",
                    [1 for _ in range(world_size)],
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
            block_cull_backend=str(value.get("block_cull_backend", "cpu")),
            block_cull_gpu_kernel_ms=float(
                value.get("block_cull_gpu_kernel_ms", 0.0)
            ),
            block_cull_gpu_d2h_ms=float(value.get("block_cull_gpu_d2h_ms", 0.0)),
            block_cull_cache_hit_cameras=int(
                value.get("block_cull_cache_hit_cameras", 0)
            ),
            block_cull_gpu_cameras=int(value.get("block_cull_gpu_cameras", 0)),
            block_cull_output_blocks=int(value.get("block_cull_output_blocks", 0)),
            plan_ms=float(value.get("plan_ms", 0.0)),
            predicted_stream_in_blocks=int(
                value.get("predicted_stream_in_blocks", 0)
            ),
            prediction_missing_blocks=int(
                value.get("prediction_missing_blocks", 0)
            ),
            prediction_extra_blocks=int(
                value.get("prediction_extra_blocks", 0)
            ),
            prediction_replanned=int(value.get("prediction_replanned", 0)),
            prediction_repair_ms=float(value.get("prediction_repair_ms", 0.0)),
            prediction_exact_plan_ms=float(
                value.get("prediction_exact_plan_ms", 0.0)
            ),
        )


@dataclass(frozen=True)
class DistributedPlannerState:
    resident: Tuple[int, ...]
    active: Tuple[int, ...]
    recency: Dict[int, float]


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

    def snapshot_state(self) -> DistributedPlannerState:
        return DistributedPlannerState(
            resident=tuple(self._resident),
            active=tuple(self._active),
            recency=dict(self._recency),
        )

    def restore_state(self, state: DistributedPlannerState) -> None:
        self._resident = list(state.resident)
        self._active = list(state.active)
        self._recency = dict(state.recency)

    def preview(
        self,
        *,
        iteration: int,
        epoch: int,
        camera_ids: Sequence[int],
        camera_blocks: Mapping[int, Sequence[int]],
        curvature_camera_ids: Optional[Sequence[int]] = None,
        curvature_camera_blocks: Optional[Mapping[int, Sequence[int]]] = None,
    ) -> Tuple[DistributedBatchPlan, DistributedPlannerState]:
        baseline = self.snapshot_state()
        try:
            plan = self.plan(
                iteration=iteration,
                epoch=epoch,
                camera_ids=camera_ids,
                camera_blocks=camera_blocks,
                curvature_camera_ids=curvature_camera_ids,
                curvature_camera_blocks=curvature_camera_blocks,
            )
            predicted_state = self.snapshot_state()
        finally:
            self.restore_state(baseline)
        return plan, predicted_state

    def plan(
        self,
        *,
        iteration: int,
        epoch: int,
        camera_ids: Sequence[int],
        camera_blocks: Mapping[int, Sequence[int]],
        curvature_camera_ids: Optional[Sequence[int]] = None,
        curvature_camera_blocks: Optional[Mapping[int, Sequence[int]]] = None,
    ) -> DistributedBatchPlan:
        global_camera_ids = [int(camera_id) for camera_id in camera_ids]
        if len(global_camera_ids) % self.world_size != 0:
            raise ValueError("Global camera count must be divisible by world_size")
        if len(set(global_camera_ids)) != len(global_camera_ids):
            raise ValueError("Distributed camera batches must contain unique camera IDs")

        def assign(
            camera_batch: Sequence[int],
            block_map: Mapping[int, Sequence[int]],
        ) -> List[List[int]]:
            camera_batch = list(camera_batch)
            if self.camera_assignment == "equal":
                local_count = len(camera_batch) // self.world_size
                return [
                    camera_batch[rank * local_count:(rank + 1) * local_count]
                    for rank in range(self.world_size)
                ]
            if self.camera_assignment == "gaussian_balanced":
                return assign_cameras_balanced(
                    camera_ids=camera_batch,
                    camera_blocks=block_map,
                    block_rows=self._block_rows,
                    world_size=self.world_size,
                )
            raise ValueError(f"Unsupported camera assignment: {self.camera_assignment}")

        rank_camera_ids = assign(global_camera_ids, camera_blocks)

        if curvature_camera_ids is None:
            if curvature_camera_blocks is not None:
                raise ValueError(
                    "curvature_camera_blocks requires curvature_camera_ids"
                )
            global_s2_camera_ids: List[int] = []
            rank_s2_camera_ids = [[] for _ in range(self.world_size)]
            curvature_camera_blocks = {}
        else:
            global_s2_camera_ids = [int(value) for value in curvature_camera_ids]
            if len(global_s2_camera_ids) != len(global_camera_ids):
                raise ValueError("S2 global camera count must equal S1")
            if len(set(global_s2_camera_ids)) != len(global_s2_camera_ids):
                raise ValueError("S2 camera batches must contain unique camera IDs")
            if curvature_camera_blocks is None:
                raise ValueError(
                    "curvature_camera_blocks is required for an S2 batch"
                )
            rank_s2_camera_ids = assign(
                global_s2_camera_ids, curvature_camera_blocks
            )
            if [len(values) for values in rank_s2_camera_ids] != [
                len(values) for values in rank_camera_ids
            ]:
                raise RuntimeError("S2 local camera counts must equal S1 on every rank")

        if self.camera_assignment == "equal":
            local_count = len(global_camera_ids) // self.world_size
            if any(len(values) != local_count for values in rank_camera_ids):
                raise RuntimeError("S1 camera assignment did not fill every rank")

        normalized_gradient_camera_blocks = {
            int(camera_id): sorted(
                {
                    int(block_id)
                    for block_id in camera_blocks.get(int(camera_id), [])
                    if 0 <= int(block_id) < len(self.block_owner)
                }
            )
            for camera_id in global_camera_ids
        }
        normalized_curvature_camera_blocks = {
            int(camera_id): sorted(
                {
                    int(block_id)
                    for block_id in curvature_camera_blocks.get(int(camera_id), [])
                    if 0 <= int(block_id) < len(self.block_owner)
                }
            )
            for camera_id in global_s2_camera_ids
        }
        global_gradient_active = sorted(
            {
                block_id
                for blocks in normalized_gradient_camera_blocks.values()
                for block_id in blocks
            }
        )
        global_curvature_active = sorted(
            {
                block_id
                for blocks in normalized_curvature_camera_blocks.values()
                for block_id in blocks
            }
        )
        global_active = sorted(
            set(global_gradient_active).union(global_curvature_active)
        )
        gradient_camera_set = set(global_camera_ids)
        union_camera_ids = list(global_camera_ids)
        union_camera_ids.extend(
            camera_id
            for camera_id in global_s2_camera_ids
            if camera_id not in gradient_camera_set
        )
        normalized_union_camera_blocks = {
            camera_id: sorted(
                set(normalized_gradient_camera_blocks.get(camera_id, [])).union(
                    normalized_curvature_camera_blocks.get(camera_id, [])
                )
            )
            for camera_id in union_camera_ids
        }
        transition = compute_topc_resident_transition(
            current_active_blocks=self._active,
            next_active_blocks=global_active,
            current_resident_blocks=self._resident,
            next_camera_ids=union_camera_ids,
            next_camera_blocks=normalized_union_camera_blocks,
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
        rank_gradient_active_blocks = [
            [
                block_id
                for block_id in global_gradient_active
                if self.block_owner[block_id] == rank
            ]
            for rank in range(self.world_size)
        ]
        rank_curvature_active_blocks = [
            [
                block_id
                for block_id in global_curvature_active
                if self.block_owner[block_id] == rank
            ]
            for rank in range(self.world_size)
        ]
        rank_resident_blocks = [
            [block_id for block_id in global_resident if self.block_owner[block_id] == rank]
            for rank in range(self.world_size)
        ]

        resident_sets = [set(values) for values in rank_resident_blocks]
        rank_owner_active_rows = [
            sum(
                self._block_rows[block_id]
                for block_id in rank_active_blocks[rank]
                if block_id in resident_sets[rank]
            )
            for rank in range(self.world_size)
        ]
        rank_participation_rows = [
            max(1, row_count) for row_count in rank_owner_active_rows
        ]
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
            global_s2_camera_ids=global_s2_camera_ids,
            rank_s2_camera_ids=rank_s2_camera_ids,
            rank_gradient_active_blocks=rank_gradient_active_blocks,
            rank_curvature_active_blocks=rank_curvature_active_blocks,
            global_gradient_active_blocks=global_gradient_active,
            global_curvature_active_blocks=global_curvature_active,
            rank_owner_active_rows=rank_owner_active_rows,
            rank_participation_rows=rank_participation_rows,
            stream_in_blocks=list(transition.stream_in_blocks),
            evict_blocks=list(transition.evict_blocks),
        )
