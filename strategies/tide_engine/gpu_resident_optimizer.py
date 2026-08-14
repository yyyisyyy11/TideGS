import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class SophiaTRPreparedStep:
    """Validated, unpublished Sophia/TR update owned by one optimizer."""

    optimizer: Any
    gaussians: Any
    iteration: int
    optimizer_step: int
    curvature_due: bool
    curvature_update_index: int
    trust_epsilon: float
    gradient_local_rows: torch.Tensor
    gradient_state_rows: torch.Tensor
    gradient_components: Dict[str, torch.Tensor]
    gradient_curvature_components: Optional[Dict[str, torch.Tensor]]
    gradient_curvature_mask: Optional[torch.Tensor]
    curvature_only_local_rows: torch.Tensor
    curvature_only_state_rows: torch.Tensor
    curvature_only_components: Optional[Dict[str, torch.Tensor]]
    union_touched_blocks: List[int]
    union_touched_slots: List[int]
    union_touched_counts: List[int]
    gradient_touched_blocks: List[int]
    gradient_touched_slots: List[int]
    gradient_touched_counts: List[int]
    curvature_touched_blocks: List[int]
    curvature_row_count: int
    beta1: float
    beta2: float
    eps: float
    gamma: float
    batch_scale: float
    quat_norm_tr: float
    active_sh_degree: int
    clipped_values: int
    rows_skipped_without_curvature: int
    status: str = "prepared"

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - torch CUDA wheels normally provide Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _fused_resident_adam_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        local_rows_ptr,
        state_rows_ptr,
        state_slots_ptr,
        bias_correction1_ptr,
        denom_scale_ptr,
        n_rows,
        learning_rate,
        beta1,
        beta2,
        eps,
        batch_size,
        WIDTH: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        total = n_rows * WIDTH
        mask = offsets < total
        row = offsets // WIDTH
        col = offsets - row * WIDTH

        local_row = tl.load(local_rows_ptr + row, mask=mask, other=0)
        state_row = tl.load(state_rows_ptr + row, mask=mask, other=0)
        state_slot = tl.load(state_slots_ptr + row, mask=mask, other=0)

        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = grad / batch_size

        state_offset = state_row * WIDTH + col
        param_offset = local_row * WIDTH + col
        exp_avg = tl.load(exp_avg_ptr + state_offset, mask=mask, other=0.0).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + state_offset, mask=mask, other=0.0).to(tl.float32)

        exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad

        bias_correction1 = tl.load(
            bias_correction1_ptr + state_slot, mask=mask, other=1.0
        ).to(tl.float32)
        denom_scale = tl.load(
            denom_scale_ptr + state_slot, mask=mask, other=1.0
        ).to(tl.float32)
        denom = tl.sqrt(exp_avg_sq) / denom_scale + eps

        param = tl.load(param_ptr + param_offset, mask=mask, other=0.0).to(tl.float32)
        param -= (learning_rate / bias_correction1) * exp_avg / denom

        tl.store(exp_avg_ptr + state_offset, exp_avg, mask=mask)
        tl.store(exp_avg_sq_ptr + state_offset, exp_avg_sq, mask=mask)
        tl.store(param_ptr + param_offset, param, mask=mask)


class GPUResidentAdam:
    """Adam with moments stored only for the current resident block set.

    Blocks keep stable state slots while resident. Eviction releases the slot and
    discards its moments; a later admission starts from zero. The update itself is
    vectorized over every touched row, avoiding the former block-by-component
    Python loop and its hundreds of thousands of tiny CUDA kernels.
    """

    COMPONENT_SPECS = (
        ("xyz", "_xyz", 3, 0),
        ("opacity", "_opacity", 1, 1),
        ("scaling", "_scaling", 3, 2),
        ("rotation", "_rotation", 4, 3),
        ("features_dc", "_features_dc", 3, 4),
        ("features_rest", "_features_rest", 45, 5),
    )
    TOTAL_WIDTH = sum(spec[2] for spec in COMPONENT_SPECS)

    def __init__(
        self,
        batch_size: int = 1,
        block_size: int = 4096,
        capacity_blocks: int = 0,
        device: str = "cuda",
    ):
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.capacity_blocks = max(0, int(capacity_blocks))
        self.device = torch.device(device)
        if (
            self.device.type == "cuda"
            and self.device.index is None
            and torch.cuda.is_available()
        ):
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.state_mode = "resident_blocks"

        self._resident_blocks = set()
        self._resident_streaks: Dict[int, int] = {}
        self._completed_streak_total = 0
        self._completed_streak_count = 0

        self._allocated_slots = 0
        self._block_to_slot: Dict[int, int] = {}
        self._slot_to_block: List[Optional[int]] = []
        self._free_slots: List[int] = []
        self._slot_steps: List[int] = []
        self._slot_initialized: List[bool] = []
        self._slot_row_counts: List[int] = []
        self._exp_avg: Dict[str, torch.Tensor] = {}
        self._exp_avg_sq: Dict[str, torch.Tensor] = {}
        self._bias_correction1: Optional[torch.Tensor] = None
        self._denom_scale: Optional[torch.Tensor] = None

        self._logical_state_blocks = 0
        self._logical_state_bytes = 0
        self._residency_corrupted = False
        self._stats = {
            "state_mode": "resident_blocks",
            "resident_target_blocks": 0,
            "resident_state_blocks": 0,
            "resident_state_bytes": 0,
            "allocated_state_bytes": 0,
            "cold_restarts": 0,
            "state_evictions": 0,
            "optimizer_rows_touched_total": 0,
            "cold_restarted_rows_touched_total": 0,
            "mean_resident_streak": 0.0,
        }

    def _physical_state_bytes(self) -> int:
        moment_bytes = self._allocated_slots * self.block_size * self.TOTAL_WIDTH * 2 * 4
        return int(moment_bytes)

    def _logical_state_bytes_per_row(self) -> int:
        return int(self.TOTAL_WIDTH * 2 * 4)

    def _update_state_stats(self) -> None:
        self._stats["resident_target_blocks"] = len(self._resident_blocks)
        self._stats["resident_state_blocks"] = int(self._logical_state_blocks)
        self._stats["resident_state_bytes"] = int(self._logical_state_bytes)
        self._stats["allocated_state_bytes"] = self._physical_state_bytes()
        streak_total = self._completed_streak_total + sum(self._resident_streaks.values())
        streak_count = self._completed_streak_count + len(self._resident_streaks)
        self._stats["mean_resident_streak"] = float(streak_total) / max(1, streak_count)

    def _normalize_columns_lr(self, columns_lr) -> Dict[str, float]:
        if columns_lr is None:
            raise RuntimeError("[GPUResidentAdam] optimizer.columns_lr is required")
        if torch.is_tensor(columns_lr):
            cols = columns_lr.detach().to(device="cpu", dtype=torch.float32)
        else:
            cols = torch.tensor(columns_lr, dtype=torch.float32)
        cols = cols.flatten().contiguous()
        if cols.numel() == 6:
            grouped = cols
        elif cols.numel() == 59:
            grouped = cols[torch.tensor([0, 3, 4, 7, 11, 14])]
        else:
            raise RuntimeError(
                f"[GPUResidentAdam] Unexpected columns_lr width={cols.numel()}; "
                "expected 6 grouped or 59 expanded entries"
            )
        return {
            name: float(grouped[group_idx].item())
            for name, _, _, group_idx in self.COMPONENT_SPECS
        }

    def _ensure_storage(self, required_slots: int) -> None:
        if required_slots <= self._allocated_slots:
            return
        target = max(required_slots, self.capacity_blocks, max(1, self._allocated_slots * 2))
        old_slots = self._allocated_slots
        old_rows = old_slots * self.block_size
        new_rows = target * self.block_size

        new_exp_avg: Dict[str, torch.Tensor] = {}
        new_exp_avg_sq: Dict[str, torch.Tensor] = {}
        for name, _, width, _ in self.COMPONENT_SPECS:
            avg = torch.empty((new_rows, width), dtype=torch.float32, device=self.device)
            avg_sq = torch.empty((new_rows, width), dtype=torch.float32, device=self.device)
            if old_rows:
                avg[:old_rows].copy_(self._exp_avg[name])
                avg_sq[:old_rows].copy_(self._exp_avg_sq[name])
            new_exp_avg[name] = avg
            new_exp_avg_sq[name] = avg_sq

        bias = torch.empty((target,), dtype=torch.float32, device=self.device)
        denom = torch.empty((target,), dtype=torch.float32, device=self.device)
        if old_slots:
            bias[:old_slots].copy_(self._bias_correction1)
            denom[:old_slots].copy_(self._denom_scale)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        self._exp_avg = new_exp_avg
        self._exp_avg_sq = new_exp_avg_sq
        self._bias_correction1 = bias
        self._denom_scale = denom
        self._slot_to_block.extend([None] * (target - old_slots))
        self._slot_steps.extend([0] * (target - old_slots))
        self._slot_initialized.extend([False] * (target - old_slots))
        self._slot_row_counts.extend([0] * (target - old_slots))
        self._free_slots.extend(range(target - 1, old_slots - 1, -1))
        self._allocated_slots = target

    def _clear_slots(self, slots: List[int]) -> None:
        if not slots:
            return
        slot_ids = torch.tensor(slots, dtype=torch.long, device=self.device)
        for name, _, width, _ in self.COMPONENT_SPECS:
            self._exp_avg[name].view(
                self._allocated_slots, self.block_size, width
            ).index_fill_(0, slot_ids, 0.0)
            self._exp_avg_sq[name].view(
                self._allocated_slots, self.block_size, width
            ).index_fill_(0, slot_ids, 0.0)

    def set_resident_blocks(self, block_ids: List[int]) -> None:
        if self._residency_corrupted:
            raise RuntimeError(
                "[GPUResidentAdam] optimizer is unusable after a failed resident "
                "slot clear; recreate it before changing the resident set"
            )
        next_resident = {int(block_id) for block_id in block_ids}
        previous_resident = set(self._resident_blocks)
        evicted = previous_resident - next_resident
        kept = previous_resident & next_resident
        incoming = next_resident - previous_resident

        self._ensure_storage(len(next_resident))

        next_resident_streaks = dict(self._resident_streaks)
        next_completed_streak_total = self._completed_streak_total
        next_completed_streak_count = self._completed_streak_count
        next_block_to_slot = dict(self._block_to_slot)
        next_slot_to_block = list(self._slot_to_block)
        next_free_slots = list(self._free_slots)
        next_slot_steps = list(self._slot_steps)
        next_slot_initialized = list(self._slot_initialized)
        next_slot_row_counts = list(self._slot_row_counts)
        next_logical_state_blocks = self._logical_state_blocks
        next_logical_state_bytes = self._logical_state_bytes
        next_stats = dict(self._stats)

        for block_id in evicted:
            streak = int(next_resident_streaks.pop(block_id, 0))
            if streak > 0:
                next_completed_streak_total += streak
                next_completed_streak_count += 1
            slot = next_block_to_slot.pop(block_id)
            if next_slot_initialized[slot]:
                next_stats["state_evictions"] += 1
                next_logical_state_blocks -= 1
                next_logical_state_bytes -= (
                    next_slot_row_counts[slot] * self._logical_state_bytes_per_row()
                )
            next_slot_to_block[slot] = None
            next_slot_steps[slot] = 0
            next_slot_initialized[slot] = False
            next_slot_row_counts[slot] = 0
            next_free_slots.append(slot)

        for block_id in kept:
            next_resident_streaks[block_id] = next_resident_streaks.get(block_id, 0) + 1
        for block_id in incoming:
            next_resident_streaks[block_id] = 1

        incoming_slots: List[int] = []
        for block_id in sorted(incoming):
            if not next_free_slots:
                raise RuntimeError("[GPUResidentAdam] internal slot allocation invariant failed")
            slot = next_free_slots.pop()
            next_block_to_slot[block_id] = slot
            next_slot_to_block[slot] = block_id
            next_slot_steps[slot] = 0
            next_slot_initialized[slot] = False
            next_slot_row_counts[slot] = 0
            incoming_slots.append(slot)

        try:
            self._clear_slots(incoming_slots)
            if incoming_slots and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except BaseException as error:
            self._residency_corrupted = True
            if hasattr(error, "add_note"):
                error.add_note(
                    "[GPUResidentAdam] resident slot clear may be partial; "
                    "optimizer is poisoned and must be recreated"
                )
            raise

        self._resident_streaks = next_resident_streaks
        self._completed_streak_total = next_completed_streak_total
        self._completed_streak_count = next_completed_streak_count
        self._block_to_slot = next_block_to_slot
        self._slot_to_block = next_slot_to_block
        self._free_slots = next_free_slots
        self._slot_steps = next_slot_steps
        self._slot_initialized = next_slot_initialized
        self._slot_row_counts = next_slot_row_counts
        self._logical_state_blocks = next_logical_state_blocks
        self._logical_state_bytes = next_logical_state_bytes
        self._stats = next_stats
        self._resident_blocks = next_resident
        self._update_state_stats()

    def _working_set_state_rows(
        self,
        manager,
        local_ids: torch.Tensor,
        *,
        assume_sorted_unique: bool = False,
        validate_mapping: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int], List[int]]:
        if local_ids.ndim != 1:
            raise ValueError("[GPUResidentAdam] sparse ids must be one-dimensional")
        if local_ids.numel() == 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty, [], [], []
        layout = []
        for block_id in manager.loaded_blocks:
            block_id = int(block_id)
            block_slice = manager.block_to_gpu_slice.get(block_id)
            if block_slice is None:
                continue
            slot = self._block_to_slot.get(block_id)
            if slot is None:
                raise RuntimeError(
                    f"[GPUResidentAdam] loaded block {block_id} has no resident state slot"
                )
            layout.append((int(block_slice.start), int(block_slice.stop), block_id, slot))
        layout.sort(key=lambda item: item[0])
        if not layout:
            raise RuntimeError("[GPUResidentAdam] working set contains no resident blocks")

        if not assume_sorted_unique and local_ids.numel() > 1:
            sorted_unique = (local_ids[1:] > local_ids[:-1]).all()
            if not bool(sorted_unique):
                raise ValueError(
                    "[GPUResidentAdam] sparse ids must be sorted and unique"
                )

        starts = torch.tensor(
            [item[0] for item in layout], dtype=torch.long, device=self.device
        )
        ends = torch.tensor(
            [item[1] for item in layout], dtype=torch.long, device=self.device
        )
        slots = torch.tensor(
            [item[3] for item in layout], dtype=torch.long, device=self.device
        )
        block_positions = torch.searchsorted(ends, local_ids, right=True)
        safe_positions = block_positions.clamp_max(len(layout) - 1)
        row_in_block = local_ids - starts[safe_positions]
        if validate_mapping:
            contained = (block_positions < len(layout)) & (row_in_block >= 0)
            contained &= local_ids < ends[safe_positions]
            if not bool(contained.all()):
                invalid = local_ids[~contained].detach().cpu().tolist()
                raise RuntimeError(
                    "[GPUResidentAdam] sparse ids are not contained in a loaded "
                    f"resident block: {invalid[:8]}"
                )
        state_slots = slots[safe_positions]
        state_rows = state_slots * self.block_size + row_in_block

        touched_positions, touched_counts = torch.unique_consecutive(
            safe_positions, return_counts=True
        )
        positions_cpu = touched_positions.detach().cpu().tolist()
        counts_cpu = touched_counts.detach().cpu().tolist()
        touched_blocks = [layout[position][2] for position in positions_cpu]
        touched_slots = [layout[position][3] for position in positions_cpu]
        return state_rows, state_slots, touched_blocks, touched_slots, counts_cpu

    def _update_touched_slot_metadata(
        self,
        manager,
        touched_blocks: List[int],
        touched_slots: List[int],
        touched_counts: List[int],
        beta1: float,
        beta2: float,
    ) -> int:
        cold_rows = 0
        bias_values = []
        denom_values = []
        for block_id, slot, count in zip(touched_blocks, touched_slots, touched_counts):
            if not self._slot_initialized[slot]:
                self._slot_initialized[slot] = True
                block_slice = manager.block_to_gpu_slice[block_id]
                row_count = int(block_slice.stop - block_slice.start)
                self._slot_row_counts[slot] = row_count
                self._logical_state_blocks += 1
                self._logical_state_bytes += row_count * self._logical_state_bytes_per_row()
                self._stats["cold_restarts"] += 1
                self._stats["cold_restarted_rows_touched_total"] += int(count)
                cold_rows += int(count)
            self._slot_steps[slot] += 1
            step = self._slot_steps[slot]
            bias_values.append(1.0 - beta1**step)
            denom_values.append(math.sqrt(1.0 - beta2**step))

        slot_ids = torch.tensor(touched_slots, dtype=torch.long, device=self.device)
        self._bias_correction1.index_copy_(
            0, slot_ids, torch.tensor(bias_values, dtype=torch.float32, device=self.device)
        )
        self._denom_scale.index_copy_(
            0, slot_ids, torch.tensor(denom_values, dtype=torch.float32, device=self.device)
        )
        return cold_rows

    def _vectorized_component_step(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        local_rows: torch.Tensor,
        state_rows: torch.Tensor,
        state_slots: torch.Tensor,
        learning_rate: float,
        beta1: float,
        beta2: float,
        eps: float,
    ) -> None:
        avg = exp_avg.index_select(0, state_rows)
        avg_sq = exp_avg_sq.index_select(0, state_rows)
        scaled_grad = grad / float(self.batch_size)
        avg.mul_(beta1).add_(scaled_grad, alpha=1.0 - beta1)
        avg_sq.mul_(beta2).addcmul_(scaled_grad, scaled_grad, value=1.0 - beta2)
        exp_avg.index_copy_(0, state_rows, avg)
        exp_avg_sq.index_copy_(0, state_rows, avg_sq)
        bias = self._bias_correction1.index_select(0, state_slots).unsqueeze(1)
        denom_scale = self._denom_scale.index_select(0, state_slots).unsqueeze(1)
        update = (avg / (avg_sq.sqrt() / denom_scale + eps)) * (learning_rate / bias)
        values = param.index_select(0, local_rows) - update
        param.index_copy_(0, local_rows, values)

    def _component_step(
        self,
        *,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        local_rows: torch.Tensor,
        state_rows: torch.Tensor,
        state_slots: torch.Tensor,
        learning_rate: float,
        beta1: float,
        beta2: float,
        eps: float,
        width: int,
    ) -> None:
        if triton is None or self.device.type != "cuda":
            self._vectorized_component_step(
                param,
                grad,
                exp_avg,
                exp_avg_sq,
                local_rows,
                state_rows,
                state_slots,
                learning_rate,
                beta1,
                beta2,
                eps,
            )
            return
        n_rows = int(local_rows.numel())
        grid = (triton.cdiv(n_rows * width, 256),)
        _fused_resident_adam_kernel[grid](
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            local_rows,
            state_rows,
            state_slots,
            self._bias_correction1,
            self._denom_scale,
            n_rows,
            learning_rate,
            beta1,
            beta2,
            eps,
            float(self.batch_size),
            WIDTH=width,
            BLOCK_SIZE=256,
            num_warps=4,
        )

    def step(
        self,
        iteration: int,
        gaussians,
        sparse_grad_local_ids: torch.Tensor,
        sparse_grad_components: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        if sparse_grad_local_ids is None or sparse_grad_components is None:
            return {
                "updated_blocks": 0,
                "updated_block_ids": [],
                "touched_rows": 0,
                "cold_rows": 0,
            }
        if sparse_grad_local_ids.numel() == 0:
            return {
                "updated_blocks": 0,
                "updated_block_ids": [],
                "touched_rows": 0,
                "cold_rows": 0,
            }
        if self._residency_corrupted:
            raise RuntimeError(
                "[GPUResidentAdam] optimizer is unusable after a failed resident "
                "slot clear; recreate it before processing another non-empty step"
            )

        manager = getattr(gaussians, "gpu_working_set_manager", None)
        if manager is None or manager.local_to_global_idx is None:
            raise RuntimeError(
                "[GPUResidentAdam] gpu_working_set_manager with local_to_global_idx is required"
            )
        local_rows = sparse_grad_local_ids.to(
            device=self.device, dtype=torch.long
        ).contiguous()
        state_rows, state_slots, touched_blocks, touched_slots, touched_counts = (
            self._working_set_state_rows(manager, local_rows)
        )

        component_lrs = self._normalize_columns_lr(
            getattr(gaussians.optimizer, "columns_lr", None)
        )
        beta1, beta2 = gaussians.optimizer.param_groups[0]["betas"]
        beta1 = float(beta1)
        beta2 = float(beta2)
        eps = float(gaussians.optimizer.param_groups[0]["eps"])
        cold_rows = self._update_touched_slot_metadata(
            manager,
            touched_blocks,
            touched_slots,
            touched_counts,
            beta1,
            beta2,
        )

        param_views = {
            name: getattr(gaussians, attr_name).data
            for name, attr_name, _, _ in self.COMPONENT_SPECS
        }
        for name, _, width, _ in self.COMPONENT_SPECS:
            if name not in sparse_grad_components:
                raise KeyError(
                    f"[GPUResidentAdam] Missing sparse_grad_components[{name!r}] "
                    f"at iter={iteration}"
                )
            param = param_views[name]
            grad = sparse_grad_components[name]
            if not param.is_contiguous() or not grad.is_contiguous():
                raise RuntimeError(
                    f"[GPUResidentAdam] {name} parameter and gradient must be contiguous"
                )
            self._component_step(
                param=param,
                grad=grad,
                exp_avg=self._exp_avg[name],
                exp_avg_sq=self._exp_avg_sq[name],
                local_rows=local_rows,
                state_rows=state_rows,
                state_slots=state_slots,
                learning_rate=component_lrs[name],
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                width=width,
            )

        touched_rows = int(local_rows.numel())
        self._stats["optimizer_rows_touched_total"] += touched_rows
        self._update_state_stats()
        return {
            "updated_blocks": len(touched_blocks),
            "updated_block_ids": list(touched_blocks),
            "touched_rows": touched_rows,
            "cold_rows": int(cold_rows),
        }

    def get_stats(self) -> Dict[str, Any]:
        self._update_state_stats()
        return dict(self._stats)


class GPUResidentSophiaTR(GPUResidentAdam):
    """Resident-block implementation of the 3DGS2-TR optimizer.

    The two FP32 state tensors store the gradient EMA and the Hutchinson
    Gauss-Newton diagonal EMA. The paper uses the raw EMAs without Adam-style
    bias correction. TideGS still discards both tensors on resident-block
    eviction; a re-admitted row is therefore held fixed until it receives a
    fresh curvature sample.
    """

    TRANSACTION_CHUNK_ROWS = 262_144

    def __init__(
        self,
        batch_size: int = 1,
        block_size: int = 4096,
        capacity_blocks: int = 0,
        device: str = "cuda",
    ):
        super().__init__(
            batch_size=batch_size,
            block_size=block_size,
            capacity_blocks=capacity_blocks,
            device=device,
        )
        self.state_mode = "resident_blocks"
        self._curvature_initialized: Optional[torch.Tensor] = None
        self._last_gradient_step: Optional[torch.Tensor] = None
        self._last_curvature_update: Optional[torch.Tensor] = None
        self._transaction_corrupted = False
        self._pending_transaction: Optional[SophiaTRPreparedStep] = None
        self._stats.update(
            {
                "algorithm": "3dgs2_tr",
                "state_mode": self.state_mode,
                "curvature_updates": 0,
                "curvature_rows_touched_total": 0,
                "rows_skipped_without_curvature_total": 0,
                "trust_region_clipped_values_total": 0,
                "last_tr_epsilon": 0.0,
                "optimizer_step": 0,
            }
        )

    def set_resident_blocks(self, block_ids: List[int]) -> None:
        if self._pending_transaction is not None:
            raise RuntimeError(
                "[GPUResidentSophiaTR] cannot change residency while a prepared "
                "step is pending"
            )
        super().set_resident_blocks(block_ids)

    def _physical_state_bytes(self) -> int:
        moment_bytes = super()._physical_state_bytes()
        rows = self._allocated_slots * self.block_size
        return int(moment_bytes + rows * (1 + 2 * 8))

    def _logical_state_bytes_per_row(self) -> int:
        return super()._logical_state_bytes_per_row() + 1 + 2 * 8

    def _ensure_storage(self, required_slots: int) -> None:
        if required_slots <= self._allocated_slots:
            return
        target = max(
            required_slots,
            self.capacity_blocks,
            max(1, self._allocated_slots * 2),
        )
        old_slots = self._allocated_slots
        old_rows = self._allocated_slots * self.block_size
        new_rows = target * self.block_size
        if old_rows and (
            self._curvature_initialized is None
            or self._last_gradient_step is None
            or self._last_curvature_update is None
        ):
            raise RuntimeError(
                "[GPUResidentSophiaTR] existing row metadata is incomplete"
            )

        # Build every tensor and Python container off to the side. Publishing
        # parent moment storage before these row tensors exist would make a
        # recoverable CUDA OOM leave the optimizer permanently half-expanded.
        new_exp_avg: Dict[str, torch.Tensor] = {}
        new_exp_avg_sq: Dict[str, torch.Tensor] = {}
        for name, _, width, _ in self.COMPONENT_SPECS:
            avg = torch.empty(
                (new_rows, width), dtype=torch.float32, device=self.device
            )
            avg_sq = torch.empty(
                (new_rows, width), dtype=torch.float32, device=self.device
            )
            if old_rows:
                avg[:old_rows].copy_(self._exp_avg[name])
                avg_sq[:old_rows].copy_(self._exp_avg_sq[name])
            new_exp_avg[name] = avg
            new_exp_avg_sq[name] = avg_sq

        bias = torch.empty((target,), dtype=torch.float32, device=self.device)
        denom = torch.empty((target,), dtype=torch.float32, device=self.device)
        if old_slots:
            bias[:old_slots].copy_(self._bias_correction1)
            denom[:old_slots].copy_(self._denom_scale)

        initialized = torch.zeros(
            (new_rows,), dtype=torch.bool, device=self.device
        )
        last_gradient_step = torch.zeros(
            (new_rows,), dtype=torch.int64, device=self.device
        )
        last_curvature_update = torch.zeros(
            (new_rows,), dtype=torch.int64, device=self.device
        )
        if old_rows:
            initialized[:old_rows].copy_(self._curvature_initialized)
            last_gradient_step[:old_rows].copy_(self._last_gradient_step)
            last_curvature_update[:old_rows].copy_(
                self._last_curvature_update
            )

        slot_to_block = list(self._slot_to_block)
        slot_to_block.extend([None] * (target - old_slots))
        slot_steps = list(self._slot_steps)
        slot_steps.extend([0] * (target - old_slots))
        slot_initialized = list(self._slot_initialized)
        slot_initialized.extend([False] * (target - old_slots))
        slot_row_counts = list(self._slot_row_counts)
        slot_row_counts.extend([0] * (target - old_slots))
        free_slots = list(self._free_slots)
        free_slots.extend(range(target - 1, old_slots - 1, -1))

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        self._exp_avg = new_exp_avg
        self._exp_avg_sq = new_exp_avg_sq
        self._bias_correction1 = bias
        self._denom_scale = denom
        self._curvature_initialized = initialized
        self._last_gradient_step = last_gradient_step
        self._last_curvature_update = last_curvature_update
        self._slot_to_block = slot_to_block
        self._slot_steps = slot_steps
        self._slot_initialized = slot_initialized
        self._slot_row_counts = slot_row_counts
        self._free_slots = free_slots
        self._allocated_slots = target

    def _clear_slots(self, slots: List[int]) -> None:
        super()._clear_slots(slots)
        if not slots or self._curvature_initialized is None:
            return
        slot_ids = torch.tensor(slots, dtype=torch.long, device=self.device)
        self._curvature_initialized.view(
            self._allocated_slots, self.block_size
        ).index_fill_(0, slot_ids, False)
        self._last_gradient_step.view(
            self._allocated_slots, self.block_size
        ).index_fill_(0, slot_ids, 0)
        self._last_curvature_update.view(
            self._allocated_slots, self.block_size
        ).index_fill_(0, slot_ids, 0)

    @staticmethod
    def _empty_step_stats() -> Dict[str, Any]:
        return {
            "updated_blocks": 0,
            "updated_block_ids": [],
            "touched_rows": 0,
            "cold_rows": 0,
            "curvature_due": False,
            "curvature_rows": 0,
            "clipped_values": 0,
            "rows_skipped_without_curvature": 0,
        }

    def _validate_component_dict(
        self,
        components: Dict[str, torch.Tensor],
        *,
        name: str,
        n_rows: int,
        require_non_negative: bool = False,
        check_finite: bool = True,
    ) -> None:
        finite_flags = []
        non_negative_flags = []
        for component_name, _, width, _ in self.COMPONENT_SPECS:
            if component_name not in components:
                raise KeyError(
                    f"[GPUResidentSophiaTR] Missing {name}[{component_name!r}]"
                )
            tensor = components[component_name]
            if tensor.shape != (n_rows, width):
                raise ValueError(
                    f"[GPUResidentSophiaTR] {name}[{component_name!r}] has "
                    f"shape={tuple(tensor.shape)}; expected {(n_rows, width)}"
                )
            if tensor.device != self.device:
                raise ValueError(
                    f"[GPUResidentSophiaTR] {name}[{component_name!r}] is on "
                    f"{tensor.device}; expected {self.device}"
                )
            if tensor.dtype != torch.float32:
                raise ValueError(
                    f"[GPUResidentSophiaTR] {name}[{component_name!r}] has "
                    f"dtype={tensor.dtype}; expected torch.float32"
                )
            if not tensor.is_contiguous():
                raise ValueError(
                    f"[GPUResidentSophiaTR] {name}[{component_name!r}] must be contiguous"
                )
            if check_finite:
                finite_flags.append(torch.isfinite(tensor).all())
            if require_non_negative:
                non_negative_flags.append((tensor >= 0.0).all())

        validity_flags = finite_flags + non_negative_flags
        if validity_flags and not bool(torch.stack(validity_flags).all()):
            if finite_flags and not bool(torch.stack(finite_flags).all()):
                raise FloatingPointError(
                    f"[GPUResidentSophiaTR] {name} contains NaN/Inf"
                )
            raise ValueError(
                "[GPUResidentSophiaTR] residual_vjp curvature must be non-negative"
            )

    def _synchronize_transaction_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _snapshot_transaction_python_state(self) -> Dict[str, Any]:
        return {
            "slot_steps": list(self._slot_steps),
            "slot_initialized": list(self._slot_initialized),
            "slot_row_counts": list(self._slot_row_counts),
            "logical_state_blocks": int(self._logical_state_blocks),
            "logical_state_bytes": int(self._logical_state_bytes),
            "stats": dict(self._stats),
        }

    def _restore_transaction_python_state(
        self, snapshot: Dict[str, Any]
    ) -> None:
        self._slot_steps[:] = snapshot["slot_steps"]
        self._slot_initialized[:] = snapshot["slot_initialized"]
        self._slot_row_counts[:] = snapshot["slot_row_counts"]
        self._logical_state_blocks = snapshot["logical_state_blocks"]
        self._logical_state_bytes = snapshot["logical_state_bytes"]
        self._stats.clear()
        self._stats.update(snapshot["stats"])

    def _build_transaction_chunk(
        self,
        *,
        param_views: Dict[str, torch.Tensor],
        sparse_grad_components: Dict[str, torch.Tensor],
        sparse_curvature_components: Optional[Dict[str, torch.Tensor]],
        curvature_update_mask: Optional[torch.Tensor],
        chunk: slice,
        local_rows: torch.Tensor,
        state_rows: torch.Tensor,
        beta1: float,
        beta2: float,
        eps: float,
        gamma: float,
        batch_scale: float,
        curvature_due: bool,
        optimizer_step: int,
        curvature_update_index: int,
        trust_epsilon: float,
        quat_norm_tr: float,
        active_sh_degree: int,
        clip_hellinger_step: Any,
        validate_candidates: bool,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        curvature_initialized = self._curvature_initialized.index_select(
            0, state_rows
        )
        if curvature_due:
            if curvature_update_mask is None:
                curvature_initialized = torch.ones_like(curvature_initialized)
            else:
                curvature_initialized = curvature_initialized | curvature_update_mask[
                    chunk
                ]
        last_gradient_step = self._last_gradient_step.index_select(0, state_rows)
        last_curvature_update = self._last_curvature_update.index_select(
            0, state_rows
        )
        if validate_candidates:
            gradient_time_valid = (last_gradient_step < optimizer_step).all()
            curvature_time_valid = (
                last_curvature_update <= curvature_update_index
            ).all()

        gradient_gap = optimizer_step - last_gradient_step
        gradient_decay = torch.pow(
            torch.full(
                gradient_gap.shape,
                beta1,
                dtype=torch.float32,
                device=self.device,
            ),
            gradient_gap.to(dtype=torch.float32),
        ).unsqueeze(1)
        curvature_gap = curvature_update_index - last_curvature_update
        curvature_decay = torch.pow(
            torch.full(
                curvature_gap.shape,
                beta2,
                dtype=torch.float32,
                device=self.device,
            ),
            curvature_gap.to(dtype=torch.float32),
        ).unsqueeze(1)
        chunk_curvature_ready = curvature_initialized.unsqueeze(1)

        raw_params: Dict[str, torch.Tensor] = {}
        proposals: Dict[str, torch.Tensor] = {}
        candidate_averages: Dict[str, torch.Tensor] = {}
        candidate_curvatures: Dict[str, torch.Tensor] = {}
        validity_flags = []
        for name, _, _, _ in self.COMPONENT_SPECS:
            avg = self._exp_avg[name].index_select(0, state_rows)
            curvature_avg = self._exp_avg_sq[name].index_select(0, state_rows)
            avg.mul_(gradient_decay).add_(
                sparse_grad_components[name][chunk],
                alpha=(1.0 - beta1) * batch_scale,
            )
            curvature_avg.mul_(curvature_decay)
            if curvature_due:
                curvature_avg.add_(
                    sparse_curvature_components[name][chunk],
                    alpha=(1.0 - beta2) * batch_scale,
                )

            coordinate_ready = curvature_avg > 0.0
            denominator = torch.where(
                coordinate_ready,
                curvature_avg + eps,
                torch.ones_like(curvature_avg),
            )
            proposal = -gamma * avg / denominator
            proposal = torch.where(
                chunk_curvature_ready & coordinate_ready,
                proposal,
                torch.zeros_like(proposal),
            )
            raw_value = param_views[name].index_select(0, local_rows)
            raw_params[name] = raw_value
            proposals[name] = proposal
            candidate_averages[name] = avg
            candidate_curvatures[name] = curvature_avg
            if validate_candidates:
                validity_flags.extend(
                    (
                        torch.isfinite(avg).all(),
                        torch.isfinite(curvature_avg).all(),
                        torch.isfinite(proposal).all(),
                        torch.isfinite(raw_value).all(),
                    )
                )

        clipped_chunk = clip_hellinger_step(
            raw_params,
            proposals,
            epsilon=trust_epsilon,
            quat_norm_tr=None if quat_norm_tr < 0.0 else quat_norm_tr,
            active_sh_degree=active_sh_degree,
        )
        chunk_rows = int(local_rows.numel())
        self._validate_component_dict(
            clipped_chunk,
            name="clipped_steps",
            n_rows=chunk_rows,
            check_finite=False,
        )
        candidate_parameters: Dict[str, torch.Tensor] = {}
        clipped_count_terms = []
        for name, _, _, _ in self.COMPONENT_SPECS:
            clipped_value = clipped_chunk[name]
            next_value = raw_params[name] + clipped_value
            candidate_parameters[name] = next_value
            if validate_candidates:
                validity_flags.extend(
                    (
                        torch.isfinite(clipped_value).all(),
                        torch.isfinite(next_value).all(),
                    )
                )
                clipped_count_terms.append(
                    torch.count_nonzero(clipped_value != proposals[name])
                )

        chunk_summary = None
        if validate_candidates:
            transaction_valid = torch.stack(validity_flags).all().to(torch.int64)
            chunk_summary = torch.stack(
                (
                    transaction_valid,
                    gradient_time_valid.to(torch.int64),
                    curvature_time_valid.to(torch.int64),
                    torch.stack(clipped_count_terms).sum(),
                    torch.count_nonzero(~curvature_initialized),
                )
            )
        return (
            candidate_averages,
            candidate_curvatures,
            candidate_parameters,
            curvature_initialized,
            chunk_summary,
        )

    def _build_curvature_only_chunk(
        self,
        *,
        sparse_curvature_components: Dict[str, torch.Tensor],
        chunk: slice,
        state_rows: torch.Tensor,
        beta2: float,
        batch_scale: float,
        curvature_update_index: int,
        validate_candidates: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        last_curvature_update = self._last_curvature_update.index_select(
            0, state_rows
        )
        curvature_gap = curvature_update_index - last_curvature_update
        curvature_decay = torch.pow(
            torch.full(
                curvature_gap.shape,
                beta2,
                dtype=torch.float32,
                device=self.device,
            ),
            curvature_gap.to(dtype=torch.float32),
        ).unsqueeze(1)
        candidates = {}
        validity_flags = []
        for name, _, _, _ in self.COMPONENT_SPECS:
            value = self._exp_avg_sq[name].index_select(0, state_rows)
            value.mul_(curvature_decay).add_(
                sparse_curvature_components[name][chunk],
                alpha=(1.0 - beta2) * batch_scale,
            )
            candidates[name] = value
            if validate_candidates:
                validity_flags.append(torch.isfinite(value).all())
        summary = None
        if validate_candidates:
            summary = torch.stack(
                validity_flags
                + [(last_curvature_update <= curvature_update_index).all()]
            ).all()
        return candidates, summary

    def _commit_transaction_chunk(
        self,
        *,
        param_views: Dict[str, torch.Tensor],
        local_rows: torch.Tensor,
        state_rows: torch.Tensor,
        candidate_averages: Dict[str, torch.Tensor],
        candidate_curvatures: Dict[str, torch.Tensor],
        candidate_parameters: Dict[str, torch.Tensor],
        curvature_initialized: torch.Tensor,
        optimizer_step: int,
        curvature_update_index: int,
    ) -> None:
        for name, _, _, _ in self.COMPONENT_SPECS:
            self._exp_avg[name].index_copy_(
                0, state_rows, candidate_averages[name]
            )
            self._exp_avg_sq[name].index_copy_(
                0, state_rows, candidate_curvatures[name]
            )
            param_views[name].index_copy_(
                0, local_rows, candidate_parameters[name]
            )
        self._curvature_initialized.index_copy_(
            0, state_rows, curvature_initialized
        )
        self._last_gradient_step.index_fill_(0, state_rows, optimizer_step)
        self._last_curvature_update.index_fill_(
            0, state_rows, curvature_update_index
        )

    def _commit_curvature_only_chunk(
        self,
        *,
        state_rows: torch.Tensor,
        candidate_curvatures: Dict[str, torch.Tensor],
        curvature_update_index: int,
    ) -> None:
        for name, _, _, _ in self.COMPONENT_SPECS:
            self._exp_avg_sq[name].index_copy_(
                0, state_rows, candidate_curvatures[name]
            )
        self._curvature_initialized.index_fill_(0, state_rows, True)
        self._last_curvature_update.index_fill_(
            0, state_rows, curvature_update_index
        )

    def _publish_touched_slot_metadata(
        self,
        manager,
        touched_blocks: List[int],
        touched_slots: List[int],
        touched_counts: List[int],
    ) -> int:
        cold_rows = 0
        for block_id, slot, count in zip(
            touched_blocks, touched_slots, touched_counts
        ):
            if not self._slot_initialized[slot]:
                self._slot_initialized[slot] = True
                block_slice = manager.block_to_gpu_slice[block_id]
                row_count = int(block_slice.stop - block_slice.start)
                self._slot_row_counts[slot] = row_count
                self._logical_state_blocks += 1
                self._logical_state_bytes += (
                    row_count * self._logical_state_bytes_per_row()
                )
                self._stats["cold_restarts"] += 1
                self._stats["cold_restarted_rows_touched_total"] += int(count)
                cold_rows += int(count)
            self._slot_steps[slot] += 1
        return cold_rows

    def _normalize_sparse_ids(self, gaussians, values, *, name: str) -> torch.Tensor:
        if values is None:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        rows = values.to(device=self.device, dtype=torch.long).contiguous()
        if rows.ndim != 1:
            raise ValueError(f"[GPUResidentSophiaTR] {name} must be one-dimensional")
        if rows.numel() == 0:
            return rows
        in_bounds = (rows >= 0).all() & (rows < int(gaussians._xyz.shape[0])).all()
        sorted_unique = (
            (rows[1:] > rows[:-1]).all()
            if rows.numel() > 1
            else torch.ones((), dtype=torch.bool, device=self.device)
        )
        if not bool(in_bounds):
            raise IndexError(
                f"[GPUResidentSophiaTR] {name} is outside the working set"
            )
        if not bool(sorted_unique):
            raise ValueError(
                f"[GPUResidentSophiaTR] {name} must be sorted and unique"
            )
        return rows

    def _validated_parameter_views(self, gaussians) -> Dict[str, torch.Tensor]:
        param_views = {
            name: getattr(gaussians, attr_name).data
            for name, attr_name, _, _ in self.COMPONENT_SPECS
        }
        for name, _, width, _ in self.COMPONENT_SPECS:
            param = param_views[name]
            if param.ndim != 2 or param.shape[1] != width:
                raise ValueError(
                    f"[GPUResidentSophiaTR] {name} parameter has "
                    f"shape={tuple(param.shape)}; expected [N, {width}]"
                )
            if param.device != self.device or param.dtype != torch.float32:
                raise ValueError(
                    f"[GPUResidentSophiaTR] {name} parameter must be FP32 on "
                    f"{self.device}; got dtype={param.dtype}, device={param.device}"
                )
            if not param.is_contiguous():
                raise ValueError(
                    f"[GPUResidentSophiaTR] {name} parameter must be contiguous"
                )
        return param_views

    def _partition_curvature_components(
        self,
        *,
        gradient_rows: torch.Tensor,
        curvature_rows: torch.Tensor,
        curvature_components: Dict[str, torch.Tensor],
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        gradient_count = int(gradient_rows.numel())
        curvature_count = int(curvature_rows.numel())
        gradient_mask = torch.zeros(
            (gradient_count,), dtype=torch.bool, device=self.device
        )
        curvature_positions_for_gradient = torch.zeros(
            (gradient_count,), dtype=torch.long, device=self.device
        )
        if gradient_count and curvature_count:
            positions = torch.searchsorted(curvature_rows, gradient_rows)
            safe_positions = positions.clamp_max(curvature_count - 1)
            gradient_mask = (positions < curvature_count) & (
                curvature_rows.index_select(0, safe_positions) == gradient_rows
            )
            curvature_positions_for_gradient = safe_positions

        aligned = {}
        gradient_positions = torch.nonzero(gradient_mask, as_tuple=False).flatten()
        matched_curvature_positions = curvature_positions_for_gradient.index_select(
            0, gradient_positions
        )
        for name, _, width, _ in self.COMPONENT_SPECS:
            value = torch.zeros(
                (gradient_count, width), dtype=torch.float32, device=self.device
            )
            if gradient_positions.numel():
                value.index_copy_(
                    0,
                    gradient_positions,
                    curvature_components[name].index_select(
                        0, matched_curvature_positions
                    ),
                )
            aligned[name] = value.contiguous()

        if not curvature_count:
            curvature_only_mask = torch.empty(
                (0,), dtype=torch.bool, device=self.device
            )
        elif not gradient_count:
            curvature_only_mask = torch.ones(
                (curvature_count,), dtype=torch.bool, device=self.device
            )
        else:
            positions = torch.searchsorted(gradient_rows, curvature_rows)
            safe_positions = positions.clamp_max(gradient_count - 1)
            overlap = (positions < gradient_count) & (
                gradient_rows.index_select(0, safe_positions) == curvature_rows
            )
            curvature_only_mask = ~overlap
        curvature_only_rows = curvature_rows[curvature_only_mask].contiguous()
        curvature_only_components = {
            name: value[curvature_only_mask].contiguous()
            for name, value in curvature_components.items()
        }
        return (
            aligned,
            gradient_mask,
            curvature_only_rows,
            curvature_only_components,
        )

    def prepare_step(
        self,
        iteration: int,
        gaussians,
        sparse_grad_local_ids: Optional[torch.Tensor],
        sparse_grad_components: Optional[Dict[str, torch.Tensor]],
        *,
        sparse_curvature_local_ids: Optional[torch.Tensor] = None,
        sparse_curvature_components: Optional[Dict[str, torch.Tensor]] = None,
        curvature_due: Optional[bool] = None,
        optimizer_step: Optional[int] = None,
    ) -> SophiaTRPreparedStep:
        from .sophia_tr_math import (
            clip_hellinger_step,
            exponential_schedule,
            optimizer_step_from_iteration,
            resolve_curvature_schedule,
        )

        if self._pending_transaction is not None:
            raise RuntimeError("[GPUResidentSophiaTR] another prepared step is pending")
        if self._residency_corrupted or self._transaction_corrupted:
            raise RuntimeError(
                "[GPUResidentSophiaTR] optimizer is unusable after a failed resident "
                "slot clear or commit; recreate it before processing another step"
            )
        args = getattr(gaussians, "args", None)
        if args is None:
            raise RuntimeError("[GPUResidentSophiaTR] gaussians.args is required")
        manager = getattr(gaussians, "gpu_working_set_manager", None)
        if manager is None or manager.local_to_global_idx is None:
            raise RuntimeError(
                "[GPUResidentSophiaTR] gpu_working_set_manager with "
                "local_to_global_idx is required"
            )

        gradient_rows = self._normalize_sparse_ids(
            gaussians, sparse_grad_local_ids, name="sparse_grad_local_ids"
        )
        interval = int(getattr(args, "paper_sophia_curvature_interval", 10))
        optimizer_step, scheduled_curvature = resolve_curvature_schedule(
            iteration=iteration,
            batch_size=max(1, int(self.batch_size)),
            interval=interval,
            optimizer_step=optimizer_step,
        )
        if curvature_due is None:
            curvature_due = scheduled_curvature
        curvature_due = bool(curvature_due)
        if curvature_due != scheduled_curvature:
            raise ValueError(
                "[GPUResidentSophiaTR] curvature_due disagrees with the configured schedule"
            )

        legacy_curvature_rows = sparse_curvature_local_ids is None
        if legacy_curvature_rows and sparse_curvature_components is not None:
            curvature_rows = gradient_rows
        else:
            curvature_rows = self._normalize_sparse_ids(
                gaussians,
                sparse_curvature_local_ids,
                name="sparse_curvature_local_ids",
            )
        gradient_count = int(gradient_rows.numel())
        curvature_count = int(curvature_rows.numel())
        if gradient_count and sparse_grad_components is None:
            raise RuntimeError(
                "[GPUResidentSophiaTR] gradient rows require sparse_grad_components"
            )
        if sparse_grad_components is not None:
            self._validate_component_dict(
                sparse_grad_components,
                name="sparse_grad_components",
                n_rows=gradient_count,
            )
        if not curvature_due and (
            curvature_count or sparse_curvature_components is not None
        ):
            raise ValueError(
                "[GPUResidentSophiaTR] curvature inputs were provided off schedule"
            )
        if curvature_due and legacy_curvature_rows and gradient_count and (
            sparse_curvature_components is None
        ):
            raise RuntimeError(
                "[GPUResidentSophiaTR] curvature is due but no Hutchinson sample was provided"
            )
        if curvature_count and sparse_curvature_components is None:
            raise RuntimeError(
                "[GPUResidentSophiaTR] curvature rows require sparse_curvature_components"
            )
        if sparse_curvature_components is not None:
            self._validate_component_dict(
                sparse_curvature_components,
                name="sparse_curvature_components",
                n_rows=curvature_count,
                require_non_negative=True,
            )

        param_views = self._validated_parameter_views(gaussians)
        (
            gradient_state_rows,
            _,
            gradient_touched_blocks,
            gradient_touched_slots,
            gradient_touched_counts,
        ) = self._working_set_state_rows(
            manager,
            gradient_rows,
            assume_sorted_unique=True,
            validate_mapping=True,
        )
        (
            _,
            _,
            curvature_touched_blocks,
            _,
            _,
        ) = self._working_set_state_rows(
            manager,
            curvature_rows,
            assume_sorted_unique=True,
            validate_mapping=True,
        )

        gradient_curvature_components = None
        gradient_curvature_mask = None
        curvature_only_rows = torch.empty(
            (0,), dtype=torch.long, device=self.device
        )
        curvature_only_components = None
        if curvature_due:
            if sparse_curvature_components is None:
                sparse_curvature_components = {
                    name: torch.empty(
                        (0, width), dtype=torch.float32, device=self.device
                    )
                    for name, _, width, _ in self.COMPONENT_SPECS
                }
            (
                gradient_curvature_components,
                gradient_curvature_mask,
                curvature_only_rows,
                curvature_only_components,
            ) = self._partition_curvature_components(
                gradient_rows=gradient_rows,
                curvature_rows=curvature_rows,
                curvature_components=sparse_curvature_components,
            )
        (
            curvature_only_state_rows,
            _,
            _,
            _,
            _,
        ) = self._working_set_state_rows(
            manager,
            curvature_only_rows,
            assume_sorted_unique=True,
            validate_mapping=True,
        )
        union_rows = (
            torch.unique(
                torch.cat((gradient_rows, curvature_rows), dim=0), sorted=True
            )
            if gradient_count or curvature_count
            else torch.empty((0,), dtype=torch.long, device=self.device)
        )
        (
            _,
            _,
            union_touched_blocks,
            union_touched_slots,
            union_touched_counts,
        ) = self._working_set_state_rows(
            manager,
            union_rows,
            assume_sorted_unique=True,
            validate_mapping=True,
        )

        beta1, beta2 = gaussians.optimizer.param_groups[0]["betas"]
        beta1 = float(beta1)
        beta2 = float(beta2)
        eps = float(gaussians.optimizer.param_groups[0]["eps"])
        gamma = float(getattr(args, "paper_sophia_gamma", 1.0))
        if not 0.0 < beta1 < 1.0 or not 0.0 < beta2 < 1.0:
            raise ValueError("[GPUResidentSophiaTR] betas must be in (0, 1)")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("[GPUResidentSophiaTR] eps must be finite and positive")
        if not math.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("[GPUResidentSophiaTR] gamma must be finite and positive")
        if self.batch_size <= 0:
            raise ValueError("[GPUResidentSophiaTR] batch_size must be positive")

        curvature_update_index = (optimizer_step - 1) // interval + 1
        total_iterations = max(1, int(getattr(args, "iterations", iteration)))
        total_optimizer_steps = optimizer_step_from_iteration(
            total_iterations, self.batch_size
        )
        trust_epsilon_initial = float(
            getattr(args, "paper_tr_epsilon_init", 1e-6)
        )
        trust_epsilon_final = float(
            getattr(args, "paper_tr_epsilon_final", 1e-8)
        )
        if total_optimizer_steps == 1:
            trust_epsilon = trust_epsilon_initial
        else:
            trust_epsilon = exponential_schedule(
                trust_epsilon_initial,
                trust_epsilon_final,
                optimizer_step - 1,
                total_optimizer_steps - 1,
            )
        quat_norm_tr = float(getattr(args, "paper_tr_quat_norm", -1.0))
        active_sh_degree = int(getattr(gaussians, "active_sh_degree", 3))
        batch_scale = 1.0 / float(self.batch_size)
        chunk_rows = max(1, int(self.TRANSACTION_CHUNK_ROWS))

        validity_summary = torch.ones(
            (3,), dtype=torch.int64, device=self.device
        )
        count_summary = torch.zeros((2,), dtype=torch.int64, device=self.device)
        for chunk_start in range(0, gradient_count, chunk_rows):
            chunk_stop = min(gradient_count, chunk_start + chunk_rows)
            chunk = slice(chunk_start, chunk_stop)
            candidates = self._build_transaction_chunk(
                param_views=param_views,
                sparse_grad_components=sparse_grad_components,
                sparse_curvature_components=gradient_curvature_components,
                curvature_update_mask=gradient_curvature_mask,
                chunk=chunk,
                local_rows=gradient_rows[chunk],
                state_rows=gradient_state_rows[chunk],
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                gamma=gamma,
                batch_scale=batch_scale,
                curvature_due=curvature_due,
                optimizer_step=optimizer_step,
                curvature_update_index=curvature_update_index,
                trust_epsilon=trust_epsilon,
                quat_norm_tr=quat_norm_tr,
                active_sh_degree=active_sh_degree,
                clip_hellinger_step=clip_hellinger_step,
                validate_candidates=True,
            )
            chunk_summary = candidates[4]
            validity_summary.mul_(chunk_summary[:3])
            count_summary.add_(chunk_summary[3:])
            del candidates
        transaction_summary = torch.cat(
            (validity_summary, count_summary)
        ).cpu().tolist()
        if gradient_count and not bool(transaction_summary[1]):
            raise RuntimeError(
                "[GPUResidentSophiaTR] optimizer_step must increase for touched rows"
            )
        if gradient_count and not bool(transaction_summary[2]):
            raise RuntimeError(
                "[GPUResidentSophiaTR] curvature update index moved backwards"
            )
        if not bool(transaction_summary[0]):
            raise FloatingPointError(
                "[GPUResidentSophiaTR] candidate optimizer update produced NaN/Inf"
            )

        curvature_only_count = int(curvature_only_rows.numel())
        curvature_only_valid = torch.ones((), dtype=torch.bool, device=self.device)
        for chunk_start in range(0, curvature_only_count, chunk_rows):
            chunk_stop = min(curvature_only_count, chunk_start + chunk_rows)
            chunk = slice(chunk_start, chunk_stop)
            _, chunk_valid = self._build_curvature_only_chunk(
                sparse_curvature_components=curvature_only_components,
                chunk=chunk,
                state_rows=curvature_only_state_rows[chunk],
                beta2=beta2,
                batch_scale=batch_scale,
                curvature_update_index=curvature_update_index,
                validate_candidates=True,
            )
            curvature_only_valid &= chunk_valid
        if not bool(curvature_only_valid):
            raise FloatingPointError(
                "[GPUResidentSophiaTR] candidate curvature update produced NaN/Inf "
                "or moved backwards"
            )

        prepared = SophiaTRPreparedStep(
            optimizer=self,
            gaussians=gaussians,
            iteration=int(iteration),
            optimizer_step=optimizer_step,
            curvature_due=curvature_due,
            curvature_update_index=curvature_update_index,
            trust_epsilon=float(trust_epsilon),
            gradient_local_rows=gradient_rows,
            gradient_state_rows=gradient_state_rows,
            gradient_components=sparse_grad_components or {},
            gradient_curvature_components=gradient_curvature_components,
            gradient_curvature_mask=gradient_curvature_mask,
            curvature_only_local_rows=curvature_only_rows,
            curvature_only_state_rows=curvature_only_state_rows,
            curvature_only_components=curvature_only_components,
            union_touched_blocks=union_touched_blocks,
            union_touched_slots=union_touched_slots,
            union_touched_counts=union_touched_counts,
            gradient_touched_blocks=gradient_touched_blocks,
            gradient_touched_slots=gradient_touched_slots,
            gradient_touched_counts=gradient_touched_counts,
            curvature_touched_blocks=curvature_touched_blocks,
            curvature_row_count=curvature_count,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            gamma=gamma,
            batch_scale=batch_scale,
            quat_norm_tr=quat_norm_tr,
            active_sh_degree=active_sh_degree,
            clipped_values=int(transaction_summary[3]),
            rows_skipped_without_curvature=int(transaction_summary[4]),
        )
        self._pending_transaction = prepared
        return prepared

    def abort_step(self, prepared: SophiaTRPreparedStep) -> None:
        if prepared.optimizer is not self or self._pending_transaction is not prepared:
            raise ValueError("[GPUResidentSophiaTR] prepared step does not belong here")
        if prepared.status != "prepared":
            raise RuntimeError(
                f"[GPUResidentSophiaTR] cannot abort a {prepared.status} step"
            )
        prepared.status = "aborted"
        self._pending_transaction = None

    def commit_step(self, prepared: SophiaTRPreparedStep) -> Dict[str, Any]:
        from .sophia_tr_math import clip_hellinger_step

        if prepared.optimizer is not self or self._pending_transaction is not prepared:
            raise ValueError("[GPUResidentSophiaTR] prepared step does not belong here")
        if prepared.status != "prepared":
            raise RuntimeError(
                f"[GPUResidentSophiaTR] cannot commit a {prepared.status} step"
            )
        python_state = None
        try:
            manager = prepared.gaussians.gpu_working_set_manager
            param_views = self._validated_parameter_views(prepared.gaussians)
            gradient_count = int(prepared.gradient_local_rows.numel())
            curvature_only_count = int(prepared.curvature_only_local_rows.numel())
            chunk_rows = max(1, int(self.TRANSACTION_CHUNK_ROWS))
            cold_gradient_rows = sum(
                int(count)
                for slot, count in zip(
                    prepared.gradient_touched_slots,
                    prepared.gradient_touched_counts,
                )
                if not self._slot_initialized[slot]
            )
            python_state = self._snapshot_transaction_python_state()
            for chunk_start in range(0, gradient_count, chunk_rows):
                chunk_stop = min(gradient_count, chunk_start + chunk_rows)
                chunk = slice(chunk_start, chunk_stop)
                candidates = self._build_transaction_chunk(
                    param_views=param_views,
                    sparse_grad_components=prepared.gradient_components,
                    sparse_curvature_components=(
                        prepared.gradient_curvature_components
                    ),
                    curvature_update_mask=prepared.gradient_curvature_mask,
                    chunk=chunk,
                    local_rows=prepared.gradient_local_rows[chunk],
                    state_rows=prepared.gradient_state_rows[chunk],
                    beta1=prepared.beta1,
                    beta2=prepared.beta2,
                    eps=prepared.eps,
                    gamma=prepared.gamma,
                    batch_scale=prepared.batch_scale,
                    curvature_due=prepared.curvature_due,
                    optimizer_step=prepared.optimizer_step,
                    curvature_update_index=prepared.curvature_update_index,
                    trust_epsilon=prepared.trust_epsilon,
                    quat_norm_tr=prepared.quat_norm_tr,
                    active_sh_degree=prepared.active_sh_degree,
                    clip_hellinger_step=clip_hellinger_step,
                    validate_candidates=False,
                )
                (
                    candidate_averages,
                    candidate_curvatures,
                    candidate_parameters,
                    curvature_initialized,
                    _,
                ) = candidates
                self._commit_transaction_chunk(
                    param_views=param_views,
                    local_rows=prepared.gradient_local_rows[chunk],
                    state_rows=prepared.gradient_state_rows[chunk],
                    candidate_averages=candidate_averages,
                    candidate_curvatures=candidate_curvatures,
                    candidate_parameters=candidate_parameters,
                    curvature_initialized=curvature_initialized,
                    optimizer_step=prepared.optimizer_step,
                    curvature_update_index=prepared.curvature_update_index,
                )
                del candidates

            for chunk_start in range(0, curvature_only_count, chunk_rows):
                chunk_stop = min(curvature_only_count, chunk_start + chunk_rows)
                chunk = slice(chunk_start, chunk_stop)
                candidate_curvatures, _ = self._build_curvature_only_chunk(
                    sparse_curvature_components=prepared.curvature_only_components,
                    chunk=chunk,
                    state_rows=prepared.curvature_only_state_rows[chunk],
                    beta2=prepared.beta2,
                    batch_scale=prepared.batch_scale,
                    curvature_update_index=prepared.curvature_update_index,
                    validate_candidates=False,
                )
                self._commit_curvature_only_chunk(
                    state_rows=prepared.curvature_only_state_rows[chunk],
                    candidate_curvatures=candidate_curvatures,
                    curvature_update_index=prepared.curvature_update_index,
                )
                del candidate_curvatures

            self._synchronize_transaction_device()
            self._publish_touched_slot_metadata(
                manager,
                prepared.union_touched_blocks,
                prepared.union_touched_slots,
                prepared.union_touched_counts,
            )
            touched_rows = gradient_count
            curvature_rows = (
                prepared.curvature_row_count if prepared.curvature_due else 0
            )
            self._stats["optimizer_rows_touched_total"] += touched_rows
            self._stats["trust_region_clipped_values_total"] += prepared.clipped_values
            self._stats["rows_skipped_without_curvature_total"] += (
                prepared.rows_skipped_without_curvature
            )
            self._stats["last_tr_epsilon"] = prepared.trust_epsilon
            self._stats["optimizer_step"] = prepared.optimizer_step
            if prepared.curvature_due:
                self._stats["curvature_updates"] += 1
                self._stats["curvature_rows_touched_total"] += curvature_rows
            self._update_state_stats()
            stats = {
                "updated_blocks": len(prepared.gradient_touched_blocks),
                "updated_block_ids": list(prepared.gradient_touched_blocks),
                "curvature_block_ids": list(prepared.curvature_touched_blocks),
                "touched_rows": touched_rows,
                "cold_rows": int(cold_gradient_rows),
                "curvature_due": prepared.curvature_due,
                "curvature_rows": curvature_rows,
                "clipped_values": prepared.clipped_values,
                "rows_skipped_without_curvature": (
                    prepared.rows_skipped_without_curvature
                ),
                "trust_region_epsilon": prepared.trust_epsilon,
                "optimizer_step": prepared.optimizer_step,
            }
            prepared.status = "committed"
            self._pending_transaction = None
            return stats
        except BaseException as error:
            self._transaction_corrupted = True
            prepared.status = "failed"
            self._pending_transaction = None
            if python_state is not None:
                try:
                    self._restore_transaction_python_state(python_state)
                except BaseException as metadata_error:
                    raise RuntimeError(
                        "[GPUResidentSophiaTR] commit failed and Python metadata could "
                        f"not be restored; optimizer is unusable (original={error!r})"
                    ) from metadata_error
            if hasattr(error, "add_note"):
                error.add_note(
                    "[GPUResidentSophiaTR] commit may be partial; optimizer is "
                    "poisoned and must be recreated"
                )
            raise

    def step(
        self,
        iteration: int,
        gaussians,
        sparse_grad_local_ids: Optional[torch.Tensor],
        sparse_grad_components: Optional[Dict[str, torch.Tensor]],
        *,
        sparse_curvature_local_ids: Optional[torch.Tensor] = None,
        sparse_curvature_components: Optional[Dict[str, torch.Tensor]] = None,
        curvature_due: Optional[bool] = None,
        optimizer_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        prepared = self.prepare_step(
            iteration=iteration,
            gaussians=gaussians,
            sparse_grad_local_ids=sparse_grad_local_ids,
            sparse_grad_components=sparse_grad_components,
            sparse_curvature_local_ids=sparse_curvature_local_ids,
            sparse_curvature_components=sparse_curvature_components,
            curvature_due=curvature_due,
            optimizer_step=optimizer_step,
        )
        return self.commit_step(prepared)
