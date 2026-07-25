import math
import types
import unittest
from unittest import mock

import torch

from strategies.tide_engine.gpu_resident_optimizer import GPUResidentAdam


COMPONENT_SPECS = GPUResidentAdam.COMPONENT_SPECS


class _Manager:
    def __init__(self, block_layout):
        self.set_layout(block_layout)

    def set_layout(self, block_layout):
        self.loaded_blocks = [block_id for block_id, _ in block_layout]
        self.block_to_gpu_slice = {
            block_id: block_slice for block_id, block_slice in block_layout
        }
        total_rows = max(block_slice.stop for _, block_slice in block_layout)
        self.local_to_global_idx = torch.arange(total_rows, device="cuda")


class _Gaussians:
    def __init__(self, manager, params, columns_lr, betas=(0.9, 0.999), eps=1e-8):
        self.gpu_working_set_manager = manager
        for name, attr_name, _, _ in COMPONENT_SPECS:
            setattr(self, attr_name, torch.nn.Parameter(params[name].clone()))
        self.optimizer = types.SimpleNamespace(
            columns_lr=torch.tensor(columns_lr, dtype=torch.float32),
            param_groups=[{"betas": betas, "eps": eps}],
        )

    def parameter_values(self):
        return {
            name: getattr(self, attr_name).detach().clone()
            for name, attr_name, _, _ in COMPONENT_SPECS
        }


def _random_params(rows, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return {
        name: torch.randn((rows, width), generator=generator, device="cuda")
        for name, _, width, _ in COMPONENT_SPECS
    }


def _random_grads(rows, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return {
        name: torch.randn((rows, width), generator=generator, device="cuda")
        for name, _, width, _ in COMPONENT_SPECS
    }


def _reference_step(
    params,
    state,
    block_layout,
    local_ids,
    grads,
    batch_size,
    columns_lr,
    betas,
    eps,
):
    beta1, beta2 = betas
    for block_id, block_slice in block_layout:
        selected = (local_ids >= block_slice.start) & (local_ids < block_slice.stop)
        positions = torch.nonzero(selected, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        rows = local_ids[positions] - block_slice.start
        row_count = block_slice.stop - block_slice.start
        block_state = state.get(block_id)
        if block_state is None:
            block_state = {
                "step": 0,
                "exp_avg": {
                    name: torch.zeros((row_count, width), device="cuda")
                    for name, _, width, _ in COMPONENT_SPECS
                },
                "exp_avg_sq": {
                    name: torch.zeros((row_count, width), device="cuda")
                    for name, _, width, _ in COMPONENT_SPECS
                },
            }
            state[block_id] = block_state
        block_state["step"] += 1
        step = block_state["step"]
        bias1 = 1.0 - beta1**step
        denom_scale = math.sqrt(1.0 - beta2**step)

        for name, _, _, group_index in COMPONENT_SPECS:
            grad = grads[name][positions] / float(batch_size)
            avg = block_state["exp_avg"][name][rows]
            avg_sq = block_state["exp_avg_sq"][name][rows]
            avg = avg * beta1 + grad * (1.0 - beta1)
            avg_sq = avg_sq * beta2 + grad.square() * (1.0 - beta2)
            block_state["exp_avg"][name][rows] = avg
            block_state["exp_avg_sq"][name][rows] = avg_sq
            update = avg / (avg_sq.sqrt() / denom_scale + eps)
            params[name][local_ids[positions]] -= update * (columns_lr[group_index] / bias1)


class GPUResidentAdamAllocationTest(unittest.TestCase):
    def test_initial_allocation_matches_actual_resident_blocks(self):
        block_size = 4
        optimizer = GPUResidentAdam(
            batch_size=1,
            block_size=block_size,
            capacity_blocks=0,
            device="cpu",
        )

        optimizer.set_resident_blocks([10, 20, 30])

        expected_bytes = 3 * block_size * GPUResidentAdam.TOTAL_WIDTH * 2 * 4
        self.assertEqual(optimizer._allocated_slots, 3)
        self.assertEqual(optimizer.get_stats()["allocated_state_bytes"], expected_bytes)

    def test_planner_capacity_does_not_control_optimizer_allocation(self):
        from strategies.tide_engine import engine as tide_engine

        args = types.SimpleNamespace(
            gaussian_block_size=4096,
            paper_resident_capacity_blocks=8192,
        )
        gaussians = types.SimpleNamespace(args=args)
        optimizer = types.SimpleNamespace(
            batch_size=32,
            block_size=4096,
            capacity_blocks=0,
        )

        with mock.patch.object(
            tide_engine, "GPUResidentAdam", return_value=optimizer
        ) as optimizer_cls:
            first = tide_engine.get_gpu_resident_optimizer(gaussians, batch_size=32)
            optimizer_cls.assert_called_once_with(
                batch_size=32,
                block_size=4096,
                capacity_blocks=0,
                device="cuda",
            )

            args.paper_resident_capacity_blocks = 16384
            second = tide_engine.get_gpu_resident_optimizer(gaussians, batch_size=32)

        self.assertIs(first, second)
        optimizer_cls.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class GPUResidentAdamTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.set_device(0)
        self.block_size = 4
        self.batch_size = 3
        self.columns_lr = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        self.betas = (0.9, 0.999)
        self.eps = 1e-15
        self.optimizer = GPUResidentAdam(
            batch_size=self.batch_size,
            block_size=self.block_size,
            capacity_blocks=2,
            device="cuda",
        )
        self.reference_state = {}

    def assert_parameters_close(self, gaussians, reference):
        for name, actual in gaussians.parameter_values().items():
            torch.testing.assert_close(actual, reference[name], rtol=2e-5, atol=2e-6)

    def run_step(self, gaussians, reference, local_ids, grads, iteration):
        stats = self.optimizer.step(
            iteration=iteration,
            gaussians=gaussians,
            sparse_grad_local_ids=local_ids,
            sparse_grad_components=grads,
        )
        layout = [
            (block_id, gaussians.gpu_working_set_manager.block_to_gpu_slice[block_id])
            for block_id in gaussians.gpu_working_set_manager.loaded_blocks
        ]
        _reference_step(
            reference,
            self.reference_state,
            layout,
            local_ids,
            grads,
            self.batch_size,
            self.columns_lr,
            self.betas,
            self.eps,
        )
        self.assert_parameters_close(gaussians, reference)
        return stats

    def test_matches_reference_across_retention_eviction_and_readmission(self):
        first_layout = [(10, slice(0, 4)), (20, slice(4, 7))]
        manager = _Manager(first_layout)
        initial = _random_params(7, seed=10)
        gaussians = _Gaussians(
            manager, initial, self.columns_lr, betas=self.betas, eps=self.eps
        )
        reference = {name: value.clone() for name, value in initial.items()}
        self.optimizer.set_resident_blocks([10, 20])

        local_ids = torch.tensor([0, 2, 4, 6], device="cuda")
        stats = self.run_step(
            gaussians, reference, local_ids, _random_grads(4, seed=20), iteration=1
        )
        self.assertEqual(stats, {"updated_blocks": 2, "updated_block_ids": [10, 20], "touched_rows": 4, "cold_rows": 4})

        local_ids = torch.tensor([1, 4, 5], device="cuda")
        stats = self.run_step(
            gaussians, reference, local_ids, _random_grads(3, seed=21), iteration=2
        )
        self.assertEqual(stats, {"updated_blocks": 2, "updated_block_ids": [10, 20], "touched_rows": 3, "cold_rows": 0})

        # Block 20 remains resident but moves to another local slice; block 30 is cold.
        second_layout = [(20, slice(0, 3)), (30, slice(3, 7))]
        manager.set_layout(second_layout)
        second_params = _random_params(7, seed=11)
        for name, attr_name, _, _ in COMPONENT_SPECS:
            setattr(gaussians, attr_name, torch.nn.Parameter(second_params[name].clone()))
        reference = {name: value.clone() for name, value in second_params.items()}
        self.reference_state.pop(10)
        self.optimizer.set_resident_blocks([20, 30])
        local_ids = torch.tensor([0, 2, 3, 6], device="cuda")
        stats = self.run_step(
            gaussians, reference, local_ids, _random_grads(4, seed=22), iteration=3
        )
        self.assertEqual(stats, {"updated_blocks": 2, "updated_block_ids": [20, 30], "touched_rows": 4, "cold_rows": 2})

        # Re-admitting block 10 must not recover the state discarded above.
        third_layout = [(10, slice(0, 4))]
        manager.set_layout(third_layout)
        third_params = _random_params(4, seed=12)
        for name, attr_name, _, _ in COMPONENT_SPECS:
            setattr(gaussians, attr_name, torch.nn.Parameter(third_params[name].clone()))
        reference = {name: value.clone() for name, value in third_params.items()}
        self.reference_state.clear()
        self.optimizer.set_resident_blocks([10])
        local_ids = torch.tensor([0, 3], device="cuda")
        stats = self.run_step(
            gaussians, reference, local_ids, _random_grads(2, seed=23), iteration=4
        )
        self.assertEqual(stats, {"updated_blocks": 1, "updated_block_ids": [10], "touched_rows": 2, "cold_rows": 2})

        aggregate = self.optimizer.get_stats()
        self.assertEqual(aggregate["cold_restarts"], 4)
        self.assertEqual(aggregate["state_evictions"], 3)
        self.assertEqual(aggregate["resident_state_blocks"], 1)


if __name__ == "__main__":
    unittest.main()
