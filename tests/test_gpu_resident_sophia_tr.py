import types
import unittest
from unittest import mock

import torch

from strategies.tide_engine.gpu_resident_optimizer import GPUResidentSophiaTR


COMPONENT_SPECS = GPUResidentSophiaTR.COMPONENT_SPECS
CLIP_TARGET = "strategies.tide_engine.sophia_tr_math.clip_hellinger_step"
TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Manager:
    def __init__(self, rows, block_id=10):
        self.loaded_blocks = [block_id]
        self.block_to_gpu_slice = {block_id: slice(0, rows)}
        self.local_to_global_idx = torch.arange(rows, device=TEST_DEVICE)

    def replace_block(self, block_id, rows):
        self.loaded_blocks = [block_id]
        self.block_to_gpu_slice = {block_id: slice(0, rows)}


class _Gaussians:
    def __init__(self, manager, params, betas=(0.9, 0.999), eps=1e-12):
        self.gpu_working_set_manager = manager
        for name, attr_name, _, _ in COMPONENT_SPECS:
            setattr(self, attr_name, torch.nn.Parameter(params[name].clone()))
        self.optimizer = types.SimpleNamespace(
            param_groups=[{"betas": betas, "eps": eps}],
        )
        self.args = types.SimpleNamespace(
            iterations=100,
            paper_sophia_curvature_interval=10,
            paper_sophia_gamma=1.0,
            paper_tr_epsilon_init=1e-6,
            paper_tr_epsilon_final=1e-8,
            paper_tr_quat_norm=-1.0,
        )
        self.active_sh_degree = 3

    def parameter_values(self):
        return {
            name: getattr(self, attr_name).detach().clone()
            for name, attr_name, _, _ in COMPONENT_SPECS
        }


def _params(rows):
    values = {
        name: torch.zeros((rows, width), device=TEST_DEVICE)
        for name, _, width, _ in COMPONENT_SPECS
    }
    values["rotation"][:, 0] = 1.0
    return values


def _components(rows, value):
    return {
        name: torch.full((rows, width), value, device=TEST_DEVICE)
        for name, _, width, _ in COMPONENT_SPECS
    }


def _identity_clip(raw_params, steps, *args, **kwargs):
    del raw_params, args, kwargs
    return steps


class GPUResidentSophiaTRTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        self.rows = 2
        self.block_size = 4
        self.betas = (0.9, 0.999)
        self.eps = 1e-12
        self.local_ids = torch.arange(self.rows, device=TEST_DEVICE)

    def _make_case(self, batch_size=1):
        manager = _Manager(self.rows)
        initial = _params(self.rows)
        gaussians = _Gaussians(
            manager,
            initial,
            betas=self.betas,
            eps=self.eps,
        )
        optimizer = GPUResidentSophiaTR(
            batch_size=batch_size,
            block_size=self.block_size,
            capacity_blocks=1,
            device=TEST_DEVICE,
        )
        optimizer.set_resident_blocks([10])
        return optimizer, gaussians, initial

    def _state_rows(self, optimizer):
        slot = optimizer._block_to_slot[10]
        return slot * self.block_size + self.local_ids

    def _assert_components_close(self, actual, expected):
        for name, _, _, _ in COMPONENT_SPECS:
            torch.testing.assert_close(
                actual[name], expected[name], rtol=2e-5, atol=2e-6
            )

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_first_curvature_step_uses_raw_emas_without_bias_correction(self, _):
        optimizer, gaussians, initial = self._make_case()
        grads = _components(self.rows, 2.0)
        curvature = _components(self.rows, 5.0)

        stats = optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=grads,
            sparse_curvature_components=curvature,
            curvature_due=True,
            optimizer_step=1,
        )

        beta1, beta2 = self.betas
        expected_avg = (1.0 - beta1) * 2.0
        expected_curvature = (1.0 - beta2) * 5.0
        expected_step = -expected_avg / (expected_curvature + self.eps)
        state_rows = self._state_rows(optimizer)
        for name, attr_name, _, _ in COMPONENT_SPECS:
            torch.testing.assert_close(
                optimizer._exp_avg[name].index_select(0, state_rows),
                torch.full_like(grads[name], expected_avg),
            )
            torch.testing.assert_close(
                optimizer._exp_avg_sq[name].index_select(0, state_rows),
                torch.full_like(curvature[name], expected_curvature),
            )
            torch.testing.assert_close(
                getattr(gaussians, attr_name).detach(),
                initial[name] + expected_step,
                rtol=2e-5,
                atol=2e-6,
            )
        self.assertEqual(stats["curvature_rows"], self.rows)
        self.assertEqual(stats["rows_skipped_without_curvature"], 0)

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_batched_tr_schedule_reaches_both_endpoints(self, clip_mock):
        optimizer, gaussians, _ = self._make_case(batch_size=4)
        gaussians.args.iterations = 40
        zero_grads = _components(self.rows, 0.0)

        optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=zero_grads,
            sparse_curvature_components=_components(self.rows, 1.0),
            curvature_due=True,
            optimizer_step=1,
        )
        optimizer.step(
            iteration=37,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=zero_grads,
            curvature_due=False,
            optimizer_step=10,
        )

        self.assertEqual(clip_mock.call_args_list[0].kwargs["epsilon"], 1e-6)
        self.assertEqual(clip_mock.call_args_list[-1].kwargs["epsilon"], 1e-8)
        self.assertEqual(optimizer.get_stats()["state_mode"], "resident_blocks")

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_non_divisible_batch_schedule_reaches_exact_final_epsilon(self, clip_mock):
        optimizer, gaussians, _ = self._make_case(batch_size=4)
        gaussians.args.iterations = 10
        zero_grads = _components(self.rows, 0.0)

        optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=zero_grads,
            sparse_curvature_components=_components(self.rows, 1.0),
            curvature_due=True,
            optimizer_step=1,
        )
        stats = optimizer.step(
            iteration=9,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=zero_grads,
            curvature_due=False,
            optimizer_step=3,
        )

        self.assertEqual(clip_mock.call_args_list[-1].kwargs["epsilon"], 1e-8)
        self.assertEqual(stats["optimizer_step"], 3)

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_single_step_schedule_uses_exact_initial_epsilon(self, clip_mock):
        optimizer, gaussians, _ = self._make_case(batch_size=4)
        gaussians.args.iterations = 3

        stats = optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 0.0),
            sparse_curvature_components=_components(self.rows, 1.0),
            curvature_due=True,
            optimizer_step=1,
        )

        self.assertEqual(clip_mock.call_args.kwargs["epsilon"], 1e-6)
        self.assertEqual(stats["optimizer_step"], 1)

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_non_curvature_step_reuses_previous_curvature_ema(self, _):
        optimizer, gaussians, _ = self._make_case()
        first_grads = _components(self.rows, 2.0)
        curvature = _components(self.rows, 5.0)
        optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=first_grads,
            sparse_curvature_components=curvature,
            curvature_due=True,
            optimizer_step=1,
        )
        before = gaussians.parameter_values()
        state_rows = self._state_rows(optimizer)
        curvature_before = {
            name: optimizer._exp_avg_sq[name].index_select(0, state_rows).clone()
            for name, _, _, _ in COMPONENT_SPECS
        }

        second_grads = _components(self.rows, 3.0)
        stats = optimizer.step(
            iteration=2,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=second_grads,
            curvature_due=False,
            optimizer_step=2,
        )

        beta1, _ = self.betas
        expected_avg = beta1 * ((1.0 - beta1) * 2.0) + (1.0 - beta1) * 3.0
        for name, attr_name, _, _ in COMPONENT_SPECS:
            curvature_after = optimizer._exp_avg_sq[name].index_select(0, state_rows)
            torch.testing.assert_close(curvature_after, curvature_before[name])
            expected_step = -expected_avg / (curvature_before[name] + self.eps)
            torch.testing.assert_close(
                getattr(gaussians, attr_name).detach(),
                before[name] + expected_step,
                rtol=2e-5,
                atol=2e-6,
            )
        self.assertFalse(stats["curvature_due"])
        self.assertEqual(stats["curvature_rows"], 0)

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_disjoint_gradient_and_curvature_blocks_do_not_write_s2_parameters(
        self, _
    ):
        rows_per_block = 2
        manager = _Manager(rows_per_block * 2)
        manager.loaded_blocks = [10, 20]
        manager.block_to_gpu_slice = {
            10: slice(0, rows_per_block),
            20: slice(rows_per_block, rows_per_block * 2),
        }
        initial = _params(rows_per_block * 2)
        gaussians = _Gaussians(manager, initial, betas=self.betas, eps=self.eps)
        optimizer = GPUResidentSophiaTR(
            batch_size=1,
            block_size=rows_per_block,
            capacity_blocks=2,
            device=TEST_DEVICE,
        )
        optimizer.set_resident_blocks([10, 20])
        gradient_rows = torch.tensor([0, 1], device=TEST_DEVICE)
        curvature_rows = torch.tensor([2, 3], device=TEST_DEVICE)

        optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=gradient_rows,
            sparse_grad_components=_components(rows_per_block, 1.0),
            sparse_curvature_local_ids=torch.arange(4, device=TEST_DEVICE),
            sparse_curvature_components=_components(4, 2.0),
            curvature_due=True,
            optimizer_step=1,
        )
        before = gaussians.parameter_values()
        curvature_slot = optimizer._block_to_slot[20]
        curvature_state_rows = (
            curvature_slot * rows_per_block
            + torch.arange(rows_per_block, device=TEST_DEVICE)
        )
        curvature_state_before = optimizer._exp_avg_sq["xyz"].index_select(
            0, curvature_state_rows
        ).clone()

        stats = optimizer.step(
            iteration=11,
            gaussians=gaussians,
            sparse_grad_local_ids=gradient_rows,
            sparse_grad_components=_components(rows_per_block, 3.0),
            sparse_curvature_local_ids=curvature_rows,
            sparse_curvature_components=_components(rows_per_block, 5.0),
            curvature_due=True,
            optimizer_step=11,
        )
        after = gaussians.parameter_values()

        for name, _, _, _ in COMPONENT_SPECS:
            torch.testing.assert_close(
                after[name].index_select(0, curvature_rows),
                before[name].index_select(0, curvature_rows),
                rtol=0,
                atol=0,
            )
        self.assertFalse(torch.equal(after["xyz"][:2], before["xyz"][:2]))
        self.assertFalse(
            torch.equal(
                optimizer._exp_avg_sq["xyz"].index_select(
                    0, curvature_state_rows
                ),
                curvature_state_before,
            )
        )
        self.assertEqual(stats["updated_block_ids"], [10])
        self.assertEqual(stats["curvature_block_ids"], [20])

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_sparse_rows_lazily_decay_over_missed_optimizer_steps(self, _):
        optimizer, gaussians, _ = self._make_case()
        optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 2.0),
            sparse_curvature_components=_components(self.rows, 5.0),
            curvature_due=True,
            optimizer_step=1,
        )

        optimizer.step(
            iteration=22,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 3.0),
            curvature_due=False,
            optimizer_step=22,
        )

        beta1, beta2 = self.betas
        expected_avg = beta1**21 * ((1.0 - beta1) * 2.0) + (1.0 - beta1) * 3.0
        expected_curvature = beta2**2 * ((1.0 - beta2) * 5.0)
        state_rows = self._state_rows(optimizer)
        for name, _, _, _ in COMPONENT_SPECS:
            torch.testing.assert_close(
                optimizer._exp_avg[name].index_select(0, state_rows),
                torch.full_like(optimizer._exp_avg[name].index_select(0, state_rows), expected_avg),
            )
            torch.testing.assert_close(
                optimizer._exp_avg_sq[name].index_select(0, state_rows),
                torch.full_like(
                    optimizer._exp_avg_sq[name].index_select(0, state_rows),
                    expected_curvature,
                ),
            )

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_curvature_updates_exactly_on_1_mod_interval_boundaries(self, _):
        optimizer, gaussians, _ = self._make_case()
        zero_grads = _components(self.rows, 0.0)
        optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=zero_grads,
            sparse_curvature_components=_components(self.rows, 5.0),
            curvature_due=True,
            optimizer_step=1,
        )
        state_rows = self._state_rows(optimizer)
        beta2 = self.betas[1]
        expected = (1.0 - beta2) * 5.0

        optimizer.step(
            iteration=10,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=zero_grads,
            curvature_due=False,
            optimizer_step=10,
        )
        torch.testing.assert_close(
            optimizer._exp_avg_sq["xyz"].index_select(0, state_rows),
            torch.full((self.rows, 3), expected, device=TEST_DEVICE),
        )

        optimizer.step(
            iteration=11,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=zero_grads,
            sparse_curvature_components=_components(self.rows, 7.0),
            curvature_due=True,
            optimizer_step=11,
        )
        expected = beta2 * expected + (1.0 - beta2) * 7.0
        optimizer.step(
            iteration=20,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=zero_grads,
            curvature_due=False,
            optimizer_step=20,
        )
        optimizer.step(
            iteration=21,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=zero_grads,
            sparse_curvature_components=_components(self.rows, 11.0),
            curvature_due=True,
            optimizer_step=21,
        )
        expected = beta2 * expected + (1.0 - beta2) * 11.0
        torch.testing.assert_close(
            optimizer._exp_avg_sq["xyz"].index_select(0, state_rows),
            torch.full((self.rows, 3), expected, device=TEST_DEVICE),
        )

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_rows_use_their_own_lazy_gradient_gap(self, _):
        optimizer, gaussians, _ = self._make_case()
        optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 1.0),
            sparse_curvature_components=_components(self.rows, 1.0),
            curvature_due=True,
            optimizer_step=1,
        )
        row_zero = self.local_ids[:1]
        optimizer.step(
            iteration=2,
            gaussians=gaussians,
            sparse_grad_local_ids=row_zero,
            sparse_grad_components=_components(1, 2.0),
            curvature_due=False,
            optimizer_step=2,
        )
        optimizer.step(
            iteration=3,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 3.0),
            curvature_due=False,
            optimizer_step=3,
        )

        beta1 = self.betas[0]
        first = (1.0 - beta1) * 1.0
        expected_row_zero = beta1 * (
            beta1 * first + (1.0 - beta1) * 2.0
        ) + (1.0 - beta1) * 3.0
        expected_row_one = beta1**2 * first + (1.0 - beta1) * 3.0
        averages = optimizer._exp_avg["xyz"].index_select(
            0, self._state_rows(optimizer)
        )
        torch.testing.assert_close(
            averages[0], torch.full_like(averages[0], expected_row_zero)
        )
        torch.testing.assert_close(
            averages[1], torch.full_like(averages[1], expected_row_one)
        )

    def test_clip_failure_leaves_state_unchanged_and_allows_retry(self):
        optimizer, gaussians, initial = self._make_case()
        grads = _components(self.rows, 1.0)
        curvature = _components(self.rows, 1.0)

        with mock.patch(CLIP_TARGET, side_effect=RuntimeError("clip failed")):
            with self.assertRaisesRegex(RuntimeError, "clip failed"):
                optimizer.step(
                    iteration=1,
                    gaussians=gaussians,
                    sparse_grad_local_ids=self.local_ids,
                    sparse_grad_components=grads,
                    sparse_curvature_components=curvature,
                    curvature_due=True,
                    optimizer_step=1,
                )

        self._assert_components_close(gaussians.parameter_values(), initial)
        slot = optimizer._block_to_slot[10]
        state_rows = slot * self.block_size + torch.arange(
            self.block_size, device=TEST_DEVICE
        )
        self.assertFalse(optimizer._slot_initialized[slot])
        for name, _, _, _ in COMPONENT_SPECS:
            self.assertFalse(
                bool(optimizer._exp_avg[name].index_select(0, state_rows).any())
            )
            self.assertFalse(
                bool(optimizer._exp_avg_sq[name].index_select(0, state_rows).any())
            )
        self.assertFalse(
            bool(optimizer._curvature_initialized.index_select(0, state_rows).any())
        )
        self.assertFalse(
            bool(optimizer._last_gradient_step.index_select(0, state_rows).any())
        )

        with mock.patch(CLIP_TARGET, side_effect=_identity_clip):
            stats = optimizer.step(
                iteration=1,
                gaussians=gaussians,
                sparse_grad_local_ids=self.local_ids,
                sparse_grad_components=grads,
                sparse_curvature_components=curvature,
                curvature_due=True,
                optimizer_step=1,
            )
        self.assertEqual(stats["touched_rows"], self.rows)

    def test_late_chunk_clip_failure_leaves_state_unchanged(self):
        optimizer, gaussians, initial = self._make_case()
        optimizer.TRANSACTION_CHUNK_ROWS = 1
        grads = _components(self.rows, 1.0)
        curvature = _components(self.rows, 1.0)
        clip_calls = 0

        def fail_second_chunk(raw_params, steps, *args, **kwargs):
            nonlocal clip_calls
            del raw_params, args, kwargs
            clip_calls += 1
            if clip_calls == 2:
                raise RuntimeError("second chunk failed")
            return steps

        with mock.patch(CLIP_TARGET, side_effect=fail_second_chunk):
            with self.assertRaisesRegex(RuntimeError, "second chunk failed"):
                optimizer.step(
                    iteration=1,
                    gaussians=gaussians,
                    sparse_grad_local_ids=self.local_ids,
                    sparse_grad_components=grads,
                    sparse_curvature_components=curvature,
                    curvature_due=True,
                    optimizer_step=1,
                )

        self._assert_components_close(gaussians.parameter_values(), initial)
        slot = optimizer._block_to_slot[10]
        state_rows = self._state_rows(optimizer)
        self.assertFalse(optimizer._slot_initialized[slot])
        for name, _, _, _ in COMPONENT_SPECS:
            self.assertFalse(
                bool(optimizer._exp_avg[name].index_select(0, state_rows).any())
            )
            self.assertFalse(
                bool(optimizer._exp_avg_sq[name].index_select(0, state_rows).any())
            )
        self.assertFalse(
            bool(optimizer._curvature_initialized.index_select(0, state_rows).any())
        )

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_chunked_transaction_matches_single_chunk(self, _):
        chunked, chunked_gaussians, _ = self._make_case()
        single, single_gaussians, _ = self._make_case()
        chunked.TRANSACTION_CHUNK_ROWS = 1
        grads = _components(self.rows, 2.0)
        curvature = _components(self.rows, 5.0)

        chunked_stats = chunked.step(
            iteration=1,
            gaussians=chunked_gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=grads,
            sparse_curvature_components=curvature,
            curvature_due=True,
            optimizer_step=1,
        )
        single_stats = single.step(
            iteration=1,
            gaussians=single_gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=grads,
            sparse_curvature_components=curvature,
            curvature_due=True,
            optimizer_step=1,
        )

        self._assert_components_close(
            chunked_gaussians.parameter_values(),
            single_gaussians.parameter_values(),
        )
        chunked_rows = self._state_rows(chunked)
        single_rows = self._state_rows(single)
        for name, _, _, _ in COMPONENT_SPECS:
            torch.testing.assert_close(
                chunked._exp_avg[name].index_select(0, chunked_rows),
                single._exp_avg[name].index_select(0, single_rows),
            )
            torch.testing.assert_close(
                chunked._exp_avg_sq[name].index_select(0, chunked_rows),
                single._exp_avg_sq[name].index_select(0, single_rows),
            )
        self.assertEqual(chunked_stats, single_stats)

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_transaction_builds_only_chunk_local_candidates(self, _):
        optimizer, gaussians, _ = self._make_case()
        optimizer.TRANSACTION_CHUNK_ROWS = 1
        original_build = optimizer._build_transaction_chunk
        built_row_counts = []

        def track_build(**kwargs):
            built_row_counts.append(int(kwargs["state_rows"].numel()))
            return original_build(**kwargs)

        with mock.patch.object(
            optimizer, "_build_transaction_chunk", side_effect=track_build
        ):
            optimizer.step(
                iteration=1,
                gaussians=gaussians,
                sparse_grad_local_ids=self.local_ids,
                sparse_grad_components=_components(self.rows, 1.0),
                sparse_curvature_components=_components(self.rows, 1.0),
                curvature_due=True,
                optimizer_step=1,
            )

        self.assertEqual(built_row_counts, [1, 1, 1, 1])

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_commit_preflight_failure_clears_pending_and_poisons_optimizer(self, _):
        optimizer, gaussians, initial = self._make_case()
        prepared = optimizer.prepare_step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 1.0),
            sparse_curvature_components=_components(self.rows, 1.0),
            curvature_due=True,
            optimizer_step=1,
        )

        with mock.patch.object(
            optimizer,
            "_validated_parameter_views",
            side_effect=RuntimeError("preflight failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                optimizer.commit_step(prepared)

        self.assertEqual(prepared.status, "failed")
        self.assertIsNone(optimizer._pending_transaction)
        self.assertTrue(optimizer._transaction_corrupted)
        self._assert_components_close(gaussians.parameter_values(), initial)
        with self.assertRaisesRegex(RuntimeError, "optimizer is unusable"):
            optimizer.prepare_step(
                iteration=1,
                gaussians=gaussians,
                sparse_grad_local_ids=self.local_ids,
                sparse_grad_components=_components(self.rows, 1.0),
                sparse_curvature_components=_components(self.rows, 1.0),
                curvature_due=True,
                optimizer_step=1,
            )

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_late_commit_failure_poisons_optimizer_and_prevents_retry(self, _):
        optimizer, gaussians, initial = self._make_case()
        optimizer.TRANSACTION_CHUNK_ROWS = 1
        original_commit = optimizer._commit_transaction_chunk
        commit_calls = 0

        def fail_second_commit(**kwargs):
            nonlocal commit_calls
            commit_calls += 1
            if commit_calls == 2:
                raise RuntimeError("second commit failed")
            return original_commit(**kwargs)

        with mock.patch.object(
            optimizer,
            "_commit_transaction_chunk",
            side_effect=fail_second_commit,
        ):
            with self.assertRaisesRegex(RuntimeError, "second commit failed"):
                optimizer.step(
                    iteration=1,
                    gaussians=gaussians,
                    sparse_grad_local_ids=self.local_ids,
                    sparse_grad_components=_components(self.rows, 1.0),
                    sparse_curvature_components=_components(self.rows, 1.0),
                    curvature_due=True,
                    optimizer_step=1,
                )

        self.assertTrue(optimizer._transaction_corrupted)
        self.assertFalse(any(optimizer._slot_initialized))
        self.assertFalse(
            torch.equal(gaussians._xyz.detach()[0], initial["xyz"][0])
        )
        torch.testing.assert_close(
            gaussians._xyz.detach()[1], initial["xyz"][1], rtol=0, atol=0
        )
        with self.assertRaisesRegex(RuntimeError, "optimizer is unusable"):
            optimizer.step(
                iteration=1,
                gaussians=gaussians,
                sparse_grad_local_ids=self.local_ids,
                sparse_grad_components=_components(self.rows, 1.0),
                sparse_curvature_components=_components(self.rows, 1.0),
                curvature_due=True,
                optimizer_step=1,
            )

    def test_gap_in_working_set_slices_is_rejected(self):
        manager = _Manager(6)
        manager.loaded_blocks = [10, 20]
        manager.block_to_gpu_slice = {10: slice(0, 2), 20: slice(4, 6)}
        gaussians = _Gaussians(manager, _params(6), betas=self.betas, eps=self.eps)
        optimizer = GPUResidentSophiaTR(
            batch_size=1,
            block_size=self.block_size,
            capacity_blocks=2,
            device=TEST_DEVICE,
        )
        optimizer.set_resident_blocks([10, 20])
        gap_row = torch.tensor([3], device=TEST_DEVICE)

        with self.assertRaisesRegex(RuntimeError, "not contained"):
            optimizer.step(
                iteration=1,
                gaussians=gaussians,
                sparse_grad_local_ids=gap_row,
                sparse_grad_components=_components(1, 1.0),
                sparse_curvature_components=_components(1, 1.0),
                curvature_due=True,
                optimizer_step=1,
            )
        self.assertFalse(any(optimizer._slot_initialized))

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_slot_reuse_clears_all_sophia_state(self, _):
        optimizer, gaussians, _ = self._make_case()
        optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 1.0),
            sparse_curvature_components=_components(self.rows, 1.0),
            curvature_due=True,
            optimizer_step=1,
        )
        old_slot = optimizer._block_to_slot[10]
        old_slot_rows = old_slot * self.block_size + torch.arange(
            self.block_size, device=TEST_DEVICE
        )
        for name, _, _, _ in COMPONENT_SPECS:
            optimizer._exp_avg[name].index_fill_(0, old_slot_rows, 7.0)
            optimizer._exp_avg_sq[name].index_fill_(0, old_slot_rows, 11.0)
        optimizer._curvature_initialized.index_fill_(0, old_slot_rows, True)
        optimizer._last_gradient_step.index_fill_(0, old_slot_rows, 13)
        optimizer._last_curvature_update.index_fill_(0, old_slot_rows, 17)

        gaussians.gpu_working_set_manager.replace_block(20, self.rows)
        optimizer.set_resident_blocks([20])
        new_slot = optimizer._block_to_slot[20]
        self.assertEqual(new_slot, old_slot)
        state_rows = new_slot * self.block_size + torch.arange(
            self.block_size, device=TEST_DEVICE
        )
        for name, _, _, _ in COMPONENT_SPECS:
            self.assertFalse(bool(optimizer._exp_avg[name].index_select(0, state_rows).any()))
            self.assertFalse(
                bool(optimizer._exp_avg_sq[name].index_select(0, state_rows).any())
            )
        self.assertFalse(
            bool(optimizer._curvature_initialized.index_select(0, state_rows).any())
        )
        self.assertFalse(bool(optimizer._last_gradient_step.index_select(0, state_rows).any()))
        self.assertFalse(
            bool(optimizer._last_curvature_update.index_select(0, state_rows).any())
        )

        before = gaussians.parameter_values()
        stats = optimizer.step(
            iteration=2,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 2.0),
            curvature_due=False,
            optimizer_step=2,
        )
        self._assert_components_close(gaussians.parameter_values(), before)
        self.assertEqual(stats["rows_skipped_without_curvature"], self.rows)

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_storage_growth_preserves_existing_block_state(self, _):
        optimizer, gaussians, _ = self._make_case()
        optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 2.0),
            sparse_curvature_components=_components(self.rows, 3.0),
            curvature_due=True,
            optimizer_step=1,
        )
        old_state_rows = self._state_rows(optimizer)
        averages_before = {
            name: optimizer._exp_avg[name].index_select(0, old_state_rows).clone()
            for name, _, _, _ in COMPONENT_SPECS
        }
        curvatures_before = {
            name: optimizer._exp_avg_sq[name].index_select(0, old_state_rows).clone()
            for name, _, _, _ in COMPONENT_SPECS
        }
        readiness_before = optimizer._curvature_initialized.index_select(
            0, old_state_rows
        ).clone()
        gradient_steps_before = optimizer._last_gradient_step.index_select(
            0, old_state_rows
        ).clone()
        curvature_steps_before = optimizer._last_curvature_update.index_select(
            0, old_state_rows
        ).clone()

        optimizer.set_resident_blocks([10, 20])
        new_state_rows = self._state_rows(optimizer)
        for name, _, _, _ in COMPONENT_SPECS:
            torch.testing.assert_close(
                optimizer._exp_avg[name].index_select(0, new_state_rows),
                averages_before[name],
            )
            torch.testing.assert_close(
                optimizer._exp_avg_sq[name].index_select(0, new_state_rows),
                curvatures_before[name],
            )
        torch.testing.assert_close(
            optimizer._curvature_initialized.index_select(0, new_state_rows),
            readiness_before,
        )
        torch.testing.assert_close(
            optimizer._last_gradient_step.index_select(0, new_state_rows),
            gradient_steps_before,
        )
        torch.testing.assert_close(
            optimizer._last_curvature_update.index_select(0, new_state_rows),
            curvature_steps_before,
        )
        incoming_slot = optimizer._block_to_slot[20]
        incoming_rows = incoming_slot * self.block_size + self.local_ids
        self.assertFalse(
            bool(optimizer._curvature_initialized.index_select(0, incoming_rows).any())
        )

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_storage_growth_failure_is_atomic_and_retryable(self, _):
        optimizer, gaussians, _ = self._make_case()
        optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 2.0),
            sparse_curvature_components=_components(self.rows, 3.0),
            curvature_due=True,
            optimizer_step=1,
        )

        tensor_state = {
            "exp_avg": dict(optimizer._exp_avg),
            "exp_avg_sq": dict(optimizer._exp_avg_sq),
            "bias": optimizer._bias_correction1,
            "denom": optimizer._denom_scale,
            "curvature_initialized": optimizer._curvature_initialized,
            "last_gradient_step": optimizer._last_gradient_step,
            "last_curvature_update": optimizer._last_curvature_update,
        }
        python_state = {
            "allocated_slots": optimizer._allocated_slots,
            "resident_blocks": set(optimizer._resident_blocks),
            "resident_streaks": dict(optimizer._resident_streaks),
            "block_to_slot": dict(optimizer._block_to_slot),
            "slot_to_block": list(optimizer._slot_to_block),
            "free_slots": list(optimizer._free_slots),
            "slot_steps": list(optimizer._slot_steps),
            "slot_initialized": list(optimizer._slot_initialized),
            "slot_row_counts": list(optimizer._slot_row_counts),
            "stats": optimizer.get_stats(),
        }

        with mock.patch(
            "strategies.tide_engine.gpu_resident_optimizer.torch.zeros",
            side_effect=torch.OutOfMemoryError("simulated metadata OOM"),
        ):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "metadata OOM"):
                optimizer.set_resident_blocks([20, 30])

        for name, tensor in tensor_state["exp_avg"].items():
            self.assertIs(optimizer._exp_avg[name], tensor)
        for name, tensor in tensor_state["exp_avg_sq"].items():
            self.assertIs(optimizer._exp_avg_sq[name], tensor)
        self.assertIs(optimizer._bias_correction1, tensor_state["bias"])
        self.assertIs(optimizer._denom_scale, tensor_state["denom"])
        self.assertIs(
            optimizer._curvature_initialized,
            tensor_state["curvature_initialized"],
        )
        self.assertIs(
            optimizer._last_gradient_step,
            tensor_state["last_gradient_step"],
        )
        self.assertIs(
            optimizer._last_curvature_update,
            tensor_state["last_curvature_update"],
        )
        self.assertEqual(
            optimizer._allocated_slots, python_state["allocated_slots"]
        )
        self.assertEqual(
            optimizer._resident_blocks, python_state["resident_blocks"]
        )
        self.assertEqual(
            optimizer._resident_streaks, python_state["resident_streaks"]
        )
        self.assertEqual(optimizer._block_to_slot, python_state["block_to_slot"])
        self.assertEqual(optimizer._slot_to_block, python_state["slot_to_block"])
        self.assertEqual(optimizer._free_slots, python_state["free_slots"])
        self.assertEqual(optimizer._slot_steps, python_state["slot_steps"])
        self.assertEqual(
            optimizer._slot_initialized, python_state["slot_initialized"]
        )
        self.assertEqual(
            optimizer._slot_row_counts, python_state["slot_row_counts"]
        )
        self.assertEqual(optimizer.get_stats(), python_state["stats"])

        optimizer.set_resident_blocks([20, 30])
        self.assertEqual(optimizer._allocated_slots, 2)
        self.assertEqual(optimizer._resident_blocks, {20, 30})

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_resident_slot_clear_failure_keeps_mapping_and_poisons(self, _):
        optimizer, gaussians, _ = self._make_case()
        optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 2.0),
            sparse_curvature_components=_components(self.rows, 3.0),
            curvature_due=True,
            optimizer_step=1,
        )
        mapping_before = dict(optimizer._block_to_slot)
        stats_before = optimizer.get_stats()
        original_clear = optimizer._clear_slots

        def fail_after_clear(slots):
            original_clear(slots)
            raise torch.OutOfMemoryError("simulated slot clear OOM")

        with mock.patch.object(
            optimizer, "_clear_slots", side_effect=fail_after_clear
        ):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "slot clear OOM"):
                optimizer.set_resident_blocks([20])

        self.assertEqual(optimizer._resident_blocks, {10})
        self.assertEqual(optimizer._block_to_slot, mapping_before)
        self.assertEqual(optimizer.get_stats(), stats_before)
        self.assertTrue(optimizer._residency_corrupted)
        with self.assertRaisesRegex(RuntimeError, "unusable"):
            optimizer.set_resident_blocks([20])
        with self.assertRaisesRegex(RuntimeError, "unusable"):
            optimizer.step(
                iteration=2,
                gaussians=gaussians,
                sparse_grad_local_ids=self.local_ids,
                sparse_grad_components=_components(self.rows, 1.0),
                curvature_due=False,
                optimizer_step=2,
            )

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_zero_curvature_coordinate_unfreezes_after_positive_sample(self, _):
        optimizer, gaussians, initial = self._make_case()
        first_curvature = _components(self.rows, 1.0)
        first_curvature["features_rest"].zero_()
        optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 1.0),
            sparse_curvature_components=first_curvature,
            curvature_due=True,
            optimizer_step=1,
        )
        torch.testing.assert_close(
            gaussians._features_rest.detach(), initial["features_rest"]
        )

        optimizer.step(
            iteration=11,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 1.0),
            sparse_curvature_components=_components(self.rows, 1.0),
            curvature_due=True,
            optimizer_step=11,
        )
        self.assertTrue(bool((gaussians._features_rest.detach() != 0.0).any()))

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_batch_sum_normalization_matches_single_sample(self, _):
        optimizer_one, gaussians_one, _ = self._make_case(batch_size=1)
        optimizer_four, gaussians_four, _ = self._make_case(batch_size=4)
        grads = _components(self.rows, 2.0)
        curvature = _components(self.rows, 5.0)

        optimizer_one.step(
            iteration=1,
            gaussians=gaussians_one,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=grads,
            sparse_curvature_components=curvature,
            curvature_due=True,
            optimizer_step=1,
        )
        optimizer_four.step(
            iteration=1,
            gaussians=gaussians_four,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components={name: value * 4.0 for name, value in grads.items()},
            sparse_curvature_components={
                name: value * 4.0 for name, value in curvature.items()
            },
            curvature_due=True,
            optimizer_step=1,
        )

        self._assert_components_close(
            gaussians_one.parameter_values(), gaussians_four.parameter_values()
        )
        state_rows_one = self._state_rows(optimizer_one)
        state_rows_four = self._state_rows(optimizer_four)
        for name, _, _, _ in COMPONENT_SPECS:
            torch.testing.assert_close(
                optimizer_one._exp_avg[name].index_select(0, state_rows_one),
                optimizer_four._exp_avg[name].index_select(0, state_rows_four),
            )
            torch.testing.assert_close(
                optimizer_one._exp_avg_sq[name].index_select(0, state_rows_one),
                optimizer_four._exp_avg_sq[name].index_select(0, state_rows_four),
            )

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_cold_rows_on_non_curvature_step_are_held_fixed_and_counted(
        self, clip_mock
    ):
        optimizer, gaussians, initial = self._make_case()

        stats = optimizer.step(
            iteration=2,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 3.0),
            curvature_due=False,
            optimizer_step=2,
        )

        proposals = clip_mock.call_args.args[1]
        for name, _, _, _ in COMPONENT_SPECS:
            torch.testing.assert_close(
                proposals[name], torch.zeros_like(proposals[name]), rtol=0, atol=0
            )
        self._assert_components_close(gaussians.parameter_values(), initial)
        self.assertEqual(stats["rows_skipped_without_curvature"], self.rows)
        self.assertEqual(stats["clipped_values"], 0)
        state_rows = self._state_rows(optimizer)
        self.assertFalse(
            bool(optimizer._curvature_initialized.index_select(0, state_rows).any())
        )
        self.assertEqual(
            optimizer.get_stats()["rows_skipped_without_curvature_total"], self.rows
        )

    @mock.patch(CLIP_TARGET, side_effect=_identity_clip)
    def test_zero_curvature_coordinates_remain_fixed_after_row_initialization(self, _):
        optimizer, gaussians, initial = self._make_case()
        curvature = _components(self.rows, 1.0)
        curvature["features_rest"].zero_()

        optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 1.0),
            sparse_curvature_components=curvature,
            curvature_due=True,
            optimizer_step=1,
        )

        torch.testing.assert_close(
            gaussians._features_rest.detach(),
            initial["features_rest"],
            rtol=0,
            atol=0,
        )
        self.assertFalse(
            bool(
                optimizer._exp_avg_sq["features_rest"]
                .index_select(0, self._state_rows(optimizer))
                .any()
            )
        )

    def test_default_negative_quaternion_cap_runs_real_hellinger_clip(self):
        optimizer, gaussians, initial = self._make_case()

        stats = optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=self.local_ids,
            sparse_grad_components=_components(self.rows, 0.0),
            sparse_curvature_components=_components(self.rows, 1.0),
            curvature_due=True,
            optimizer_step=1,
        )

        self._assert_components_close(gaussians.parameter_values(), initial)
        self.assertEqual(stats["rows_skipped_without_curvature"], 0)


if __name__ == "__main__":
    unittest.main()
