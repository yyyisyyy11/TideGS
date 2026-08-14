import itertools
import math
import unittest

import torch

from strategies.tide_engine.sophia_tr_math import (
    _quat_trace_coefficients,
    build_3dgs2_residual_tuple_from_ssim_map,
    build_3dgs2_residual_vector,
    build_3dgs2_residual_vector_from_ssim_map,
    clip_hellinger_step,
    estimate_residual_vjp_curvature,
    exponential_schedule,
    optimizer_step_from_iteration,
    rademacher_like,
    resolve_curvature_schedule,
    should_update_curvature,
)
from utils.loss_utils import ssim
from utils.sh_utils import eval_sh


def _identity_raw(rows=1, dtype=torch.float64):
    rotation = torch.zeros((rows, 4), dtype=dtype)
    rotation[:, 0] = 1.0
    return {
        "xyz": torch.zeros((rows, 3), dtype=dtype),
        "opacity": torch.zeros((rows, 1), dtype=dtype),
        "scaling": torch.zeros((rows, 3), dtype=dtype),
        "rotation": rotation,
        "features_dc": torch.zeros((rows, 3), dtype=dtype),
        "features_rest": torch.zeros((rows, 45), dtype=dtype),
    }


def _steps_like(params, value):
    return {name: torch.full_like(tensor, value) for name, tensor in params.items()}


def _normalized_rotation(raw_quaternion):
    quaternion = raw_quaternion / torch.linalg.vector_norm(raw_quaternion)
    w, x, y, z = quaternion.unbind()
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y))),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x))),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))),
        )
    )


class SophiaTRScheduleTest(unittest.TestCase):
    def test_optimizer_steps_account_for_tide_batch_size(self):
        self.assertEqual(optimizer_step_from_iteration(1, 4), 1)
        self.assertEqual(optimizer_step_from_iteration(4, 4), 1)
        self.assertEqual(optimizer_step_from_iteration(5, 4), 2)
        self.assertEqual(optimizer_step_from_iteration(40, 4), 10)
        self.assertEqual(optimizer_step_from_iteration(41, 4), 11)
        with self.assertRaisesRegex(ValueError, ">= 1"):
            optimizer_step_from_iteration(0, 4)
        with self.assertRaisesRegex(ValueError, ">= 1"):
            optimizer_step_from_iteration(1, 0)

    def test_curvature_uses_the_paper_modulo_schedule(self):
        due = [step for step in range(1, 23) if should_update_curvature(step, 10)]
        self.assertEqual(due, [1, 11, 21])
        with self.assertRaisesRegex(ValueError, ">= 1"):
            should_update_curvature(1, 0)

    def test_shared_schedule_resolves_and_validates_explicit_clock(self):
        self.assertEqual(
            resolve_curvature_schedule(
                iteration=41,
                batch_size=4,
                interval=10,
            ),
            (11, True),
        )
        self.assertEqual(
            resolve_curvature_schedule(
                iteration=44,
                batch_size=4,
                interval=10,
                optimizer_step=11,
            ),
            (11, True),
        )
        with self.assertRaisesRegex(ValueError, "iteration-derived"):
            resolve_curvature_schedule(
                iteration=45,
                batch_size=4,
                interval=10,
                optimizer_step=11,
            )
        with self.assertRaisesRegex(TypeError, "not bool"):
            resolve_curvature_schedule(
                iteration=1,
                batch_size=4,
                interval=10,
                optimizer_step=True,
            )

    def test_exponential_schedule_has_exact_endpoints_and_geometric_midpoint(self):
        self.assertEqual(exponential_schedule(1e-6, 1e-8, 0, 30_000), 1e-6)
        self.assertAlmostEqual(
            exponential_schedule(1e-6, 1e-8, 15_000, 30_000), 1e-7, places=18
        )
        self.assertEqual(
            exponential_schedule(1e-6, 1e-8, 30_000, 30_000), 1e-8
        )
        self.assertEqual(
            exponential_schedule(1e-6, 1e-8, 40_000, 30_000), 1e-8
        )


class SophiaTRResidualTest(unittest.TestCase):
    def test_precomputed_ssim_map_builds_reusable_tuple_and_vector(self):
        image = torch.tensor(
            [[[0.0, 0.25], [0.5, 1.0]]], dtype=torch.float64
        )
        target = torch.tensor(
            [[[0.5, 0.25], [0.0, 0.75]]], dtype=torch.float64
        )
        ssim_map = torch.tensor(
            [[[0.8, 1.0], [0.2, -0.5]]], dtype=torch.float64
        )
        weight = 0.25

        residual_tuple = build_3dgs2_residual_tuple_from_ssim_map(
            image, target, ssim_map, weight
        )
        residual_vector = build_3dgs2_residual_vector_from_ssim_map(
            image, target, ssim_map, weight
        )
        expected_loss = (1.0 - weight) * (image - target).abs().mean()
        expected_loss = expected_loss + weight * (1.0 - ssim_map).mean()

        self.assertEqual(len(residual_tuple), 2)
        self.assertEqual(residual_tuple[0].shape, image.shape)
        torch.testing.assert_close(
            residual_vector,
            torch.cat(tuple(value.reshape(-1) for value in residual_tuple)),
        )
        torch.testing.assert_close(
            0.5 * residual_vector.square().sum(), expected_loss
        )

    def test_half_squared_norm_matches_l1_dssim_loss(self):
        generator = torch.Generator().manual_seed(19)
        image = torch.rand((3, 13, 17), generator=generator, dtype=torch.float64)
        target = torch.rand((3, 13, 17), generator=generator, dtype=torch.float64)
        weight = 0.2

        residual = build_3dgs2_residual_vector(image, target, weight)
        expected = (1.0 - weight) * (image - target).abs().mean()
        expected = expected + weight * (1.0 - ssim(image, target))

        self.assertEqual(residual.shape, (2 * image.numel(),))
        torch.testing.assert_close(
            0.5 * residual.square().sum(), expected, rtol=2e-7, atol=2e-9
        )
        self.assertTrue(torch.isfinite(residual).all())

    def test_identical_images_have_zero_residual_and_finite_gradient(self):
        image = torch.full((1, 3, 11, 11), 0.25, dtype=torch.float64, requires_grad=True)
        target = image.detach().clone()
        residual = build_3dgs2_residual_vector(image, target)
        loss = 0.5 * residual.square().sum()
        loss.backward()

        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=1e-12)
        self.assertTrue(torch.isfinite(image.grad).all())
        torch.testing.assert_close(image.grad, torch.zeros_like(image.grad), rtol=0, atol=0)

    def test_random_residual_vjp_is_finite_at_zero_loss_components(self):
        image = torch.tensor(
            [[[0.25, 0.5], [0.75, 0.25]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        target = torch.tensor(
            [[[0.25, 0.0], [0.5, 0.25]]], dtype=torch.float64
        )
        ssim_map = 1.0 - 0.1 * (image - target).square()
        residuals = build_3dgs2_residual_tuple_from_ssim_map(
            image, target, ssim_map
        )
        probes = rademacher_like(
            residuals, generator=torch.Generator().manual_seed(31)
        )

        (vjp,) = torch.autograd.grad(
            outputs=residuals, inputs=(image,), grad_outputs=probes
        )

        self.assertTrue(torch.isfinite(vjp).all())


class RademacherTest(unittest.TestCase):
    def test_samples_are_signed_one_and_seed_reproducible(self):
        reference = {
            "xyz": torch.zeros((5, 3), dtype=torch.float64),
            "opacity": torch.zeros((5, 1), dtype=torch.float64),
        }
        first = rademacher_like(
            reference, generator=torch.Generator().manual_seed(7)
        )
        second = rademacher_like(
            reference, generator=torch.Generator().manual_seed(7)
        )
        for name in reference:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
            self.assertTrue(torch.all((first[name] == -1.0) | (first[name] == 1.0)))
            torch.testing.assert_close(reference[name], torch.zeros_like(reference[name]))


class ResidualSpaceHutchinsonTest(unittest.TestCase):
    @staticmethod
    def _jacobian():
        return torch.tensor(
            [
                [1.0, 2.0, -1.0],
                [0.5, -1.0, 3.0],
                [-2.0, 0.25, 1.5],
                [1.25, -0.75, 0.5],
            ],
            dtype=torch.float64,
        )

    def test_complete_residual_probe_expectation_is_exactly_unbiased(self):
        jacobian = self._jacobian()
        probes = torch.tensor(
            list(itertools.product((-1.0, 1.0), repeat=jacobian.shape[0])),
            dtype=jacobian.dtype,
        )

        estimates = (probes @ jacobian).square()
        exact_diagonal = jacobian.square().sum(dim=0)

        torch.testing.assert_close(
            estimates.mean(dim=0), exact_diagonal, rtol=0, atol=1e-12
        )

    def test_monte_carlo_residual_probes_converge_to_gauss_newton_diagonal(self):
        jacobian = self._jacobian()
        probes = rademacher_like(
            torch.empty((20_000, jacobian.shape[0]), dtype=jacobian.dtype),
            generator=torch.Generator().manual_seed(2026),
        )

        estimate = (probes @ jacobian).square().mean(dim=0)
        exact_diagonal = jacobian.square().sum(dim=0)

        torch.testing.assert_close(estimate, exact_diagonal, rtol=0.02, atol=0.03)

    def test_runtime_helper_matches_seeded_probe_and_preserves_graph(self):
        jacobian = self._jacobian()
        parameters = torch.tensor(
            [0.25, -0.5, 1.5], dtype=torch.float64, requires_grad=True
        )
        residual = jacobian @ parameters
        expected_probe = rademacher_like(
            (residual,), generator=torch.Generator().manual_seed(73)
        )[0]
        expected = (jacobian.T @ expected_probe).square()

        (actual,) = estimate_residual_vjp_curvature(
            (residual,),
            (parameters,),
            1,
            generator=torch.Generator().manual_seed(73),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)

        loss = 0.5 * residual.square().sum()
        loss.backward()
        torch.testing.assert_close(parameters.grad, jacobian.T @ residual)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_ssim_vjp_then_loss_backward_on_cuda(self):
        try:
            from clm_kernels import FusedSSIMMap
        except ImportError as error:
            self.skipTest(f"clm_kernels extension is unavailable: {error}")

        image = torch.linspace(
            0.05, 0.95, 3 * 16 * 16, device="cuda", dtype=torch.float32
        ).reshape(1, 3, 16, 16).requires_grad_(True)
        target = image.detach().flip(-1).contiguous()
        ssim_map = FusedSSIMMap.apply(
            0.01**2, 0.03**2, image, target, "same", True
        )
        residuals = build_3dgs2_residual_tuple_from_ssim_map(
            image, target, ssim_map
        )

        (curvature,) = estimate_residual_vjp_curvature(
            residuals, (image,), 2
        )
        self.assertTrue(torch.isfinite(curvature).all())
        self.assertTrue(torch.all(curvature >= 0.0))

        loss = 0.5 * sum(value.square().sum() for value in residuals)
        loss.backward()
        self.assertIsNotNone(image.grad)
        self.assertTrue(torch.isfinite(image.grad).all())


class HellingerClipTest(unittest.TestCase):
    def test_identity_case_matches_paper_parameterwise_bounds(self):
        params = _identity_raw()
        steps = _steps_like(params, 10.0)
        steps["scaling"][0] = torch.tensor([10.0, -10.0, 0.0], dtype=torch.float64)
        steps["features_dc"][0] = torch.tensor([10.0, -10.0, 0.0], dtype=torch.float64)
        steps["features_rest"].zero_()
        epsilon = 0.003

        clipped = clip_hellinger_step(params, steps, epsilon)

        mass = 0.5
        position_bound = math.sqrt(-8.0 * math.log(1.0 - epsilon / mass))
        scale_delta = math.sqrt(2.0 * epsilon / mass)
        opacity_delta = math.sqrt(4.0 * mass * epsilon)
        opacity_bound = math.log((mass + opacity_delta) / (mass - opacity_delta))
        color_delta = math.sqrt(4.0 * 0.5 * epsilon / mass)
        dc_bound = color_delta / 0.28209479177387814

        torch.testing.assert_close(
            clipped["xyz"], torch.full_like(clipped["xyz"], position_bound)
        )
        torch.testing.assert_close(
            clipped["scaling"][0, 0],
            torch.tensor(math.log1p(scale_delta), dtype=torch.float64),
        )
        torch.testing.assert_close(
            clipped["scaling"][0, 1],
            torch.tensor(math.log1p(-scale_delta), dtype=torch.float64),
        )
        self.assertEqual(clipped["scaling"][0, 2].item(), 0.0)
        torch.testing.assert_close(
            clipped["opacity"], torch.full_like(clipped["opacity"], opacity_bound)
        )
        torch.testing.assert_close(
            clipped["features_dc"][0, :2],
            torch.tensor([dc_bound, -dc_bound], dtype=torch.float64),
        )
        self.assertEqual(clipped["features_dc"][0, 2].item(), 0.0)
        torch.testing.assert_close(
            clipped["features_rest"], torch.zeros_like(clipped["features_rest"])
        )
        torch.testing.assert_close(clipped["rotation"], steps["rotation"])

    def test_position_bound_uses_rotated_covariance_diagonal(self):
        params = _identity_raw()
        params["scaling"][0] = torch.log(
            torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
        )
        params["rotation"][0] = torch.tensor(
            [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
            dtype=torch.float64,
        )
        steps = _steps_like(params, 100.0)
        epsilon = 1e-4

        clipped = clip_hellinger_step(params, steps, epsilon)

        covariance_diagonal = torch.tensor([9.0, 4.0, 25.0], dtype=torch.float64)
        expected = torch.sqrt(
            -8.0 * covariance_diagonal * math.log(1.0 - epsilon / 0.5)
        ).reshape(1, 3)
        torch.testing.assert_close(clipped["xyz"], expected)

    def test_extreme_valid_log_scale_respects_the_physical_relative_bound(self):
        params = _identity_raw(rows=2)
        params["scaling"].fill_(-100.0)
        steps = _steps_like(params, 0.0)
        steps["scaling"][0].fill_(1e6)
        steps["scaling"][1].fill_(-1e6)
        epsilon = 1e-6

        clipped = clip_hellinger_step(params, steps, epsilon)

        before = torch.exp(params["scaling"])
        after = torch.exp(params["scaling"] + clipped["scaling"])
        opacity = torch.sigmoid(params["opacity"])
        bound = before * torch.sqrt(2.0 * epsilon / opacity)
        self.assertTrue(torch.all((after - before).abs() <= bound * (1.0 + 1e-10)))

    def test_subnormal_scale_and_degenerate_quaternion_freeze_geometry(self):
        params = _identity_raw(rows=2, dtype=torch.float32)
        params["scaling"][0].fill_(-100.0)
        params["rotation"][1].zero_()
        steps = _steps_like(params, 1.0)

        clipped = clip_hellinger_step(params, steps, 1e-6)

        for name in ("xyz", "scaling", "rotation"):
            torch.testing.assert_close(
                clipped[name], torch.zeros_like(clipped[name]), rtol=0, atol=0
            )

    def test_saturated_zero_opacity_does_not_jump_in_logit_space(self):
        params = _identity_raw()
        params["opacity"].fill_(-1000.0)
        steps = _steps_like(params, 0.0)
        steps["opacity"].fill_(1000.0)

        clipped = clip_hellinger_step(params, steps, 1e-6)

        torch.testing.assert_close(
            clipped["opacity"], torch.zeros_like(clipped["opacity"]), rtol=0, atol=0
        )

    def test_transparent_scale_step_does_not_underflow(self):
        params = _identity_raw(dtype=torch.float32)
        params["opacity"].fill_(-20.0)
        steps = _steps_like(params, 0.0)
        steps["scaling"].fill_(-1e20)

        clipped = clip_hellinger_step(params, steps, 1e-6)

        after = torch.exp(params["scaling"] + clipped["scaling"])
        self.assertTrue(torch.all(torch.isfinite(after)))
        self.assertTrue(torch.all(after >= 2.0 * torch.finfo(torch.float32).tiny))

    def test_scale_updates_stay_inside_reusable_numeric_range(self):
        params = _identity_raw(rows=2, dtype=torch.float32)
        finfo = torch.finfo(torch.float32)
        minimum = 2.0 * finfo.tiny
        maximum = 0.5 * math.sqrt(finfo.max)
        params["scaling"][0].fill_(math.log(3.0 * finfo.tiny))
        params["scaling"][1].fill_(math.log(0.9 * maximum))
        steps = _steps_like(params, 0.0)
        steps["scaling"][0].fill_(-1e20)
        steps["scaling"][1].fill_(1e20)

        clipped = clip_hellinger_step(params, steps, 0.0625)

        after = torch.exp(params["scaling"] + clipped["scaling"])
        self.assertTrue(torch.all(after[0] >= minimum))
        self.assertTrue(torch.all(after[1] <= maximum))
        self.assertTrue(torch.all(torch.isfinite(after.square())))

    def test_quaternion_norm_cap_is_disabled_by_default_and_configurable(self):
        params = _identity_raw()
        steps = _steps_like(params, 0.0)
        steps["rotation"].fill_(10.0)

        default = clip_hellinger_step(params, steps, 1e-6)
        guarded = clip_hellinger_step(params, steps, 1e-6, quat_norm_tr=0.01)

        torch.testing.assert_close(default["rotation"], steps["rotation"])
        self.assertTrue(torch.all(guarded["rotation"].abs() <= 0.01 + 1e-12))

    def test_anisotropic_quaternion_beta_matches_autograd_second_derivative(self):
        raw_quaternion = torch.tensor(
            [1.4, 0.4, -0.6, 1.2], dtype=torch.float64, requires_grad=True
        )
        scaling = torch.tensor([0.7, 1.5, 3.0], dtype=torch.float64)
        base_rotation = _normalized_rotation(raw_quaternion.detach())
        relative_rotation = base_rotation @ _normalized_rotation(raw_quaternion).T
        scaled_relative = (
            scaling[:, None] * relative_rotation / scaling[None, :]
        )
        trace_expression = scaled_relative.square().sum()
        first_derivative = torch.autograd.grad(
            trace_expression, raw_quaternion, create_graph=True
        )[0]
        autograd_beta = torch.stack(
            tuple(
                torch.autograd.grad(
                    first_derivative[index],
                    raw_quaternion,
                    retain_graph=True,
                )[0][index]
                for index in range(4)
            )
        )

        closed_form_beta = _quat_trace_coefficients(
            raw_quaternion.detach().reshape(1, 4), scaling.reshape(1, 3)
        )[0]

        self.assertTrue(torch.all(autograd_beta > 0.0))
        torch.testing.assert_close(
            closed_form_beta, autograd_beta, rtol=2e-10, atol=2e-10
        )

    def test_combined_sh_update_uses_one_provable_scale_per_color(self):
        params = _identity_raw()
        steps = _steps_like(params, 0.0)
        steps["features_dc"].fill_(1.0)
        steps["features_rest"].fill_(1.0)
        epsilon = 1e-4

        clipped = clip_hellinger_step(params, steps, epsilon)

        degree_bounds = [
            math.sqrt((2 * degree + 1) / (4.0 * math.pi))
            for degree in range(4)
        ]
        change_bound = sum(
            bound * math.sqrt(width)
            for bound, width in zip(degree_bounds, (1, 3, 5, 7))
        )
        physical_radius = math.sqrt(4.0 * 0.5 * epsilon / 0.5)
        expected_scale = physical_radius / change_bound

        torch.testing.assert_close(
            clipped["features_dc"],
            torch.full_like(clipped["features_dc"], expected_scale),
        )
        torch.testing.assert_close(
            clipped["features_rest"],
            torch.full_like(clipped["features_rest"], expected_scale),
        )

    def test_sh_update_satisfies_color_bound_for_sampled_view_directions(self):
        params = _identity_raw()
        params["features_dc"][0] = torch.tensor(
            [2.0, 1.5, 1.0], dtype=torch.float64
        )
        generator = torch.Generator().manual_seed(2027)
        params["features_rest"].copy_(
            0.02
            * torch.randn(
                params["features_rest"].shape,
                generator=generator,
                dtype=torch.float64,
            )
        )
        steps = _steps_like(params, 0.0)
        steps["features_dc"].copy_(
            torch.tensor([[2.0, -3.0, 4.0]], dtype=torch.float64)
        )
        steps["features_rest"].copy_(
            torch.randn(
                steps["features_rest"].shape,
                generator=generator,
                dtype=torch.float64,
            )
        )
        epsilon = 1e-4

        clipped = clip_hellinger_step(params, steps, epsilon)
        coefficients = torch.cat(
            (
                params["features_dc"].reshape(1, 1, 3),
                params["features_rest"].reshape(1, 15, 3),
            ),
            dim=1,
        )[0].T
        coefficient_step = torch.cat(
            (
                clipped["features_dc"].reshape(1, 1, 3),
                clipped["features_rest"].reshape(1, 15, 3),
            ),
            dim=1,
        )[0].T
        directions = torch.randn(
            (20_000, 3), generator=generator, dtype=torch.float64
        )
        directions = directions / torch.linalg.vector_norm(
            directions, dim=1, keepdim=True
        )
        expanded = coefficients.unsqueeze(0).expand(directions.shape[0], -1, -1)
        expanded_after = (coefficients + coefficient_step).unsqueeze(0).expand_as(
            expanded
        )
        before = torch.clamp_min(eval_sh(3, expanded, directions) + 0.5, 0.0)
        after = torch.clamp_min(eval_sh(3, expanded_after, directions) + 0.5, 0.0)
        opacity = 0.5
        paper_bound = torch.sqrt(4.0 * before * epsilon / opacity)

        self.assertTrue(torch.all((after - before).abs() <= paper_bound + 1e-12))

    def test_zero_global_color_lower_bound_freezes_that_channel(self):
        params = _identity_raw()
        params["features_rest"][0, 0] = 2.0
        steps = _steps_like(params, 0.0)
        steps["features_dc"][0, 0] = 1.0
        steps["features_rest"][0, 0::3] = 1.0

        clipped = clip_hellinger_step(params, steps, 1e-3)

        self.assertEqual(clipped["features_dc"][0, 0].item(), 0.0)
        self.assertFalse(bool(clipped["features_rest"][0, 0::3].any()))

    def test_inactive_sh_degrees_are_held_fixed(self):
        params = _identity_raw()
        steps = _steps_like(params, 0.0)
        steps["features_rest"].fill_(1.0)

        clipped = clip_hellinger_step(
            params, steps, 1e-3, active_sh_degree=1
        )

        self.assertTrue(bool(clipped["features_rest"][0, :9].any()))
        self.assertFalse(bool(clipped["features_rest"][0, 9:].any()))

    def test_zero_step_is_bitwise_zero_and_inputs_are_not_modified(self):
        params = _identity_raw(rows=2)
        params["scaling"][0] = torch.tensor([1000.0, -1000.0, 0.0])
        params["opacity"][:] = torch.tensor([[1000.0], [-1000.0]])
        params["rotation"].zero_()
        params["features_dc"][0] = 1e300
        steps = _steps_like(params, 0.0)
        params_before = {name: value.clone() for name, value in params.items()}
        steps_before = {name: value.clone() for name, value in steps.items()}

        clipped = clip_hellinger_step(params, steps, 1e-6)

        for name in params:
            torch.testing.assert_close(params[name], params_before[name], rtol=0, atol=0)
            torch.testing.assert_close(steps[name], steps_before[name], rtol=0, atol=0)
            torch.testing.assert_close(clipped[name], torch.zeros_like(clipped[name]), rtol=0, atol=0)
            self.assertTrue(torch.isfinite(clipped[name]).all())

    def test_extreme_finite_inputs_produce_only_finite_steps(self):
        params = _identity_raw(rows=2)
        params["scaling"][:] = torch.tensor(
            [[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 0.0]],
            dtype=torch.float64,
        )
        params["opacity"][:] = torch.tensor([[1000.0], [-1000.0]])
        params["rotation"].zero_()
        params["features_dc"][:] = 1e300
        params["features_rest"][:] = -1e300
        steps = _steps_like(params, 1e300)

        clipped = clip_hellinger_step(params, steps, 1e-6)

        for value in clipped.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_smaller_epsilon_never_expands_a_coordinate_bound(self):
        params = _identity_raw(rows=2)
        steps = _steps_like(params, 10.0)
        small = clip_hellinger_step(params, steps, 1e-8)
        large = clip_hellinger_step(params, steps, 1e-4)

        for name in params:
            self.assertTrue(torch.all(small[name].abs() <= large[name].abs() + 1e-12))

    def test_component_shapes_are_validated(self):
        params = _identity_raw()
        steps = _steps_like(params, 1.0)
        steps["features_rest"] = torch.zeros((1, 44), dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "shape"):
            clip_hellinger_step(params, steps, 1e-6)

    def test_active_sh_degree_is_validated(self):
        params = _identity_raw()
        steps = _steps_like(params, 0.0)
        with self.assertRaisesRegex(ValueError, r"\[0, 3\]"):
            clip_hellinger_step(params, steps, 1e-6, active_sh_degree=4)
        with self.assertRaisesRegex(TypeError, "integer"):
            clip_hellinger_step(params, steps, 1e-6, active_sh_degree=True)


if __name__ == "__main__":
    unittest.main()
