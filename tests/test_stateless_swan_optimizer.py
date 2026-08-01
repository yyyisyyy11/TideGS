import math
import unittest
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from strategies.tide_engine.gpu_resident_optimizer import GPUStatelessSWAN
    from strategies.tide_engine.runtime import validate_tide_runtime_args
else:
    GPUStatelessSWAN = None
    validate_tide_runtime_args = None


@unittest.skipUnless(torch is not None, 'PyTorch is required')
class StatelessSWANOptimizerTest(unittest.TestCase):
    COMPONENT_WIDTHS = {
        'xyz': 3,
        'opacity': 1,
        'scaling': 3,
        'rotation': 4,
        'features_dc': 3,
        'features_rest': 45,
    }
    COMPONENT_ATTRS = {
        'xyz': '_xyz',
        'opacity': '_opacity',
        'scaling': '_scaling',
        'rotation': '_rotation',
        'features_dc': '_features_dc',
        'features_rest': '_features_rest',
    }

    @staticmethod
    def swan_reference(mean_grads, eps, steps=10, beta=0.8):
        touched_rows, width = mean_grads.shape
        matrix = mean_grads.transpose(0, 1).to(dtype=torch.float32)
        matrix = matrix / matrix.square().mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
        if touched_rows < width:
            return matrix.transpose(0, 1), True
        matrix = matrix / matrix.norm().clamp_min(eps)
        identity = torch.eye(width, dtype=matrix.dtype, device=matrix.device)
        y = matrix @ matrix.transpose(0, 1)
        z = identity
        for _ in range(steps):
            transform = beta * (3.0 * identity - z @ y)
            y, z = y @ transform, transform @ z
        matrix = z @ matrix
        matrix = matrix * (
            math.sqrt(float(width * touched_rows)) / matrix.norm().clamp_min(eps)
        )
        return matrix.transpose(0, 1), False

    def make_gaussians(self, rows, columns_lr, eps):
        context = SimpleNamespace(columns_lr=columns_lr, param_groups=[{'eps': eps}])
        gaussians = SimpleNamespace(optimizer=context)
        for component_index, (name, width) in enumerate(self.COMPONENT_WIDTHS.items()):
            setattr(
                gaussians,
                self.COMPONENT_ATTRS[name],
                torch.full((rows, width), 10.0 + component_index),
            )
        return gaussians

    def test_hybrid_swan_matches_reference_and_keeps_untouched_rows(self):
        torch.manual_seed(7)
        touched_rows = 48
        rows = touched_rows + 2
        local_ids = torch.arange(touched_rows, dtype=torch.long)
        columns_lr = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        eps = 1e-6
        gaussians = self.make_gaussians(rows, columns_lr, eps)
        before = {
            name: getattr(gaussians, attr).clone()
            for name, attr in self.COMPONENT_ATTRS.items()
        }
        grads = {
            name: torch.randn(touched_rows, width)
            for name, width in self.COMPONENT_WIDTHS.items()
        }
        grads['features_rest'][0].zero_()

        optimizer = GPUStatelessSWAN(batch_size=4, device='cpu')
        stats = optimizer.step(1, gaussians, local_ids, grads)

        for component_index, (name, width) in enumerate(self.COMPONENT_WIDTHS.items()):
            expected = before[name].clone()
            mean_grad = grads[name] / 4.0
            if name in GPUStatelessSWAN.SWAN_COMPONENTS:
                update, used_fallback = self.swan_reference(mean_grad, eps)
                self.assertFalse(used_fallback, name)
                self.assertGreater(float(update.abs().max()), 1.0, name)
                update = update.clamp(min=-1.0, max=1.0)
            else:
                update = mean_grad / (mean_grad.abs() + eps)
            expected.index_add_(0, local_ids, update, alpha=-columns_lr[component_index])
            actual = getattr(gaussians, self.COMPONENT_ATTRS[name])
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(actual[-2:], before[name][-2:])
            applied_direction = (
                before[name][local_ids] - actual[local_ids]
            ) / columns_lr[component_index]
            self.assertLessEqual(float(applied_direction.abs().max()), 1.0)

        self.assertEqual(
            stats,
            {
                'touched_rows': touched_rows,
                'state_bytes': 0,
                'swan_whitened_components': 2,
                'swan_gradnorm_fallback_components': 0,
                'normalized_sgd_components': 4,
            },
        )
        self.assertFalse(hasattr(optimizer, 'state'))
        self.assertFalse(optimizer.log_update_stats)
        optimizer_stats = optimizer.get_stats()
        self.assertEqual(optimizer_stats['persistent_state_bytes'], 0)
        self.assertEqual(optimizer_stats['optimizer_rows_touched_total'], touched_rows)
        self.assertEqual(optimizer_stats['swan_whitened_components_total'], 2)
        self.assertEqual(optimizer_stats['swan_gradnorm_fallback_components_total'], 0)
        self.assertEqual(optimizer_stats['normalized_sgd_components_total'], 4)

        # A completely zero Gaussian gradient must remain a zero update.
        torch.testing.assert_close(
            gaussians._features_rest[0],
            before['features_rest'][0],
        )

    def test_whitened_component_has_target_norm_and_improves_conditioning(self):
        torch.manual_seed(11)
        base = torch.randn(128, 1)
        grads = torch.cat((base, base + 0.05 * torch.randn(128, 1)), dim=1)
        optimizer = GPUStatelessSWAN(batch_size=16, device='cpu')
        update, used_fallback = optimizer._swan_direction(grads / 16.0, eps=1e-6)

        self.assertFalse(used_fallback)
        normalized = (grads / 16.0).transpose(0, 1)
        normalized = normalized / normalized.square().mean(dim=1, keepdim=True).sqrt()
        self.assertLess(
            float(torch.linalg.cond(update.transpose(0, 1))),
            float(torch.linalg.cond(normalized)),
        )
        self.assertAlmostEqual(float(update.norm()), math.sqrt(128 * 2), places=4)

    def test_uses_gradnorm_when_touched_rows_are_narrower_than_component(self):
        grads = torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
        optimizer = GPUStatelessSWAN(batch_size=2, device='cpu')
        update, used_fallback = optimizer._swan_direction(grads / 2.0, eps=1e-6)

        expected, expected_fallback = self.swan_reference(grads / 2.0, 1e-6)
        self.assertTrue(used_fallback)
        self.assertTrue(expected_fallback)
        torch.testing.assert_close(update, expected)

    def test_step_counts_gradnorm_fallback_components(self):
        rows = 3
        gaussians = self.make_gaussians(rows, [0.1] * 6, eps=1e-6)
        grads = {
            name: torch.ones((rows, width))
            for name, width in self.COMPONENT_WIDTHS.items()
        }
        stats = GPUStatelessSWAN(batch_size=1, device='cpu').step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=torch.arange(rows),
            sparse_grad_components=grads,
        )

        self.assertEqual(stats['swan_whitened_components'], 2)
        self.assertEqual(stats['swan_gradnorm_fallback_components'], 0)
        self.assertEqual(stats['normalized_sgd_components'], 4)

    def test_rejects_nonfinite_gradients(self):
        optimizer = GPUStatelessSWAN(batch_size=1, device='cpu')
        with self.assertRaisesRegex(FloatingPointError, 'Non-finite raw gradient'):
            optimizer._swan_direction(torch.tensor([[float('nan')]]), eps=1e-6)

    def test_all_zero_component_gradient_stays_zero(self):
        optimizer = GPUStatelessSWAN(batch_size=1, device='cpu')
        update, used_fallback = optimizer._swan_direction(
            torch.zeros((48, 45)),
            eps=1e-6,
        )
        self.assertFalse(used_fallback)
        torch.testing.assert_close(update, torch.zeros_like(update))

    def test_empty_input_keeps_state_empty(self):
        optimizer = GPUStatelessSWAN(batch_size=4, device='cpu')
        stats = optimizer.step(
            iteration=1,
            gaussians=SimpleNamespace(),
            sparse_grad_local_ids=torch.empty((0,), dtype=torch.long),
            sparse_grad_components={},
        )
        self.assertEqual(stats, {'touched_rows': 0, 'state_bytes': 0})
        self.assertEqual(optimizer.get_stats()['persistent_state_bytes'], 0)


@unittest.skipUnless(torch is not None, 'PyTorch is required')
class StatelessSWANConfigTest(unittest.TestCase):
    @staticmethod
    def make_args(state_mode):
        return SimpleNamespace(
            pure_ssd_offload=True,
            use_ssd_offload=True,
            clm_offload=True,
            naive_offload=False,
            no_offload=False,
            ssd_execution_mode='paper',
            paper_block_reader_backend='tiered_cache',
            paper_optimizer_backend='gpu_resident',
            paper_optimizer_state_mode=state_mode,
            paper_optimizer_deferred_mode='off',
            sparse_adam=False,
            paper_free_unified_params=True,
            disable_auto_densification=True,
        )

    def test_tide_runtime_accepts_stateless_mode(self):
        validate_tide_runtime_args(self.make_args('none'))

    def test_tide_runtime_rejects_resident_state(self):
        with self.assertRaisesRegex(RuntimeError, 'paper_optimizer_state_mode none'):
            validate_tide_runtime_args(self.make_args('resident_blocks'))


if __name__ == '__main__':
    unittest.main()
