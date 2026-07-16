import unittest
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from strategies.tide_engine.gpu_resident_optimizer import (
        GPUStatelessNormalizedSGD,
    )
    from strategies.tide_engine.runtime import validate_tide_runtime_args
else:
    GPUStatelessNormalizedSGD = None
    validate_tide_runtime_args = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class StatelessNormalizedOptimizerTest(unittest.TestCase):
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

    def test_updates_only_touched_rows_with_component_learning_rates(self):
        columns_lr = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        eps = 0.5
        context = SimpleNamespace(
            columns_lr=columns_lr,
            param_groups=[{'eps': eps}],
        )
        gaussians = SimpleNamespace(optimizer=context)
        before = {}
        grads = {}

        for component_index, (name, width) in enumerate(self.COMPONENT_WIDTHS.items()):
            attr_name = self.COMPONENT_ATTRS[name]
            params = torch.full((4, width), 10.0 + component_index)
            setattr(gaussians, attr_name, params)
            before[name] = params.clone()

            component_grads = torch.arange(
                1,
                2 * width + 1,
                dtype=torch.float32,
            ).reshape(2, width)
            component_grads[0, 0] = 0.0
            component_grads[1] *= -1.0
            grads[name] = component_grads

        local_ids = torch.tensor([1, 3], dtype=torch.long)
        optimizer = GPUStatelessNormalizedSGD(batch_size=2, device='cpu')
        stats = optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=local_ids,
            sparse_grad_components=grads,
        )

        for component_index, (name, width) in enumerate(self.COMPONENT_WIDTHS.items()):
            expected = before[name].clone()
            mean_grad = grads[name] / 2.0
            normalized = mean_grad / (mean_grad.abs() + eps)
            expected.index_add_(
                0,
                local_ids,
                normalized,
                alpha=-columns_lr[component_index],
            )
            actual = getattr(gaussians, self.COMPONENT_ATTRS[name])
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual[0], before[name][0])
            torch.testing.assert_close(actual[2], before[name][2])

        self.assertEqual(stats, {'touched_rows': 2, 'state_bytes': 0})
        self.assertFalse(hasattr(optimizer, 'state'))
        self.assertEqual(optimizer.get_stats()['persistent_state_bytes'], 0)
        self.assertEqual(optimizer.get_stats()['optimizer_rows_touched_total'], 2)

        # The first touched opacity gradient is exactly zero and must remain unchanged.
        actual_delta = before['opacity'][1, 0] - gaussians._opacity[1, 0]
        self.assertAlmostEqual(float(actual_delta), 0.0, places=6)
        nonzero_delta = before['opacity'][3, 0] - gaussians._opacity[3, 0]
        expected_nonzero = columns_lr[1] * ((-2.0 / 2.0) / (abs(-2.0 / 2.0) + eps))
        self.assertAlmostEqual(float(nonzero_delta), expected_nonzero, places=6)

    def test_matches_bias_corrected_adam_on_first_cold_start_step(self):
        learning_rate = 0.25
        eps = 1e-3
        beta1, beta2 = 0.9, 0.999
        context = SimpleNamespace(
            columns_lr=[learning_rate] * 6,
            param_groups=[{'eps': eps}],
        )
        gaussians = SimpleNamespace(optimizer=context)
        grads = {}

        for name, width in self.COMPONENT_WIDTHS.items():
            setattr(gaussians, self.COMPONENT_ATTRS[name], torch.zeros((2, width)))
            component_grads = torch.linspace(-2.0, 2.0, steps=width).reshape(1, width)
            grads[name] = component_grads

        optimizer = GPUStatelessNormalizedSGD(batch_size=4, device='cpu')
        optimizer.step(
            iteration=1,
            gaussians=gaussians,
            sparse_grad_local_ids=torch.tensor([1]),
            sparse_grad_components=grads,
        )

        for name in self.COMPONENT_WIDTHS:
            mean_grad = grads[name] / 4.0
            exp_avg = (1.0 - beta1) * mean_grad
            exp_avg_sq = (1.0 - beta2) * mean_grad.square()
            adam_denom = exp_avg_sq.sqrt() / ((1.0 - beta2) ** 0.5) + eps
            cold_start_adam_update = (
                exp_avg / adam_denom
            ) * (learning_rate / (1.0 - beta1))
            actual = getattr(gaussians, self.COMPONENT_ATTRS[name])
            torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]))
            torch.testing.assert_close(actual[1], -cold_start_adam_update[0])

    def test_empty_input_keeps_state_empty(self):
        optimizer = GPUStatelessNormalizedSGD(batch_size=4, device='cpu')
        stats = optimizer.step(
            iteration=1,
            gaussians=SimpleNamespace(),
            sparse_grad_local_ids=torch.empty((0,), dtype=torch.long),
            sparse_grad_components={},
        )
        self.assertEqual(stats, {'touched_rows': 0, 'state_bytes': 0})
        self.assertEqual(optimizer.get_stats()['persistent_state_bytes'], 0)


@unittest.skipUnless(torch is not None, "PyTorch is required")
class StatelessOptimizerConfigTest(unittest.TestCase):
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
