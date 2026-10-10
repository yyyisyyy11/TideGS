import unittest


try:
    import torch
except ImportError:  # pragma: no cover - exercised in the CUDA environment
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required for hook tests")
class PartialGradientHookTest(unittest.TestCase):
    def _tensors(self):
        return {
            "xyz": torch.ones((4, 3), requires_grad=True),
            "opacity": torch.ones((4, 1), requires_grad=True),
            "scaling": torch.ones((4, 3), requires_grad=True),
            "rotation": torch.ones((4, 4), requires_grad=True),
            "features": torch.ones((4, 16, 3), requires_grad=True),
        }

    def test_counts_all_zero_single_field_and_multiple_field_rows(self):
        from strategies.tide_engine.gsplat_backend import (
            PARTIAL_GRADIENT_CONTEXT,
            begin_partial_gradient_metrics,
            consume_partial_gradient_metrics,
        )

        tensors = self._tensors()
        tensors["opacity"].data[3].zero_()
        context = begin_partial_gradient_metrics(**tensors)
        # Row 0 is a valid all-zero-gradient record. Row 1 has only xyz.
        # Row 2 has multiple non-zero fields. Row 3 is a zero-opacity sentinel.
        loss = tensors["xyz"][1].sum()
        loss = loss + tensors["xyz"][2].sum() + tensors["features"][2].sum()
        loss.backward()
        total, all_zero = consume_partial_gradient_metrics(
            {PARTIAL_GRADIENT_CONTEXT: context}
        )
        self.assertEqual((total, all_zero), (3, 1))

    def test_nan_inf_records_are_excluded_from_denominator(self):
        from strategies.tide_engine.gsplat_backend import (
            PARTIAL_GRADIENT_CONTEXT,
            begin_partial_gradient_metrics,
            consume_partial_gradient_metrics,
        )

        tensors = self._tensors()
        tensors["xyz"].data[2, 0] = float("nan")
        tensors["opacity"].data[3, 0] = float("inf")
        context = begin_partial_gradient_metrics(**tensors)
        tensors["xyz"][1].sum().backward()
        total, all_zero = consume_partial_gradient_metrics(
            {PARTIAL_GRADIENT_CONTEXT: context}
        )
        self.assertEqual(total, 2)
        self.assertEqual(all_zero, 1)

    def test_counters_cannot_be_consumed_twice(self):
        from strategies.tide_engine.gsplat_backend import (
            PARTIAL_GRADIENT_CONTEXT,
            begin_partial_gradient_metrics,
            consume_partial_gradient_metrics,
        )

        tensors = self._tensors()
        context = begin_partial_gradient_metrics(**tensors)
        tensors["xyz"][1].sum().backward()
        meta = {PARTIAL_GRADIENT_CONTEXT: context}
        consume_partial_gradient_metrics(meta)
        with self.assertRaises(RuntimeError):
            consume_partial_gradient_metrics(meta)


if __name__ == "__main__":
    unittest.main()
