"""Opt-in real SSIM stream tests; execute with an external process timeout.

    TIDE_RUN_NATIVE_CUDA=1 timeout 120s python -m unittest tests.test_micro_pipeline_native_cuda -v

These tests compare the actual fused loss and its image gradient on a blocked
default stream versus a non-default compute stream. CPU discovery skips them.
"""

import os
import unittest

try:
    import torch
except ImportError:
    torch = None


RUN_CUDA = (torch is not None and torch.cuda.is_available()
            and os.environ.get("TIDE_RUN_NATIVE_CUDA") == "1")


@unittest.skipUnless(RUN_CUDA, "set TIDE_RUN_NATIVE_CUDA=1 on a CUDA host")
class MicroPipelineNativeCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utils.micro_pipeline_native_capabilities import check_micro_pipeline_native_capabilities
        check_micro_pipeline_native_capabilities(require_tile_mask=True)
        import clm_kernels
        cls.clm = clm_kernels
        cls.device = torch.device("cuda", torch.cuda.current_device())

    def _evaluate(self, img1, img2, padding):
        image = img1.detach().clone().requires_grad_()
        loss = self.clm.fused_ssim(image, img2, padding=padding, train=True)
        gradient, = torch.autograd.grad(loss, (image,))
        return loss.detach().clone(), gradient.detach().clone()

    def test_nondefault_stream_preserves_actual_loss_and_backward(self):
        torch.manual_seed(139)
        # Odd dimensions also exercise edge padding and partial tiles.
        first = torch.rand((2, 3, 47, 65), device=self.device)
        second = torch.rand_like(first)
        torch.cuda.synchronize(self.device)
        for padding in ("same", "valid"):
            with self.subTest(padding=padding):
                expected = self._evaluate(first, second, padding)
                torch.cuda.synchronize(self.device)
                compute = torch.cuda.Stream(device=self.device)
                default = torch.cuda.default_stream(self.device)
                default_blocked = torch.cuda.Event()
                ready = torch.cuda.Event()
                # Preallocate inputs and warm this stream's allocator/autograd
                # path before blocking default. A first cudaMalloc may otherwise
                # synchronize the device and hide the ordering being tested.
                with torch.cuda.stream(compute):
                    shifted_first = first.mul(0.7).add(0.1)
                    shifted_second = second.mul(0.8).add(0.05)
                    warm = self._evaluate(shifted_first, shifted_second, padding)
                    ready.record(compute)
                torch.cuda.synchronize(self.device)
                del warm
                # A launch accidentally routed to the default stream runs only
                # after the consumer snapshot below, exposing missing ordering.
                with torch.cuda.stream(default):
                    torch.cuda._sleep(300_000_000)
                    default_blocked.record(default)
                with torch.cuda.stream(compute):
                    # Change inputs so stale outputs from the reference cannot
                    # make a default-stream launch appear correct.
                    actual = self._evaluate(shifted_first, shifted_second, padding)
                    ready.record(compute)
                ready.synchronize()
                self.assertFalse(
                    default_blocked.query(),
                    "default-stream delay ended before the consumer; increase test delay",
                )
                # The expectation itself runs after draining the deliberately
                # blocked default stream; actual tensors are already snapshots.
                torch.cuda.synchronize(self.device)
                expected_shifted = self._evaluate(shifted_first, shifted_second, padding)
                torch.cuda.synchronize(self.device)
                for observed, reference in zip(actual, expected_shifted):
                    torch.testing.assert_close(observed, reference, rtol=1e-4, atol=1e-5)
                # Confirm this is a sensitive loss input, not an invariant case.
                self.assertGreater(float((expected[0] - expected_shifted[0]).abs()), 1e-6)

    def test_native_forward_and_backward_guard_input_device(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("device guard requires two CUDA devices")
        from clm_kernels import _C
        original = torch.cuda.current_device()
        input_device = self.device.index
        other = (input_device + 1) % torch.cuda.device_count()
        image = torch.rand((1, 3, 17, 19), device=self.device)
        target = torch.rand_like(image)
        grad_map = torch.ones_like(image)
        torch.cuda.synchronize(self.device)
        try:
            torch.cuda.set_device(other)
            outputs = _C.fusedssim(0.01 ** 2, 0.03 ** 2, image, target, True)
            gradient = _C.fusedssim_backward(
                0.01 ** 2, 0.03 ** 2, image, target, grad_map, *outputs[1:],
            )
            self.assertEqual(torch.cuda.current_device(), other)
            self.assertEqual(gradient.device, image.device)
            torch.cuda.synchronize(self.device)
            self.assertTrue(torch.isfinite(gradient).all().item())
        finally:
            torch.cuda.set_device(original)


if __name__ == "__main__":
    unittest.main()
