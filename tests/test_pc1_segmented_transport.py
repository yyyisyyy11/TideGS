"""Execute packet packing, leaf cuts, transpose transport and slot behavior."""

from concurrent.futures import Future
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.pc1_tensor_stub import Backend, Event, Tensor


class SegmentedTransportTest(unittest.TestCase):
    def test_tile_row_mask_broadcasts_over_both_radius_axes(self):
        backend = Backend()
        module = backend.load()
        for radii, keep, expected in (
            ([[2, 4], [6, 8], [10, 12]], [True, False, True], [[2, 4], [0, 0], [10, 12]]),
            ([[2, 4], [6, 8]], [True, False], [[2, 4], [0, 0]]),
            ([2, 6, 10], [True, False, True], [2, 0, 10]),
        ):
            with self.subTest(radii=radii):
                actual = module._mask_radii(Tensor(radii), Tensor(keep))
                np.testing.assert_array_equal(actual.values, expected)

    def setup_pipeline(self, *, world_size=1, rows=3, trace=False):
        self.backend = Backend(world_size)
        self.module = self.backend.load()
        self.worker = self.backend.worker(self.module, default_timeout_s=2)
        self.addCleanup(self.cleanup_worker)
        self.parameters = {name: Tensor(0.0, requires_grad=True)
                           for name in self.module.FLOAT_ATTRIBUTES}
        self.originals = []

        def projection(**kwargs):
            outputs = {}
            for index, (name, width) in enumerate(zip(self.module.FLOAT_ATTRIBUTES, (2, 1, 3, 1, 3))):
                shape = (rows,) if width == 1 else (rows, width)
                coefficient = np.arange(np.prod(shape), dtype=float).reshape(shape) + index + 1
                outputs[name] = Tensor(coefficient.copy(), True,
                                       (self.parameters[name], coefficient))
            self.originals.append(outputs)
            counts = [Tensor(rows)] if world_size == 1 else [Tensor(1), Tensor(2)]
            return dict(outputs, means=Tensor(np.zeros((3, 3))),
                        radii=Tensor(np.ones((rows, 2), dtype=np.int32)),
                        camera_ids=Tensor(np.zeros(rows, dtype=np.int64)),
                        gaussian_ids=Tensor(np.arange(rows, dtype=np.int64)),
                        cnts=counts, C_world=[1] * world_size, N_world=[3] * world_size,
                        meta={}, projection_start=Event(), projection_end=Event())

        self.projection_patch = patch.object(self.module, '_projection_prepare', projection)
        self.render_patch = patch.object(self.module, '_render_prepared',
            lambda state: (Tensor(np.zeros((1, 2, 2, 3))), Tensor(0.0), state['meta']))
        self.projection_patch.start()
        self.render_patch.start()
        self.addCleanup(self.projection_patch.stop)
        self.addCleanup(self.render_patch.stop)
        self.pipeline = self.module.DistributedRasterizationPipeline(
            comm_worker=self.worker, buffer_count=2, batch_index=11, capture_timeline=trace)
        return self.pipeline

    def cleanup_worker(self):
        if self.worker._aborted is None and not self.worker._closed:
            self.worker.close(self.worker.resolve_deadline(None))
        else:
            self.worker.thread.join(timeout=0.3)

    def submit(self, index=0):
        return self.pipeline.submit_projection(microbatch_index=index, means=Tensor(0.0))

    @staticmethod
    def loss(handle, *, requires_grad=True):
        values = [None if name == 'depths' else np.arange(leaf.values.size).reshape(leaf.shape) + 0.5
                  for name, leaf in handle.image_leaves.items()]
        return SimpleNamespace(requires_grad=requires_grad, grad_fn=None,
                               gradient_values=values)

    def test_detached_c1_preserves_original_graph_and_only_g_updates_parameters(self):
        self.setup_pipeline(world_size=2)
        handle = self.submit().start_forward_exchange()
        original = dict(handle.projection_outputs)
        handle.wait_and_render()
        for name in self.module.FLOAT_ATTRIBUTES:
            self.assertIs(handle.projection_outputs[name], original[name])
            self.assertIsNot(handle.image_leaves[name], original[name])
            self.assertIsNone(handle.image_leaves[name].source)
            self.assertTrue(handle.image_leaves[name].requires_grad)
        loss = self.loss(handle)
        handle.intermediate_backward(loss)
        self.assertTrue(all(param.grad is None for param in self.parameters.values()))
        self.assertTrue(all(leaf.grad is None for leaf in handle.image_leaves.values()))
        event = handle.final_backward()
        self.assertIs(event, handle.backward_done)
        self.assertEqual(len(self.backend.autograd.backward_calls), 1)
        grad_call = self.backend.autograd.grad_calls[0]
        self.assertEqual(grad_call[2], dict(allow_unused=True, create_graph=False, retain_graph=False))
        for index, name in enumerate(self.module.FLOAT_ATTRIBUTES):
            coefficient = original[name].source[1]
            gradient = loss.gradient_values[index]
            expected = 0.0 if gradient is None else float(np.sum(coefficient * gradient[[1, 2, 0]]))
            self.assertAlmostEqual(self.parameters[name].grad, expected)
        calls = self.backend.calls
        self.assertEqual([stage for stage, _, _ in calls], ['C1'] * 4 + ['C2'])
        c2 = calls[-1][2]
        self.assertEqual(c2['input_split_sizes'], [2, 1])
        self.assertEqual(c2['output_split_sizes'], [1, 2])
        self.assertTrue(all(stream is self.worker.comm_stream for stream in self.backend.waited_work))
        self.assertEqual(len(self.backend.device_initializations), 1)
        snapshot = handle.snapshot()
        self.assertTrue(all(not isinstance(value, Tensor) for value in snapshot.values()))
        self.assertTrue(any(record['thread_role'] == 'comm' for record in snapshot['cpu_trace']))
        self.pipeline.drain()
        self.pipeline.close()
        self.assertTrue(self.worker.thread.is_alive())

    def test_completed_future_still_requires_compute_event_dependency(self):
        self.setup_pipeline()
        handle = self.submit().start_forward_exchange()
        result = self.module.wait_future(handle.c1_future, self.worker.resolve_deadline(None))
        result.done_event.ready = False
        self.assertTrue(handle.c1_future.done())
        handle.wait_and_render()
        self.assertIn(result.done_event, self.pipeline.compute_stream.waits)
        self.assertFalse(result.done_event.query())
        result.done_event.ready = True
        handle.intermediate_backward(self.loss(handle))
        handle.final_backward()
        self.pipeline.drain()
        self.pipeline.close()

    def test_worker_does_not_attach_results_or_mutate_main_thread_state(self):
        self.setup_pipeline()
        handle = self.submit().start_forward_exchange()
        original_state = dict(handle.state)
        self.module.wait_future(handle.c1_future, self.worker.resolve_deadline(None))
        self.assertEqual(handle.phase, 'C1_QUEUED')
        self.assertEqual(handle.image_leaves, {})
        self.assertIsNone(handle.c1_result)
        self.assertEqual(handle.state.keys(), original_state.keys())
        for key, value in original_state.items():
            self.assertIs(handle.state[key], value)
        self.assertEqual(self.pipeline._active, {0: handle})
        handle.wait_and_render()
        handle.intermediate_backward(self.loss(handle))
        handle.final_backward()
        self.pipeline.drain()
        self.pipeline.close()

    def test_slot_reuse_requires_g_and_keeps_buffer_lease_until_event_completion(self):
        self.setup_pipeline()
        handle = self.submit().start_forward_exchange()
        with self.assertRaisesRegex(RuntimeError, 'still in use'):
            self.submit(2)
        handle.wait_and_render()
        handle.intermediate_backward(self.loss(handle))
        with self.assertRaisesRegex(RuntimeError, 'still in use'):
            self.submit(2)
        final = handle.final_backward()
        final.ready = False
        newer = self.submit(2)
        self.assertIn(final, self.pipeline.compute_stream.waits)
        self.assertEqual(len(self.pipeline._retired), 1)
        self.assertEqual(handle.projection_outputs, {})
        self.assertIsNone(handle.state)
        final.ready = True
        newer.start_forward_exchange().wait_and_render()
        newer.intermediate_backward(self.loss(newer))
        newer.final_backward()
        self.pipeline.drain()
        self.pipeline.close()

    def test_duplicate_stages_and_cross_thread_graph_access_are_rejected(self):
        self.setup_pipeline()
        handle = self.submit().start_forward_exchange()
        with self.assertRaisesRegex(RuntimeError, 'requires PROJECTED'):
            handle.start_forward_exchange()
        with self.assertRaisesRegex(RuntimeError, 'requires C2_QUEUED'):
            handle.final_backward()
        failures = []
        thread = threading.Thread(target=lambda: self.capture(failures, handle.wait_and_render))
        thread.start()
        thread.join(timeout=1)
        self.assertRegex(str(failures[0]), 'main thread')
        handle.wait_and_render()
        handle.intermediate_backward(self.loss(handle))
        with self.assertRaisesRegex(RuntimeError, 'requires RENDERED'):
            handle.intermediate_backward(self.loss(handle))
        handle.final_backward()
        self.pipeline.drain()
        self.pipeline.close()
        with self.assertRaisesRegex(RuntimeError, 'shutdown'):
            self.submit(2)

    @staticmethod
    def capture(failures, function):
        try:
            function()
        except Exception as error:
            failures.append(error)

    def test_zero_rows_and_loss_without_differentiable_path_still_transport_c2(self):
        for rows in (0, 3):
            with self.subTest(rows=rows):
                self.setup_pipeline(rows=rows)
                handle = self.submit().start_forward_exchange()
                handle.wait_and_render()
                handle.intermediate_backward(self.loss(handle, requires_grad=False))
                handle.final_backward()
                self.assertEqual(len(self.backend.autograd.grad_calls), 0)
                self.assertEqual(handle.c2_result.layout.send_splits, (rows,))
                for parameter in self.parameters.values():
                    self.assertEqual(parameter.grad, 0.0)
                self.pipeline.drain()
                self.pipeline.close()
                self.cleanup_worker()
                self.projection_patch.stop()
                self.render_patch.stop()

    def test_trace_switch_preserves_gradient_calls_and_transport_sequence(self):
        results = []
        for enabled in (False, True):
            self.setup_pipeline(world_size=2, trace=enabled)
            handle = self.submit().start_forward_exchange()
            handle.wait_and_render()
            handle.intermediate_backward(self.loss(handle))
            handle.final_backward()
            results.append(([stage for stage, _, _ in self.backend.calls],
                            [parameter.grad for parameter in self.parameters.values()]))
            self.pipeline.drain()
            self.pipeline.close()
            self.cleanup_worker()
            self.projection_patch.stop()
            self.render_patch.stop()
        self.assertEqual(results[0], results[1])

    def test_partial_collective_failure_quarantines_buffers_before_result_publication(self):
        self.setup_pipeline(world_size=2)
        original = self.module._raw_exchange
        observed = []

        def failing_exchange(data, send_splits, recv_splits, retained):
            output = original(data, send_splits, recv_splits, retained)
            observed.extend((data, output))
            raise ValueError('failure after first payload submission')

        with patch.object(self.module, '_raw_exchange', failing_exchange):
            handle = self.submit().start_forward_exchange()
            with self.assertRaisesRegex(ValueError, 'first payload'):
                handle.wait_and_render()
        self.worker.thread.join(timeout=1)
        self.assertTrue(self.pipeline._aborted)
        self.assertIsNone(handle.c1_result)
        self.assertTrue(observed)
        self.assertTrue(any(item is self.worker._transport_inflight
                            for item in self.worker._quarantine))
        for tensor in observed:
            self.assertTrue(any(item is tensor for item in self.worker._transport_inflight))
        self.assertEqual(self.pipeline._active, {})
        self.assertIn(handle, self.pipeline._quarantine)


if __name__ == '__main__':
    unittest.main()
