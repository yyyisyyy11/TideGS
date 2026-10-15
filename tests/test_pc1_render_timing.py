"""Execute the real Future/attach/render path with independent stream clocks."""

from concurrent.futures import Future
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.pc1_tensor_stub import Backend, Event, Tensor


class TimedEvent(Event):
    def __init__(self, ms=0):
        super().__init__()
        self.ms = ms

    def record(self, stream):
        self.ms = stream.ms

    def elapsed_time(self, end):
        return end.ms - self.ms


class TimedStream:
    def __init__(self, ms):
        self.ms = ms
        self.waits = []

    def wait_event(self, event):
        self.waits.append(event)
        self.ms = max(self.ms, event.ms)


class PC1RenderTimingTest(unittest.TestCase):
    def test_render_excludes_compute_work_after_exchange(self):
        backend = Backend()
        module = backend.load()
        compute = TimedStream(7.0)  # P(i+1) ends at 7; C1(i) ended at 2.
        pipeline = SimpleNamespace(compute_stream=compute, rank=0, batch_index=1,
            _ensure_not_aborted=lambda: None, _ensure_main_thread=lambda: None,
            _deadline=lambda value: module.time.monotonic() + 1)
        identity = module.CommunicationIdentity(0, 1, 0, 0, 'C1', 0)
        result = SimpleNamespace(identity=identity, done_event=TimedEvent(2.0),
            floats=tuple(Tensor(np.zeros((1, width))) for width in (2, 1, 3, 1, 3)),
            radii=Tensor([[1, 1]]), camera_ids=Tensor([0]), gaussian_ids=Tensor([0]),
            events=(('c1_wait_start', TimedEvent()), ('c1_wait_end', TimedEvent())),
            worker_begin_ns=1, worker_end_ns=2, native_thread_id=7, thread_id=8)
        future = Future()
        future.set_result(result)
        handle = module.DistributedRasterizationHandle(pipeline, 0, 0,
            state={'C_world': [1], 'meta': {}}, phase='C1_QUEUED', c1_future=future)

        def render(state):
            compute.ms += 3.0
            return 'image', 'alpha', {}

        with patch.object(module, '_render_prepared', render), patch.object(module, '_cuda_event', TimedEvent):
            self.assertEqual(handle.wait_and_render()[0], 'image')
        self.assertEqual(compute.waits, [result.done_event])
        self.assertEqual(handle.render_start.ms, 7.0)
        self.assertEqual(handle.render_done.ms, 10.0)
        self.assertEqual(handle.timing()['render_ms'], 3.0)
        self.assertIs(handle.snapshot()['render_start'], handle.render_start)


if __name__ == '__main__':
    unittest.main()
