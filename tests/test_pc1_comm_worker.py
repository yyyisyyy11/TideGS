"""Real CPU-thread/Future/deadline behavior, with no torch dependency."""

import importlib.util
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest


_PATH = Path(__file__).resolve().parents[1] / 'submodules/gsplat/gsplat/_pipeline_worker.py'
_SPEC = importlib.util.spec_from_file_location('pc1_worker_behavior', _PATH)
worker_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = worker_module
_SPEC.loader.exec_module(worker_module)


class DeviceEvent:
    def __init__(self, ready=True):
        self.ready = threading.Event()
        if ready:
            self.ready.set()

    def query(self):
        return self.ready.is_set()


def request(index, stage='C1', timeout=2, slot=None):
    return SimpleNamespace(
        identity=worker_module.CommunicationIdentity(0, 3, index,
            index % 2 if slot is None else slot, stage, index),
        deadline=worker_module.deadline_after(timeout))


class CommWorkerBehaviorTest(unittest.TestCase):
    def test_nonfinite_timeout_or_deadline_is_rejected(self):
        for value in (float('nan'), float('inf'), -1, 0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'finite|positive'):
                    worker_module.FIFOCommWorker(
                        execute=lambda stage, task: None, default_timeout_s=value)
        for value in (float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'finite'):
                worker_module.remaining(value, 'test')

    def worker(self, execute, **kwargs):
        worker = worker_module.FIFOCommWorker(execute=execute, **kwargs)
        self.addCleanup(self.cleanup, worker)
        return worker

    @staticmethod
    def cleanup(worker):
        if worker._aborted is None and not worker._closed:
            worker.close(worker_module.deadline_after(2))
        elif not worker._closed:
            worker.thread.join(timeout=0.2)

    def test_single_daemon_fifo_preserves_window_collective_order(self):
        order, threads, initialized = [], [], []

        def execute(stage, task):
            time.sleep(0.003 if task.identity.microbatch_index == 0 else 0.001)
            order.append((stage, task.identity.microbatch_index))
            threads.append(threading.get_native_id())
            return SimpleNamespace(done_event=DeviceEvent())

        worker = self.worker(execute, initialize=lambda: initialized.append(
            threading.get_native_id()), buffer_count=2)
        futures = [worker.enqueue_c1(request(0)), worker.enqueue_c1(request(1)),
                   worker.enqueue_c2(request(0, 'C2')), worker.enqueue_c2(request(1, 'C2'))]
        for future in futures:
            worker_module.wait_future(future, worker_module.deadline_after(2))
        worker.drain(worker_module.deadline_after(2))
        self.assertEqual(order, [('C1', 0), ('C1', 1), ('C2', 0), ('C2', 1)])
        self.assertEqual(len(set(threads)), 1)
        self.assertEqual(threads[0], initialized[0])
        self.assertNotEqual(threads[0], threading.get_native_id())
        self.assertTrue(worker.thread.daemon)
        self.assertEqual(worker._queue.maxsize, 4)
        self.assertEqual(len(initialized), 1)

    def test_future_publication_does_not_claim_gpu_completion(self):
        event = DeviceEvent(False)
        worker = self.worker(lambda stage, task: SimpleNamespace(done_event=event))
        result = worker_module.wait_future(worker.enqueue_c1(request(0)),
                                          worker_module.deadline_after(2))
        self.assertFalse(result.done_event.query())
        timer = threading.Timer(0.03, event.ready.set)
        timer.start()
        self.addCleanup(timer.join)
        start = time.monotonic()
        worker.drain(worker_module.deadline_after(2))
        self.assertGreater(time.monotonic() - start, 0.01)

    def test_queue_timeout_fails_every_pending_future_and_rejects_new_work(self):
        gate, entered = threading.Event(), threading.Event()

        def execute(stage, task):
            entered.set()
            gate.wait(timeout=1)
            return SimpleNamespace(done_event=DeviceEvent())

        worker = self.worker(execute, buffer_count=1)
        self.addCleanup(gate.set)
        first = worker.enqueue_c1(request(0))
        self.assertTrue(entered.wait(timeout=1))
        queued = [worker.enqueue_c1(request(i)) for i in (1, 2)]
        start = time.monotonic()
        with self.assertRaises(worker_module.PipelineTimeoutError):
            worker.enqueue_c1(request(3, timeout=0.02))
        self.assertLess(time.monotonic() - start, 0.3)
        for future in (first, *queued):
            self.assertTrue(future.done())
            self.assertIsInstance(future.exception(), TimeoutError)
        with self.assertRaisesRegex(RuntimeError, 'aborted'):
            worker.enqueue_c1(request(4))
        gate.set()
        worker.thread.join(timeout=1)
        self.assertFalse(worker.thread.is_alive())
        self.assertTrue(worker._quarantine)

    def test_future_timeout_and_gpu_drain_timeout_use_absolute_deadlines(self):
        from concurrent.futures import Future
        start = time.monotonic()
        with self.assertRaises(worker_module.PipelineTimeoutError):
            worker_module.wait_future(Future(), start + 0.015)
        self.assertLess(time.monotonic() - start, 0.15)
        worker = self.worker(lambda stage, task: SimpleNamespace(done_event=DeviceEvent(False)))
        future = worker.enqueue_c1(request(0))
        worker_module.wait_future(future, worker_module.deadline_after(1))
        with self.assertRaises(worker_module.PipelineTimeoutError):
            worker.drain(worker_module.deadline_after(0.015))
        self.assertIsNotNone(worker._aborted)

    def test_worker_exception_fails_queue_and_calls_failure_controller_once(self):
        gate, entered, failures = threading.Event(), threading.Event(), []

        def execute(stage, task):
            entered.set()
            gate.wait(timeout=1)
            raise ValueError('injected transport failure')

        worker = self.worker(execute, on_failure=failures.append)
        self.addCleanup(gate.set)
        first = worker.enqueue_c1(request(0))
        self.assertTrue(entered.wait(timeout=1))
        second = worker.enqueue_c2(request(0, 'C2'))
        gate.set()
        for future in (first, second):
            with self.assertRaisesRegex(ValueError, 'injected transport failure'):
                worker_module.wait_future(future, worker_module.deadline_after(1))
        worker.thread.join(timeout=1)
        worker.abort(RuntimeError('repeat abort'))
        self.assertEqual(len(failures), 1)
        self.assertEqual(len(worker._quarantine), 2)

    def test_healthy_shutdown_reuses_one_thread_across_batches(self):
        worker = self.worker(lambda stage, task: SimpleNamespace(done_event=DeviceEvent()))
        identity = worker.thread.ident
        for _ in range(3):
            future = worker.enqueue_c1(request(0))
            worker_module.wait_future(future, worker_module.deadline_after(1))
            worker.drain(worker_module.deadline_after(1))
            self.assertEqual(worker.thread.ident, identity)
            self.assertTrue(worker.thread.is_alive())
            self.assertEqual(worker._pending, {})
            self.assertEqual(worker._published, [])
        worker.close(worker_module.deadline_after(1))
        worker.close(worker_module.deadline_after(1))
        self.assertFalse(worker.thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            worker.enqueue_c1(request(0))

    def test_abort_never_waits_for_blocked_device_submission(self):
        entered, gate = threading.Event(), threading.Event()

        def execute(stage, task):
            entered.set()
            gate.wait(timeout=1)
            return SimpleNamespace(done_event=DeviceEvent())

        worker = self.worker(execute)
        self.addCleanup(gate.set)
        future = worker.enqueue_c1(request(0))
        self.assertTrue(entered.wait(timeout=1))
        start = time.monotonic()
        worker.abort(RuntimeError('abort now'))
        self.assertLess(time.monotonic() - start, 0.1)
        self.assertTrue(future.done())
        self.assertTrue(worker.thread.is_alive())
        gate.set()

    def test_result_returned_after_deadline_is_quarantined_before_publication(self):
        result = SimpleNamespace(done_event=DeviceEvent())

        def execute(stage, task):
            time.sleep(0.025)
            return result

        worker = self.worker(execute)
        future = worker.enqueue_c1(request(0, timeout=0.01))
        with self.assertRaises(worker_module.PipelineTimeoutError):
            worker_module.wait_future(future, worker_module.deadline_after(1))
        worker.thread.join(timeout=1)
        self.assertTrue(any(item is result for item in worker._quarantine))


if __name__ == '__main__':
    unittest.main()
