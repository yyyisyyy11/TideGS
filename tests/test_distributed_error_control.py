"""CPU fault injection: workers exit even while their main thread is blocked."""

import ast
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
import weakref

from utils.distributed_error_control import (
    PC1ErrorController, PC1FatalError, PC1_FATAL_EXIT_CODE,
)


class MemoryStore:
    def __init__(self, values=None):
        self.values = {} if values is None else values

    def set(self, key, value):
        self.values[key] = value

    def check(self, keys):
        return all(key in self.values for key in keys)

    def get(self, key):
        return self.values[key]


class SharedArrayStore:
    """Store API over CPU shared memory; no test-server socket is required."""

    slot_size = 4096

    def __init__(self, values):
        self.values = values

    @staticmethod
    def rank(key):
        return int(key.rsplit("-", 1)[-1])

    def set(self, key, value):
        encoded = value.encode("utf-8") if isinstance(value, str) else value
        if len(encoded) >= self.slot_size:
            raise ValueError("test Store value exceeds slot size")
        offset = self.rank(key) * self.slot_size
        with self.values.get_lock():
            self.values[offset:offset + len(encoded)] = encoded
            self.values[offset + len(encoded)] = 0

    def check(self, keys):
        with self.values.get_lock():
            return all(self.values[self.rank(key) * self.slot_size] != 0 for key in keys)

    def get(self, key):
        offset = self.rank(key) * self.slot_size
        with self.values.get_lock():
            return bytes(self.values[offset:offset + self.slot_size]).split(b"\0", 1)[0]


def blocked_worker(rank, values, ready, reasons, local_failure):
    store = SharedArrayStore(values)
    reports = SharedArrayStore(reasons)

    def exit_callback(reason):
        reports.set(f"rank-{rank}", reason)
        os._exit(PC1_FATAL_EXIT_CODE)

    controller = PC1ErrorController(
        store=store, rank=rank, world_size=2 if local_failure else 1,
        stage_timeout=2.0 if local_failure else 0.3,
        poll_interval=0.01, exit_callback=exit_callback,
    )
    ready.wait(timeout=5)
    with controller.operation("before matching collective"):
        if local_failure and rank == 0:
            raise RuntimeError("injected failure before collective submission")
        # Simulates the CPU wait in CUDA split -> CPU conversion. The main
        # thread never reaches a later Gloo error checkpoint or cleanup.
        threading.Event().wait(20)


class DistributedErrorControlTest(unittest.TestCase):
    def controller(self, **kwargs):
        self.reasons = []
        controller = PC1ErrorController(
            store=MemoryStore(), rank=0, world_size=1,
            poll_interval=0.005, exit_callback=self.reasons.append, **kwargs,
        )
        self.addCleanup(self.stop_controller, controller)
        return controller

    @staticmethod
    def stop_controller(controller):
        controller._stop.set()
        controller._watchdog.join(timeout=2)

    def test_repeated_and_nested_operations_and_healthy_close(self):
        controller = self.controller()
        for index in range(3):
            with controller.operation(f"outer {index}"):
                with controller.operation("inner"):
                    self.assertEqual(len(controller._operations), 2)
            self.assertEqual(controller._operations, {})
        self.assertEqual(self.reasons, [])
        store = controller.store
        controller.close()
        controller.close()
        self.assertIs(controller.store, store)
        self.assertFalse(controller._watchdog.is_alive())
        controller.release_store()
        self.assertIsNone(controller.store)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            with controller.operation("late"):
                self.fail("closed controller entered")

    def test_stage_exception_is_fatal_and_keeps_original_cause(self):
        controller = self.controller()
        original = ValueError("projection failed")
        with self.assertRaises(PC1FatalError) as caught:
            with controller.operation("projection"):
                raise original
        self.assertIs(caught.exception.__cause__, original)
        self.assertEqual(len(self.reasons), 1)
        self.assertIn("ValueError: projection failed", self.reasons[0])
        self.assertTrue(controller.failed)
        with self.assertRaises(PC1FatalError):
            controller.close()
        with self.assertRaises(PC1FatalError):
            with controller.operation("next batch"):
                self.fail("failed controller reused")
        self.assertEqual(len(self.reasons), 1)

    def test_nested_scope_cannot_extend_the_outer_deadline(self):
        controller = self.controller(stage_timeout=0.15)
        callback_done = threading.Event()
        controller._exit_callback = lambda reason: (self.reasons.append(reason), callback_done.set())
        with self.assertRaises(PC1FatalError):
            with controller.operation("outer collective"):
                threading.Event().wait(0.08)
                with controller.operation("inner payload"):
                    self.assertTrue(callback_done.wait(0.15))
        self.assertIn("phase=outer collective", self.reasons[0])
        self.assertEqual(len(self.reasons), 1)

    def test_peer_failure_does_not_require_matching_operation_or_collective(self):
        controller = self.controller()
        callback_done = threading.Event()
        controller._exit_callback = lambda reason: (self.reasons.append(reason), callback_done.set())
        controller.store.set("pc1-fatal-rank-0", b"peer pre-submit failure")
        self.assertTrue(callback_done.wait(2))
        self.assertIn("peer pre-submit failure", self.reasons[0])

    def test_control_channel_failure_is_fatal(self):
        class BrokenStore(MemoryStore):
            def check(self, keys):
                raise OSError("control connection lost")

        callback_done = threading.Event()
        reasons = []
        controller = PC1ErrorController(
            store=BrokenStore(), rank=0, world_size=1,
            exit_callback=lambda reason: (reasons.append(reason), callback_done.set()),
        )
        self.addCleanup(self.stop_controller, controller)
        self.assertTrue(callback_done.wait(2))
        self.assertIn("control channel failed", reasons[0])

    def test_failure_to_publish_does_not_prevent_local_exit(self):
        controller = self.controller()

        def broken_set(key, value):
            raise OSError("publish failed")

        controller.store.set = broken_set
        with self.assertRaises(PC1FatalError):
            controller.fail("submit", RuntimeError("local failure"))
        self.assertEqual(len(self.reasons), 1)
        self.assertIn("local failure", self.reasons[0])

    def test_close_during_operation_is_rejected(self):
        controller = self.controller()
        with controller.operation("active"):
            with self.assertRaisesRegex(RuntimeError, "during an operation"):
                controller.close()
        controller.close()

    def test_invalid_timeouts_are_rejected(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                PC1ErrorController(store=MemoryStore(), rank=0, world_size=1, stage_timeout=timeout)

    @unittest.skipUnless(os.name == "posix", "requires POSIX stderr pipe flags")
    def test_default_fatal_exit_does_not_block_on_full_stderr_pipe(self):
        program = """import fcntl, os
from utils.distributed_error_control import _exit_worker
flags = fcntl.fcntl(2, fcntl.F_GETFL)
fcntl.fcntl(2, fcntl.F_SETFL, flags | os.O_NONBLOCK)
try:
    while True:
        os.write(2, b'x' * 4096)
except BlockingIOError:
    pass
fcntl.fcntl(2, fcntl.F_SETFL, flags)
_exit_worker('fatal with a full blocking stderr pipe')
"""
        worker = subprocess.Popen(
            [sys.executable, "-c", program], stderr=subprocess.PIPE,
            cwd=Path(__file__).resolve().parents[1],
        )
        try:
            self.assertEqual(worker.wait(timeout=2), PC1_FATAL_EXIT_CODE)
        finally:
            if worker.poll() is None:
                worker.kill()
            worker.communicate(timeout=2)

    def test_default_fatal_exit_keeps_cpu_diagnostics_when_logging_is_available(self):
        program = """from utils.distributed_error_control import _exit_worker
_exit_worker('PC1 fatal rank=0 phase=projection: injected CPU error')
"""
        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1], timeout=2,
        )
        self.assertEqual(result.returncode, PC1_FATAL_EXIT_CODE)
        self.assertIn("PC1 fatal rank=0 phase=projection: injected CPU error", result.stderr)

    def run_workers(self, local_failure):
        context = mp.get_context("spawn")
        count = 2 if local_failure else 1
        values = context.Array("B", count * SharedArrayStore.slot_size)
        reasons = context.Array("B", count * SharedArrayStore.slot_size)
        ready = context.Barrier(count + 1)
        workers = [context.Process(
            target=blocked_worker,
            args=(rank, values, ready, reasons, local_failure),
        ) for rank in range(count)]
        try:
            for worker in workers:
                worker.start()
            ready.wait(timeout=10)
            for worker in workers:
                worker.join(timeout=4)
            self.assertTrue(all(not worker.is_alive() for worker in workers))
            self.assertEqual([worker.exitcode for worker in workers], [PC1_FATAL_EXIT_CODE] * count)
            reports = SharedArrayStore(reasons)
            return [(rank, reports.get(f"rank-{rank}").decode("utf-8")) for rank in range(count)]
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=2)

    def test_main_thread_blocked_without_exception_exits_at_stage_deadline(self):
        reasons = self.run_workers(local_failure=False)
        self.assertEqual(len(reasons), 1)
        self.assertIn("stage exceeded 0.3s", reasons[0][1])

    def test_pre_collective_failure_exits_both_cpu_workers_in_bounded_time(self):
        reasons = dict(self.run_workers(local_failure=True))
        self.assertIn("injected failure before collective submission", reasons[0])
        self.assertIn("peer failure", reasons[1])


class PipelineAbortLifecycleTest(unittest.TestCase):
    def test_abort_quarantines_handles_and_never_claims_successful_drain(self):
        # Execute the actual production classes without importing gsplat CUDA
        # extensions or requiring PyTorch on the CPU-only development host.
        path = Path(__file__).resolve().parents[1] / "submodules/gsplat/gsplat/distributed_pipeline.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name in {"DistributedRasterizationHandle", "DistributedRasterizationPipeline"}]
        module = ast.Module(body=[ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0,
        ), *classes], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"dataclass": dataclass}
        exec(compile(module, str(path), "exec"), namespace)
        pipeline = namespace["DistributedRasterizationPipeline"].__new__(namespace["DistributedRasterizationPipeline"])
        pipeline._aborted = pipeline._drained = pipeline._closed = False
        pipeline._quarantine = []
        handle = namespace["DistributedRasterizationHandle"](pipeline, 0, 0)
        retained = weakref.ref(handle)
        pipeline._active = {0: handle}
        pipeline.abort()
        pipeline.abort()
        self.assertFalse(pipeline._drained)
        self.assertEqual(pipeline._active, {})
        self.assertEqual(pipeline._quarantine, [handle])
        for operation in (pipeline.drain, pipeline.close, pipeline.submit_projection,
                          handle.start_forward_exchange, handle.wait_and_render,
                          lambda: handle.mark_backward_done(None)):
            with self.subTest(operation=operation), self.assertRaisesRegex(RuntimeError, "worker must exit"):
                operation()
        del handle
        self.assertIsNotNone(retained())


if __name__ == "__main__":
    unittest.main()
