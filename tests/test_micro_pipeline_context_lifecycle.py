"""CPU behavior checks for the rank-owned worker and healthy teardown order."""

import ast
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import math
import os
from datetime import timedelta
from pathlib import Path
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from utils.distributed_error_control import MicroPipelineErrorController, MicroPipelineFatalError
from utils.pipeline_mode import normalize_pipeline_mode


def context_class(dist):
    path = Path(__file__).parents[1] / "utils/distributed.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "DistributedContext")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"distributed_mode", "_positive_timeout"}]
    namespace = dict(dataclass=dataclass, field=field, Any=object, List=list,
                     dist=dist, torch=SimpleNamespace(
                         device=lambda *args: args,
                         cuda=SimpleNamespace(is_available=lambda: True, set_device=lambda value: None)),
                     time=time, json=json, os=os, math=math, timedelta=timedelta,
                     normalize_pipeline_mode=normalize_pipeline_mode,
                     MicroPipelineErrorController=MicroPipelineErrorController,
                     _DISTRIBUTED_MODES={"off", "gaussian_sharded"})
    exec(compile(ast.Module(body=[*functions, node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["DistributedContext"]


class MicroPipelineContextLifecycleTest(unittest.TestCase):
    def test_watchdog_shutdown_has_a_deadline_even_when_control_io_blocks(self):
        entered, release = threading.Event(), threading.Event()
        reasons = []

        def check(keys):
            entered.set()
            release.wait(timeout=2)
            return False

        controller = MicroPipelineErrorController(
            store=SimpleNamespace(check=check, set=lambda *args: None),
            rank=0, world_size=1, stage_timeout=1,
            poll_interval=0.01, exit_callback=reasons.append,
        )
        try:
            self.assertTrue(entered.wait(timeout=1))
            with self.assertRaisesRegex(MicroPipelineFatalError, "watchdog shutdown"):
                controller.close(time.monotonic() + 0.02)
            self.assertEqual(len(reasons), 1)
        finally:
            release.set()
            controller._watchdog.join(timeout=2)

    def test_native_loading_precedes_one_persistent_worker(self):
        calls = []
        main_thread = threading.get_native_id()
        native = ModuleType("utils.micro_pipeline_native_capabilities")

        def check(**kwargs):
            self.assertEqual(threading.get_native_id(), main_thread)
            calls.append(("native", kwargs))
            return {"abi": 1}

        native.check_micro_pipeline_native_capabilities = check
        transport = ModuleType("gsplat.distributed_pipeline")

        def worker(**kwargs):
            calls.append(("worker", kwargs["buffer_count"]))
            return SimpleNamespace(buffer_count=kwargs["buffer_count"])

        transport.PipelineCommWorker = worker
        controller = SimpleNamespace(stage_timeout=1, fail=lambda *args: None,
                                     operation=lambda phase: nullcontext())
        context = context_class(None)(mode="gaussian_sharded", micro_pipeline_error_controller=controller)
        with patch.dict("sys.modules", {"utils.micro_pipeline_native_capabilities": native,
                                        "gsplat.distributed_pipeline": transport}):
            with patch("builtins.print"):
                first = context.ensure_pipeline_comm_worker(buffer_count=2, require_tile_mask=True)
            second = context.ensure_pipeline_comm_worker(buffer_count=2)
            self.assertIs(first, second)
            with self.assertRaisesRegex(RuntimeError, "cannot change"):
                context.ensure_pipeline_comm_worker(buffer_count=3)
        self.assertEqual(calls, [("native", {"require_tile_mask": True}), ("worker", 2)])

    def test_worker_finishes_before_watchdog_and_groups(self):
        calls = []
        worker = SimpleNamespace(drain=lambda deadline: calls.append("drain"),
                                 close=lambda deadline: calls.append("worker close"))
        controller = SimpleNamespace(stage_timeout=1, operation=lambda phase: nullcontext(),
                                     close=lambda: calls.append("controller close"),
                                     release_store=lambda: calls.append("release store"))
        dist = SimpleNamespace(is_initialized=lambda: True,
                               barrier=lambda **kwargs: calls.append("gloo barrier"),
                               destroy_process_group=lambda *args: calls.append(("destroy", args)))
        context = context_class(dist)(mode="gaussian_sharded", error_group="gloo",
                                      micro_pipeline_error_controller=controller, pipeline_comm_worker=worker)
        context.close()
        self.assertEqual(calls, ["drain", "worker close", "controller close", "gloo barrier",
                                 "release store", ("destroy", ("gloo",)), ("destroy", ())])

    def test_failed_drain_leaves_supervision_and_groups_alive(self):
        calls = []

        def drain(deadline):
            raise TimeoutError("GPU event pending")

        controller = SimpleNamespace(stage_timeout=1, operation=lambda phase: nullcontext(),
                                     close=lambda: calls.append("controller close"))
        worker = SimpleNamespace(drain=drain, close=lambda deadline: calls.append("worker close"))
        context = context_class(None)(mode="gaussian_sharded", micro_pipeline_error_controller=controller,
                                      pipeline_comm_worker=worker)
        with self.assertRaisesRegex(TimeoutError, "GPU event pending"):
            context.close()
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
