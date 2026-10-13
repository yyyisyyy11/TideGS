"""Check actual handle timing with independent fake compute/comm clocks."""

import ast
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest


class Event:
    def __init__(self, ms=0):
        self.ms = ms

    def record(self, stream):
        self.ms = stream.ms

    def elapsed_time(self, end):
        return end.ms - self.ms


class Stream:
    def __init__(self, ms):
        self.ms = ms

    def wait_event(self, event):
        self.ms = max(self.ms, event.ms)


class PC1RenderTimingTest(unittest.TestCase):
    def test_render_excludes_compute_work_after_exchange(self):
        path = Path(__file__).parents[1] / "submodules/gsplat/gsplat/distributed_pipeline.py"
        tree = ast.parse(path.read_text())
        handle_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                            and node.name == "DistributedRasterizationHandle")
        compute = Stream(7.0)  # P(i+1) runs until 7ms; C1(i) finished at 2ms.

        def render(state):
            compute.ms += 3.0
            return "image", "alpha", {}

        namespace = {
            "__name__": __name__, "dataclass": dataclass,
            "torch": SimpleNamespace(cuda=SimpleNamespace(stream=lambda stream: nullcontext())),
            "time": SimpleNamespace(perf_counter_ns=lambda: 123),
            "_cuda_event": Event,
            "_record_stream_recursive": lambda *args: None,
            "_nvtx_push": lambda name: None, "_nvtx_pop": lambda: None,
            "_render_prepared": render,
        }
        module = ast.Module(body=[ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0,
        ), handle_class], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        handle = namespace["DistributedRasterizationHandle"](
            pipeline=SimpleNamespace(compute_stream=compute, _ensure_not_aborted=lambda: None),
            slot=0, microbatch_index=0, state={}, _started=True, exchange_done=Event(2.0),
        )
        self.assertEqual(handle.wait_and_render()[0], "image")
        self.assertEqual(handle.render_start.ms, 7.0)
        self.assertEqual(handle.render_done.ms, 10.0)
        self.assertEqual(handle.timing()["render_ms"], 3.0)
        self.assertIs(handle.snapshot()["render_start"], handle.render_start)


if __name__ == "__main__":
    unittest.main()
