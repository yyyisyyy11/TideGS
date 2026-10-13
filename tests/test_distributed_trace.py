"""CPU contract tests for opt-in tracing; no CUDA/PyTorch installation needed.

Fake nodes run registered hooks around nonzero simulated device work. This
checks graph cuts, stream attribution, hook lifetime and collective call parity;
it does not claim to validate NCCL/CUDA timing on hardware.
"""

import ast
import gc
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np


GSPLAT = Path(__file__).resolve().parents[1] / "submodules/gsplat/gsplat"


class HookHandle:
    def __init__(self, hooks, hook):
        self.hooks, self.hook = hooks, hook

    def remove(self):
        if self.hook in self.hooks:
            self.hooks.remove(self.hook)


class Node:
    def __init__(self, name, *parents):
        self._name = name
        self.next_functions = tuple((parent, 0) for parent in parents)
        self.prehooks, self.posthooks = [], []

    def name(self):
        return self._name

    def register_prehook(self, hook):
        self.prehooks.append(hook)
        return HookHandle(self.prehooks, hook)

    def register_hook(self, hook):
        self.posthooks.append(hook)
        return HookHandle(self.posthooks, hook)

    def run(self, cuda, stream_id, work_ms=3.0):
        cuda.stream = stream_id
        # These sentinels must remain untouched by observational hooks.
        gradients = (object(), None)
        for hook in self.prehooks:
            assert hook(gradients) is None
        cuda.advance(work_ms)
        for hook in self.posthooks:
            assert hook(gradients, gradients) is None


class FakeCuda:
    def __init__(self):
        self.local = threading.local()
        self.times = {}
        self.events = []
        self.ranges = []
        self.nvtx = SimpleNamespace(range_push=self.push, range_pop=self.pop)

    @property
    def stream(self):
        return getattr(self.local, "stream", 0)

    @stream.setter
    def stream(self, value):
        self.local.stream = value

    def current_stream(self, device):
        return SimpleNamespace(cuda_stream=self.stream)

    def advance(self, amount):
        self.times[self.stream] = self.times.get(self.stream, 0.0) + amount

    def push(self, name):
        if not hasattr(self.local, "stack"):
            self.local.stack = []
        self.local.stack.append(name)
        self.ranges.append(("push", threading.get_ident(), name))

    def pop(self):
        self.ranges.append(("pop", threading.get_ident(), self.local.stack.pop()))

    def Event(self, enable_timing):
        assert enable_timing
        cuda = self

        class Event:
            def record(self, stream):
                self.stream = stream.cuda_stream
                self.time = cuda.times.get(self.stream, 0.0)

            def elapsed_time(self, other):
                return other.time - self.time

        event = Event()
        self.events.append(event)
        return event


def load_trace():
    cuda = FakeCuda()
    fake_torch = SimpleNamespace(
        device=lambda device: SimpleNamespace(type=str(device).split(":")[0]),
        cuda=cuda,
    )
    spec = importlib.util.spec_from_file_location(
        "pc1_trace_under_test", GSPLAT / "distributed_trace.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"torch": fake_torch}):
        spec.loader.exec_module(module)
    return module.MicrobatchTrace, cuda


def tensor(node):
    return SimpleNamespace(grad_fn=node)


class DistributedTraceTest(unittest.TestCase):
    def setUp(self):
        self.Trace, self.cuda = load_trace()

    def test_real_node_spans_partition_multiple_exchanges_and_branches(self):
        leaf = Node("torch::autograd::AccumulateGrad")
        projection = Node("ProjectionBackward", leaf)
        sh = Node("SphericalHarmonicsBackward", leaf)
        c2a = Node("_AlltoAllBackward", projection)
        c2b = Node("_AlltoAllBackward", sh)
        unpack = Node("CatBackward", c2a, c2b)
        render = Node("RasterizeBackward", unpack)
        loss = Node("LossBackward", render)
        trace = self.Trace("cuda:0", 7)
        # All rank outputs share each actual communication node.
        trace.watch_exchange([tensor(c2a), tensor(c2a)])
        trace.watch_exchange([tensor(c2b), tensor(c2b)])
        trace.watch_backward(tensor(loss))
        trace.watch_backward(tensor(loss))  # Registration remains idempotent.
        for node in (loss, render, unpack, c2a, c2b, projection, sh, leaf):
            self.assertEqual(len(node.prehooks), 1)
            node.run(self.cuda, 11 if node in (c2a, c2b) else 22)
        intervals = trace.snapshot()
        self.assertEqual(len(intervals), 8)
        self.assertEqual([record["stage"] for record in intervals], [
            "IntermediateGrad", "IntermediateGrad", "IntermediateGrad",
            "C2", "C2", "FinalGrad", "FinalGrad", "AccumulateGrad",
        ])
        for record in intervals:
            self.assertEqual(record["microbatch_index"], 7)
            self.assertEqual(record["stream_id"], 11 if record["stage"] == "C2" else 22)
            self.assertEqual(record["start_event"].stream, record["stream_id"])
            self.assertEqual(record["end_event"].stream, record["stream_id"])
            self.assertEqual(record["start_event"].elapsed_time(record["end_event"]), 3.0)
            self.assertTrue(record["complete"])
        self.assertEqual(self.cuda.local.stack, [])

    def test_unreachable_exchange_does_not_relabel_loss_branch(self):
        unrelated = Node("_AlltoAllBackward", Node("UnrelatedProjection"))
        loss = Node("LossBackward", Node("CameraAdjustmentBackward"))
        trace = self.Trace("cuda", 0)
        trace.watch_exchange(tensor(unrelated))
        trace.watch_backward(tensor(loss))
        loss.run(self.cuda, 1)
        loss.next_functions[0][0].run(self.cuda, 1)
        self.assertEqual([span["stage"] for span in trace.snapshot()], [
            "IntermediateGrad", "IntermediateGrad",
        ])

    def test_explicit_projection_cut_works_without_communication(self):
        leaf = Node("torch::autograd::AccumulateGrad")
        gaussian = Node("GaussianScaleBackward", leaf)
        projection = Node("ProjectionBackward", gaussian)
        render = Node("RasterizeBackward", projection)
        trace = self.Trace("cuda", 1)
        trace.watch_node([tensor(projection), tensor(projection)], "FinalGrad")
        trace.watch_backward(tensor(render))
        for node in (render, projection, gaussian, leaf):
            node.run(self.cuda, 2)
        self.assertEqual([span["stage"] for span in trace.snapshot()], [
            "IntermediateGrad", "FinalGrad", "FinalGrad", "AccumulateGrad",
        ])

    def test_disabled_trace_creates_no_hooks_events_or_ranges(self):
        node = Node("_AlltoAllBackward")
        trace = self.Trace("cpu", 0, enabled=False)
        trace.watch_exchange(tensor(node))
        trace.watch_node(tensor(node), "FinalGrad")
        trace.watch_backward(tensor(node))
        with trace.scope("Loss"):
            pass
        self.assertEqual(node.prehooks, [])
        self.assertEqual(node.posthooks, [])
        self.assertEqual(self.cuda.events, [])
        self.assertEqual(self.cuda.ranges, [])
        self.assertEqual(trace.snapshot(), [])

    def test_enabled_trace_rejects_cpu_instead_of_faking_gpu_time(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            self.Trace("cpu", 0)

    def test_close_removes_hooks_and_snapshot_releases_graph(self):
        node = Node("_AlltoAllBackward")
        reference = weakref.ref(node)
        trace = self.Trace("cuda", 0)
        trace.watch_exchange(tensor(node))
        node.run(self.cuda, 3)
        snapshot = trace.snapshot()
        trace.close()
        trace.close()
        self.assertEqual(node.prehooks, [])
        self.assertEqual(node.posthooks, [])
        self.assertEqual(trace._nodes, {})
        del node
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(len(snapshot), 1)
        snapshot[0]["stage"] = "edited copy"
        self.assertEqual(trace.snapshot()[0]["stage"], "C2")

    def test_hooks_on_different_threads_have_balanced_thread_local_nvtx(self):
        trace = self.Trace("cuda", 0)
        nodes = [Node("_AlltoAllBackward"), Node("RasterizeBackward")]
        trace.watch_exchange(tensor(nodes[0]))
        trace.watch_node(tensor(nodes[1]), "IntermediateGrad")
        barrier = threading.Barrier(2)
        failures = []

        def run(node, stream):
            try:
                self.cuda.stream = stream
                gradients = (object(),)
                node.prehooks[0](gradients)
                barrier.wait(timeout=5)
                self.cuda.advance(4)
                node.posthooks[0](gradients, gradients)
                self.assertEqual(self.cuda.local.stack, [])
            except Exception as error:
                failures.append(error)

        threads = [threading.Thread(target=run, args=(node, stream))
                   for node, stream in zip(nodes, (55, 66))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=6)
            self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        for record in trace.snapshot():
            self.assertEqual(record["start_event"].stream, record["end_event"].stream)
            self.assertEqual(record["start_event"].elapsed_time(record["end_event"]), 4)
        for tid in {item[1] for item in self.cuda.ranges}:
            operations = [item[0] for item in self.cuda.ranges if item[1] == tid]
            self.assertEqual(operations, ["push", "pop"])

    def test_scope_preserves_exception_and_balances_nvtx(self):
        trace = self.Trace("cuda", 0)
        with self.assertRaisesRegex(RuntimeError, "original"):
            with trace.scope("Loss"):
                self.cuda.advance(5)
                raise RuntimeError("original")
        self.assertEqual(self.cuda.local.stack, [])
        span = trace.snapshot()[0]
        self.assertEqual(span["start_event"].elapsed_time(span["end_event"]), 5)


class ArrayTensor:
    """Enough real array semantics to execute the original exchange helper."""
    def __init__(self, values, requires_grad=False):
        self.values = np.asarray(values)
        self.shape, self.dtype = self.values.shape, self.values.dtype
        self.device = "cuda:0"
        self.requires_grad = requires_grad
        self.grad_fn = None

    def __len__(self):
        return len(self.values)

    def numel(self):
        return self.values.size

    def reshape(self, *shape):
        return ArrayTensor(self.values.reshape(*shape), self.requires_grad)

    view = reshape

    def split(self, sizes, dim=0):
        values = np.split(self.values, np.cumsum(sizes)[:-1], axis=dim)
        return tuple(ArrayTensor(value, self.requires_grad) for value in values)


def load_collective_helper():
    calls = []

    def cat(tensors, dim=0):
        return ArrayTensor(np.concatenate([t.values for t in tensors], axis=dim),
                           any(t.requires_grad for t in tensors))

    def exchange(outputs, inputs, *, differentiable):
        calls.append((differentiable, outputs, inputs))
        node = Node("_AlltoAllBackward") if differentiable else None
        for output, source in zip(outputs, reversed(inputs)):
            output.values[:] = source.values
            output.grad_fn = node
            output.requires_grad = differentiable
        return tuple(outputs)

    fake_torch = SimpleNamespace(
        cat=cat,
        split=lambda value, sizes, dim: value.split(sizes, dim),
        empty=lambda shape, dtype, device: ArrayTensor(np.empty(shape, dtype=dtype)),
        distributed=SimpleNamespace(all_to_all=lambda outputs, inputs: exchange(
            outputs, inputs, differentiable=False)),
    )
    namespace = {
        "torch": fake_torch,
        "distF": SimpleNamespace(all_to_all=lambda outputs, inputs: exchange(
            outputs, inputs, differentiable=True)),
        "Tensor": ArrayTensor,
    }
    source = ast.parse((GSPLAT / "distributed.py").read_text())
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                    and node.name == "all_to_all_tensor_list")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", level=0,
                       names=[ast.alias(name="annotations")]), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(GSPLAT / "distributed.py"),
                 "exec"), namespace)
    return namespace["all_to_all_tensor_list"], calls


class CollectiveTracingParityTest(unittest.TestCase):
    def test_enabled_disabled_and_absent_trace_preserve_collective_and_outputs(self):
        helper, calls = load_collective_helper()
        Trace, cuda = load_trace()
        inputs = [ArrayTensor(np.arange(8).reshape(4, 2), True),
                  ArrayTensor(np.arange(4).reshape(4, 1), True)]
        traces = [None, Trace("cuda", 0, enabled=False), Trace("cuda", 0)]
        results = [helper(2, inputs, [2, 2], [2, 2], trace=trace) for trace in traces]
        self.assertEqual(len(calls), 3)  # No extra forward collective.
        self.assertTrue(all(call[0] for call in calls))
        for output in results:
            for actual, expected in zip(output, results[0]):
                np.testing.assert_array_equal(actual.values, expected.values)
        self.assertEqual(traces[1]._nodes, {})
        self.assertEqual(len(traces[2]._nodes), 1)
        node = next(iter(traces[2]._nodes.values()))
        node.run(cuda, 9, 6)
        self.assertEqual(traces[2].snapshot()[0]["stage"], "C2")

    def test_nondifferentiable_metadata_never_creates_c2(self):
        helper, calls = load_collective_helper()
        Trace, _ = load_trace()
        trace = Trace("cuda", 0)
        inputs = [ArrayTensor(np.arange(4).reshape(4, 1))]
        helper(2, inputs, [2, 2], [2, 2], trace=trace)
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0][0])
        self.assertEqual(trace._nodes, {})

    def test_single_rank_returns_original_tensors_without_events_or_collective(self):
        helper, calls = load_collective_helper()
        Trace, cuda = load_trace()
        trace = Trace("cuda", 0)
        inputs = [ArrayTensor(np.ones((4, 1)), True)]
        self.assertIs(helper(1, inputs, [4], trace=trace), inputs)
        self.assertEqual(calls, [])
        self.assertEqual(cuda.events, [])


if __name__ == "__main__":
    unittest.main()
