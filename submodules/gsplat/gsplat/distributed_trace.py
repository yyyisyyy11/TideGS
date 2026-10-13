"""Opt-in CUDA intervals for the actual PC1 backward autograd nodes.

Events bracket node execution on the stream selected by the autograd engine.
For NCCL this is the caller-stream readiness interval, including the collective's
stream wait, rather than a claim about pure NCCL kernel occupancy.  Each node is
reported separately: merging the first/last timestamps would count idle gaps
and potentially overlap unrelated streams.  No CUDA synchronization is added.
"""

from __future__ import annotations

from contextlib import contextmanager
import threading
import time
from typing import Any

import torch


class MicrobatchTrace:
    """Observe one microbatch without replacing Functions or gradient tensors.

    Pass this object to ``all_to_all_tensor_list(trace=...)`` during C1, then call
    ``watch_backward(loss)`` before the ordinary single ``loss.backward()``.
    ``snapshot()`` contains only event objects and scalar metadata.  Call
    ``close()`` after backward returns to release retained node references/hooks.
    Read event timings only after the caller's existing measurement sync.

    ``AccumulateGrad`` measures the real leaf accumulation node. It does not
    describe every reduction/gradient addition or an optimizer update. In
    particular, autograd engine input-buffer additions happen between node
    invocations and are not independently timed by these hooks.
    """

    def __init__(self, device, microbatch_index: int, enabled: bool = True):
        self.enabled = bool(enabled)
        self.device = torch.device(device) if self.enabled else device
        self.microbatch_index = int(microbatch_index)
        if self.enabled and self.device.type != "cuda":
            raise ValueError("MicrobatchTrace requires a CUDA device when enabled")
        self._intervals: list[dict[str, Any]] = []
        # Keeping the Python wrappers alive makes identity-based graph cuts
        # stable; remove_hooks releases these after backward, before reuse.
        self._nodes: dict[int, Any] = {}
        self._node_stages: dict[int, str] = {}
        self._exchange_nodes: set[int] = set()
        self._handles: list[Any] = []
        self._thread_state = threading.local()
        self._lock = threading.Lock()

    def _start(self, stage: str, node_name: str = "") -> dict[str, Any]:
        stream = torch.cuda.current_stream(self.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        interval = {
            "stage": stage,
            "microbatch_index": self.microbatch_index,
            "start_event": start,
            "end_event": end,
            "stream_id": int(stream.cuda_stream),
            "node_name": node_name,
            "host_start_ns": time.perf_counter_ns(),
            "host_end_ns": None,
            "complete": False,
            "timing_kind": "cuda_node_interval" if node_name else "cuda_stage_interval",
        }
        torch.cuda.nvtx.range_push(
            f"Tide:S1:MB{self.microbatch_index}:{stage}"
            + (f":{node_name}" if node_name else "")
        )
        start.record(stream)
        with self._lock:
            self._intervals.append(interval)
        return interval

    def _finish(self, interval: dict[str, Any]) -> None:
        # Autograd executes a node's pre/post hooks on the same thread and
        # stream.  Never record its end on the engine caller's default stream.
        interval["end_event"].record(torch.cuda.current_stream(self.device))
        interval["host_end_ns"] = time.perf_counter_ns()
        interval["complete"] = True
        torch.cuda.nvtx.range_pop()

    @contextmanager
    def scope(self, stage: str):
        """Optional forward/loss interval; disabled scopes allocate no events."""
        if not self.enabled:
            yield
            return
        interval = self._start(stage)
        try:
            yield
        finally:
            self._finish(interval)

    def _watch(self, node, stage: str) -> None:
        identity = id(node)
        if identity in self._nodes:
            return
        self._nodes[identity] = node
        self._node_stages[identity] = stage
        node_name = node.name()

        def before(grad_outputs):
            pending = getattr(self._thread_state, "pending", None)
            if pending is None:
                pending = self._thread_state.pending = {}
            pending.setdefault(identity, []).append(self._start(stage, node_name))
            # None preserves the original grad_outputs, including unused ones.

        def after(grad_inputs, grad_outputs):
            interval = self._thread_state.pending[identity].pop()
            self._finish(interval)
            # None preserves the exact original grad_inputs.

        self._handles.extend((node.register_prehook(before), node.register_hook(after)))

    def watch_node(self, tensors, stage: str) -> None:
        """Optionally label kernel nodes; list outputs sharing a node dedupe.

        Use FinalGrad for projection/SH outputs, or IntermediateGrad for raster
        outputs. This is useful when world_size=1 has no C2 graph cut.
        """
        if not self.enabled:
            return
        if not isinstance(tensors, (tuple, list)):
            tensors = (tensors,)
        for tensor in tensors:
            node = getattr(tensor, "grad_fn", None)
            if node is not None:
                self._watch(node, stage)

    def watch_exchange(self, tensors) -> None:
        """Mark every actual differentiable all-to-all node as a C2 cut."""
        if not self.enabled:
            return
        if not isinstance(tensors, (tuple, list)):
            tensors = (tensors,)
        for tensor in tensors:
            node = getattr(tensor, "grad_fn", None)
            if node is not None:
                self._exchange_nodes.add(id(node))
                self._watch(node, "C2")

    def watch_backward(self, loss) -> None:
        """Classify reachable nodes on either side of all observed C2 cuts.

        Loss/render and packing backward downstream of C2 are IntermediateGrad.
        C2 ancestors, including projection/SH and Gaussian-side packing, are
        FinalGrad. Multiple exchanges/branches produce multiple real intervals.
        The metadata nodes that do not require gradients never become C2.
        """
        if not self.enabled:
            return
        roots = loss if isinstance(loss, (tuple, list)) else (loss,)
        reachable = {}
        pending = [getattr(tensor, "grad_fn", None) for tensor in roots]
        while pending:
            node = pending.pop()
            if node is None or id(node) in reachable:
                continue
            reachable[id(node)] = node
            pending.extend(parent for parent, _ in node.next_functions)

        final = set()
        # Explicit FinalGrad kernel labels are additional cuts for world_size=1.
        cuts = self._exchange_nodes | {
            identity for identity, stage in self._node_stages.items()
            if stage == "FinalGrad"
        }
        pending = [
            parent for identity in cuts if identity in reachable
            for parent, _ in reachable[identity].next_functions
        ]
        while pending:
            node = pending.pop()
            if node is None or id(node) in final:
                continue
            final.add(id(node))
            pending.extend(parent for parent, _ in node.next_functions)

        for identity, node in reachable.items():
            if identity in self._nodes:
                continue
            if node.name().split("::")[-1] == "AccumulateGrad":
                stage = "AccumulateGrad"
            else:
                stage = "FinalGrad" if identity in final else "IntermediateGrad"
            self._watch(node, stage)

    def snapshot(self) -> list[dict[str, Any]]:
        """Copy timing metadata without including tensors, nodes or hooks."""
        with self._lock:
            return [dict(interval) for interval in self._intervals]

    def remove_hooks(self) -> None:
        """Release graph references after backward has finished on the host."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._nodes.clear()
        self._node_stages.clear()
        self._exchange_nodes.clear()

    def close(self) -> None:
        self.remove_hooks()
