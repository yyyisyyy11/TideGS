"""Opt-in real two-rank numerical, event and repeated-batch PC1 checks.

Use the bounded parent: TIDE_RUN_CUDA_SMOKE=1 python tools/run_pc1_cuda_smoke.py
Direct entry: torchrun --standalone --nproc_per_node=2 -m unittest \
    tests.test_distributed_pipeline_cuda -v
CPU discovery skips cleanly when torch/CUDA are absent.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import timedelta
import gc
from functools import wraps
import os
from pathlib import Path
import sys
import threading
import time
import unittest

try:
    import torch
    import torch.distributed as dist
except ImportError:
    torch = None
    dist = None


_RUN_SMOKE = (
    torch is not None and dist is not None and torch.cuda.is_available()
    and os.environ.get("TIDE_RUN_CUDA_SMOKE") == "1"
    and int(os.environ.get("WORLD_SIZE", "1")) == 2
)
_MICROBATCHES = 4


@contextmanager
def _smoke_nvtx(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _smoke_batch_ranges(function):
    @wraps(function)
    def measured(*args, **kwargs):
        with _smoke_nvtx("tide.batch.step"), _smoke_nvtx("tide.batch.compute"):
            return function(*args, **kwargs)
    return measured


def cuda_smoke_inputs(rank, local_rank, camera_count=_MICROBATCHES, *, case="uneven"):
    """Different owner sizes, with explicit empty/sentinel/zero-peer cases."""
    torch.manual_seed(17 + rank)
    device = torch.device("cuda", local_rank)
    count = 32 + 11 * rank
    if case == "sentinel" and rank == 0:
        count = 1
    means = torch.randn((count, 3), device=device) * 0.15
    means[:, 2] = means[:, 2].abs() + 2.0
    if case in {"all_empty", "packed_zero"} and (rank == 0 or case == "all_empty"):
        means[:, 2] = -2.0
    if case == "sentinel" and rank == 0:
        means.zero_()  # Actual empty-owner sentinel projects to zero packed rows.
    means.requires_grad_()
    quats = torch.zeros((count, 4), device=device)
    quats[:, 0] = 1.0
    quats.requires_grad_()
    scales = torch.full((count, 3), 0.08, device=device, requires_grad=True)
    opacities = torch.full((count,), 0.5, device=device)
    if case == "sentinel" and rank == 0:
        opacities.zero_()
    opacities.requires_grad_()
    colors = (torch.randn((count, 16, 3), device=device) * 0.05).requires_grad_()
    viewmats = torch.eye(4, device=device).reshape(1, 4, 4).repeat(camera_count, 1, 1)
    Ks = torch.tensor(
        [[6.0, 0.0, 4.0], [0.0, 6.0, 4.0], [0.0, 0.0, 1.0]], device=device,
    ).reshape(1, 3, 3).repeat(camera_count, 1, 1)
    targets = torch.linspace(0.2, 0.65, 8 * 8 * 3, device=device).reshape(1, 8, 8, 3)
    targets = targets.repeat(camera_count, 1, 1, 1)
    for index in range(camera_count):
        viewmats[index, 0, 3] = 0.07 * rank + 0.025 * index
        if case == "zero_peer" and rank == 1:
            viewmats[index, 0, 3] = 1000.0
        viewmats[index, 1, 3] = -0.03 * rank + 0.015 * index
        Ks[index, 0, 0] += 0.2 * rank + 0.1 * index
        targets[index] += 0.025 * rank + 0.01 * index
    return means, quats, scales, opacities, colors, viewmats, Ks, targets


def submit_smoke_projection(
    pipeline, layout, values, index, background=None, *, sh_degree=0, deadline=None,
):
    means, quats, scales, opacities, colors, viewmats, Ks, _ = values
    global_viewmats, global_Ks = layout.microbatch_cameras(index)
    return pipeline.submit_projection(
        microbatch_index=index, deadline=deadline, means=means, quats=quats, scales=scales,
        opacities=opacities, colors=colors,
        viewmats=viewmats[index:index + 1], Ks=Ks[index:index + 1],
        viewmats_global=global_viewmats, Ks_global=global_Ks,
        width=8, height=8, sh_degree=sh_degree,
        backgrounds=None if background is None else background.reshape(1, 3),
        near_plane=0.01, far_plane=1e10, radius_clip=0.0, eps2d=0.3,
        tile_size=16, rasterize_mode="classic", N_world=layout.N_world,
        C_world=layout.microbatch_C_world,
    )


class _SmokeController:
    # The parent process imposes a hard deadline even if CUDA blocks Python.
    stage_timeout = 60.0

    @staticmethod
    def operation(phase):
        return nullcontext()


@unittest.skipUnless(_RUN_SMOKE, "set TIDE_RUN_CUDA_SMOKE=1 on a two-GPU host")
class DistributedPipelineCudaSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rank = int(os.environ["RANK"])
        cls.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(cls.local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl", init_method="env://", timeout=timedelta(seconds=60),
            )
        sys.path.insert(0, str(Path(__file__).parents[1] / "submodules" / "gsplat"))
        from utils.pc1_native_capabilities import check_pc1_native_capabilities
        from gsplat.distributed_pipeline import PipelineCommWorker

        check_pc1_native_capabilities()
        cls.worker = PipelineCommWorker(
            device=torch.device("cuda", cls.local_rank),
            comm_stream=torch.cuda.Stream(priority=-1), buffer_count=2,
            default_timeout_s=60.0,
        )
        cls.controller = _SmokeController()
        cls.batch_index = 0
        cls._real_loss_warmed = False

    @classmethod
    def tearDownClass(cls):
        deadline = time.monotonic() + 60.0
        cls.worker.drain(deadline)
        cls.worker.close(deadline)
        if dist.is_initialized():
            dist.destroy_process_group()

    def _inputs(self, camera_count=_MICROBATCHES, case="uneven"):
        return cuda_smoke_inputs(self.rank, self.local_rank, camera_count, case=case)

    @staticmethod
    def _loss(images, targets, real_loss=False):
        if real_loss:
            from strategies.base_engine import torch_compiled_loss

            return sum(
                torch_compiled_loss(
                    image.permute(2, 0, 1).contiguous(),
                    (target * 255.0).permute(2, 0, 1).contiguous(),
                )
                for image, target in zip(images, targets)
            )
        return ((images - targets) ** 2).mean()

    def _warm_loss(self, values):
        if type(self)._real_loss_warmed:
            return
        image = torch.zeros_like(values[-1][:1], requires_grad=True)
        torch.autograd.grad(self._loss(image, values[-1][:1], True), image)
        torch.cuda.current_stream().synchronize()
        type(self)._real_loss_warmed = True

    @staticmethod
    def _gradients(values):
        return tuple(
            torch.zeros_like(t) if t.grad is None else t.grad.detach().clone()
            for t in values[:5]
        )

    @_smoke_batch_ranges
    def _baseline(self, values, background=None, *, sh_degree=0, real_loss=False, **unused):
        import gsplat

        if real_loss:
            self._warm_loss(values)
        means, quats, scales, opacities, colors, viewmats, Ks, targets = values
        outputs, total = [], means.new_zeros(())
        for index in range(len(viewmats)):
            rendered, _, _ = gsplat.rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=colors, viewmats=viewmats[index:index + 1],
                Ks=Ks[index:index + 1], width=8, height=8, sh_degree=sh_degree,
                packed=True, distributed=True,
                backgrounds=None if background is None else background.reshape(1, 3),
            )
            loss = self._loss(rendered, targets[index:index + 1], real_loss)
            outputs.append(rendered.detach().clone())
            total = total + loss.detach()
            if loss.requires_grad:
                loss.backward()
        return torch.cat(outputs), total, self._gradients(values)

    @_smoke_batch_ranges
    def _pc1(
        self, values, background=None, *, capture_timeline=True,
        sh_degree=0, real_loss=False, nondefault_stream=False, serial_reference=False,
    ):
        from gsplat.distributed_pipeline import (
            distributed_rasterization_pipeline, prepare_batch_camera_layout,
        )
        from strategies.tide_engine.pc1_schedule import run_pc1_windows

        if real_loss:
            self._warm_loss(values)
        means, _, _, _, _, viewmats, Ks, targets = values
        compute_stream = (
            torch.cuda.Stream() if nondefault_stream else torch.cuda.current_stream()
        )
        compute_stream.wait_stream(torch.cuda.current_stream())
        snapshots, outputs = [], []
        total = means.new_zeros(())
        type(self).batch_index += 1
        with torch.cuda.stream(compute_stream):
            layout = prepare_batch_camera_layout(
                means=means, viewmats=viewmats, Ks=Ks, microbatch=1,
            )
            pipeline = distributed_rasterization_pipeline(
                compute_stream=compute_stream, comm_stream=self.worker.comm_stream,
                comm_worker=self.worker, buffer_count=2, capture_timeline=capture_timeline,
                batch_index=type(self).batch_index,
            )

            def project(index, *, deadline):
                handle = submit_smoke_projection(
                    pipeline, layout, values, index, background,
                    sh_degree=sh_degree, deadline=deadline,
                )
                handle.start_forward_exchange(deadline=deadline)
                return handle

            def image(index, handle, *, deadline):
                nonlocal total
                before_grads = [(id(t.grad), None if t.grad is None else t.grad._version)
                                for t in values[:5]]
                rendered, _, _ = handle.wait_and_render(deadline=deadline)
                outputs.append(rendered.detach().clone())
                loss = self._loss(rendered, targets[index:index + 1], real_loss)
                total = total + loss.detach()
                handle.intermediate_backward(loss, deadline=deadline)
                # P and M must never accumulate into Gaussian parameter leaves.
                self.assertEqual(before_grads, [
                    (id(t.grad), None if t.grad is None else t.grad._version)
                    for t in values[:5]
                ])

            def final(index, handle, *, deadline):
                handle.final_backward(deadline=deadline)
                snapshots.append(handle.snapshot())

            run_pc1_windows(
                range(len(viewmats)), buffer_count=1 if serial_reference else 2,
                controller=self.controller, project=project,
                image_backward=image, final_backward=final,
            )
            pipeline.drain(time.monotonic() + 60.0)
            pipeline.close()
        torch.cuda.current_stream().wait_stream(compute_stream)
        return torch.cat(outputs), total, self._gradients(values), snapshots

    def _assert_numerical_match(self, actual, expected):
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-4, atol=1e-5)
        self.assertEqual(len(actual[2]), 5)
        for actual_grad, expected_grad in zip(actual[2], expected[2]):
            self.assertTrue(torch.isfinite(actual_grad).all().item())
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-5)

    def _compare_three_paths(self, *, count=4, case="uneven", background=None, **kwargs):
        baseline = self._baseline(self._inputs(count, case), background, **kwargs)
        serial = self._pc1(
            self._inputs(count, case), background, serial_reference=True, **kwargs,
        )
        threaded = self._pc1(self._inputs(count, case), background, **kwargs)
        self._assert_numerical_match(serial[:3], baseline)
        self._assert_numerical_match(threaded[:3], baseline)
        self._assert_events(threaded[3], count, kwargs.get("capture_timeline", True))
        return threaded

    def test_one_through_four_microbatches_and_slot_reuse(self):
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                self._compare_three_paths(count=count)

    def test_backgrounds_sh_trace_and_nondefault_stream(self):
        device = torch.device("cuda", self.local_rank)
        for label, background, degree, trace, stream in (
            ("default", None, 0, False, False),
            ("white", (1.0, 1.0, 1.0), 3, True, True),
            ("custom", (0.1, 0.2, 0.3), 3, False, True),
        ):
            with self.subTest(background=label, sh_degree=degree, trace=trace, stream=stream):
                color = None if background is None else torch.tensor(background, device=device)
                self._compare_three_paths(
                    background=color, sh_degree=degree, capture_timeline=trace,
                    nondefault_stream=stream,
                )

    def test_zero_peer_packed_zero_and_empty_owner_sentinel(self):
        for case in ("zero_peer", "packed_zero", "all_empty", "sentinel"):
            with self.subTest(case=case):
                self._compare_three_paths(case=case, sh_degree=3)

    def test_actual_compiled_loss_on_nondefault_compute_stream(self):
        self._compare_three_paths(real_loss=True, nondefault_stream=True, sh_degree=3)

    def test_repeated_batches_match_adam_parameters_and_all_states(self):
        trajectories = []
        for mode in ("off", "serial", "pc1"):
            values = self._inputs()
            optimizer = torch.optim.Adam(values[:5], lr=1e-3, foreach=False)
            memories, peaks, tensor_counts, worker_counts = [], [], [], []
            for _ in range(3):
                torch.cuda.reset_peak_memory_stats()
                optimizer.zero_grad(set_to_none=True)
                if mode == "off":
                    result = self._baseline(values, sh_degree=3)
                else:
                    result = self._pc1(
                        values, sh_degree=3, serial_reference=mode == "serial",
                    )
                # Adam state must exist for every Gaussian parameter component,
                # including features with zero gradient at the active SH degree.
                for parameter in values[:5]:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                optimizer.step()
                del result
                gc.collect()
                torch.cuda.current_stream().synchronize()
                memories.append(torch.cuda.memory_allocated())
                peaks.append(torch.cuda.max_memory_allocated())
                tensor_counts.append(sum(isinstance(value, torch.Tensor) for value in gc.get_objects()))
                worker_counts.append(sum(t is self.worker.thread for t in threading.enumerate()))
            self.assertEqual(worker_counts, [1, 1, 1])
            self.assertLessEqual(memories[-1], memories[0] + 2 * 1024 * 1024)
            self.assertLessEqual(tensor_counts[-1], tensor_counts[0])
            print(f"rank={self.rank} mode={mode} allocated_bytes={memories} "
                  f"peak_bytes={peaks} live_tensors={tensor_counts} comm_threads={worker_counts}",
                  flush=True)
            trajectories.append((
                tuple(t.detach().clone() for t in values[:5]),
                tuple({key: value.detach().clone() for key, value in optimizer.state[t].items()}
                      for t in values[:5]),
            ))
        for parameters, states in trajectories[1:]:
            for actual, expected in zip(parameters, trajectories[0][0]):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
            for actual_state, expected_state in zip(states, trajectories[0][1]):
                self.assertEqual(set(actual_state), set(expected_state))
                for key in actual_state:
                    torch.testing.assert_close(
                        actual_state[key], expected_state[key], rtol=1e-5, atol=1e-7,
                    )

    def _assert_events(self, snapshots, count, traced):
        self.assertEqual(len(snapshots), count)
        self.assertEqual([s["pipeline_buffer_slot"] for s in snapshots],
                         [index % 2 for index in range(count)])
        for index, snapshot in enumerate(snapshots):
            self.assertEqual(snapshot["microbatch_index"], index)
            for name in (
                "projection_start", "projection_end", "projection_done",
                "metadata_start", "metadata_end", "payload_start", "payload_end",
                "exchange_done", "render_start", "render_done", "intermediate_start",
                "intermediate_done", "c2_payload_start", "c2_payload_end",
                "final_start", "backward_done",
            ):
                self.assertIsNotNone(snapshot[name], (index, name))
                self.assertTrue(snapshot[name].query(), (index, name))
            for ready, consumer in (
                ("exchange_done", "render_start"),
                ("intermediate_done", "c2_payload_start"),
                ("c2_payload_end", "final_start"),
            ):
                self.assertGreaterEqual(snapshot[ready].elapsed_time(snapshot[consumer]), -0.01)
            window_last = min(count - 1, (index // 2) * 2 + 1)
            self.assertGreaterEqual(
                snapshots[window_last]["projection_done"].elapsed_time(snapshot["render_start"]),
                -0.01,
            )
            if index >= 2:
                self.assertGreaterEqual(
                    snapshots[index - 2]["backward_done"].elapsed_time(snapshot["projection_start"]),
                    -0.01,
                )
            if not traced:
                self.assertFalse(snapshot.get("backward_trace"))
            else:
                spans = snapshot["backward_trace"]
                self.assertTrue(spans)
                self.assertTrue({"IntermediateGrad", "FinalGrad"}.issubset(
                    {span["stage"] for span in spans}
                ))
                for span in spans:
                    self.assertTrue(span["complete"])
                    self.assertTrue(span["end_event"].query())
        # These events bound stage envelopes; Nsight is required to prove
        # actual overlap between compute kernels and NCCL kernels.
        for index in range(0, count - 1, 2):
            left, right = snapshots[index], snapshots[index + 1]
            start = left["metadata_start"]
            overlap = max(0.0, min(
                start.elapsed_time(left["payload_end"]),
                start.elapsed_time(right["projection_end"]),
            ) - max(0.0, start.elapsed_time(right["projection_start"])))
            print(f"rank={self.rank} P({index + 1})/C1({index}) envelope_estimate_ms={overlap:.3f}",
                  flush=True)


if __name__ == "__main__":
    unittest.main()
