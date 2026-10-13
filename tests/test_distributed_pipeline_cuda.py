"""Opt-in, real two-rank CUDA/NCCL numerical and event-order smoke tests.

Run in the pinned two-GPU environment:
    TIDE_RUN_CUDA_SMOKE=1 torchrun --standalone --nproc_per_node=2 \
        -m unittest tests.test_distributed_pipeline_cuda

Fatal workers are isolated in tests.test_distributed_pipeline_cuda_faults.
CPU discovery imports this module and skips without torch/CUDA.
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import timedelta
from pathlib import Path

try:
    import torch
    import torch.distributed as dist
except ImportError:  # CPU developer hosts can have no torch at all.
    torch = None
    dist = None


_RUN_SMOKE = (
    torch is not None
    and dist is not None
    and torch.cuda.is_available()
    and os.environ.get("TIDE_RUN_CUDA_SMOKE") == "1"
    and int(os.environ.get("WORLD_SIZE", "1")) == 2
)
_MICROBATCHES = 4


def cuda_smoke_inputs(rank, local_rank, camera_count=_MICROBATCHES):
    """Deterministic shards with distinct cameras/targets on every rank/MB."""
    torch.manual_seed(17 + rank)
    device = torch.device("cuda", local_rank)
    count = 32
    means = torch.randn((count, 3), device=device) * 0.15
    means[:, 2] = means[:, 2].abs() + 2.0
    means.requires_grad_()
    quats = torch.zeros((count, 4), device=device)
    quats[:, 0] = 1.0
    quats.requires_grad_()
    scales = torch.full((count, 3), 0.08, device=device, requires_grad=True)
    opacities = torch.full((count,), 0.5, device=device, requires_grad=True)
    colors = (torch.randn((count, 16, 3), device=device) * 0.05).requires_grad_()
    viewmats = torch.eye(4, device=device).reshape(1, 4, 4).repeat(camera_count, 1, 1)
    Ks = torch.tensor(
        [[6.0, 0.0, 4.0], [0.0, 6.0, 4.0], [0.0, 0.0, 1.0]], device=device,
    ).reshape(1, 3, 3).repeat(camera_count, 1, 1)
    targets = torch.linspace(0.2, 0.65, 8 * 8 * 3, device=device).reshape(1, 8, 8, 3)
    targets = targets.repeat(camera_count, 1, 1, 1)
    for index in range(camera_count):
        viewmats[index, 0, 3] = 0.07 * rank + 0.025 * index
        viewmats[index, 1, 3] = -0.03 * rank + 0.015 * index
        Ks[index, 0, 0] += 0.2 * rank + 0.1 * index
        targets[index] += 0.025 * rank + 0.01 * index
    return means, quats, scales, opacities, colors, viewmats, Ks, targets


def submit_smoke_projection(pipeline, layout, values, index, background=None):
    means, quats, scales, opacities, colors, viewmats, Ks, _ = values
    global_viewmats, global_Ks = layout.microbatch_cameras(index)
    return pipeline.submit_projection(
        microbatch_index=index, means=means, quats=quats, scales=scales,
        opacities=opacities, colors=colors,
        viewmats=viewmats[index:index + 1], Ks=Ks[index:index + 1],
        viewmats_global=global_viewmats, Ks_global=global_Ks,
        width=8, height=8, sh_degree=0,
        backgrounds=None if background is None else background.reshape(1, 3),
        near_plane=0.01, far_plane=1e10, radius_clip=0.0, eps2d=0.3,
        tile_size=16, rasterize_mode="classic", N_world=layout.N_world,
        C_world=layout.microbatch_C_world,
    )


@unittest.skipUnless(
    _RUN_SMOKE,
    "set TIDE_RUN_CUDA_SMOKE=1 and launch with torchrun on two GPUs",
)
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
        submodule = Path(__file__).parents[1] / "submodules" / "gsplat"
        sys.path.insert(0, str(submodule))
        # An opted-in GPU run must fail if its pinned extension is unusable.
        import gsplat  # noqa: F401

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _inputs(self):
        return cuda_smoke_inputs(self.rank, self.local_rank)

    @staticmethod
    def _loss(images, targets):
        return ((images - targets) ** 2).mean()

    def _baseline(self, values, background=None):
        import gsplat

        means, quats, scales, opacities, colors, viewmats, Ks, targets = values
        outputs = []
        total = means.new_zeros(())
        for index in range(_MICROBATCHES):
            rendered, _, _ = gsplat.rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=colors, viewmats=viewmats[index:index + 1],
                Ks=Ks[index:index + 1], width=8, height=8, sh_degree=0,
                packed=True, distributed=True,
                backgrounds=None if background is None else background.reshape(1, 3),
            )
            outputs.append(rendered.detach())
            total = total + self._loss(rendered, targets[index:index + 1])
        total.backward()
        gradients = tuple(t.grad.detach().clone() for t in values[:5])
        return torch.cat(outputs, dim=0), total.detach(), gradients

    def _pc1(self, values, background=None, capture_timeline=True):
        from gsplat.distributed_pipeline import (
            distributed_rasterization_pipeline, prepare_batch_camera_layout,
        )

        means, _, _, _, _, viewmats, Ks, targets = values
        layout = prepare_batch_camera_layout(
            means=means, viewmats=viewmats, Ks=Ks, microbatch=1,
        )
        compute_stream = torch.cuda.current_stream()
        pipeline = distributed_rasterization_pipeline(
            compute_stream=compute_stream, comm_stream=torch.cuda.Stream(priority=-1),
            buffer_count=2, capture_timeline=capture_timeline,
        )
        snapshots, outputs = [], []
        total = means.new_zeros(())
        current = submit_smoke_projection(pipeline, layout, values, 0, background)
        for index in range(_MICROBATCHES):
            current.start_forward_exchange()
            next_handle = None
            if index + 1 < _MICROBATCHES:
                next_handle = submit_smoke_projection(
                    pipeline, layout, values, index + 1, background,
                )
            rendered, _, _ = current.wait_and_render()
            outputs.append(rendered.detach())
            loss = self._loss(rendered, targets[index:index + 1])
            total = total + loss.detach()
            current.trace_backward(loss)
            loss.backward()
            backward_done = torch.cuda.Event(enable_timing=True)
            backward_done.record(compute_stream)
            current.mark_backward_done(backward_done)
            # Save after backward; saving before it silently misses all stage spans.
            snapshots.append(current.snapshot())
            current = next_handle
        pipeline.drain()
        pipeline.close()
        gradients = tuple(t.grad.detach().clone() for t in values[:5])
        return torch.cat(outputs, dim=0), total.detach(), gradients, snapshots

    def test_off_and_pc1_match_with_reuse_and_real_stage_events(self):
        for label, color in (
            ("default", None), ("white", (1.0, 1.0, 1.0)),
            ("custom", (0.1, 0.2, 0.3)),
        ):
            with self.subTest(background=label):
                background = None if color is None else torch.tensor(
                    color, device=torch.device("cuda", self.local_rank),
                )
                baseline = self._baseline(self._inputs(), background)
                traced = self._pc1(self._inputs(), background)
                self._assert_numerical_match(traced[:3], baseline)
                self._assert_events(traced[3], label)
                if label == "default":
                    untraced = self._pc1(self._inputs(), capture_timeline=False)
                    self._assert_numerical_match(untraced[:3], baseline)
                    self._assert_numerical_match(untraced[:3], traced[:3])
                    for snapshot in untraced[3]:
                        self.assertFalse(snapshot.get("backward_trace"))

    def _assert_numerical_match(self, actual, expected):
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-4, atol=1e-5)
        self.assertEqual(len(actual[2]), 5)
        for actual_grad, expected_grad in zip(actual[2], expected[2]):
            self.assertTrue(torch.isfinite(actual_grad).all().item())
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-5)

    def _assert_events(self, snapshots, label):
        self.assertEqual(len(snapshots), _MICROBATCHES)
        self.assertEqual([s["pipeline_buffer_slot"] for s in snapshots], [0, 1, 0, 1])
        stages = {"IntermediateGrad", "C2", "FinalGrad", "AccumulateGrad"}
        # The existing drain is the only timing sync required by these checks.
        for index, snapshot in enumerate(snapshots):
            self.assertEqual(snapshot["microbatch_index"], index)
            for name in (
                "projection_start", "projection_end", "projection_done",
                "metadata_start", "metadata_end", "payload_start", "payload_end",
                "exchange_done", "render_start", "render_done", "backward_done",
            ):
                self.assertIsNotNone(snapshot[name], (index, name))
                self.assertTrue(snapshot[name].query(), (index, name))
            self.assertGreaterEqual(
                snapshot["exchange_done"].elapsed_time(snapshot["render_start"]), -0.01,
            )
            # P(i+1) is enqueued on compute before Render(i); Render(i) must
            # respect both that stream order and its cross-stream C1 dependency.
            if index + 1 < _MICROBATCHES:
                self.assertGreaterEqual(
                    snapshots[index + 1]["projection_done"].elapsed_time(snapshot["render_start"]),
                    -0.01,
                )
            if index >= 2:
                self.assertGreaterEqual(
                    snapshots[index - 2]["backward_done"].elapsed_time(snapshot["projection_start"]),
                    -0.01,
                )
            spans = snapshot["backward_trace"]
            self.assertTrue(spans, index)
            self.assertTrue(stages.issubset({span["stage"] for span in spans}), index)
            for span in spans:
                self.assertTrue(span["complete"], (index, span["stage"]))
                self.assertEqual(span["microbatch_index"], index)
                self.assertIsInstance(span["stream_id"], int)
                self.assertTrue(span["node_name"])
                self.assertIsNotNone(span["start_event"])
                self.assertIsNotNone(span["end_event"])
                self.assertTrue(span["end_event"].query())
                self.assertGreaterEqual(span["start_event"].elapsed_time(span["end_event"]), 0.0)
        overlaps = []
        for index in range(_MICROBATCHES - 1):
            c1_start, c1_end = snapshots[index]["metadata_start"], snapshots[index]["payload_end"]
            next_start = snapshots[index + 1]["projection_start"]
            next_end = snapshots[index + 1]["projection_end"]
            overlap_ms = max(
                0.0,
                min(c1_start.elapsed_time(c1_end), c1_start.elapsed_time(next_end))
                - max(0.0, c1_start.elapsed_time(next_start)),
            )
            print(
                f"rank={self.rank} background={label} "
                f"P({index + 1})/C1({index}) overlap_ms={overlap_ms:.3f}", flush=True,
            )
            overlaps.append(overlap_ms)
        if os.environ.get("TIDE_REQUIRE_PC1_OVERLAP") == "1":
            self.assertGreater(max(overlaps), 0.0)


if __name__ == "__main__":
    unittest.main()
