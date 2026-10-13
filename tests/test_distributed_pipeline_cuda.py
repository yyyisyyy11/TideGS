"""Opt-in two-rank CUDA/NCCL smoke test for the PC1 pipeline.

Run on a two-GPU host with the pinned gsplat environment:

    TIDE_RUN_CUDA_SMOKE=1 torchrun --nproc_per_node=2 \
        -m unittest tests.test_distributed_pipeline_cuda

The normal unit-test discovery skips this file unless the explicit opt-in is
set.  This keeps CPU-only development environments useful while retaining a
real numerical and event-ordering check for CI or a cluster node.
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
except Exception:  # pragma: no cover - exercised only on a CPU-only host.
    torch = None
    dist = None


_RUN_SMOKE = (
    torch is not None
    and dist is not None
    and torch.cuda.is_available()
    and os.environ.get("TIDE_RUN_CUDA_SMOKE") == "1"
    and int(os.environ.get("WORLD_SIZE", "1")) == 2
)


@unittest.skipUnless(
    _RUN_SMOKE,
    "set TIDE_RUN_CUDA_SMOKE=1 and launch this test with torchrun on two GPUs",
)
class DistributedPipelineCudaSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rank = int(os.environ["RANK"])
        cls.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(cls.local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(seconds=60),
            )
        submodule = Path(__file__).parents[1] / "submodules" / "gsplat"
        if str(submodule) not in sys.path:
            sys.path.insert(0, str(submodule))
        try:
            import gsplat  # noqa: F401
            from gsplat.distributed_pipeline import (  # noqa: F401
                prepare_batch_camera_layout,
                distributed_rasterization_pipeline,
            )
        except Exception as error:
            raise unittest.SkipTest(f"pinned gsplat CUDA extension unavailable: {error}")

    def _inputs(self):
        torch.manual_seed(17)
        device = torch.device("cuda", self.local_rank)
        count = 32
        means = torch.randn((count, 3), device=device) * 0.15
        means[:, 2] = means[:, 2].abs() + 2.0
        means.requires_grad_()
        quats = torch.zeros((count, 4), device=device)
        quats[:, 0] = 1.0
        quats.requires_grad_()
        scales = torch.full((count, 3), 0.08, device=device, requires_grad=True)
        opacities = torch.full((count,), 0.5, device=device, requires_grad=True)
        colors = torch.randn((count, 16, 3), device=device) * 0.05
        colors.requires_grad_()
        viewmats = torch.eye(4, device=device).reshape(1, 4, 4).repeat(2, 1, 1)
        Ks = torch.tensor(
            [[6.0, 0.0, 4.0], [0.0, 6.0, 4.0], [0.0, 0.0, 1.0]],
            device=device,
        ).reshape(1, 3, 3).repeat(2, 1, 1)
        targets = torch.full((2, 8, 8, 3), 0.5, device=device)
        return means, quats, scales, opacities, colors, viewmats, Ks, targets

    @staticmethod
    def _loss(images, targets):
        return ((images - targets) ** 2).mean()

    def _baseline(self, values, background=None):
        import gsplat

        means, quats, scales, opacities, colors, viewmats, Ks, targets = values
        outputs = []
        total = means.new_zeros(())
        for index in range(2):
            rendered, _, _ = gsplat.rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmats[index:index + 1],
                Ks=Ks[index:index + 1],
                width=8,
                height=8,
                sh_degree=0,
                packed=True,
                distributed=True,
                backgrounds=None if background is None else background.reshape(1, 3),
            )
            outputs.append(rendered.detach())
            total = total + self._loss(rendered, targets[index:index + 1])
        total.backward()
        gradients = tuple(
            tensor.grad.detach().clone()
            for tensor in (means, quats, scales, opacities, colors)
        )
        return torch.cat(outputs, dim=0), total.detach(), gradients

    def _pc1(self, values, background=None):
        from gsplat.distributed_pipeline import (
            distributed_rasterization_pipeline,
            prepare_batch_camera_layout,
        )

        means, quats, scales, opacities, colors, viewmats, Ks, targets = values
        layout = prepare_batch_camera_layout(
            means=means, viewmats=viewmats, Ks=Ks, microbatch=1
        )
        compute_stream = torch.cuda.current_stream()
        pipeline = distributed_rasterization_pipeline(
            compute_stream=compute_stream,
            comm_stream=torch.cuda.Stream(priority=-1),
            buffer_count=2,
        )
        current = None
        snapshots = []
        outputs = []
        total = means.new_zeros(())
        for index in range(2):
            if current is None:
                current = pipeline.submit_projection(
                    microbatch_index=index,
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmats[index:index + 1],
                    Ks=Ks[index:index + 1],
                    viewmats_global=layout.microbatch_cameras(index)[0],
                    Ks_global=layout.microbatch_cameras(index)[1],
                    width=8,
                    height=8,
                    sh_degree=0,
                    backgrounds=None if background is None else background.reshape(1, 3),
                    near_plane=0.01,
                    far_plane=1e10,
                    radius_clip=0.0,
                    eps2d=0.3,
                    tile_size=16,
                    rasterize_mode="classic",
                    N_world=layout.N_world,
                    C_world=layout.microbatch_C_world,
                )
            current.start_forward_exchange()
            next_handle = None
            if index == 0:
                next_handle = pipeline.submit_projection(
                    microbatch_index=1,
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmats[1:2],
                    Ks=Ks[1:2],
                    viewmats_global=layout.microbatch_cameras(1)[0],
                    Ks_global=layout.microbatch_cameras(1)[1],
                    width=8,
                    height=8,
                    sh_degree=0,
                    backgrounds=None if background is None else background.reshape(1, 3),
                    near_plane=0.01,
                    far_plane=1e10,
                    radius_clip=0.0,
                    eps2d=0.3,
                    tile_size=16,
                    rasterize_mode="classic",
                    N_world=layout.N_world,
                    C_world=layout.microbatch_C_world,
                )
            rendered, _, _ = current.wait_and_render()
            outputs.append(rendered.detach())
            total = total + self._loss(rendered, targets[index:index + 1])
            snapshots.append(current.snapshot())
            # The backward event is intentionally part of the test: slot reuse
            # must not race autograd writes.
            total_piece = self._loss(rendered, targets[index:index + 1])
            total_piece.backward()
            backward_done = torch.cuda.Event(enable_timing=True)
            backward_done.record(compute_stream)
            current.mark_backward_done(backward_done)
            current = next_handle
        pipeline.drain()
        pipeline.close()
        gradients = tuple(
            tensor.grad.detach().clone()
            for tensor in (means, quats, scales, opacities, colors)
        )
        return torch.cat(outputs, dim=0), total.detach(), gradients, snapshots

    def test_off_and_pc1_match_and_overlap_events_exist(self):
        for label, color in (
            ("default", None),
            ("white", (1.0, 1.0, 1.0)),
            ("custom", (0.1, 0.2, 0.3)),
        ):
            with self.subTest(background=label):
                background = None if color is None else torch.tensor(
                    color, device=torch.device("cuda", self.local_rank)
                )
                self._assert_off_and_pc1_match(background, label)

    def _assert_off_and_pc1_match(self, background, label):
        baseline_values = self._inputs()
        baseline_images, baseline_loss, baseline_gradients = self._baseline(
            baseline_values, background
        )
        pc1_values = tuple(
            value.detach().clone().requires_grad_(True)
            if torch.is_tensor(value) and value.is_floating_point()
            and value.requires_grad
            else value.detach().clone() if torch.is_tensor(value) else value
            for value in self._inputs()
        )
        pc1_images, pc1_loss, pc1_gradients, snapshots = self._pc1(pc1_values, background)
        torch.testing.assert_close(pc1_images, baseline_images, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(pc1_loss, baseline_loss, rtol=1e-4, atol=1e-5)
        for pc1_gradient, baseline_gradient in zip(pc1_gradients, baseline_gradients):
            torch.testing.assert_close(pc1_gradient, baseline_gradient, rtol=1e-4, atol=1e-5)
        self.assertEqual(len(snapshots), 2)
        for snapshot in snapshots:
            for name in ("projection_start", "projection_end", "exchange_done", "render_done"):
                self.assertIsNotNone(snapshot[name])
        c1_start = snapshots[0]["metadata_start"]
        c1_end = snapshots[0]["payload_end"]
        next_start = snapshots[1]["projection_start"]
        next_end = snapshots[1]["projection_end"]
        self.assertIsNotNone(c1_start)
        self.assertIsNotNone(c1_end)
        self.assertIsNotNone(next_start)
        self.assertIsNotNone(next_end)
        c1_ms = float(c1_start.elapsed_time(c1_end))
        next_start_ms = float(c1_start.elapsed_time(next_start))
        next_end_ms = float(c1_start.elapsed_time(next_end))
        overlap_ms = max(0.0, min(c1_ms, next_end_ms) - max(0.0, next_start_ms))
        print(f"rank={self.rank} background={label} measured P(1)/C1(0) overlap_ms={overlap_ms:.3f}")
        if os.environ.get("TIDE_REQUIRE_PC1_OVERLAP") == "1":
            self.assertGreater(overlap_ms, 0.0)

    def test_error_group_exits_after_c1_submission(self):
        """Inject rank-0 failure after C1 submission without a NCCL barrier."""

        if os.environ.get("TIDE_RUN_CUDA_ABORT_TEST") != "1":
            self.skipTest("set TIDE_RUN_CUDA_ABORT_TEST=1 to run the abort smoke test")
        from gsplat.distributed_pipeline import (
            distributed_rasterization_pipeline,
            prepare_batch_camera_layout,
        )

        values = self._inputs()
        means, quats, scales, opacities, colors, viewmats, Ks, _ = values
        layout = prepare_batch_camera_layout(
            means=means, viewmats=viewmats, Ks=Ks, microbatch=1
        )
        pipeline = distributed_rasterization_pipeline(
            compute_stream=torch.cuda.current_stream(),
            comm_stream=torch.cuda.Stream(priority=-1),
            buffer_count=2,
        )
        handle = pipeline.submit_projection(
            microbatch_index=0,
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats[:1],
            Ks=Ks[:1],
            viewmats_global=layout.microbatch_cameras(0)[0],
            Ks_global=layout.microbatch_cameras(0)[1],
            width=8,
            height=8,
            sh_degree=0,
            backgrounds=torch.zeros((1, 3), device=means.device),
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            eps2d=0.3,
            tile_size=16,
            rasterize_mode="classic",
            N_world=layout.N_world,
            C_world=layout.microbatch_C_world,
        )
        handle.start_forward_exchange()
        local_error = None
        try:
            if self.rank == 0:
                raise RuntimeError("injected rank-0 failure after C1 submit")
        except BaseException as error:
            local_error = f"{type(error).__name__}: {error}"
        # Abort only drops Python handles.  It intentionally does not call a
        # default-group barrier or synchronize the NCCL stream on this path.
        pipeline.abort()
        error_group = dist.new_group(backend="gloo")
        statuses = [None, None]
        dist.all_gather_object(
            statuses,
            {"rank": self.rank, "error": local_error},
            group=error_group,
        )
        self.assertTrue(any(status["error"] for status in statuses))
        self.assertEqual(statuses[0]["rank"], 0)
        dist.destroy_process_group(error_group)


if __name__ == "__main__":
    unittest.main()
