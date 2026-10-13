"""Disposable real CUDA worker, launched only by the fatal smoke parent.

Test-only wrappers inject failures at real metadata/payload boundaries. No
CUDA/NCCL cleanup is attempted after abort; the fatal callback still calls the
production os._exit(86), adding only a small CPU evidence file beforehand.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def _write_record(path, value):
    # Files belong to one rank each. A killed/interrupted write fails the
    # parent JSON check instead of silently reporting a passing fatal exit.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, json.dumps(value).encode("utf-8"))
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=("before_c1_split", "mid_payload_collective"))
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--stage-timeout", type=float, default=4.0)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    from utils.distributed import DistributedContext
    from utils.distributed_error_control import PC1_FATAL_EXIT_CODE, _exit_worker
    from tests.test_distributed_pipeline_cuda import cuda_smoke_inputs, submit_smoke_projection

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ["TIDE_PG_TIMEOUT_SEC"] = "60"
    os.environ["TIDE_PC1_ERROR_TIMEOUT_SEC"] = "30"
    # Setup/first extension use can be slow; arm the short stage timeout only
    # after the real camera-layout/projection setup and healthy synchronization.
    os.environ["TIDE_PC1_STAGE_TIMEOUT_SEC"] = "60"
    os.environ["TIDE_PC1_CONTROL_IO_TIMEOUT_SEC"] = "0.25"
    # Ensure the pre-C1 case reaches the CUDA split materialization rather
    # than waiting inside Work.wait due to an inherited NCCL debug setting.
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    context = DistributedContext.initialize(SimpleNamespace(
        tide_distributed_mode="gaussian_sharded", tide_forward_pipeline="pc1",
    ))
    controller = context.pc1_error_controller
    controller.poll_interval = 0.01

    def fatal_exit(reason):
        _write_record(args.result_dir / f"rank-{rank}-fatal.json", {
            "rank": rank, "exit_code": PC1_FATAL_EXIT_CODE,
            "time": time.monotonic(), "reason": reason,
        })
        _exit_worker(reason)

    # Keep the actual context/controller/watchdog path; instrument only the
    # final public callback, so a launcher-issued SIGTERM cannot pass this test.
    controller._exit_callback = fatal_exit
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submodules" / "gsplat"))
    import gsplat.distributed_pipeline as pc1

    values = cuda_smoke_inputs(rank, local_rank, camera_count=1)
    layout = pc1.prepare_batch_camera_layout(
        means=values[0], viewmats=values[5], Ks=values[6], microbatch=1,
    )
    pipeline = pc1.distributed_rasterization_pipeline(
        compute_stream=torch.cuda.current_stream(),
        comm_stream=torch.cuda.Stream(priority=-1), buffer_count=2,
    )
    handle = submit_smoke_projection(pipeline, layout, values, 0)
    torch.cuda.synchronize()
    dist.barrier(group=context.error_group)
    controller.stage_timeout = args.stage_timeout
    control_store = controller.store

    def peer_wait(phase):
        _write_record(args.result_dir / "peer_wait.json", {
            "rank": rank, "phase": phase, "time": time.monotonic(),
        })
        control_store.set("test-peer-wait", phase)

    def inject_failure():
        # Wait on CPU evidence only, never a damaged NCCL group. Store wait has
        # its own explicit timeout shorter than the controller stage deadline.
        control_store.wait(["test-peer-wait"], timedelta(seconds=args.stage_timeout / 2))
        # Give the peer time to enter the following real CUDA wait. The wait
        # under test is the unmatched NCCL work, never this scheduling delay.
        time.sleep(min(0.1, args.stage_timeout / 10))
        _write_record(args.result_dir / "fault.json", {
            "scenario": args.scenario, "rank": rank, "time": time.monotonic(),
        })
        pipeline.abort()
        # Abort must quarantine live tensors, and neither cleanup method may
        # falsely claim successful drainage of this partially submitted C1.
        assert any(retained is handle for retained in pipeline._quarantine)
        for cleanup in (pipeline.drain, pipeline.close):
            try:
                cleanup()
            except RuntimeError as error:
                assert "aborted" in str(error)
            else:
                raise AssertionError("aborted PC1 cleanup unexpectedly succeeded")
        controller.fail(args.scenario, RuntimeError("injected rank-0 C1 failure"))

    original_metadata = pc1.all_to_all_int32
    original_payload = pc1.all_to_all_tensor_list
    payload_calls = 0

    def metadata(world_size, splits, device=None):
        collected = original_metadata(world_size, splits, device=device)
        if args.scenario == "before_c1_split" and rank == 1:
            peer_wait("cuda_split_read")
            # Real pending NCCL metadata, not a sleep or CPU mock. Rank 0
            # never enters C1, so materializing this CUDA split must wait.
            collected[0].item()
            raise AssertionError("missing-peer CUDA split read unexpectedly completed")
        # For the payload case, prove both ranks completed the metadata before
        # either can inject failure at the second real payload exchange.
        for split in collected:
            if isinstance(split, torch.Tensor):
                split.item()
        return collected

    def payload(*positional, **keywords):
        nonlocal payload_calls
        payload_calls += 1
        if args.scenario == "mid_payload_collective" and payload_calls == 2:
            if rank == 0:
                inject_failure()
            peer_wait("payload_collective")
            exchanged = original_payload(*positional, **keywords)
            # On rank 1 the actual second payload collective has no matching
            # rank-0 submission. This public CUDA wait must be interrupted by
            # CPU supervision, not NCCL's much longer built-in timeout.
            torch.cuda.current_stream().synchronize()
            raise AssertionError("missing-peer payload collective unexpectedly completed")
        exchanged = original_payload(*positional, **keywords)
        if args.scenario == "mid_payload_collective" and payload_calls == 1:
            torch.cuda.current_stream().synchronize()
        return exchanged

    pc1.all_to_all_int32 = metadata
    pc1.all_to_all_tensor_list = payload
    with controller.operation(f"smoke-{args.scenario}"):
        if rank == 0 and args.scenario == "before_c1_split":
            inject_failure()
        handle.start_forward_exchange()
        raise AssertionError("fatal fault worker unexpectedly returned from C1")


if __name__ == "__main__":
    main()
