"""Disposable real CUDA rank for PC1 Future/worker failure tests.

Only test wrappers inject errors. All transport is the actual explicit C1/C2
NCCL path, supervised by the production CPU controller. No damaged CUDA/NCCL
cleanup is attempted. The parent checks each process's actual exit code 86.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

from tests.pc1_fault_evidence import SCENARIOS


def _write_record(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        data = memoryview(json.dumps(value).encode("utf-8"))
        while data:
            data = data[os.write(descriptor, data):]
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=tuple(SCENARIOS))
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--stage-timeout", type=float, default=4.0)
    args = parser.parse_args()
    if args.stage_timeout <= 0:
        parser.error("stage-timeout must be positive")

    import torch
    import torch.distributed as dist
    from utils.distributed import DistributedContext
    from utils.distributed_error_control import PC1_FATAL_EXIT_CODE, _exit_worker
    from tests.test_distributed_pipeline_cuda import cuda_smoke_inputs, submit_smoke_projection

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ["TIDE_PG_TIMEOUT_SEC"] = "60"
    os.environ["TIDE_PC1_ERROR_TIMEOUT_SEC"] = "30"
    os.environ["TIDE_PC1_STAGE_TIMEOUT_SEC"] = "60"
    os.environ["TIDE_PC1_CONTROL_IO_TIMEOUT_SEC"] = "0.25"
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
            "native_thread_id": threading.get_native_id(),
        })
        _exit_worker(reason)

    controller._exit_callback = fatal_exit
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submodules" / "gsplat"))
    import gsplat.distributed_pipeline as pc1

    # Native load/ABI checking and real projection setup precede the short
    # measured deadline. The rank reuses exactly one context-owned worker.
    worker = context.ensure_pipeline_comm_worker(buffer_count=2)
    values = cuda_smoke_inputs(rank, local_rank, camera_count=1)
    layout = pc1.prepare_batch_camera_layout(
        means=values[0], viewmats=values[5], Ks=values[6], microbatch=1,
    )
    pipeline = pc1.distributed_rasterization_pipeline(
        compute_stream=torch.cuda.current_stream(), comm_worker=worker,
        buffer_count=2, batch_index=0,
    )
    handle = submit_smoke_projection(pipeline, layout, values, 0)
    torch.cuda.synchronize()
    dist.barrier(group=context.error_group)
    controller.stage_timeout = args.stage_timeout
    pipeline.deadline = time.monotonic() + args.stage_timeout
    control_store = controller.store
    worker_stage = threading.local()
    close_entered = threading.Event()

    def store_wait(keys):
        control_store.wait(keys, timedelta(seconds=args.stage_timeout / 2))

    def peer_wait(*, gpu_work=True, **details):
        _write_record(args.result_dir / "peer_wait.json", {
            "rank": rank, "phase": SCENARIOS[args.scenario],
            "time": time.monotonic(), "gpu_work_submitted": gpu_work,
            "native_thread_id": threading.get_native_id(),
            **details,
        })
        control_store.set("test-peer-wait", SCENARIOS[args.scenario])

    def record_fault():
        keys = ["test-peer-wait"]
        if args.scenario == "shutdown_pending":
            keys += [f"test-shutdown-rank-{value}" for value in range(2)]
        store_wait(keys)
        # Peer has already submitted unmatched NCCL/GPU work. This short delay
        # only lets it reach the public CUDA wait that the watchdog interrupts.
        time.sleep(min(0.1, args.stage_timeout / 10))
        _write_record(args.result_dir / "fault.json", {
            "scenario": args.scenario, "rank": rank, "time": time.monotonic(),
            "native_thread_id": threading.get_native_id(),
        })

    def inject_failure():
        record_fault()
        # Raising on the comm thread exercises its real on_failure callback;
        # raising on the owner thread exercises controller.operation(). Neither
        # test wrapper mutates the owner's graph/active-slot dictionaries.
        raise RuntimeError(f"injected rank-0 {args.scenario} failure")

    original_c1 = worker._c1
    original_c2 = worker._c2

    def c1(request):
        worker_stage.stage = "C1"
        worker_stage.payload_index = 0
        if args.scenario == "shutdown_pending" and rank == 0:
            if not close_entered.wait(args.stage_timeout / 2):
                raise TimeoutError("shutdown did not begin with pending C1")
            inject_failure()
        return original_c1(request)

    def c2(request):
        worker_stage.stage = "C2"
        worker_stage.payload_index = 0
        return original_c2(request)

    worker._c1, worker._c2 = c1, c2
    original_collective = dist.all_to_all_single

    def collective(output, data, *positional, **keywords):
        metadata_target = (
            getattr(worker_stage, "stage", None) == "C1" and data.dim() == 1
            and args.scenario in ("before_c1", "c1_metadata", "shutdown_pending")
        )
        if metadata_target and rank == 0:
            inject_failure()
        work = original_collective(output, data, *positional, **keywords)
        if metadata_target and rank == 1:
            # A real async Work exists before recording evidence; retain it in
            # this blocked frame until the production controller exits the rank.
            work.wait()
            peer_wait()
            torch.cuda.current_stream().synchronize()
            raise AssertionError("unmatched C1 metadata unexpectedly completed")
        return work

    dist.all_to_all_single = collective
    original_payload = pc1._raw_exchange

    def payload(data, send_splits, recv_splits, retained):
        worker_stage.payload_index += 1
        stage = getattr(worker_stage, "stage", None)
        target = ((stage == "C1" and args.scenario == "c1_payload"
                   and worker_stage.payload_index == 2)
                  or (stage == "C2" and args.scenario in ("before_c2", "c2_payload")))
        if target and rank == 0:
            inject_failure()
        output = original_payload(data, send_splits, recv_splits, retained)
        if target and rank == 1:
            peer_wait()
            torch.cuda.current_stream().synchronize()
            raise AssertionError("unmatched payload unexpectedly completed")
        return output

    pc1._raw_exchange = payload
    original_enqueue_c2 = worker.enqueue_c2

    def enqueue_c2(request):
        if args.scenario == "before_c2" and rank == 0:
            assert handle.intermediate_done is not None
            inject_failure()
        return original_enqueue_c2(request)

    worker.enqueue_c2 = enqueue_c2
    original_backward = torch.autograd.backward

    def backward(*positional, **keywords):
        if args.scenario != "final_gradient":
            return original_backward(*positional, **keywords)
        if rank == 0:
            inject_failure()
        # We are inside final_backward's G stage, before invoking its actual
        # autograd engine. Observe real pending GPU work before any first-use
        # backward allocation can wait on the delayed stream. This marker does
        # not claim that a native Gaussian backward kernel has been entered.
        torch.cuda._sleep(int(args.stage_timeout * 2_000_000_000))
        pending = torch.cuda.Event()
        pending.record(torch.cuda.current_stream())
        if pending.query():
            raise AssertionError("G-stage delay completed before its entry marker")
        peer_wait(gpu_pending_at_marker=True, native_backward_entered=False)
        original_backward(*positional, **keywords)
        torch.cuda.current_stream().synchronize()
        raise AssertionError("G-stage fault did not interrupt peer GPU wait")

    torch.autograd.backward = backward
    original_drain = worker.drain

    def drain(deadline=None):
        if args.scenario == "shutdown_pending":
            _write_record(args.result_dir / f"rank-{rank}-shutdown.json", {
                "rank": rank, "time": time.monotonic(),
                "pending_future": not handle.c1_future.done(),
            })
            control_store.set(f"test-shutdown-rank-{rank}", "entered")
            close_entered.set()
        return original_drain(deadline)

    worker.drain = drain

    class BrokenControlStore:
        def check(self, keys):
            if control_store.check(["test-break-control"]):
                raise ConnectionError("injected PC1 control channel failure")
            return control_store.check(keys)

        def get(self, key):
            return control_store.get(key)

        def set(self, key, value):
            return control_store.set(key, value)

    if args.scenario == "control_channel":
        controller.store = BrokenControlStore()

    with controller.operation(f"fault-{args.scenario}"):
        if args.scenario == "before_c1" and rank == 0:
            inject_failure()
        handle.start_forward_exchange(deadline=pipeline.deadline)
        if args.scenario == "shutdown_pending":
            context.close()
            raise AssertionError("shutdown with pending task unexpectedly succeeded")
        images, _, _ = handle.wait_and_render(deadline=pipeline.deadline)
        if args.scenario == "control_channel":
            if rank == 1:
                peer_wait(gpu_work=False)
                store_wait(["test-break-control"])
            else:
                record_fault()
                control_store.set("test-break-control", "disconnected")
            # The actual watchdog's Store exception, not an explicit fail(),
            # must trigger each rank's production exit path.
            threading.Event().wait(args.stage_timeout * 2)
            raise AssertionError("control-channel failure was not detected")
        loss = ((images - values[7][:1]) ** 2).mean()
        handle.intermediate_backward(loss, deadline=pipeline.deadline)
        handle.final_backward(deadline=pipeline.deadline)
        raise AssertionError("fatal fault worker unexpectedly completed its window")


if __name__ == "__main__":
    main()
