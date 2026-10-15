"""Small, explicit distributed runtime wrapper for TideGS training."""

from __future__ import annotations

import os
import math
import json
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, List

import torch
import torch.distributed as dist
from utils.tide_trace import tide_range
from utils.distributed_error_control import PC1ErrorController


_DISTRIBUTED_MODES = {"off", "gaussian_sharded"}


def _positive_timeout(name: str, default: str) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def distributed_mode(args: Any) -> str:
    return str(getattr(args, "tide_distributed_mode", "off")).lower()


def distributed_requested(args: Any) -> bool:
    return distributed_mode(args) != "off"


@dataclass(frozen=True)
class DistributedContext:
    mode: str = "off"
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    error_group: Any = field(default=None, repr=False, compare=False)
    pc1_error_controller: Any = field(default=None, repr=False, compare=False)
    pipeline_comm_worker: Any = field(default=None, repr=False, compare=False)
    pc1_native_runtime: Any = field(default=None, repr=False, compare=False)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0

    @classmethod
    def initialize(cls, args: Any) -> "DistributedContext":
        mode = distributed_mode(args)
        if mode not in _DISTRIBUTED_MODES:
            raise ValueError(
                f"Invalid tide_distributed_mode={mode!r}; expected one of "
                f"{sorted(_DISTRIBUTED_MODES)}"
            )
        if mode == "off":
            if int(os.environ.get("WORLD_SIZE", "1")) > 1:
                raise RuntimeError(
                    "torchrun detected but tide_distributed_mode=off; enable "
                    "gaussian_sharded mode or launch a single process"
                )
            return cls()

        missing = [name for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if name not in os.environ]
        if missing:
            raise RuntimeError(
                "Distributed TideGS must be launched with torchrun; missing environment "
                f"variables: {', '.join(missing)}"
            )

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size <= 1:
            raise RuntimeError("gaussian_sharded mode requires WORLD_SIZE > 1")
        if not torch.cuda.is_available():
            raise RuntimeError("gaussian_sharded mode requires CUDA")

        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            # NCCL's built-in collective timeout is 10 minutes, and a compaction burst can
            # legitimately exceed it: the compaction barrier is an all-gather that the other
            # ranks wait on while one rank streams tens of GiB to the SSD.  On 2026-09-25 that
            # aborted a 200K run at iteration ~70k with
            # "Watchdog caught collective operation timeout: ... OpType=ALLGATHER ...
            #  Timeout(ms)=600000".  Raise it so a long maintenance burst cannot kill the job;
            # TIDE_PG_TIMEOUT_SEC overrides without another code edit.
            pg_timeout = timedelta(
                seconds=int(os.environ.get("TIDE_PG_TIMEOUT_SEC", 3600))
            )
            dist.init_process_group(
                backend="nccl", init_method="env://", timeout=pg_timeout
            )
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise RuntimeError(
                "torch.distributed process-group metadata does not match torchrun"
            )

        pc1 = str(getattr(args, "tide_forward_pipeline", "off")).lower() == "pc1"
        # Gloo error checkpoints alone cannot interrupt a main thread blocked
        # in CUDA/NCCL. PC1 additionally needs independent fatal supervision.
        if pc1:
            setup_timeout = timedelta(seconds=_positive_timeout("TIDE_PC1_ERROR_TIMEOUT_SEC", "30"))
            io_timeout = timedelta(seconds=_positive_timeout("TIDE_PC1_CONTROL_IO_TIMEOUT_SEC", "1"))
            stage_timeout = _positive_timeout("TIDE_PC1_STAGE_TIMEOUT_SEC", "60")
            # A healthy peer can still be completing a protected host stage
            # when another rank reaches its CPU checkpoint. Gloo must not
            # time out earlier than the controller's permitted stage skew.
            runtime_timeout = max(setup_timeout, timedelta(
                seconds=stage_timeout + 2 * io_timeout.total_seconds() + 5,
            ))
            error_group = dist.new_group(
                backend="gloo", timeout=runtime_timeout,
            )
        else:
            error_group = dist.new_group(backend="gloo")
        pc1_error_controller = None
        if pc1:
            control_store = None
            control_port = [None]
            if rank == 0:
                control_store = dist.TCPStore(
                    host_name="0.0.0.0", port=0, world_size=world_size,
                    is_master=True, timeout=setup_timeout, wait_for_workers=False,
                )
                control_port[0] = control_store.port
            # This CPU startup exchange allocates a unique port per job. The
            # watchdog never runs collectives on this (or the NCCL) group.
            dist.broadcast_object_list(control_port, src=0, group=error_group)
            if rank != 0:
                control_store = dist.TCPStore(
                    host_name=os.environ["MASTER_ADDR"], port=int(control_port[0]),
                    world_size=world_size, is_master=False,
                    timeout=setup_timeout, wait_for_workers=False,
                )
            control_store.set_timeout(io_timeout)
            pc1_error_controller = PC1ErrorController(
                store=control_store, rank=rank, world_size=world_size,
                stage_timeout=stage_timeout,
            )

        args.gpu = local_rank
        context = cls(
            mode=mode,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            error_group=error_group,
            pc1_error_controller=pc1_error_controller,
        )
        setattr(args, "_tide_distributed_context", context)
        return context

    def barrier(self) -> None:
        if self.enabled:
            with tide_range("tide.sync.barrier"):
                dist.barrier()

    def broadcast_object(self, value: Any, src: int = 0) -> Any:
        if not self.enabled:
            return value
        payload = [value if self.rank == src else None]
        with tide_range("tide.comm.object_collective"):
            dist.broadcast_object_list(payload, src=src, device=torch.device("cuda", self.local_rank))
        return payload[0]

    def all_gather_object(self, value: Any, *, group=None) -> List[Any]:
        if not self.enabled:
            return [value]
        gathered: List[Any] = [None for _ in range(self.world_size)]
        with tide_range("tide.comm.object_collective"):
            dist.all_gather_object(gathered, value, group=group)
        return gathered

    def ensure_pipeline_comm_worker(
        self, *, buffer_count: int = 2, comm_stream=None,
        require_tile_mask: bool = False,
    ):
        """Load native capabilities on the owner thread, then reuse one worker.

        The frozen context owns the runtime for its entire rank lifetime; batch
        pipelines borrow it and must not shut it down between batches.
        """
        if not self.enabled or self.pc1_error_controller is None:
            raise RuntimeError("PC1 communication requires an initialized error controller")
        buffer_count = int(buffer_count)
        if buffer_count < 2:
            raise ValueError("PC1 buffer_count must be at least 2")
        worker = self.pipeline_comm_worker
        if worker is not None:
            if worker.buffer_count != buffer_count:
                raise RuntimeError("PC1 buffer_count cannot change during a rank lifetime")
            return worker

        from utils.pc1_native_capabilities import check_pc1_native_capabilities
        from gsplat.distributed_pipeline import PipelineCommWorker

        # Loading/JIT compilation here is intentional: the communication thread
        # must never be the first caller of a CUDA extension.
        with self.pc1_error_controller.operation("PC1 native initialization and worker startup"):
            runtime = check_pc1_native_capabilities(require_tile_mask=require_tile_mask)
            print(
                f"[PC1 native rank={self.rank}] " + json.dumps(runtime, sort_keys=True),
                flush=True,
            )
            worker = PipelineCommWorker(
                device=torch.device("cuda", self.local_rank),
                comm_stream=comm_stream,
                buffer_count=buffer_count,
                default_timeout_s=self.pc1_error_controller.stage_timeout,
                on_failure=lambda error: self.pc1_error_controller.fail(
                    "PC1 communication worker", error,
                ),
            )
        object.__setattr__(self, "pc1_native_runtime", runtime)
        object.__setattr__(self, "pipeline_comm_worker", worker)
        return worker

    def close(self) -> None:
        worker = self.pipeline_comm_worker
        if worker is not None:
            controller = self.pc1_error_controller
            deadline = time.monotonic() + controller.stage_timeout
            # Keep fatal supervision alive while queue, GPU events and worker
            # shutdown are still capable of blocking.
            with controller.operation("PC1 communication worker shutdown"):
                worker.drain(deadline)
                worker.close(deadline)
        if self.enabled and dist.is_initialized():
            if self.pc1_error_controller is not None:
                self.pc1_error_controller.close()
                # Rank 0 owns the TCPStore server. All watchdogs must stop
                # before that server is released during healthy shutdown.
                dist.barrier(group=self.error_group)
                self.pc1_error_controller.release_store()
            if self.error_group is not None:
                dist.destroy_process_group(self.error_group)
            dist.destroy_process_group()


def get_distributed_context(args: Any) -> DistributedContext:
    context = getattr(args, "_tide_distributed_context", None)
    return context if context is not None else DistributedContext()
