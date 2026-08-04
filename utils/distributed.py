"""Small, explicit distributed runtime wrapper for TideGS training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List

import torch
import torch.distributed as dist


_DISTRIBUTED_MODES = {"off", "gaussian_sharded"}


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
            dist.init_process_group(backend="nccl", init_method="env://")
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise RuntimeError("torch.distributed process-group metadata does not match torchrun")

        args.gpu = local_rank
        context = cls(mode=mode, rank=rank, local_rank=local_rank, world_size=world_size)
        setattr(args, "_tide_distributed_context", context)
        return context

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def broadcast_object(self, value: Any, src: int = 0) -> Any:
        if not self.enabled:
            return value
        payload = [value if self.rank == src else None]
        dist.broadcast_object_list(payload, src=src, device=torch.device("cuda", self.local_rank))
        return payload[0]

    def all_gather_object(self, value: Any) -> List[Any]:
        if not self.enabled:
            return [value]
        gathered: List[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, value)
        return gathered

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def get_distributed_context(args: Any) -> DistributedContext:
    context = getattr(args, "_tide_distributed_context", None)
    return context if context is not None else DistributedContext()
