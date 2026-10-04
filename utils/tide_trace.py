"""Shared NVTX ranges for TideGS pipeline profiling."""

from contextlib import contextmanager

import torch


@contextmanager
def tide_range(name):
    """Emit one balanced NVTX range without synchronizing CUDA."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
