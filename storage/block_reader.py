"""BlockReader abstraction for paper-mode out-of-core materialization.

The out-of-core training loop needs to fetch per-block Gaussian parameters
from one of several possible sources:

  1. ``UnifiedParamsBlockReader``
     Slices ``_unified_params`` (a pinned-CPU (N, 59) tensor) directly.
     Fastest when the whole parameter table already fits in RAM.
     Column layout:  xyz(3) | opacity(1) | scaling(3) | rotation(4)
                     | features_dc(3) | features_rest(45)

  2. ``TieredCacheBlockReader``
     Routes reads through ``TieredCacheManager``, which serves hits from
     an in-RAM LRU cache and streams misses from the on-SSD log segments.
     This is the paper-correct path — VRAM, RAM, and SSD are the three
     explicit tiers, and the full parameter table never needs to be
     RAM-resident.
     Column layout:  xyz(3) | scaling(3) | rotation(4) | opacity(1)
                     | features_dc(3) | features_rest(45)

Both readers return 2-D tensors shaped (rows, 59).  They also advertise
their column layout via the ``layout`` property so callers can parse the
block data without hard-coding a source-specific offset table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import torch


class BlockLayout(Enum):
    """Column order of a 59-wide block tensor.

    UNIFIED corresponds to the historical ``_unified_params`` layout.
    CACHE corresponds to the SSD base file layout used by
    ``LogStorageManager`` / ``TieredCacheManager``.
    """

    UNIFIED = "unified"  # xyz | opacity | scaling | rotation | dc | rest
    CACHE = "cache"      # xyz | scaling  | rotation | opacity  | dc | rest


@dataclass(frozen=True)
class BlockBatch:
    tensor: torch.Tensor
    block_ids: tuple[int, ...]
    block_slices: Dict[int, slice]
    layout: BlockLayout


def _pack_block_batch(
    blocks: Dict[int, torch.Tensor],
    block_ids: Sequence[int],
    layout: BlockLayout,
    out: Optional[torch.Tensor] = None,
) -> BlockBatch:
    ordered_ids = tuple(int(block_id) for block_id in block_ids)
    missing = [block_id for block_id in ordered_ids if block_id not in blocks]
    if missing:
        raise KeyError(f"BlockReader omitted blocks; first missing={missing[:8]}")

    pieces = [blocks[block_id] for block_id in ordered_ids]
    total_rows = sum(int(piece.shape[0]) for piece in pieces)
    if out is None:
        out = torch.empty((total_rows, 59), dtype=torch.float32)
    elif out.device.type != "cpu" or out.dim() != 2 or tuple(out.shape) != (total_rows, 59):
        raise ValueError(
            f"BlockBatch output must be CPU ({total_rows}, 59); got {tuple(out.shape)} "
            f"on {out.device}"
        )

    if len(pieces) == 1:
        out.copy_(pieces[0])
    elif pieces:
        torch.cat(pieces, dim=0, out=out)

    block_slices = {}
    offset = 0
    for block_id, piece in zip(ordered_ids, pieces):
        next_offset = offset + int(piece.shape[0])
        block_slices[block_id] = slice(offset, next_offset)
        offset = next_offset
    return BlockBatch(out, ordered_ids, block_slices, layout)


@runtime_checkable
class BlockReader(Protocol):
    """Protocol for block-level parameter readers.

    Implementations must be thread-safe enough that a single training
    thread (plus optional prefetch thread) can call ``read_blocks`` and
    ``hint_future`` concurrently.  The returned tensors are CPU tensors
    whose rows are indexed by *within-block* offset (0 .. block_size-1).
    """

    total_gaussians: int
    block_size: int
    num_blocks: int
    layout: BlockLayout

    def read_blocks(self, block_ids: List[int]) -> Dict[int, torch.Tensor]:
        ...

    def read_batch(
        self,
        block_ids: List[int],
        out: Optional[torch.Tensor] = None,
    ) -> BlockBatch:
        ...

    def hint_future(
        self,
        block_ids: List[int],
        *,
        target_iteration: Optional[int] = None,
    ) -> int:
        ...

    def contains_block(self, block_id: int) -> bool:
        ...


class UnifiedParamsBlockReader:
    """Serve block reads by slicing a fully-resident ``_unified_params``.

    This preserves the existing ``fast_ram`` behaviour exactly: a block
    is a (block_size, 59) view into the global CPU pinned tensor, so
    reads are zero-copy.
    """

    layout = BlockLayout.UNIFIED

    def __init__(self, unified_params: torch.Tensor, block_size: int):
        if unified_params is None:
            raise ValueError("UnifiedParamsBlockReader requires a non-None unified_params tensor")
        if unified_params.dim() != 2 or unified_params.shape[1] != 59:
            raise ValueError(
                f"UnifiedParamsBlockReader expects (N, 59) tensor; got shape {tuple(unified_params.shape)}"
            )
        if unified_params.device.type != 'cpu':
            raise ValueError(
                f"UnifiedParamsBlockReader expects a CPU tensor; got device={unified_params.device}"
            )
        self._unified_params = unified_params
        self.block_size = int(block_size)
        self.total_gaussians = int(unified_params.shape[0])
        self.num_blocks = (self.total_gaussians + self.block_size - 1) // self.block_size

    def read_blocks(self, block_ids: List[int]) -> Dict[int, torch.Tensor]:
        """Return a dict {block_id: (rows, 59) CPU tensor view}."""
        out: Dict[int, torch.Tensor] = {}
        data = self._unified_params.data
        for bid in block_ids:
            bid = int(bid)
            if bid < 0 or bid >= self.num_blocks:
                continue
            start = bid * self.block_size
            end = min(start + self.block_size, self.total_gaussians)
            if end <= start:
                continue
            out[bid] = data[start:end]
        return out

    def read_batch(
        self,
        block_ids: List[int],
        out: Optional[torch.Tensor] = None,
    ) -> BlockBatch:
        return _pack_block_batch(self.read_blocks(block_ids), block_ids, self.layout, out)

    def hint_future(
        self,
        block_ids: List[int],
        *,
        target_iteration: Optional[int] = None,
    ) -> int:
        # In-memory slicing has no prefetch semantics.
        return 0

    def contains_block(self, block_id: int) -> bool:
        block_id = int(block_id)
        return 0 <= block_id < self.num_blocks


class TieredCacheBlockReader:
    """Serve block reads through ``TieredCacheManager``.

    Hits are resolved from the in-RAM LRU cache; misses stream from the
    on-SSD log segments via ``storage.read_blocks``.  Dirty blocks that
    were evicted but not yet flushed are also resolved correctly (the
    cache checks its ``flushing_buffer`` before going to SSD).
    """

    layout = BlockLayout.CACHE

    def __init__(
        self,
        cache_manager,
        total_gaussians: int,
        block_size: int,
        before_read: Optional[Callable[[List[int]], None]] = None,
        filter_hint: Optional[Callable[[List[int]], List[int]]] = None,
    ):
        if cache_manager is None:
            raise ValueError("TieredCacheBlockReader requires a non-None cache_manager")
        self._cache = cache_manager
        self.total_gaussians = int(total_gaussians)
        self.block_size = int(block_size)
        self.num_blocks = (self.total_gaussians + self.block_size - 1) // self.block_size
        self._before_read = before_read
        self._filter_hint = filter_hint

    def read_blocks(self, block_ids: List[int]) -> Dict[int, torch.Tensor]:
        valid_ids = [int(b) for b in block_ids if 0 <= int(b) < self.num_blocks]
        if not valid_ids:
            return {}
        if self._before_read is not None:
            self._before_read(valid_ids)
        return self._cache.prefetch(valid_ids)

    def read_batch(
        self,
        block_ids: List[int],
        out: Optional[torch.Tensor] = None,
    ) -> BlockBatch:
        valid_ids = [int(b) for b in block_ids if 0 <= int(b) < self.num_blocks]
        return _pack_block_batch(self.read_blocks(valid_ids), valid_ids, self.layout, out)

    def hint_future(
        self,
        block_ids: List[int],
        *,
        target_iteration: Optional[int] = None,
    ) -> int:
        valid_ids = [int(b) for b in block_ids if 0 <= int(b) < self.num_blocks]
        if self._filter_hint is not None:
            valid_ids = self._filter_hint(valid_ids)
        if not valid_ids:
            return 0
        prefetcher = getattr(self._cache, 'prefetch_future', None)
        if callable(prefetcher):
            if target_iteration is None:
                return int(prefetcher(valid_ids) or 0)
            return int(
                prefetcher(valid_ids, target_iteration=int(target_iteration)) or 0
            )
        return 0

    def contains_block(self, block_id: int) -> bool:
        block_id = int(block_id)
        return 0 <= block_id < self.num_blocks


def parse_block_row_components(
    block_tensor: torch.Tensor,
    layout: BlockLayout,
) -> Dict[str, torch.Tensor]:
    """Slice a (rows, 59) block tensor into its six named components.

    Returns views (no copy) keyed by component name: ``xyz``, ``opacity``,
    ``scaling``, ``rotation``, ``features_dc``, ``features_rest``.
    """
    if block_tensor.dim() != 2 or block_tensor.shape[1] != 59:
        raise ValueError(
            f"parse_block_row_components expects (rows, 59) tensor; got shape {tuple(block_tensor.shape)}"
        )
    if layout == BlockLayout.UNIFIED:
        return {
            'xyz': block_tensor[:, 0:3],
            'opacity': block_tensor[:, 3:4],
            'scaling': block_tensor[:, 4:7],
            'rotation': block_tensor[:, 7:11],
            'features_dc': block_tensor[:, 11:14],
            'features_rest': block_tensor[:, 14:59],
        }
    if layout == BlockLayout.CACHE:
        return {
            'xyz': block_tensor[:, 0:3],
            'scaling': block_tensor[:, 3:6],
            'rotation': block_tensor[:, 6:10],
            'opacity': block_tensor[:, 10:11],
            'features_dc': block_tensor[:, 11:14],
            'features_rest': block_tensor[:, 14:59],
        }
    raise ValueError(f"Unknown BlockLayout: {layout!r}")


def resolve_block_reader_backend(
    requested_backend: str,
    ssd_execution_mode: str,
) -> str:
    """Resolve the ``auto`` alias of ``--paper_block_reader_backend``.

    Returns one of ``'unified_params'`` or ``'tiered_cache'``.  ``auto`` maps
    to ``tiered_cache`` when in paper mode, and ``unified_params`` otherwise.
    """
    requested = str(requested_backend).lower()
    if requested not in {"auto", "unified_params", "tiered_cache"}:
        raise ValueError(
            f"Invalid paper_block_reader_backend={requested!r}; expected one of "
            f"auto, unified_params, tiered_cache"
        )
    if requested == "auto":
        return "tiered_cache" if str(ssd_execution_mode).lower() == "paper" else "unified_params"
    return requested


__all__ = [
    'BlockLayout',
    'BlockBatch',
    'BlockReader',
    'UnifiedParamsBlockReader',
    'TieredCacheBlockReader',
    'parse_block_row_components',
    'resolve_block_reader_backend',
]
