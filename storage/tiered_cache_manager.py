"""
Tiered Cache Manager with RAM buffering and LRU eviction.

Manages the CPU RAM layer between GPU and SSD:
- LRU cache for frequently accessed blocks
- Dirty tracking for modified blocks
- Asynchronous GPU->RAM sync
- Intelligent prefetching and eviction
"""

import threading
import time
from collections import OrderedDict
from queue import Queue, Empty, Full
from typing import Dict, List, Set, Optional, Tuple
import torch
import psutil


class TieredCacheManager:
    """
    RAM Cache & Write Buffer for tiered storage system.

    Architecture:
    - cache_data: In-RAM copy of Gaussian blocks
    - dirty_set: Tracks blocks modified by GPU (needs flush to SSD)
    - lru_queue: Maintains access order for eviction
    - async_sync: Background thread for GPU->RAM transfers
    """

    def __init__(
        self,
        storage_manager,
        max_ram_gb: float = 16.0,
        block_size: int = 4096,
        point_dim: int = 59,
        prefetch_distance: int = 5,
        eviction_threshold: float = 0.8,
        verbose: bool = True,
    ):
        """
        Initialize Tiered Cache Manager.

        Args:
            storage_manager: LogStorageManager instance
            max_ram_gb: Maximum RAM usage in GB
            block_size: Number of points per block
            point_dim: Dimension per point
            prefetch_distance: Number of future blocks to prefetch
            eviction_threshold: RAM usage ratio to trigger eviction (0-1)
        """

        # 保存底层存储引用
        # 当 RAM 缓存 Miss时，需要调用它从SSD 中读取
        # 当 RAM 缓存中有脏数据需要罗盘时，需要调用它写入 SSD
        self.storage = storage_manager
        self.max_ram_bytes = int(max_ram_gb * 1024 * 1024 * 1024)
        self.block_size = block_size
        self.point_dim = point_dim
        self.prefetch_distance = prefetch_distance
        self.eviction_threshold = eviction_threshold
        self.verbose = bool(verbose)

        # Bytes per block
        self.bytes_per_block = block_size * point_dim * 4  # float32

        # Cache storage
        # 核心缓存结构
        # 用OrderedDict 来实现 Least Recently Used （LRU）
        # - 最新访问的移动到末尾
        # - 内存满时，弹出最前面的（最久未使用的）
        self.cache_data: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._cache_entry_bytes: Dict[int, int] = {}
        self._resident_ram_bytes = 0
        # 记录脏块的id，如果id在这个set中，那么说明它在内存中的版本比SSD中的版本更新
        # 驱逐前必须先flush，写回SSD
        self.dirty_set: Set[int] = set()
        self.block_versions: Dict[int, int] = {}
        self.block_origin_iterations: Dict[int, int] = {}
        self.cache_lock = threading.RLock()
        
        # [FIX] Flushing buffer to prevent race condition
        # 正在写入 SSD 的 dirty blocks 会暂存在这里，防止 prefetch 读到旧数据
        # Key: block_id, Value: (tensor, timestamp)
        self.flushing_buffer: Dict[int, Tuple[torch.Tensor, float, int]] = {}
        self.flushing_lock = threading.RLock()

        # Async sync queue: (block_id, gpu_tensor, cuda_stream)
        # 生产者消费者模式，用于异步将GPU上的更新同步回RAM
        # 队列元素: (block_id, gpu_tensor, cuda_stream)
        # maxsize = 100, 限制队列长度，防止GPU跑太快导致内存堆积
        self.sync_queue: Queue = Queue(maxsize=100)
        self.sync_thread = None
        self.sync_running = False

        # Async SSD writeback queue (dirty RAM -> SSD patch log)
        self.flush_queue: Queue = Queue(maxsize=128)
        self.flush_thread = None
        self.flush_running = False

        # Async SSD -> RAM future prefetch queue.  Urgent prefetch() remains
        # synchronous fallback; this queue warms Delta+ blocks off the main path.
        self.future_prefetch_queue: Queue = Queue(maxsize=64)
        self.future_prefetch_thread = None
        self.future_prefetch_running = False
        self.future_pending: Set[int] = set()
        self.future_lock = threading.RLock()

        # Per-block SSD read coordination.  The thread that creates an event
        # owns the SSD read; all other callers wait and reuse the cached result.
        self.inflight_reads: Dict[int, threading.Event] = {}
        self.inflight_lock = threading.RLock()

        # Prefetch tracking， 防止重复预取同一个block
        self.prefetch_history: Set[int] = set()
        self._foreground_iteration: Optional[int] = None
        self.timeline_enabled = False
        self._io_events: List[Dict[str, object]] = []
        self._io_events_lock = threading.Lock()

        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'flushes': 0,
            'prefetches': 0,
            'syncs': 0,
            'async_flush_requests': 0,
            'flush_queue_drops': 0,
            'async_flush_jobs': 0,
            'async_flush_blocks': 0,
            'async_flush_time': 0.0,
            'sync_flush_jobs': 0,
            'sync_flush_blocks': 0,
            'sync_flush_time': 0.0,
            'urgent_prefetch_calls': 0,
            'urgent_prefetch_blocks': 0,
            'urgent_prefetch_miss_blocks': 0,
            'urgent_prefetch_time': 0.0,
            'future_prefetch_submitted': 0,
            'future_prefetch_skipped': 0,
            'future_prefetch_dropped': 0,
            'future_prefetch_jobs': 0,
            'future_prefetch_blocks': 0,
            'future_prefetch_time': 0.0,
            'future_prefetch_errors': 0,
            'future_prefetch_reserved': 0,
            'urgent_storage_read_calls': 0,
            'urgent_storage_read_blocks': 0,
            'urgent_storage_read_time': 0.0,
            'future_storage_read_calls': 0,
            'future_storage_read_blocks': 0,
            'future_storage_read_time': 0.0,
            'future_materialize_time': 0.0,
            'ssd_bytes_read_urgent': 0,
            'ssd_bytes_read_future': 0,
            'ssd_bytes_written_async': 0,
            'ssd_bytes_written_sync': 0,
            'inflight_wait_blocks': 0,
            'inflight_wait_time': 0.0,
            'inflight_fallback_blocks': 0,
        }

        # Start async worker threads
        self._start_sync_thread()
        self._start_flush_thread()
        self._start_future_prefetch_thread()

    def set_foreground_iteration(self, iteration: Optional[int]) -> None:
        self._foreground_iteration = (
            None if iteration is None else int(iteration)
        )

    def set_timeline_enabled(self, enabled: bool) -> None:
        """Enable precise host-clock events for an explicit profiling run."""
        self.timeline_enabled = bool(enabled)

    def _record_io_event(self, **event) -> None:
        if self.timeline_enabled:
            event.setdefault("thread_id", threading.get_native_id())
            event.setdefault("timing_source", "host_monotonic")
        with self._io_events_lock:
            self._io_events.append(dict(event))

    def record_io_event(self, **event) -> None:
        self._record_io_event(**event)

    def record_partitioned_io_events(
        self,
        *,
        operation: str,
        tier: str,
        block_tensors: Dict[int, torch.Tensor],
        service_ms: float,
        block_origin_iterations: Optional[Dict[int, int]] = None,
        origin_iteration: Optional[int] = None,
    ) -> None:
        if not block_tensors:
            return
        groups: Dict[Optional[int], Dict[str, int]] = {}
        for raw_block_id, tensor in block_tensors.items():
            block_id = int(raw_block_id)
            origin = None
            if block_origin_iterations is not None:
                origin = block_origin_iterations.get(block_id)
            if origin is None:
                origin = origin_iteration
            if origin is not None:
                origin = int(origin)
            values = groups.setdefault(origin, {"blocks": 0, "bytes": 0})
            values["blocks"] += 1
            values["bytes"] += self._tensor_nbytes(tensor)

        total_bytes = sum(values["bytes"] for values in groups.values())
        total_blocks = sum(values["blocks"] for values in groups.values())
        for origin, values in groups.items():
            weight = (
                float(values["bytes"]) / float(total_bytes)
                if total_bytes > 0
                else float(values["blocks"]) / float(total_blocks)
            )
            self._record_io_event(
                operation=operation,
                tier=tier,
                origin_iteration=origin,
                target_iteration=None,
                blocks=values["blocks"],
                bytes=values["bytes"],
                service_ms=float(service_ms) * weight,
            )

    def drain_io_events(self) -> List[Dict[str, object]]:
        with self._io_events_lock:
            events = list(self._io_events)
            self._io_events.clear()
        return events

    def _start_sync_thread(self):
        """Start background thread for async GPU->RAM sync."""
        # 启动后台线程，用于异步将 GPU 数据 同步到 CPU RAM
        self.sync_running = True
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True) # 这里创建了线程，指定目标函数，并且daemon=True表示线程会随着主程序退出而退出
        self.sync_thread.start()
        self._log("[TieredCache] Async sync thread started")

    def _start_flush_thread(self):
        """Start background thread for async RAM->SSD dirty writeback."""
        self.flush_running = True
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()
        self._log("[TieredCache] Async flush thread started")

    def _start_future_prefetch_thread(self):
        """Start background thread for SSD->RAM future prefetch."""
        self.future_prefetch_running = True
        self.future_prefetch_thread = threading.Thread(
            target=self._future_prefetch_worker,
            daemon=True,
        )
        self.future_prefetch_thread.start()
        self._log("[TieredCache] Async future prefetch thread started")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        element_size = getattr(tensor, 'element_size', None)
        bytes_per_element = int(element_size()) if callable(element_size) else 4
        return int(tensor.numel()) * bytes_per_element

    @classmethod
    def _unique_storage_bytes(cls, tensors) -> int:
        storages = {}
        for tensor in tensors:
            if tensor is None:
                continue
            try:
                storage = tensor.untyped_storage()
                key = (
                    str(tensor.device),
                    int(storage.data_ptr()),
                    int(storage.nbytes()),
                )
                storages[key] = int(storage.nbytes())
            except (AttributeError, RuntimeError):
                storages[("tensor", id(tensor))] = cls._tensor_nbytes(tensor)
        return sum(storages.values())

    @staticmethod
    def _clone_cache_block(tensor: torch.Tensor) -> torch.Tensor:
        """Create one cache-owned storage allocation for one block."""
        if not hasattr(tensor, 'device'):
            return tensor.clone()
        if tensor.device.type != 'cpu':
            raise ValueError(
                f"RAM cache requires CPU tensors, got device={tensor.device}"
            )
        pinned = bool(tensor.is_pinned())
        try:
            owned = torch.empty(
                tuple(tensor.shape),
                dtype=tensor.dtype,
                device='cpu',
                pin_memory=pinned,
            )
        except RuntimeError:
            owned = torch.empty(
                tuple(tensor.shape),
                dtype=tensor.dtype,
                device='cpu',
            )
        owned.copy_(tensor)
        return owned

    def _store_cache_block_locked(
        self,
        block_id: int,
        tensor: torch.Tensor,
    ) -> None:
        block_id = int(block_id)
        self._resident_ram_bytes -= self._cache_entry_bytes.get(block_id, 0)
        block_bytes = self._tensor_nbytes(tensor)
        self.cache_data[block_id] = tensor
        self.cache_data.move_to_end(block_id)
        self._cache_entry_bytes[block_id] = block_bytes
        self._resident_ram_bytes += block_bytes

    def _pop_lru_cache_block_locked(self) -> Tuple[int, torch.Tensor]:
        block_id, tensor = self.cache_data.popitem(last=False)
        self._resident_ram_bytes -= self._cache_entry_bytes.pop(block_id, 0)
        return block_id, tensor

    def _lookup_cached_or_flushing(
        self,
        block_id: int,
        promote_flushing: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        """Return a RAM-resident block without touching SSD."""
        with self.cache_lock:
            tensor = self.cache_data.get(block_id)
            if tensor is not None:
                self.cache_data.move_to_end(block_id)
                return tensor, 'cache'

            with self.flushing_lock:
                flushing_entry = self.flushing_buffer.get(block_id)
                if flushing_entry is None:
                    return None, None
                tensor, _, flushing_version = flushing_entry
                if promote_flushing:
                    cached_tensor = self.cache_data.get(block_id)
                    if cached_tensor is not None:
                        self.cache_data.move_to_end(block_id)
                        return cached_tensor, 'cache'
                    self._store_cache_block_locked(block_id, tensor)
                    self.block_versions[block_id] = max(
                        self.block_versions.get(block_id, 0),
                        int(flushing_version),
                    )
                return tensor, 'flushing'

    def _publish_storage_block(
        self,
        block_id: int,
        tensor: torch.Tensor,
        loaded_version: int,
        *,
        promote_flushing: bool,
    ) -> Tuple[Optional[torch.Tensor], bool, bool]:
        """Publish one SSD result without replacing a newer RAM-known version."""
        block_id = int(block_id)
        loaded_version = int(loaded_version)
        with self.cache_lock:
            with self.flushing_lock:
                known_version = int(self.block_versions.get(block_id, -1))
                cached_tensor = self.cache_data.get(block_id)
                flushing_entry = self.flushing_buffer.get(block_id)

                if flushing_entry is not None:
                    flushing_tensor, _, flushing_version = flushing_entry
                    flushing_version = int(flushing_version)
                    if (
                        flushing_version >= loaded_version
                        and flushing_version >= known_version
                    ):
                        if promote_flushing and cached_tensor is None:
                            self._store_cache_block_locked(
                                block_id,
                                flushing_tensor,
                            )
                        self.block_versions[block_id] = max(
                            known_version,
                            flushing_version,
                        )
                        return flushing_tensor, False, False

                if cached_tensor is not None and known_version >= loaded_version:
                    self.cache_data.move_to_end(block_id)
                    return cached_tensor, False, False

                if known_version > loaded_version:
                    return None, False, True

                self._store_cache_block_locked(block_id, tensor)
                self.block_versions[block_id] = loaded_version
                return tensor, True, False

    def _read_current_storage_blocks(
        self,
        block_ids: List[int],
        *,
        promote_flushing: bool,
        max_attempts: int = 3,
    ):
        """Read and version-arbitrate SSD blocks, retrying stale completions."""
        pending = sorted(set(int(block_id) for block_id in block_ids))
        resolved: Dict[int, torch.Tensor] = {}
        inserted = 0
        physical_blocks = 0
        physical_bytes = 0
        read_elapsed = 0.0
        materialize_elapsed = 0.0
        read_start_ns = None
        read_end_ns = None
        read_attempts = 0

        for _ in range(max_attempts):
            if not pending:
                break
            attempt_start_ns = (
                time.perf_counter_ns() if self.timeline_enabled else None
            )
            read_start = time.perf_counter()
            loaded, loaded_versions = self._read_storage_blocks(pending)
            read_elapsed += time.perf_counter() - read_start
            read_attempts += 1
            if attempt_start_ns is not None:
                if read_start_ns is None:
                    read_start_ns = attempt_start_ns
                read_end_ns = time.perf_counter_ns()
            physical_blocks += len(loaded)
            physical_bytes += sum(
                self._tensor_nbytes(tensor) for tensor in loaded.values()
            )

            retry = []
            materialize_start = time.perf_counter()
            for block_id in pending:
                tensor = loaded.get(block_id)
                if tensor is None:
                    retry.append(block_id)
                    continue
                selected, was_inserted, stale = self._publish_storage_block(
                    block_id,
                    tensor,
                    int(loaded_versions.get(block_id, 0)),
                    promote_flushing=promote_flushing,
                )
                if stale or selected is None:
                    retry.append(block_id)
                    continue
                resolved[block_id] = selected
                inserted += int(was_inserted)
            materialize_elapsed += time.perf_counter() - materialize_start
            pending = retry

        if pending:
            known_versions = {
                block_id: int(self.block_versions.get(block_id, -1))
                for block_id in pending
            }
            storage_versions = self._storage_versions(pending)
            raise RuntimeError(
                "SSD read could not catch the latest RAM-known block versions: "
                f"blocks={pending[:8]} known={known_versions} "
                f"storage={storage_versions}"
            )

        return {
            "blocks": resolved,
            "inserted": inserted,
            "physical_blocks": physical_blocks,
            "physical_bytes": physical_bytes,
            "read_elapsed": read_elapsed,
            "materialize_elapsed": materialize_elapsed,
            "read_start_ns": read_start_ns,
            "read_end_ns": read_end_ns,
            "read_attempts": read_attempts,
        }

    def _storage_versions(self, block_ids: List[int]) -> Dict[int, int]:
        getter = getattr(self.storage, "get_block_versions", None)
        if not callable(getter):
            return {int(block_id): 0 for block_id in block_ids}
        return {
            int(block_id): int(version)
            for block_id, version in getter(block_ids).items()
        }

    def _read_storage_blocks(
        self,
        block_ids: List[int],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
        reader = getattr(self.storage, "read_blocks_with_versions", None)
        if callable(reader):
            blocks, versions = reader(block_ids)
            return blocks, {
                int(block_id): int(version)
                for block_id, version in versions.items()
            }
        blocks = self.storage.read_blocks(block_ids)
        return blocks, self._storage_versions(list(blocks))

    def get_block_versions(self, block_ids: List[int]) -> Dict[int, int]:
        storage_missing = []
        versions = {}
        with self.cache_lock:
            for raw_block_id in block_ids:
                block_id = int(raw_block_id)
                if block_id in self.block_versions:
                    versions[block_id] = int(self.block_versions[block_id])
                else:
                    storage_missing.append(block_id)
        if storage_missing:
            versions.update(self._storage_versions(storage_missing))
        return versions

    def _reserve_inflight_read(self, block_id: int) -> Tuple[threading.Event, bool]:
        """Reserve the right to read one block from SSD."""
        with self.inflight_lock:
            event = self.inflight_reads.get(block_id)
            if event is not None:
                return event, False

            event = threading.Event()
            self.inflight_reads[block_id] = event
            return event, True

    def _complete_inflight_reads(self, block_ids: List[int]) -> None:
        """Wake waiters for completed, skipped, or failed SSD reads."""
        events = []
        with self.inflight_lock:
            for block_id in block_ids:
                event = self.inflight_reads.pop(int(block_id), None)
                if event is not None:
                    events.append(event)

        for event in events:
            event.set()

    def _wait_for_inflight_read(self, event: threading.Event) -> None:
        t0 = time.time()
        event.wait()
        self.stats['inflight_wait_blocks'] += 1
        self.stats['inflight_wait_time'] += time.time() - t0

    def wait_for_prefetches(self) -> None:
        """Wait until queued and in-flight future reads have completed."""
        if hasattr(self, "future_prefetch_queue"):
            self.future_prefetch_queue.join()
        while True:
            with self.inflight_lock:
                events = list(self.inflight_reads.values())
            if not events:
                return
            for event in events:
                event.wait()

    def _read_claimed_blocks(
        self,
        block_ids: List[int],
        result: Dict[int, torch.Tensor],
        cache_hits: List[int],
        log_flushing_hit: bool = False,
    ) -> Tuple[int, float]:
        """Read caller-owned misses from SSD and publish them to cache."""
        if not block_ids:
            return 0, 0.0

        to_read = []
        resolved_before_read = []
        for block_id in block_ids:
            tensor, source = self._lookup_cached_or_flushing(block_id)
            if tensor is not None:
                result[block_id] = tensor
                cache_hits.append(block_id)
                resolved_before_read.append(block_id)
                if log_flushing_hit and source == 'flushing':
                    print(f"[TieredCache] ⚠️ Race avoided: block {block_id} found in flushing_buffer")
            else:
                to_read.append(block_id)

        if resolved_before_read:
            self._complete_inflight_reads(resolved_before_read)

        if not to_read:
            return 0, 0.0

        try:
            loaded = self._read_current_storage_blocks(
                to_read,
                promote_flushing=False,
            )
            elapsed = float(loaded["read_elapsed"])
            bytes_read = int(loaded["physical_bytes"])
            read_blocks = int(loaded["physical_blocks"])
            self.stats['urgent_storage_read_calls'] += 1
            self.stats['urgent_storage_read_blocks'] += read_blocks
            self.stats['urgent_storage_read_time'] += elapsed
            self.stats['ssd_bytes_read_urgent'] += bytes_read
            self._record_io_event(
                operation="ssd_read_urgent",
                tier="ssd",
                origin_iteration=self._foreground_iteration,
                target_iteration=self._foreground_iteration,
                blocks=read_blocks,
                bytes=bytes_read,
                service_ms=elapsed * 1000.0,
            )

            result.update(loaded["blocks"])

            return read_blocks, elapsed
        finally:
            self._complete_inflight_reads(to_read)

    def _flush_dirty_blocks(
        self,
        dirty_blocks: Dict[int, torch.Tensor],
        block_versions: Optional[Dict[int, int]] = None,
        origin_iteration: Optional[int] = None,
        block_origin_iterations: Optional[Dict[int, int]] = None,
        mode: str = 'async',
    ):
        """Flush a batch of dirty blocks to SSD and clear flushing metadata."""
        if not dirty_blocks:
            return

        block_versions = block_versions or {}
        bytes_written = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in dirty_blocks.values()
        )
        t0 = time.time()
        try:
            self.storage.write_patch(
                dirty_blocks,
                block_versions=block_versions or None,
            )
            elapsed = time.time() - t0

            with self.flushing_lock:
                for block_id in dirty_blocks.keys():
                    flushed_version = block_versions.get(block_id)
                    current_entry = self.flushing_buffer.get(block_id)
                    if current_entry is None:
                        continue
                    _, _, buffered_version = current_entry
                    if flushed_version is None or buffered_version <= flushed_version:
                        self.flushing_buffer.pop(block_id, None)

            with self.cache_lock:
                for block_id in dirty_blocks.keys():
                    flushed_version = block_versions.get(block_id)
                    if flushed_version is None:
                        self.dirty_set.discard(block_id)
                        self.block_origin_iterations.pop(block_id, None)
                        continue
                    if self.block_versions.get(block_id, 0) == flushed_version:
                        self.dirty_set.discard(block_id)
                        self.block_origin_iterations.pop(block_id, None)

            self.stats['flushes'] += 1
            if mode == 'async':
                self.stats['async_flush_jobs'] += 1
                self.stats['async_flush_blocks'] += len(dirty_blocks)
                self.stats['async_flush_time'] += elapsed
                self.stats['ssd_bytes_written_async'] += bytes_written
            else:
                self.stats['sync_flush_jobs'] += 1
                self.stats['sync_flush_blocks'] += len(dirty_blocks)
                self.stats['sync_flush_time'] += elapsed
                self.stats['ssd_bytes_written_sync'] += bytes_written
            self.record_partitioned_io_events(
                operation=f"ssd_write_{mode}",
                tier="ssd",
                block_tensors=dirty_blocks,
                block_origin_iterations=block_origin_iterations,
                origin_iteration=origin_iteration,
                service_ms=elapsed * 1000.0,
            )
        except Exception as e:
            print(f"[TieredCache] ERROR during SSD write: {e}")
            with self.cache_lock:
                for block_id, tensor in dirty_blocks.items():
                    failed_version = int(block_versions.get(block_id, 0))
                    current_version = int(self.block_versions.get(block_id, -1))
                    if (
                        block_id not in self.cache_data
                        and current_version <= failed_version
                    ):
                        self._store_cache_block_locked(block_id, tensor)
                        self.block_versions[block_id] = failed_version
                    if current_version <= failed_version:
                        self.dirty_set.add(block_id)
                        failed_origin = (
                            block_origin_iterations.get(block_id)
                            if block_origin_iterations is not None
                            else origin_iteration
                        )
                        if failed_origin is not None:
                            self.block_origin_iterations[block_id] = int(
                                failed_origin
                            )
            with self.flushing_lock:
                for block_id in dirty_blocks:
                    failed_version = int(block_versions.get(block_id, 0))
                    current_entry = self.flushing_buffer.get(block_id)
                    if (
                        current_entry is not None
                        and int(current_entry[2]) <= failed_version
                    ):
                        self.flushing_buffer.pop(block_id, None)
            raise

    def _enqueue_dirty_flush(
        self,
        dirty_blocks: Dict[int, torch.Tensor],
        block_versions: Optional[Dict[int, int]] = None,
        origin_iteration: Optional[int] = None,
        block_origin_iterations: Optional[Dict[int, int]] = None,
        reason: str = 'background',
    ) -> bool:
        """Schedule dirty RAM blocks for background SSD writeback."""
        if not dirty_blocks:
            return True

        try:
            self.flush_queue.put_nowait(
                (
                    reason,
                    dirty_blocks,
                    block_versions or {},
                    None if origin_iteration is None else int(origin_iteration),
                    block_origin_iterations or {},
                )
            )
            self.stats['async_flush_requests'] += 1
            return True
        except Full:
            self.stats['flush_queue_drops'] += 1
            print(f"[TieredCache] Flush queue full; falling back to synchronous write for {len(dirty_blocks)} blocks")
            return False

    def _flush_worker(self):
        """Background worker for async RAM->SSD dirty block writeback."""
        while self.flush_running or not self.flush_queue.empty():
            try:
                item = self.flush_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                if len(item) == 5:
                    (
                        reason,
                        dirty_blocks,
                        block_versions,
                        origin_iteration,
                        block_origin_iterations,
                    ) = item
                elif len(item) == 4:
                    reason, dirty_blocks, block_versions, origin_iteration = item
                    block_origin_iterations = None
                else:
                    reason, dirty_blocks, block_versions = item
                    origin_iteration = None
                    block_origin_iterations = None
                self._flush_dirty_blocks(
                    dirty_blocks,
                    block_versions=block_versions,
                    origin_iteration=origin_iteration,
                    block_origin_iterations=block_origin_iterations,
                    mode='async',
                )
            except Exception as e:
                print(f"[TieredCache] Flush worker error ({reason}): {e}")
            finally:
                self.flush_queue.task_done()

    def _future_prefetch_worker(self):
        """Background worker for SSD->RAM future block prefetch."""
        while self.future_prefetch_running or not self.future_prefetch_queue.empty():
            try:
                item = self.future_prefetch_queue.get(timeout=0.1)
            except Empty:
                continue

            enqueue_ns = None
            if (
                isinstance(item, tuple)
                and len(item) == 4
                and isinstance(item[2], (list, tuple, set))
            ):
                origin_iteration, target_iteration, block_ids, enqueue_ns = item
            elif (
                isinstance(item, tuple)
                and len(item) == 3
                and isinstance(item[2], (list, tuple, set))
            ):
                origin_iteration, target_iteration, block_ids = item
            elif (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[1], (list, tuple, set))
            ):
                origin_iteration, target_iteration, block_ids = None, item[0], item[1]
            else:
                origin_iteration, target_iteration, block_ids = None, None, item
            block_ids = [int(block_id) for block_id in block_ids]
            dequeue_ns = (
                time.perf_counter_ns() if self.timeline_enabled else None
            )
            if enqueue_ns is not None and dequeue_ns is not None:
                self._record_io_event(
                    operation="prefetch_queue_wait",
                    tier="cpu_queue",
                    lane="cpu",
                    origin_iteration=origin_iteration,
                    target_iteration=target_iteration,
                    blocks=len(block_ids),
                    bytes=0,
                    service_ms=(dequeue_ns - int(enqueue_ns)) / 1e6,
                    start_ns=int(enqueue_ns),
                    end_ns=dequeue_ns,
                )
            loaded_count = 0
            t0 = time.time()
            try:
                to_load = []
                for block_id in block_ids:
                    tensor, _ = self._lookup_cached_or_flushing(block_id, promote_flushing=True)
                    if tensor is None:
                        to_load.append(block_id)

                if to_load:
                    loaded = self._read_current_storage_blocks(
                        to_load,
                        promote_flushing=True,
                    )
                    read_elapsed = float(loaded["read_elapsed"])
                    bytes_read = int(loaded["physical_bytes"])
                    read_blocks = int(loaded["physical_blocks"])
                    materialize_elapsed = float(
                        loaded["materialize_elapsed"]
                    )
                    self.stats['future_storage_read_calls'] += 1
                    self.stats['future_storage_read_blocks'] += read_blocks
                    self.stats['future_storage_read_time'] += read_elapsed
                    self.stats['ssd_bytes_read_future'] += bytes_read
                    read_event = {
                        "operation": "ssd_read_future",
                        "tier": "ssd",
                        "origin_iteration": origin_iteration,
                        "target_iteration": target_iteration,
                        "blocks": read_blocks,
                        "bytes": bytes_read,
                        "service_ms": read_elapsed * 1000.0,
                    }
                    if (
                        self.timeline_enabled
                        and int(loaded["read_attempts"]) == 1
                        and loaded["read_start_ns"] is not None
                    ):
                        read_event.update(
                            lane="ssd",
                            start_ns=int(loaded["read_start_ns"]),
                            end_ns=int(loaded["read_end_ns"]),
                        )
                    self._record_io_event(**read_event)
                    self._record_io_event(
                        operation="cpu_materialize_future",
                        tier="cpu_cache",
                        origin_iteration=origin_iteration,
                        target_iteration=target_iteration,
                        blocks=len(loaded["blocks"]),
                        bytes=sum(
                            self._tensor_nbytes(tensor)
                            for tensor in loaded["blocks"].values()
                        ),
                        service_ms=materialize_elapsed * 1000.0,
                    )
                    loaded_count += int(loaded["inserted"])
                    self.stats['future_materialize_time'] += materialize_elapsed

                cache_ready_ns = (
                    time.perf_counter_ns() if self.timeline_enabled else None
                )
                if enqueue_ns is not None and cache_ready_ns is not None:
                    self._record_io_event(
                        operation="prefetch_end_to_end",
                        tier="ssd_to_ram",
                        lane="cpu",
                        origin_iteration=origin_iteration,
                        target_iteration=target_iteration,
                        blocks=len(block_ids),
                        bytes=0,
                        service_ms=(cache_ready_ns - int(enqueue_ns)) / 1e6,
                        start_ns=int(enqueue_ns),
                        end_ns=cache_ready_ns,
                        status="cache_ready",
                    )

                elapsed = time.time() - t0
                self.stats['future_prefetch_jobs'] += 1
                self.stats['future_prefetch_blocks'] += int(loaded_count)
                self.stats['future_prefetch_time'] += elapsed
                if loaded_count:
                    self.stats['prefetches'] += int(loaded_count)
                self._maybe_evict()
            except Exception as e:
                failed_ns = (
                    time.perf_counter_ns() if self.timeline_enabled else None
                )
                if enqueue_ns is not None and failed_ns is not None:
                    self._record_io_event(
                        operation="prefetch_end_to_end",
                        tier="ssd_to_ram",
                        lane="cpu",
                        origin_iteration=origin_iteration,
                        target_iteration=target_iteration,
                        blocks=len(block_ids),
                        bytes=0,
                        service_ms=(failed_ns - int(enqueue_ns)) / 1e6,
                        start_ns=int(enqueue_ns),
                        end_ns=failed_ns,
                        status="failed",
                    )
                self.stats['future_prefetch_errors'] += 1
                print(f"[TieredCache] Future prefetch worker error: {e}")
            finally:
                self._complete_inflight_reads(block_ids)
                with self.future_lock:
                    for block_id in block_ids:
                        self.future_pending.discard(block_id)
                self.future_prefetch_queue.task_done()


    def _sync_worker(self):
        """Background worker for async D2H transfers."""
        while self.sync_running:
            try:
                # Get sync request (timeout to allow shutdown), 生产者/主线程是sync_from_gpu
                item = self.sync_queue.get(timeout=0.1) # 这是一个阻塞操作，如果队列空，线程会在这里暂停 (挂起)，不消耗cpu，一旦sync_from_gpu中有数据，线程会被唤醒，继续执行. 每0.1秒检查一次sync_running标志，确保能shutdown
                block_id, gpu_tensor, cuda_stream = item

                # Wait for CUDA stream to finish
                if cuda_stream is not None:
                    cuda_stream.synchronize()

                # Copy to CPU (pinned memory for faster transfer)
                cpu_tensor = gpu_tensor.cpu().clone()

                # Update cache
                with self.cache_lock:
                    self._store_cache_block_locked(block_id, cpu_tensor)
                    self.block_versions[block_id] = self.block_versions.get(block_id, 0) + 1
                    self.dirty_set.add(block_id) # 标记为脏，也就是修改过，未来需要写回SSD

                self.stats['syncs'] += 1

                # Check if eviction is needed
                self._maybe_evict()

            except Empty:
                continue
            except Exception as e:
                print(f"[TieredCache] Sync worker error: {e}")

    def sync_from_gpu( # TODO: 这段代码认为 Adam 在GPU上进行，GPU更新完了，把结果传回CPU，但是我们的想法是：如果是 CPU 优化，流程会是：GPU 算梯度 -> 梯度传回 CPU -> CPU 做 Adam 更新
        self,
        block_ids: List[int],
        gpu_tensors: List[torch.Tensor],
        cuda_stream: Optional[torch.cuda.Stream] = None
    ):
        """
        Asynchronously sync updated blocks from GPU to RAM.

        Called after optimizer.step() to persist gradient updates.

        Args:
            block_ids: List of block IDs
            gpu_tensors: List of corresponding GPU tensors
            cuda_stream: CUDA stream for async copy (optional)
        """
        if len(block_ids) != len(gpu_tensors):
            raise ValueError("block_ids and gpu_tensors must have same length")

        for block_id, gpu_tensor in zip(block_ids, gpu_tensors):
            # Enqueue async sync (non-blocking)
            self.sync_queue.put((block_id, gpu_tensor, cuda_stream))

    def _is_valid_block_tensor(
        self,
        block_id: int,
        tensor: Optional[torch.Tensor],
        context: str,
    ) -> bool:
        if tensor is None or tensor.numel() == 0:
            print(f"[TieredCache] WARNING: skip empty block {block_id} during {context}")
            return False
        if tensor.dim() != 2 or tensor.shape[1] != self.point_dim:
            raise ValueError(
                f"Invalid block {block_id} shape {tuple(tensor.shape)} during {context}; "
                f"expected (*, {self.point_dim})"
            )
        if tensor.shape[0] > self.block_size:
            raise ValueError(
                f"Invalid block {block_id} rows {tensor.shape[0]} during {context}; "
                f"expected at most {self.block_size}"
            )
        return True

    def upsert_dirty_blocks(self, block_tensors: Dict[int, torch.Tensor], clone: bool = False) -> int:
        """Refresh RAM cache entries from CPU tensors and mark them dirty.

        This path is used by the paper-mode bridge after GPU resident updates:
        modified blocks are staged in the warm RAM cache before background SSD
        writeback is scheduled.
        """
        if not block_tensors:
            return 0

        staged = 0
        with self.cache_lock:
            for block_id, tensor in block_tensors.items():
                cached_tensor = self._clone_cache_block(tensor)
                if not self._is_valid_block_tensor(block_id, cached_tensor, 'upsert_dirty_blocks'):
                    continue
                self._store_cache_block_locked(block_id, cached_tensor)
                self.block_versions[block_id] = self.block_versions.get(block_id, 0) + 1
                self.dirty_set.add(block_id)
                staged += 1

        self._maybe_evict()
        return staged

    def upsert_dirty_block_batch(
        self,
        block_tensors: Dict[int, torch.Tensor],
        block_versions: Optional[Dict[int, int]] = None,
        origin_iteration: Optional[int] = None,
        block_origin_iterations: Optional[Dict[int, int]] = None,
    ) -> int:
        """Copy staged writeback views into independently owned cache blocks."""
        if not block_tensors:
            return 0

        items = []
        for block_id, tensor in block_tensors.items():
            if not self._is_valid_block_tensor(
                block_id, tensor, 'upsert_dirty_block_batch'
            ):
                continue
            items.append((int(block_id), tensor))
        if not items:
            return 0

        candidates = []
        with self.cache_lock:
            for block_id, tensor in items:
                current_version = int(self.block_versions.get(block_id, 0))
                incoming_version = (
                    current_version + 1
                    if block_versions is None
                    else int(block_versions[block_id])
                )
                if incoming_version > current_version:
                    candidates.append(
                        (block_id, tensor, incoming_version)
                    )
        candidates = [
            (block_id, self._clone_cache_block(tensor), incoming_version)
            for block_id, tensor, incoming_version in candidates
        ]

        staged = 0
        with self.cache_lock:
            for block_id, owned, incoming_version in candidates:
                current_version = int(self.block_versions.get(block_id, 0))
                if incoming_version <= current_version:
                    continue
                self._store_cache_block_locked(block_id, owned)
                self.block_versions[block_id] = incoming_version
                block_origin = (
                    block_origin_iterations.get(block_id)
                    if block_origin_iterations is not None
                    else origin_iteration
                )
                if block_origin is not None:
                    self.block_origin_iterations[block_id] = int(block_origin)
                self.dirty_set.add(block_id)
                staged += 1

        self._maybe_evict()
        return staged

    def submit_dirty_blocks(self, block_ids: List[int], clone: bool = True, reason: str = 'paper_writeback') -> int:
        """Submit dirty RAM blocks for asynchronous SSD writeback."""
        if not block_ids:
            return 0

        dirty_blocks: Dict[int, torch.Tensor] = {}
        dirty_versions: Dict[int, int] = {}
        dirty_origins: Dict[int, int] = {}

        with self.cache_lock:
            for block_id in block_ids:
                if block_id not in self.cache_data or block_id not in self.dirty_set:
                    continue
                tensor = self.cache_data[block_id]
                if not self._is_valid_block_tensor(block_id, tensor, 'submit_dirty_blocks'):
                    self.dirty_set.discard(block_id)
                    continue
                dirty_blocks[block_id] = tensor.clone() if clone else tensor
                dirty_versions[block_id] = self.block_versions.get(block_id, 0)
                if block_id in self.block_origin_iterations:
                    dirty_origins[block_id] = int(
                        self.block_origin_iterations[block_id]
                    )

        if not dirty_blocks:
            return 0

        with self.flushing_lock:
            current_time = time.time()
            for block_id, tensor in dirty_blocks.items():
                version = dirty_versions.get(block_id, 0)
                existing = self.flushing_buffer.get(block_id)
                if existing is None or existing[2] <= version:
                    self.flushing_buffer[block_id] = (tensor, current_time, version)

        enqueued = self._enqueue_dirty_flush(
            dirty_blocks,
            block_versions=dirty_versions,
            block_origin_iterations=dirty_origins,
            reason=reason,
        )
        if not enqueued:
            self._flush_dirty_blocks(
                dirty_blocks,
                block_versions=dirty_versions,
                block_origin_iterations=dirty_origins,
                mode='sync',
            )

        return len(dirty_blocks)

    def prefetch(
        self,
        needed_block_ids: List[int],
        future_block_ids: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Prefetch blocks from cache or SSD.

        Args:
            needed_block_ids: Blocks needed immediately
            future_block_ids: Blocks that will be needed soon (for prefetching)

        Returns:
            Dictionary mapping block_id -> tensor (CPU)
        """
        result = {}
        cache_hits = []
        read_ids = []
        wait_entries = []
        seen_block_ids = set()
        self.stats['urgent_prefetch_calls'] += 1

        # Check order: cache_data -> flushing_buffer -> in-flight read -> SSD.
        for raw_block_id in needed_block_ids:
            block_id = int(raw_block_id)
            if block_id in seen_block_ids:
                continue
            seen_block_ids.add(block_id)

            tensor, source = self._lookup_cached_or_flushing(block_id)
            if tensor is not None:
                result[block_id] = tensor
                cache_hits.append(block_id)
                if source == 'flushing':
                    print(f"[TieredCache] ⚠️ Race avoided: block {block_id} found in flushing_buffer")
                continue

            event, claimed = self._reserve_inflight_read(block_id)
            if claimed:
                read_ids.append(block_id)
            else:
                wait_entries.append((block_id, event))

        actual_reads, read_time = self._read_claimed_blocks(
            read_ids,
            result,
            cache_hits,
            log_flushing_hit=True,
        )

        while wait_entries:
            fallback_ids = []
            next_wait_entries = []
            for block_id, event in wait_entries:
                self._wait_for_inflight_read(event)
                tensor, source = self._lookup_cached_or_flushing(block_id)
                if tensor is not None:
                    result[block_id] = tensor
                    cache_hits.append(block_id)
                    if source == 'flushing':
                        print(f"[TieredCache] ⚠️ Race avoided: block {block_id} found in flushing_buffer")
                    continue

                retry_event, claimed = self._reserve_inflight_read(block_id)
                if claimed:
                    fallback_ids.append(block_id)
                    self.stats['inflight_fallback_blocks'] += 1
                else:
                    next_wait_entries.append((block_id, retry_event))

            fallback_reads, fallback_time = self._read_claimed_blocks(
                fallback_ids,
                result,
                cache_hits,
                log_flushing_hit=True,
            )
            actual_reads += fallback_reads
            read_time += fallback_time
            wait_entries = next_wait_entries

        self.stats['urgent_prefetch_blocks'] += actual_reads
        self.stats['urgent_prefetch_miss_blocks'] += actual_reads
        self.stats['urgent_prefetch_time'] += read_time

        # Update stats
        self.stats['cache_hits'] += len(cache_hits)
        self.stats['cache_misses'] += actual_reads

        # Background prefetch for future blocks
        # if future_block_ids:
        #     self._background_prefetch(future_block_ids)

        # Trigger eviction if needed
        self._maybe_evict()

        return result

    def prefetch_future(
        self,
        block_ids: List[int],
        *,
        target_iteration: Optional[int] = None,
    ) -> int:
        """Prefetch future blocks in background without blocking urgent fetches."""
        to_prefetch = []
        skipped = 0

        with self.cache_lock:
            for block_id in sorted(set(int(block_id) for block_id in block_ids)):
                if block_id in self.cache_data:
                    self.cache_data.move_to_end(block_id)
                    skipped += 1
                    continue

                with self.flushing_lock:
                    flushing_entry = self.flushing_buffer.get(block_id)
                if flushing_entry is not None:
                    tensor, _, version = flushing_entry
                    self._store_cache_block_locked(block_id, tensor)
                    self.block_versions[block_id] = max(
                        self.block_versions.get(block_id, 0),
                        int(version),
                    )
                    skipped += 1
                    continue

                with self.future_lock:
                    if block_id in self.future_pending:
                        skipped += 1
                        continue
                    _, claimed = self._reserve_inflight_read(block_id)
                    if not claimed:
                        skipped += 1
                        continue
                    self.future_pending.add(block_id)
                to_prefetch.append(block_id)

        self.stats['future_prefetch_skipped'] += int(skipped)

        if not to_prefetch:
            return 0

        enqueue_ns = time.perf_counter_ns() if self.timeline_enabled else None
        try:
            item = (
                    self._foreground_iteration,
                    None if target_iteration is None else int(target_iteration),
                    to_prefetch,
                )
            if enqueue_ns is not None:
                item += (enqueue_ns,)
            self.future_prefetch_queue.put_nowait(item)
            if enqueue_ns is not None:
                self._record_io_event(
                    operation="prefetch_hint_enqueue",
                    tier="cpu_queue",
                    lane="cpu",
                    origin_iteration=self._foreground_iteration,
                    target_iteration=target_iteration,
                    blocks=len(to_prefetch),
                    bytes=0,
                    service_ms=0.0,
                    start_ns=enqueue_ns,
                    end_ns=enqueue_ns,
                    status="submitted",
                )
            self.stats['future_prefetch_submitted'] += len(to_prefetch)
            self.stats['future_prefetch_reserved'] += len(to_prefetch)
            return len(to_prefetch)
        except Full:
            self._complete_inflight_reads(to_prefetch)
            with self.future_lock:
                for block_id in to_prefetch:
                    self.future_pending.discard(block_id)
            if enqueue_ns is not None:
                dropped_ns = time.perf_counter_ns()
                self._record_io_event(
                    operation="prefetch_hint_enqueue",
                    tier="cpu_queue",
                    lane="cpu",
                    origin_iteration=self._foreground_iteration,
                    target_iteration=target_iteration,
                    blocks=len(to_prefetch),
                    bytes=0,
                    service_ms=(dropped_ns - enqueue_ns) / 1e6,
                    start_ns=enqueue_ns,
                    end_ns=dropped_ns,
                    status="dropped_queue_full",
                )
            self.stats['future_prefetch_dropped'] += len(to_prefetch)
            print(f"[TieredCache] Future prefetch queue full; dropped {len(to_prefetch)} blocks")
            return 0

    def _maybe_evict(self):
        """Check RAM usage and trigger eviction if needed."""
        current_usage = self._get_ram_usage()
        threshold_bytes = self.max_ram_bytes * self.eviction_threshold

        if current_usage > threshold_bytes:
            self.evict_and_flush(target_ram_bytes=int(threshold_bytes))

    def _get_ram_usage(self) -> int:
        """Get physical bytes owned by resident cache entries."""
        with self.cache_lock:
            return int(self._resident_ram_bytes)

    def _get_flushing_ram_usage(self) -> int:
        with self.flushing_lock:
            return self._unique_storage_bytes(
                entry[0] for entry in self.flushing_buffer.values()
            )

    def _get_managed_ram_usage(self) -> int:
        with self.cache_lock:
            with self.flushing_lock:
                tensors = list(self.cache_data.values())
                tensors.extend(
                    entry[0] for entry in self.flushing_buffer.values()
                )
                return self._unique_storage_bytes(tensors)

    def evict_and_flush(
        self,
        num_blocks: Optional[int] = None,
        target_ram_bytes: Optional[int] = None,
    ):
        """
        Evict blocks from RAM cache using LRU policy.

        Critical logic (RACE CONDITION FIXED):
        - Clean blocks: Just remove from cache
        - Dirty blocks: Move to flushing_buffer -> Flush to SSD -> Remove from flushing_buffer
        
        This prevents race condition where:
        1. evict removes block from cache
        2. prefetch reads stale data from SSD (write not finished)
        3. prefetch overwrites cache with stale data

        Args:
            num_blocks: Number of blocks to evict (default: evict 20% of cache)
        """
        if num_blocks is None and target_ram_bytes is None:
            num_blocks = max(1, int(len(self.cache_data) * 0.2))

        victims = []
        dirty_victims = {}
        dirty_versions = {}
        dirty_origins = {}

        with self.cache_lock:
            resident_bytes = int(self._resident_ram_bytes)
            max_victims = (
                len(self.cache_data)
                if num_blocks is None
                else min(int(num_blocks), len(self.cache_data))
            )
            # Select victims from LRU tail
            for _ in range(max_victims):
                if not self.cache_data:
                    break
                if (
                    target_ram_bytes is not None
                    and resident_bytes <= int(target_ram_bytes)
                ):
                    break

                # Pop from front (oldest)
                block_id, tensor = self._pop_lru_cache_block_locked()
                victims.append(block_id)
                resident_bytes -= self._tensor_nbytes(tensor)

                # Check if dirty
                if block_id in self.dirty_set:
                    if not self._is_valid_block_tensor(block_id, tensor, 'evict_and_flush'):
                        self.dirty_set.discard(block_id)
                        continue
                    dirty_victims[block_id] = tensor
                    dirty_versions[block_id] = self.block_versions.get(block_id, 0)
                    if block_id in self.block_origin_iterations:
                        dirty_origins[block_id] = int(
                            self.block_origin_iterations[block_id]
                        )
                    # Publish under cache_lock so readers cannot observe a gap.
                    with self.flushing_lock:
                        version = int(dirty_versions[block_id])
                        existing = self.flushing_buffer.get(block_id)
                        if existing is None or int(existing[2]) <= version:
                            self.flushing_buffer[block_id] = (
                                tensor,
                                time.time(),
                                version,
                            )

        # Flush dirty blocks to SSD on the background writeback worker.
        if dirty_victims:
            enqueued = self._enqueue_dirty_flush(
                dirty_victims,
                block_versions=dirty_versions,
                block_origin_iterations=dirty_origins,
                reason='evict',
            )
            if not enqueued:
                self._flush_dirty_blocks(
                    dirty_victims,
                    block_versions=dirty_versions,
                    block_origin_iterations=dirty_origins,
                    mode='sync',
                )

        self.stats['evictions'] += len(victims)

        if victims:
            print(f"[TieredCache] Evicted {len(victims)} blocks ({len(dirty_victims)} dirty)")

    def flush_all_dirty(self):
        """Flush all dirty blocks to SSD (e.g., at end of epoch)."""
        with self.cache_lock:
            dirty_blocks = {
                block_id: self.cache_data[block_id]
                for block_id in self.dirty_set
                if block_id in self.cache_data
            }
            dirty_versions = {
                block_id: self.block_versions.get(block_id, 0)
                for block_id in dirty_blocks.keys()
            }
            dirty_origins = {
                block_id: int(self.block_origin_iterations[block_id])
                for block_id in dirty_blocks
                if block_id in self.block_origin_iterations
            }

        if dirty_blocks:
            print(f"[TieredCache] Flushing {len(dirty_blocks)} dirty blocks to SSD")

            with self.flushing_lock:
                current_time = time.time()
                for block_id, tensor in dirty_blocks.items():
                    version = dirty_versions.get(block_id, self.block_versions.get(block_id, 0))
                    self.flushing_buffer[block_id] = (tensor, current_time, version)

            self._flush_dirty_blocks(
                dirty_blocks,
                block_versions=dirty_versions,
                block_origin_iterations=dirty_origins,
                mode='sync',
            )

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.cache_lock:
            cache_size = len(self.cache_data)
            dirty_count = len(self.dirty_set)
            ram_usage_mb = self._get_ram_usage() / 1024 / 1024
        
        with self.flushing_lock:
            flushing_count = len(self.flushing_buffer)
        flushing_ram_usage_mb = self._get_flushing_ram_usage() / 1024 / 1024
        managed_ram_usage_mb = self._get_managed_ram_usage() / 1024 / 1024

        flush_queue_size = self.flush_queue.qsize() if hasattr(self, 'flush_queue') else 0
        future_queue_size = self.future_prefetch_queue.qsize() if hasattr(self, 'future_prefetch_queue') else 0
        with self.future_lock:
            future_pending_count = len(self.future_pending)
        with self.inflight_lock:
            inflight_read_count = len(self.inflight_reads)

        hit_rate = 0.0
        total_accesses = self.stats['cache_hits'] + self.stats['cache_misses']
        if total_accesses > 0:
            hit_rate = self.stats['cache_hits'] / total_accesses

        return {
            **self.stats,
            'cache_size': cache_size,
            'dirty_blocks': dirty_count,
            'flushing_blocks': flushing_count,  # [NEW]
            'flush_queue_size': flush_queue_size,
            'future_queue_size': future_queue_size,
            'future_pending_blocks': future_pending_count,
            'future_prefetch_queue_size': future_queue_size,
            'future_prefetch_pending': future_pending_count,
            'inflight_read_blocks': inflight_read_count,
            'ram_usage_mb': ram_usage_mb,
            'flushing_ram_usage_mb': flushing_ram_usage_mb,
            'managed_ram_usage_mb': managed_ram_usage_mb,
            'hit_rate': hit_rate,
            'max_ram_mb': self.max_ram_bytes / 1024 / 1024
        }

    def shutdown(self):
        """Shutdown cache manager and flush all dirty blocks."""
        self._log("[TieredCache] Shutting down...")

        self.sync_running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5.0)

        if self.flush_thread:
            self.flush_queue.join()

        if self.future_prefetch_thread:
            self.future_prefetch_queue.join()

        self.flush_all_dirty()

        self.future_prefetch_running = False
        if self.future_prefetch_thread:
            self.future_prefetch_thread.join(timeout=5.0)

        self.flush_running = False
        if self.flush_thread:
            self.flush_thread.join(timeout=5.0)

        max_wait = 10.0  # seconds
        start_time = time.time()
        while True:
            with self.flushing_lock:
                if not self.flushing_buffer:
                    break
                remaining = len(self.flushing_buffer)

            if time.time() - start_time > max_wait:
                print(f"[TieredCache] WARNING: {remaining} blocks still in flushing_buffer after {max_wait}s")
                break

            time.sleep(0.1)

        stats = self.get_stats()
        print(
            "[TieredCache] Shutdown summary: "
            f"hits={stats.get('cache_hits', 0)} "
            f"misses={stats.get('cache_misses', 0)} "
            f"hit_rate={stats.get('hit_rate', 0.0):.3f} "
            f"dirty={stats.get('dirty_blocks', 0)} "
            f"sync_flush_blocks={stats.get('sync_flush_blocks', 0)} "
            f"ram_mb={stats.get('ram_usage_mb', 0.0):.1f}"
        )
