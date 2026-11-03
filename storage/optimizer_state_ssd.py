"""Single-card, block-granular Adam state backing store.

The GPU optimizer owns the CPU tier. This module owns only the durable SSD
tier and is independent from the Gaussian parameter cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


STATE_FORMAT_VERSION = 2
_SUPPORTED_READ_VERSIONS = {1, STATE_FORMAT_VERSION}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(dict(payload)), handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    """Persist directory entry updates after an atomic rename when supported."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    result = {
        "row_count": int(state["row_count"]),
        "step": int(state["step"]),
        "exp_avg": {},
        "exp_avg_sq": {},
    }
    for key in ("exp_avg", "exp_avg_sq"):
        for name, tensor in state[key].items():
            if not torch.is_tensor(tensor):
                raise TypeError(f"optimizer state {key}[{name!r}] must be a tensor")
            result[key][str(name)] = tensor.detach().to(device="cpu").contiguous().clone()
    return result


def _validate_state(
    state: Mapping[str, Any],
    component_widths: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    result = _cpu_state(state)
    row_count = int(result["row_count"])
    if row_count <= 0:
        raise ValueError(f"optimizer state row_count must be positive, got {row_count}")
    if set(result["exp_avg"]) != set(result["exp_avg_sq"]):
        raise ValueError("optimizer state components differ between exp_avg and exp_avg_sq")
    if component_widths is not None and set(result["exp_avg"]) != set(component_widths):
        raise ValueError("optimizer state components do not match the optimizer layout")
    for name in result["exp_avg"]:
        avg = result["exp_avg"][name]
        avg_sq = result["exp_avg_sq"][name]
        if tuple(avg.shape) != tuple(avg_sq.shape):
            raise ValueError(f"optimizer state shape mismatch for component {name!r}")
        if int(avg.shape[0]) != row_count:
            raise ValueError(
                f"optimizer state component {name!r} has {avg.shape[0]} rows, "
                f"expected {row_count}"
            )
        if component_widths is not None and int(avg.shape[1]) != int(component_widths[name]):
            raise ValueError(f"optimizer state width mismatch for component {name!r}")
    return result


class SSDOptimizerStateStore:
    """Asynchronous SSD store with atomic replacement and checksum validation."""

    def __init__(
        self,
        root: str | Path,
        *,
        reset: bool = False,
        allow_cold_start: bool = True,
        component_widths: Optional[Mapping[str, int]] = None,
        index_write_batch: int = 64,
        index_write_interval_s: float = 1.0,
    ):
        self.root = Path(root)
        self.blocks_dir = self.root / "blocks"
        self.index_path = self.root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        self.allow_cold_start = bool(allow_cold_start)
        self.component_widths = {
            str(name): int(width) for name, width in (component_widths or {}).items()
        }
        self.index_write_batch = max(1, int(index_write_batch))
        self.index_write_interval_s = max(0.0, float(index_write_interval_s))
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="optimizer-ssd")
        self._pending: Dict[int, Future] = {}
        self._next_generation: Dict[int, int] = {}
        self._latest_submitted_version: Dict[int, int] = {}
        self._index: Dict[int, Dict[str, Any]] = {}
        self._warnings = 0
        self._index_dirty = False
        self._index_updates = 0
        self._last_index_write = time.monotonic()
        self._index_write_count = 0
        self._state_write_count = 0
        self._load_existing(reset=reset)

    def _load_existing(self, *, reset: bool) -> None:
        if reset:
            # Do not inherit orphaned files from a prior run using the same
            # state directory.  Only files owned by this store are removed.
            try:
                self.index_path.unlink()
            except FileNotFoundError:
                pass
            for path in self.blocks_dir.iterdir():
                if path.is_file() or path.is_symlink():
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
            return
        if not self.index_path.is_file():
            return
        try:
            with open(self.index_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if int(payload.get("state_format_version", -1)) not in _SUPPORTED_READ_VERSIONS:
                raise ValueError("unsupported optimizer state format version")
            saved_widths = {
                str(name): int(width)
                for name, width in dict(payload.get("component_widths", {})).items()
            }
            if self.component_widths and saved_widths and saved_widths != self.component_widths:
                raise ValueError("optimizer state component widths do not match the optimizer")
            entries = payload.get("blocks", {})
            if not isinstance(entries, dict):
                raise ValueError("optimizer state index blocks must be an object")
            for key, metadata in entries.items():
                block_id = int(key)
                self._index[block_id] = dict(metadata)
                self._next_generation[block_id] = int(metadata.get("generation", 0)) + 1
                state_version = int(metadata.get("state_version", 0))
                if state_version > 0:
                    self._latest_submitted_version[block_id] = state_version
        except Exception:
            self._warnings += 1
            self._index.clear()
            self._next_generation.clear()
            self._latest_submitted_version.clear()

    def _write_index(self, *, force: bool = False) -> None:
        if not force and not self._index_dirty:
            return
        payload = {
            "state_format_version": STATE_FORMAT_VERSION,
            "component_widths": dict(self.component_widths),
            "blocks": {
                str(block_id): metadata
                for block_id, metadata in sorted(self._index.items())
            },
        }
        _atomic_json_dump(payload, self.index_path)
        self._index_dirty = False
        self._index_updates = 0
        self._last_index_write = time.monotonic()
        self._index_write_count += 1

    def _maybe_write_index(self) -> None:
        if not self._index_dirty:
            return
        due_count = self._index_updates >= self.index_write_batch
        due_time = (
            self.index_write_interval_s == 0.0
            or time.monotonic() - self._last_index_write >= self.index_write_interval_s
        )
        if due_count or due_time:
            self._write_index()

    def _cleanup_old_generations(self, block_id: int, keep: Path) -> None:
        prefix = f"block_{int(block_id)}_"
        for path in self.blocks_dir.glob(f"{prefix}*.pt"):
            if path != keep:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def _write_state_file(
        self,
        block_id: int,
        generation: int,
        state: Mapping[str, Any],
        state_version: int,
    ) -> Optional[Dict[str, Any]]:
        # A newer version may have been queued while an older write was
        # waiting behind another block write.  Skip the stale write entirely;
        # the newer per-block future will publish the stable file.
        with self._lock:
            latest = int(self._latest_submitted_version.get(int(block_id), 0))
            if latest > 0 and (
                int(state_version) == 0 or latest > int(state_version)
            ):
                return None
        checked = _validate_state(state, self.component_widths or None)
        filename = f"block_{int(block_id)}.pt"
        final_path = self.blocks_dir / filename
        temporary = final_path.with_name(f".{final_path.name}.{int(generation)}.tmp")
        torch.save(checked, temporary)
        _fsync_file(temporary)
        os.replace(temporary, final_path)
        _fsync_directory(final_path.parent)
        self._cleanup_old_generations(block_id, final_path)
        metadata = {
            "path": str(Path("blocks") / filename),
            "generation": int(generation),
            "state_version": int(state_version),
            "row_count": int(checked["row_count"]),
            "step": int(checked["step"]),
            "bytes": int(final_path.stat().st_size),
            "sha256": _sha256(final_path),
        }
        with self._lock:
            self._index[int(block_id)] = metadata
            self._index_dirty = True
            self._index_updates += 1
            self._state_write_count += 1
            self._maybe_write_index()
        return metadata

    def _write_after(
        self,
        previous: Optional[Future],
        block_id: int,
        generation: int,
        state: Mapping[str, Any],
        state_version: int,
    ):
        if previous is not None:
            previous.result()
        return self._write_state_file(block_id, generation, state, state_version)

    def save_async(
        self,
        block_id: int,
        state: Mapping[str, Any],
        *,
        state_version: int = 0,
    ) -> Future:
        block_id = int(block_id)
        staged = _cpu_state(state)
        with self._lock:
            generation = int(self._next_generation.get(block_id, 0))
            self._next_generation[block_id] = generation + 1
            state_version = int(state_version)
            latest = int(self._latest_submitted_version.get(block_id, 0))
            if latest > 0 and state_version == 0:
                future = self._pending.get(block_id)
                if future is not None:
                    return future
                skipped = Future()
                skipped.set_result(None)
                return skipped
            if state_version > 0:
                if latest > state_version:
                    future = self._pending.get(block_id)
                    if future is not None:
                        return future
                    skipped = Future()
                    skipped.set_result(None)
                    return skipped
                self._latest_submitted_version[block_id] = state_version
            previous = self._pending.get(block_id)
            future = self._executor.submit(
                self._write_after,
                previous,
                block_id,
                generation,
                staged,
                state_version,
            )
            self._pending[block_id] = future
            return future

    def _path_for_metadata(self, metadata: Mapping[str, Any]) -> Path:
        path = Path(str(metadata["path"]))
        return path if path.is_absolute() else self.root / path

    def _load_metadata(
        self,
        block_id: int,
        metadata: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        path = self._path_for_metadata(metadata)
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_bytes = int(metadata.get("bytes", 0))
            if expected_bytes and path.stat().st_size != expected_bytes:
                raise ValueError(f"size mismatch for {path}")
            checksum = str(metadata.get("sha256", ""))
            if checksum and _sha256(path) != checksum:
                raise ValueError(f"checksum mismatch for {path}")
            loaded = torch.load(path, map_location="cpu")
            return _validate_state(loaded, self.component_widths or None)
        except Exception as error:
            self._warnings += 1
            if not self.allow_cold_start:
                raise RuntimeError(
                    f"failed to load optimizer state block {block_id}: {error}"
                ) from error
            print(
                "[SSD OPTIMIZER STATE] WARNING: block "
                f"{block_id} could not be loaded; using cold-start: {error}"
            )
            with self._lock:
                self._index.pop(int(block_id), None)
                self._index_dirty = True
                self._index_updates += 1
            return None

    def load(
        self,
        block_id: int,
        *,
        min_state_version: int = 0,
    ) -> Optional[Dict[str, Any]]:
        block_id = int(block_id)
        self.flush_block(block_id)
        with self._lock:
            metadata = self._index.get(block_id)
            if metadata is not None and int(metadata.get("state_version", 0)) < int(min_state_version):
                return None
        if metadata is None:
            return None
        return self._load_metadata(block_id, metadata)

    def state_version(self, block_id: int) -> int:
        with self._lock:
            return int(self._index.get(int(block_id), {}).get("state_version", 0))

    def flush_block(self, block_id: int) -> None:
        block_id = int(block_id)
        with self._lock:
            future = self._pending.get(block_id)
        if future is not None:
            future.result()
            with self._lock:
                if self._pending.get(block_id) is future:
                    self._pending.pop(block_id, None)

    def flush(self) -> None:
        with self._lock:
            futures = list(self._pending.values())
        for future in futures:
            future.result()
        with self._lock:
            self._pending.clear()
            self._write_index(force=True)

    def _copy_entry(self, metadata: Mapping[str, Any], destination_blocks: Path) -> Dict[str, Any]:
        source = self._path_for_metadata(metadata)
        filename = Path(str(metadata["path"])).name
        if filename.startswith("block_") and filename.count("_") >= 2:
            filename = f"block_{Path(filename).stem.split('_')[1]}.pt"
        destination = destination_blocks / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        copied = dict(metadata)
        copied["path"] = str(Path("blocks") / filename)
        copied["bytes"] = int(destination.stat().st_size)
        copied["sha256"] = _sha256(destination)
        return copied

    @staticmethod
    def _write_state_file_to(
        destination_blocks: Path,
        block_id: int,
        generation: int,
        state: Mapping[str, Any],
        state_version: int = 0,
    ) -> Dict[str, Any]:
        checked = _validate_state(state)
        filename = f"block_{int(block_id)}.pt"
        final_path = destination_blocks / filename
        temporary = final_path.with_name(f".{final_path.name}.{int(generation)}.tmp")
        torch.save(checked, temporary)
        _fsync_file(temporary)
        os.replace(temporary, final_path)
        _fsync_directory(final_path.parent)
        return {
            "path": str(Path("blocks") / filename),
            "generation": int(generation),
            "state_version": int(state_version),
            "row_count": int(checked["row_count"]),
            "step": int(checked["step"]),
            "bytes": int(final_path.stat().st_size),
            "sha256": _sha256(final_path),
        }

    def export_checkpoint(
        self,
        checkpoint_dir: str | Path,
        resident_states: Mapping[int, Mapping[str, Any]],
        *,
        state_overrides: Optional[Mapping[int, Mapping[str, Any]]] = None,
        state_versions: Optional[Mapping[int, int]] = None,
        block_size: int,
        optimizer_algorithm: str = "adam",
    ) -> Dict[str, Any]:
        """Export one complete, latest-wins optimizer state snapshot."""
        self.flush()
        checkpoint_root = Path(checkpoint_dir) / "optimizer_state_ssd"
        destination_blocks = checkpoint_root / "blocks"
        destination_blocks.mkdir(parents=True, exist_ok=True)
        exported: Dict[int, Dict[str, Any]] = {}
        with self._lock:
            entries = dict(self._index)
        for block_id, metadata in entries.items():
            # Validate before copying so a damaged old SSD entry is omitted
            # from the checkpoint and follows the documented cold-start path.
            if self._load_metadata(int(block_id), metadata) is None:
                continue
            exported[int(block_id)] = self._copy_entry(metadata, destination_blocks)
        merged: Dict[int, Mapping[str, Any]] = dict(state_overrides or {})
        merged.update({int(k): v for k, v in resident_states.items()})
        versions = {int(k): int(v) for k, v in (state_versions or {}).items()}
        for block_id, state in merged.items():
            block_id = int(block_id)
            generation = int(exported.get(block_id, {}).get("generation", 0)) + 1
            exported[block_id] = self._write_state_file_to(
                destination_blocks,
                block_id,
                generation,
                state,
                versions.get(block_id, int(exported.get(block_id, {}).get("state_version", 0))),
            )

        index_path = checkpoint_root / "index.json"
        _atomic_json_dump(
            {
                "state_format_version": STATE_FORMAT_VERSION,
                "block_size": int(block_size),
                "optimizer_algorithm": str(optimizer_algorithm),
                "component_widths": dict(self.component_widths),
                "blocks": {
                    str(block_id): metadata
                    for block_id, metadata in sorted(exported.items())
                },
            },
            index_path,
        )
        manifest_path = checkpoint_root / "manifest.json"
        manifest = {
            "state_format_version": STATE_FORMAT_VERSION,
            "optimizer_algorithm": str(optimizer_algorithm),
            "block_size": int(block_size),
            "component_widths": dict(self.component_widths),
            "state_block_count": len(exported),
            "optimizer_state_mode": "resident_ssd_cache",
            "optimizer_state_saved": bool(exported),
            "adam_moments_saved": bool(exported),
            "optimizer_state_format_version": STATE_FORMAT_VERSION,
            "state_dir": str(checkpoint_root.resolve()),
            "index": str(index_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "optimizer_state_manifest": str(manifest_path.resolve()),
        }
        _atomic_json_dump(manifest, manifest_path)
        # The checkpoint is a snapshot. Keep the live store index pointed at
        # its own state directory so later writebacks do not read from an old
        # checkpoint and runtime SSD accounting remains accurate.
        with self._lock:
            self._write_index(force=True)
        return manifest

    def import_checkpoint(self, manifest: Mapping[str, Any] | str | Path) -> None:
        manifest_path = (
            Path(manifest)
            if isinstance(manifest, (str, Path))
            else Path(str(manifest["manifest"]))
        )
        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("state_format_version", -1)) not in _SUPPORTED_READ_VERSIONS:
            raise ValueError("unsupported optimizer state format version")
        index_path = Path(str(payload["index"]))
        if not index_path.is_absolute():
            index_path = manifest_path.parent / index_path
        with open(index_path, "r", encoding="utf-8") as handle:
            index_payload = json.load(handle)
        if int(index_payload.get("state_format_version", -1)) not in _SUPPORTED_READ_VERSIONS:
            raise ValueError("unsupported optimizer state index version")
        saved_widths = {
            str(name): int(width)
            for name, width in dict(index_payload.get("component_widths", {})).items()
        }
        if self.component_widths and saved_widths and saved_widths != self.component_widths:
            raise ValueError("optimizer state component widths do not match the optimizer")
        imported: Dict[int, Dict[str, Any]] = {}
        for key, metadata in dict(index_payload.get("blocks", {})).items():
            copied = dict(metadata)
            source = Path(str(copied["path"]))
            source = source if source.is_absolute() else index_path.parent / source
            block_id = int(key)
            try:
                if not source.is_file():
                    raise FileNotFoundError(source)
                expected_bytes = int(copied.get("bytes", 0))
                if expected_bytes and source.stat().st_size != expected_bytes:
                    raise ValueError(f"size mismatch for {source}")
                checksum = str(copied.get("sha256", ""))
                if checksum and _sha256(source) != checksum:
                    raise ValueError(f"checksum mismatch for {source}")

                # Stage imported files into the live state directory. A
                # hardlink is instantaneous on the same filesystem and keeps
                # the state alive even when checkpoint pruning removes the
                # source directory; cross-device filesystems fall back to a
                # copy. The temporary name makes replacement atomic.
                destination = self.blocks_dir / f"block_{block_id}.pt"
                if source.resolve() != destination.resolve():
                    temporary = destination.with_name(
                        f".{destination.name}.import.tmp"
                    )
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
                    try:
                        os.link(source, temporary)
                    except OSError:
                        shutil.copy2(source, temporary)
                    os.replace(temporary, destination)
                    _fsync_directory(destination.parent)
                copied["path"] = str(Path("blocks") / destination.name)
                copied["bytes"] = int(destination.stat().st_size)
                copied["sha256"] = _sha256(destination)
                imported[block_id] = copied
            except Exception as error:
                if not self.allow_cold_start:
                    raise RuntimeError(
                        f"failed to import optimizer state block {block_id}: {error}"
                    ) from error
                self._warnings += 1
                print(
                    "[SSD OPTIMIZER STATE] WARNING: checkpoint block "
                    f"{block_id} could not be imported; using cold-start: {error}"
                )
        with self._lock:
            self._index = imported
            self._next_generation = {
                block_id: int(metadata.get("generation", 0)) + 1
                for block_id, metadata in imported.items()
            }
            self._latest_submitted_version = {
                block_id: int(metadata.get("state_version", 0))
                for block_id, metadata in imported.items()
                if int(metadata.get("state_version", 0)) > 0
            }
            self._index_dirty = True
            self._write_index(force=True)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            entries = list(self._index.values())
            pending = len(self._pending)
            index_writes = self._index_write_count
            state_writes = self._state_write_count
        return {
            "blocks": len(entries),
            "bytes": sum(int(entry.get("bytes", 0)) for entry in entries),
            "warnings": int(self._warnings),
            "pending": pending,
            "index_writes": index_writes,
            "state_writes": state_writes,
        }

    def record_warning(self) -> None:
        self._warnings += 1

    def close(self) -> None:
        self.flush()
        self._executor.shutdown(wait=True)
