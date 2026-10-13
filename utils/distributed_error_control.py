"""CPU-only fatal control for PC1; a failed NCCL batch is never resumed.

The watchdog uses a separate Store, not a collective on either process group.
It can terminate a worker whose main thread is waiting in a CUDA/NCCL call.
No CUDA APIs, stream synchronization, or private PyTorch APIs are used here.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import os
import threading
import time
from typing import Callable


PC1_FATAL_EXIT_CODE = 86


class PC1FatalError(RuntimeError):
    """Raised only if an injected fatal-exit callback returns."""


def _exit_worker(reason: str) -> None:
    # Avoid Python/CUDA teardown and process-group destruction after failure.
    # A full stderr pipe or stalled log file must never block the fatal exit.
    # Give a daemon logger only a bounded chance to preserve CPU diagnostics.
    def log_best_effort():
        try:
            os.write(2, (reason[:4096] + "\n").encode("utf-8", errors="replace"))
        except (OSError, ValueError):
            pass

    try:
        logger = threading.Thread(target=log_best_effort, name="pc1-fatal-log", daemon=True)
        logger.start()
        logger.join(timeout=0.05)
    except (OSError, RuntimeError):
        pass
    finally:
        os._exit(PC1_FATAL_EXIT_CODE)


class PC1ErrorController:
    """Bound PC1 stage waits and propagate fatal errors independently of NCCL.

    ``store`` implements the public torch.distributed.Store set/check/get API
    and must have a short I/O timeout. All production failures exit the worker;
    injecting ``exit_callback`` permits CPU tests without terminating the test
    runner. A successful operation does not imply CUDA work has completed.
    """

    def __init__(
        self, *, store, rank: int, world_size: int,
        stage_timeout: float = 60.0, poll_interval: float = 0.1,
        exit_callback: Callable[[str], None] = _exit_worker,
    ):
        for name, value in (("stage_timeout", stage_timeout), ("poll_interval", poll_interval)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"PC1 {name} must be positive and finite")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid PC1 rank/world_size")
        self.store = store
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.stage_timeout = float(stage_timeout)
        self.poll_interval = float(poll_interval)
        self._exit_callback = exit_callback
        self._lock = threading.Lock()
        self._operations = {}
        self._reason = None
        self._closed = False
        self._stop = threading.Event()
        self._watchdog = threading.Thread(
            target=self._watch, name=f"pc1-error-control-rank-{rank}", daemon=True,
        )
        self._watchdog.start()

    @staticmethod
    def _error_key(rank: int) -> str:
        return f"pc1-fatal-rank-{rank}"

    @property
    def failed(self) -> bool:
        with self._lock:
            return self._reason is not None

    def _raise_if_unusable(self) -> None:
        with self._lock:
            reason, closed = self._reason, self._closed
        if reason is not None:
            raise PC1FatalError(reason)
        if closed:
            raise RuntimeError("PC1 error controller is closed")

    @contextmanager
    def operation(self, phase: str):
        """Guard one host stage, including blocking calls and error checkpoints.

        Repeated and nested scopes are supported. Nested scopes cannot extend
        an outer scope's deadline. Exceptions retain their cause in CPU tests;
        production exits immediately rather than unwinding CUDA resources.
        """
        self._raise_if_unusable()
        token = object()
        with self._lock:
            self._operations[token] = (time.monotonic() + self.stage_timeout, str(phase))
        try:
            yield
            self._raise_if_unusable()
        except BaseException as error:
            self.fail(phase, error)
            raise
        finally:
            with self._lock:
                self._operations.pop(token, None)

    def fail(self, phase: str, error: BaseException) -> None:
        """Publish a local failure and exit; never wait for a peer collective."""
        reason = (
            f"PC1 fatal rank={self.rank} phase={phase}: "
            f"{type(error).__name__}: {error}; NCCL batch cannot be resumed"
        )
        self._trip(reason, publish=True)
        # A returning callback is allowed only for tests. Keep them terminal.
        with self._lock:
            reason = self._reason
        raise PC1FatalError(reason) from error

    def _trip(self, reason: str, *, publish: bool) -> None:
        with self._lock:
            if self._reason is not None:
                return
            self._reason = reason
        self._stop.set()
        if publish:
            try:
                self.store.set(self._error_key(self.rank), reason)
            except Exception:
                # Store failure must not prevent the local fatal exit. Peers
                # also detect control-channel failure or their stage timeout.
                pass
        self._exit_callback(reason)

    def _expired_stage(self):
        with self._lock:
            stages = list(self._operations.values())
        if stages:
            deadline, phase = min(stages)
            if time.monotonic() >= deadline:
                return phase
        return None

    def _watch(self) -> None:
        while not self._stop.is_set():
            phase = self._expired_stage()
            if phase is not None:
                self._trip(
                    f"PC1 fatal rank={self.rank} phase={phase}: stage exceeded "
                    f"{self.stage_timeout:g}s; NCCL batch cannot be resumed",
                    publish=True,
                )
                return
            try:
                for rank in range(self.world_size):
                    key = self._error_key(rank)
                    if self.store.check([key]):
                        message = self.store.get(key)
                        if isinstance(message, bytes):
                            message = message.decode("utf-8", errors="replace")
                        self._trip(
                            f"PC1 fatal rank={self.rank}: peer failure: {message}",
                            publish=False,
                        )
                        return
            except Exception as error:
                self._trip(
                    f"PC1 fatal rank={self.rank}: control channel failed: "
                    f"{type(error).__name__}: {error}", publish=False,
                )
                return
            self._stop.wait(self.poll_interval)

    def close(self) -> None:
        """Stop the healthy watchdog before process-group teardown."""
        with self._lock:
            if self._closed:
                return
            if self._reason is not None:
                raise PC1FatalError(self._reason)
            if self._operations:
                raise RuntimeError("cannot close PC1 error controller during an operation")
            self._closed = True
        self._stop.set()
        self._watchdog.join()

    def release_store(self) -> None:
        """Release only after every rank has stopped its healthy watchdog."""
        with self._lock:
            if not self._closed:
                raise RuntimeError("PC1 store release requires controller.close()")
        self.store = None
