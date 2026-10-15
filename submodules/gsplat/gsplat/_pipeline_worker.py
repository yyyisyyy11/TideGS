"""Torch-free FIFO worker and absolute-deadline primitives for PC1.

GPU event readiness is deliberately independent from Future completion.  The
backend publishes an event after enqueueing work; drain polls that event with a
deadline rather than calling an unbounded CUDA synchronize.
"""

from __future__ import annotations

from concurrent.futures import Future, InvalidStateError, TimeoutError as FutureTimeout
from collections import deque
from dataclasses import dataclass
import math
import queue
import threading
import time


class PipelineTimeoutError(TimeoutError):
    pass


def deadline_after(timeout_s: float) -> float:
    if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0:
        raise ValueError("PC1 timeout must be finite and positive")
    return time.monotonic() + float(timeout_s)


def remaining(deadline: float, phase: str) -> float:
    if not math.isfinite(float(deadline)):
        raise ValueError("PC1 deadline must be finite")
    seconds = float(deadline) - time.monotonic()
    if seconds <= 0:
        raise PipelineTimeoutError(f"PC1 deadline exceeded during {phase}")
    return seconds


def wait_future(future: Future, deadline: float):
    try:
        return future.result(timeout=remaining(deadline, "Future wait"))
    except FutureTimeout as error:
        raise PipelineTimeoutError("PC1 deadline exceeded during Future wait") from error


def wait_event(event, deadline: float, phase: str = "GPU event") -> None:
    while not event.query():
        time.sleep(min(0.002, remaining(deadline, phase)))


@dataclass(frozen=True)
class CommunicationIdentity:
    rank: int
    batch_index: int
    microbatch_index: int
    slot: int
    stage: str
    comm_sequence: int
    schedule_version: str = "pc1-window-v2"


@dataclass(frozen=True)
class WorkerSubmission:
    request: object
    future: Future
    deadline: float
    enqueue_ns: int


class FIFOCommWorker:
    """One persistent daemon, bounded queue, no executor shutdown waits.

    ``execute(stage, request)`` must return a result exposing ``done_event``.
    It must not wait for device completion before publishing that result.
    ``on_failure`` receives exactly one exception argument, once.
    """

    def __init__(self, *, execute, initialize=None, buffer_count: int = 2,
                 on_failure=None, default_timeout_s: float = 120.0,
                 name: str = "pc1-comm"):
        if int(buffer_count) <= 0:
            raise ValueError("PC1 buffer_count must be positive")
        if not math.isfinite(float(default_timeout_s)) or float(default_timeout_s) <= 0:
            raise ValueError("PC1 default_timeout_s must be finite and positive")
        self.buffer_count = int(buffer_count)
        self.default_timeout_s = float(default_timeout_s)
        self._execute = execute
        self._initialize = initialize
        self._on_failure = on_failure
        self._queue = queue.Queue(maxsize=2 * self.buffer_count)
        self._lock = threading.Lock()
        self._pending = {}
        self._published = []
        self._quarantine = []
        self._aborted = None
        self._closed = False
        self._closing = False
        self._sequence = 0
        self._stop = threading.Event()
        self._ready = Future()
        self.cpu_records = deque(maxlen=4096)
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()
        try:
            wait_future(self._ready, self.resolve_deadline(None))
        except BaseException as error:
            self.abort(error)
            raise

    def resolve_deadline(self, deadline):
        value = deadline_after(self.default_timeout_s) if deadline is None else float(deadline)
        if not math.isfinite(value):
            raise ValueError("PC1 deadline must be finite")
        return value

    def next_sequence(self) -> int:
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
            return sequence

    def _ensure_healthy(self):
        if self._aborted is not None:
            raise RuntimeError("PC1 communication worker aborted") from self._aborted
        if self._closed or self._closing:
            raise RuntimeError("PC1 communication worker is closed")

    def _enqueue(self, stage, request):
        deadline = self.resolve_deadline(getattr(request, "deadline", None))
        future = Future()
        # A published NCCL request is never cancellable through Future.cancel().
        future.set_running_or_notify_cancel()
        submission = WorkerSubmission(request, future, deadline, time.perf_counter_ns())
        with self._lock:
            self._ensure_healthy()
            self._published = [result for result in self._published
                               if not result.done_event.query()]
            if request.identity.stage != stage:
                raise ValueError("PC1 request stage does not match enqueue operation")
            self._pending[future] = submission
        try:
            while True:
                with self._lock:
                    self._ensure_healthy()
                try:
                    self._queue.put((stage, submission), timeout=min(
                        0.01, remaining(deadline, "queue enqueue")))
                    break
                except queue.Full:
                    continue
        except BaseException as error:
            self.abort(error)
            raise
        self._record("enqueue", submission)
        return future

    def enqueue_c1(self, request):
        return self._enqueue("C1", request)

    def enqueue_c2(self, request):
        return self._enqueue("C2", request)

    def _record(self, action, submission):
        identity = submission.request.identity
        record = dict(vars(identity))
        record.update(action=action, host_ns=time.perf_counter_ns(),
                      thread_id=threading.get_ident(),
                      native_thread_id=threading.get_native_id(),
                      thread_role="comm" if threading.current_thread() is self.thread else "main")
        with self._lock:
            self.cpu_records.append(record)

    def _run(self):
        try:
            if self._initialize is not None:
                self._initialize()
            self._ready.set_result(None)
            while not self._stop.is_set():
                try:
                    stage, submission = self._queue.get(timeout=0.02)
                except queue.Empty:
                    continue
                result = None
                try:
                    with self._lock:
                        if self._aborted is not None:
                            self._quarantine.append(submission)
                            continue
                    remaining(submission.deadline, "worker begin")
                    self._record("worker_begin", submission)
                    result = self._execute(stage, submission.request)
                    remaining(submission.deadline, "worker submission")
                    self._record("worker_end", submission)
                    publish = False
                    with self._lock:
                        if self._aborted is not None:
                            self._quarantine.extend((submission, result))
                        else:
                            self._published.append(result)
                            publish = True
                    if publish:
                        # User Future callbacks must never execute under our
                        # state lock (they may inspect or drain the worker).
                        try:
                            submission.future.set_result(result)
                        except InvalidStateError:
                            # Abort may have failed this Future after the
                            # result was prepared but before CPU publication.
                            with self._lock:
                                if self._aborted is None:
                                    raise
                        with self._lock:
                            self._pending.pop(submission.future, None)
                except BaseException:
                    if result is not None:
                        with self._lock:
                            self._quarantine.append(result)
                    raise
                finally:
                    self._queue.task_done()
                    # An idle persistent worker must not keep the previous
                    # batch's detached transport result in Python frame locals.
                    submission = result = None
        except BaseException as error:
            if not self._ready.done():
                self._ready.set_exception(error)
            self.abort(error)

    def drain(self, deadline=None):
        deadline = self.resolve_deadline(deadline)
        try:
            with self._lock:
                self._ensure_healthy()
                pending = tuple(self._pending)
            for future in pending:
                wait_future(future, deadline)
            with self._lock:
                published = tuple(self._published)
            for result in published:
                wait_event(result.done_event, deadline, "communication drain")
            with self._lock:
                self._published = [result for result in self._published
                                   if not any(result is old for old in published)]
        except BaseException as error:
            self.abort(error)
            raise

    def close(self, deadline=None):
        deadline = self.resolve_deadline(deadline)
        with self._lock:
            if self._closed:
                return
            self._ensure_healthy()
        self.drain(deadline)
        with self._lock:
            self._closing = True
            self._stop.set()
        self.thread.join(timeout=remaining(deadline, "worker shutdown"))
        if self.thread.is_alive():
            error = PipelineTimeoutError("PC1 deadline exceeded during worker shutdown")
            self.abort(error)
            raise error
        with self._lock:
            self._closed = True

    def abort(self, error):
        if error is None:
            error = RuntimeError("PC1 communication worker aborted")
        with self._lock:
            if self._aborted is not None:
                return
            self._aborted = error
            self._stop.set()
            self._quarantine.extend(self._pending.values())
            self._quarantine.extend(self._published)
            self._published.clear()
            futures = tuple(self._pending)
            self._pending.clear()
        for future in futures:
            try:
                if not future.done():
                    future.set_exception(error)
            except InvalidStateError:
                # A concurrent publication won; submitted GPU buffers are
                # still quarantined regardless of which Future state won.
                pass
        # Never synchronize CUDA, join, destroy a communicator or release
        # possibly in-flight buffers on this failure path.
        if self._on_failure is not None:
            try:
                self._on_failure(error)
            except BaseException:
                pass
