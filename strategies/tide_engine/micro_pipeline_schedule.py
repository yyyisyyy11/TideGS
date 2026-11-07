"""Deterministic Micro Pipeline host scheduling, independent of the CUDA runtime."""

from __future__ import annotations

import math
import time


MICRO_PIPELINE_SCHEDULE_VERSION = "micro-pipeline-window-v2"


def run_micro_pipeline_windows(
    microbatches, *, buffer_count, controller, project, image_backward,
    final_backward, clock=time.monotonic,
):
    """Execute each fixed window as all P/C1, all M/C2, then ordered G.

    Callbacks receive one absolute monotonic deadline. A window operation also
    keeps Python work between callbacks under the independent watchdog. No
    completion-order decisions or success-path rank collectives belong here.
    Only the current window's handles are retained by this executor.
    """
    buffer_count = int(buffer_count)
    if buffer_count < 1:
        raise ValueError("Micro Pipeline buffer_count must be positive")
    timeout = float(controller.stage_timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Micro Pipeline window timeout must be positive and finite")
    iterator = iter(microbatches)
    window_index = 0
    while True:
        window = []
        for _ in range(buffer_count):
            try:
                window.append(next(iterator))
            except StopIteration:
                break
        if not window:
            return
        deadline = clock() + timeout

        def checked(callback, *args):
            if clock() >= deadline:
                raise TimeoutError(f"Micro Pipeline window {window_index} deadline expired")
            value = callback(*args, deadline=deadline)
            if clock() >= deadline:
                raise TimeoutError(f"Micro Pipeline window {window_index} deadline expired")
            return value

        with controller.operation(f"Micro Pipeline window {window_index}"):
            handles = [checked(project, item) for item in window]
            for item, handle in zip(window, handles):
                checked(image_backward, item, handle)
            for item, handle in zip(window, handles):
                checked(final_backward, item, handle)
            # Do not carry projection graphs from this window into the next.
            del handle, handles
        window_index += 1
