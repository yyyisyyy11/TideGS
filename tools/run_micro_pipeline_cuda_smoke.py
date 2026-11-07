"""Launch the opted-in two-rank Micro Pipeline suite with a process-level deadline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=600.0)
    arguments = parser.parse_args()
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")
    if os.environ.get("TIDE_RUN_CUDA_SMOKE") != "1":
        parser.error("set TIDE_RUN_CUDA_SMOKE=1 to opt into real CUDA/NCCL execution")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("launch this parent without torchrun")
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=2", "--max_restarts=0", "--module", "unittest",
        "tests.test_distributed_pipeline_cuda", "-v",
    ]
    process = subprocess.Popen(
        command, cwd=Path(__file__).resolve().parents[1], env=os.environ.copy(),
        start_new_session=True,
    )
    try:
        return process.wait(timeout=arguments.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
        print(f"Micro Pipeline numerical smoke exceeded {arguments.timeout:g}s", file=sys.stderr)
        return 124
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
