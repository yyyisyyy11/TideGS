"""Opt-in parent process for fatal two-rank PC1 CUDA tests.

Run once, WITHOUT an outer torchrun:
    TIDE_RUN_CUDA_ABORT_TEST=1 python -m unittest \
        tests.test_distributed_pipeline_cuda_faults -v

Each case launches its own torchrun job. Both workers must report their own
controller fatal exit; torchrun killing the peer is not a passing result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


_RUN_FAULTS = (
    os.environ.get("TIDE_RUN_CUDA_ABORT_TEST") == "1"
    and torch is not None
    and torch.cuda.is_available()
    and torch.cuda.device_count() >= 2
)


@unittest.skipUnless(_RUN_FAULTS, "set TIDE_RUN_CUDA_ABORT_TEST=1 on a two-GPU host")
class DistributedPipelineCudaFaultTest(unittest.TestCase):
    def test_rank0_failure_before_c1_interrupts_peer_cuda_split_read(self):
        self._run_fault("before_c1_split")

    def test_rank0_failure_mid_payload_interrupts_peer_collective(self):
        self._run_fault("mid_payload_collective")

    def _run_fault(self, scenario):
        self.assertEqual(
            int(os.environ.get("WORLD_SIZE", "1")), 1,
            "launch this parent unittest without torchrun",
        )
        stage_timeout = float(os.environ.get("TIDE_CUDA_FAULT_STAGE_TIMEOUT_SEC", "4"))
        process_timeout = float(os.environ.get("TIDE_CUDA_FAULT_PROCESS_TIMEOUT_SEC", "90"))
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="tide-pc1-cuda-fault-") as result_dir:
            command = [
                sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node=2", "--max_restarts=0", "--monitor_interval=5",
                "--module", "tests.pc1_cuda_fault_worker",
                "--scenario", scenario, "--result-dir", result_dir,
                "--stage-timeout", str(stage_timeout),
            ]
            process = subprocess.Popen(
                command, cwd=root, env=os.environ.copy(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, start_new_session=True,
            )
            try:
                output, _ = process.communicate(timeout=process_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate(timeout=10)
                self.fail(f"{scenario}: torchrun exceeded {process_timeout:g}s\n{output[-12000:]}")
            self.assertNotEqual(process.returncode, 0, output[-12000:])
            result_path = Path(result_dir)
            fault_path = result_path / "fault.json"
            self.assertTrue(fault_path.is_file(), output[-12000:])
            fault = json.loads(fault_path.read_text())
            self.assertEqual(fault["scenario"], scenario)
            marker_path = result_path / "peer_wait.json"
            self.assertTrue(marker_path.is_file(), output[-12000:])
            marker = json.loads(marker_path.read_text())
            expected_wait = "cuda_split_read" if scenario == "before_c1_split" else "payload_collective"
            self.assertEqual(marker["phase"], expected_wait)
            self.assertLessEqual(marker["time"], fault["time"])
            for rank in range(2):
                exit_path = result_path / f"rank-{rank}-fatal.json"
                self.assertTrue(
                    exit_path.is_file(),
                    f"rank {rank} never reached its controller fatal exit; launcher kill is insufficient\n"
                    + output[-12000:],
                )
                record = json.loads(exit_path.read_text())
                self.assertEqual(record["exit_code"], 86)
                self.assertEqual(record["rank"], rank)
                self.assertIn("PC1 fatal", record["reason"])
                if rank == 0:
                    self.assertIn("RuntimeError: injected rank-0 C1 failure", record["reason"])
                # Same-host monotonic clocks measure fault propagation, without
                # counting import, extension compilation or torchrun startup.
                elapsed = record["time"] - fault["time"]
                self.assertGreaterEqual(elapsed, 0.0)
                self.assertLessEqual(elapsed, stage_timeout + 2.0, record)


if __name__ == "__main__":
    unittest.main()
