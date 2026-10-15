"""External parent for disposable two-rank CUDA/NCCL Micro Pipeline failure tests.

    TIDE_RUN_CUDA_ABORT_TEST=1 python -m unittest \
        tests.test_distributed_pipeline_cuda_faults -v

Run WITHOUT torchrun. The parent starts two independent rank processes and
requires BOTH actual exit codes to be 86. No launcher can kill a peer and make
this test pass. Startup and fault propagation have separate bounded deadlines.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from tests.micro_pipeline_fault_evidence import SCENARIOS, validate_fault_evidence

try:
    import torch
except ImportError:
    torch = None


RUN_FAULTS = (
    os.environ.get("TIDE_RUN_CUDA_ABORT_TEST") == "1"
    and torch is not None
    and torch.cuda.is_available()
    and torch.cuda.device_count() >= 2
)


@unittest.skipUnless(RUN_FAULTS, "set TIDE_RUN_CUDA_ABORT_TEST=1 on a two-GPU host")
class DistributedPipelineCudaFaultTest(unittest.TestCase):
    def test_rank0_failure_before_c1(self):
        self._run_fault("before_c1")

    def test_rank0_failure_in_c1_metadata(self):
        self._run_fault("c1_metadata")

    def test_rank0_failure_mid_c1_payload(self):
        self._run_fault("c1_payload")

    def test_rank0_failure_after_intermediate_before_c2(self):
        self._run_fault("before_c2")

    def test_rank0_failure_mid_c2_payload(self):
        self._run_fault("c2_payload")

    def test_rank0_failure_in_final_gradient(self):
        self._run_fault("final_gradient")

    def test_control_channel_failure(self):
        self._run_fault("control_channel")

    def test_shutdown_while_worker_task_is_pending(self):
        self._run_fault("shutdown_pending")

    def _run_fault(self, scenario):
        self.assertEqual(int(os.environ.get("WORLD_SIZE", "1")), 1,
                         "launch this parent unittest without torchrun")
        stage_timeout = float(os.environ.get("TIDE_CUDA_FAULT_STAGE_TIMEOUT_SEC", "4"))
        process_timeout = float(os.environ.get("TIDE_CUDA_FAULT_PROCESS_TIMEOUT_SEC", "90"))
        self.assertGreater(stage_timeout, 0)
        self.assertGreater(process_timeout, stage_timeout)
        root = Path(__file__).resolve().parents[1]
        # Each rank uses env:// rendezvous directly. No elastic supervisor is
        # present, so each process can exit only through its own path or timeout.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        with tempfile.TemporaryDirectory(prefix=f"tide-micro-pipeline-{scenario}-") as result_dir:
            result_path = Path(result_dir)
            command = [sys.executable, "-m", "tests.micro_pipeline_cuda_fault_worker",
                       "--scenario", scenario, "--result-dir", result_dir,
                       "--stage-timeout", str(stage_timeout)]
            processes = []
            logs = []
            deadline = time.monotonic() + process_timeout
            try:
                for rank in range(2):
                    environment = os.environ.copy()
                    environment.update({
                        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                        "RANK": str(rank), "LOCAL_RANK": str(rank),
                        "WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2",
                        "PYTHONUNBUFFERED": "1",
                    })
                    log = (result_path / f"rank-{rank}.log").open("w")
                    logs.append(log)
                    processes.append(subprocess.Popen(
                        command, cwd=root, env=environment, stdout=log,
                        stderr=subprocess.STDOUT, text=True, start_new_session=True,
                    ))
                while any(process.poll() is None for process in processes):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{scenario}: rank process exceeded {process_timeout:g}s")
                    time.sleep(0.05)
                validate_fault_evidence(
                    result_path, scenario,
                    {rank: process.returncode for rank, process in enumerate(processes)},
                    stage_timeout,
                )
            except BaseException as error:
                # Kill only after deciding this run failed; cleanup deaths can
                # never be accepted as evidence of bounded production exits.
                for process in processes:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                for process in processes:
                    process.wait(timeout=10)
                output = "\n".join(
                    f"rank {rank}:\n{(result_path / f'rank-{rank}.log').read_text()[-12000:]}"
                    for rank in range(len(processes))
                )
                raise AssertionError(f"{error}\n{output}") from error
            finally:
                for log in logs:
                    log.close()


class FaultEvidenceCpuTest(unittest.TestCase):
    """Check acceptance/rejection behavior without CUDA or subprocess mocks."""

    @staticmethod
    def _write(root, name, value):
        (root / name).write_text(json.dumps(value))

    def _fixture(self, root, scenario):
        self._write(root, "fault.json", {"rank": 0, "scenario": scenario, "time": 10.0})
        self._write(root, "peer_wait.json", {
            "rank": 1, "phase": SCENARIOS[scenario], "time": 9.9,
            "gpu_work_submitted": scenario != "control_channel",
            "gpu_pending_at_marker": scenario == "final_gradient",
            "native_backward_entered": False,
        })
        for rank in range(2):
            reason = ("Micro Pipeline fatal control channel failed: ConnectionError"
                      if scenario == "control_channel" else
                      f"Micro Pipeline fatal RuntimeError: injected rank-0 {scenario} failure")
            self._write(root, f"rank-{rank}-fatal.json", {
                "rank": rank, "exit_code": 86, "time": 10.1, "reason": reason,
            })
            self._write(root, f"rank-{rank}-shutdown.json", {
                "rank": rank, "time": 9.95, "pending_future": True,
            })

    def test_accepts_each_scenario_only_with_both_rank_exit_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scenario in SCENARIOS:
                with self.subTest(scenario=scenario):
                    self._fixture(root, scenario)
                    validate_fault_evidence(root, scenario, {0: 86, 1: 86}, 4.0)

    def test_rejects_peer_killed_by_launcher_even_with_fatal_intent_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root, "c1_payload")
            with self.assertRaisesRegex(AssertionError, "actual returncode=-15"):
                validate_fault_evidence(root, "c1_payload", {0: 86, 1: -15}, 4.0)

    def test_rejects_missing_rank_fatal_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root, "c2_payload")
            (root / "rank-1-fatal.json").unlink()
            with self.assertRaisesRegex(AssertionError, "missing rank-1-fatal.json"):
                validate_fault_evidence(root, "c2_payload", {0: 86, 1: 86}, 4.0)

    def test_rejects_late_exit_and_wrong_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root, "final_gradient")
            self._write(root, "rank-1-fatal.json", {
                "rank": 1, "exit_code": 86, "time": 17.0, "reason": "Micro Pipeline fatal peer failure",
            })
            with self.assertRaisesRegex(AssertionError, "exceeds its deadline"):
                validate_fault_evidence(root, "final_gradient", {0: 86, 1: 86}, 4.0)
            self._fixture(root, "final_gradient")
            self._write(root, "peer_wait.json", {"rank": 1, "phase": "old_c1", "time": 9.9})
            with self.assertRaisesRegex(AssertionError, "wrong peer wait phase"):
                validate_fault_evidence(root, "final_gradient", {0: 86, 1: 86}, 4.0)

    def test_rejects_shutdown_without_pending_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root, "shutdown_pending")
            self._write(root, "rank-1-shutdown.json", {
                "rank": 1, "time": 9.95, "pending_future": False,
            })
            with self.assertRaisesRegex(AssertionError, "shutdown had no pending task"):
                validate_fault_evidence(root, "shutdown_pending", {0: 86, 1: 86}, 4.0)

    def test_g_entry_marker_requires_observed_pending_gpu_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root, "final_gradient")
            marker = json.loads((root / "peer_wait.json").read_text())
            marker["gpu_pending_at_marker"] = False
            self._write(root, "peer_wait.json", marker)
            with self.assertRaisesRegex(AssertionError, "no observed pending GPU work"):
                validate_fault_evidence(root, "final_gradient", {0: 86, 1: 86}, 4.0)


if __name__ == "__main__":
    unittest.main()
