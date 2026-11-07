"""Torch-free checks for disposable rank fault evidence."""

import json
from pathlib import Path


SCENARIOS = {
    "before_c1": "c1_metadata_gpu_wait",
    "c1_metadata": "c1_metadata_gpu_wait",
    "c1_payload": "c1_payload_gpu_wait",
    "before_c2": "c2_payload_gpu_wait",
    "c2_payload": "c2_payload_gpu_wait",
    "final_gradient": "final_gradient_entry_gpu_pending",
    "control_channel": "control_channel_armed",
    "shutdown_pending": "shutdown_pending_gpu_wait",
}


def validate_fault_evidence(result_dir, scenario, returncodes, stage_timeout):
    """A launcher kill, missing injection, or merely intended exit cannot pass."""
    root = Path(result_dir)

    def require(condition, message):
        if not condition:
            raise AssertionError(f"{scenario}: {message}")

    def read(name):
        path = root / name
        require(path.is_file(), f"missing {name}; launcher kill is insufficient")
        return json.loads(path.read_text())

    fault = read("fault.json")
    require(fault["scenario"] == scenario and fault["rank"] == 0, "wrong injection record")
    peer = read("peer_wait.json")
    require(peer["rank"] == 1 and peer["phase"] == SCENARIOS[scenario], "wrong peer wait phase")
    require(peer["time"] <= fault["time"], "fault occurred before peer entered the tested phase")
    if scenario != "control_channel":
        require(peer.get("gpu_work_submitted") is True, "peer never submitted real GPU work")
    if scenario == "final_gradient":
        require(peer.get("gpu_pending_at_marker") is True,
                "G-stage marker has no observed pending GPU work")
        require(peer.get("native_backward_entered") is False,
                "G-stage marker must describe entry, before invoking native backward")
    if scenario == "shutdown_pending":
        for rank in range(2):
            closing = read(f"rank-{rank}-shutdown.json")
            require(closing["pending_future"] is True, f"rank {rank} shutdown had no pending task")
            require(closing["time"] <= fault["time"], f"rank {rank} never entered shutdown before failure")
    for rank in range(2):
        require(returncodes.get(rank) == 86,
                f"rank {rank} actual returncode={returncodes.get(rank)!r}, expected its own exit 86")
        record = read(f"rank-{rank}-fatal.json")
        require(record["rank"] == rank and record["exit_code"] == 86, f"rank {rank} invalid fatal record")
        require("Micro Pipeline fatal" in record["reason"], f"rank {rank} did not use the fatal controller")
        if scenario == "control_channel":
            require("control channel failed" in record["reason"], f"rank {rank} missed store failure")
        elif rank == 0:
            require(f"RuntimeError: injected rank-0 {scenario} failure" in record["reason"],
                    "rank 0 did not report the injected failure")
        elapsed = record["time"] - fault["time"]
        require(0 <= elapsed <= stage_timeout + 2.0,
                f"rank {rank} fatal exit after {elapsed:.3f}s exceeds its deadline")
    return {"fault": fault, "peer": peer}
