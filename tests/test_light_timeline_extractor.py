"""The bundled extractor accepts light-profile records from four ranks."""

import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest


EXTRACTOR = (
    Path(__file__).resolve().parents[1]
    / "tools/micro_pipeline_timeline/extract_superpod_comparison.py"
)
STEPS = 13
GLOBAL_BSZ = 64


def _nvtx(connection, start, end, thread, name):
    connection.execute(
        "INSERT INTO NVTX_EVENTS(start,end,globalTid,text,textId) VALUES(?,?,?,?,NULL)",
        (start, end, thread, name),
    )


def _rank_fixture(run_dir, rank):
    pid = 1000 + rank
    global_pid = pid << 24
    main_thread = global_pid | pid
    worker_thread = global_pid | (pid + 100)
    rank_dir = run_dir / "train" / (f"rank_{rank}" if rank else "")
    rank_dir.mkdir(parents=True, exist_ok=True)
    report_dir = run_dir / "nsys"
    report_dir.mkdir(exist_ok=True)
    (report_dir / f"rank{rank}.nsys-rep").write_bytes(b"synthetic report fixture")
    sqlite_path = report_dir / f"rank{rank}.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.executescript("""
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            CREATE TABLE PROCESSES(globalPid INTEGER, pid INTEGER);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER,
                                     text TEXT, textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                start INTEGER, end INTEGER, globalPid INTEGER, deviceId INTEGER,
                streamId INTEGER, shortName INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
                start INTEGER, end INTEGER, globalPid INTEGER, deviceId INTEGER,
                streamId INTEGER, copyKind INTEGER, bytes INTEGER);
        """)
        connection.executemany(
            "INSERT INTO StringIds VALUES(?,?)",
            [(1, "projection_kernel"), (2, "ncclKernel_SendRecv")],
        )
        connection.execute("INSERT INTO PROCESSES VALUES(?,?)", (global_pid, pid))
        for batch in range(1, STEPS + 1):
            base = 100_000_000_000 + batch * 1_000_000_000
            _nvtx(connection, base, base + 900_000_000, main_thread, "tide.batch.step")
            _nvtx(connection, base + 10_000_000, base + 20_000_000,
                  main_thread, "tide.cpu.camera.schedule")
            _nvtx(connection, base + 100_000_000, base + 750_000_000,
                  main_thread, "tide.batch.compute")
            _nvtx(connection, base + 120_000_000, base + 180_000_000,
                  main_thread, f"Tide:S1:MB0:Projection|rank={rank},batch={batch},slot=0")
            _nvtx(connection, base + 650_000_000, base + 700_000_000,
                  main_thread, "Tide:S1:Optimizer:adam")
            _nvtx(connection, base + 710_000_000, base + 720_000_000,
                  main_thread, "tide.cpu.gradient.sparse_prepare")
            _nvtx(connection, base + 800_000_000, base + 820_000_000,
                  main_thread, "tide.cpu.plan.finalize.total")
            if rank == 0:
                _nvtx(connection, base + 805_000_000, base + 815_000_000,
                      main_thread, "tide.cpu.plan.active_refresh")
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?)",
                (base + 150_000_000, base + 400_000_000, global_pid, rank, 1, 1),
            )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?)",
                (base + 300_000_000, base + 350_000_000, global_pid, rank, 2, 2),
            )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?,?,?,?,?)",
                (base + 110_000_000, base + 130_000_000, global_pid, rank, 3, 1, 4096),
            )
            if batch == 10:
                _nvtx(connection, base + 280_000_000, base + 380_000_000,
                      main_thread, "tide.cpu.resident.read_batch")
                _nvtx(connection, base + 310_000_000, base + 320_000_000,
                      main_thread, "tide.io.ssd.read.service")
                _nvtx(connection, base + 300_000_000, base + 350_000_000,
                      worker_thread, "tide.io.ssd.read.service")
    connection.close()

    metrics_path = rank_dir / f"metrics_batch_rank{rank}.tsv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("iteration", "rank", "optimizer_step", "local_cameras",
                        "pipeline_mode", "intermediate_grad_ms"),
            delimiter="\t",
        )
        writer.writeheader()
        for batch in range(1, STEPS + 1):
            writer.writerow({"iteration": 1 + (batch - 1) * GLOBAL_BSZ,
                             "rank": rank, "optimizer_step": batch,
                             "local_cameras": 16, "pipeline_mode": "micro",
                             "intermediate_grad_ms": ""})

    def event(name, start, end, **fields):
        return {"name": name, "rank": rank, "pid": pid,
                "start_ns": start, "end_ns": end, "duration_ns": end - start,
                "timing_source": "host_monotonic", **fields}

    events = []
    for batch in range(1, STEPS + 1):
        base = 100_000_000_000 + batch * 1_000_000_000
        events.append(event("batch_identity", base, base,
                            iteration=1 + (batch - 1) * GLOBAL_BSZ))
        if batch == 10:
            events.extend((
                event("ssd_read_future", base + 290_000_000, base + 360_000_000,
                      thread_id=pid + 100, target_iteration=641, blocks=2, bytes=8192),
                event("prefetch_end_to_end", base + 270_000_000,
                      base + 370_000_000, thread_id=pid + 100,
                      target_iteration=641, blocks=2, status="cache_ready"),
            ))
    timeline_path = rank_dir / f"timeline_events_rank{rank}.jsonl"
    timeline_path.write_text(
        "".join(json.dumps(item) + "\n" for item in events), encoding="utf-8"
    )
    return sqlite_path


class LightTimelineExtractorTest(unittest.TestCase):
    def test_four_rank_light_capture_keeps_timeline_and_unknown_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            sqlites = [_rank_fixture(run_dir, rank) for rank in range(4)]
            (run_dir / "train" / "args.json").write_text(
                json.dumps({"tide_forward_pipeline": "micro", "bsz": GLOBAL_BSZ,
                            "iterations": 769, "tide_light_timeline": True,
                            "tide_detailed_metrics": False}),
                encoding="utf-8",
            )
            output = root / "derived"
            result = subprocess.run(
                [sys.executable, "-B", str(EXTRACTOR), "--run-dir", str(run_dir),
                 "--mode", "micro", "--world-size", "4", "--global-bsz", "64",
                 "--expected-steps", str(STEPS), "--sqlite",
                 *(str(path) for path in sqlites), "--batch-start", "10",
                 "--batch-count", "2", "--timeline-ranks", "0", "1", "2", "3",
                 "--batch-starts", "10", "--output-dir", str(output)],
                text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["recording_mode"], "light")
            self.assertEqual(summary["record_count"], 4 * STEPS)
            self.assertEqual([item["pid"] for item in summary["ranks"]],
                             [1000, 1001, 1002, 1003])
            self.assertEqual(len(summary["timeline_windows"]), 4)
            for rank in range(4):
                self.assertEqual(summary["ranks"][rank]["prefetch_request_count"], 1)
                timeline = json.loads((output / "windows" / f"rank-{rank}"
                                       / "B10-11" / "micro.timeline.data.json").read_text())
                self.assertEqual(timeline["metadata"]["recording_mode"], "light")
                self.assertEqual(timeline["metadata"]["pid"], 1000 + rank)
                self.assertEqual({item["kind"] for item in timeline["gpu_bins"]},
                                 {"compute", "nccl", "copy"})
                self.assertEqual({item["read_kind"] for item in timeline["ssd_reads"]},
                                 {"urgent", "prefetch"})
                self.assertEqual([item["number"] for item in timeline["batches"]],
                                 [10, 11])
                if rank == 0:
                    self.assertTrue(any(item["name"] == "Plan active refresh (host call)"
                                        for item in timeline["cpu_phases"]))
            with (output / "batch_metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["metrics_intermediate_grad_ms"], "")


if __name__ == "__main__":
    unittest.main()
