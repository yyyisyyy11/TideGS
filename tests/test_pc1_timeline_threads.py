"""Evidence-based CPU thread roles and SVG subrows, without Nsight or CUDA."""

import importlib.util
import os
from pathlib import Path
import unittest
from xml.etree.ElementTree import fromstring


SHARED_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "experiment/timeline/scripts"
BUNDLED_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "tools/pc1_timeline"
SCRIPT_DIR = Path(os.environ.get("TIDE_TIMELINE_SCRIPTS", str(
    SHARED_SCRIPT_DIR if SHARED_SCRIPT_DIR.is_dir() else BUNDLED_SCRIPT_DIR)))


def load_script(name):
    path = SCRIPT_DIR / f"{name}.py"
    if not path.is_file():
        raise unittest.SkipTest("timeline scripts unavailable; set TIDE_TIMELINE_SCRIPTS")
    spec = importlib.util.spec_from_file_location(f"pc1_threads_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extractor = load_script("extract_superpod_comparison")
renderer = load_script("timeline_svg")


class PC1TimelineThreadsTest(unittest.TestCase):
    def fixture(self):
        pid = 123
        process = pid << 24
        names = ("main", "comm", "autograd", "ssd", "watchdog", "wait_only",
                 "enqueue_only", "bare_stage", "legacy_autograd", "legacy_comm",
                 "worker", "unlabelled", "accumulate")
        tids = {name: process | (pid + index) for index, name in enumerate(names)}
        begin, finish = 1_000_000_000, 2_000_000_000
        scopes = (
            ("main", "Tide:S1:MB0:C1:FutureWait|tid=123"),
            ("main", "Tide:S1:MB0:C1:Metadata"),
            ("main", "Tide:S1:MB0:FinalGrad:ProjectionBackward"),
            ("comm", "Tide:S1:MB0:C1:Metadata|schedule=pc1-window-v2"),
            ("comm", "Tide:S1:MB0:C2:Payload|schedule=pc1-window-v2"),
            ("autograd", "Tide:S1:MB0:IntermediateGrad:RasterizeBackward|rank=0"),
            ("autograd", "Tide:S1:MB0:AccumulateGrad:torch::autograd::AccumulateGrad"),
            ("ssd", extractor.READ),
            ("watchdog", "ProcessGroupNCCL watchdog"),
            ("wait_only", "Tide:S1:MB0:C1:FutureWait"),
            ("wait_only", "Tide:S1:MB0:C2:Wait"),
            ("enqueue_only", "Tide:S1:MB0:C1:Enqueue"),
            ("bare_stage", "Tide:S1:MB0:FinalGrad"),
            ("legacy_autograd", "Tide:S1:MB0:C2:AlltoAllBackward"),
            ("legacy_comm", "Tide:S2:MB0:C1:Payload"),
            ("worker", "Tide:S1:MB0:C2:Worker"),
            ("unlabelled", "tide.cpu.materialize"),
            ("accumulate", "Tide:S2:MB0:AccumulateGrad"),
        )
        nvtx = [extractor.event_row(
            begin + index * 2_000_000, begin + (index + 1) * 2_000_000,
            "CPU", "nvtx", label, tid=tids[name], rowid=index + 1,
        ) for index, (name, label) in enumerate(scopes)]
        reads, _ = extractor.select_ssd_services(nvtx, tids["main"])
        for row in reads:
            row.update(read_kind="prefetch", request_kind="prefetch", detail="SSD prefetch worker")
        waits = [extractor.event_row(
            begin + 50_000_000, begin + 60_000_000, "CPU", "os_wait",
            "pthread_cond_timedwait", tid=tid, rowid=100 + index,
        ) for index, tid in enumerate(tids.values())]
        rank_data = {
            "rank": 0, "pid": pid, "global_pid": process, "main_tid": tids["main"],
            "nvtx": nvtx, "cpu_fragments": extractor.exclusive_cpu_phases(nvtx),
            "waits": waits, "reads": reads, "device_ids": [0],
            "gpu": [extractor.event_row(begin, begin + 20_000_000, "GPU", "compute",
                                         "actual kernel", device=0, stream=1)],
            "ssd_clock_calibration": {}, "provenance": {}, "diagnostics": [], "warnings": [],
        }
        records = [{"batch": 2, "iteration": 33, "start_ns": begin,
                    "end_ns": finish, "cycle_seconds": 1.0}]
        return rank_data, records, tids

    def test_roles_require_nvtx_evidence_and_main_identity_takes_priority(self):
        rank_data, records, tids = self.fixture()
        timeline = extractor.build_timeline(rank_data, records, "pc1", 2, 1)
        expected = {tids[name]: role for name, role in {
            "main": "main", "comm": "comm", "autograd": "autograd",
            "ssd": "unknown", "watchdog": "unknown", "wait_only": "unknown",
            "enqueue_only": "unknown", "bare_stage": "unknown", "unlabelled": "unknown",
            "legacy_autograd": "autograd", "legacy_comm": "comm", "worker": "comm",
            "accumulate": "autograd",
        }.items()}
        self.assertEqual({row["global_tid"]: row["thread_role"] for row in timeline["cpu_waits"]},
                         expected)
        for row in timeline["cpu_phases"]:
            self.assertEqual(row["thread_role"], expected[row["global_tid"]])
        self.assertTrue(any(row["global_tid"] == tids["legacy_autograd"]
                            for row in timeline["cpu_phases"]))
        for key in ("cpu_phases", "cpu_waits", "ssd_reads"):
            for row in timeline[key]:
                self.assertEqual(row["thread_id"], row["global_tid"] & 0xFFFFFF)
        self.assertEqual(timeline["ssd_reads"][0]["global_tid"], tids["ssd"])
        self.assertEqual(timeline["ssd_reads"][0]["read_kind"], "prefetch")

    def test_nonmain_threads_without_evidence_remain_unknown(self):
        rank_data, records, tids = self.fixture()
        rank_data.update(nvtx=[], cpu_fragments=[])
        timeline = extractor.build_timeline(rank_data, records, "off", 2, 1)
        for row in timeline["cpu_waits"]:
            self.assertEqual(row["thread_role"], "main" if row["global_tid"] == tids["main"]
                             else "unknown")

    def test_svg_reserves_distinct_unknown_row_inside_cpu_lane(self):
        rank_data, records, _ = self.fixture()
        timeline = extractor.build_timeline(rank_data, records, "pc1", 2, 1)
        svg, height = renderer.render_static(timeline)
        root = fromstring(f"<svg>{svg}</svg>")
        by_role = {}
        for node in root.findall("rect"):
            role = node.get("data-thread-role")
            if role is not None:
                by_role.setdefault(role, set()).add(float(node.get("y")))
                self.assertGreaterEqual(float(node.get("y")), 148)
                self.assertLessEqual(float(node.get("y")) + float(node.get("height")), 226)
        self.assertEqual(set(by_role), {"main", "comm", "autograd", "unknown"})
        self.assertTrue(by_role["unknown"].isdisjoint(by_role["autograd"]))
        self.assertEqual(len(set().union(*by_role.values())), 8)
        ssd = [node for node in root.findall("rect") if node.get("data-kind") == "ssd"]
        self.assertTrue(ssd)
        self.assertTrue(all(float(node.get("y")) >= 230 for node in ssd))
        self.assertEqual(height, 372)

    def test_svg_unrecognized_or_missing_roles_use_unknown_row(self):
        rank_data, records, _ = self.fixture()
        timeline = extractor.build_timeline(rank_data, records, "pc1", 2, 1)
        phase = timeline["cpu_phases"][0]
        phase.pop("thread_role")
        phase["thread_id"] = 987
        wait = timeline["cpu_waits"][0]
        wait["thread_role"] = "worker"
        wait["thread_id"] = 988
        svg, _ = renderer.render_static(timeline)
        root = fromstring(f"<svg>{svg}</svg>")
        for thread_id, kind in ((987, "phase"), (988, "wait")):
            node = next(node for node in root.findall("rect")
                        if node.get("data-kind") == kind
                        and f"thread {thread_id}" in node.get("data-tooltip", ""))
            self.assertEqual(node.get("data-thread-role"), "unknown")
            self.assertIn(f"unknown thread {thread_id}", node.get("data-tooltip"))


if __name__ == "__main__":
    unittest.main()
