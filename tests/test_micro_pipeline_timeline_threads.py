"""Evidence-based CPU thread roles and SVG subrows, without Nsight or CUDA."""

import csv
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from xml.etree.ElementTree import fromstring


SHARED_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "experiment/timeline/scripts"
BUNDLED_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "tools/micro_pipeline_timeline"
SCRIPT_DIR = Path(os.environ.get("TIDE_TIMELINE_SCRIPTS", str(
    SHARED_SCRIPT_DIR if SHARED_SCRIPT_DIR.is_dir() else BUNDLED_SCRIPT_DIR)))


def load_script(name):
    path = SCRIPT_DIR / f"{name}.py"
    if not path.is_file():
        raise unittest.SkipTest("timeline scripts unavailable; set TIDE_TIMELINE_SCRIPTS")
    spec = importlib.util.spec_from_file_location(f"micro_pipeline_threads_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extractor = load_script("extract_superpod_comparison")
renderer = load_script("timeline_svg")


class MicroPipelineTimelineThreadsTest(unittest.TestCase):
    def test_old_and_new_foreground_wrappers_do_not_count_as_ssd_service(self):
        main_tid = (123 << 24) | 123
        worker_tid = (123 << 24) | 456
        def scope(start, end, name, tid, rowid):
            return extractor.event_row(start, end, "CPU", "nvtx", name,
                                       tid=tid, rowid=rowid)
        nvtx = [
            scope(0, 100, extractor.READ, main_tid, 1),
            scope(20, 50, extractor.READ, main_tid, 2),
            scope(110, 210, extractor.READ_BATCH, main_tid, 3),
            scope(130, 160, extractor.READ, main_tid, 4),
            scope(220, 250, extractor.READ, main_tid, 5),
            scope(300, 330, extractor.READ, worker_tid, 6),
        ]
        reads, rejected = extractor.select_ssd_services(nvtx, main_tid)
        self.assertEqual({row["source_rowid"] for row in reads}, {2, 4, 6})
        self.assertEqual({row["source_rowid"]: row["reason"] for row in rejected}, {
            1: "enclosing_read_service_wrapper",
            3: "explicit_read_batch_wrapper",
            5: "unverified_main_thread_read_batch_wrapper",
        })
        self.assertEqual({row["source_rowid"]: row["thread_kind"] for row in reads}, {
            2: "main_nested_storage", 4: "main_nested_storage",
            6: "background_storage",
        })

    def test_bubble_labels_render_as_host_scopes(self):
        labels = (
            "tide.cpu.plan.preview.total", "tide.cpu.plan.finalize.total",
            "tide.cpu.camera.prefetch.submit", "tide.cpu.camera.h2d.submit",
            "tide.comm.plan.broadcast", "tide.io.ssd.prefetch.submit",
            "tide.cpu.resident.read_batch", "tide.cpu.resident.cache_fetch",
            "tide.cpu.resident.pack", "tide.cpu.resident.unpack.submit",
            "tide.cpu.resident.evict", "tide.cpu.resident.evict.stage",
            "tide.cpu.resident.evict.submit", "tide.cpu.resident.load",
            "tide.cpu.resident.bind", "tide.cpu.resident.active_leaves",
            "tide.comm.micro.layout", "tide.cpu.bounds.gpu_submit",
            "tide.wait.bounds.d2h", "tide.comm.bounds.gather",
            "tide.cpu.metrics.cuda_timing", "tide.cpu.metrics.timeline_emit",
            "tide.comm.metrics.gather", "tide.comm.batch.barrier",
            "tide.comm.loss.gather", "Tide:S1:Optimizer:adam",
            "Tide:S1:Optimizer:3dgs2_tr:Prepare",
            "Tide:S1:Optimizer:3dgs2_tr:Commit",
        )
        for label in labels:
            with self.subTest(label=label):
                self.assertIsNotNone(extractor.phase_name(label))

    def test_nested_bubble_scopes_have_exclusive_fragments(self):
        main_tid = (123 << 24) | 123
        scopes = (
            (0, 100, "tide.cpu.resident.load"),
            (20, 70, "tide.cpu.resident.read_batch"),
            (30, 45, "tide.cpu.resident.cache_fetch"),
            (45, 65, "tide.cpu.resident.pack"),
        )
        events = [extractor.event_row(start, end, "CPU", "nvtx", label,
                                      tid=main_tid, rowid=index)
                  for index, (start, end, label) in enumerate(scopes, 1)]
        fragments = extractor.exclusive_cpu_phases(events)
        totals = {}
        for fragment in fragments:
            totals[fragment["name"]] = totals.get(fragment["name"], 0) + (
                fragment["end_ns"] - fragment["start_ns"])
        self.assertEqual(totals, {
            "tide.cpu.resident.load": 50,
            "tide.cpu.resident.read_batch": 15,
            "tide.cpu.resident.cache_fetch": 15,
            "tide.cpu.resident.pack": 20,
        })

    def host_gap_fixture(self, mode):
        main_tid = (123 << 24) | 123
        worker_tid = (123 << 24) | 456
        def scope(start, end, name, rowid, tid=main_tid):
            return extractor.event_row(start, end, "CPU", "nvtx", name,
                                       tid=tid, rowid=rowid)
        schedules = [scope(base + 100, base + 110, extractor.SCHEDULE, 100 + batch)
                     for batch, base in enumerate((0, 1000, 2000), 1)]
        steps = [scope(base, base + 980, extractor.STEP, 200 + batch)
                 for batch, base in enumerate((0, 1000, 2000), 1)]
        nvtx = [*steps, *schedules]
        for batch, base in enumerate((0, 1000, 2000), 1):
            projection_name = ("tide.cpu.projection.submit" if mode == "off"
                               else f"Tide:S1:MB0:Projection|rank=0,batch={batch}")
            nvtx += [scope(base + 150, base + 400, "tide.cpu.resident.load", 300 + batch),
                     scope(base + 200, base + 250, extractor.READ_BATCH, 400 + batch),
                     scope(base + 210, base + 230, "tide.cpu.resident.cache_fetch", 500 + batch),
                     scope(base + 230, base + 245, "tide.cpu.resident.pack", 600 + batch),
                     scope(base + 500, base + 550, projection_name, 700 + batch),
                     scope(base + 700, base + 750, "tide.cpu.gradient.sparse_prepare", 800 + batch),
                     scope(base + 750, base + 760, "Tide:S1:Optimizer:adam", 900 + batch),
                     scope(base + 760, base + 780, "tide.cpu.writeback.mark_dirty", 1000 + batch)]
            if mode == "off":
                nvtx.append(scope(base + 560, base + 570, projection_name, 1100 + batch))
            if batch < 3:
                nvtx.append(scope(base + 800, base + 850, "tide.cpu.plan.finalize.total", 1200 + batch))
        nvtx.append(scope(125, 350, "tide.cpu.resident.load", 1400, tid=worker_tid))
        return {"rank": 0, "main_tid": main_tid, "main_nvtx": [row for row in nvtx
                if row["global_tid"] == main_tid], "cpu_fragments": extractor.exclusive_cpu_phases(nvtx),
                "rejected_reads": [], "schedules": schedules, "steps": steps,
                "gpu": [extractor.event_row(720, 780, "GPU", "compute", "kernel", rowid=1)]}

    def test_host_gap_boundaries_and_exclusive_accounting(self):
        for mode in ("off", "micro"):
            with self.subTest(mode=mode):
                rank_data = self.host_gap_fixture(mode)
                gaps = extractor.host_gap_records(rank_data, mode)
                self.assertEqual(len(gaps), 12)
                pre = next(row for row in gaps if row["batch"] == 1 and row["gap"] == "pre_projection")
                self.assertEqual((pre["start_ns"], pre["end_ns"]), (100, 500))
                self.assertEqual(pre["stage_exclusive_seconds"]["cache_to_gpu"], 50 / extractor.NS)
                self.assertEqual(pre["unclassified_seconds"], 340 / extractor.NS)
                post = next(row for row in gaps if row["batch"] == 1 and row["gap"] == "post_backward_host_tail")
                self.assertEqual((post["start_ns"], post["end_ns"]), (700, 800))
                self.assertEqual(post["stage_exclusive_seconds"]["update_tail"], 80 / extractor.NS)
                self.assertEqual(post["gpu_active_union_seconds"], 60 / extractor.NS)
                transition = next(row for row in gaps if row["batch"] == 2 and row["gap"] == "prior_finalize_to_schedule")
                self.assertEqual((transition["start_ns"], transition["end_ns"]), (850, 1100))
                forward = next(row for row in gaps if row["batch"] == 1 and row["gap"] == "finalize_to_next_projection")
                self.assertEqual((forward["start_ns"], forward["end_ns"]), (800, 1500))
                summary = extractor.host_gap_aggregate(gaps, 1, 2)
                self.assertEqual(summary["post_backward_host_tail"]["available_count"], 2)
                self.assertAlmostEqual(summary["post_backward_host_tail"]["duration"]["mean_seconds"],
                                       100 / extractor.NS)

    def test_host_gap_duplicate_sparse_boundary_is_rejected(self):
        rank_data = self.host_gap_fixture("micro")
        rank_data["main_nvtx"].append(extractor.event_row(
            780, 790, "CPU", "nvtx", "tide.cpu.gradient.sparse_prepare",
            tid=rank_data["main_tid"], rowid=2000))
        with self.assertRaisesRegex(ValueError, "expected one sparse_prepare"):
            extractor.host_gap_records(rank_data, "micro")

    def test_reused_h2d_label_uses_camera_or_resident_context(self):
        main_tid = (123 << 24) | 123
        def scope(start, end, label, rowid):
            return extractor.event_row(start, end, "CPU", "nvtx", label,
                                       tid=main_tid, rowid=rowid)
        scopes = [
            scope(0, 100, "tide.gpu.transfer.h2d", 1),
            scope(10, 90, "tide.cpu.camera.h2d.submit", 2),
            scope(100, 200, "tide.cpu.resident.load", 3),
            scope(110, 190, "tide.cpu.resident.h2d.submit", 4),
            scope(120, 180, "tide.gpu.transfer.h2d", 5),
            scope(200, 220, "tide.gpu.transfer.h2d", 6),
        ]
        rank_data = {"rank": 0, "main_tid": main_tid, "main_nvtx": scopes,
                     "cpu_fragments": extractor.exclusive_cpu_phases(scopes),
                     "rejected_reads": []}
        start, end = scope(0, 1, "start", 10), scope(220, 221, "end", 11)
        gap = extractor.host_gap_window(rank_data, 1, "off", "pre_projection",
                                        start, "start", end, "start", [])
        self.assertEqual(gap["stage_exclusive_seconds"]["camera_transfer"],
                         100 / extractor.NS)
        self.assertEqual(gap["stage_exclusive_seconds"]["cache_to_gpu"],
                         80 / extractor.NS)
        self.assertEqual(gap["unclassified_seconds"], 40 / extractor.NS)

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
            ("comm", "Tide:S1:MB0:C1:Metadata|schedule=micro-pipeline-window-v2"),
            ("comm", "Tide:S1:MB0:C2:Payload|schedule=micro-pipeline-window-v2"),
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
        timeline = extractor.build_timeline(rank_data, records, "micro", 2, 1)
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

    def test_unverified_legacy_foreground_read_is_not_labeled_storage_service(self):
        rank_data, records, tids = self.fixture()
        wrapper = extractor.event_row(1_100_000_000, 1_110_000_000, "CPU", "nvtx",
                                      extractor.READ, tid=tids["main"], rowid=2000)
        rank_data["nvtx"].append(wrapper)
        rank_data["cpu_fragments"] = extractor.exclusive_cpu_phases(rank_data["nvtx"])
        rank_data["rejected_reads"] = [{**wrapper,
                                        "reason": "unverified_main_thread_read_batch_wrapper"}]
        timeline = extractor.build_timeline(rank_data, records, "micro", 2, 1)
        wrapper_phases = [row for row in timeline["cpu_phases"]
                          if row["source_rowid"] == wrapper["source_rowid"]]
        self.assertTrue(wrapper_phases)
        self.assertEqual({row["name"] for row in wrapper_phases},
                         {"Gaussian cache fetch / pack (legacy wrapper)"})

    def test_legacy_timeline_mode_renders_current_label(self):
        rank_data, records, _ = self.fixture()
        timeline = extractor.build_timeline(rank_data, records, "pc1", 2, 1)
        self.assertEqual(timeline["metadata"]["mode"], "micro")
        timeline["metadata"]["mode"] = "pc1"
        self.assertTrue(renderer.display_labels(timeline)["compact_title"].startswith("MICRO ·"))

    def test_legacy_metric_modes_keep_rank_and_iteration_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.tsv"
            rows = [dict(rank=0, optimizer_step=batch, iteration=1 + (batch - 1) * 32,
                         local_cameras=16, pipeline_mode="pc1") for batch in range(1, 95)]
            def write():
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
                    writer.writeheader()
                    writer.writerows(rows)
            write()
            self.assertEqual(len(extractor.read_metrics(path, 0, "micro")), 94)
            rows[0]["rank"] = 1
            write()
            with self.assertRaisesRegex(ValueError, "rank/optimizer clock mismatch"):
                extractor.read_metrics(path, 0, "micro")
            rows[0]["rank"] = 0
            rows[1]["iteration"] = 1
            write()
            with self.assertRaisesRegex(ValueError, "not a from-scratch"):
                extractor.read_metrics(path, 0, "micro")

    def test_nonmain_threads_without_evidence_remain_unknown(self):
        rank_data, records, tids = self.fixture()
        rank_data.update(nvtx=[], cpu_fragments=[])
        timeline = extractor.build_timeline(rank_data, records, "off", 2, 1)
        for row in timeline["cpu_waits"]:
            self.assertEqual(row["thread_role"], "main" if row["global_tid"] == tids["main"]
                             else "unknown")

    def test_svg_reserves_distinct_unknown_row_inside_cpu_lane(self):
        rank_data, records, _ = self.fixture()
        timeline = extractor.build_timeline(rank_data, records, "micro", 2, 1)
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
        timeline = extractor.build_timeline(rank_data, records, "micro", 2, 1)
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
