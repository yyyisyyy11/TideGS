"""Behavioral checks of the production PC1 window executor (no torch)."""

from contextlib import contextmanager
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref

from strategies.tide_engine.pc1_schedule import run_pc1_windows


class _Controller:
    stage_timeout = 10.0

    def __init__(self):
        self.operations = []
        self.active = False

    @contextmanager
    def operation(self, phase):
        self.operations.append(phase)
        self.active = True
        try:
            yield
        finally:
            self.active = False


class _Handle:
    pass


class PC1ScheduleTest(unittest.TestCase):
    def test_batch_entry_rejects_mismatched_rank_window_boundaries(self):
        path = Path(__file__).parents[1] / "strategies/tide_engine/distributed_engine.py"
        parsed = ast.parse(path.read_text())
        functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"_validate_distributed_batch_contract",
                                       "_camera_global_ids", "_error_description"}]
        module = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *functions,
        ], type_ignores=[])
        namespace = {"resolve_curvature_schedule": lambda **kwargs: (1, False)}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)

        def gather(value):
            peer = dict(value)
            peer["contract"] = dict(value["contract"], pipeline_buffer_count=3)
            return [value, peer]

        with self.assertRaisesRegex(RuntimeError, "pipeline schedule"):
            namespace["_validate_distributed_batch_contract"](
                args=SimpleNamespace(bsz=4, tide_forward_pipeline="pc1",
                                     tide_camera_microbatch=1, tide_pipeline_buffer_count=2),
                batched_cameras=[SimpleNamespace(global_idx=0), SimpleNamespace(global_idx=1)],
                curvature_cameras=[], optimizer_step=1,
                plan=SimpleNamespace(
                    iteration=1, rank_s1_camera_ids={0: [0, 1]}, rank_s2_camera_ids={0: []},
                    global_s1_camera_ids=[0, 1], global_s2_camera_ids=[],
                ),
                context=SimpleNamespace(rank=0, all_gather_object=gather),
            )

    def _execute(self, count, buffers):
        controller = _Controller()
        calls, active, deadlines, weak_handles = [], {}, [], []

        def project(index, *, deadline):
            self.assertTrue(controller.active)
            # Every previous window must finish G before any slot is reused.
            slot = index % buffers
            self.assertNotIn(slot, active)
            handle = _Handle()
            handle.index = index
            active[slot] = handle
            weak_handles.append(weakref.ref(handle))
            self.assertLessEqual(len(active), buffers)
            calls.extend([("P", index), ("C1", index)])
            deadlines.append(deadline)
            return handle

        def image(index, handle, *, deadline):
            self.assertTrue(controller.active)
            self.assertEqual(index, handle.index)
            calls.extend([("M", index), ("C2", index)])
            deadlines.append(deadline)

        def final(index, handle, *, deadline):
            self.assertTrue(controller.active)
            self.assertEqual(active.pop(index % buffers), handle)
            calls.append(("G", index))
            deadlines.append(deadline)

        run_pc1_windows(
            range(count), buffer_count=buffers, controller=controller,
            project=project, image_backward=image, final_backward=final,
            clock=lambda: 100.0,
        )
        expected = []
        for start in range(0, count, buffers):
            window = list(range(start, min(count, start + buffers)))
            expected.extend((stage, index) for index in window for stage in ("P", "C1"))
            expected.extend((stage, index) for index in window for stage in ("M", "C2"))
            expected.extend(("G", index) for index in window)
        self.assertEqual(calls, expected)
        self.assertFalse(active)
        self.assertTrue(all(value == 110.0 for value in deadlines))
        self.assertTrue(all(ref() is None for ref in weak_handles))
        self.assertEqual(len(controller.operations), (count + buffers - 1) // buffers)

    def test_one_through_four_microbatches_and_tail_window(self):
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                self._execute(count, 2)

    def test_arbitrary_fixed_window_and_serial_reference(self):
        self._execute(7, 3)
        self._execute(4, 1)
        self._execute(0, 2)

    def test_one_deadline_includes_python_work_between_stages(self):
        controller = _Controller()
        now, calls = [0.0], []

        def project(index, *, deadline):
            calls.append(("P", index, deadline))
            now[0] += 3.0
            return index

        def image(index, handle, *, deadline):
            calls.append(("M", index, deadline))
            now[0] += 5.0

        with self.assertRaisesRegex(TimeoutError, "window 0 deadline"):
            run_pc1_windows(
                range(2), buffer_count=2, controller=controller,
                project=project, image_backward=image,
                final_backward=lambda *args, **kwargs: self.fail("G after timeout"),
                clock=lambda: now[0],
            )
        self.assertEqual(calls, [("P", 0, 10.0), ("P", 1, 10.0), ("M", 0, 10.0)])
        self.assertFalse(controller.active)

    def test_exception_stops_window_without_later_final_gradient(self):
        controller = _Controller()

        def broken_image(*args, **kwargs):
            raise RuntimeError("injected image failure")

        with self.assertRaisesRegex(RuntimeError, "injected image failure"):
            run_pc1_windows(
                range(4), buffer_count=2, controller=controller,
                project=lambda index, **kwargs: index,
                image_backward=broken_image,
                final_backward=lambda *args, **kwargs: self.fail("unexpected G"),
            )
        self.assertEqual(controller.operations, ["PC1 window 0"])
        self.assertFalse(controller.active)


if __name__ == "__main__":
    unittest.main()
