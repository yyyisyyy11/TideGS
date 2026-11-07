"""Exercise the real CPU planner and training finalize helper without CUDA/torch."""

import ast
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from strategies.tide_engine.distributed_plan import (
    DistributedBatchPlan,
    DistributedBatchPlanner,
    DistributedPlannerState,
    assign_cameras_balanced,
)
from tests.test_micro_pipeline_mode import arguments_parser


ROOT = Path(__file__).resolve().parents[1]
CAMERAS = list(range(8))


def _blocks(values):
    return {camera: list(values) if camera == 0 else [] for camera in CAMERAS}


def _planner(*, capacity=6, assignment="equal", policy="topc_strict_active_first"):
    planner = DistributedBatchPlanner(
        block_owner=[0, 1, 2, 3, 0, 1, 2, 3],
        total_points=29, block_size=4, world_size=4,
        resident_capacity_blocks=capacity, resident_lambda=0.3,
        resident_recency_decay=0.95, balanced_seed_fraction=1.0,
        camera_assignment=assignment, resident_selection_policy=policy,
    )
    planner.restore_state(DistributedPlannerState(
        resident=(0, 1, 2, 3, 6, 7), active=(0, 1, 2),
        recency={0: 2.0, 1: 1.0, 7: 0.5},
    ))
    return planner


class _Context:
    enabled = True
    is_rank0 = True
    rank = 0

    def __init__(self):
        self.statuses = []
        self.broadcasts = []

    def all_gather_object(self, value):
        self.statuses.append(value)
        return [value] + [{"rank": rank, "error": None} for rank in range(1, 4)]

    def broadcast_object(self, value):
        self.broadcasts.append(value)
        return value


class OptionalFinalizePlanTest(unittest.TestCase):
    def setUp(self):
        path = ROOT / "train_tidegs.py"
        names = {"_normalize_camera_blocks", "_split_camera_blocks", "_union_camera_ids",
                 "_run_synchronized_training_phase", "_finalize_distributed_plan"}
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if (
            isinstance(node, ast.FunctionDef) and node.name in names
        ) or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "CULL_METRIC_FIELDS"
                    for target in node.targets)
        )]
        self.ranges = []
        self.open_ranges = []

        @contextmanager
        def tide_range(name):
            self.ranges.append(name)
            self.open_ranges.append(name)
            try:
                yield
            finally:
                self.assertEqual(self.open_ranges.pop(), name)

        self.schedule = SimpleNamespace(batch_indices=CAMERAS, epoch=1)
        namespace = dict(time=time, tide_range=tide_range,
                         DistributedBatchPlan=DistributedBatchPlan,
                         get_camera_batch_schedule=Mock(return_value=self.schedule))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        self.finalize = namespace["_finalize_distributed_plan"]
        self.normalize = namespace["_normalize_camera_blocks"]
        self.context = _Context()

    def tearDown(self):
        self.assertEqual(self.open_ranges, [])

    def _preview(self, planner, blocks, s2=None, *, iteration=65):
        predicted, state = planner.preview(
            iteration=iteration, epoch=1, camera_ids=CAMERAS, camera_blocks=blocks,
            curvature_camera_ids=(sorted(s2) if s2 is not None else None),
            curvature_camera_blocks=s2,
        )
        union = {**blocks, **(s2 or {})}
        return predicted, dict(bounds_generation=1, predicted_state=state,
                               union_camera_blocks=self.normalize(union))

    def _finalize(self, planner, predicted, preview, blocks, *, requested=False,
                  generation=2, s2=None):
        union = {**blocks, **(s2 or {})}
        self.storage = SimpleNamespace(
            get_bounds_generation=Mock(return_value=generation),
            get_visible_blocks_batch=Mock(return_value=(generation, union)),
        )
        return self.finalize(
            context=self.context, planner=planner, storage_adapter=self.storage,
            training_schedule=object(), iteration=predicted.iteration, batch_size=8,
            schedule_ordering="trajectory", predicted_plan=predicted,
            rank0_preview=preview, finalize_plan=requested,
            curvature_camera_ids=(sorted(s2) if s2 is not None else None),
        )[0]

    def test_cli_default_and_explicit_full_finalize(self):
        parser, _ = arguments_parser()
        self.assertFalse(parser.parse_args([]).finalize_plan)
        self.assertTrue(parser.parse_args(["--finalize_plan"]).finalize_plan)

    def test_refresh_changes_active_assignment_and_rows_without_resident_selection(self):
        before = {camera: [0, 1] if camera == 0 else [2] for camera in CAMERAS}
        after = {0: [7], 1: [2], 2: [], 3: [2], 4: [7], 5: [], 6: [2], 7: []}
        s2 = {camera + 8: blocks for camera, blocks in _blocks([3]).items()}
        for assignment in ("equal", "gaussian_balanced"):
            for policy in ("topc_strict_active_first", "topc_balanced_active_first"):
                with self.subTest(assignment=assignment, policy=policy):
                    planner = _planner(assignment=assignment, policy=policy)
                    predicted, preview = self._preview(
                        planner, before, {camera + 8: blocks for camera, blocks in _blocks([1]).items()})
                    with patch("strategies.tide_engine.distributed_plan.compute_topc_resident_transition",
                               side_effect=AssertionError("resident selection on fast path")):
                        actual = self._finalize(planner, predicted, preview, after, s2=s2)
                    self.assertEqual(actual.global_active_blocks, [2, 3, 7])
                    self.assertEqual(actual.global_gradient_active_blocks, [2, 7])
                    self.assertEqual(actual.global_curvature_active_blocks, [3])
                    self.assertEqual(actual.rank_active_blocks, [[], [], [2], [3, 7]])
                    self.assertEqual(actual.rank_owner_active_rows, [0, 0, 4, 5])
                    self.assertEqual(actual.rank_participation_rows, [1, 1, 4, 5])
                    for field in ("rank_resident_blocks", "global_resident_blocks",
                                  "stream_in_blocks", "evict_blocks"):
                        self.assertEqual(getattr(actual, field), getattr(predicted, field))
                    if assignment == "gaussian_balanced":
                        expected = assign_cameras_balanced(
                            camera_ids=CAMERAS, camera_blocks=after,
                            block_rows=[4, 4, 4, 4, 4, 4, 4, 1], world_size=4)
                        self.assertEqual(actual.rank_camera_ids, expected)
                        self.assertNotEqual(actual.rank_camera_ids, predicted.rank_camera_ids)
                    else:
                        self.assertEqual(actual.rank_camera_ids, [[0, 1], [2, 3], [4, 5], [6, 7]])
                    self.assertEqual(actual.plan_resolution, "active_refresh")
                    self.assertEqual(actual.prediction_replanned, 0)
                    self.assertIsNone(actual.prediction_missing_blocks)
                    self.assertIsNone(actual.prediction_extra_blocks)
                    self.assertEqual(actual.prediction_exact_plan_ms, 0.0)
                    self.assertEqual(planner.snapshot_state(), replace(
                        preview["predicted_state"], active=(2, 3, 7)))
                    self.assertEqual(self.storage.get_visible_blocks_batch.call_count, 1)
                    self.assertIn("tide.cpu.plan.active_refresh", self.ranges)

    def test_unchanged_bounds_uses_cached_visibility(self):
        for requested in (False, True):
            with self.subTest(requested=requested):
                planner = _planner()
                predicted, preview = self._preview(planner, _blocks([0, 1]))
                with patch.object(planner, "plan", side_effect=AssertionError("unexpected replan")):
                    actual = self._finalize(planner, predicted, preview, _blocks([7]),
                                            requested=requested, generation=1)
                self.storage.get_visible_blocks_batch.assert_not_called()
                self.assertEqual(actual.global_active_blocks, [0, 1])
                self.assertEqual(planner.snapshot_state(), preview["predicted_state"])
                self.assertEqual(actual.prediction_replanned, 0)
                self.assertEqual(actual.plan_resolution,
                                 "full_requested" if requested else "active_refresh")
                self.assertEqual(actual.prediction_missing_blocks, 0 if requested else None)

    def test_all_fallbacks_plan_once_from_original_state(self):
        cases = [
            ("fallback_policy", "topc_balanced", 4, [0], [1]),
            ("fallback_preview_capacity", "topc_strict_active_first", 2, [0, 1, 2], [0]),
            ("fallback_active_capacity", "topc_strict_active_first", 2, [0], [0, 1, 2]),
            ("fallback_resident_coverage", "topc_strict_active_first", 4, [0], [0, 4]),
        ]
        for reason, policy, capacity, old, new in cases:
            with self.subTest(reason=reason):
                planner = _planner(capacity=capacity, policy=policy)
                baseline = planner.snapshot_state()
                predicted, preview = self._preview(planner, _blocks(old))
                reference = _planner(capacity=capacity, policy=policy)
                expected = reference.plan(iteration=65, epoch=1, camera_ids=CAMERAS,
                                          camera_blocks=_blocks(new))
                original_plan = planner.plan

                def plan_from_baseline(**kwargs):
                    self.assertEqual(planner.snapshot_state(), baseline)
                    return original_plan(**kwargs)

                with patch.object(planner, "plan", side_effect=plan_from_baseline) as full:
                    actual = self._finalize(planner, predicted, preview, _blocks(new))
                self.assertEqual(full.call_count, 1)
                self.assertEqual(actual.plan_resolution, reason)
                self.assertEqual(actual.prediction_replanned, 1)
                self.assertEqual(actual.prediction_missing_blocks,
                                 len(set(expected.stream_in_blocks) - set(predicted.stream_in_blocks)))
                self.assertEqual(actual.prediction_extra_blocks,
                                 len(set(predicted.stream_in_blocks) - set(expected.stream_in_blocks)))
                for field in ("rank_active_blocks", "rank_resident_blocks", "stream_in_blocks",
                              "evict_blocks", "rank_camera_ids", "rank_owner_active_rows"):
                    self.assertEqual(getattr(actual, field), getattr(expected, field))
                self.assertEqual(planner.snapshot_state(), reference.snapshot_state())
                self.assertEqual(self.storage.get_visible_blocks_batch.call_count, 1)

    def test_unchanged_visibility_after_recull_preserves_old_full_mode_shortcut(self):
        for requested in (False, True):
            with self.subTest(requested=requested):
                planner = _planner()
                predicted, preview = self._preview(planner, _blocks([0]))
                with patch.object(planner, "plan", side_effect=AssertionError("unexpected replan")):
                    actual = self._finalize(planner, predicted, preview, _blocks([0]),
                                            requested=requested)
                self.assertEqual(self.storage.get_visible_blocks_batch.call_count, 1)
                self.assertEqual(actual.prediction_replanned, 0)
                self.assertEqual(planner.snapshot_state(), preview["predicted_state"])

    def test_requested_full_finalize_matches_existing_plan_and_recency(self):
        planner = _planner()
        predicted, preview = self._preview(planner, _blocks([0]))
        reference = _planner()
        expected = reference.plan(iteration=65, epoch=1, camera_ids=CAMERAS,
                                  camera_blocks=_blocks([0, 4]))
        with patch.object(planner, "refresh_active", side_effect=AssertionError("fast path")):
            actual = self._finalize(planner, predicted, preview, _blocks([0, 4]), requested=True)
        self.assertEqual(actual.plan_resolution, "full_requested")
        self.assertEqual(actual.prediction_replanned, 1)
        self.assertEqual(actual.stream_in_blocks, expected.stream_in_blocks)
        self.assertEqual(actual.rank_active_blocks, expected.rank_active_blocks)
        self.assertEqual(planner.snapshot_state(), reference.snapshot_state())

    def test_refresh_rejection_and_construction_failure_do_not_commit_preview(self):
        planner = _planner()
        baseline = planner.snapshot_state()
        predicted, preview = self._preview(planner, _blocks([0]))
        with patch.object(planner, "_build_plan", side_effect=ValueError("bad active layout")):
            with self.assertRaisesRegex(RuntimeError, "final distributed planning.*bad active layout"):
                self._finalize(planner, predicted, preview, _blocks([1]))
        self.assertEqual(planner.snapshot_state(), baseline)
        self.assertEqual(self.context.broadcasts, [])
        self.assertIn("bad active layout", self.context.statuses[-1]["error"])
        rejected, reason = planner.refresh_active(
            predicted_plan=predicted, predicted_state=preview["predicted_state"],
            camera_blocks=_blocks([4]))
        self.assertIsNone(rejected)
        self.assertEqual(reason, "fallback_resident_coverage")
        self.assertEqual(planner.snapshot_state(), baseline)

    def test_cull_failure_is_published_before_broadcast(self):
        planner = _planner()
        baseline = planner.snapshot_state()
        predicted, preview = self._preview(planner, _blocks([0]))
        storage = SimpleNamespace(get_bounds_generation=lambda: 2,
                                  get_visible_blocks_batch=Mock(side_effect=OSError("cull failed")))
        with self.assertRaisesRegex(RuntimeError, "final distributed planning.*cull failed"):
            self.finalize(context=self.context, planner=planner, storage_adapter=storage,
                          training_schedule=[], iteration=65, batch_size=8,
                          schedule_ordering="trajectory", predicted_plan=predicted,
                          rank0_preview=preview)
        self.assertEqual(planner.snapshot_state(), baseline)
        self.assertEqual(self.context.broadcasts, [])

    def test_multiple_refreshes_advance_recency_once_and_state_round_trips(self):
        planner = _planner(capacity=-1)
        for batch in range(4):
            predicted, preview = self._preview(planner, _blocks([0, 1]), iteration=65 + 64 * batch)
            actual = self._finalize(planner, predicted, preview, _blocks([2, 7]))
            self.assertEqual(planner.snapshot_state().recency, preview["predicted_state"].recency)
            self.assertEqual(DistributedBatchPlan.from_dict(actual.to_dict()), actual)
            state = planner.snapshot_state()
            restored = _planner(capacity=-1)
            restored.restore_state(state)
            planner = restored
            self.assertEqual(planner.snapshot_state().active, (2, 7))

    def test_legacy_payload_preserves_numeric_comparison_defaults(self):
        planner = _planner()
        predicted, _ = self._preview(planner, _blocks([0]))
        payload = predicted.to_dict()
        for field in ("plan_resolution", "prediction_missing_blocks", "prediction_extra_blocks"):
            payload.pop(field)
        restored = DistributedBatchPlan.from_dict(payload)
        self.assertEqual(restored.plan_resolution, "initial")
        self.assertEqual(restored.prediction_missing_blocks, 0)
        self.assertEqual(restored.prediction_extra_blocks, 0)


if __name__ == "__main__":
    unittest.main()
