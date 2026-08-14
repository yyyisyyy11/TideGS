import unittest

import numpy as np

from strategies.tide_engine.distributed_plan import (
    DistributedBatchPlan,
    DistributedBatchPlanner,
    assign_cameras_balanced,
    build_balanced_block_owner,
    build_stable_block_owner,
)
from strategies.tide_engine.resident_policy import compute_topc_resident_transition


class DistributedPlanTest(unittest.TestCase):
    def test_stable_owner_is_round_robin(self):
        owner = build_stable_block_owner(num_blocks=11, world_size=4)
        np.testing.assert_array_equal(
            owner,
            np.asarray([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2], dtype=np.int32),
        )

    def test_owner_map_is_deterministic_complete_and_balanced(self):
        kwargs = dict(
            num_blocks=17,
            total_points=65,
            block_size=4,
            world_size=4,
            visibility_counts=[20, 1, 1, 1, 10, 1, 1, 1, 5, 1, 1, 1, 2, 1, 1, 1, 0],
        )
        first = build_balanced_block_owner(**kwargs)
        second = build_balanced_block_owner(**kwargs)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (17,))
        self.assertTrue(np.all((first >= 0) & (first < 4)))
        self.assertTrue(all(np.any(first == rank) for rank in range(4)))
        weights = np.asarray(
            [4 * (1 + count) for count in kwargs["visibility_counts"][:-1]] + [1],
            dtype=np.int64,
        )
        loads = [int(weights[first == rank].sum()) for rank in range(4)]
        self.assertLessEqual(max(loads) - min(loads), int(weights.max()))

    def test_camera_assignment_uses_each_camera_once_and_fills_each_rank(self):
        camera_ids = list(range(16))
        camera_blocks = {camera_id: list(range(camera_id % 5 + 1)) for camera_id in camera_ids}
        assignments = assign_cameras_balanced(
            camera_ids=camera_ids,
            camera_blocks=camera_blocks,
            block_rows=[4, 4, 4, 4, 4],
            world_size=4,
        )
        self.assertEqual([len(values) for values in assignments], [4, 4, 4, 4])
        self.assertEqual(sorted(value for values in assignments for value in values), camera_ids)

    def test_planner_enforces_unique_owner_residency_and_round_trips(self):
        owner = np.asarray([block_id % 4 for block_id in range(24)], dtype=np.int32)
        planner = DistributedBatchPlanner(
            block_owner=owner,
            total_points=96,
            block_size=4,
            world_size=4,
            resident_capacity_blocks=3,
            resident_lambda=0.3,
            resident_recency_decay=0.95,
            balanced_seed_fraction=0.5,
            camera_assignment="gaussian_balanced",
        )
        camera_ids = list(range(16))
        camera_blocks = {
            camera_id: sorted({camera_id % 24, (camera_id + 4) % 24, (camera_id + 8) % 24})
            for camera_id in camera_ids
        }
        plan = planner.plan(
            iteration=1,
            epoch=0,
            camera_ids=camera_ids,
            camera_blocks=camera_blocks,
        )
        restored = DistributedBatchPlan.from_dict(plan.to_dict())
        self.assertEqual(restored, plan)
        self.assertEqual(plan.global_s1_camera_ids, plan.global_camera_ids)
        self.assertEqual(plan.rank_s1_camera_ids, plan.rank_camera_ids)
        self.assertEqual(plan.global_s2_camera_ids, [])
        self.assertEqual(plan.rank_s2_camera_ids, [[], [], [], []])
        self.assertEqual(
            plan.rank_gradient_active_blocks, plan.rank_active_blocks
        )
        self.assertEqual(plan.rank_curvature_active_blocks, [[], [], [], []])
        self.assertEqual([len(values) for values in plan.rank_camera_ids], [4, 4, 4, 4])
        self.assertEqual(
            sorted(value for values in plan.rank_camera_ids for value in values), camera_ids
        )
        seen = set()
        for rank, blocks in enumerate(plan.rank_resident_blocks):
            self.assertLessEqual(len(blocks), 3)
            self.assertTrue(all(owner[block_id] == rank for block_id in blocks))
            self.assertFalse(seen.intersection(blocks))
            seen.update(blocks)
        self.assertLessEqual(len(plan.global_resident_blocks), 3)
        self.assertEqual(seen, set(plan.global_resident_blocks))
        self.assertEqual(
            set(plan.global_active_blocks),
            {
                block_id
                for rank_blocks in plan.rank_active_blocks
                for block_id in rank_blocks
            },
        )
        self.assertEqual(
            set(plan.stream_in_blocks),
            set(plan.global_resident_blocks),
        )

    def test_planner_rejects_duplicate_camera_ids(self):
        planner = DistributedBatchPlanner(
            block_owner=np.asarray([0, 1, 2, 3], dtype=np.int32),
            total_points=16,
            block_size=4,
            world_size=4,
            resident_capacity_blocks=1,
            resident_lambda=0.3,
            resident_recency_decay=0.95,
            balanced_seed_fraction=1.0,
        )
        with self.assertRaisesRegex(ValueError, "unique camera IDs"):
            planner.plan(
                iteration=1,
                epoch=0,
                camera_ids=[0, 1, 2, 2],
                camera_blocks={camera_id: [camera_id] for camera_id in range(4)},
            )

    def test_curvature_batch_uses_union_residency_and_separate_active_sets(self):
        owner = np.asarray([block_id % 2 for block_id in range(8)], dtype=np.int32)
        planner = DistributedBatchPlanner(
            block_owner=owner,
            total_points=32,
            block_size=4,
            world_size=2,
            resident_capacity_blocks=8,
            resident_lambda=1.0,
            resident_recency_decay=0.95,
            balanced_seed_fraction=1.0,
            camera_assignment="gaussian_balanced",
        )
        plan = planner.plan(
            iteration=1,
            epoch=0,
            camera_ids=[0, 1, 2, 3],
            camera_blocks={0: [0], 1: [1], 2: [2], 3: [3]},
            curvature_camera_ids=[2, 3, 4, 5],
            curvature_camera_blocks={2: [2, 4], 3: [3, 5], 4: [6], 5: [7]},
        )

        self.assertEqual(plan.global_s1_camera_ids, [0, 1, 2, 3])
        self.assertEqual(plan.global_s2_camera_ids, [2, 3, 4, 5])
        self.assertEqual(
            [len(values) for values in plan.rank_s1_camera_ids], [2, 2]
        )
        self.assertEqual(
            [len(values) for values in plan.rank_s2_camera_ids], [2, 2]
        )
        self.assertEqual(plan.global_gradient_active_blocks, [0, 1, 2, 3])
        self.assertEqual(
            plan.global_curvature_active_blocks, [2, 3, 4, 5, 6, 7]
        )
        self.assertEqual(plan.global_union_active_blocks, list(range(8)))
        self.assertEqual(plan.global_resident_blocks, list(range(8)))
        self.assertEqual(plan.rank_owner_active_rows, [16, 16])
        self.assertEqual(plan.rank_participation_rows, [16, 16])
        self.assertEqual(
            DistributedBatchPlan.from_dict(plan.to_dict()), plan
        )

    def test_empty_owner_uses_one_collective_participation_row(self):
        planner = DistributedBatchPlanner(
            block_owner=np.asarray([0, 0], dtype=np.int32),
            total_points=8,
            block_size=4,
            world_size=2,
            resident_capacity_blocks=1,
            resident_lambda=1.0,
            resident_recency_decay=0.95,
            balanced_seed_fraction=1.0,
            camera_assignment="equal",
        )

        plan = planner.plan(
            iteration=1,
            epoch=0,
            camera_ids=[0, 1],
            camera_blocks={0: [0], 1: [0]},
        )

        self.assertEqual(plan.rank_owner_active_rows, [4, 0])
        self.assertEqual(plan.rank_participation_rows, [4, 1])

    def test_curvature_batch_must_match_s1_global_size(self):
        planner = DistributedBatchPlanner(
            block_owner=np.asarray([0, 1, 0, 1], dtype=np.int32),
            total_points=16,
            block_size=4,
            world_size=2,
            resident_capacity_blocks=4,
            resident_lambda=0.3,
            resident_recency_decay=0.95,
            balanced_seed_fraction=1.0,
            camera_assignment="equal",
        )
        with self.assertRaisesRegex(ValueError, "S2 global camera count"):
            planner.plan(
                iteration=1,
                epoch=0,
                camera_ids=[0, 1, 2, 3],
                camera_blocks={camera_id: [camera_id] for camera_id in range(4)},
                curvature_camera_ids=[0, 1],
                curvature_camera_blocks={0: [0], 1: [1]},
            )

    def test_plan_rejects_inconsistent_rank_s2_assignment(self):
        with self.assertRaisesRegex(ValueError, "every global S2 camera"):
            DistributedBatchPlan(
                iteration=1,
                epoch=0,
                global_camera_ids=[0, 1],
                rank_camera_ids=[[0], [1]],
                rank_resident_blocks=[[0], [1]],
                rank_active_blocks=[[0], [1]],
                global_s2_camera_ids=[2, 3],
                rank_s2_camera_ids=[[2], [2]],
            )

    def test_legacy_payload_defaults_to_s1_only(self):
        payload = {
            "iteration": 7,
            "epoch": 1,
            "global_camera_ids": [10, 11],
            "rank_camera_ids": [[10], [11]],
            "rank_resident_blocks": [[0], [1]],
            "rank_active_blocks": [[0], [1]],
        }

        plan = DistributedBatchPlan.from_dict(payload)

        self.assertEqual(plan.global_s1_camera_ids, [10, 11])
        self.assertEqual(plan.rank_s1_camera_ids, [[10], [11]])
        self.assertEqual(plan.global_s2_camera_ids, [])
        self.assertEqual(plan.rank_s2_camera_ids, [[], []])
        self.assertEqual(plan.global_gradient_active_blocks, [0, 1])
        self.assertEqual(plan.global_curvature_active_blocks, [])
        self.assertEqual(plan.global_union_active_blocks, [0, 1])

    def test_preview_does_not_advance_planner_until_state_is_committed(self):
        planner = DistributedBatchPlanner(
            block_owner=np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32),
            total_points=24,
            block_size=4,
            world_size=2,
            resident_capacity_blocks=4,
            resident_lambda=0.3,
            resident_recency_decay=0.95,
            balanced_seed_fraction=1.0,
            camera_assignment="equal",
        )
        camera_ids = [0, 1]
        camera_blocks = {0: [0, 2], 1: [1, 3]}
        baseline = planner.snapshot_state()

        preview, predicted_state = planner.preview(
            iteration=1,
            epoch=0,
            camera_ids=camera_ids,
            camera_blocks=camera_blocks,
        )

        self.assertEqual(planner.snapshot_state(), baseline)
        committed = planner.plan(
            iteration=1,
            epoch=0,
            camera_ids=camera_ids,
            camera_blocks=camera_blocks,
        )
        self.assertEqual(committed, preview)
        self.assertEqual(planner.snapshot_state(), predicted_state)

    def test_preview_state_can_be_committed_without_replanning(self):
        planner = DistributedBatchPlanner(
            block_owner=np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32),
            total_points=24,
            block_size=4,
            world_size=2,
            resident_capacity_blocks=4,
            resident_lambda=0.3,
            resident_recency_decay=0.95,
            balanced_seed_fraction=1.0,
            camera_assignment="equal",
        )
        preview, predicted_state = planner.preview(
            iteration=1,
            epoch=0,
            camera_ids=[0, 1],
            camera_blocks={0: [0, 2], 1: [1, 3]},
        )
        planner.restore_state(predicted_state)

        next_plan = planner.plan(
            iteration=3,
            epoch=0,
            camera_ids=[2, 3],
            camera_blocks={2: [2, 4], 3: [3, 5]},
        )

        self.assertEqual(set(preview.global_resident_blocks), {0, 1, 2, 3})
        self.assertEqual(set(next_plan.stream_in_blocks), {4, 5})

    def test_changed_prediction_can_be_replanned_from_original_state(self):
        kwargs = dict(
            block_owner=np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32),
            total_points=24,
            block_size=4,
            world_size=2,
            resident_capacity_blocks=2,
            resident_lambda=0.3,
            resident_recency_decay=0.95,
            balanced_seed_fraction=1.0,
            camera_assignment="equal",
        )
        planner = DistributedBatchPlanner(**kwargs)
        baseline = planner.snapshot_state()
        predicted, _ = planner.preview(
            iteration=1,
            epoch=0,
            camera_ids=[0, 1],
            camera_blocks={0: [0], 1: [1]},
        )
        exact = planner.plan(
            iteration=1,
            epoch=0,
            camera_ids=[0, 1],
            camera_blocks={0: [2], 1: [3]},
        )

        reference = DistributedBatchPlanner(**kwargs).plan(
            iteration=1,
            epoch=0,
            camera_ids=[0, 1],
            camera_blocks={0: [2], 1: [3]},
        )
        self.assertEqual(baseline.resident, ())
        self.assertEqual(set(predicted.stream_in_blocks), {0, 1})
        self.assertEqual(exact, reference)

    def test_cull_metadata_round_trips_through_plan_payload(self):
        plan = DistributedBatchPlan(
            iteration=1,
            epoch=0,
            global_camera_ids=[0, 1],
            rank_camera_ids=[[0], [1]],
            rank_resident_blocks=[[0], [1]],
            rank_active_blocks=[[0], [1]],
            block_cull_ms=12.0,
            block_cull_backend="gpu",
            block_cull_gpu_kernel_ms=4.0,
            block_cull_gpu_d2h_ms=1.5,
            block_cull_cache_hit_cameras=3,
            block_cull_gpu_cameras=5,
            block_cull_output_blocks=42,
        )
        self.assertEqual(DistributedBatchPlan.from_dict(plan.to_dict()), plan)

    def test_lambda_one_prioritizes_next_active_blocks_over_recency(self):
        transition = compute_topc_resident_transition(
            current_active_blocks=[0],
            next_active_blocks=[1],
            current_resident_blocks=[0],
            previous_recency_scores={0: 1.0},
            lambda_weight=1.0,
            resident_capacity_blocks=1,
        )
        self.assertEqual(transition.next_resident_blocks, [1])


if __name__ == "__main__":
    unittest.main()
