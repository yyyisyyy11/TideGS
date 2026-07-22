import unittest

import numpy as np

from strategies.tide_engine.distributed_plan import (
    DistributedBatchPlan,
    DistributedBatchPlanner,
    assign_cameras_balanced,
    build_balanced_block_owner,
)


class DistributedPlanTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
