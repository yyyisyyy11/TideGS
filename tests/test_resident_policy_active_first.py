"""Active-first TopC resident selection (ported from upstream TideGS e4a29c0)."""

import unittest
from types import SimpleNamespace

from strategies.tide_engine.resident_policy import compute_topc_resident_transition
from strategies.tide_engine.runtime import resolve_current_iteration_resident_blocks


def _transition(**overrides):
    kwargs = dict(
        current_active_blocks=[0, 1, 2, 3],
        next_active_blocks=[4, 5, 6, 7, 8],
        current_resident_blocks=[0, 1, 2, 3],
        next_camera_ids=[10, 11],
        next_camera_blocks={10: [4, 5, 6], 11: [7, 8]},
        previous_recency_scores={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
        lambda_weight=0.3,
        recency_decay=0.95,
        resident_capacity_blocks=6,
        balanced_camera_seeds=True,
        balanced_seed_fraction=0.25,
    )
    kwargs.update(overrides)
    return compute_topc_resident_transition(**kwargs)


class ActiveFirstTransitionTest(unittest.TestCase):
    def test_default_behaviour_is_unchanged_and_can_drop_active_blocks(self):
        legacy = _transition()
        self.assertFalse(legacy.enforce_next_active_coverage)
        self.assertEqual(legacy.resident_selection_policy, "topc_balanced")
        self.assertLess(legacy.next_active_coverage, 5)

    def test_all_next_active_blocks_are_resident_when_they_fit(self):
        t = _transition(enforce_next_active_coverage=True)
        self.assertEqual(t.resident_selection_policy, "topc_balanced_active_first")
        self.assertTrue({4, 5, 6, 7, 8} <= set(t.next_resident_blocks))
        self.assertEqual(t.next_active_coverage, 5)
        self.assertLessEqual(len(t.next_resident_blocks), 6)
        self.assertEqual(len(t.optional_selected_blocks), 1)  # one spare slot keeps a stale block

    def test_only_next_active_blocks_are_selected_when_they_exceed_capacity(self):
        t = _transition(enforce_next_active_coverage=True, resident_capacity_blocks=3)
        self.assertEqual(len(t.next_resident_blocks), 3)
        self.assertTrue(set(t.next_resident_blocks) <= {4, 5, 6, 7, 8})
        self.assertEqual(t.optional_selected_blocks, [])
        self.assertEqual(t.next_active_coverage, 3)

    def test_strict_variant_has_no_camera_seeds(self):
        t = _transition(enforce_next_active_coverage=True, balanced_camera_seeds=False)
        self.assertEqual(t.resident_selection_policy, "topc_strict_active_first")
        self.assertEqual(t.camera_seed_blocks, [])
        self.assertEqual(t.next_active_coverage, 5)

    def test_transition_sets_are_consistent(self):
        for capacity in (1, 3, 5, 6, 9):
            t = _transition(enforce_next_active_coverage=True, resident_capacity_blocks=capacity)
            keep, stream_in, evict = map(set, (t.keep_resident_blocks, t.stream_in_blocks, t.evict_blocks))
            self.assertFalse(keep & stream_in or keep & evict or stream_in & evict)
            self.assertEqual(keep | stream_in, set(t.next_resident_blocks))
            self.assertEqual(keep | evict, set(t.current_resident_blocks))
            self.assertEqual(t.next_active_coverage, min(5, capacity))


class SingleRankRuntimeTest(unittest.TestCase):
    def _resolve(self, policy, capacity):
        gaussians = SimpleNamespace(_paper_expected_resident_blocks=[], _paper_resident_recency_scores={})
        args = SimpleNamespace(
            paper_resident_selection_policy=policy,
            paper_resident_capacity_blocks=capacity,
            paper_resident_lambda=0.3,
            paper_resident_recency_decay=0.95,
            paper_balanced_seed_fraction=0.25,
        )
        return resolve_current_iteration_resident_blocks(
            gaussians, args, [2, 4, 6, 8, 10], num_total_blocks=16,
            current_camera_blocks={0: [2, 4, 6], 1: [8, 10]},
        )

    def test_active_first_policy_is_accepted_by_runtime(self):
        blocks, source = self._resolve("topc_balanced_active_first", 8)
        self.assertEqual(source, "bootstrap_topc_over_k1")
        self.assertEqual(blocks, [2, 4, 6, 8, 10])

    def test_active_first_respects_capacity(self):
        blocks, _ = self._resolve("topc_balanced_active_first", 3)
        self.assertEqual(len(blocks), 3)
        self.assertTrue(set(blocks) <= {2, 4, 6, 8, 10})

    def test_unknown_policy_still_falls_back_to_passthrough(self):
        blocks, source = self._resolve("passthrough_active_set", 3)
        self.assertEqual(source, "passthrough_active_set")
        self.assertEqual(blocks, [2, 4, 6, 8, 10])


if __name__ == "__main__":
    unittest.main()
