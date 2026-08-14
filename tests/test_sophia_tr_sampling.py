import unittest

import numpy as np

from strategies.tide_engine.sophia_tr_sampling import sample_s2_camera_ids


class SophiaTRSamplingTest(unittest.TestCase):
    def test_sampling_is_stateless_and_deterministic_for_seed_and_step(self):
        population = list(range(64))
        first_s1 = list(range(8))
        second_s1 = list(range(8, 16))
        np.random.seed(123)
        state_before = np.random.get_state()

        first = sample_s2_camera_ids(
            population_camera_ids=population,
            s1_camera_ids=first_s1,
            seed=2026,
            optimizer_step=11,
        )
        repeated = sample_s2_camera_ids(
            population_camera_ids=population,
            s1_camera_ids=first_s1,
            seed=2026,
            optimizer_step=11,
        )
        same_size_different_s1 = sample_s2_camera_ids(
            population_camera_ids=population,
            s1_camera_ids=second_s1,
            seed=2026,
            optimizer_step=11,
        )
        state_after = np.random.get_state()

        self.assertEqual(first, repeated)
        self.assertEqual(first, same_size_different_s1)
        self.assertEqual(len(first), len(first_s1))
        self.assertEqual(len(set(first)), len(first))
        self.assertEqual(state_before[0], state_after[0])
        np.testing.assert_array_equal(state_before[1], state_after[1])
        self.assertEqual(state_before[2:], state_after[2:])

    def test_sampling_uses_step_and_allows_s1_overlap(self):
        population = list(range(32))
        s1 = list(population)
        first = sample_s2_camera_ids(
            population_camera_ids=population,
            s1_camera_ids=s1,
            seed=7,
            optimizer_step=1,
        )
        later = sample_s2_camera_ids(
            population_camera_ids=population,
            s1_camera_ids=s1,
            seed=7,
            optimizer_step=11,
        )

        self.assertEqual(set(first), set(s1))
        self.assertEqual(set(later), set(s1))
        self.assertNotEqual(first, later)

    def test_sampling_rejects_invalid_batches(self):
        with self.assertRaisesRegex(ValueError, "unique camera IDs"):
            sample_s2_camera_ids(
                population_camera_ids=[0, 0, 1],
                s1_camera_ids=[0],
                seed=1,
                optimizer_step=1,
            )
        with self.assertRaisesRegex(ValueError, "missing"):
            sample_s2_camera_ids(
                population_camera_ids=[0, 1, 2],
                s1_camera_ids=[3],
                seed=1,
                optimizer_step=1,
            )
        with self.assertRaisesRegex(ValueError, ">= 1"):
            sample_s2_camera_ids(
                population_camera_ids=[0, 1, 2],
                s1_camera_ids=[0],
                seed=1,
                optimizer_step=0,
            )


if __name__ == "__main__":
    unittest.main()
