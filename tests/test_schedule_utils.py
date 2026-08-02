import os
import sys
import unittest

# Import schedule_utils directly to avoid triggering storage/__init__.py
# which has heavy dependencies (torch, etc.).
_sched_dir = os.path.join(os.path.dirname(__file__), "..", "storage")
sys.path.insert(0, os.path.abspath(_sched_dir))
import schedule_utils  # noqa: E402

DEFAULT_EPOCH_SEED_BASE = schedule_utils.DEFAULT_EPOCH_SEED_BASE
_circular_slice = schedule_utils._circular_slice
get_camera_batch_schedule = schedule_utils.get_camera_batch_schedule


class CircularSliceTest(unittest.TestCase):
    def test_basic_slice(self):
        schedule = [0, 1, 2, 3, 4]
        result = _circular_slice(schedule, 1, 3)
        self.assertEqual(result, [1, 2, 3])

    def test_wrap_around(self):
        schedule = [0, 1, 2, 3, 4]
        result = _circular_slice(schedule, 3, 4)
        self.assertEqual(result, [3, 4, 0, 1])

    def test_empty_schedule(self):
        self.assertEqual(_circular_slice([], 0, 5), [])


class MicrobatchShuffleTest(unittest.TestCase):
    """Tests for the microbatch_shuffle schedule ordering mode."""

    def setUp(self):
        # 10 cameras in TSP order; batch_size=4 → 3 batches (ceil(10/4))
        self.schedule = list(range(10))
        self.bsz = 4

    def _get_epoch_batches(self, schedule, iteration, bsz, ordering, epoch_seed_base):
        """Collect all batches for the epoch containing `iteration`."""
        sched = get_camera_batch_schedule(
            training_schedule=schedule,
            iteration=iteration,
            batch_size=bsz,
            epoch_seed_base=epoch_seed_base,
            schedule_ordering=ordering,
        )
        n_batches = sched.num_batches
        epoch_base_iteration = sched.epoch * n_batches * bsz + 1
        batches = []
        for i in range(n_batches):
            it = epoch_base_iteration + i * bsz
            b = get_camera_batch_schedule(
                training_schedule=schedule,
                iteration=it,
                batch_size=bsz,
                epoch_seed_base=epoch_seed_base,
                schedule_ordering=ordering,
            )
            batches.append(b.batch_indices)
        return batches

    # ── microbatch_shuffle specific tests ──

    def test_preserves_tsp_order_within_each_microbatch(self):
        """Each micro-batch must preserve contiguous TSP order (possibly wrapped)."""
        batches = self._get_epoch_batches(
            self.schedule, iteration=1, bsz=self.bsz,
            ordering="microbatch_shuffle", epoch_seed_base=DEFAULT_EPOCH_SEED_BASE,
        )
        for batch in batches:
            # Verify that the cameras in this batch appear in the same relative
            # order as they do in the canonical TSP schedule.
            positions = [self.schedule.index(c) for c in batch]
            # After unwrapping, the TSP positions should be monotonic within each
            # contiguous segment.
            for i in range(len(positions) - 1):
                # Either positions increase by 1 (same contiguous segment),
                # or we wrapped around (position drops from high to low).
                diff = (positions[i + 1] - positions[i]) % len(self.schedule)
                self.assertEqual(
                    diff, 1,
                    f"Non-consecutive TSP neighbors in batch: {batch} "
                    f"at positions {positions}",
                )

    def test_reproducible_same_epoch_same_seed(self):
        """Same epoch + same seed must produce identical batch order."""
        batches_a = self._get_epoch_batches(
            self.schedule, iteration=1, bsz=self.bsz,
            ordering="microbatch_shuffle", epoch_seed_base=42,
        )
        batches_b = self._get_epoch_batches(
            self.schedule, iteration=1, bsz=self.bsz,
            ordering="microbatch_shuffle", epoch_seed_base=42,
        )
        self.assertEqual(batches_a, batches_b)

    def test_different_epoch_different_batch_order(self):
        """Different epochs must use different deterministic batch orders."""
        # Epoch 0
        batches_e0 = self._get_epoch_batches(
            self.schedule, iteration=1, bsz=self.bsz,
            ordering="microbatch_shuffle", epoch_seed_base=DEFAULT_EPOCH_SEED_BASE,
        )
        # Epoch 1: iteration past epoch 0 boundary
        n_cams = len(self.schedule)
        n_batches = -(-n_cams // self.bsz)  # ceil division
        epoch1_iteration = n_batches * self.bsz + 1
        batches_e1 = self._get_epoch_batches(
            self.schedule, iteration=epoch1_iteration, bsz=self.bsz,
            ordering="microbatch_shuffle", epoch_seed_base=DEFAULT_EPOCH_SEED_BASE,
        )
        # The batch orders should differ (probabilistically; seed 42 guarantees it).
        self.assertNotEqual(batches_e0, batches_e1)

    def test_tail_batch_wraps_around(self):
        """Tail batch must wrap around to fill batch_size cameras.

        With schedule=[0..9] and bsz=4, the three conceptual batches are:
          [0,1,2,3], [4,5,6,7], [8,9,0,1]
        The tail batch [8,9,0,1] wraps from the end to the beginning.
        """
        batches = self._get_epoch_batches(
            self.schedule, iteration=1, bsz=self.bsz,
            ordering="microbatch_shuffle", epoch_seed_base=DEFAULT_EPOCH_SEED_BASE,
        )
        # Collect all batch contents into a set of (position, camera_id) for the
        # tail batch's structure check.
        all_cam_seen = []
        for batch in batches:
            all_cam_seen.extend(batch)
        # The tail batch (whichever order it appears in) must contain cameras
        # from the end AND the beginning of the schedule.
        tail_batch = None
        for batch in batches:
            positions = [self.schedule.index(c) for c in batch]
            # A wrapped batch has a position drop > 1 somewhere
            for i in range(len(positions) - 1):
                if (positions[i + 1] - positions[i]) % len(self.schedule) != 1:
                    # This batch does NOT have consecutive TSP order — shouldn't happen
                    pass
            # Check if this batch wraps: the set of cameras spans both ends
            if max(positions) - min(positions) >= len(self.schedule) - self.bsz:
                tail_batch = batch
                break
        self.assertIsNotNone(tail_batch, "No tail batch with wrap-around found")
        # Verify the tail batch has cameras from both ends
        self.assertIn(8, tail_batch)
        self.assertIn(9, tail_batch)
        self.assertIn(0, tail_batch)

    def test_epoch_camera_offset_is_zero(self):
        """microbatch_shuffle must always report epoch_camera_offset=0."""
        sched = get_camera_batch_schedule(
            training_schedule=self.schedule,
            iteration=1,
            batch_size=self.bsz,
            schedule_ordering="microbatch_shuffle",
        )
        self.assertEqual(sched.epoch_camera_offset, 0)

    # ── Regression tests: trajectory mode ──

    def test_trajectory_epoch0_starts_at_zero(self):
        """Trajectory mode epoch 0 must start at camera offset 0."""
        sched = get_camera_batch_schedule(
            training_schedule=self.schedule,
            iteration=1,
            batch_size=self.bsz,
            schedule_ordering="trajectory",
        )
        self.assertEqual(sched.epoch_camera_offset, 0)
        self.assertEqual(sched.batch_indices, [0, 1, 2, 3])

    def test_trajectory_nonzero_epoch_has_offset(self):
        """Trajectory mode epoch >= 1 must have a random epoch_camera_offset."""
        n_cams = len(self.schedule)
        n_batches = -(-n_cams // self.bsz)
        epoch1_iteration = n_batches * self.bsz + 1
        sched = get_camera_batch_schedule(
            training_schedule=self.schedule,
            iteration=epoch1_iteration,
            batch_size=self.bsz,
            schedule_ordering="trajectory",
            epoch_seed_base=42,
        )
        # With seed 42 + epoch 1 = 43, the offset should be deterministic but non-zero.
        self.assertGreater(sched.epoch_camera_offset, 0)

    def test_trajectory_preserves_tsp_order(self):
        """Trajectory batches must preserve TSP order (contiguous slices)."""
        sched = get_camera_batch_schedule(
            training_schedule=self.schedule,
            iteration=1,
            batch_size=3,
            schedule_ordering="trajectory",
        )
        positions = [self.schedule.index(c) for c in sched.batch_indices]
        for i in range(len(positions) - 1):
            diff = (positions[i + 1] - positions[i]) % len(self.schedule)
            self.assertEqual(diff, 1)

    # ── Regression tests: shuffle mode ──

    def test_shuffle_epoch_camera_offset_zero(self):
        """Shuffle mode must always report epoch_camera_offset=0."""
        sched = get_camera_batch_schedule(
            training_schedule=self.schedule,
            iteration=1,
            batch_size=self.bsz,
            schedule_ordering="shuffle",
        )
        self.assertEqual(sched.epoch_camera_offset, 0)

    def test_shuffle_different_epochs_produce_different_permutations(self):
        """Shuffle mode must produce different permutations across epochs."""
        batches_e0 = self._get_epoch_batches(
            self.schedule, iteration=1, bsz=self.bsz,
            ordering="shuffle", epoch_seed_base=DEFAULT_EPOCH_SEED_BASE,
        )
        n_cams = len(self.schedule)
        n_batches = -(-n_cams // self.bsz)
        epoch1_iteration = n_batches * self.bsz + 1
        batches_e1 = self._get_epoch_batches(
            self.schedule, iteration=epoch1_iteration, bsz=self.bsz,
            ordering="shuffle", epoch_seed_base=DEFAULT_EPOCH_SEED_BASE,
        )
        self.assertNotEqual(batches_e0, batches_e1)

    def test_shuffle_reproducible(self):
        """Shuffle mode must be reproducible with same seed."""
        batches_a = self._get_epoch_batches(
            self.schedule, iteration=1, bsz=self.bsz,
            ordering="shuffle", epoch_seed_base=42,
        )
        batches_b = self._get_epoch_batches(
            self.schedule, iteration=1, bsz=self.bsz,
            ordering="shuffle", epoch_seed_base=42,
        )
        self.assertEqual(batches_a, batches_b)

    # ── Edge cases ──

    def test_batch_size_one(self):
        """bsz=1: each camera is its own batch; order shuffled but cameras unchanged."""
        sched = get_camera_batch_schedule(
            training_schedule=[5, 3, 1],
            iteration=1,
            batch_size=1,
            schedule_ordering="microbatch_shuffle",
        )
        self.assertEqual(len(sched.batch_indices), 1)
        self.assertIn(sched.batch_indices[0], [5, 3, 1])

    def test_batch_size_equals_num_cameras(self):
        """bsz == num_cameras: single batch containing all cameras in TSP order."""
        sched = get_camera_batch_schedule(
            training_schedule=[5, 3, 1],
            iteration=1,
            batch_size=3,
            schedule_ordering="microbatch_shuffle",
        )
        self.assertEqual(sched.num_batches, 1)
        self.assertEqual(sched.within_epoch_idx, 0)
        # Single batch: must be the full TSP schedule
        self.assertEqual(sched.batch_indices, [5, 3, 1])

    def test_raises_on_negative_batch_size(self):
        with self.assertRaises(ValueError):
            get_camera_batch_schedule(
                training_schedule=[0, 1, 2],
                iteration=1,
                batch_size=0,
                schedule_ordering="microbatch_shuffle",
            )

    def test_raises_on_empty_schedule(self):
        with self.assertRaises(ValueError):
            get_camera_batch_schedule(
                training_schedule=[],
                iteration=1,
                batch_size=4,
                schedule_ordering="microbatch_shuffle",
            )


if __name__ == "__main__":
    unittest.main()
