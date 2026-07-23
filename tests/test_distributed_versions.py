import unittest

from strategies.tide_engine.distributed_engine import DistributedResidentState


class DistributedResidentVersionTest(unittest.TestCase):
    def test_old_commit_cannot_clear_newer_gpu_dirty_version(self):
        state = DistributedResidentState(block_versions={3: 5})
        state.mark_dirty_blocks([3], iteration=1)
        first = state.begin_writeback([3])
        self.assertEqual(first, {3: 6})
        self.assertEqual(state.pending_commit_versions, {3: 6})

        state.mark_dirty_blocks([3], iteration=33)
        state.complete_writeback([3], first)

        self.assertEqual(state.dirty_blocks(), [3])
        self.assertEqual(state.versions_for_blocks([3]), {3: 7})
        self.assertEqual(state.dirty_origin_iterations, {3: 33})

    def test_failed_commit_returns_pending_version_to_gpu_dirty(self):
        state = DistributedResidentState(block_versions={4: 2})
        state.mark_dirty_blocks([4], iteration=9)
        pending = state.begin_writeback([4])
        state.cancel_writeback(pending)

        self.assertEqual(state.dirty_blocks(), [4])
        self.assertEqual(state.pending_commit_versions, {})
        self.assertEqual(state.versions_for_blocks([4]), {4: 3})


if __name__ == "__main__":
    unittest.main()
