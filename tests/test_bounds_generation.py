import threading
import unittest

import numpy as np
import torch

from storage.gaussian_block import FrustumCuller
from storage.tide_storage_adapter import TideStorageAdapter


class _Culler:
    def __init__(self):
        self.calls = []

    def update_block_bounds(self, block_ids, bounds):
        self.calls.append((block_ids.copy(), bounds.copy()))


class _Camera:
    R = np.eye(3, dtype=np.float32)
    T = np.zeros(3, dtype=np.float32)
    FovY = np.deg2rad(45.0)
    width = 640
    height = 480


class _RepairCuller(_Culler):
    num_blocks = 4

    def __init__(self, visible):
        super().__init__()
        self.visible = set(visible)
        self.full_culls = 0
        self.subset_culls = []

    def cull(self, **kwargs):
        self.full_culls += 1
        return sorted(self.visible)

    def cull_subset(self, *, block_ids, **kwargs):
        ids = set(int(block_id) for block_id in block_ids)
        self.subset_culls.append(ids)
        return sorted(ids & self.visible)


class _GpuCuller:
    def __init__(self):
        self.calls = []
        self.last_cull_timing = {"kernel_ms": 1.5, "d2h_ms": 0.5}

    def cull_batch(self, view_matrices, proj_matrices):
        self.calls.append(len(view_matrices))
        return [[index] for index in range(len(view_matrices))]


class _BoundsPayload:
    block_ids = [1]

    def wait(self):
        return [1], torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.float32)


class BoundsGenerationTest(unittest.TestCase):
    def test_update_preserves_versioned_cache_and_advances_generation(self):
        adapter = TideStorageAdapter.__new__(TideStorageAdapter)
        adapter._bounds_state_lock = threading.RLock()
        adapter._bounds_generation = 4
        adapter._visibility_cache = {7: (4, (1, 2))}
        adapter.block_bounds = np.zeros((3, 6), dtype=np.float32)
        adapter.culler = _Culler()
        adapter.execution_metrics = {"paper_bounds_refresh_blocks": 0}
        new_bounds = np.asarray([[1, 2, 3, 4, 5, 6]], dtype=np.float32)

        self.assertEqual(adapter.update_block_bounds([2], new_bounds), 1)

        self.assertEqual(adapter.get_bounds_generation(), 5)
        self.assertEqual(adapter._visibility_cache, {7: (4, (1, 2))})
        self.assertEqual(adapter._bounds_change_log[5], (2,))
        np.testing.assert_array_equal(adapter.block_bounds[2], new_bounds[0])
        self.assertEqual(len(adapter.culler.calls), 1)
        self.assertEqual(adapter.execution_metrics["paper_bounds_refresh_blocks"], 1)

    def test_background_refresh_is_visible_after_fence(self):
        adapter = TideStorageAdapter.__new__(TideStorageAdapter)
        adapter._bounds_state_lock = threading.RLock()
        adapter._bounds_generation = 0
        adapter._visibility_cache = {2: (0, (0,))}
        adapter.block_bounds = np.zeros((2, 6), dtype=np.float32)
        adapter.culler = _Culler()
        adapter.execution_metrics = {
            "paper_bounds_refresh_blocks": 0,
            "background_bounds_jobs": 0,
            "background_bounds_blocks": 0,
            "background_bounds_waits": 0,
        }
        adapter._bounds_refresh_queue = __import__("queue").Queue()
        adapter._bounds_refresh_error = None
        adapter._bounds_refresh_thread = None
        adapter._start_bounds_refresh_worker()
        self.addCleanup(self._shutdown_bounds_worker, adapter)

        self.assertEqual(adapter.submit_bounds_refresh(_BoundsPayload()), 1)
        adapter.wait_for_bounds_refresh()

        self.assertEqual(adapter.get_bounds_generation(), 1)
        self.assertEqual(adapter._visibility_cache, {2: (0, (0,))})
        self.assertEqual(adapter._bounds_change_log[1], (1,))
        np.testing.assert_array_equal(
            adapter.block_bounds[1],
            np.asarray([1, 1, 1, 2, 2, 2], dtype=np.float32),
        )

    @staticmethod
    def _shutdown_bounds_worker(adapter):
        adapter.wait_for_bounds_refresh()
        adapter._bounds_refresh_queue.put(None)
        adapter._bounds_refresh_queue.join()
        adapter._bounds_refresh_thread.join(timeout=2.0)

    def test_incremental_visibility_repair_tracks_changed_blocks(self):
        adapter = TideStorageAdapter.__new__(TideStorageAdapter)
        adapter._bounds_state_lock = threading.RLock()
        adapter._bounds_generation = 4
        adapter._bounds_change_log = __import__("collections").OrderedDict()
        adapter._bounds_change_history_limit = 64
        adapter._visibility_cache = {0: (4, (0, 1))}
        adapter.block_bounds = np.zeros((4, 6), dtype=np.float32)
        adapter.culler = _RepairCuller({0})
        adapter.cameras = [_Camera()]
        adapter.scene_radius = 10.0
        adapter.use_6plane = True
        adapter.paper_debug_logging = False
        adapter.execution_metrics = {"paper_bounds_refresh_blocks": 0}

        adapter.update_block_bounds(
            [1], np.asarray([[1, 1, 1, 2, 2, 2]], dtype=np.float32)
        )
        self.assertEqual(adapter.get_visible_blocks(0, wait_for_refresh=False), [0])
        self.assertEqual(adapter.culler.subset_culls, [{1}])
        self.assertEqual(adapter.culler.full_culls, 0)

        adapter.culler.visible.add(2)
        adapter.update_block_bounds(
            [2], np.asarray([[2, 2, 2, 3, 3, 3]], dtype=np.float32)
        )
        self.assertEqual(adapter.get_visible_blocks(0, wait_for_refresh=False), [0, 2])
        self.assertEqual(adapter.culler.subset_culls[-1], {2})
        self.assertEqual(adapter.execution_metrics["visibility_incremental_repairs"], 2)
        self.assertEqual(adapter.execution_metrics["visibility_repaired_blocks"], 2)

    def test_debug_coordinate_check_does_not_require_clip_variables(self):
        adapter = TideStorageAdapter.__new__(TideStorageAdapter)
        adapter._bounds_state_lock = threading.RLock()
        adapter._bounds_generation = 0
        adapter._bounds_change_log = __import__("collections").OrderedDict()
        adapter._visibility_cache = {}
        adapter.block_bounds = np.asarray(
            [[-1, -1, -1, 1, 1, 1]], dtype=np.float32
        )
        adapter.culler = _RepairCuller({0})
        adapter.cameras = [_Camera()]
        adapter.scene_radius = 10.0
        adapter.use_6plane = True
        adapter.paper_debug_logging = True
        adapter.execution_metrics = {}

        self.assertEqual(adapter.get_visible_blocks(0, wait_for_refresh=False), [0])

    def test_gpu_batch_cull_reuses_same_generation_cache(self):
        adapter = TideStorageAdapter.__new__(TideStorageAdapter)
        adapter._bounds_state_lock = threading.RLock()
        adapter._bounds_generation = 0
        adapter._visibility_cache = {}
        adapter.cameras = [_Camera(), _Camera()]
        adapter.scene_radius = 10.0
        adapter.gpu_culler = _GpuCuller()
        adapter.wait_for_bounds_refresh = lambda: None

        _, first = adapter.get_visible_blocks_batch([0, 1])
        self.assertEqual(first, {0: [0], 1: [1]})
        self.assertEqual(adapter.gpu_culler.calls, [2])
        self.assertEqual(adapter._batch_cull_metrics, {
            "block_cull_backend": "gpu",
            "block_cull_gpu_kernel_ms": 1.5,
            "block_cull_gpu_d2h_ms": 0.5,
            "block_cull_cache_hit_cameras": 0,
            "block_cull_gpu_cameras": 2,
            "block_cull_output_blocks": 2,
        })

        _, second = adapter.get_visible_blocks_batch([0, 1])
        self.assertEqual(second, first)
        self.assertEqual(adapter.gpu_culler.calls, [2])
        self.assertEqual(adapter._batch_cull_metrics, {
            "block_cull_backend": "gpu",
            "block_cull_gpu_kernel_ms": 0.0,
            "block_cull_gpu_d2h_ms": 0.0,
            "block_cull_cache_hit_cameras": 2,
            "block_cull_gpu_cameras": 0,
            "block_cull_output_blocks": 0,
        })

    def test_subset_culling_matches_full_six_plane_result(self):
        rng = np.random.default_rng(7)
        mins = rng.uniform(-4.0, 4.0, size=(128, 3)).astype(np.float32)
        extents = rng.uniform(0.05, 0.8, size=(128, 3)).astype(np.float32)
        bounds = np.concatenate([mins, mins + extents], axis=1)
        culler = FrustumCuller(bounds, scene_radius=10.0, verbose=False)
        view = np.eye(4, dtype=np.float32)
        projection = np.eye(4, dtype=np.float32)

        full = culler.cull(
            camera_position=np.zeros(3, dtype=np.float32),
            view_matrix=view,
            projection_matrix=projection,
            use_6plane=True,
        )
        subset = culler.cull_subset(
            camera_position=np.zeros(3, dtype=np.float32),
            view_matrix=view,
            projection_matrix=projection,
            block_ids=np.arange(128),
            use_6plane=True,
        )
        self.assertEqual(subset, full)
        subset_from_set = culler.cull_subset(
            camera_position=np.zeros(3, dtype=np.float32),
            view_matrix=view,
            projection_matrix=projection,
            block_ids=set(range(128)),
            use_6plane=True,
        )
        self.assertEqual(subset_from_set, full)


if __name__ == "__main__":
    unittest.main()
