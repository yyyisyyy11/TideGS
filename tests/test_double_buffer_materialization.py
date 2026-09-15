import unittest
import threading
import time

import torch

from storage.block_reader import BlockLayout, _pack_block_batch
from strategies.tide_engine.double_buffer_gpu import DoubleBufferGPUWorkingSet


def _components(global_ids):
    rows = global_ids.to(dtype=torch.float32).unsqueeze(1)
    return {
        'xyz': rows + torch.tensor([0.1, 0.2, 0.3], device=rows.device),
        'scaling': rows + torch.tensor([1.1, 1.2, 1.3], device=rows.device),
        'rotation': rows + torch.tensor([2.1, 2.2, 2.3, 2.4], device=rows.device),
        'opacity': rows + 3.1,
        'features_dc': rows + torch.tensor([4.1, 4.2, 4.3], device=rows.device),
        'features_rest': rows + torch.arange(45, device=rows.device) * 0.01 + 5.0,
    }


def _pack_block(global_ids, layout):
    values = _components(global_ids)
    if layout == BlockLayout.CACHE:
        order = ('xyz', 'scaling', 'rotation', 'opacity', 'features_dc', 'features_rest')
    else:
        order = ('xyz', 'opacity', 'scaling', 'rotation', 'features_dc', 'features_rest')
    return torch.cat([values[name] for name in order], dim=1).cpu()


class _Reader:
    def __init__(self, layout, blocks):
        self.layout = layout
        self.blocks = blocks
        self.calls = []

    def read_blocks(self, block_ids):
        self.calls.append(list(block_ids))
        return {block_id: self.blocks[block_id] for block_id in block_ids}

    def read_batch(self, block_ids, out=None):
        return _pack_block_batch(
            self.read_blocks(block_ids),
            block_ids,
            self.layout,
            out,
        )


class _BlockingReader(_Reader):
    def __init__(self, layout, blocks):
        super().__init__(layout, blocks)
        self.started = threading.Event()
        self.release = threading.Event()

    def read_blocks(self, block_ids):
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test did not release the blocking reader")
        return super().read_blocks(block_ids)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class DoubleBufferMaterializationTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.set_device(0)

    def test_reused_staging_slot_exposes_only_requested_rows(self):
        manager = DoubleBufferGPUWorkingSet(
            num_total_gaussians=32,
            block_size=4,
            device='cuda',
            verbose=False,
        )
        self.addCleanup(manager.clear)

        first_cpu, first_gpu, first_slot = manager._acquire_host_staging(8)
        manager._acquire_host_staging(4)
        reused_cpu, reused_gpu, reused_slot = manager._acquire_host_staging(3)

        self.assertEqual(first_slot, reused_slot)
        self.assertEqual(tuple(reused_cpu.shape), (3, 59))
        self.assertEqual(tuple(reused_gpu.shape), (3, 59))
        self.assertEqual(first_cpu.data_ptr(), reused_cpu.data_ptr())
        self.assertEqual(first_gpu.data_ptr(), reused_gpu.data_ptr())

    def _run_layout(self, layout):
        manager = DoubleBufferGPUWorkingSet(
            num_total_gaussians=14,
            block_size=4,
            device='cuda',
            verbose=False,
        )
        self.addCleanup(manager.clear)

        source_ids = torch.tensor([4, 5, 6, 7, 12, 13], device='cuda')
        source_values = _components(source_ids)
        source = manager.active_buffer
        source.xyz = source_values['xyz']
        source.scaling = source_values['scaling']
        source.rotation = source_values['rotation']
        source.opacity = source_values['opacity']
        source.features_dc = source_values['features_dc']
        source.features_rest = source_values['features_rest']
        source.local_to_global_idx = source_ids
        source.loaded_blocks = [1, 3]
        source.block_to_local_slice = {1: slice(0, 4), 3: slice(4, 6)}
        source.num_gaussians = 6

        cold_ids = torch.tensor([8, 9, 10, 11])
        reader = _Reader(layout, {2: _pack_block(cold_ids, layout)})
        manager.start_prefetch(
            iteration=17,
            visible_block_ids=[3, 2, 1],
            resident_block_ids=[1, 3],
            evicted_block_ids=[0],
            block_reader=reader,
        )
        self.assertTrue(manager.wait_for_prefetch(17))
        target = manager.loading_buffer

        expected_ids = torch.tensor([4, 5, 6, 7, 12, 13, 8, 9, 10, 11], device='cuda')
        expected = _components(expected_ids)
        torch.testing.assert_close(target.local_to_global_idx, expected_ids)
        for name, expected_tensor in expected.items():
            torch.testing.assert_close(getattr(target, name), expected_tensor)

        self.assertEqual(target.loaded_blocks, [1, 3, 2])
        self.assertEqual(target.resident_blocks, [1, 3])
        self.assertEqual(target.streamed_blocks, [2])
        self.assertEqual(target.evicted_blocks, [0])
        self.assertEqual(reader.calls, [[2]])
        self.assertEqual(target.block_to_local_slice[1], slice(0, 4))
        self.assertEqual(target.block_to_local_slice[3], slice(4, 6))
        self.assertEqual(target.block_to_local_slice[2], slice(6, 10))
        self.assertEqual(target._block_starts[[1, 2, 3]].cpu().tolist(), [0, 6, 4])

        stats = manager.get_stats()
        self.assertEqual(stats['batched_materializations'], 1)
        self.assertEqual(stats['batched_h2d_calls'], 1)
        self.assertEqual(stats['batched_resident_blocks'], 2)
        self.assertEqual(stats['batched_streamed_blocks'], 1)

        for name in expected:
            getattr(source, name).add_(10.0)
        handoff = manager.finalize_retained_blocks([1, 3], target='loading')
        self.assertEqual(handoff.ready_blocks, 2)
        self.assertEqual(handoff.copied_blocks, 2)
        manager.prefetch_complete_event.synchronize()
        refreshed_expected = _components(expected_ids)
        for name in refreshed_expected:
            refreshed_expected[name][:6].add_(10.0)
            torch.testing.assert_close(getattr(target, name), refreshed_expected[name])

        manager.swap_buffers()
        query = torch.tensor([4, 12, 8], device='cuda')
        self.assertEqual(manager.global_to_local(query).cpu().tolist(), [0, 4, 6])

    def test_cache_layout(self):
        self._run_layout(BlockLayout.CACHE)

    def test_unified_layout(self):
        self._run_layout(BlockLayout.UNIFIED)

    def test_dirty_blocks_are_written_back_only_on_eviction(self):
        manager = DoubleBufferGPUWorkingSet(
            num_total_gaussians=16,
            block_size=4,
            device='cuda',
            verbose=False,
        )
        self.addCleanup(manager.clear)

        self.assertEqual(manager.mark_dirty_blocks([0, 1, 1]), 2)
        self.assertEqual(manager.dirty_blocks_for_eviction([1, 2]), [1])
        manager.mark_blocks_written_back([1])
        self.assertEqual(manager.dirty_blocks(), [0])

    def test_resident_plan_runs_on_background_worker(self):
        manager = DoubleBufferGPUWorkingSet(
            num_total_gaussians=16,
            block_size=4,
            device='cuda',
            verbose=False,
        )
        self.addCleanup(manager.clear)
        started = threading.Event()
        release = threading.Event()

        def planner():
            started.set()
            if not release.wait(timeout=2.0):
                raise TimeoutError("resident planner test was not released")
            return {"planned": True}

        future = manager.submit_resident_plan(17, planner)
        self.assertTrue(started.wait(timeout=1.0))
        self.assertFalse(future.done())
        release.set()
        self.assertEqual(future.result(timeout=2.0), {"planned": True})
        self.assertEqual(manager.get_stats()['resident_plan_jobs'], 1)

    def test_persistent_slots_keep_omega_and_fill_only_delta(self):
        manager = DoubleBufferGPUWorkingSet(
            num_total_gaussians=16,
            block_size=4,
            device='cuda',
            verbose=False,
        )
        self.addCleanup(manager.clear)

        source_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda')
        source_values = _components(source_ids)
        source = manager.active_buffer
        for name, tensor in source_values.items():
            setattr(source, name, tensor)
        source.local_to_global_idx = source_ids.clone()
        source.loaded_blocks = [0, 1]
        source.block_to_local_slice = {0: slice(0, 4), 1: slice(4, 8)}
        source.num_gaussians = 8

        cold_ids = torch.tensor([8, 9, 10, 11])
        reader = _Reader(
            BlockLayout.CACHE,
            {2: _pack_block(cold_ids, BlockLayout.CACHE)},
        )
        reuse_fences = []
        manager.start_prefetch(
            iteration=17,
            visible_block_ids=[1, 2],
            resident_block_ids=[1],
            evicted_block_ids=[0],
            block_reader=reader,
            defer_resident_copy=True,
            before_target_reuse=lambda: reuse_fences.append('activation'),
        )

        for name in source_values:
            getattr(source, name)[4:8].add_(10.0)
        copy_resident_blocks = manager._copy_resident_blocks
        manager._copy_resident_blocks = lambda *args, **kwargs: self.fail(
            "persistent Omega handoff must not copy resident tensors"
        )
        try:
            handoff = manager.finalize_retained_blocks([1], target='loading')
        finally:
            manager._copy_resident_blocks = copy_resident_blocks
        self.assertEqual(handoff.ready_blocks, 1)
        self.assertEqual(handoff.copied_blocks, 0)
        self.assertTrue(manager.wait_for_prefetch(17))
        self.assertEqual(reuse_fences, [])
        manager.swap_buffers()
        self.assertEqual(reuse_fences, ['activation'])

        self.assertIs(manager.active_buffer, source)
        self.assertEqual(manager.active_buffer.loaded_blocks, [2, 1])
        self.assertEqual(manager.active_buffer.block_to_local_slice[2], slice(0, 4))
        self.assertEqual(manager.active_buffer.block_to_local_slice[1], slice(4, 8))
        self.assertEqual(reader.calls, [[2]])

        expected_ids = torch.tensor([8, 9, 10, 11, 4, 5, 6, 7], device='cuda')
        expected = _components(expected_ids)
        for name in expected:
            expected[name][4:8].add_(10.0)
            torch.testing.assert_close(getattr(source, name), expected[name])
        torch.testing.assert_close(source.local_to_global_idx, expected_ids)
        self.assertEqual(manager.get_stats()['persistent_slot_updates'], 1)
        self.assertEqual(manager.get_stats()['persistent_omega_reuses'], 1)
        self.assertEqual(manager.get_stats()['resident_refresh_blocks'], 0)

    def test_block_reader_prefetch_returns_before_read_and_defers_resident_copy(self):
        manager = DoubleBufferGPUWorkingSet(
            num_total_gaussians=12,
            block_size=4,
            device='cuda',
            verbose=False,
        )
        self.addCleanup(manager.clear)

        source_ids = torch.tensor([0, 1, 2, 3], device='cuda')
        source_values = _components(source_ids)
        source = manager.active_buffer
        for name, tensor in source_values.items():
            setattr(source, name, tensor)
        source.local_to_global_idx = source_ids
        source.loaded_blocks = [0]
        source.block_to_local_slice = {0: slice(0, 4)}
        source.num_gaussians = 4

        cold_ids = torch.tensor([4, 5, 6, 7])
        reader = _BlockingReader(
            BlockLayout.CACHE,
            {1: _pack_block(cold_ids, BlockLayout.CACHE)},
        )
        started_at = time.perf_counter()
        manager.start_prefetch(
            iteration=17,
            visible_block_ids=[0, 1],
            resident_block_ids=[0],
            block_reader=reader,
            defer_resident_copy=True,
        )
        elapsed = time.perf_counter() - started_at
        self.assertLess(elapsed, 0.5)
        self.assertTrue(reader.started.wait(timeout=1.0))

        for name in source_values:
            getattr(source, name).add_(10.0)
        refresh_started = time.perf_counter()
        try:
            handoff = manager.finalize_retained_blocks([0], target='loading')
            self.assertEqual(handoff.ready_blocks, 1)
            self.assertEqual(handoff.copied_blocks, 1)
            self.assertLess(time.perf_counter() - refresh_started, 1.5)
            self.assertFalse(reader.release.is_set())
        finally:
            reader.release.set()
        self.assertTrue(manager.wait_for_prefetch(17))

        target = manager.loading_buffer
        expected_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda')
        expected = _components(expected_ids)
        for name in expected:
            expected[name][:4].add_(10.0)
            torch.testing.assert_close(getattr(target, name), expected[name])


if __name__ == '__main__':
    unittest.main()
