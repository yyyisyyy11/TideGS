import unittest

import torch

from storage.block_reader import BlockLayout, _pack_block_batch
from strategies.tide_engine.gpu_working_set import GPUWorkingSet


def _components(global_ids):
    rows = global_ids.to(dtype=torch.float32).unsqueeze(1)
    return {
        "xyz": rows + torch.tensor([0.1, 0.2, 0.3], device=rows.device),
        "scaling": rows + torch.tensor([1.1, 1.2, 1.3], device=rows.device),
        "rotation": rows
        + torch.tensor([2.1, 2.2, 2.3, 2.4], device=rows.device),
        "opacity": rows + 3.1,
        "features_dc": rows
        + torch.tensor([4.1, 4.2, 4.3], device=rows.device),
        "features_rest": rows
        + torch.arange(45, device=rows.device) * 0.01
        + 5.0,
    }


def _cache_block(block_id, block_size, total_gaussians):
    start = block_id * block_size
    end = min(start + block_size, total_gaussians)
    values = _components(torch.arange(start, end))
    return torch.cat(
        [
            values["xyz"],
            values["scaling"],
            values["rotation"],
            values["opacity"],
            values["features_dc"],
            values["features_rest"],
        ],
        dim=1,
    )


class _Reader:
    layout = BlockLayout.CACHE

    def __init__(self, total_gaussians, block_size):
        self.total_gaussians = total_gaussians
        self.block_size = block_size
        self.num_blocks = (total_gaussians + block_size - 1) // block_size
        self.blocks = {
            block_id: _cache_block(block_id, block_size, total_gaussians)
            for block_id in range(self.num_blocks)
        }
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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class GPUWorkingSetRetentionTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.set_device(0)

    def _manager(self, total_gaussians=16, block_size=4):
        manager = GPUWorkingSet(
            num_total_gaussians=total_gaussians,
            block_size=block_size,
            device="cuda",
            verbose=False,
        )
        self.addCleanup(manager.clear)
        return manager

    def _load(self, manager, reader, blocks, enable_retention):
        tensors, stats = manager.load_visible_blocks_with_retention(
            visible_block_ids=blocks,
            enable_retention=enable_retention,
            block_reader=reader,
            allow_gpu_hotspots=True,
        )
        torch.cuda.synchronize()
        return tensors, stats

    @staticmethod
    def _bind_as_parameters(manager, tensors):
        attributes = {
            "xyz": "gpu_xyz",
            "scaling": "gpu_scaling",
            "rotation": "gpu_rotation",
            "opacity": "gpu_opacity",
            "features_dc": "gpu_features_dc",
            "features_rest": "gpu_features_rest",
        }
        for name, attribute in attributes.items():
            setattr(
                manager,
                attribute,
                torch.nn.Parameter(tensors[name].requires_grad_(True)),
            )
        return manager._persistent_component_tensors()

    def test_empty_resident_set_supports_collective_sentinel(self):
        manager = self._manager()
        reader = _Reader(total_gaussians=16, block_size=4)

        tensors, empty_stats = self._load(
            manager,
            reader,
            [],
            enable_retention=False,
        )

        self.assertEqual(
            {name: tuple(tensor.shape) for name, tensor in tensors.items()},
            {
                "xyz": (0, 3),
                "scaling": (0, 3),
                "rotation": (0, 4),
                "opacity": (0, 1),
                "features_dc": (0, 3),
                "features_rest": (0, 45),
            },
        )
        self.assertEqual(empty_stats["total_count"], 0)
        self.assertEqual(empty_stats["hit_rate"], 0.0)
        self.assertEqual(manager.loaded_blocks, [])
        self.assertEqual(manager.block_to_gpu_slice, {})

        tensors, _ = self._load(manager, reader, [0], enable_retention=True)
        self._bind_as_parameters(manager, tensors)
        tensors, empty_stats = self._load(
            manager,
            reader,
            [],
            enable_retention=True,
        )

        self.assertEqual(empty_stats["total_count"], 0)
        self.assertEqual(empty_stats["num_gaussians"], 0)
        self.assertEqual(manager.loaded_blocks, [])
        self.assertEqual(manager.block_to_gpu_slice, {})
        self.assertTrue(bool((manager.local_to_global_idx == -1).all()))

    def test_retained_block_stays_in_place_and_only_delta_is_read(self):
        manager = self._manager()
        reader = _Reader(total_gaussians=16, block_size=4)
        tensors, first_stats = self._load(
            manager,
            reader,
            [0, 1],
            enable_retention=False,
        )
        tensors = self._bind_as_parameters(manager, tensors)
        pointers = {name: tensor.data_ptr() for name, tensor in tensors.items()}
        retained_slice = manager.block_to_gpu_slice[1]
        with torch.no_grad():
            manager.gpu_xyz[retained_slice].add_(10.0)
        retained_xyz = manager.gpu_xyz[retained_slice].clone()

        tensors, second_stats = self._load(
            manager,
            reader,
            [1, 2],
            enable_retention=True,
        )

        self.assertEqual(reader.calls, [[0, 1], [2]])
        self.assertEqual(manager.block_to_gpu_slice[1], retained_slice)
        self.assertEqual(manager.block_to_gpu_slice[2], slice(0, 4))
        self.assertEqual(
            {name: tensor.data_ptr() for name, tensor in tensors.items()},
            pointers,
        )
        torch.testing.assert_close(manager.gpu_xyz[retained_slice], retained_xyz)
        self.assertEqual(first_stats["gpu_slot_growth_blocks"], 2)
        self.assertEqual(second_stats["gpu_slot_growth_blocks"], 0)
        self.assertEqual(second_stats["hotspot_count"], 1)
        self.assertEqual(second_stats["cold_count"], 1)
        self.assertEqual(second_stats["h2d_bytes"], 4 * 59 * 4)
        self.assertEqual(
            manager.local_to_global_idx[manager.block_to_gpu_slice[2]].tolist(),
            [8, 9, 10, 11],
        )

        _, third_stats = self._load(
            manager,
            reader,
            [1, 2],
            enable_retention=True,
        )
        self.assertEqual(reader.calls, [[0, 1], [2]])
        self.assertEqual(third_stats["cold_count"], 0)
        self.assertEqual(third_stats["h2d_bytes"], 0)

    def test_growth_preserves_retained_values_and_partial_block(self):
        manager = self._manager(total_gaussians=10, block_size=4)
        reader = _Reader(total_gaussians=10, block_size=4)
        tensors, _ = self._load(
            manager,
            reader,
            [0],
            enable_retention=False,
        )
        tensors = self._bind_as_parameters(manager, tensors)
        old_pointer = tensors["xyz"].data_ptr()
        with torch.no_grad():
            manager.gpu_xyz[manager.block_to_gpu_slice[0]].add_(20.0)
        retained_xyz = manager.gpu_xyz[
            manager.block_to_gpu_slice[0]
        ].clone()

        tensors, stats = self._load(
            manager,
            reader,
            [0, 1, 2],
            enable_retention=True,
        )

        self.assertNotEqual(tensors["xyz"].data_ptr(), old_pointer)
        self.assertEqual(stats["gpu_slot_capacity_blocks"], 3)
        self.assertEqual(stats["gpu_slot_growth_blocks"], 2)
        torch.testing.assert_close(
            manager.gpu_xyz[manager.block_to_gpu_slice[0]],
            retained_xyz,
        )
        self.assertEqual(
            manager.block_to_gpu_slice[2].stop
            - manager.block_to_gpu_slice[2].start,
            2,
        )
        self.assertEqual(
            manager.local_to_global_idx[manager.block_to_gpu_slice[2]].tolist(),
            [8, 9],
        )

    def test_slot_overwrite_waits_for_evicted_block_d2h(self):
        manager = self._manager(total_gaussians=8, block_size=4)
        reader = _Reader(total_gaussians=8, block_size=4)
        tensors, _ = self._load(manager, reader, [0], enable_retention=False)
        self._bind_as_parameters(manager, tensors)
        expected = reader.blocks[0].clone()

        payload = manager.stage_updated_blocks([0])
        self.assertIsNotNone(payload)
        self._load(manager, reader, [1], enable_retention=True)
        written_back = payload.wait()[0]
        payload.release()

        torch.testing.assert_close(written_back, expected)
        torch.testing.assert_close(
            manager.gpu_xyz[manager.block_to_gpu_slice[1]],
            _components(torch.arange(4, 8, device="cuda"))["xyz"],
        )


if __name__ == "__main__":
    unittest.main()
