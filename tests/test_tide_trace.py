from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_pipeline_ranges_use_balanced_context_managers():
    for relative_path in (
        "strategies/tide_engine/engine.py",
        "strategies/tide_engine/runtime.py",
        "train_tidegs.py",
    ):
        source = _source(relative_path)
        assert "range_push(" not in source
        assert "range_pop(" not in source


def test_trace_names_are_backend_neutral_and_fine_grained():
    source = "\n".join(
        _source(path)
        for path in (
            "strategies/tide_engine/engine.py",
            "strategies/tide_engine/runtime.py",
            "utils/distributed.py",
        )
    )
    for obsolete in (
        "tide.cpu.materialize",
        "tide.gpu.cull.block",
        "tide.comm.nccl",
        "tide.gpu.forward.projection_cull",
        "tide.gpu.forward.render",
    ):
        assert obsolete not in source
    for required in (
        "tide.cull.block",
        "tide.cull.gaussian_filter",
        "tide.comm.object_collective",
        "tide.gpu.forward.projection",
        "tide.gpu.forward.sh",
        "tide.gpu.forward.tile_mask",
        "tide.gpu.forward.isect",
        "tide.gpu.forward.rasterize",
        "tide.gpu.backward.scatter",
        "tide.gpu.optimizer.update",
    ):
        assert required in source


def test_batch_and_camera_ranges_cover_their_named_work():
    source = _source("train_tidegs.py")
    assert 'with tide_range("tide.batch.step"):' in source
    assert 'with tide_range("tide.batch.compute"):' in source
    assert 'with tide_range("tide.cpu.camera.schedule"):' in source
    assert 'with tide_range("tide.wait.camera_prefetch"):' in source


def test_wait_and_order_ranges_do_not_use_ambiguous_names():
    source = "\n".join(
        _source(path)
        for path in (
            "strategies/tide_engine/engine.py",
            "strategies/tide_engine/gpu_working_set.py",
            "storage/tide_storage_adapter.py",
            "storage/pure_ssd_checkpoint.py",
        )
    )
    assert 'tide.cpu.order.tsp' in source
    assert 'tide.wait.writeback' not in source


def test_writeback_submit_is_at_queue_boundary_only():
    source = _source("storage/tide_storage_adapter.py")
    start = source.index("    def submit_cache_writeback(")
    end = source.index("\n    def ", start + 1)
    function_source = source[start:end]
    assert function_source.count("tide.io.ssd.write.submit") == 1
    assert "self._cache_commit_queue.put(job)" in function_source


def test_legacy_metrics_and_events_remain_present():
    source = _source("strategies/tide_engine/engine.py")
    for historical_name in (
        "gaussian_cull_forward",
        "gaussian_projection_cull_ms",
        "backward_ms",
        "optimizer_ms",
        "block_cull_backend",
    ):
        assert historical_name in source


def test_tide_range_balances_normal_path_without_sync():
    torch = pytest.importorskip("torch")
    from utils.tide_trace import tide_range

    with patch.object(torch.cuda.nvtx, "range_push") as push, patch.object(
        torch.cuda.nvtx, "range_pop"
    ) as pop, patch.object(torch.cuda, "synchronize") as synchronize:
        with tide_range("tide.test"):
            pass

    push.assert_called_once_with("tide.test")
    pop.assert_called_once_with()
    synchronize.assert_not_called()


def test_tide_range_pops_on_exception():
    torch = pytest.importorskip("torch")
    from utils.tide_trace import tide_range

    with patch.object(torch.cuda.nvtx, "range_push") as push, patch.object(
        torch.cuda.nvtx, "range_pop"
    ) as pop:
        with pytest.raises(RuntimeError, match="trace failure"):
            with tide_range("tide.test"):
                raise RuntimeError("trace failure")

    push.assert_called_once_with("tide.test")
    pop.assert_called_once_with()
