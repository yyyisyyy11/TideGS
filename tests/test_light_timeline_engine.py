"""Source-level guards for the profiling switch in the GPU training path."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_light_mode_keeps_batch_and_storage_metrics_without_micro_trace():
    train = (ROOT / "train_tidegs.py").read_text(encoding="utf-8")
    engine = (
        ROOT / "strategies/tide_engine/distributed_engine.py"
    ).read_text(encoding="utf-8")

    assert 'args.tide_detailed_metrics or getattr(args, "tide_light_timeline", False)' in train
    assert "enable_timing=metrics_enabled" in train
    assert "if metrics_enabled\n        else None" in train
    assert train.count("if metrics_enabled:") >= 2

    assert "timeline_enabled = bool(metrics_writer.detailed_timeline_enabled)" in engine
    assert engine.count("capture_timeline=timeline_enabled") == 2
    assert 'if timeline_enabled:\n        with tide_range("tide.cpu.metrics.timeline_emit"):' in engine
    assert 'if stage == "AccumulateGrad" and not timeline_enabled:' in engine
    assert 'backward_stage_ms[stage] = None' in engine
