import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
PIPELINE = ROOT / "submodules/gsplat/gsplat/distributed_pipeline.py"
ENGINE = ROOT / "strategies/tide_engine/distributed_engine.py"


class DistributedPipelineContractTest(unittest.TestCase):
    def test_pipeline_module_compiles_and_exposes_stages(self):
        source = PIPELINE.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(PIPELINE))
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        self.assertIn("DistributedRasterizationPipeline", names)
        self.assertIn("DistributedRasterizationHandle", names)
        self.assertIn("distributed_rasterization_pipeline", names)
        self.assertIn("_projection_prepare", names)
        self.assertIn("_forward_exchange", names)
        self.assertIn("_render_prepared", names)

    def test_stage_order_is_explicit(self):
        source = PIPELINE.read_text(encoding="utf-8")
        self.assertIn("stream.wait_event(self.projection_done)", source)
        self.assertIn("compute_stream.wait_event(self.exchange_done)", source)
        self.assertIn("_forward_exchange(self.state, stream, self.microbatch_index)", source)
        self.assertIn("_render_prepared(self.state)", source)

    def test_engine_keeps_off_path_and_pc1_switch(self):
        source = ENGINE.read_text(encoding="utf-8")
        self.assertIn('pipeline_mode not in {"off", "pc1"}', source)
        self.assertIn('if pipeline_mode == "pc1":', source)
        self.assertIn('if pipeline_mode == "pc1":\n            continue', source)

    def test_no_global_cuda_synchronize_added(self):
        source = PIPELINE.read_text(encoding="utf-8")
        self.assertNotIn("torch.cuda.synchronize", source)

    def test_camera_layout_is_checked_before_tensor_gather(self):
        source = PIPELINE.read_text(encoding="utf-8")
        self.assertIn("class BatchCameraLayout", source)
        self.assertIn("def prepare_batch_camera_layout", source)
        self.assertIn("C_world = [int(value) for value in all_gather_int32", source)
        self.assertIn("N_world = [int(value) for value in all_gather_int32", source)
        self.assertIn("viewmats_global, Ks_global = all_gather_tensor_list", source)
        self.assertIn("reshape(\n            self.world_size, self.local_camera_count", source)
        layout_source = source.split("def prepare_batch_camera_layout", 1)[1].split(
            "def _projection_prepare", 1
        )[0]
        self.assertEqual(layout_source.count("all_gather_int32"), 2)
        self.assertEqual(layout_source.count("all_gather_tensor_list"), 1)
        projection_source = source.split("def _projection_prepare", 1)[1]
        self.assertNotIn("all_gather_int32", projection_source)
        self.assertNotIn("all_gather_tensor_list", projection_source)
        self.assertIn("pc1 requires the same number of local cameras", source)

    def test_completed_handles_are_snapshotted(self):
        source = ENGINE.read_text(encoding="utf-8")
        self.assertIn("pc1_current.snapshot()", source)
        self.assertIn("pc1_current.mark_backward_done(backward_end)", source)
        self.assertIn('"PC1 pipeline drain", pc1_pipeline.drain', source)
        self.assertIn("pc1_pipeline.close()", source)
        self.assertIn('phase="S1 pc1 pipeline drain"', source)

    def test_pipeline_rejects_submission_after_drain_or_close(self):
        source = PIPELINE.read_text(encoding="utf-8")
        self.assertIn("cannot submit projection after pipeline shutdown", source)

    def test_forward_loss_boundaries_and_error_group(self):
        source = ENGINE.read_text(encoding="utf-8")
        self.assertIn("loss_start = torch.cuda.Event", source)
        self.assertIn('"loss_ms": loss_ms', source)
        self.assertIn('timing_source="cuda_event_relative_to_batch_anchor"', source)
        self.assertIn("group=pc1_error_group", source)
        self.assertIn("S1 pc1 local camera layout preparation", source)
        self.assertIn("S1 pc1 batch camera layout collective", source)
        self.assertIn('f"Tide:S1:MB{index}:Loss"', source)
        self.assertIn('pc1_current.trace_backward(loss)', source)
        timeline_source = source.split(
            "    if timeline_enabled:\n        anchor_event", 1
        )[1]
        pc1_timeline = timeline_source.split('if pipeline_mode == "pc1":', 1)[1]
        pc1_timeline = pc1_timeline.split("else:", 1)[0]
        self.assertNotIn("gaussian_cull_forward", pc1_timeline)
        self.assertNotIn("host_submit_plus_cuda_event", pc1_timeline)

    def test_error_group_is_separate_from_default_process_group(self):
        source = (ROOT / "utils/distributed.py").read_text(encoding="utf-8")
        self.assertIn('dist.new_group(backend="gloo")', source)
        self.assertIn("dist.all_gather_object(gathered, value, group=group)", source)
        self.assertIn("dist.destroy_process_group(self.error_group)", source)

    def test_metrics_schema_contains_loss_duration(self):
        metrics = (ROOT / "strategies/tide_engine/distributed_metrics.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"loss_ms",', metrics)
        self.assertIn('float(row.get("loss_ms", 0.0))', metrics)


if __name__ == "__main__":
    unittest.main()
