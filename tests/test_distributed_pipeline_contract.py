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
        self.assertIn("C_world = all_gather_int32", source)
        self.assertIn("pc1 requires the same number of local cameras", source)

    def test_completed_handles_are_snapshotted(self):
        source = ENGINE.read_text(encoding="utf-8")
        self.assertIn("pc1_current.snapshot()", source)
        self.assertIn("pc1_pipeline.close()", source)


if __name__ == "__main__":
    unittest.main()
