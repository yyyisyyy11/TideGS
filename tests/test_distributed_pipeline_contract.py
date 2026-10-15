"""Only critical negative source contracts; scheduling/transport run behavior tests."""

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]
PIPELINE = ROOT / 'submodules/gsplat/gsplat/distributed_pipeline.py'


class DistributedPipelineContractTest(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(PIPELINE.read_text(), filename=str(PIPELINE))

    def test_projection_cannot_submit_collectives(self):
        projection = next(node for node in self.tree.body
                          if isinstance(node, ast.FunctionDef) and node.name == '_projection_prepare')
        forbidden = {'all_to_all', 'all_to_all_single', 'all_gather',
                     'all_gather_int32', 'all_gather_tensor_list', 'all_reduce', 'barrier'}
        for call in (node for node in ast.walk(projection) if isinstance(node, ast.Call)):
            function = call.func
            name = function.id if isinstance(function, ast.Name) else getattr(function, 'attr', '')
            self.assertNotIn(name, forbidden)

    def test_pc1_has_no_implicit_differentiable_communication(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, 'torch.distributed.nn.functional')
                self.assertNotIn('all_to_all_tensor_list', [alias.name for alias in node.names])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertFalse(node.func.attr == 'synchronize',
                                 'PC1 must use bounded event queries, never global/device synchronization')


if __name__ == '__main__':
    unittest.main()
