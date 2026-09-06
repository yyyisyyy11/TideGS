import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace


RUNTIME_PATH = (
    Path(__file__).resolve().parents[1]
    / "strategies"
    / "tide_engine"
    / "runtime.py"
)


def _load_runtime_validator():
    tree = ast.parse(RUNTIME_PATH.read_text(encoding="utf-8"))
    names = {"_require_attr", "_require_lower", "validate_tide_runtime_args"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Any": object, "math": math}
    exec(compile(module, str(RUNTIME_PATH), "exec"), namespace)
    return namespace["validate_tide_runtime_args"]


def _valid_args(**overrides):
    values = {
        "pure_ssd_offload": True,
        "use_ssd_offload": True,
        "clm_offload": True,
        "naive_offload": False,
        "no_offload": False,
        "ssd_execution_mode": "paper",
        "paper_block_reader_backend": "tiered_cache",
        "paper_optimizer_backend": "gpu_resident",
        "paper_optimizer_state_mode": "resident_blocks",
        "paper_optimizer_deferred_mode": "off",
        "paper_free_unified_params": True,
        "disable_auto_densification": True,
        "tide_tile_contribution_mode": "off",
        "tide_tile_alpha_threshold": 1.0 / 255.0,
        "tide_grad_zero_metrics": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TileContributionArgsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validate = staticmethod(_load_runtime_validator())

    def test_off_and_apply_do_not_require_gradient_metrics(self):
        self.validate(_valid_args(tide_tile_contribution_mode="off"))
        self.validate(_valid_args(tide_tile_contribution_mode="apply"))

    def test_profile_requires_gradient_metrics(self):
        with self.assertRaisesRegex(RuntimeError, "requires"):
            self.validate(_valid_args(tide_tile_contribution_mode="profile"))
        self.validate(
            _valid_args(
                tide_tile_contribution_mode="profile",
                tide_grad_zero_metrics=True,
            )
        )

    def test_mode_and_threshold_are_bounded(self):
        with self.assertRaisesRegex(RuntimeError, "off, profile, or apply"):
            self.validate(_valid_args(tide_tile_contribution_mode="unknown"))
        for threshold in (0.0, -1.0, 1.0 / 255.0 + 1e-6, float("nan")):
            with self.assertRaisesRegex(RuntimeError, r"\(0, 1/255\]"):
                self.validate(
                    _valid_args(tide_tile_alpha_threshold=threshold)
                )


if __name__ == "__main__":
    unittest.main()
