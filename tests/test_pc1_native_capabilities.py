"""Behavioral capability checks with no torch/CUDA dependency."""

import types
import unittest

from utils.pc1_native_capabilities import (
    CLM_GIL_OPS, GSPLAT_GIL_OPS, validate_pc1_native_capabilities,
)


def native_module(ops, binding, path):
    result = types.SimpleNamespace(
        __file__=path, PC1_NATIVE_ABI_VERSION=1, PC1_NATIVE_BINDING=binding,
        PC1_GIL_RELEASED_OPS=tuple(ops), PC1_SSIM_CURRENT_STREAM=True,
    )
    for op in ops:
        setattr(result, op, lambda: None)
    return result


class Pc1NativeCapabilitiesTest(unittest.TestCase):
    def setUp(self):
        self.gsplat = native_module(GSPLAT_GIL_OPS, "gsplat/cuda/ext.cpp", "/actual/gsplat.so")
        self.clm = native_module(CLM_GIL_OPS + ("tile_contribution_mask",),
                                 "clm_kernels/ext.cpp", "/actual/clm.so")

    def validate(self, **kwargs):
        return validate_pc1_native_capabilities(self.gsplat, self.clm, **kwargs)

    def test_reports_actual_extension_paths_versions_and_operations(self):
        result = self.validate(require_tile_mask=True, runtime={"code_commit": "test-sha"})
        self.assertEqual(result["gsplat"]["path"], "/actual/gsplat.so")
        self.assertEqual(result["clm_kernels"]["path"], "/actual/clm.so")
        self.assertEqual(result["gsplat"]["abi_version"], 1)
        self.assertTrue(result["clm_kernels"]["ssim_current_stream"])
        self.assertEqual(result["runtime"]["code_commit"], "test-sha")
        self.assertIn("projection_ewa_3dgs_packed_fwd", result["gsplat"]["gil_released_ops"])

    def test_both_old_extensions_fail_with_paths_and_rebuild_instructions(self):
        del self.gsplat.PC1_NATIVE_ABI_VERSION
        del self.clm.PC1_NATIVE_ABI_VERSION
        with self.assertRaisesRegex(RuntimeError, "Rebuild BOTH extensions") as error:
            self.validate()
        self.assertIn("/actual/gsplat.so", str(error.exception))
        self.assertIn("/actual/clm.so", str(error.exception))
        self.assertIn("build_gsplat.py", str(error.exception))

    def test_missing_gil_operation_fails(self):
        self.gsplat.PC1_GIL_RELEASED_OPS = GSPLAT_GIL_OPS[1:]
        with self.assertRaisesRegex(RuntimeError, "missing GIL release: projection_ewa_3dgs_packed_fwd"):
            self.validate()

    def test_advertised_but_absent_native_operation_fails(self):
        del self.clm.fusedssim_backward
        with self.assertRaisesRegex(RuntimeError, "missing native entry points: fusedssim_backward"):
            self.validate()

    def test_missing_current_stream_capability_fails(self):
        self.clm.PC1_SSIM_CURRENT_STREAM = False
        with self.assertRaisesRegex(RuntimeError, "PC1_SSIM_CURRENT_STREAM=True"):
            self.validate()

    def test_legacy_binding_cannot_claim_modern_capability(self):
        self.gsplat.PC1_NATIVE_BINDING = "gsplat/cuda/csrc/ext.cpp"
        with self.assertRaisesRegex(RuntimeError, "PC1_NATIVE_BINDING"):
            self.validate()

    def test_tile_mask_requirement_matches_enabled_path(self):
        self.clm.PC1_GIL_RELEASED_OPS = CLM_GIL_OPS
        self.validate(require_tile_mask=False)
        with self.assertRaisesRegex(RuntimeError, "missing GIL release: tile_contribution_mask"):
            self.validate(require_tile_mask=True)

    def test_malformed_operation_list_fails_cleanly(self):
        self.gsplat.PC1_GIL_RELEASED_OPS = [3, None]
        with self.assertRaisesRegex(RuntimeError, "sequence of operation names"):
            self.validate()

    def test_missing_loaded_module_fails_cleanly(self):
        with self.assertRaisesRegex(RuntimeError, "<unavailable>"):
            validate_pc1_native_capabilities(None, self.clm)


if __name__ == "__main__":
    unittest.main()
