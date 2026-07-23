import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "tools" / "patch_gsplat_distributed_packed.py"
SPEC = importlib.util.spec_from_file_location("patch_gsplat_distributed_packed", SCRIPT_PATH)
PATCH_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH_MODULE)


class GsplatPackedPatchTest(unittest.TestCase):
    def test_inserts_local_image_ids_once(self):
        source = (
            f"{PATCH_MODULE.BUGGY_IMAGE_IDS}\nbefore\n"
            f"{PATCH_MODULE.ORIGINAL}after\n"
        )
        patched, changed = PATCH_MODULE.patch_rendering_source(source)
        self.assertTrue(changed)
        self.assertIn(PATCH_MODULE.PATCHED, patched)

    def test_is_idempotent(self):
        source = (
            f"{PATCH_MODULE.BUGGY_IMAGE_IDS}\nbefore\n"
            f"{PATCH_MODULE.PATCHED}after\n"
        )
        patched, changed = PATCH_MODULE.patch_rendering_source(source)
        self.assertFalse(changed)
        self.assertEqual(patched, source)

    def test_rejects_unknown_source_layout(self):
        with self.assertRaisesRegex(RuntimeError, "unknown source layout"):
            PATCH_MODULE.patch_rendering_source("unrelated source")


if __name__ == "__main__":
    unittest.main()
