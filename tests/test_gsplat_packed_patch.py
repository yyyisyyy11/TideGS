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

    def test_inserts_projection_timing_once(self):
        source = (
            f"before\n{PATCH_MODULE.PROJECTION_START_ANCHOR}"
            f"project\n{PATCH_MODULE.PROJECTION_END_ANCHOR}after\n"
        )
        patched, changed = PATCH_MODULE.patch_projection_timing_source(source)
        self.assertTrue(changed)
        self.assertIn(PATCH_MODULE.PROJECTION_TIMING_MARKER, patched)
        patched_again, changed_again = (
            PATCH_MODULE.patch_projection_timing_source(patched)
        )
        self.assertFalse(changed_again)
        self.assertEqual(patched_again, patched)

    def test_projection_timing_rejects_unknown_source_layout(self):
        with self.assertRaisesRegex(RuntimeError, "unknown source layout"):
            PATCH_MODULE.patch_projection_timing_source("unrelated source")

    def test_inserts_owner_projection_survivor_mask_once(self):
        source = f"before\n{PATCH_MODULE.OWNER_SURVIVOR_ANCHOR}after\n"
        patched, changed = PATCH_MODULE.patch_owner_survivor_source(source)
        self.assertTrue(changed)
        self.assertIn(PATCH_MODULE.OWNER_SURVIVOR_MARKER, patched)
        patched_again, changed_again = PATCH_MODULE.patch_owner_survivor_source(
            patched
        )
        self.assertFalse(changed_again)
        self.assertEqual(patched_again, patched)

    def test_owner_survivor_rejects_unknown_source_layout(self):
        with self.assertRaisesRegex(RuntimeError, "unknown source layout"):
            PATCH_MODULE.patch_owner_survivor_source("unrelated source")


if __name__ == "__main__":
    unittest.main()
