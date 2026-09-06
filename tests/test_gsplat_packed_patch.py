import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "tools" / "patch_gsplat_distributed_packed.py"
SPEC = importlib.util.spec_from_file_location("patch_gsplat_distributed_packed", SCRIPT_PATH)
PATCH_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH_MODULE)


class GsplatPackedPatchTest(unittest.TestCase):
    @staticmethod
    def _gsplat_153_layout():
        return (
            "def rasterization():\n"
            "    meta = {}\n"
            "    with_ut = False\n"
            "    compensations = None\n"
            "    if True:\n"
            f"        {PATCH_MODULE.BUGGY_IMAGE_IDS}\n"
            "        if True:\n"
            f"{PATCH_MODULE.ORIGINAL}"
            "            pass\n"
            f"{PATCH_MODULE.PROJECTION_START_ANCHOR}"
            "        pass\n"
            f"{PATCH_MODULE.PROJECTION_END_ANCHOR}"
            "        pass\n"
            f"{PATCH_MODULE.TILE_CONTRIBUTION_ANCHOR}"
            '            "x": 1,\n'
            "        }\n"
            "    )\n"
        )

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

    def test_projection_timing_rejects_unknown_existing_patch(self):
        with self.assertRaisesRegex(RuntimeError, "unknown existing"):
            PATCH_MODULE.patch_projection_timing_source(
                f"# {PATCH_MODULE.PROJECTION_TIMING_MARKER}\n"
            )

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

    def test_inserts_tile_contribution_patch_once(self):
        source = f"before\n{PATCH_MODULE.TILE_CONTRIBUTION_ANCHOR}after\n"
        patched, changed = PATCH_MODULE.patch_tile_contribution_source(source)
        self.assertTrue(changed)
        self.assertIn(PATCH_MODULE.TILE_CONTRIBUTION_BODY, patched)
        patched_again, changed_again = (
            PATCH_MODULE.patch_tile_contribution_source(patched)
        )
        self.assertFalse(changed_again)
        self.assertEqual(patched_again, patched)

    def test_tile_patch_rejects_unknown_existing_patch(self):
        with self.assertRaisesRegex(RuntimeError, "unknown existing"):
            PATCH_MODULE.patch_tile_contribution_source(
                f"# {PATCH_MODULE.TILE_CONTRIBUTION_MARKER}\n"
            )

    def test_tile_patch_rejects_legacy_secondary_rasterizer_anchor(self):
        secondary = (
            f"{PATCH_MODULE.TILE_CONTRIBUTION_PREFIX}"
            "    # Identify intersecting tiles\n"
        )
        with self.assertRaisesRegex(RuntimeError, "unknown source layout"):
            PATCH_MODULE.patch_tile_contribution_source(secondary)

    def test_tile_patch_migrates_old_owner_survivor_layout(self):
        source = (
            f"{PATCH_MODULE.TILE_CONTRIBUTION_PREFIX}"
            f"{PATCH_MODULE.OWNER_SURVIVOR_LEGACY_PATCHED}"
            "            pass\n"
            "        }\n"
            "    )\n"
        )
        tiled, tile_changed = PATCH_MODULE.patch_tile_contribution_source(source)
        migrated, owner_changed = PATCH_MODULE.patch_owner_survivor_source(tiled)
        self.assertTrue(tile_changed)
        self.assertTrue(owner_changed)
        self.assertIn(PATCH_MODULE.TILE_CONTRIBUTION_BODY, migrated)
        self.assertIn(PATCH_MODULE.OWNER_SURVIVOR_PATCHED, migrated)
        self.assertNotIn(PATCH_MODULE.OWNER_SURVIVOR_LEGACY_PATCHED, migrated)

    def test_full_patch_compiles_gsplat_153_layout(self):
        source = self._gsplat_153_layout()
        for patcher in (
            PATCH_MODULE.patch_rendering_source,
            PATCH_MODULE.patch_projection_timing_source,
            PATCH_MODULE.patch_tile_contribution_source,
            PATCH_MODULE.patch_owner_survivor_source,
        ):
            source, changed = patcher(source)
            self.assertTrue(changed)
        compile(source, "gsplat-1.5.3-rendering.py", "exec")
        for patcher in (
            PATCH_MODULE.patch_rendering_source,
            PATCH_MODULE.patch_projection_timing_source,
            PATCH_MODULE.patch_tile_contribution_source,
            PATCH_MODULE.patch_owner_survivor_source,
        ):
            source_again, changed = patcher(source)
            self.assertFalse(changed)
            self.assertEqual(source_again, source)


if __name__ == "__main__":
    unittest.main()
