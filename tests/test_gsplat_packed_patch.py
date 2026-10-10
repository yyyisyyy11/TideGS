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
            "    means = quats = scales = opacities = colors = None\n"
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
            f"{PATCH_MODULE.PARTIAL_GRADIENT_RETURN_ANCHOR}"
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

    def test_inserts_partial_gradient_hooks_once(self):
        source = (
            f"before\n{PATCH_MODULE.PATCHED}"
            f"{PATCH_MODULE.PARTIAL_GRADIENT_RETURN_ANCHOR}"
        )
        patched, changed = PATCH_MODULE.patch_partial_gradient_source(source)
        self.assertTrue(changed)
        self.assertIn(PATCH_MODULE.PARTIAL_GRADIENT_CALL_MARKER, patched)
        self.assertIn(PATCH_MODULE.PARTIAL_GRADIENT_MARKER, patched)
        patched_again, changed_again = PATCH_MODULE.patch_partial_gradient_source(
            patched
        )
        self.assertFalse(changed_again)
        self.assertEqual(patched_again, patched)

    def test_partial_gradient_rejects_unknown_source_layout(self):
        with self.assertRaisesRegex(RuntimeError, "unknown source layout"):
            PATCH_MODULE.patch_partial_gradient_source("unrelated source")

    def test_partial_gradient_picks_the_return_in_the_receiving_function(self):
        """真实 gsplat 1.5.3 里 return 锚点出现两次。

        `rasterization()`（公开入口，含分布式相机修正）和它调用的 `_rasterization()`
        各有一个 `return render_colors, render_alphas, meta`。要挂 meta 的是**前者**。
        这条测试用同样的布局，断言补丁落在接收点所在的那个函数里、且只落一次。
        """
        anchor = PATCH_MODULE.PARTIAL_GRADIENT_RETURN_ANCHOR
        source = (
            f"def rasterization(\n"
            f"{PATCH_MODULE.PATCHED}"
            f"{anchor}"                       # ← 正确的插入点（接收点所在函数）
            f"\n\ndef _rasterization(\n"
            f"    meta = {{}}\n"
            f"{anchor}"                       # ← 另一个函数里的同名 return
        )
        patched, changed = PATCH_MODULE.patch_partial_gradient_source(source)
        self.assertTrue(changed)
        head, _, tail = patched.partition("\ndef _rasterization(")
        # 前半段（rasterization）里应当有标记，后半段（_rasterization）里不该有
        self.assertIn(PATCH_MODULE.PARTIAL_GRADIENT_MARKER, head)
        self.assertNotIn(PATCH_MODULE.PARTIAL_GRADIENT_MARKER, tail)
        # 两个函数的 return 都还在（只多了插入的那几行，没有吃掉 return）
        self.assertEqual(patched.count(anchor), 2)

    def test_partial_gradient_rejects_return_outside_the_receiving_function(self):
        anchor = PATCH_MODULE.PARTIAL_GRADIENT_RETURN_ANCHOR
        source = (
            f"{PATCH_MODULE.PATCHED}"
            f"\n\ndef _rasterization(\n"
            f"    meta = {{}}\n"
            f"{anchor}"                       # 接收点之后已经没有同函数的 return 了
        )
        with self.assertRaisesRegex(RuntimeError, "different function"):
            PATCH_MODULE.patch_partial_gradient_source(source)

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
            PATCH_MODULE.patch_partial_gradient_source,
        ):
            source, changed = patcher(source)
            self.assertTrue(changed)
        compile(source, "gsplat-1.5.3-rendering.py", "exec")
        for patcher in (
            PATCH_MODULE.patch_rendering_source,
            PATCH_MODULE.patch_projection_timing_source,
            PATCH_MODULE.patch_tile_contribution_source,
            PATCH_MODULE.patch_owner_survivor_source,
            PATCH_MODULE.patch_partial_gradient_source,
        ):
            source_again, changed = patcher(source)
            self.assertFalse(changed)
            self.assertEqual(source_again, source)


if __name__ == "__main__":
    unittest.main()
