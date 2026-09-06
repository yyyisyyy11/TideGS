import importlib.util
import sys
import types
import unittest
from pathlib import Path


_installed_fake_torch = False
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.SimpleNamespace()
    _installed_fake_torch = True


MODULE_PATH = (
    Path(__file__).parents[1]
    / "strategies"
    / "tide_engine"
    / "gsplat_backend.py"
)
SPEC = importlib.util.spec_from_file_location("gsplat_backend_under_test", MODULE_PATH)
BACKEND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKEND)
if _installed_fake_torch:
    sys.modules.pop("torch", None)


def _unpatched_rasterization():
    return None


def _patched_rasterization():
    _tide_projection_events = None
    return _tide_projection_events


def _tile_patched_rasterization():
    _tide_owner_tile_survivor_mask = None
    _tide_tile_contribution_mask = None
    _tide_original_projection_radii = None
    _tide_radius_mask = None
    return (
        _tide_owner_tile_survivor_mask,
        _tide_tile_contribution_mask,
        _tide_original_projection_radii,
        _tide_radius_mask,
    )


class _Event:
    def __init__(self, elapsed=0.0):
        self.elapsed = float(elapsed)
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True

    def elapsed_time(self, other):
        return other.elapsed


class GsplatBackendGuardTest(unittest.TestCase):
    def test_detailed_metrics_reject_missing_projection_patch(self):
        gsplat = types.SimpleNamespace(rasterization=_unpatched_rasterization)
        with self.assertRaisesRegex(RuntimeError, "projection timing patch"):
            BACKEND._require_projection_timing_fix(gsplat)

    def test_detailed_metrics_accept_projection_patch(self):
        gsplat = types.SimpleNamespace(rasterization=_patched_rasterization)
        BACKEND._require_projection_timing_fix(gsplat)

    def test_tile_mask_rejects_incomplete_patch(self):
        gsplat = types.SimpleNamespace(rasterization=_unpatched_rasterization)
        with self.assertRaisesRegex(RuntimeError, "tile-mask patch"):
            BACKEND._require_tile_contribution_fix(gsplat)

    def test_tile_mask_accepts_complete_patch(self):
        gsplat = types.SimpleNamespace(rasterization=_tile_patched_rasterization)
        BACKEND._require_tile_contribution_fix(gsplat)

    def test_projection_elapsed_time_is_nonnegative(self):
        start = _Event()
        end = _Event(3.5)
        self.assertEqual(
            BACKEND.projection_elapsed_ms(
                {BACKEND.PROJECTION_TIMING_FIX: (start, end)}
            ),
            3.5,
        )
        self.assertTrue(end.synchronized)


if __name__ == "__main__":
    unittest.main()
