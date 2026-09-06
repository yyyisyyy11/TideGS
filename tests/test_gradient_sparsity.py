import unittest

from strategies.tide_engine.gradient_schema import (
    GRADIENT_COMPONENTS,
    GRADIENT_WIDTHS,
    active_parameter_width,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None


class ActiveParameterWidthTest(unittest.TestCase):
    def test_active_sh_widths(self):
        self.assertEqual(
            [active_parameter_width(degree) for degree in range(4)],
            [14, 23, 38, 59],
        )

    def test_active_sh_degree_range_is_checked(self):
        for degree in (-1, 4):
            with self.assertRaisesRegex(ValueError, "active_sh_degree"):
                active_parameter_width(degree)


@unittest.skipIf(torch is None, "torch is unavailable")
class GradientSparsityProfileTest(unittest.TestCase):
    @staticmethod
    def _components(rows=3):
        return {
            name: torch.zeros((rows, width), dtype=torch.float32)
            for name, width in GRADIENT_WIDTHS.items()
        }

    def test_active_sh_profile_ignores_inactive_rest_coefficients(self):
        from strategies.tide_engine.gradient_sparsity import (
            profile_gradient_sparsity,
        )

        components = self._components(rows=2)
        components["features_rest"][0, -1] = 1.0
        components["features_rest"][1, 0] = 1.0
        profile = profile_gradient_sparsity(
            components,
            torch.tensor([True, True]),
            prefix="test",
            active_sh_degree=0,
            include_active=True,
        )
        stats = profile["stats"]
        self.assertEqual(int(stats["test_all_zero_gradient_gaussians"]), 0)
        self.assertEqual(int(stats["test_active_all_zero_gradient_gaussians"]), 2)
        self.assertEqual(int(stats["test_active_parameter_elements"]), 28)

    def test_tile_survivor_profile_and_sample_rows(self):
        from strategies.tide_engine.gradient_sparsity import (
            profile_gradient_sparsity,
        )

        components = self._components()
        components["xyz"][0, 0] = 1.0
        components["opacity"][2, 0] = 2.0
        profile = profile_gradient_sparsity(
            components,
            torch.tensor([True, False, True]),
            prefix="tile_mask_keep",
            active_sh_degree=3,
            sample_rows=-1,
            include_active=True,
        )
        stats = profile["stats"]
        self.assertEqual(int(stats["tile_mask_keep_unique_gaussians"]), 2)
        self.assertEqual(
            int(stats["tile_mask_keep_nonfinite_gradient_elements"]), 0
        )
        self.assertEqual(profile["sample_owner_rows"].tolist(), [0, 2])
        self.assertEqual(tuple(profile["sample_gradients"].shape), (2, 59))


if __name__ == "__main__":
    unittest.main()
