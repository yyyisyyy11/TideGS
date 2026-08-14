import unittest
from types import SimpleNamespace

from arguments import _apply_pure_ssd_release_defaults, _apply_tide_aliases


def _args(*, algorithm="adam", tide_algorithm="", optimizer_alias=""):
    return SimpleNamespace(
        pure_ssd_offload=True,
        paper_optimizer_algorithm=algorithm,
        tide_optimizer_algorithm=tide_algorithm,
        optimizer=optimizer_alias,
        use_ssd_offload=False,
        clm_offload=False,
        naive_offload=False,
        no_offload=False,
        pure_ssd_init_backend="auto",
        ssd_execution_mode="fast_ram",
        paper_block_reader_backend="auto",
        paper_optimizer_backend="cpu",
        paper_optimizer_state_mode="full_cpu",
        paper_free_unified_params=False,
        disable_auto_densification=False,
        sparse_adam=False,
        tide_optimizer_deferred_mode="",
        tide_resident_selection_policy="",
        tide_resident_lambda="",
        tide_resident_recency_decay="",
        tide_balanced_seed_fraction="",
        tide_resident_capacity_blocks="",
        tide_optimizer_state_mode="",
        tide_optimizer_backend="",
        tide_block_reader_backend="",
        tide_free_unified_params=False,
        tide_debug_logging=False,
    )


class OptimizerAlgorithmDefaultTest(unittest.TestCase):
    def test_pure_ssd_keeps_adam_as_default(self):
        args = _args()

        _apply_pure_ssd_release_defaults(args)

        self.assertEqual(args.paper_optimizer_algorithm, "adam")

    def test_explicit_3dgs2_tr_alias_is_preserved(self):
        args = _args(tide_algorithm="3dgs2_tr")

        _apply_tide_aliases(args)
        _apply_pure_ssd_release_defaults(args)

        self.assertEqual(args.paper_optimizer_algorithm, "3dgs2_tr")

    def test_explicit_adam_alias_is_preserved(self):
        args = _args(algorithm="3dgs2_tr", tide_algorithm="adam")

        _apply_tide_aliases(args)
        _apply_pure_ssd_release_defaults(args)

        self.assertEqual(args.paper_optimizer_algorithm, "adam")

    def test_public_optimizer_alias_is_supported(self):
        args = _args(optimizer_alias="3dgs2_tr")

        _apply_tide_aliases(args)

        self.assertEqual(args.paper_optimizer_algorithm, "3dgs2_tr")


if __name__ == "__main__":
    unittest.main()
