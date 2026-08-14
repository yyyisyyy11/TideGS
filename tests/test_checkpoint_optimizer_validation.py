import types
import unittest

from strategies.tide_engine.checkpoint_validation import (
    build_optimizer_provenance,
    validate_distributed_checkpoint_resume,
    validate_pure_ssd_checkpoint_optimizer,
)


def _args(**overrides):
    values = {
        "paper_optimizer_algorithm": "3dgs2_tr",
        "bsz": 16,
        "iterations": 30_000,
        "paper_optimizer_backend": "gpu_resident",
        "paper_optimizer_state_mode": "resident_blocks",
        "paper_sophia_beta1": 0.9,
        "paper_sophia_beta2": 0.999,
        "paper_sophia_curvature_interval": 10,
        "paper_sophia_hutchinson_samples": 1,
        "paper_sophia_curvature_estimator": "residual_vjp",
        "paper_sophia_gamma": 1.0,
        "paper_sophia_epsilon": 1e-15,
        "paper_sophia_curvature_seed": 17,
        "paper_tr_epsilon_init": 1e-6,
        "paper_tr_epsilon_final": 1e-8,
        "paper_tr_quat_norm": -1.0,
        "paper_resident_capacity_blocks": 2048,
        "paper_resident_selection_policy": "topc_balanced",
        "paper_resident_lambda": 0.3,
        "paper_resident_recency_decay": 0.95,
        "paper_balanced_seed_fraction": 0.25,
        "lambda_dssim": 0.2,
        "ssd_schedule_ordering": "trajectory",
        "tide_camera_assignment": "equal",
        "tide_camera_microbatch": 4,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _v3_root(args=None, **overrides):
    args = args or _args()
    root = {
        "checkpoint_version": 3,
        "world_size": 2,
        "global_bsz": 16,
        "next_iteration": 177,
        "global_capacity_blocks": 2048,
        "camera_assignment": "equal",
        "camera_microbatch": 4,
        "optimizer_provenance": build_optimizer_provenance(args),
        "global_optimizer_step": 11,
        "optimizer_state_mode": "cold_start_per_rank",
        "optimizer_ema_saved": False,
        "optimizer_curvature_saved": False,
    }
    root.update(overrides)
    return root


class CheckpointOptimizerValidationTest(unittest.TestCase):
    def test_future_distributed_manifest_version_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported distributed checkpoint version: 4",
        ):
            validate_distributed_checkpoint_resume(
                _args(),
                _v3_root(checkpoint_version=4),
                world_size=2,
            )

    def test_matching_v3_configuration_is_accepted(self):
        args = _args()
        validate_distributed_checkpoint_resume(
            args,
            _v3_root(args),
            world_size=2,
            global_bsz=16,
        )

    def test_algorithm_aliases_are_normalized(self):
        args = _args()
        root = _v3_root(args)
        root["optimizer_provenance"]["paper_optimizer_algorithm"] = "sophia-tr"
        validate_distributed_checkpoint_resume(args, root, world_size=2)

    def test_algorithm_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "optimizer mismatch"):
            validate_distributed_checkpoint_resume(
                _args(paper_optimizer_algorithm="adam"),
                _v3_root(),
                world_size=2,
            )

    def test_legacy_distributed_checkpoint_only_allows_adam(self):
        root = {
            "checkpoint_version": 2,
            "world_size": 2,
            "global_bsz": 16,
            "global_capacity_blocks": 2048,
            "camera_assignment": "equal",
            "camera_microbatch": 4,
        }
        with self.assertRaisesRegex(ValueError, "only resume with Adam"):
            validate_distributed_checkpoint_resume(_args(), root, world_size=2)

        validate_distributed_checkpoint_resume(
            _args(paper_optimizer_algorithm="adam"),
            root,
            world_size=2,
        )

    def test_world_batch_capacity_and_camera_mismatches_are_rejected(self):
        cases = {
            "world_size": (_args(), _v3_root(), {"world_size": 4}),
            "global_bsz": (_args(bsz=8), _v3_root(), {"world_size": 2}),
            "paper_resident_capacity_blocks": (
                _args(paper_resident_capacity_blocks=1024),
                _v3_root(),
                {"world_size": 2},
            ),
            "tide_camera_assignment": (
                _args(tide_camera_assignment="gaussian_balanced"),
                _v3_root(),
                {"world_size": 2},
            ),
            "tide_camera_microbatch": (
                _args(tide_camera_microbatch=2),
                _v3_root(),
                {"world_size": 2},
            ),
        }
        for expected, (args, root, kwargs) in cases.items():
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    validate_distributed_checkpoint_resume(args, root, **kwargs)

    def test_each_tr_field_and_seed_mismatch_is_rejected(self):
        changes = {
            "paper_sophia_beta1": 0.8,
            "paper_sophia_beta2": 0.99,
            "paper_sophia_curvature_interval": 11,
            "paper_sophia_hutchinson_samples": 2,
            "paper_sophia_curvature_estimator": "other",
            "paper_sophia_gamma": 0.5,
            "paper_sophia_epsilon": 1e-12,
            "paper_sophia_curvature_seed": 18,
            "paper_tr_epsilon_init": 2e-6,
            "paper_tr_epsilon_final": 2e-8,
            "paper_tr_quat_norm": 1.0,
        }
        for name, value in changes.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    validate_distributed_checkpoint_resume(
                        _args(**{name: value}),
                        _v3_root(),
                        world_size=2,
                    )

    def test_missing_v3_field_is_rejected(self):
        root = _v3_root()
        del root["optimizer_provenance"]["paper_sophia_curvature_seed"]
        with self.assertRaisesRegex(ValueError, "paper_sophia_curvature_seed"):
            validate_distributed_checkpoint_resume(_args(), root, world_size=2)

    def test_optimizer_total_steps_preserves_schedule_semantics(self):
        root = _v3_root(_args(iterations=30_000))
        validate_distributed_checkpoint_resume(
            _args(iterations=29_999),
            root,
            world_size=2,
        )
        with self.assertRaisesRegex(ValueError, "optimizer_total_steps"):
            validate_distributed_checkpoint_resume(
                _args(iterations=30_001),
                root,
                world_size=2,
            )

    def test_v3_requires_cold_start_metadata_and_valid_step(self):
        with self.assertRaisesRegex(ValueError, "cold_start_per_rank"):
            validate_distributed_checkpoint_resume(
                _args(),
                _v3_root(optimizer_state_mode="cold_start"),
                world_size=2,
            )
        with self.assertRaisesRegex(ValueError, "global_optimizer_step"):
            validate_distributed_checkpoint_resume(
                _args(),
                _v3_root(global_optimizer_step=-1),
                world_size=2,
            )

    def test_legacy_single_rank_without_algorithm_rejects_3dgs2_tr(self):
        with self.assertRaisesRegex(ValueError, "predates 3DGS2-TR"):
            validate_pure_ssd_checkpoint_optimizer(_args(), {"args": {}})

    def test_legacy_single_rank_without_algorithm_accepts_adam(self):
        validate_pure_ssd_checkpoint_optimizer(
            _args(paper_optimizer_algorithm="adam"),
            {"args": {}},
        )

    def test_single_rank_resume_preserves_optimizer_step_schedule(self):
        saved = _args(iterations=30_000)
        manifest = {"args": dict(vars(saved))}
        validate_pure_ssd_checkpoint_optimizer(
            _args(iterations=29_999),
            manifest,
        )
        with self.assertRaisesRegex(ValueError, "optimizer_total_steps"):
            validate_pure_ssd_checkpoint_optimizer(
                _args(iterations=30_001),
                manifest,
            )


if __name__ == "__main__":
    unittest.main()
