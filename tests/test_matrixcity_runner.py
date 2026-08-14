import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "train_matrixcity_1b.sh"


class MatrixCityRunnerTest(unittest.TestCase):
    def _dry_run(self, *extra_args):
        with tempfile.TemporaryDirectory() as root:
            env = os.environ.copy()
            for name in (
                "OPTIMIZER_ALGORITHM",
                "SOPHIA_BETA1",
                "SOPHIA_BETA2",
                "SOPHIA_CURVATURE_INTERVAL",
                "SOPHIA_HUTCHINSON_SAMPLES",
                "SOPHIA_CURVATURE_SEED",
                "SOPHIA_GAMMA",
                "SOPHIA_EPSILON",
                "TR_EPSILON_INIT",
                "TR_EPSILON_FINAL",
            ):
                env.pop(name, None)
            command = [
                "bash",
                str(RUNNER),
                "--mode",
                "train",
                "--root",
                root,
                "--run-tag",
                "test-run",
                "--src",
                "/tmp/matrixcity",
                "--ply",
                "/tmp/matrixcity.ply",
                "--manifest",
                "/tmp/prebuilt.json",
                "--iterations",
                "176",
                "--bsz",
                "16",
                "--capacity",
                "64",
                "--dry-run",
            ]
            command.extend(extra_args)
            subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            run_root = Path(root) / "output" / "runs" / "test-run"
            return (
                (run_root / "commands.sh").read_text(encoding="utf-8"),
                (run_root / "planned_runs.tsv").read_text(encoding="utf-8"),
            )

    def test_default_dry_run_keeps_adam(self):
        commands, plan = self._dry_run()

        self.assertIn("--optimizer adam", commands)
        self.assertIn("_adam_iter176", plan)
        self.assertNotIn("3dgs2_tr", commands)
        self.assertNotIn("--paper_sophia_", commands)
        self.assertNotIn("--paper_tr_", commands)

    def test_explicit_3dgs2_tr_and_checkpoint_are_rendered(self):
        commands, plan = self._dry_run(
            "--optimizer",
            "3dgs2_tr",
            "--checkpoint-iter",
            "160",
        )

        self.assertIn("--optimizer 3dgs2_tr", commands)
        self.assertIn("--checkpoint_iterations 160", commands)
        self.assertIn("_3dgs2_tr_iter176", plan)
        self.assertIn("--paper_sophia_beta1 0.9", commands)
        self.assertIn("--paper_sophia_beta2 0.999", commands)
        self.assertIn("--paper_sophia_curvature_interval 10", commands)
        self.assertIn("--paper_sophia_hutchinson_samples 1", commands)
        self.assertIn("--paper_sophia_curvature_seed 1", commands)
        self.assertIn("--paper_sophia_gamma 1.0", commands)
        self.assertIn("--paper_tr_epsilon_init 1e-6", commands)
        self.assertIn("--paper_tr_epsilon_final 1e-8", commands)

    def test_explicit_3dgs2_tr_parameters_are_rendered(self):
        commands, _ = self._dry_run(
            "--optimizer",
            "3dgs2_tr",
            "--sophia-beta1",
            "0.8",
            "--sophia-beta2",
            "0.95",
            "--curvature-interval",
            "4",
            "--hutchinson-samples",
            "3",
            "--curvature-seed",
            "17",
            "--sophia-gamma",
            "0.5",
            "--sophia-epsilon",
            "1e-12",
            "--tr-epsilon-init",
            "2e-6",
            "--tr-epsilon-final",
            "2e-8",
        )

        expected = (
            "--paper_sophia_beta1 0.8",
            "--paper_sophia_beta2 0.95",
            "--paper_sophia_curvature_interval 4",
            "--paper_sophia_hutchinson_samples 3",
            "--paper_sophia_curvature_seed 17",
            "--paper_sophia_gamma 0.5",
            "--paper_sophia_epsilon 1e-12",
            "--paper_tr_epsilon_init 2e-6",
            "--paper_tr_epsilon_final 2e-8",
        )
        for option in expected:
            self.assertIn(option, commands)

    def test_four_gpu_command_uses_generic_torchrun(self):
        commands, plan = self._dry_run(
            "--optimizer",
            "3dgs2_tr",
            "--gpus",
            "0,1,2,3",
            "--camera-microbatch",
            "2",
        )

        self.assertIn("-m torch.distributed.run", commands)
        self.assertIn("--nproc_per_node=4", commands)
        self.assertIn("--tide_distributed_mode gaussian_sharded", commands)
        self.assertIn("--tide_camera_microbatch 2", commands)
        self.assertIn("_dist4_gaussian_balanced", plan)

    def test_resume_does_not_emit_prebuilt_manifest(self):
        commands, _ = self._dry_run(
            "--mode",
            "resume",
            "--start-checkpoint",
            "/tmp/checkpoint",
            "--resume-to-iter",
            "1000",
            "--checkpoint-iter",
            "1000",
        )

        self.assertIn("--start_checkpoint /tmp/checkpoint", commands)
        self.assertIn("--checkpoint_iterations 1000", commands)
        self.assertNotIn("--pure_ssd_prebuilt_manifest", commands)


if __name__ == "__main__":
    unittest.main()
