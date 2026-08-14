import tempfile
import unittest
from pathlib import Path

from tools.compare_pure_ssd_quality import (
    build_argparser,
    build_paired_rows,
    compare_quality,
    validate_camera_alignment,
)
from tools.pure_ssd_quality_utils import write_tsv


FIELDS = (
    "camera_index",
    "image_name",
    "psnr",
    "ssim",
    "lpips_alex",
    "render_ms",
)


def make_row(camera_index, image_name, offset=0.0):
    return {
        "camera_index": camera_index,
        "image_name": image_name,
        "psnr": 20.0 + offset,
        "ssim": 0.8 + offset,
        "lpips_alex": 0.2 + offset,
        "render_ms": 5.0 + offset,
    }


class CameraAlignmentTest(unittest.TestCase):
    def test_camera_mismatch_fails_clearly(self):
        adam = [make_row(0, "a"), make_row(1, "b")]
        candidate = [make_row(0, "a"), make_row(2, "c")]
        with self.assertRaisesRegex(ValueError, "camera sets or order do not match"):
            validate_camera_alignment(adam, candidate)

    def test_candidate_minus_adam_delta(self):
        adam = [make_row(0, "a")]
        candidate = [make_row(0, "a", offset=0.5)]
        paired = build_paired_rows(adam, candidate)
        self.assertEqual(len(paired), 1)
        self.assertAlmostEqual(paired[0]["delta_psnr"], 0.5)
        self.assertAlmostEqual(paired[0]["delta_lpips_alex"], 0.5)
        self.assertAlmostEqual(paired[0]["candidate_psnr"], 20.5)

    def test_cli_uses_generic_candidate_name(self):
        args = build_argparser().parse_args(
            ["--adam", "/adam", "--candidate", "/candidate", "--output-dir", "/out"]
        )
        self.assertEqual(args.candidate, "/candidate")

    def test_compare_writes_tsv_and_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adam_dir = root / "adam"
            candidate_dir = root / "candidate"
            output_dir = root / "comparison"
            write_tsv(
                adam_dir / "per_camera_metrics.tsv",
                [make_row(0, "a"), make_row(1, "b")],
                FIELDS,
            )
            write_tsv(
                candidate_dir / "per_camera_metrics.tsv",
                [make_row(0, "a", 0.1), make_row(1, "b", -0.1)],
                FIELDS,
            )
            summary = compare_quality(adam_dir, candidate_dir, output_dir)
            self.assertEqual(summary["camera_count"], 2)
            self.assertEqual(summary["delta_definition"], "Candidate - Adam")
            self.assertTrue((output_dir / "paired_quality_comparison.tsv").is_file())
            self.assertTrue((output_dir / "paired_quality_comparison.json").is_file())


if __name__ == "__main__":
    unittest.main()
