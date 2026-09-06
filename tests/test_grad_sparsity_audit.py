import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from strategies.tide_engine.distributed_metrics import (
    GRAD_METRIC_VALUE_FIELDS,
    GRAD_ZERO_GLOBAL_FIELDS,
)
from strategies.tide_engine.gradient_schema import (
    GRADIENT_WIDTHS,
    GRAD_NEAR_ZERO_THRESHOLDS,
)


SCRIPT_PATH = Path(__file__).parents[1] / "tools" / "audit_grad_sparsity_metrics.py"
SPEC = importlib.util.spec_from_file_location("audit_grad_sparsity_metrics", SCRIPT_PATH)
AUDIT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT_MODULE)


def _write_tsv(path, fieldnames, row):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


def _all_zero_profile_row():
    row = {field: 0 for field in GRAD_ZERO_GLOBAL_FIELDS}
    row.update(
        {
            "iteration": 1,
            "world_size": 1,
            "optimizer_step": 1,
            "active_sh_degree": 0,
            "active_parameter_width": 14,
            "near_zero_threshold": 1e-8,
            "tile_contribution_mode": "profile",
            "tile_alpha_threshold": 1.0 / 255.0,
            "tile_drop_gradients_observed": 1,
            "projection_cull_unique_gaussians": 1,
            "projection_cull_parameter_elements": 59,
            "projection_cull_zero_gradient_elements": 59,
            "projection_cull_all_zero_gradient_gaussians": 1,
            "projection_cull_active_parameter_elements": 14,
            "projection_cull_active_zero_gradient_elements": 14,
            "projection_cull_active_all_zero_gradient_gaussians": 1,
            "projection_cull_zero_gradient_elements_59_gaussians": 1,
            "projection_cull_near_zero_elements_59_gaussians": 1,
            "projection_cull_active_zero_gradient_elements_14_gaussians": 1,
            "projection_cull_active_near_zero_elements_14_gaussians": 1,
            "tile_mask_rejected_gaussians": 1,
            "tile_mask_rejected_all_zero_gradient_gaussians": 1,
            "tile_mask_projection_pairs": 1,
            "tile_mask_candidate_tile_pairs_before": 1,
        }
    )
    for token, _ in GRAD_NEAR_ZERO_THRESHOLDS:
        row[f"projection_cull_abs_le_{token}_gradient_elements"] = 59
        row[f"projection_cull_active_abs_le_{token}_gradient_elements"] = 14
    for component, width in GRADIENT_WIDTHS.items():
        prefix = f"projection_cull_{component}"
        row[f"{prefix}_parameter_elements"] = width
        row[f"{prefix}_zero_gradient_elements"] = width
        for token, _ in GRAD_NEAR_ZERO_THRESHOLDS:
            row[f"{prefix}_abs_le_{token}_gradient_elements"] = width
    return row


class GradSparsityAuditTest(unittest.TestCase):
    def _audit(self, grad_row, optimizer_touched_rows):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_tsv(
                run_dir / "metrics_grad_zero_global.tsv",
                GRAD_ZERO_GLOBAL_FIELDS,
                grad_row,
            )
            _write_tsv(
                run_dir / "metrics_batch_global.tsv",
                ["iteration", "global_resident_blocks", "optimizer_touched_rows"],
                {
                    "iteration": 1,
                    "global_resident_blocks": 1,
                    "optimizer_touched_rows": optimizer_touched_rows,
                },
            )
            return AUDIT_MODULE.audit(run_dir)

    def test_profile_all_zero_would_drop_row_passes(self):
        result = self._audit(_all_zero_profile_row(), optimizer_touched_rows=0)
        self.assertEqual(
            result["checks"]["profile_would_drop_rows_have_zero_gradient"],
            "PASS",
        )
        self.assertEqual(
            result["checks"]["apply_equivalence_under_profiled_threshold"],
            "PASS",
        )

    def test_profile_nonzero_would_drop_row_fails(self):
        row = _all_zero_profile_row()
        row.update(
            {
                "projection_cull_all_zero_gradient_gaussians": 0,
                "tile_mask_rejected_all_zero_gradient_gaussians": 0,
                "tile_mask_rejected_nonzero_gradient_gaussians": 1,
                "would_drop_nonzero_gradient_gaussians": 1,
            }
        )
        with self.assertRaisesRegex(AssertionError, "false negatives"):
            self._audit(row, optimizer_touched_rows=1)

    def test_apply_cannot_claim_unobserved_would_drop_gradient(self):
        row = _all_zero_profile_row()
        row["tile_contribution_mode"] = "apply"
        row["tile_drop_gradients_observed"] = 0
        row["would_drop_nonzero_gradient_gaussians"] = 1
        with self.assertRaisesRegex(AssertionError, "observation mode"):
            self._audit(row, optimizer_touched_rows=0)

    def test_apply_without_profile_is_not_marked_equivalent(self):
        row = _all_zero_profile_row()
        row["tile_contribution_mode"] = "apply"
        row["tile_drop_gradients_observed"] = 0
        result = self._audit(row, optimizer_touched_rows=0)
        self.assertEqual(
            result["checks"]["profile_would_drop_rows_have_zero_gradient"],
            "NOT_CHECKED",
        )
        self.assertEqual(
            result["checks"]["apply_equivalence_under_profiled_threshold"],
            "NOT_ESTABLISHED",
        )


if __name__ == "__main__":
    unittest.main()
