#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/nvme0n1/Tsinghua_Node11/ylunhan
REPO="$BASE/TideGS-git"
RUN_BASE="$BASE/TideGS-runs"
PY="$BASE/miniconda3/envs/tidegs-dist/bin/python"
RUNNER="$REPO/scripts/run_iat_grad_sparsity.sh"
AUDITOR="$REPO/tools/audit_grad_sparsity_metrics.py"
ANALYZER="$REPO/tools/analyze_scene_locality.py"

OFFSETS=(0 6464 12928 19392 25792 32256 38720 45184)
STAMP="${TIDE_MATRIX_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MANIFEST="$RUN_BASE/output/scene_locality_${STAMP}_manifest.tsv"
ANALYSIS_DIR="$RUN_BASE/output/scene_locality_${STAMP}_analysis"

gpu_set_is_free() {
  local gpu_set="$1"
  local gpu_id gpu_pids
  IFS=',' read -r -a gpu_ids <<< "$gpu_set"
  (( ${#gpu_ids[@]} == 4 )) || return 1
  for gpu_id in "${gpu_ids[@]}"; do
    gpu_pids="$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader,nounits)"
    [[ -z "${gpu_pids//[[:space:]]/}" ]] || return 1
  done
}

if [[ -n "${TIDE_GPUS:-}" ]]; then
  GPUS="$TIDE_GPUS"
elif gpu_set_is_free "2,3,4,5"; then
  GPUS="2,3,4,5"
elif gpu_set_is_free "4,5,6,7"; then
  GPUS="4,5,6,7"
else
  echo "No free four-GPU set: tried 2,3,4,5 and 4,5,6,7" >&2
  exit 3
fi

mkdir -p "$RUN_BASE/output" "$ANALYSIS_DIR"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  run_tag offset sh_mode model_path schedule_metadata audit_json > "$MANIFEST"

"$PY" - "${OFFSETS[@]}" <<'PY'
import sys

offsets = [int(value) for value in sys.argv[1:]]
window_size = 1024
windows = [set(range(offset, offset + window_size)) for offset in offsets]
for i, left in enumerate(windows):
    for j, right in enumerate(windows[:i]):
        overlap = left & right
        if overlap:
            raise SystemExit(
                f"TSP windows overlap: offsets {offsets[i]} and {offsets[j]}"
            )
print(f"validated {len(offsets)} disjoint TSP windows")
PY

validate_run() {
  local run_tag="$1"
  local model_name="$2"
  local expected_batches="$3"
  local expected_samples="$4"
  local expected_offset="$5"
  local model_path="$RUN_BASE/output/runs/$run_tag/$model_name"
  local audit_json="$RUN_BASE/output/runs/$run_tag/audit_grad_sparsity.json"

  "$PY" "$AUDITOR" "$model_path" > "$audit_json"
  "$PY" - "$model_path" "$expected_batches" "$expected_samples" "$expected_offset" <<'PY'
import csv
import glob
import json
import os
import sys

model_path, expected_batches, expected_samples, expected_offset = sys.argv[1:]
expected_batches = int(expected_batches)
expected_samples = int(expected_samples)
expected_offset = int(expected_offset)

def read_tsv(name):
    with open(os.path.join(model_path, name), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))

grad_rows = read_tsv("metrics_grad_zero_global.tsv")
batch_rows = read_tsv("metrics_batch_global.tsv")
camera_rows = read_tsv("metrics_camera_batches.tsv")
if not (len(grad_rows) == len(batch_rows) == len(camera_rows) == expected_batches):
    raise SystemExit(
        "row count mismatch: "
        f"grad={len(grad_rows)} batch={len(batch_rows)} "
        f"camera={len(camera_rows)} expected={expected_batches}"
    )

rank_files = glob.glob(os.path.join(model_path, "metrics_grad_zero_rank*.tsv"))
rank_files += glob.glob(
    os.path.join(model_path, "rank_*", "metrics_grad_zero_rank*.tsv")
)
if len(rank_files) != 4:
    raise SystemExit(f"expected 4 rank metric files, found {len(rank_files)}")

sample_files = glob.glob(
    os.path.join(model_path, "**", "grad_sample_*.pt"), recursive=True
)
if len(sample_files) != expected_samples:
    raise SystemExit(
        f"expected {expected_samples} gradient samples, found {len(sample_files)}"
    )

camera_ids = []
for row in camera_rows:
    if int(row["trajectory_start_offset"]) != expected_offset:
        raise SystemExit(f"unexpected trajectory offset in camera trace: {row}")
    batch_ids = json.loads(row["camera_ids_json"])
    if len(batch_ids) != int(row["camera_count"]):
        raise SystemExit("camera_count does not match camera_ids_json")
    camera_ids.extend(int(value) for value in batch_ids)
if len(camera_ids) != expected_batches * 64:
    raise SystemExit(f"unexpected camera total: {len(camera_ids)}")
if len(set(camera_ids)) != len(camera_ids):
    raise SystemExit("camera IDs repeat within the analysis window")

with open(os.path.join(model_path, "schedule_metadata.json"), encoding="utf-8") as handle:
    metadata = json.load(handle)
if int(metadata["effective_trajectory_start_offset"]) != expected_offset:
    raise SystemExit("schedule metadata offset mismatch")
if metadata["analysis_window_camera_ids"] != camera_ids:
    raise SystemExit("schedule metadata camera IDs differ from batch trace")
print(
    f"validated run: batches={expected_batches} ranks=4 "
    f"samples={expected_samples} cameras={len(camera_ids)}"
)
PY
}

cleanup_cache() {
  local run_tag="$1"
  local model_name="$2"
  local cache_root cache_path
  cache_root="$(realpath -m "$RUN_BASE/ssd_cache")"
  cache_path="$(realpath -m "$RUN_BASE/ssd_cache/$run_tag/$model_name")"
  case "$cache_path" in
    "$cache_root"/*) ;;
    *)
      echo "Refusing to clean cache outside $cache_root: $cache_path" >&2
      exit 4
      ;;
  esac
  if [[ -d "$cache_path" ]]; then
    rm -rf -- "$cache_path"
  fi
  rmdir "$RUN_BASE/ssd_cache/$run_tag" 2>/dev/null || true
}

SMOKE_TAG="scene_locality_smoke_offset6464_${STAMP}"
SMOKE_MODEL="scene_locality_smoke_bsz64_cap8192_offset6464_iter64"
TIDE_GPUS="$GPUS" \
TIDE_ITERATIONS=64 \
TIDE_TRAJECTORY_START_OFFSET=6464 \
TIDE_FORCE_ACTIVE_SH_DEGREE=-1 \
TIDE_GRAD_SAMPLE_ROWS=128 \
TIDE_RUN_TAG="$SMOKE_TAG" \
TIDE_MODEL_NAME="$SMOKE_MODEL" \
  "$RUNNER"
validate_run "$SMOKE_TAG" "$SMOKE_MODEL" 1 4 6464
cleanup_cache "$SMOKE_TAG" "$SMOKE_MODEL"

for OFFSET in "${OFFSETS[@]}"; do
  for SH_MODE in default forced_sh3; do
    if [[ "$SH_MODE" == "default" ]]; then
      FORCE_SH=-1
    else
      FORCE_SH=3
    fi
    RUN_TAG="scene_locality_iat_offset${OFFSET}_${SH_MODE}_${STAMP}"
    MODEL_NAME="scene_locality_bsz64_cap8192_offset${OFFSET}_${SH_MODE}_iter1000"
    TIDE_GPUS="$GPUS" \
    TIDE_ITERATIONS=1000 \
    TIDE_BSZ=64 \
    TIDE_CAPACITY_BLOCKS=8192 \
    TIDE_TRAJECTORY_START_OFFSET="$OFFSET" \
    TIDE_FORCE_ACTIVE_SH_DEGREE="$FORCE_SH" \
    TIDE_GRAD_SAMPLE_ROWS=8192 \
    TIDE_RUN_TAG="$RUN_TAG" \
    TIDE_MODEL_NAME="$MODEL_NAME" \
      "$RUNNER"
    validate_run "$RUN_TAG" "$MODEL_NAME" 16 64 "$OFFSET"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$RUN_TAG" \
      "$OFFSET" \
      "$SH_MODE" \
      "$RUN_BASE/output/runs/$RUN_TAG/$MODEL_NAME" \
      "$RUN_BASE/output/runs/$RUN_TAG/$MODEL_NAME/schedule_metadata.json" \
      "$RUN_BASE/output/runs/$RUN_TAG/audit_grad_sparsity.json" >> "$MANIFEST"
    cleanup_cache "$RUN_TAG" "$MODEL_NAME"
  done
done

"$PY" "$ANALYZER" \
  --manifest "$MANIFEST" \
  --output-dir "$ANALYSIS_DIR" \
  --historical-default \
    "$RUN_BASE/output/runs/grad_near_zero_adam_iat_4gpu_bsz64_cap8192_20260821_055755/train_bsz64_cap8192_lam0p3_decay0p95_seed0p25_adam_iter1000_dist4_equal" \
    "$RUN_BASE/output/runs/grad_near_zero_adam_iat_4gpu_bsz64_cap8192_20260821_165826/train_bsz64_cap8192_lam0p3_decay0p95_seed0p25_adam_iter1000_dist4_equal" \
  --historical-forced-sh3 \
    "$RUN_BASE/output/runs/grad_forcedsh3_iat_4gpu_bsz64_cap8192_20260822_033232/train_bsz64_cap8192_lam0p3_decay0p95_seed0p25_adam_iter1000_dist4_equal_forcedsh3"

echo "Scene-locality matrix completed"
echo "Manifest: $MANIFEST"
echo "Analysis: $ANALYSIS_DIR"
