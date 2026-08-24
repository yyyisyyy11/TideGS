#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/nvme0n1/Tsinghua_Node11/ylunhan
REPO="$BASE/TideGS-git"
RUN_BASE="$BASE/TideGS-runs"
PY="$BASE/miniconda3/envs/tidegs-dist/bin/python"
RUNNER="$REPO/scripts/run_iat_grad_sparsity.sh"
AUDITOR="$REPO/tools/audit_grad_sparsity_metrics.py"

OFFSETS=(
  860 2581 4302 6023 7744 9465 11186 12908 14629 16350
  18071 19792 21513 23234 24955 26676 28397 30118 31839 33560
  35281 37002 38724 40445 42166 43887 45608 47329 49050 50771
)
EXISTING_OFFSETS=(0 6464 12928 19392 25792 32256 38720 45184)
DATASET_CAMERAS=51632
ITERATIONS=5000
BSZ=64
EXPECTED_BATCHES=$(( (ITERATIONS + BSZ - 1) / BSZ ))
EXPECTED_SAMPLES=$(( EXPECTED_BATCHES * 4 ))
EXPECTED_CAMERA_EVENTS=$(( EXPECTED_BATCHES * BSZ ))
STAMP="${TIDE_MATRIX_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MANIFEST="$RUN_BASE/output/scene_locality_30offset_5000_${STAMP}_manifest.tsv"
STATUS="$RUN_BASE/output/scene_locality_30offset_5000_${STAMP}_status.tsv"
LAUNCHER_COPY="$RUN_BASE/output/scene_locality_30offset_5000_${STAMP}_commands.sh"

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

mkdir -p "$RUN_BASE/output"
cp "$0" "$LAUNCHER_COPY"
if [[ ! -f "$MANIFEST" ]]; then
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    run_tag offset sh_mode model_path schedule_metadata audit_json > "$MANIFEST"
fi
if [[ ! -f "$STATUS" ]]; then
  printf '%s\t%s\t%s\t%s\n' timestamp offset state detail > "$STATUS"
fi

"$PY" - "$DATASET_CAMERAS" "${EXISTING_OFFSETS[@]}" -- "${OFFSETS[@]}" <<'PY'
import sys

num_cameras = int(sys.argv[1])
separator = sys.argv.index("--")
existing = {int(value) for value in sys.argv[2:separator]}
offsets = [int(value) for value in sys.argv[separator + 1:]]
if len(offsets) != 30 or len(set(offsets)) != 30:
    raise SystemExit("expected 30 unique offsets")
if any(value < 0 or value >= num_cameras for value in offsets):
    raise SystemExit("offset outside canonical schedule")
if existing.intersection(offsets):
    raise SystemExit(f"new offsets overlap existing starts: {existing.intersection(offsets)}")
cyclic_gaps = [
    (offsets[(index + 1) % len(offsets)] - offsets[index]) % num_cameras
    for index in range(len(offsets))
]
if min(cyclic_gaps) < 1721 or max(cyclic_gaps) > 1722:
    raise SystemExit(f"offsets are not evenly spaced: {cyclic_gaps}")
print(
    f"validated offsets=30 min_gap={min(cyclic_gaps)} "
    f"max_gap={max(cyclic_gaps)} existing_overlap=0"
)
PY

validate_run() {
  local run_tag="$1"
  local model_name="$2"
  local expected_offset="$3"
  local model_path="$RUN_BASE/output/runs/$run_tag/$model_name"
  local audit_json="$RUN_BASE/output/runs/$run_tag/audit_grad_sparsity.json"

  "$PY" "$AUDITOR" "$model_path" > "$audit_json"
  "$PY" - \
    "$model_path" \
    "$EXPECTED_BATCHES" \
    "$EXPECTED_SAMPLES" \
    "$EXPECTED_CAMERA_EVENTS" \
    "$expected_offset" <<'PY'
import csv
import glob
import json
import os
import sys

model_path, expected_batches, expected_samples, expected_cameras, expected_offset = sys.argv[1:]
expected_batches = int(expected_batches)
expected_samples = int(expected_samples)
expected_cameras = int(expected_cameras)
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
if len(camera_ids) != expected_cameras:
    raise SystemExit(f"unexpected camera event total: {len(camera_ids)}")
if len(set(camera_ids)) != len(camera_ids):
    raise SystemExit("camera IDs repeat within one 5000-iteration window")

with open(os.path.join(model_path, "schedule_metadata.json"), encoding="utf-8") as handle:
    metadata = json.load(handle)
if int(metadata["effective_trajectory_start_offset"]) != expected_offset:
    raise SystemExit("schedule metadata offset mismatch")
if metadata["analysis_window_camera_ids"] != camera_ids:
    raise SystemExit("schedule metadata camera IDs differ from batch trace")
print(
    f"validated run: batches={expected_batches} ranks=4 "
    f"samples={expected_samples} camera_events={len(camera_ids)}"
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

record_manifest_once() {
  local run_tag="$1"
  local offset="$2"
  local model_path="$3"
  local audit_json="$4"
  if awk -F '\t' -v tag="$run_tag" 'NR > 1 && $1 == tag {found=1} END {exit !found}' "$MANIFEST"; then
    return
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_tag" \
    "$offset" \
    default \
    "$model_path" \
    "$model_path/schedule_metadata.json" \
    "$audit_json" >> "$MANIFEST"
}

for OFFSET in "${OFFSETS[@]}"; do
  RUN_TAG="scene_locality30_iat_offset${OFFSET}_default_${STAMP}"
  MODEL_NAME="scene_locality_bsz64_cap8192_offset${OFFSET}_default_iter5000"
  RUN_ROOT="$RUN_BASE/output/runs/$RUN_TAG"
  MODEL_PATH="$RUN_ROOT/$MODEL_NAME"
  AUDIT_JSON="$RUN_ROOT/audit_grad_sparsity.json"

  if [[ -f "$AUDIT_JSON" ]]; then
    validate_run "$RUN_TAG" "$MODEL_NAME" "$OFFSET"
    record_manifest_once "$RUN_TAG" "$OFFSET" "$MODEL_PATH" "$AUDIT_JSON"
    cleanup_cache "$RUN_TAG" "$MODEL_NAME"
    printf '%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$OFFSET" skipped already_valid >> "$STATUS"
    continue
  fi

  printf '%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$OFFSET" started "$RUN_TAG" >> "$STATUS"
  TIDE_GPUS="$GPUS" \
  TIDE_ITERATIONS="$ITERATIONS" \
  TIDE_BSZ="$BSZ" \
  TIDE_CAPACITY_BLOCKS=8192 \
  TIDE_TRAJECTORY_START_OFFSET="$OFFSET" \
  TIDE_FORCE_ACTIVE_SH_DEGREE=-1 \
  TIDE_GRAD_SAMPLE_ROWS=8192 \
  TIDE_RUN_TAG="$RUN_TAG" \
  TIDE_MODEL_NAME="$MODEL_NAME" \
    "$RUNNER"
  validate_run "$RUN_TAG" "$MODEL_NAME" "$OFFSET"
  record_manifest_once "$RUN_TAG" "$OFFSET" "$MODEL_PATH" "$AUDIT_JSON"
  cleanup_cache "$RUN_TAG" "$MODEL_NAME"
  printf '%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$OFFSET" completed "$RUN_TAG" >> "$STATUS"
done

echo "Scene-locality 30-offset matrix completed"
echo "Stamp: $STAMP"
echo "GPUs: $GPUS"
echo "Manifest: $MANIFEST"
echo "Status: $STATUS"
