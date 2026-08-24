#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/nvme0n1/Tsinghua_Node11/ylunhan
REPO="$BASE/TideGS-git"
RUN_BASE="$BASE/TideGS-runs"
PY="$BASE/miniconda3/envs/tidegs-dist/bin/python"
SCENE=/essfs10/public_data/zhongch_data/MatrixCity_big_city/big_city/aerial/pose/all_blocks
DENSE_PLY=/essfs10/public_data/zhongch_data/MatrixCity_big_city/big_city/aerial/all_1B_ds4_ratio6_1115251200_active.ply
DECODED="$BASE/tidegs_4gpu_smoke/decoded_images_full_hier"
DEFAULT_MANIFEST=/essfs10/public_data/zhongch_data/tmp_output/Flex-3DGS/ssd_cache/20260516_101136_1b_pure_ssd_r_8iter/20260516_101145/streaming_init_manifest.json
SCHEDULE_CACHE="$RUN_BASE/schedule_cache/oneb_fullcams_trajectory"

ITERATIONS="${TIDE_ITERATIONS:-1000}"
BSZ="${TIDE_BSZ:-64}"
CAPACITY_BLOCKS="${TIDE_CAPACITY_BLOCKS:-8192}"
FORCE_ACTIVE_SH_DEGREE="${TIDE_FORCE_ACTIVE_SH_DEGREE:--1}"
TRAJECTORY_START_OFFSET="${TIDE_TRAJECTORY_START_OFFSET:-0}"
GPUS="${TIDE_GPUS:-2,3,4,5}"
MANIFEST="${TIDE_MANIFEST:-$DEFAULT_MANIFEST}"
GRAD_SAMPLE_ROWS="${TIDE_GRAD_SAMPLE_ROWS:-8192}"
RUN_TAG="${TIDE_RUN_TAG:-grad_near_zero_adam_iat_4gpu_bsz${BSZ}_cap${CAPACITY_BLOCKS}_offset${TRAJECTORY_START_OFFSET}_$(date +%Y%m%d_%H%M%S)}"
MODEL_SUFFIX="_offset${TRAJECTORY_START_OFFSET}"
if (( FORCE_ACTIVE_SH_DEGREE >= 0 )); then
  MODEL_SUFFIX="${MODEL_SUFFIX}_forcedsh${FORCE_ACTIVE_SH_DEGREE}"
fi
MODEL_NAME="${TIDE_MODEL_NAME:-train_bsz${BSZ}_cap${CAPACITY_BLOCKS}_lam0p3_decay0p95_seed0p25_adam_iter${ITERATIONS}_dist4_equal${MODEL_SUFFIX}}"
RUN_ROOT="$RUN_BASE/output/runs/$RUN_TAG"
MODEL_PATH="$RUN_ROOT/$MODEL_NAME"
SSD_CACHE="$RUN_BASE/ssd_cache/$RUN_TAG/$MODEL_NAME"

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
if (( ${#GPU_IDS[@]} != 4 )); then
  echo "TIDE_GPUS must contain exactly four comma-separated GPU indices, got: $GPUS" >&2
  exit 2
fi
declare -A SEEN_GPU_IDS=()
for GPU_ID in "${GPU_IDS[@]}"; do
  if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU index in TIDE_GPUS: $GPU_ID" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPU_IDS[$GPU_ID]:-}" ]]; then
    echo "Duplicate GPU index in TIDE_GPUS: $GPU_ID" >&2
    exit 2
  fi
  SEEN_GPU_IDS[$GPU_ID]=1
  GPU_PIDS="$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -n "${GPU_PIDS//[[:space:]]/}" ]]; then
    echo "GPU $GPU_ID is not free; compute PIDs: $GPU_PIDS" >&2
    exit 3
  fi
done

mkdir -p "$RUN_ROOT" "$MODEL_PATH" "$SSD_CACHE"
cp "$0" "$RUN_ROOT/commands.sh"

cd "$REPO"
git rev-parse HEAD > "$RUN_ROOT/code_head.txt"
git status --short > "$RUN_ROOT/code_status.txt"
git diff -- train_tidegs.py scene/dataset_readers.py \
  arguments/__init__.py storage/schedule_utils.py \
  strategies/tide_engine/distributed_engine.py \
  strategies/tide_engine/distributed_metrics.py \
  strategies/tide_engine/gsplat_backend.py \
  tools/patch_gsplat_distributed_packed.py \
  tools/audit_grad_sparsity_metrics.py \
  tools/analyze_scene_locality.py \
  scripts/run_iat_grad_sparsity.sh \
  scripts/run_iat_scene_locality_matrix.sh > "$RUN_ROOT/code_changes.patch"
for UNTRACKED_FILE in \
  tools/audit_grad_sparsity_metrics.py \
  tools/analyze_scene_locality.py \
  scripts/run_iat_grad_sparsity.sh \
  scripts/run_iat_scene_locality_matrix.sh; do
  if ! git ls-files --error-unmatch "$UNTRACKED_FILE" >/dev/null 2>&1; then
    git diff --no-index /dev/null "$UNTRACKED_FILE" \
      >> "$RUN_ROOT/code_changes.patch" || true
  fi
done

printf '%s\n' \
  "RUN_TAG=$RUN_TAG" \
  "RUN_ROOT=$RUN_ROOT" \
  "MODEL_PATH=$MODEL_PATH" \
  "GPUS=$GPUS" \
  "OPTIMIZER=adam" \
  "ITERATIONS=$ITERATIONS" \
  "BSZ=$BSZ" \
  "CAPACITY_BLOCKS=$CAPACITY_BLOCKS" \
  "FORCE_ACTIVE_SH_DEGREE=$FORCE_ACTIVE_SH_DEGREE" \
  "TRAJECTORY_START_OFFSET=$TRAJECTORY_START_OFFSET" \
  "ORDERING=trajectory" \
  "MANIFEST=$MANIFEST" \
  "GRAD_METRICS_INTERVAL=1" \
  "GRAD_NEAR_ZERO_THRESHOLD=1e-8" \
  "GRAD_SAMPLE_ROWS_PER_RANK_BATCH=$GRAD_SAMPLE_ROWS" | tee "$RUN_ROOT/settings.txt"

PYTHONDONTWRITEBYTECODE=1 \
PYTHONWARNINGS='ignore:TORCH_CUDA_ARCH_LIST is not set:UserWarning' \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES="$GPUS" \
"$PY" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  "$REPO/train_tidegs.py" \
  -s "$SCENE" \
  --model_path "$MODEL_PATH" \
  --iterations "$ITERATIONS" \
  --dense_ply_file "$DENSE_PLY" \
  --decode_dataset_path "$DECODED" \
  --bsz "$BSZ" \
  --debug_max_train_cameras -1 \
  --debug_camera_sample_mode linspace \
  --debug_camera_sample_start 0 \
  --disable_auto_densification \
  --sparse_adam \
  --enable_timer \
  --check_gpu_memory \
  --check_cpu_memory \
  --initial_point_cloud_downsampled_ratio 1.0 \
  --use_ssd_offload \
  --pure_ssd_offload \
  --pure_ssd_init_backend streaming \
  --pure_ssd_prebuilt_manifest "$MANIFEST" \
  --use_6plane \
  --ssd_cache_dir "$SSD_CACHE" \
  --gaussian_block_size 4096 \
  --max_ram_gb 32 \
  --num_clusters 64 \
  --ssd_schedule_ordering trajectory \
  --tide_trajectory_start_offset "$TRAJECTORY_START_OFFSET" \
  --tide_optimizer_backend gpu_resident \
  --optimizer adam \
  --tide_block_reader_backend tiered_cache \
  --tide_optimizer_deferred_mode off \
  --tide_resident_selection_policy topc_balanced \
  --tide_resident_lambda 0.3 \
  --tide_resident_recency_decay 0.95 \
  --tide_balanced_seed_fraction 0.25 \
  --tide_resident_capacity_blocks "$CAPACITY_BLOCKS" \
  --tide_optimizer_state_mode resident_blocks \
  --tide_distributed_mode gaussian_sharded \
  --tide_camera_assignment equal \
  --tide_camera_microbatch 4 \
  --tide_owner_balance_samples 256 \
  --projection_max_cameras_per_chunk 2 \
  --pure_ssd_checkpoint_mode incremental \
  --pure_ssd_checkpoint_patch_mode hardlink \
  --pure_ssd_checkpoint_keep_last 2 \
  --tide_storage_max_patch_files 32 \
  --tide_storage_max_patch_gb 64 \
  --tide_storage_min_free_gb 64 \
  --tide_storage_compaction_interval_iterations 5000 \
  --tide_storage_compaction_target_patch_files 8 \
  --tide_storage_compaction_rank_concurrency 2 \
  --tide_storage_compaction_emergency_free_gb -1 \
  --tide_storage_idle_compaction_seconds 0 \
  --tide_debug_logging \
  --tide_detailed_metrics \
  --tide_grad_zero_metrics \
  --tide_grad_zero_metrics_interval 1 \
  --tide_grad_near_zero_threshold 1e-8 \
  --tide_grad_sample_rows "$GRAD_SAMPLE_ROWS" \
  --tide_grad_stats_chunk_rows 262144 \
  --tide_force_active_sh_degree "$FORCE_ACTIVE_SH_DEGREE" \
  --tide_free_unified_params \
  --pure_ssd_schedule_cache_dir "$SCHEDULE_CACHE" \
  2>&1 | tee "$RUN_ROOT/terminal.log"

"$PY" - "$MODEL_PATH/schedule_metadata.json" "$RUN_ROOT/settings.txt" <<'PY'
import json
import sys

metadata_path, settings_path = sys.argv[1:]
with open(metadata_path, "r", encoding="utf-8") as handle:
    metadata = json.load(handle)
fields = (
    "effective_trajectory_start_offset",
    "canonical_schedule_sha256",
    "rotated_schedule_sha256",
    "canonical_first_camera_id",
    "canonical_last_camera_id",
    "rotated_first_camera_id",
    "rotated_last_camera_id",
    "analysis_window_camera_count",
)
with open(settings_path, "a", encoding="utf-8") as handle:
    for field in fields:
        handle.write(f"{field.upper()}={metadata[field]}\n")
PY
