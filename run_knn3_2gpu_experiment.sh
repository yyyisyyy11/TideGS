#!/usr/bin/env bash
#SBATCH --job-name=tidegs-knn3-2gpu
#SBATCH --account=sponge
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=800G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/sponge/czhongak/tidegs/runs_lunhan/slurm-%j.out
# Submit:  module load slurm && sbatch /home/czhongak/code/proj_3dgs_training/lunhan/TideGS/run_knn3_2gpu_experiment.sh
# Resume:  RESUME_FROM=<run>/train_knn3_4gpu/checkpoints/<iter> sbatch <same path>
# Check:   bash <same path> --check-only    (inside an allocation with 2 GPUs)
# ============================================================================
# knn3 SSD base + multi-GPU "pure SSD" training + upstream 200-view evaluation.
#
# The number of GPUs is controlled by NGPU and defaults to 2, matching the filename.
# These settings come from a validated 4-GPU baseline; changing NGPU only partitions the same computation across ranks.
#
# Usage:
#   1) Adjust paths in the CONFIG section if needed; defaults are derived from DATA_ROOT.
#   2) Run directly on a node with allocated GPUs:
#          bash scripts/run_knn3_2gpu_experiment.sh
#      The script does not submit itself; run it in an allocated scheduler environment
#      (for Slurm, use srun --pty bash or submit this script with sbatch).
#   3) To check the environment without training: bash scripts/run_knn3_2gpu_experiment.sh --check-only
#
# The script has three stages: preflight -> training -> evaluation. Preflight failures explain what is missing and how to fix it.
# ============================================================================
set -euo pipefail

say()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 2; }

# ============================== CONFIG ======================================
# Repository root (defaults to the parent of this script's directory).
REPO="${REPO:-/home/czhongak/code/proj_3dgs_training/lunhan/TideGS}"

# Python interpreter: must provide torch 2.4.x, CUDA 12.x, gsplat 1.5.x, and simple-knn (distCUDA2).
PYTHON="${PYTHON:-/project/sponge/czhongak/envs/tidegs/bin/python}"
# The distributed path needs gsplat>=1.5.3 with the TideGS patches (tools/patch_gsplat_distributed_packed.py).
# The shared env keeps gsplat 1.0.0 for the main TideGS repo, so the patched 1.5.3 lives in its own
# directory and is put first on PYTHONPATH for every Python call in this script.
PYDEPS="${PYDEPS:-/scratch/sponge/czhongak/envs/lunhan_tidegs_pydeps}"
export PYTHONPATH="$PYDEPS${PYTHONPATH:+:$PYTHONPATH}"

# Scene directory (contains transforms_train.json and transforms_test.json).
SCENE_DIR="${SCENE_DIR:-/project/sponge/czhongak/data/big_city/aerial/pose/all_blocks}"
# Decoded-image cache (60698 raw files, already built; training rebuilds it only if dataset_raw is missing).
DECODE_DIR="${DECODE_DIR:-/scratch/sponge/czhongak/tidegs/decoded_imgs}"

# SSD base directory (contains base_file.bin, block_bounds.npy, and streaming_init_manifest.json).
SSD_BASE_DIR="${SSD_BASE_DIR:-/scratch/sponge/czhongak/tidegs/ssd_base_1b_knn3}"

# Output tag and artifact directories for this run.
OUT_BASE="${OUT_BASE:-/scratch/sponge/czhongak/tidegs/runs_lunhan}"
TAG="${TAG:-knn3_${NGPU:-2}gpu_$(date +%Y%m%d_%H%M%S)${SLURM_JOB_ID:+_job$SLURM_JOB_ID}}"
RUN_ROOT="${RUN_ROOT:-$OUT_BASE/$TAG}"
MODEL="$RUN_ROOT/train_knn3_4gpu"
CACHE="$RUN_ROOT/ssd_cache"
SCHED_CACHE="${SCHED_CACHE:-$OUT_BASE/schedule_cache/oneb_bigcity_tsp}"

# Resident selection policy. Active-first keeps every block the batch needs whenever it fits the cap.
RESIDENT_POLICY="${RESIDENT_POLICY:-topc_balanced_active_first}"

# Optional resume from a distributed checkpoint of an earlier run of this script (same NGPU and BSZ).
RESUME_FROM="${RESUME_FROM:-}"
# Checkpoints to keep (-1 = all), so every CHECKPOINTS entry can still be evaluated later.
KEEP_LAST="${KEEP_LAST:--1}"
# The ported 200-view evaluator only reads single-process checkpoints; see stage 6.
RUN_EVAL="${RUN_EVAL:-0}"

# Training scale. NGPU must divide 51632 (2, 4, and 8 work; 3 requires camera trimming).
NGPU="${NGPU:-2}"
BSZ="${BSZ:-64}"                      # **Global** batch; keep unchanged for comparison with the 4-GPU baseline.
ITERS="${ITERS:-200000}"              # Camera iterations (not optimizer steps).
CHECKPOINTS="${CHECKPOINTS:-50000 100000 150000 200000}"

# Resident block capacity is specified **per GPU**; the global value is per-GPU capacity × NGPU.
# 6144 per GPU with 2 GPUs gives a global capacity of 12288.
PER_CARD_CAP="${PER_CARD_CAP:-6144}"

# Validate numeric parameters before arithmetic. With set -u, non-numeric values would otherwise
# be interpreted as variable names and produce an "unbound variable" error.
for _v in NGPU PER_CARD_CAP BSZ ITERS; do
  _val="${!_v}"
  if [[ ! "$_val" =~ ^[0-9]+$ ]] || (( _val <= 0 )); then
    fail "$_v must be a positive integer, got '$_val' (do not combine multiple assignments in one environment variable)"
  fi
done
unset _v _val

if [[ -z "${RESIDENT_CAP:-}" ]]; then
  RESIDENT_CAP=$(( PER_CARD_CAP * NGPU ))
fi
if [[ ! "$RESIDENT_CAP" =~ ^[0-9]+$ ]] || (( RESIDENT_CAP <= 0 )); then
  fail "RESIDENT_CAP must be a positive integer, got '$RESIDENT_CAP'"
fi

# Total training cameras (fixed at 51632 for this dataset).
N_TRAIN_CAM_TOTAL=51632
N_TEST_CAM_TOTAL=9066

# Expected SSD base size and variant (used by preflight).
BASE_EXPECT_BYTES=263199283200
BASE_EXPECT_SCALE_MODE=knn3
# ============================================================================

CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then CHECK_ONLY=1; fi

# Print the parsed configuration before preflight so it is visible even when preflight fails.
say "Config: NGPU=$NGPU  BSZ=$BSZ(global)  ITERS=$ITERS  resident capacity=$PER_CARD_CAP/GPU = $RESIDENT_CAP(global)  policy=$RESIDENT_POLICY"
say "  base=$SSD_BASE_DIR"
say "  scene=$SCENE_DIR"
say "  output=$RUN_ROOT"

# ---------------------------------------------------------------------------
# 0. GPU visibility.
# ---------------------------------------------------------------------------
say "Preflight 0/N: interpreter and GPU"
[[ -x "$(command -v "$PYTHON")" ]] || fail "Interpreter not found: $PYTHON"
"$PYTHON" - <<PYEOF || fail "torch/CUDA is unavailable; check the driver and CUDA_VISIBLE_DEVICES"
import torch, sys
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
n = torch.cuda.device_count()
print("Visible GPUs:", n, [torch.cuda.get_device_name(i) for i in range(n)])
assert n >= $NGPU, f"Only {n} GPUs are visible, but NGPU=$NGPU"
PYEOF
"$PYTHON" - "$REPO" <<'GSPLAT_EOF' || fail "gsplat check failed; expected the patched gsplat 1.5.3 in PYDEPS=$PYDEPS.
       Install: $PYTHON -m pip install --no-deps --target $PYDEPS gsplat==1.5.3+pt24cu124 --index-url https://docs.gsplat.studio/whl/pt24cu124
       Patch:   cd $REPO && $PYTHON tools/patch_gsplat_distributed_packed.py --path $PYDEPS/gsplat/rendering.py"
import sys
sys.path.insert(0, sys.argv[1])
import gsplat
from strategies.tide_engine import gsplat_backend as g
print("  gsplat", gsplat.__version__, gsplat.__file__)
g.require_distributed_gsplat()
g._require_projection_timing_fix(gsplat)   # required by --tide_detailed_metrics
g.require_gsplat_cuda_backend()            # loads the compiled kernels on this node's GPU
print("  gsplat distributed guard, timing patch and CUDA backend: OK")
GSPLAT_EOF

# The camera count must be divisible by the rank count (distributed_plan.py enforces this).
if (( N_TRAIN_CAM_TOTAL % NGPU != 0 )); then
  fail "Training camera count $N_TRAIN_CAM_TOTAL is not divisible by NGPU=$NGPU; distributed planning will fail.
       Use NGPU=2, 4, or 8. If NGPU=3 is required, also set --num_train_cameras to $((N_TRAIN_CAM_TOTAL / NGPU * NGPU))"
fi

# ---------------------------------------------------------------------------
# 1. SSD base.
# ---------------------------------------------------------------------------
say "Preflight 1/N: SSD base"
for f in base_file.bin block_bounds.npy streaming_init_manifest.json; do
  [[ -e "$SSD_BASE_DIR/$f" ]] || fail "Missing $SSD_BASE_DIR/$f

  If you only have a PLY, generate it with the following command (about 37 minutes; knn3 requires simple-knn/distCUDA2):
      cd $REPO && $PYTHON -m storage.streaming_ply_init \\
        --ply <PLY path> --output $SSD_BASE_DIR \\
        --scale-mode knn3 --block-size 4096 --bucket-bits 10 --sort-memory-mb 4096

  Note that --sort-memory-mb must remain 4096: knn3 uses the in-memory sort unit as its neighborhood,
  so changing it produces different scaling values and incomparable results."
done

actual_bytes=$(stat -c %s "$SSD_BASE_DIR/base_file.bin")
[[ "$actual_bytes" == "$BASE_EXPECT_BYTES" ]] || fail "base_file.bin is $actual_bytes bytes; expected $BASE_EXPECT_BYTES.
       A size mismatch means this is not the same 1B base or the transfer is incomplete."

"$PYTHON" - "$SSD_BASE_DIR/streaming_init_manifest.json" "$BASE_EXPECT_SCALE_MODE" <<'PYEOF' \
  || fail "Manifest scale_mode does not match; see the actual value above."
import json, sys
m = json.load(open(sys.argv[1])); want = sys.argv[2]
got = m.get("scale_mode")
print("  scale_mode =", got, " total_points =", m.get("total_points"),
      " num_blocks =", m.get("num_blocks"), " block_size =", m.get("block_size"))
assert got == want, f"Expected scale_mode={want}, got {got}"
sort_mb = (m.get("external_sort") or {}).get("max_sort_memory_mb")
print("  sort_memory_mb =", sort_mb)
if sort_mb != 4096:
    print("  WARNING: base built with --sort-memory-mb", sort_mb, "(handoff baseline: 4096);",
          "knn3 scales differ, so results are not directly comparable with that baseline")
PYEOF

# The old and new bases have exactly the same byte size; only the three scaling columns differ.
# The scale_mode check above is the only reliable discriminator.

if [[ -n "$RESUME_FROM" ]]; then
  [[ -f "$RESUME_FROM/pure_ssd_distributed_checkpoint.json" ]] \
    || fail "RESUME_FROM is not a distributed checkpoint (no pure_ssd_distributed_checkpoint.json): $RESUME_FROM"
  say "  resume from $RESUME_FROM"
fi

# ---------------------------------------------------------------------------
# 2. Scene and cameras.
# ---------------------------------------------------------------------------
say "Preflight 2/N: scene and camera JSON"
[[ -d "$SCENE_DIR" ]] || fail "Scene directory not found: $SCENE_DIR"
for f in transforms_train.json transforms_test.json; do
  [[ -e "$SCENE_DIR/$f" ]] || fail "Missing $SCENE_DIR/$f"
done
"$PYTHON" - "$SCENE_DIR" "$N_TRAIN_CAM_TOTAL" "$N_TEST_CAM_TOTAL" <<'PYEOF' \
  || fail "Camera count mismatch"
import json, sys, os
d = sys.argv[1]
tr = json.load(open(os.path.join(d, "transforms_train.json")))
te = json.load(open(os.path.join(d, "transforms_test.json")))
print(f"  train frames = {len(tr['frames'])}   test frames = {len(te['frames'])}")
assert len(tr["frames"]) == int(sys.argv[2]), "Training camera count mismatch"
assert len(te["frames"]) == int(sys.argv[3]), "Test camera count mismatch"
fn = tr["frames"][0]["file_name"]
print("  Example file_name =", fn)
assert "/" in fn, "file_name has no block prefix; image keys must include the block name to avoid collisions"
PYEOF

# ---------------------------------------------------------------------------
# 3. Decoded-image cache (training builds it automatically if missing).
# ---------------------------------------------------------------------------
say "Preflight 3/N: decoded-image cache"
if [[ -d "$DECODE_DIR/dataset_raw" ]]; then
  n=$(find "$DECODE_DIR/dataset_raw" -type f 2>/dev/null | wc -l)
  say "  dataset_raw exists with $n files (expected 60698)"
  [[ "$n" == "60698" ]] || say "  ⚠️ File count differs from the expected value; the cache may be incomplete. Training will reuse it without rebuilding missing files."
else
  avail_gb=$(df -BG --output=avail "$(dirname "$DECODE_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9')
  say "  dataset_raw is missing; **the first training run will build it automatically**, using about 469 GiB"
  say "  Available space: ${avail_gb:-unknown} GiB (need at least 469 GiB)"
  [[ -z "${avail_gb:-}" || "$avail_gb" -ge 469 ]] || fail "Less than 469 GiB is available"
fi

# ---------------------------------------------------------------------------
# 4. Evaluator and lpips.
# ---------------------------------------------------------------------------
say "Preflight 4/N: evaluator"
for f in scripts/evaluate_missing_checkpoints.py tools/evaluate_pure_ssd_native.py tools/eval_protocol.py; do
  [[ -e "$REPO/$f" ]] || fail "Missing evaluator file $REPO/$f"
done
"$PYTHON" -c "import lpips" 2>/dev/null \
  || say "  ⚠️ lpips is not installed; the LPIPS metric will fail (PSNR/SSIM are unaffected). Install it with pip install lpips."

mkdir -p "$MODEL" "$CACHE" "$SCHED_CACHE"
say "All preflight checks passed. NGPU=$NGPU BSZ=$BSZ(global) ITERS=$ITERS resident capacity=$PER_CARD_CAP/GPU = $RESIDENT_CAP(global)"
say "Output directory: $RUN_ROOT"

if (( CHECK_ONLY )); then
  say "--check-only was specified; stopping here."
  exit 0
fi

# ---------------------------------------------------------------------------
# 5. Training (all settings except NGPU and resident capacity match the validated 4-GPU baseline).
# ---------------------------------------------------------------------------
{
  echo "host=$(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "ngpu=$NGPU bsz=$BSZ iters=$ITERS per_card_cap=$PER_CARD_CAP resident_cap=$RESIDENT_CAP resident_policy=$RESIDENT_POLICY"
  echo "repo_commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "ssd_base=$SSD_BASE_DIR"
  echo "scene=$SCENE_DIR"
  echo "resume_from=${RESUME_FROM:-none}"
  echo "pythonpath=$PYTHONPATH"
  echo "slurm_job_id=${SLURM_JOB_ID:-none}"
} | tee "$RUN_ROOT/run_info.txt"

export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Make SIGBUS/SIGSEGV print a Python traceback instead of only "Signal 7".
export PYTHONFAULTHANDLER=1

if [[ -n "$RESUME_FROM" ]]; then
  INIT_ARGS=(--start_checkpoint "$RESUME_FROM")
else
  INIT_ARGS=(--pure_ssd_prebuilt_manifest "$SSD_BASE_DIR/streaming_init_manifest.json")
fi

cd "$REPO"
say "===== Training started $(date -Is) ====="
"$PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="$NGPU" "$REPO/train_tidegs.py" \
  -s "$SCENE_DIR" \
  --model_path "$MODEL" --iterations "$ITERS" --checkpoint_iterations $CHECKPOINTS \
  --dense_ply_file "$SSD_BASE_DIR/base_file.bin" \
  --decode_dataset_path "$DECODE_DIR" --bsz "$BSZ" \
  --debug_max_train_cameras -1 --debug_camera_sample_mode linspace --debug_camera_sample_start 0 \
  --disable_auto_densification --sparse_adam --enable_timer --check_gpu_memory --check_cpu_memory \
  --initial_point_cloud_downsampled_ratio 1.0 --use_ssd_offload --pure_ssd_offload \
  --pure_ssd_init_backend streaming "${INIT_ARGS[@]}" \
  --use_6plane --ssd_cache_dir "$CACHE" --gaussian_block_size 4096 --max_ram_gb 32 --num_clusters 64 \
  --ssd_schedule_ordering trajectory --tide_optimizer_backend gpu_resident --optimizer adam \
  --tide_block_reader_backend tiered_cache --tide_optimizer_deferred_mode off \
  --tide_block_cull_backend gpu \
  --tide_resident_selection_policy "$RESIDENT_POLICY" --tide_resident_lambda 0.3 \
  --tide_resident_recency_decay 0.95 --tide_balanced_seed_fraction 0.25 \
  --tide_resident_capacity_blocks "$RESIDENT_CAP" \
  --tide_optimizer_state_mode resident_blocks --tide_distributed_mode gaussian_sharded \
  --tide_camera_assignment equal --tide_camera_microbatch 4 --tide_owner_balance_samples 256 \
  --projection_max_cameras_per_chunk 2 --pure_ssd_checkpoint_mode incremental \
  --pure_ssd_checkpoint_patch_mode hardlink --pure_ssd_checkpoint_keep_last "$KEEP_LAST" \
  --tide_storage_max_patch_files 32 --tide_storage_max_patch_gb 64 --tide_storage_min_free_gb 64 \
  --tide_storage_compaction_interval_iterations 5000 --tide_storage_compaction_target_patch_files 8 \
  --tide_storage_compaction_rank_concurrency 2 --tide_storage_compaction_emergency_free_gb -1 \
  --tide_storage_idle_compaction_seconds 0 --tide_debug_logging --tide_detailed_metrics \
  --quiet --tide_free_unified_params --pure_ssd_schedule_cache_dir "$SCHED_CACHE" \
  2>&1 | tee -a "$RUN_ROOT/train.log"
say "===== Training finished $(date -Is) ====="

# ---------------------------------------------------------------------------
# 6. Evaluation (upstream 200-view protocol).
# ---------------------------------------------------------------------------
# Non-fatal: training and checkpoint creation have succeeded, so an evaluation failure should not mark the whole run as failed.
# gaussian_sharded training writes pure_ssd_distributed_checkpoint.json plus one rank_<r>/ sub-checkpoint per
# rank, and each rank only holds trained values for the blocks it owns. The ported 200-view evaluator looks for
# a single pure_ssd_checkpoint.json, so on these checkpoints it reports every iteration as unavailable and exits 0.
# It stays off until it merges ranks by block_owner; RUN_EVAL=1 is only meaningful for single-process checkpoints.
if [[ "$RUN_EVAL" != "1" ]]; then
  say "Evaluation skipped (RUN_EVAL=$RUN_EVAL): the 200-view evaluator cannot read distributed checkpoints yet."
  say "Checkpoints: $MODEL/checkpoints/"
  exit 0
fi
say "===== Evaluation started $(date -Is) ====="
"$PYTHON" "$REPO/scripts/evaluate_missing_checkpoints.py" \
  --run_dir "$MODEL" \
  --iterations $CHECKPOINTS \
  --views 200 \
  --camera_sample_mode linspace \
  --lpips auto \
  --python "$PYTHON" \
  --no_wait_compaction \
  2>&1 | tee -a "$RUN_ROOT/eval.log" \
  || say "⚠️ Evaluation exited non-zero; checkpoints are intact and evaluation can be rerun (completed evaluations are skipped)."

say "===== Evaluation finished $(date -Is) ====="
say "Result directory: $MODEL/evaluations/"
say "Metrics: $MODEL/evaluations/*/metrics_summary.json (see mean_psnr / mean_ssim)"
