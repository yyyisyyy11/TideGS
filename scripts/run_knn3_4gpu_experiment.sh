#!/usr/bin/env bash
# ============================================================================
# knn3 SSD base + 多卡「pure SSD」训练 + 上游 200 视角评测
#
# 卡数由 NGPU 决定，默认为 2。文件名里的 "4gpu" 说的是配置血缘——这套参数来自
# 已跑通的 4 卡基线，NGPU 只是把同一份计算摊到不同数量的 rank 上。
#
# 用法：
#   1) 按需修改下面 CONFIG 段的路径（都有默认值，默认按 DATA_ROOT 推）
#   2) 在**已经分配到 GPU 的节点上**直接运行：
#          bash scripts/run_knn3_4gpu_experiment.sh
#      脚本不自己提交作业——请在调度器分配好的环境里跑（Slurm 用 srun --pty bash、
#      或 sbatch 提交本脚本）。
#   3) 想先只检查环境不训练：  bash scripts/run_knn3_4gpu_experiment.sh --check-only
#
# 脚本做三段：预检 -> 训练 -> 评测。预检失败会明确告诉你缺什么、怎么补。
# ============================================================================
set -euo pipefail

say()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 2; }

# ============================== CONFIG ======================================
# 仓库根目录（默认取本脚本所在目录的上一级）
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Python 解释器：必须已装好 torch 2.4.x + CUDA 12.x + gsplat 1.5.x + simple-knn(distCUDA2)
PYTHON="${PYTHON:-python}"

# 数据根目录：下面应包含 matrixcity/ 、decoded_* / 、pointcloud/
DATA_ROOT="${DATA_ROOT:-$REPO/data}"

# 场景目录（含 transforms_train.json / transforms_test.json）
SCENE_DIR="${SCENE_DIR:-$DATA_ROOT/matrixcity/big_city/aerial/pose/all_blocks}"
# 解码图像缓存目录（缺失时训练会自动构建，约 469 GiB，需要足够剩余空间）
DECODE_DIR="${DECODE_DIR:-$DATA_ROOT/decoded_matrixcity_full_v1}"

# SSD base 目录（含 base_file.bin / block_bounds.npy / streaming_init_manifest.json）
SSD_BASE_DIR="${SSD_BASE_DIR:-$DATA_ROOT/ssd_base_1b}"

# 输出：本次运行的 tag 与产物目录
TAG="${TAG:-knn3_4gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO/runs/$TAG}"
MODEL="$RUN_ROOT/train_knn3_4gpu"
CACHE="$RUN_ROOT/ssd_cache"
SCHED_CACHE="${SCHED_CACHE:-$REPO/runs/schedule_cache/oneb_bigcity_tsp}"

# 训练规模。NGPU 必须是 51632 的约数（2、4、8 都可以；3 不行，需要裁剪相机）
NGPU="${NGPU:-2}"
BSZ="${BSZ:-64}"                      # **全局** batch；保持不变才能与 4 卡基线可比
ITERS="${ITERS:-200000}"              # 相机迭代数（不是优化器步数）
CHECKPOINTS="${CHECKPOINTS:-50000 100000 150000 200000}"

# 驻留块上限按**每张卡**给，全局值 = 每卡 × NGPU。
# 6144/卡、2 卡 → 全局 12288。
PER_CARD_CAP="${PER_CARD_CAP:-6144}"

# 数值参数必须在校验通过之后才能拿去算术运算：非数字在 set -u 下会被当成变量名，
# 报 "unbound variable" 而不是"你写错了"。
for _v in NGPU PER_CARD_CAP BSZ ITERS; do
  _val="${!_v}"
  if [[ ! "$_val" =~ ^[0-9]+$ ]] || (( _val <= 0 )); then
    fail "$_v 必须是正整数，实际是 '$_val'（注意别把多个赋值挤进一个环境变量）"
  fi
done
unset _v _val

if [[ -z "${RESIDENT_CAP:-}" ]]; then
  RESIDENT_CAP=$(( PER_CARD_CAP * NGPU ))
fi
if [[ ! "$RESIDENT_CAP" =~ ^[0-9]+$ ]] || (( RESIDENT_CAP <= 0 )); then
  fail "RESIDENT_CAP 必须是正整数，实际是 '$RESIDENT_CAP'"
fi

# 训练相机总数（该数据集固定 51632）
N_TRAIN_CAM_TOTAL=51632
N_TEST_CAM_TOTAL=9066

# SSD base 的期望大小与版本（用于预检）
BASE_EXPECT_BYTES=263199283200
BASE_EXPECT_SCALE_MODE=knn3
# ============================================================================

CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then CHECK_ONLY=1; fi

# 先把解析出来的配置打出来，便于核对（预检不过也能看到）
say "配置 NGPU=$NGPU  BSZ=$BSZ(全局)  ITERS=$ITERS  驻留上限=$PER_CARD_CAP/卡 = $RESIDENT_CAP(全局)"
say "  base=$SSD_BASE_DIR"
say "  场景=$SCENE_DIR"
say "  输出=$RUN_ROOT"

# ---------------------------------------------------------------------------
# 0. 前置：GPU 可见性
# ---------------------------------------------------------------------------
say "预检 0/N  解释器与 GPU"
[[ -x "$(command -v "$PYTHON")" ]] || fail "找不到解释器：$PYTHON"
"$PYTHON" - <<PYEOF || fail "torch/CUDA 不可用，检查驱动与 CUDA_VISIBLE_DEVICES"
import torch, sys
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "torch.cuda.is_available() 为 False"
n = torch.cuda.device_count()
print("可见 GPU 数:", n, [torch.cuda.get_device_name(i) for i in range(n)])
assert n >= $NGPU, f"只看到 {n} 张卡，但 NGPU=$NGPU"
PYEOF

# 相机总数必须能被 rank 数整除（distributed_plan.py 有硬校验）
if (( N_TRAIN_CAM_TOTAL % NGPU != 0 )); then
  fail "训练相机 $N_TRAIN_CAM_TOTAL 不能被 NGPU=$NGPU 整除，分布式规划会直接报错。
       NGPU 取 2 / 4 / 8；若必须用 3，要同时把 --num_train_cameras 设成 $((N_TRAIN_CAM_TOTAL / NGPU * NGPU))"
fi

# ---------------------------------------------------------------------------
# 1. 前置：SSD base
# ---------------------------------------------------------------------------
say "预检 1/N  SSD base"
for f in base_file.bin block_bounds.npy streaming_init_manifest.json; do
  [[ -e "$SSD_BASE_DIR/$f" ]] || fail "缺少 $SSD_BASE_DIR/$f

  若手上只有 PLY，用下面命令生成（约 37 分钟，需 knn3 依赖 simple-knn/distCUDA2）：
      cd $REPO && $PYTHON -m storage.streaming_ply_init \\
        --ply <PLY路径> --output $SSD_BASE_DIR \\
        --scale-mode knn3 --block-size 4096 --bucket-bits 10 --sort-memory-mb 4096

  注意 --sort-memory-mb 必须同为 4096：knn3 的邻域就是内存内排序单元，
  换值会得到不同的 scaling，结果不可比。"
done

actual_bytes=$(stat -c %s "$SSD_BASE_DIR/base_file.bin")
[[ "$actual_bytes" == "$BASE_EXPECT_BYTES" ]] || fail "base_file.bin 大小是 $actual_bytes，期望 $BASE_EXPECT_BYTES。
       大小不对说明这不是同一份 1B base（或传输不完整）。"

"$PYTHON" - "$SSD_BASE_DIR/streaming_init_manifest.json" "$BASE_EXPECT_SCALE_MODE" <<'PYEOF' \
  || fail "manifest 的 scale_mode 不匹配（见上面的实际值）"
import json, sys
m = json.load(open(sys.argv[1])); want = sys.argv[2]
got = m.get("scale_mode")
print("  scale_mode =", got, " total_points =", m.get("total_points"),
      " num_blocks =", m.get("num_blocks"), " block_size =", m.get("block_size"))
assert got == want, f"期望 scale_mode={want}，实际 {got}"
PYEOF

# 旧的 base 与新 base **字节数完全相同**，差异只在 scaling 三列，看大小分不出来。
# 唯一可靠的判据就是上面这个 scale_mode。

# ---------------------------------------------------------------------------
# 2. 前置：场景与相机
# ---------------------------------------------------------------------------
say "预检 2/N  场景与相机 JSON"
[[ -d "$SCENE_DIR" ]] || fail "找不到场景目录 $SCENE_DIR"
for f in transforms_train.json transforms_test.json; do
  [[ -e "$SCENE_DIR/$f" ]] || fail "缺少 $SCENE_DIR/$f"
done
"$PYTHON" - "$SCENE_DIR" "$N_TRAIN_CAM_TOTAL" "$N_TEST_CAM_TOTAL" <<'PYEOF' \
  || fail "相机数量不符"
import json, sys, os
d = sys.argv[1]
tr = json.load(open(os.path.join(d, "transforms_train.json")))
te = json.load(open(os.path.join(d, "transforms_test.json")))
print(f"  train frames = {len(tr['frames'])}   test frames = {len(te['frames'])}")
assert len(tr["frames"]) == int(sys.argv[2]), "训练相机数不符"
assert len(te["frames"]) == int(sys.argv[3]), "测试相机数不符"
fn = tr["frames"][0]["file_name"]
print("  样例 file_name =", fn)
assert "/" in fn, "file_name 里没有块名前缀——图像键必须带块名，否则会串块"
PYEOF

# ---------------------------------------------------------------------------
# 3. 前置：解码缓存（缺失则由训练自动构建）
# ---------------------------------------------------------------------------
say "预检 3/N  解码图像缓存"
if [[ -d "$DECODE_DIR/dataset_raw" ]]; then
  n=$(find "$DECODE_DIR/dataset_raw" -type f 2>/dev/null | wc -l)
  say "  已存在 dataset_raw，$n 个文件（期望 60698）"
  [[ "$n" == "60698" ]] || say "  ⚠️ 文件数与期望不符，可能不完整；训练会按已有文件复用，不会补建"
else
  avail_gb=$(df -BG --output=avail "$(dirname "$DECODE_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9')
  say "  dataset_raw 不存在——**首次训练会自动构建**，约 469 GiB"
  say "  当前可用空间：${avail_gb:-未知} GiB（需 ≥ 469 GiB）"
  [[ -z "${avail_gb:-}" || "$avail_gb" -ge 469 ]] || fail "剩余空间不足 469 GiB"
fi

# ---------------------------------------------------------------------------
# 4. 前置：评测器与 lpips
# ---------------------------------------------------------------------------
say "预检 4/N  评测器"
for f in scripts/evaluate_missing_checkpoints.py tools/evaluate_pure_ssd_native.py tools/eval_protocol.py; do
  [[ -e "$REPO/$f" ]] || fail "缺少评测器文件 $REPO/$f"
done
"$PYTHON" -c "import lpips" 2>/dev/null \
  || say "  ⚠️ 没装 lpips——评测的 LPIPS 指标会失败（PSNR/SSIM 不受影响）。可 pip install lpips"

mkdir -p "$MODEL" "$CACHE" "$SCHED_CACHE"
say "全部预检通过。NGPU=$NGPU BSZ=$BSZ(全局) ITERS=$ITERS 驻留上限=$PER_CARD_CAP/卡 = $RESIDENT_CAP(全局)"
say "输出目录：$RUN_ROOT"

if (( CHECK_ONLY )); then
  say "--check-only 指定，到此为止。"
  exit 0
fi

# ---------------------------------------------------------------------------
# 5. 训练（除 NGPU 与驻留上限外，其余参数与已验证的 4 卡基线逐字一致）
# ---------------------------------------------------------------------------
{
  echo "host=$(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "ngpu=$NGPU bsz=$BSZ iters=$ITERS per_card_cap=$PER_CARD_CAP resident_cap=$RESIDENT_CAP"
  echo "repo_commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "ssd_base=$SSD_BASE_DIR"
  echo "scene=$SCENE_DIR"
} | tee "$RUN_ROOT/run_info.txt"

export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# 让 SIGBUS/SIGSEGV 打印 Python 调用栈，而不是只留一行 "Signal 7"
export PYTHONFAULTHANDLER=1

cd "$REPO"
say "===== 训练开始 $(date -Is) ====="
"$PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="$NGPU" "$REPO/train_tidegs.py" \
  -s "$SCENE_DIR" \
  --model_path "$MODEL" --iterations "$ITERS" --checkpoint_iterations $CHECKPOINTS \
  --dense_ply_file "$SSD_BASE_DIR/base_file.bin" \
  --decode_dataset_path "$DECODE_DIR" --bsz "$BSZ" \
  --debug_max_train_cameras -1 --debug_camera_sample_mode linspace --debug_camera_sample_start 0 \
  --disable_auto_densification --sparse_adam --enable_timer --check_gpu_memory --check_cpu_memory \
  --initial_point_cloud_downsampled_ratio 1.0 --use_ssd_offload --pure_ssd_offload \
  --pure_ssd_init_backend streaming --pure_ssd_prebuilt_manifest "$SSD_BASE_DIR/streaming_init_manifest.json" \
  --use_6plane --ssd_cache_dir "$CACHE" --gaussian_block_size 4096 --max_ram_gb 32 --num_clusters 64 \
  --ssd_schedule_ordering trajectory --tide_optimizer_backend gpu_resident --optimizer adam \
  --tide_block_reader_backend tiered_cache --tide_optimizer_deferred_mode off \
  --tide_block_cull_backend gpu \
  --tide_resident_selection_policy topc_balanced --tide_resident_lambda 0.3 \
  --tide_resident_recency_decay 0.95 --tide_balanced_seed_fraction 0.25 \
  --tide_resident_capacity_blocks "$RESIDENT_CAP" \
  --tide_optimizer_state_mode resident_blocks --tide_distributed_mode gaussian_sharded \
  --tide_camera_assignment equal --tide_camera_microbatch 4 --tide_owner_balance_samples 256 \
  --projection_max_cameras_per_chunk 2 --pure_ssd_checkpoint_mode incremental \
  --pure_ssd_checkpoint_patch_mode hardlink --pure_ssd_checkpoint_keep_last 2 \
  --tide_storage_max_patch_files 32 --tide_storage_max_patch_gb 64 --tide_storage_min_free_gb 64 \
  --tide_storage_compaction_interval_iterations 5000 --tide_storage_compaction_target_patch_files 8 \
  --tide_storage_compaction_rank_concurrency 2 --tide_storage_compaction_emergency_free_gb -1 \
  --tide_storage_idle_compaction_seconds 0 --tide_debug_logging --tide_detailed_metrics \
  --quiet --tide_free_unified_params --pure_ssd_schedule_cache_dir "$SCHED_CACHE" \
  2>&1 | tee -a "$RUN_ROOT/train.log"
say "===== 训练结束 $(date -Is) ====="

# ---------------------------------------------------------------------------
# 6. 评测（上游 200 视角协议）
# ---------------------------------------------------------------------------
# 非致命：训练已经成功、checkpoint 已落盘，评测失败不该把整体判成失败。
say "===== 评测开始 $(date -Is) ====="
"$PYTHON" "$REPO/scripts/evaluate_missing_checkpoints.py" \
  --run_dir "$MODEL" \
  --iterations $CHECKPOINTS \
  --views 200 \
  --camera_sample_mode linspace \
  --lpips auto \
  --python "$PYTHON" \
  --no_wait_compaction \
  2>&1 | tee -a "$RUN_ROOT/eval.log" \
  || say "⚠️ 评测非零退出；checkpoint 完好，可重跑评测（它会跳过已完成的）"

say "===== 评测结束 $(date -Is) ====="
say "结果目录：$MODEL/evaluations/"
say "指标文件：$MODEL/evaluations/*/metrics_summary.json   （看 mean_psnr / mean_ssim）"
