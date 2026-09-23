# handoff/knn3-4gpu — 可直接跑的实验分支

这个分支把**跑通这个实验所需的全部东西**放在一处：训练代码、上游评测器、运行脚本、数据要求。
目的只有一个——换一台机器也能跑起来，不需要再去别处凑代码。

---

## 1. 这个分支由什么组成

| 部分 | 来源 |
|---|---|
| 训练代码（主体） | `9c7c270` "fix: correct _write_paper_phase1_log call in tile-mask apply path" |
| 上游评测器（4 个文件） | cherry-pick 自 `ad6bc18` "Port the upstream evaluation pipeline" |
| USS 采样节流（3 行） | cherry-pick 自 `4ac3a4c` "Throttle USS sampling in the memory monitor" |
| 运行脚本 | `scripts/run_knn3_4gpu_experiment.sh`（本分支新增） |

`ad6bc18` 的改动是**纯新增**（4 个文件、1357 行，不动任何既有文件），`4ac3a4c` 只改了内存监视器：
把读 `/proc/<pid>/smaps` 的 USS 采样从每次降到每 100 次。**它不影响任何数值结果**，纯粹是避免
在一个映射了几百 GB 的进程上反复走 smaps 拖慢训练。

评测器的四个文件与上游逐字节一致的是 `eval_protocol.py` / `evaluate_missing_checkpoints.py` /
`test_eval_protocol.py`；`evaluate_pure_ssd_native.py` 是上游逻辑针对本 fork 的运行时的适配版，
改动集中在四个调用点，详见 [`tools/PORTED_EVALUATOR.md`](tools/PORTED_EVALUATOR.md)。

---

## 2. 硬件要求

| 项 | 要求 | 依据 |
|---|---|---|
| GPU | **80 GB 卡，同一节点上 2 或 4 张** | 实测峰值显存 39.1 GiB（每卡驻留 2048 时）；**40 GB 卡装不下** |
| GPU 架构 | sm_80（A100/A800） | 与已有基线数值可比。换架构需重编 CUDA 扩展，且浮点行为不同 |
| CPU | ≥ 48 核（每卡 12 核） | 已验证配置按 48 核提交 |
| 主机内存 | **≥ 512 GiB，建议 1 TB** | 实测每 rank 的进程树 RSS 到 115 GiB；另一次 run 的 cgroup 峰值到过 397 GB |
| 存储 | **强烈建议本地 NVMe** | 同一份配置在本地 NVMe 上比在共享并行文件系统上快约 **2.9 倍** |

---

## 3. 存储与数据要求

| 数据 | 大小 | 必要性 | 怎么得到 |
|---|---|---|---|
| SSD base | `base_file.bin` = **263,199,283,200 B**（≈245 GiB） | **必需** | 传输，或用下面的命令从 PLY 生成（约 37 分钟） |
| 源 PLY | 30,111,782,613 B（≈28 GiB） | 只在生成 base 时需要 | 传输 |
| 解码图像缓存 | **503,453,491,200 B**（≈469 GiB），60698 个文件 | **必需** | **训练会自动构建**（见下），或传输 |
| 原始 PNG 图像 | 量级 100 GiB（未实测） | 构建解码缓存时需要 | 传输 |
| 相机 JSON | 几 MB | **必需** | 随场景目录一起 |
| checkpoint / SSD 缓存 | 数百 GiB | 运行中产生 | `--tide_storage_max_patch_files 32 --tide_storage_max_patch_gb 64`；每 5000 迭代一次 compaction，单次约 165 GiB 读写 |

**建议按 ≥ 1.5 TB 高速可用空间准备**（base 245 GiB + 解码缓存 469 GiB + 缓存与 checkpoint 数百 GiB）。

**解码缓存不需要手工准备**：训练启动时若 `<decode_dataset_path>/dataset_raw` 不存在，
会自行创建目录、校验剩余空间（不足会直接 assert 失败）、然后并行解码全部图像写入
（`scene/__init__.py`）。存在则直接复用。

**从 PLY 生成 base**：

```bash
python -m storage.streaming_ply_init \
  --ply <PLY> --output <SSD_BASE_DIR> \
  --scale-mode knn3 --block-size 4096 --bucket-bits 10 --sort-memory-mb 4096
```

需要 `simple-knn` / `distCUDA2` 这个 CUDA 扩展（**只有生成 base 才需要，训练不需要**——
训练用的是预生成的 base）。

⚠️ `--sort-memory-mb` **必须是 4096**：knn3 的邻域就是"内存内排序单元"，
换值会算出不同的 scaling，与已有基线不可比。

---

## 4. 软件环境

| 包 | 已验证版本 |
|---|---|
| Python | 3.10 |
| PyTorch | 2.4.0+cu124 |
| CUDA | 12.4 |
| gsplat | 1.5.3（需为本机架构编译的 CUDA 扩展） |
| numpy / scipy / Pillow / plyfile | 任意近期版本 |
| lpips | 仅用于 LPIPS 指标；缺了不影响 PSNR/SSIM |

`gsplat` 这类 CUDA 扩展必须在目标机器上针对其架构重新编译。

---

## 5. 跑起来

```bash
git clone <repo-url> && cd TideGS
git checkout handoff/knn3-4gpu

# 只检查环境，不训练
DATA_ROOT=/path/to/data bash scripts/run_knn3_4gpu_experiment.sh --check-only

# 正式跑（在已分配到 4 张卡的节点上）
DATA_ROOT=/path/to/data bash scripts/run_knn3_4gpu_experiment.sh
```

脚本的分段：**预检 → 训练 → 评测**。预检会逐项检查 GPU 可见性、base 的大小与 `scale_mode`、
相机数量、解码缓存、评测器文件，缺什么直接告诉你缺什么、怎么补。任何一步不过就停下，不会白跑。

可用环境变量覆盖：`PYTHON` `DATA_ROOT` `SCENE_DIR` `DECODE_DIR` `SSD_BASE_DIR`
`RUN_ROOT` `SCHED_CACHE` `NGPU` `BSZ` `ITERS` `CHECKPOINTS` `PER_CARD_CAP` `RESIDENT_CAP` `TAG`。

**卡数与驻留上限**：默认 `NGPU=2`；驻留上限按**每张卡**给（`PER_CARD_CAP=6144`），
全局值自动等于 `每卡 × NGPU`（2 卡 → 12288，4 卡 → 24576）。想直接指定全局值就用 `RESIDENT_CAP`。
`NGPU` 必须是 51632 的约数（2 / 4 / 8 可以，3 不行）。

⚠️ **驻留上限会改变结果，不只是显存旋钮。** 只有驻留下来的块会被优化
（`--tide_optimizer_state_mode resident_blocks`），所以改这个值就等于改变了"哪些 Gaussian
每次会被更新"。已验证的 4 卡基线用的是**每卡 2048 / 全局 8192**；复现它要
`PER_CARD_CAP=2048 NGPU=4`，不是默认值。想和已有曲线并列比较时，务必先把这一项对齐。

⚠️ 每卡 6144 个驻留块（≈25.2M 个 Gaussian）是基线每卡量的 3 倍，峰值显存会明显高于
基线在 2048/卡 时的 39.1 GiB。`--check_gpu_memory` 会在日志里打显存，留意别贴上限。

---

## 6. 你会拿到什么

训练产物在 `$RUN_ROOT/train_knn3_4gpu/`：

- `checkpoints/` — 按 `CHECKPOINTS` 落盘，`keep_last=2`
- `mem_monitor.csv` — 内存时间线（列：`iter, wall_s, rss_gb, uss_gb, tree_rss_gb, avail_gb, total_gb, swap_used_gb, num_children`）
- `metrics_*.tsv` — 逐迭代的各阶段耗时
- `timeline_events_rank*.jsonl` — 含显存字段
- `evaluations/*/metrics_summary.json` — **最终指标：`mean_psnr` / `mean_ssim`**

评测用的是上游的 **200 视角固定 linspace 协议**（对 `transforms_test.json` 取固定子集，
用所选帧的 sha256 驱动跳过与续跑），每个 checkpoint 一套，比全量 9066 张便宜约 45 倍。

---

## 7. 坑（按踩到的概率排序）

1. **新旧 base 的 `base_file.bin` 字节数完全相同**（都是 263,199,283,200），差异只在
   `scaling` 那三列。看大小、看文件列表都分不出来。**唯一可靠判据是 manifest 的 `scale_mode`**，
   本实验要的是 `knn3`（旧版是 `morton_bucket_density_clamped`）。脚本的预检就是干这个的。

2. **图像缓存键必须带块名。** 相机 `file_name` 形如 `big_high_block_1/0000.png`；
   9066 张测试图里只有 **3772** 个 basename 不重复，只用 basename 做键会把 A 块的相机
   配上 B 块的 GT。历史上正是这个 bug，后改为块感知键。

3. **评测器从未端到端跑通过一次。** 它的协议层有 18 个单测且全过，导入与签名也静态核对过，
   但**没有执行过一次真实评测**。第一次跑请保留完整日志。

4. **多卡长跑在这个实验的历史上不稳定。** 原集群上五次 4 卡长跑：三次直接失败、
   两次死于无调用栈的低层信号（SIGKILL / SIGBUS，根因未确定），只有一次把 50 万迭代跑完
   （约 2 天）但死在评测阶段。**所以把 checkpoint 间隔设密一些**，脚本默认 5 万。

5. **每个 rank 下恒定有 33 个子进程，出处不明。** `mem_monitor.csv` 的 `num_children`
   列在两次 run 的共 610 行里恒为 33，但仓库里所有并发都是线程，
   唯一的多进程函数没有调用点（死代码）。这条与"SIGBUS 可能来自共享内存映射"的假设
   处在同一区域，尚未查清。

6. **多 rank 运行至今没有产出过 PSNR。** 也就是说"`gaussian_sharded` 分布式正确性"
   目前没有任何端到端证据。若拿 2 卡结果外推 4 卡，注意梯度归约的元数（一个 Gaussian
   的贡献来自几个 rank）上限就是 rank 数——2 卡只走 k≤2 的路径，k=3/4 的分支没被覆盖。
