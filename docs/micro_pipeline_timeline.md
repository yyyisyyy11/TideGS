# Micro Pipeline 线程、分段反向与验收

**当前实现统一称为 TideGS Micro Pipeline，运行模式为 `micro`；默认仍为 `off`，默认两个 slot。当前仅有本地 CPU 证据，CUDA/NCCL、原生编译、GIL 实测、实际 overlap 和性能收益均待验证。**

## 改名与兼容

新命令使用 `--tide_forward_pipeline micro`；旧配置值 `pc1` 在入口规范化为 `micro`。新模块、脚本、指标和能力标识使用 `micro_pipeline` / `MICRO_PIPELINE`。旧 `TIDE_PC1_*` 超时变量及原生 `PC1_*` 能力可读，新变量优先。分析工具兼容旧模式、schedule 标签与报告；历史实验及验证原始日志保持原名。

实验人员可先读 [200 字简介](micro_pipeline_experiment_brief.md)，本轮本地回归见 [验证记录](micro_pipeline_validation.md)。

## 定义

| 阶段 | 执行者 | 内容 |
| --- | --- | --- |
| P | 主线程 / compute stream | Projection、SH、参数变换图及 detached 发送数据 |
| C1 | 常驻通信线程 / comm stream | 行数、radii、浮点属性、整数 ID 的显式交换 |
| M | 主线程 / compute stream | Render、真实 Loss、对接收端独立叶子的 IntermediateGrad |
| C2 | 常驻通信线程 / comm stream | 使用 C1 反向 split 返回中间梯度，不再交换行数 |
| G | 主线程 / compute stream | 原始投影输出的 FinalGrad，按 microbatch 顺序累积共享 Gaussian 叶子 |
| Optimizer | 主线程 | 全 batch drain 后一次更新；S2、bounds sync、writeback 保持原有顺序 |

每个 rank 的 `DistributedContext` 复用一个 daemon FIFO 通信线程，队列容量为 `2 × buffer_count`。batch pipeline 只管理自己的 slot；关闭 batch pipeline 不关闭 context 的通信线程。worker 不执行投影、Loss 或参数反向，不修改主线程的图或 active-slot 状态。

## 调度

默认窗口内提交顺序为 `P0/C1入队 → P1/C1入队 → M0/C2入队 → M1/C2入队 → G0 → G1`，collective 顺序为 `C1(0), C1(1), C2(0), C2(1)`。更大窗口采用全部 P、全部 M、全部 G；尾窗口正常处理，单个 microbatch 自然串行。各 rank 在 batch 入口核对窗口配置，不按 Future 完成速度调整顺序。

**Future 完成仅代表 CPU 已发布结果和 event，消费者仍必须 `compute_stream.wait_event(done_event)`。** NCCL `Work.wait()` 在 comm stream 建立结果可用依赖之后才记录 event；slot 在 G 提交后复用，并等待上一使用者的 backward event。跨 stream tensor 使用 `record_stream()`，另用显式 lease 保活 buffer 和 Work。窗口间不做全设备同步，batch 末尾 drain 使用绝对 deadline 查询 event。

## 梯度

原始带图输出与 C1 detached 输入分开保存。接收浮点属性变为独立叶子，M 使用不保图的 `autograd.grad()`；未用属性或无可微路径返回同形零梯度，仍参与 C2。C2 保持 peer 内行顺序，不额外平均或乘 world size。

$$
g_r=\frac{\partial L}{\partial r},\qquad g_z=A^\mathsf{T}g_r,\qquad g_\theta\mathrel{+}=\left(\frac{\partial z}{\partial\theta}\right)^\mathsf{T}g_z.
$$

**只有 G 更新 Gaussian 参数的 `.grad`。** 每个 microbatch 保留独立变换图，共享 batch 叶子；不复制完整参数梯度。不改变背景、相机顺序、Loss 缩放、sentinel、survivor mask、稀疏梯度或 partial-gradient hook 的语义。打包使用显式列宽，支持零行。

## 原生能力

| 扩展 | 必需能力 |
| --- | --- |
| gsplat | ABI 1；绑定来源 `gsplat/cuda/ext.cpp`；packed projection、SH、tile intersection / offset、raster 前后向的 GIL release 列表 |
| clm_kernels | ABI 1；SSIM 前后向 GIL release、当前 PyTorch stream；启用 tile mask 时检查对应操作 |

启动时主线程加载实际扩展，记录 commit、Python / PyTorch / CUDA、扩展路径和能力；缺失能力明确失败。gsplat 使用外层已有现代构建适配器，JIT fallback 也排除旧 `cuda/csrc/ext.cpp`。SSIM 用输入 device guard 和当前 stream 显式 launch。真实 Loss 按 camera shape 在独立 image 叶子上完成前后向编译预热，不更新训练参数。

```bash
export TIDE_CODE=/path/to/TideGS
python /path/to/proj_tidegs_isc/ssh/superpod/micro_pipeline/build_gsplat.py --check-only
python /path/to/proj_tidegs_isc/ssh/superpod/micro_pipeline/build_gsplat.py build_ext --inplace --force
cd "$TIDE_CODE/submodules/clm_kernels"
python setup.py build_ext --inplace --force
cd "$TIDE_CODE"
```

## 故障与关闭

窗口、排队、worker、Future、drain 共享各自已建立的单调时钟 deadline；轮询不延长 deadline。窗口级独立 CPU watchdog 覆盖未细分的 Python 工作。窗口内没有成功路径逐 microbatch Gloo 集合检查；batch 入口和 drain 后保留必要一致性检查。

| 行为 | 约束 |
| --- | --- |
| abort | 拒绝新任务、失败相关 Future、保留在途 buffer；不取消已提交 CUDA/NCCL、不 CUDA 同步、不无限 join |
| 健康关闭 | batch drain → 通信线程 drain / close → watchdog close → 控制 Store 释放 → process group 销毁 |
| 致命故障 | 独立控制器让每个训练进程自行以退出码 86 退出，不能依赖主线程到达 checkpoint |

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| TIDE_MICRO_PIPELINE_STAGE_TIMEOUT_SEC | 60 | 窗口和受保护阶段的秒数；初次编译较慢时提前调整 |
| TIDE_MICRO_PIPELINE_CONTROL_IO_TIMEOUT_SEC | 1 | 控制通道 I/O 秒数 |
| TIDE_MICRO_PIPELINE_ERROR_TIMEOUT_SEC | 30 | 控制 Store 初始化秒数 |

故障测试覆盖 C1 前、C1 metadata、C1 payload、IntermediateGrad 后 C2 前、C2 payload、G、控制通道、关闭时仍有 pending 任务。测试父进程直接启动两个 rank，设置进程级超时，必须同时取得两个 rank 的实际退出码与自身 fatal 文件。

## Timeline

保留 GPU、CPU、SSD 主轨，CPU 内区分 main、comm、autograd、unknown，并保留真实 native thread ID。comm 由通信执行标签识别，autograd 由反向节点标签识别；SSD 预取、watchdog 等没有反向证据的线程进入独立 unknown 行。NVTX 名称使用 `Tide:S1:MB<i>:<stage>[:node]|rank=...,batch=...,slot=...,seq=...,tid=...,schedule=micro-pipeline-window-v2`。普通范围在本线程用异常安全的 push/pop；autograd hook 使用可配对清理的 range ID，避免异常时跨线程 pop。

| 记录 | 统计口径 |
| --- | --- |
| Projection、Render、Loss、IntermediateGrad、C2、FinalGrad、AccumulateGrad | 父阶段用于组织，节点用于细分；聚合区间先取并集 |
| CPU enqueue、worker、Future wait | host 单调时钟及真实提交线程 |
| GPU event | 相对 batch GPU anchor；不能替代实际 kernel 交集 |
| gsplat_forward_ms | 各 microbatch 从 P 到 Render 完成的延迟之和 |
| backward_ms | 各 microbatch 从 IntermediateGrad 到最终累积结束的延迟之和 |
| train_ms | forward / Loss / backward / Optimizer / S2 的约定区间并集；writer 保留此结果 |
| projection_c1_overlap_ms | **envelope estimate**，不能当作 Nsight 实际 overlap |

## 实际 overlap

共享分析脚本位于项目外层 `experiment/timeline/scripts/`；Git 内的 [tools/micro_pipeline_timeline](../tools/micro_pipeline_timeline/README.md) 保存同内容的可审阅快照与 SHA-256 manifest，独立 checkout 可直接使用。历史实验快照、原始报告和派生结果保持不变。新版按同进程全部线程建立 NVTX 索引，用 runtime correlation ID 关联真实 kernel；嵌套范围只归属最深节点一次。无法关联的 kernel 保留为 `unknown`，缺少目标必要阶段或 kernel 证据时 overlap 为 null，不按时间接近推断阶段。

| target_type | 实际交集 |
| --- | --- |
| projection_c1 | P(i+1) 与 C1(i) |
| image_c1 | Render / Loss / IntermediateGrad(i) 与 C1(i+1) |
| image_intermediate_c1 | 上一项中 IntermediateGrad 的独立子项 |
| intermediate_c2 | IntermediateGrad(i+1) 与 C2(i) |
| final_c2 | FinalGrad / AccumulateGrad(i) 与 C2(i+1) |

只计算**同设备、不同 stream**的 kernel 区间交集，先取并集，按配对 / batch / rank 输出样本数。slot 回绕标识新窗口，不产生跨窗口目标配对。旧报告继续读取，仅有旧 P/C1 标签时不声称存在分段反向目标。

提交顺序摘要的 `target_next_projection_c1_overlap_ms_per_batch` 只对证据完整的目标 batch 求均值；同 batch 任一目标配对未知，则整个 batch 保持未知。摘要同时输出 complete / unknown / no-pair batch 数及 `target_next_projection_c1_coverage_fraction`；全部未知时均值为 `null`，已证实没有交集时保留 `0`。新版使用 batch 目标区间并集；旧版保留完整配对求和口径。fine-tag 归属量只表示已归属的下界。

```bash
export TIDE_RUN_CUDA_SMOKE=1
export TIDE_NSYS_OUTPUT=/new/output/nsys
mkdir "$TIDE_NSYS_OUTPUT"
timeout 600s torchrun --standalone --nproc_per_node=2 --no-python \
  bash -c 'nsys profile --trace=cuda,nvtx,osrt --force-overwrite=false -o "$TIDE_NSYS_OUTPUT/rank$LOCAL_RANK" python -m unittest tests.test_distributed_pipeline_cuda -v'
nsys export --type sqlite --output /new/output/rank0.sqlite /new/output/nsys/rank0.nsys-rep
nsys export --type sqlite --output /new/output/rank1.sqlite /new/output/nsys/rank1.nsys-rep
python /path/to/proj_tidegs_isc/experiment/timeline/scripts/analyze_micro_pipeline_overlap.py \
  --run-dir /new/output/run --sqlite /new/output/rank0.sqlite /new/output/rank1.sqlite \
  --output /new/output/derived/micro-pipeline-overlap --emit-kernels
python /path/to/proj_tidegs_isc/experiment/timeline/scripts/analyze_micro_pipeline_submission_order.py \
  --derived-dir /new/output/derived --output /new/output/derived/submission-order
```

上例在每个 rank 单独采集；分析 smoke 时使用 `--trim-batches 0`。代表性训练采集应复用已有逐 rank profiling wrapper，保存各 rank 的独立报告；微小 smoke 不强制出现 overlap。分析器默认丢弃捕获的首尾完整 batch，可用参数明确指定；不得把 CPU 提交范围当作 GPU 执行时间。

## 配对短跑

`tools/run_micro_pipeline_paired_benchmark.py` 使用同一份 JSON 配置，执行至少两轮交替顺序：`off → micro`、`micro → off`。只启动本地命令，不提交 Slurm 作业；每次要求新目录。`--prepare-only` 保存计划与环境信息，不运行训练。

| 配置字段 | 要求 |
| --- | --- |
| command | 现有已验证短跑的 argv 列表，包含且只包含一次 `{mode}`，并包含 `{run_dir}`、`{checkpoint}` |
| checkpoint | 两模式共同初始 checkpoint；文件计算内容 hash，目录记录文件大小 / mtime 的元数据指纹 |
| camera_schedule | 固定相机参考文件或目录，必填；`{camera_schedule}` 占位符可选，实际次序另核验输出 camera TSV |
| working_directory / environment | 共同工作目录和字符串环境变量；不经过 shell 求值 |
| rounds / timeout_seconds | 至少两轮；单次运行的外部进程组超时 |
| warmup_batches / exclude_tail_batches | 明确排除的预热及尾部 batch 数 |
| expected_world_size / seed | 预期 rank 数和固定种子；可用 `{seed}` 传入命令 |

```bash
python tools/run_micro_pipeline_paired_benchmark.py run /path/to/paired-config.json \
  --output-dir /new/output/prepare-only --prepare-only
python tools/run_micro_pipeline_paired_benchmark.py run /path/to/paired-config.json \
  --output-dir /new/output/paired-runs
python tools/run_micro_pipeline_paired_benchmark.py summarize /new/output/paired-runs \
  --overlap-report round-00-off=/new/output/round00-off-analysis.json \
  --overlap-report round-00-micro=/new/output/round00-micro-analysis.json \
  --overlap-report round-01-micro=/new/output/round01-micro-analysis.json \
  --overlap-report round-01-off=/new/output/round01-off-analysis.json
```

配置应使用真实训练入口 `train_tidegs.py` 与原有参数，如 `--start_checkpoint {checkpoint}`、`--tide_forward_pipeline {mode}`、`--log_folder {run_dir}`；沿用已验证的数据、优化器及短跑步数配置。训练入口没有新增相机调度文件参数。

**`batch_total_ms` 只作为引擎内部耗时诊断，不能代表完整训练周期。** 完整周期及加速比要求独立 Nsight 分析的 `tide.batch.step / step_wall_ms`，逐 rank 精确关联 metrics 文件与测量 iteration，并核对实际 camera 次序。缺少证据时 `performance_status=unverified`。摘要保留各 rank 均值 / 中位数 / P95、吞吐、显存峰值、CPU 等待和独立 overlap 报告；不把 event estimate 代入实际 overlap。

## 复现入口

| 验收入口 | 启用变量 |
| --- | --- |
| 双 rank 数值 smoke | `TIDE_RUN_CUDA_SMOKE=1` |
| 原生 SSIM stream / device 测试 | `TIDE_RUN_NATIVE_CUDA=1` |
| 双 rank 故障退出 | `TIDE_RUN_CUDA_ABORT_TEST=1` |

```bash
python3 -m compileall strategies/tide_engine submodules/gsplat/gsplat utils tests tools
python3 -m unittest discover -s tests
git diff --check
TIDE_RUN_CUDA_SMOKE=1 python tools/run_micro_pipeline_cuda_smoke.py --timeout 600
TIDE_RUN_NATIVE_CUDA=1 timeout 120s python -m unittest tests.test_micro_pipeline_native_cuda -v
TIDE_RUN_CUDA_ABORT_TEST=1 python -m unittest tests.test_distributed_pipeline_cuda_faults -v
timeout 240s nsys profile --trace=cuda,nvtx,osrt -o /new/output/micro-pipeline-gil \
  python tools/probe_micro_pipeline_native_gil.py --include-backward --output /new/output/micro-pipeline-gil.json
```

**原生 CUDA 验收需要双卡，结果必须为 2 项执行、0 项跳过。** `OK (skipped=...)` 表示尚未完成验收；设置环境变量不能代替 CUDA 及设备数量条件。

数值 smoke 对比 off、串行分段参考、窗口 Micro Pipeline，覆盖 1–4 个 microbatch、SH 0/3、背景、零分片 / 零行 / sentinel、trace 开关、非默认 stream、MSE / 真实 Loss、多 batch 的全部参数梯度和 Adam 状态。图像容差为 `rtol=1e-4, atol=1e-4`，Loss / 梯度为 `rtol=1e-4, atol=1e-5`；Adam 使用测试内明确的严格容差，不为通过而放宽。真实 GIL 探针只证明原生调用期间其他线程可推进，不能当作训练加速证据。

## 验收标准

| 结论 | 必需证据 | 当前状态 |
| --- | --- | --- |
| 本地实现与行为回归 | CPU 调度、transport、生命周期、能力、指标、跨线程分析测试 | 见 [验证记录](micro_pipeline_validation.md) |
| CUDA 实现正确 | 两扩展编译、双 rank 数值 / 故障 / 多 batch 生命周期通过 | **未执行** |
| 流水线实际形成 | 代表性负载四类目标真实 kernel 交集 | **未执行** |
| 性能改善 | 相同状态 / 相机 / 配置 / GPU 的至少两轮交替配对完整周期与吞吐对照 | **未执行** |

性能对照必须报告各 rank 周期均值 / 中位数 / P95、吞吐、峰值显存、CPU 等待和真实 overlap；有 overlap 但吞吐下降也应如实记录。当前没有 GPU 采集结果，不生成虚构的 `.nsys-rep` 或性能数字，不自动提交长训练作业。
