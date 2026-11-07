# Micro Pipeline 验证记录

**结论：实现及本地回归完成；CUDA/NCCL、原生编译、GIL 实测、实际 overlap 与性能收益待验证。全量测试未全通过，错误集合与干净基线一致。**

## 范围

| 项目 | 记录 |
| --- | --- |
| 日期 / 分支 | 2026-10-15 / `feat/micro-pipeline` |
| 固定基线 | `31675ed4d1e80627220e739d69238be1dddeea72` |
| 本地环境 | macOS、Python 3.12.14、numpy 可用；无 PyTorch、pytest、psutil、nvcc、nvidia-smi |
| 工作树 | 当前 Micro Pipeline checkout（本地物理目录名保留） |
| 范围 | 常驻通信线程、显式 C1/C2、分段梯度、固定窗口、原生能力检查、故障控制、trace 与验证工具 |
| 默认值 | pipeline 为 `off`，buffer_count 为 2；`micro` 为正式名称，旧 `pc1` 为输入别名 |

## 改名回归

**当前正式名称为 TideGS Micro Pipeline，运行模式为 `micro`。代码、原生标识、脚本、测试、分析快照、文档及分支已统一；旧配置 / 超时变量 / 原生产物 / 报告保留读取兼容。** 本地工作树物理目录与历史原始证据保留原名。

| 检查 | 通过 | 错误 | 跳过 |
| --- | ---: | ---: | ---: |
| 定向回归，162 项 | **147** | 0 | 15 |
| 独立 Git 分析快照，33 项 | **33** | 0 | 0 |
| 完整 unittest，391 项 | 330 | 23 | 38 |
| 新 SuperPOD 构建适配器 / runtime | **23** | 0 | 0 |
| compileall / analyzer self-test / Shell 语法 / diff / 7 个脚本哈希 | 通过 | 0 | — |

**完整回归的 23 项错误与改名前完全相同，无新增错误。** 15 项 GPU 跳过仍是 5 项数值、2 项原生、8 项故障测试；未执行真实 CUDA/NCCL、GIL、overlap 或性能验收。新增兼容回归覆盖 CLI / checkpoint / runtime 规范化、新旧超时变量优先级、原生能力属性、旧 manifest / 报告、双模式 CSV 与 events 的一致选择。新 SSH 验收入口在无 GPU 时分别拒绝 5 skip / 2 skip，均以退出码 1 结束。

证据：[results.json](validation/micro-pipeline-rename-20261015/results.json)、[定向日志](validation/micro-pipeline-rename-20261015/focused-unittest.log)、[完整日志](validation/micro-pipeline-rename-20261015/full-unittest.log)。当前实验简介见 [micro_pipeline_experiment_brief.md](micro_pipeline_experiment_brief.md)。

## 历史修复复验

**本轮 3 项 P2 已修复。** 新记录保存到独立目录，首轮原始日志保持不变。

| 问题 | 修复后的行为 |
| --- | --- |
| 未知 overlap 被稀释为零 | 仅完整目标 batch 进入均值，输出完整 / 未知 / 无配对数量与覆盖率；全未知为 null，实测零仍为零 |
| 非 main / comm 线程被误标 autograd | 明确反向节点标签才能认证 autograd；SSD / watchdog 无证据时进入独立 unknown 子行，保留真实 tid |
| 原生 CUDA 命令变量错误 | 改用 TIDE_RUN_NATIVE_CUDA；双卡、两项执行、零跳过才算验收 |

| 检查 | 通过 | 错误 | 跳过 |
| --- | ---: | ---: | ---: |
| Micro Pipeline 定向回归，139 项 | **124** | 0 | 15 |
| Git 内分析快照回归，27 项 | **27** | 0 | 0 |
| 完整 unittest，376 项 | 315 | 23 | 38 |
| compileall / 共享脚本 py_compile / diff 检查 | 通过 | 0 | — |

**23 项错误测试集合与首轮及干净基线相同，无新增错误。** 五个共享脚本与 Git 快照的 SHA-256 一致；新增 14 项行为测试覆盖未知汇总、覆盖率、旧 schema、线程证据和 SVG 四个独立 CPU 子行。CUDA / NCCL 实机验收仍未执行。

证据：[results.json](validation/pc1-20261015-review-fixes/results.json)、[定向日志](validation/pc1-20261015-review-fixes/focused-unittest.log)、[快照分析日志](validation/pc1-20261015-review-fixes/bundled-analysis-unittest.log)、[完整日志](validation/pc1-20261015-review-fixes/full-unittest.log)。

## 首轮结果

以下为 `7e9317d` 的首轮记录；保留原始证据路径。

| 检查 | 通过 | 错误 | 跳过 | 结论 |
| --- | ---: | ---: | ---: | --- |
| Micro Pipeline 定向回归，125 项 | 110 | 0 | 15 | **本地通过**；跳过数值 CUDA 5 项、原生 CUDA 2 项、故障 8 项 |
| 完整 unittest，362 项 | 301 | 23 | 38 | **未全通过**；无断言失败，23 项错误与基线完全相同 |
| 干净基线 unittest，291 项 | 242 | 23 | 26 | 同一解释器、临时原始 checkout；已有依赖缺失和测试 stub 错误 |
| Git 内分析快照回归 | 13 | 0 | 0 | 独立设置脚本目录后通过 |
| 现代 gsplat 构建适配器 | 7 | 0 | 0 | 仅适配器行为，**未编译扩展** |
| compileall / 共享脚本 py_compile | — | 0 | — | 通过 |
| analyzer self-test / 构建输入检查 | — | 0 | — | 通过；输入检查不等于 CUDA 编译 |
| diff 检查 / 快照 SHA-256 / SVG 结构 | — | 0 | — | 五个脚本与共享目录相同；三类 CPU 子行位于 CPU 轨内 |

所有数值与原始日志见 [results.json](validation/pc1-20261015/results.json)、[定向日志](validation/pc1-20261015/focused-unittest.log)、[完整日志](validation/pc1-20261015/full-unittest.log)、[干净基线日志](validation/pc1-20261015/baseline-unittest.log)。**新增错误测试集合为空**；未通过安装依赖或修改无关模块掩盖基线错误。

## 行为覆盖

| 类别 | 本地已检查 |
| --- | --- |
| 调度 | 固定 C1/C2 FIFO、1–4 microbatch、尾窗口、K 扩展、按顺序 G、绝对 deadline |
| 所有权 | 主线程管理图和 slot；Future 与 GPU event 分离；先加载能力再启动一个常驻 worker |
| 生命周期 | slot 未 G 不能复用；跨 stream 保活；发布 / abort 竞态；提交中途失败保留 buffer；健康关闭顺序及有界 watchdog close |
| 分段 transport | 显式列布局、零行 / 零分片、反向 splits、无额外缩放、M 不写参数梯度 |
| 原生能力 | 缺 ABI、错误 binding、缺 GIL 操作或当前 stream 能力时明确失败 |
| 故障证据 | 每 rank 自身 fatal / 实际退出码验证器；G 入口 pending marker 可达性，**故障场景未实际在 GPU 执行** |
| 指标 | train_ms 区间并集及 deferred writer 保留；延迟与真实 kernel overlap 分开 |
| 分析 | 跨 main / comm / autograd 关联、最深 scope 去重、同设备异 stream、unknown 审计、缺 phase / kernel 为 null、旧 schema、slot 回绕 |
| 配对工具 | 交替顺序、新目录、共同输入指纹、外部超时、预热 / 尾部排除、精确 rank / metrics source 关联、重复 rank / camera iteration 拒绝 |

## 待验收

| 验收项 | 状态 / 下一步 |
| --- | --- |
| 原生构建 | 在 CUDA 环境重新编译 gsplat 与 clm_kernels，核对实际加载路径和能力 |
| 数值 | 双 rank 对比 off、串行分段参考与 micro；覆盖真实 Loss、非默认 stream、全部参数与 Adam 状态 |
| 生命周期 / 故障 | 多 batch 显存、tensor、线程增长检查；8 个故障场景各 rank 自行以 86 退出 |
| GIL | 真实 native 前后向期间，另一 Python 线程的标记与 CUDA 提交证据 |
| 实际 overlap | 代表负载逐项报告四类目标真实 kernel 交集，缺证据 / 零交集分别明确标注 |
| 性能 | 至少两轮交替配对；完整 tide.batch.step 周期、吞吐、显存、CPU 等待及独立 overlap 报告 |

**未启动新的训练、Slurm 或长跑作业；没有 GPU 性能数字或采集报告。**

## 复现

线程、数学语义、构建、CUDA smoke、故障、GIL / Nsight 与配对命令见 [micro_pipeline_timeline.md](micro_pipeline_timeline.md)。历史测试模块清单保存在原始 results.json；复现当前实现时将模块名前缀 `test_pc1_` 替换为 `test_micro_pipeline_`，或使用本次改名验证记录中的模块清单。

```bash
cd /path/to/TideGS
python3 -m compileall -q strategies/tide_engine submodules/gsplat/gsplat utils tests tools
python3 -m unittest discover -s tests
TIDE_TIMELINE_SCRIPTS="$PWD/tools/micro_pipeline_timeline" python3 -m unittest tests.test_micro_pipeline_overlap_analysis tests.test_micro_pipeline_submission_order_summary tests.test_micro_pipeline_timeline_threads -v
python3 tools/micro_pipeline_timeline/analyze_micro_pipeline_overlap.py --self-test
git diff --check
```

## 交付索引

| 交付物 | 位置 |
| --- | --- |
| 本次改名证据 | [micro-pipeline-rename-20261015](validation/micro-pipeline-rename-20261015/results.json) |
| 实验简介 | [micro_pipeline_experiment_brief.md](micro_pipeline_experiment_brief.md) |
| 设计与复现 | [micro_pipeline_timeline.md](micro_pipeline_timeline.md) |
| 测试证据及错误对照 | [validation/pc1-20261015](validation/pc1-20261015/results.json) |
| 三项 P2 修复复验 | [validation/pc1-20261015-review-fixes](validation/pc1-20261015-review-fixes/results.json) |
| 可移植共享分析脚本 | [tools/micro_pipeline_timeline](../tools/micro_pipeline_timeline/README.md)，附源位置和 SHA-256 |
| 数值外部超时入口 | `tools/run_micro_pipeline_cuda_smoke.py` |
| 原生 GIL 探针 | `tools/probe_micro_pipeline_native_gil.py` |
| 交替配对工具 | `tools/run_micro_pipeline_paired_benchmark.py` |

提交按原生能力、分段 transport、训练调度与生命周期、验收工具与文档分组。历史实验脚本、原始报告及派生结果未修改。
