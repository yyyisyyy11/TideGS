# TideGS Micro Pipeline 分析脚本快照

保存项目外层 `experiment/timeline/scripts/` 的活动分析脚本快照。`manifest.json` 记录源位置和 SHA-256；历史实验快照保持原样。本分支的 `extract_superpod_comparison.py` 从共享脚本 SHA-256 `479122543d46e7d7fdb545a0726f9e52136ec267ea2d54b1dc851f2508983991` 同步后，加入轻量记录模式的来源标记和空指标处理。

## 运行

本机测试优先使用外层共享脚本；独立 checkout 可直接使用本目录。Overlap 与提交顺序分析仅使用标准库，无需 CUDA。

```bash
python tools/micro_pipeline_timeline/analyze_micro_pipeline_overlap.py --self-test
TIDE_TIMELINE_SCRIPTS="$PWD/tools/micro_pipeline_timeline" python -m unittest tests.test_micro_pipeline_overlap_analysis tests.test_micro_pipeline_submission_order_summary tests.test_micro_pipeline_timeline_threads -v
```

## 命名与兼容

| 项目 | 当前值 | 历史输入兼容 |
|---|---|---|
| 模式 | `micro` / `off` | `pc1` 归一化为 `micro` |
| Schedule | `micro-pipeline-window-v2` | 仍可解析 `pc1-window-v2` tag |
| Overlap 前缀 | `micro-pipeline-overlap` | 默认缺少新前缀时读取 `pc1-overlap` |
| 汇总字段 | `observed_micro_pipeline_pairs`、`assessable_micro_pipeline_pairs` | paired benchmark 读取旧 `*_pc1_pairs` 字段 |
| 历史入口 | `analyze_pc1_*.py` 轻量 wrapper | 提交顺序旧入口保留 `pc1-overlap` 默认前缀 |

**未知 overlap 保留 null；完整目标 batch 才参与均值，并报告覆盖率。无证据的 CPU worker 保持 unknown。**

`extract_superpod_comparison.py` 支持显式 `--expected-steps`，限制为 2/4 rank、每 rank 16 个 camera。轻量记录沿用详细模式的逐批 TSV 与 SSD 请求 JSONL；GPU/CPU/SSD 绘图仍以 Nsight CUPTI/NVTX 原始区间为准。`summary.json`、`windows.json` 和每张图的 metadata 标出 `recording_mode=light`，未采集的逐节点指标保留空值，不当作零。任意短跑的实际 overlap 使用 `analyze_micro_pipeline_overlap.py`；轻量模式没有逐节点 trace，不能用缺失的 trace 推断零 overlap。

若在此仓库运行提取器测试，显式设置 `TIDE_TIMELINE_SCRIPTS` 为本目录；默认测试环境可能优先找到仓库外层的旧共享脚本。配对性能工具为 `tools/run_micro_pipeline_paired_benchmark.py`，可汇总旧 PC1 manifest。
