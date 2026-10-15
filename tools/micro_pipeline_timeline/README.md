# TideGS Micro Pipeline 分析脚本快照

保存项目外层 `experiment/timeline/scripts/` 的五个活动分析脚本。`manifest.json` 记录源位置和 SHA-256；历史实验快照保持原样。

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

`extract_superpod_comparison.py` 保留 94 个 optimizer step、每 rank 16 个 camera 的采集约束。任意短跑的实际 overlap 使用 `analyze_micro_pipeline_overlap.py`。配对性能工具为 `tools/run_micro_pipeline_paired_benchmark.py`，新计划交替运行 `off` / `micro`，也可汇总旧 PC1 manifest。
