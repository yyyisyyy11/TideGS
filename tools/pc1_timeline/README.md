# PC1 分析脚本快照

这里保存项目外层 `experiment/timeline/scripts/` 本轮修改的五个共享脚本，便于 Git 审阅及远端复现。`manifest.json` 记录原始位置和 SHA-256；没有修改历史实验快照。

本机优先使用外层共享脚本；独立 checkout 可直接运行本目录脚本。Overlap 与提交顺序分析入口使用标准库，不要求 CUDA；`extract_superpod_comparison.py` 仍沿用原有 94 个 optimizer step、每 rank 16 个 camera 的采集约束。任意短跑的实际 overlap 使用 `analyze_pc1_overlap.py`。

```bash
python tools/pc1_timeline/analyze_pc1_overlap.py --self-test
TIDE_TIMELINE_SCRIPTS="$PWD/tools/pc1_timeline" python -m unittest tests.test_pc1_overlap_analysis -v
```

修改共享脚本后应同步快照及 manifest，并重新运行分析行为测试。原始 `.nsys-rep`、SQLite 和派生 CSV 应保存到新的实验输出目录。
