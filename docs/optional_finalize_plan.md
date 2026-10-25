# 可选 Finalize Plan

**默认刷新最新 active 与相机分配，满足条件时复用 preview 的 resident；加 `--finalize_plan` 恢复原完整 finalize 行为。** 仅作用于 Gaussian-sharded 分布式路径，OFF / Micro 共用。

## 开关

| 参数 | finalize 行为 |
| --- | --- |
| 不传 `--finalize_plan` | 检查 bounds generation，必要时重新 cull；更新 S1/S2 active、相机分配、owner / participation 行数；复用预测 resident、stream-in、evict、recency |
| `--finalize_plan` | 保持旧逻辑：可见 block map 改变时完整规划，否则提交 preview 状态 |

首批没有 preview，始终正常完整规划。单卡路径与 checkpoint 格式保持原样；`args.json` 自动记录开关值。

## 自动回退

**快路径必须使用 active-first 策略，且 preview 与刷新后的 active 均未超全局容量，最新 active 被预测 resident 完整覆盖。** 容量小于零表示不设上限；不是每 rank 容量。

| `plan_resolution` | 含义 |
| --- | --- |
| `initial` | 首批正常规划；旧 payload 缺字段也使用此默认值 |
| `active_refresh` | 更新 active / 相机布局，复用预测 resident |
| `full_requested` | 显式启用完整 finalize；可见性未变时仍沿用旧版跳过重算的逻辑 |
| `fallback_policy` | 非 `topc_strict_active_first` / `topc_balanced_active_first` |
| `fallback_preview_capacity` | preview 的 union active 超容量 |
| `fallback_active_capacity` | 最新 union active 超容量 |
| `fallback_resident_coverage` | 最新 union active 未被预测 resident 完整覆盖 |

回退从未提交 preview 的原 planner 状态执行，复用本次已取得的可见性，不重复 cull。成功快路径只提交一次 preview recency，同时把 planner active 更新到最新值；回退由完整 planner 推进一次 recency。

## 指标与 NVTX

| 字段 / 标签 | 语义 |
| --- | --- |
| `prediction_replanned` | 实际调用完整 planner 为 1，否则为 0 |
| `prediction_missing_blocks` / `prediction_extra_blocks` | 快路径为 `None`，rank / global TSV 留空；完整路径比较 stream-in，未变化的显式完整模式为 0 |
| `prediction_repair_ms` | bounds generation 检查及必要的重新 cull |
| `prediction_exact_plan_ms` | 完整重算耗时；未执行为 0 |
| `plan_ms` | preview 规划 + 本次 active refresh 尝试 + 完整重算耗时 |
| `tide.cpu.plan.active_refresh` | 新增的批级 CPU active refresh 范围，包括回退条件检查 |
| `tide.cpu.plan.finalize` | 实际完整 planner 调用边界 |
| `tide.cpu.plan.finalize.total` / JSONL `finalize_plan` | 保留原外层范围，覆盖 bounds 检查、刷新或回退及原有广播 |

分支内提取器识别 active refresh，并归入 `plan_prefetch`。这些 NVTX 都是 CPU 调用范围；GPU 活动仍以 CUPTI 为准。没有增加 CUDA synchronize、rank barrier 或细粒度事件。

## 本地验证

行为回归覆盖 CLI、四 rank、equal / gaussian_balanced、两种 active-first 策略、S1/S2、尾块、空 rank、四类回退、原状态与 recency、连续多批、payload 兼容、异常传播和 TSV 空值。原有详细 / 轻量日志与四 rank 提取器回归同时运行。

2026-10-25 本地结果：**61 项定向测试通过**。另以 `c18fc2688e5d85a549edbc2f1f130a76fa3f9d6e` 原源码做固定种子 `31675` 对照：5 种策略 × 4 种容量（-1 / 0 / 2 / 6）× 2 种相机分配 × 12 批，480 次完整规划及 480 次显式 finalize 的所有非耗时字段、planner resident / active / recency 一致。样本交替包含 S2，覆盖 bounds 未变、重 cull 后可见性相同与不同的情况。

附加 camera-layout / checkpoint-setup 回归 9 项通过、1 项因缺 PyTorch 跳过；resident-policy / checkpoint-compat 两个附加模块也因缺 PyTorch 无法导入，未计入上述通过数。

本机无 PyTorch / CUDA：训练入口通过 AST 加载真实 CPU helper；日志写盘测试仅替换 NVTX 上下文，执行实际 writer。**未提交 GPU 作业，尚无训练数值或性能收益的实机结论。** 旧报告的规划耗时不能直接视作此次节省值。
