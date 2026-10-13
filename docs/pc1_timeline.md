# PC1 timeline 与异常退出

启用 `--tide_forward_pipeline pc1 --tide_detailed_metrics` 后，原有每个 rank 的
`timeline_events_rank*.jsonl` 会记录以下阶段；NVTX 同时用于 Nsight Systems。
无需修改训练算法或额外调用全局 CUDA synchronize。

## 阶段口径

| stage / 名称 | 含义 | 粒度 |
| --- | --- | --- |
| Projection | 投影、SH 等前向准备 | 每个 microbatch |
| C1:Wait | 通信 stream 等待 projection | 每个 microbatch |
| C1:Metadata | 交换通信分片大小 | 每个 microbatch |
| C1:Payload | 将投影结果分发到相机 owner | 每个 microbatch |
| Render | tile 相交、排序及光栅化 | 每个 microbatch |
| Loss | 前向 loss 计算 | 每个 microbatch |
| IntermediateGrad | 相机 owner 上 loss/render 及接收端解包的反向计算 | 每个真实 autograd 节点 |
| C2 | 反向 all-to-all，将中间梯度送回 Gaussian owner | 每个真实通信 autograd 节点 |
| FinalGrad | Gaussian owner 上投影、SH、参数变换及发送端打包的反向计算 | 每个真实 autograd 节点 |
| AccumulateGrad | 叶子参数的梯度累加节点 | 每个真实 autograd 节点 |
| Optimizer | 所有 microbatch 累积后更新参数 | 每个 batch 一次 |

`AccumulateGrad` 不包含 autograd 内部 InputBuffer 的所有分支加法。
`IntermediateGrad / FinalGrad` 覆盖节点内的 CUDA 工作区间，也会包含节点内部等待；
它们不是 Nsight 导出的“纯 kernel 占用时间”。`C2` 含 caller stream 等待通信结果可用的时间。
若要细分 NCCL kernel 的实际执行/通信等待，仍应查看 `.nsys-rep`。

## 记录方式

- 名称为 `Tide:S1:MB<i>:<stage>`；反向节点另存 `node_name`、`stream_id`、`node_index`。
- 反向事件由真实 autograd 节点 pre/post hooks 记录，使用节点执行 stream 的 CUDA events。
  不替换梯度、不新增反向传播，也不假定一个 microbatch 只有一次 C2。
- `Render` 使用 compute stream 等待 C1 后的独立 `render_start`。
  C1 提前结束但 `P(i+1)` 尚未完成的时间不会算进 Render。
- PC1 JSONL 不再额外画覆盖整段的 `Backward`，避免与细粒度事件重复；
  `backward_ms` 仍保留整段时长，Nsight 中的 Backward 外层 NVTX 仅作为父范围。
- `timing_source=cuda_event_relative_to_batch_anchor`：同一 GPU 内的相对位置来自 CUDA events。
  host 时间只用作锚点，跨 rank/GPU 与 CPU 的精确对齐请以 Nsight 的原始时间轴为准。
- CUDA hook 细分只在 detailed metrics 开启时采集；off 模式保持原同步路径。

## 统计

新增 `intermediate_grad_ms`、`c2_ms`、`final_grad_ms`、`grad_accumulate_ms`，
分别对该 rank、该 batch 同阶段的节点区间取并集；不同阶段若重叠，不能直接相加当作 batch 耗时。
PC1 的 `train_ms` 对 forward/loss/backward/optimizer（以及启用时的 curvature）区间取并集，
避免把重叠的 forward 请求与 backward 重复相加。`gsplat_forward_ms` 保留逐 microbatch
的端到端 forward 延迟之和，不代表纯前向 kernel 时间。完整 batch 耗时仍看 `batch_total_ms`。

## 异常退出

Gloo checkpoint 之外增加独立 CPU TCPStore watchdog。每个 PC1 阶段都带有 deadline；
某个 rank 报错、控制通道断开或阶段超时，会让 worker 以退出码 86 终止。
完整 microbatch 也有同一 deadline，覆盖图像打包、显存查询、event 记录及检查点；
嵌套的子阶段不能延长它。首次编译或单个 microbatch 确实很慢时，应提前调大超时。
这是失败训练进程的退出机制，不会尝试恢复损坏的 NCCL communicator。
`abort()` 保留可能仍在使用的张量，并禁止再次 submit、drain 或 close。

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| TIDE_PC1_STAGE_TIMEOUT_SEC | 60 | 单个受保护 host 阶段的超时秒数 |
| TIDE_PC1_CONTROL_IO_TIMEOUT_SEC | 1 | 独立控制通道 I/O 超时秒数 |
| TIDE_PC1_ERROR_TIMEOUT_SEC | 30 | 控制 Store 的创建/连接超时秒数 |

Gloo 运行超时取 `max(ERROR_TIMEOUT_SEC, STAGE_TIMEOUT_SEC + 2 × CONTROL_IO_TIMEOUT_SEC + 5)`，
默认 67 秒，避免健康 peer 尚在阶段内时被 30 秒 checkpoint 超时误杀。
fatal 只保留 CPU 错误文本，独立 daemon 尽力写日志，最多等待 0.05 秒；日志堵塞时直接退出。

控制端口由 rank 0 动态分配，通过 Gloo 广播；多节点时该端口必须能从其他 rank 访问。
健康退出先停止各 watchdog，再释放控制 Store。正常训练不增加 NCCL collective 或 CUDA 同步。

fatal 路径只尽力保留 CPU 错误文本，不再查询 CUDA 显存或补写 OOM memory point，
避免诊断动作再次阻塞。已有正常 memory points 保留。

## 待 GPU 验证

数值测试对比 off、PC1 trace 开/关的图像、loss 和参数梯度，使用 4 个 microbatch、
2 个 buffer slot，并检查独立 Render 起点和反向各阶段事件。

```bash
TIDE_RUN_CUDA_SMOKE=1 torchrun --standalone --nproc_per_node=2 \
  -m unittest tests.test_distributed_pipeline_cuda -v
```

故障测试单独启动两个 disposable worker，父进程不要再套 torchrun。覆盖一个 rank
在 C1 前及 payload 中途失败，另一个 rank 在真实 CUDA/NCCL 等待中自行退出。
两 rank 都必须留下自身控制器的退出证据，不能靠 launcher 杀掉 peer 算通过。

```bash
TIDE_RUN_CUDA_ABORT_TEST=1 python -m unittest \
  tests.test_distributed_pipeline_cuda_faults -v
```

后续可在数值测试外层加 `nsys profile --trace=cuda,nvtx,osrt` 观察实际 overlap。
CPU 测试与事件存在检查不能替代这个硬件验证；微小测试数据也不保证可观的重叠时长。
