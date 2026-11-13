# TorchRec R2 与本地 batch 8,192 的 GPU 通信对账

## 结论

`中文Response.md` 的 Criteo Terabyte 行 **111.127 GiB** 是 8 rank、5 epoch、全体 rank 的前后向 **embedding 协议传输量估算**；每 epoch 的 1,048,576 条样本分成 16 个 global batch，因此共 **80 个 global batch**。本轮 GPU 日志中 **61.56 MB/卡/步** 只统计 NCCL 点对点远端 `send` API 的逻辑 payload，未包含 Row-wise 的 AllGather/ReduceScatter、DP embedding 梯度 AllReduce；也包含 R2 排除的部分 ID/路由数据。它不能代表 TorchRec 的全部 embedding 通信。两组输入的活跃 ID 容量略有不同，但均为 18 DP、7 Table-wise、1 Row-wise。

R2 的 111.127 GiB 按全体 rank、80 步折合约 **1,491.52 MB/global batch**，再按 8 卡均摊为 **186.44 MB/卡/步**。R2 的 RAPID **45.468 GiB** 同法折合 **76.28 MB/卡/步**。这只是把总量归一化，不能与单一 NCCL API 的 61.56 MB 直接比较，也不能据此声称本轮 GPU RAPID 的通信量：RAPID 尚未按 8,192 重新完成 CPS/ACP 与有效训练。

## 按 TorchRec 的 18/7/1 计划拆分

R2 使用 `code/Rapid/experiments/baselines/torchrec_count.py` 的 `table_bytes`，按每次跨端传输只计一次。下表由旧 R2 总数及同配置逐表公式分解；DP 行是用旧总数减去固定的 Table-wise 与 Row-wise 行，故与新抽样分片的复算 DP 数值相差约 0.016 GiB。

| embedding 协议部分 | R2 的 5 epoch 全 rank | 折合每卡每步 | GPU NCCL 对应操作 |
| --- | ---: | ---: | --- |
| 7 张 Table-wise | 30.625 GiB | 51.38 MB | 点对点 send/recv 的 pooled 输出及梯度 |
| 表 20 Row-wise | 35.000 GiB | 58.72 MB | ReduceScatter 与 AllGather |
| 18 张 DP | 45.502 GiB | 76.34 MB | embedding 梯度 AllReduce 的 ring 传输估算 |
| 合计 | **111.127 GiB** | **186.44 MB** | 不含 dense 梯度、ID 和 metadata |

Row-wise 不能直接拿计数器打印的 `4.19 MB` 当跨卡传输量。`ncclReduceScatter` 的 `recvcount` 是每 rank **输出分片**的大小，`ncclAllGather` 的 `sendcount` 是每 rank **输入分片**的大小；两者都是 `8,192 × 128 × 4 = 4,194,304 B`。在 8 rank ring 中，每个 rank 为每个集合通信移动另外 7 个分片，因此两次合计每 rank 约 `2 × 7 × 4,194,304 = 58,720,256 B`，全体 rank 约 **469.76 MB/步**。乘以 80 步即 **35.000 GiB**。`table_bytes` 中 Row-wise 的 `p * unit` 正是对全体 rank 计数，没有额外重复乘 8。Nsight 中相应 kernel 名含 `RING_LL`；这里采用 ring 传输公式作口径对照，不把 NCCL API count 冒充物理链路计数器。

当前 GPU 前 30 步的 NCCL API 日志给出远端 `send` **61.56 MB/卡/步**；其中不只有 Table-wise embedding，故比 R2 的 Table-wise **51.38 MB/卡/步**大并不矛盾。TorchRec 的 AllReduce API 输入约 **107.93 MB/卡/步**；扣掉约 **64.18 MB** 公共 dense 梯度，剩余约 **43.75 MB** 是 DP embedding 梯度输入。8 rank ring 的对应传输估算 `2 × 7/8 × 43.75 ≈ 76.56 MB/卡/步`，接近 R2 的 **76.34 MB**。再加上 Row-wise 的约 **58.72 MB**，得到的量级约 `61.56 + 76.56 + 58.72 = 196.84 MB/卡/步`；比 R2 的 **186.44 MB** 多约 5.6%，包含点对点 ID/路由 payload、数据切片差异及采样窗口差异，不能当成逐字节相等的验证。

## 与 HugeCTR 比较时的边界

本轮 GPU 报告的 HugeCTR **487.79 MB/卡/步** 是原生 **26 MP 全 Unique** 的远端 `send`；TorchRec **61.56 MB/卡/步** 是其自动 **18/7/1** 计划的远端 `send`。两者同为 NCCL 点对点 API 口径，可解释当前原生 HugeCTR 的通信压力；它们不构成 RAPID 与 TorchRec 的 8,192 对照。HugeCTR 只把表 20 改为 Reduction 后，同口径值降至 **289.98 MB/卡/步**。R2 的 RAPID **45.468 GiB** 来自另一份 CPU 回放及 RAPID 物理放置计划，不能与原生 HugeCTR 的 487.79 MB 对换。

核对文件：`/tmp/rapid_response_recheck2026/terabyte_torchrec_dim128_plan.json`、`/tmp/rapid_criteo_multiday_20261103/torchrec_b8192_plan.json`、`/tmp/rapid_criteo_multiday_20261103/torchrec_b8192_30_nccl_remote.log`、`/tmp/rapid_criteo_multiday_20261103/hugectr_native_b8192_30_nccl_remote.log`。没有读取 `Rapid/store`，也没有写新数据集。
