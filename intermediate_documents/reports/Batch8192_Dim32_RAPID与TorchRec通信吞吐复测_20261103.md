# 本地 batch 8,192、dim 32：RAPID 与正常 TorchRec AutoShard 复测

## 配对配置与范围

8×V100 32 GiB、每卡 batch 8,192、全局 batch 65,536、26 张 multi-hot 表、embedding dim 32、DCNv2 风格模型、FP32、SGD lr=0.01、embedding 零初始化。HugeCTR 的 DCNv2 bottom MLP 固定输出 128 维，与 TorchRec 保持一致；本次只把 embedding 降到 32 维。Kaggle 与 Terabyte 各使用已从完整来源抽样并全局 shuffle 的 1,048,576 行子集，每组循环 16 个全局 batch，训练 510 步，不做评估。数据沿用 `/dev/shm/rapid_shuffle2026/{kaggle,terabyte}`；RAPID 复用对应的 **8,192 batch** CPS/ACP 计划及预物化输入，没有读取 `Rapid/store` 或新建数据集。计划清单均记录 `full_record_audit=true`。

TorchRec 使用**正常 AutoShard**，不带 `--row-wise-only` 或 `--no-data-parallel`；RAPID+HugeCTR 按各数据集静态 ACP：Kaggle 25 Unique + 表 20 Reduction，Terabyte 26 Unique。两框架设置 `NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`；此前同机 NCCL channel 核查为 SHM/direct/direct，不走 GPU P2P/NVLink。没有本轮逐参数对齐 dense 权重与算子，因此只比较同形态训练系统吞吐，不声称严格数值等价或 AUC 相同。

下文将三种口径分开：`fit`/训练循环的真实 GPU 前向、反向、更新墙钟；已有数据和计划就绪后的**进程总墙钟**（含输入设置与编译，不含下载、shuffle、离线 CPS/ACP）；独立 30 步进程的 NCCL API 逻辑 payload。通信追踪进程不用于吞吐排名。

## 510 步端到端训练循环与作业时间

总训练样本出现次数为 `510 × 65,536 = 33,423,360`。吞吐按**完整训练循环墙钟**相除，不采用 HugeCTR 的 `Time(100 iters)` 内部块计时。

| 数据集 | 方法 | 完整训练循环 | 吞吐 | 进程总墙钟 |
| --- | --- | ---: | ---: | ---: |
| Kaggle | RAPID，预物化 | 24.149 s | **1,384,061 samples/s** | 67.102 s |
| Kaggle | RAPID，在线映射 | 24.159 s | 1,383,450 samples/s | 137.030 s |
| Kaggle | TorchRec 正常 AutoShard | 27.931 s | 1,196,646 samples/s | 45.033 s |
| Terabyte | RAPID，预物化 | 25.034 s | **1,335,104 samples/s** | 67.700 s |
| Terabyte | RAPID，在线映射 | 25.070 s | 1,333,217 samples/s | 152.904 s |
| Terabyte | TorchRec 正常 AutoShard | 28.310 s | 1,180,608 samples/s | 45.410 s |

按预物化 RAPID 对比，训练循环吞吐在 Kaggle/Terabyte 分别比 TorchRec 高 **15.66% / 13.09%**。在线与预物化的 RAPID 训练循环接近，五个打印 loss 检查点在各自数据集内完全一致，支持这两种输入对应同一训练序列。两框架的训练窗口 `/proc/self/io` `read_bytes` 均为 0，输入没有实质磁盘读取。TorchRec 预格式化锁页 host batch 并异步 H2D，HugeCTR 使用内存驻留输入和 buffered reader；本轮未单独做 GPU 全驻留输入消融，不能把一切输入开销严格归零。

短作业**进程总耗时**的排序反过来：RAPID 在线输入设置在 Kaggle/Terabyte 分别花 **77.360 / 94.091 s**，HugeCTR 编译分别 **33.263 / 31.361 s**；TorchRec `torchrun` 的外层 shell `time real` 分别 **45.033 / 45.410 s**。即使输入已预物化，HugeCTR 的本轮初始化与编译也使 510 步进程总时长约 67 s；在线映射进一步增加约 70–85 s。上述进程时间没有包括此前离线生成 CPS/ACP 计划的成本，不应称作从原始下载到训练完成的全流程总耗时。

## 30 步 NCCL API 通信量

用 `code/trace_nccl_payload.c` 在训练循环开始前清零，对独立的前 30 步计数；HugeCTR 单进程 8 GPU 的总数除以 `8 × 30`，TorchRec 8 个 rank 求和后同样除以 `8 × 30`。单位为 **十进制 MB/卡/步**。远端点对点过滤了 `peer==rank`；`recv` 与 `send` 同量，不重复相加。

| 数据集 | 方法 | 远端 `send` | AllReduce 输入 | AllGather 输入 | ReduceScatter 输出 |
| --- | --- | ---: | ---: | ---: | ---: |
| Kaggle | RAPID | **29.731** | 23.254 | 0 | 0 |
| Kaggle | TorchRec AutoShard | 10.552 | **33.341** | 1.049 | 1.049 |
| Terabyte | RAPID | **18.972** | 23.254 | 0 | 0 |
| Terabyte | TorchRec AutoShard | **23.024** | **34.163** | 1.049 | 1.049 |

TorchRec 的实际计划：Kaggle **23 DP + 2 table-wise + 1 row-wise**，Terabyte **18 DP + 7 table-wise + 1 row-wise**。它的 AllReduce 输入比 RAPID 多约 10–11 MB/卡/步，主要与 DP 小表同步有关；每组还包含一张 row-wise 表的 AllGather/ReduceScatter。Kaggle 仅看点对点 `send` 会得出 TorchRec 更省的结论，Terabyte 则相反。不同 NCCL 操作的 API `count` 含义不同，四列不能直接求和当物理 PCIe 字节。

若仅作可比的 **8-rank ring 等效传输估算**，使用 `remote_send + 1.75 × AllReduce_input + 7 × (AllGather_input + ReduceScatter_output)`，Kaggle 为 RAPID **70.43**、TorchRec **83.58 MB/卡/步**；Terabyte 为 **59.67 / 97.49 MB/卡/步**。该公式假设相应集合通信采用 ring，忽略协议包头、路由以外的额外搬运和 NCCL 实际算法选择；它**不是链路实测值**。它说明为什么 Kaggle 即使 RAPID 点对点 `send` 更多，合并 DP 梯度与 row-wise 集合通信后仍可能有更少的总体传输量。当前两组 RAPID 较低的估算通信与较高的训练循环吞吐方向一致，但单轮数据不能证明全部速度差均由通信造成。

## 与此前 128 维结果的关系

此前 `RAPID 45.468 GiB < TorchRec 111.127 GiB` 是 **128 维、8,192 batch、5 epoch 的 CPU embedding 协议估算**，不是本次 32 维的 NCCL API 计数。此前有效 GPU 对照是另一组 **128 维、2,048 batch**，该组 RAPID 与 TorchRec 远端 `send` 约 26.56 / 26.69 MB，TorchRec 因表放置与通信计算重叠而更快。本轮改变了 dim、local batch、TorchRec 实际逐表计划及 RAPID ACP，不能把两个吞吐排名或通信数字拼成一条因果链。旧的 32 维报告只测预热后短窗口，且早期曾混用 HugeCTR 内部块计时；本表使用两边完整 510 步墙钟，因此数值不直接沿用旧表。

## 原始记录

- Kaggle：`/tmp/rapid_shuffle2026_results/kaggle/rapid_materialized_b8192_d32_510.log`、`rapid_online_b8192_d32_510.log`、`torchrec_auto_b8192_d32_510.json`、同名 `.log` 与 `_plan.json`；通信追踪同目录 `rapid_materialized_b8192_d32_30_nccl.log`、`torchrec_auto_b8192_d32_30_nccl.log`。
- Terabyte：`/tmp/rapid_shuffle2026_results/terabyte/` 中对应的同名文件。代码为 `code/hugectr_criteo.py` 与 `code/torchrec_criteo.py`。两组 32 维均在 V100 32 GiB 上完成 510 步，无须再降维。
