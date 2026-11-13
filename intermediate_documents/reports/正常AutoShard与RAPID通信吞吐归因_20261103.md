# 正常 TorchRec AutoShard 与 RAPID：通信量、关键路径和吞吐归因

## 先对齐两个实验

`中文Response.md` R2 的 Terabyte **RAPID 45.468 GiB、TorchRec 111.127 GiB**，是 8 rank、本地 batch 8,192、5 epoch 的 CPU 回放 **embedding 协议总量估算**，没有包含 dense 梯度、ID/metadata、通信重叠、计算或启动时间。折合每卡每步约 76.28 / 186.44 MB；这是 R2 自身的口径。当前多日 Criteo 子集在 8,192 batch 的 RAPID 旧计划编译 OOM，尚无有效的 RAPID 对 TorchRec GPU 吞吐配对，不能用 R2 字节数解释该配置的 GPU 排名。当前该 batch 的 321,611 samples/s 是**原生 HugeCTR 全 Unique**，不是 RAPID。

有完整配对吞吐和 GPU 时间线的是另一组 **本地 batch 2,048**：8×V100 32 GiB、26 表、dim 128、FP32、SGD 0.01、零 embedding 初始化、同一份全局 shuffle 的多日 Criteo 子集，510 步无评估，输入在 `/dev/shm`。两边都设 `NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`；NCCL channel 核验为 SHM/direct/direct，未观察到 P2P/NVLink channel。本报告只用这组同 batch 的 GPU 数据解释“RAPID 通信更少却较慢”。

## 有效 GPU 对照

| 配置 | 完整 510 步训练循环墙钟 | 吞吐 | 前 30 步远端 NCCL send | AllReduce 输入 |
| --- | ---: | ---: | ---: | ---: |
| RAPID+HugeCTR，26 MP，在线映射 | 28.118 s | 297,167 samples/s | 26.56 MB/卡/步 | 64.18 MB/卡/步 |
| RAPID+HugeCTR，同一放置的预物化输入 | 28.036 s | 298,036 samples/s | 同在线方案的物化 30 步约 26.56 MB | 64.18 MB/卡/步 |
| TorchRec 正常 AutoShard：13 DP + 13 table-wise | 25.762 / 25.648 s | 324,349 / 325,784 samples/s | **26.69 MB/卡/步** | **71.27 MB/卡/步** |
| TorchRec 约束为 26 table-wise（诊断消融） | 27.842 / 27.467 s | 300,113 / 304,219 samples/s | 50.96 MB/卡/步 | 64.18 MB/卡/步 |

NCCL `send` 是过滤本卡 `peer==rank` 后的单方向 **API 逻辑 payload**，包括部分路由数据，不等于全部 embedding 传输或 PCIe 实际链路字节；AllReduce 栏是输入张量长度，不是 ring 实际传输量。TorchRec 原计划的额外约 7.09 MB AllReduce 输入来自 13 张 DP 小表的梯度同步；公共 dense 梯度两边约 64.18 MB。**RAPID 与正常 AutoShard 的远端 send 是 26.56 对 26.69 MB，差不到 0.5%，而不是 R2 所暗示的这组运行时有 59% 差距。** 单靠这两个数也不能推出完整物理通信量排序。

TorchRec 正常计划比 RAPID 训练循环快约 9%；TorchRec 自身关闭 DP 后，吞吐降到约 300–304k，与 RAPID 约 297–298k 接近。这个同框架消融表明小表 DP 的规划选择对原 TorchRec 性能有明显影响；它同时改变了通信和算子安排，不能只把变化归于减少字节。

## 为什么更多 NCCL 时间仍更快

Nsight Systems 在训练区间采 30 步，逐卡计算 GPU kernel 时间区间并集；带 profiler 的墙钟受扰动，吞吐仍以上述无 profiler 的 510 步为准。

| 每卡每步 GPU 时间 | RAPID+HugeCTR | TorchRec 正常 AutoShard |
| --- | ---: | ---: |
| NCCL kernel 区间并集 | 33.335 ms | **35.390 ms** |
| 非 NCCL kernel 区间并集 | 20.775 ms | **23.866 ms** |
| 两类 kernel 重叠 | 0.000 ms | **11.395 ms** |
| GPU kernel 忙碌区间并集 | 54.111 ms | **47.861 ms** |
| 无 profiler 完整训练循环平均每步 | 55.134 ms | 50.513 / 50.291 ms |

因此关键路径近似遵循 `通信时间 + 其他计算时间 - 重叠时间`：RAPID 为 54.11 ms，TorchRec 为 47.86 ms。TorchRec 的 NCCL 和其他 kernel **各自都更长**，但有约 11.4 ms 的并行；RAPID 这轮没有观察到并行。Nsight 中 RAPID 每卡仅一条 GPU kernel stream；TorchRec 有两条，一条承载 NCCL、一条承载其他 kernel。GPU 忙碌并集约差 6.25 ms/步，与无 profiler 墙钟差约 4.6–4.8 ms/步量级相近。它定位了关键路径差异，但不能证明仅增加一条 stream 就能得到同样收益，因为两边通信协议、表放置和算子实现仍不同。

H2D memcpy 时间累计约 RAPID 0.34、TorchRec 0.56 ms/卡/步，远小于两者每步差距；两边 510 步训练区间 `read_bytes=0`。同一 TorchRec AutoShard 计划下，本次额外把 64 个循环 batch 预先放入 GPU 后，完整循环为 **323,774 samples/s**，与锁页预取的 324,349 / 325,784 接近且略低；导出的计划仍是 13 DP + 13 table-wise。这支持 TorchRec 输入整理和 H2D 不是主要瓶颈。RAPID 在线映射与预物化分别为 297,167 / 298,036 samples/s，也不支持逐步在线映射是主要稳态瓶颈。RAPID 在线输入设置另花约 73.53 秒，预物化输入设置约 7.19 秒；若把启动或离线规划也计入短作业端到端时间，须另报，不能与训练循环吞吐混用。

已有一轮 RAPID 打开 HugeCTR 内置 `train_intra_iteration_overlap` 后 `FIT_SECONDS=27.984`，原在线运行 28.118 秒，约快 0.5%；该开关没有消除正常 AutoShard 的吞吐差距。此消融只做一轮，不能把微小变化当成稳定加速。

## 结论与边界

通信量确实影响吞吐：原生 HugeCTR 26 MP 的远端 send 为 154.07 MB/卡/步，RAPID 通过放置、调度和压缩降到 26.56 MB，训练吞吐也从约 210k 提高到约 297–298k。但跨框架比较要看**同配置、全部通信操作在关键路径上的时间**。当前有效 2,048 GPU 配对里，RAPID 与 TorchRec AutoShard 的远端 send 已接近；TorchRec 通过 13 张小表 DP 与通信/计算重叠取得更短关键路径。R2 的 8,192 CPU 总量减少是真实的协议估算，尚没有同 batch 的 RAPID GPU 吞吐来检验其端到端收益。

原始材料：`/tmp/rapid_criteo_multiday_20261103/hugectr_rapid_ram_510_noeval.log`、`hugectr_materialized_ram_510_noeval.log`、`hugectr_rapid_overlap_510_noeval.log`、`torchrec_ram_510_noeval_wall.json`、`torchrec_dp_510_repeat.json`、`torchrec_mp_510_audit.json`、`torchrec_mp_510_repeat.json`、`torchrec_auto_gpu_input_510.json`、`nsys_hugectr_fit_30.sqlite`、`nsys_torchrec_fit_30.sqlite` 及 `*_30_nccl_remote.log`。没有读取或写入新数据集。
