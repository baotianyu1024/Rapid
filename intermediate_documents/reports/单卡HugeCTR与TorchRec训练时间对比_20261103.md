# 单卡 HugeCTR 与 TorchRec 训练时间对比（2026-11-03）

## 实验口径

- 单张 V100 32 GiB（GPU 0），每卡 batch 8,192，embedding dim 128，26 张表，DCNv2 风格模型，FP32，SGD 学习率 0.01，embedding 零初始化。
- 复用 `/dev/shm/rapid_criteo_multiday_20261103` 中已全局 shuffle 的多日 Criteo 子集：1,048,576 条样本。没有读取 `Rapid/store`，本次没有下载或写入新数据集。510 个训练 step 会循环使用这个固定子集；无验证，不能由本实验推断 AUC。
- TorchRec 使用单卡自动规划（14 张 data parallel、12 张 table-wise）和预格式化、锁页、异步 H2D 输入。HugeCTR 使用原生多热 AsyncDataReader，分别测试 26 表全 Unique、表 20/21 Reduction、26 表全 Reduction。HugeCTR 日志中的 reader `shuffle=OFF` 指训练时不再次打乱；输入文件在预处理阶段已全局 shuffle。
- 以下训练时间是完整 510 步循环的墙钟，不包含数据准备、模型初始化、编译，也不使用 HugeCTR 内部每 100 步计时替代完整循环。吞吐按 510 × 8,192 / 循环秒数计算。单卡不存在跨 GPU 远端通信；`NCCL_P2P_DISABLE=1` 在这里不构成传输路径对照。

## 结果

| 框架与策略 | 510 步训练墙钟 | 平均每步 | 训练吞吐 |
| --- | ---: | ---: | ---: |
| TorchRec AutoShard，第 1 轮 | 37.401 s | 73.336 ms | 111,706 samples/s |
| TorchRec AutoShard，第 2 轮 | 37.851 s | 74.218 ms | 110,377 samples/s |
| HugeCTR，26 表全 Unique | 42.482 s | 83.297 ms | 98,347 samples/s |
| HugeCTR，仅表 20/21 Reduction | 38.381 s | 75.257 ms | 108,853 samples/s |
| HugeCTR，26 表全 Reduction，第 1 轮 | 36.146 s | 70.874 ms | 115,585 samples/s |
| HugeCTR，26 表全 Reduction，第 2 轮 | 36.170 s | 70.921 ms | 115,508 samples/s |

两轮均值：TorchRec 37.626 s、73.777 ms/步；HugeCTR 全 Reduction 36.158 s、70.898 ms/步，即 HugeCTR 的完整训练循环约快 3.9%。全 Unique HugeCTR 比 TorchRec 两轮均值慢约 12.9%。单卡 HugeCTR 三种策略在第 100、200、300、400、500 步打印的 loss 一致；TorchRec 两轮末步 loss 一致。两框架的 dense 参数初值与算子未逐参数配对，不能据此声称跨框架数值严格等价。

## 耗时来源与资源

HugeCTR 全 Reduction 的 `model.compile()` 为 74.830 s，进程总墙钟 112.729 s；训练循环只占其中 36.146 s。因此如果把一次性启动也算进去，它的一次短作业明显比表中的训练循环长。TorchRec 输入加载约 1.377 s、预格式化约 5.090 s，均未计入训练循环；目前没有与 HugeCTR 完全同口径的 TorchRec 进程总时间，不能用上述初始化数值比较两个框架的完整作业总时间。

单卡 128 维都能放下，无需降到 32 维。HugeCTR 运行中 `nvidia-smi` 采样约 22 GiB；TorchRec 报告 `max_memory_allocated=12,743,511,040` 字节（11.87 GiB），两者统计口径不同。HugeCTR 全 Reduction 训练区间 `FIT_READ_BYTES=0`；TorchRec 第 2 轮训练区间 `read_bytes=0`，第 1 轮为 77,824 字节。因数据已驻留内存且训练区间无实质磁盘读取，这次测量没有显示磁盘读取瓶颈；尚未用输入压力实验给出两框架的数据管线吞吐上限。

Nsight Systems 对 HugeCTR 两种极端策略各采前 30 步。GPU kernel 时间累计：全 Unique 81.627 ms/步，全 Reduction 70.073 ms/步，均为单条 stream；带 profiler 的数字仅用于归因，吞吐以无 profiler 的 510 步实验为准。两者 `volta_sgemm` 分别为 56.948 和 57.105 ms/步，几乎相同；Unique 路径的 `one_to_one_atomic_vec4` 为 10.523 ms/步、`partition_and_unique_kernel` 为 1.874 ms/步，Reduction 路径的 `multi_to_one_reduce_vec4` 为 1.252 ms/步。单卡仍出现少量 `ncclDevKernel_SendRecv` 自调用（1.506 对 0.615 ms/步），不代表跨卡流量。各 kernel 时间是分类累计，不能简单相加当作完整步墙钟。证据支持本地 embedding 去重/聚合路径是这两种 HugeCTR 策略差距的重要来源；不能仅凭单卡测试归因多卡的全部差距。

## 与多卡结果的关系

8 卡、每卡 batch 8,192 时，TorchRec AutoShard 两轮约 507,444 / 506,945 samples/s，HugeCTR 全 Unique 321,611、表 20/21 Reduction 两轮约 426,522 / 426,476 samples/s。单卡全 Reduction 略快并不表示 8 卡全 Reduction 最优：8 卡的表放置、通信量及 overlap 都不同。表 20/21 Reduction 是 HugeCTR 原生策略消融，不是已重新完成的 RAPID CPS/ACP 方案；该 batch 的 RAPID+HugeCTR 尚无有效完整吞吐。

## 原始记录

- TorchRec 两轮：`/tmp/rapid_criteo_multiday_20261103/torchrec_single_b8192_d128_510.json`、`/tmp/rapid_criteo_multiday_20261103/torchrec_single_b8192_d128_510_repeat.json`；同名 `.log`。
- HugeCTR：同目录 `hugectr_single_b8192_d128_unique_510.log`、`hugectr_single_b8192_d128_reduction20_21_510.log`、`hugectr_single_b8192_d128_reduction_all_510.log`、`hugectr_single_b8192_d128_reduction_all_510_repeat.log`。
- Nsight：同目录 `nsys_hugectr_single_b8192_d128_unique_30.sqlite`、`nsys_hugectr_single_b8192_d128_reduction_all_30.sqlite` 及同名 `.nsys-rep`、`.log`。
