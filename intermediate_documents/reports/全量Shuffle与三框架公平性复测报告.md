# 全范围 Shuffle 与内存驻留输入复测

与 `中文Response.md` 表 R2 同口径的 128 维、5 epoch、全 rank 前反向通信对照见[通信量与Response_R2复核_20261102.md](通信量与Response_R2复核_20261102.md)。下文每卡每批的前向通信表不能直接与 R2 的 GiB 总量比较。

日期：2026-11-02。使用 8 张 V100-SXM2 32 GiB；`NCCL_P2P_DISABLE=1`，NCCL 通过主机共享内存路径通信。本文的“吞吐”是预热 30 步后测量 80 步、三次独立运行的稳态训练吞吐，不含预处理、CPS/ACP 规划、模型构建和输入初始化。Criteo 两组为 DCNv2 风格、26 张 multi-hot 表、32 维 embedding；Taobao 为 DeepFM 风格、4 张 one-hot 表。全局 batch 65,536，本地 batch 8,192，输入含 16 个不同的全局 batch，训练时循环这 16 批。

## 数据范围与预处理

| 数据 | 可用原始范围 | 本次输入 | 抽样及顺序 | SHA256 |
| --- | ---: | ---: | --- | --- |
| Kaggle | 45,840,617 行 | 1,048,576 行 | seed 2026 从完整原始文件均匀无放回抽样，再打乱 | `0c971ffa05d48c08aed81e17dbf395c725f11191b967d639b8b19e1bc1292aec` |
| Terabyte | 已重新下载的 8 个 Parquet 分片，共 6,884,005 行 | 1,048,576 行 | seed 42 从这 8 片全部行均匀无放回抽样，再打乱 | `5262120cfd5fd5c104481cf9570e672464ddc9ffd9418c5d168bfd0f4ca71eff` |
| Taobao | 合并编码后 26,557,961 行 | 完整文件打乱后取前 1,048,576 行 | seed 42 先打乱全量，再提取 4 表 | 子集 `1c4cb17ca5be49b1374738d59e369915f2df92d22ee6d28490ebf2af9d59bdec` |

Kaggle 新文件在数据盘 `fresh_datasets/criteo_kaggle_uniform_1048576_seed2026.bin`，Terabyte 在 `fresh_datasets/criteo_terabyte_uniform_1048576_seed42.bin`。Terabyte 8 片分别抽中 232,916 / 116,810 / 116,409 / 116,802 / 116,120 / 116,611 / 116,205 / 116,703 行，抽中原始行索引最小 0、最大 6,883,995。对打乱后的 10 个输出位置，逐条回查 7 个分片的标签和表 20 第一原始 ID，均一致。Taobao 全量 15 表文件与 4 表提取文件的 26,557,961 行保留字段逐行相同；本次 1,048,576 行与旧 RAM 公平性实验输入的 SHA256 相同，因此 Taobao 的三轮 GPU 日志沿用该次测试，未重复训练。

默认的 Criteo TSV 预处理现在先扫描全部源数据，`--max_rows` 从完整源数据抽样，然后打乱输出；Parquet 转换同样支持全范围的 `--max-rows`。Taobao 本来就全量打乱。转换、提取、离线计划、tmpfs 拷贝及归档均在写入数据前检查目标容量；大数据和中间文件写入 `/tmp`、`/dev/shm` 或数据盘，代码和本报告留在 home。小型端到端测试验证 TSV 的 100 行输入抽取 10 行时确实覆盖后半段、固定种子顺序可复现。

## 通信量与 ACP

ACP 在训练开始前按完整 16 个 batch 的逐表均值静态选择，不在每个 batch 动态切换。下表的 Unit 是每卡每全局 batch 的**前向远端 FP32 embedding payload**；HugeCTR/RAPID 统计远端 Unique 向量与 Reduction 固定量的较小者，GMP 统计其根节点及镜像放置下的远端唯一 embedding 请求，TorchRec 根据当前训练规划器的 sharding 方式推算逻辑向量传输。它们是 trace/协议模型，不是 GPU 链路字节计数；不含 ID、offset、dense 梯度、反向传播以及 NCCL 实现开销。Criteo 为 32 维、Taobao HugeCTR/RAPID/GMP 为 128 维；Taobao TorchRec 实际为 132 维，表中使用其实际维度。

| 数据 | HugeCTR hash+ACP 远端向量/卡/批 | RAPID 最终计划+ACP | GMP 远端请求 | TorchRec 前向协议 MiB/卡/批 |
| --- | ---: | ---: | ---: | ---: |
| Kaggle | 136,272.23 | 108,953.50 | 77,918.91 | 70.00 |
| Terabyte | 253,175.30 | 72,347.94 | 26,572.09 | 105.00 |
| Taobao | 24,535.03 | 5,829.11 | 2,582.66 | 115.50 |

| 数据 | hash ACP | 初始 CPS 后 ACP | 最终额外优化计划 ACP | 关键表 Unique 远端量，hash / 初始 CPS / 最终 |
| --- | --- | --- | --- | --- |
| Kaggle | 25 Unique，表 20 Reduction | 25 Unique，表 20 Reduction | 25 Unique，表 20 Reduction | 表 20：393,317.52 / 77,350.93 / 67,717.95 |
| Terabyte | 24 Unique，表 20、21 Reduction | 26 Unique | 26 Unique | 表 20：249,070.88 / 23,672.16 / 19,679.92；表 21：80,997.50 / 11,590.19 / 4,669.63 |
| Taobao | 4 Unique | 4 Unique | 4 Unique | 四表分别见逐表 JSON |

比较阈值是同一远端口径的 `B × (G−1) = 8,192 × 7 = 57,344` 个向量/卡/批，而不是把远端 Unique 与包含本地项的 `B × G` 比较。Kaggle 表 20 最终 67,717.95，故仍选择 Reduction。这个值与此前用户测得的 68,493.41 接近但并不相同，样本、种子及计划优化过程并非同一实现。初始 CPS 与最终额外 owner/sample 优化计划分别列出，不能把后者称为论文原始 CPS。

全范围抽样后，Kaggle 原始 hash 的表 20 每批远端 Unique 均值仍稳定在 391,563.13–395,719.50；初始 CPS 后的逐批值却从 35,370.38 升到 111,304。这说明该斜坡主要来自顺序处理 batch 的 CPS 规划状态，而不是原始文件前缀的数据漂移。本轮离线命令仍未启用论文中的 METIS 子图聚类初始化，因此这些是当前实现的 CPS/ACP 复测，不能宣称完整复现论文原版规划。

最终两组 Criteo 的逐表 `final_optimized_traffic.json` 与导出的 HugeCTR `acp.json` 独立重算结果逐值相同，26 张表的选择零不一致。

按前向 FP32 payload 换算，Kaggle 的 HugeCTR hash+ACP / RAPID / GMP 分别约 16.64 / 13.30 / 9.51 MiB/卡/批；Terabyte 为 30.91 / 8.83 / 3.24；Taobao 为 11.98 / 2.85 / 1.26。GMP 的远端请求计数仅在每个请求实际返回一个向量的前提下对应这些字节。TorchRec 的实际规划含 data-parallel 表，反向还包含梯度同步；其协议模型另行记录反向估算，不能只凭前向表格推断总通信或吞吐。

TorchRec 训练实际规划为 Kaggle 23 张 data-parallel、2 张 table-wise、1 张 row-wise；Terabyte 18 / 7 / 1；Taobao 4 张 table-wise。按该规划的逻辑反向 embedding payload，三组分别约 204.68 / 250.66 / 115.50 MiB/卡/批；这仍是协议估算，data-parallel 梯度同步、实际 NCCL 算法及重叠执行会影响真实链路字节与墙钟时间。ID 和 KJT 长度通信未计入上述 MiB。

最终优化计划逐表的远端 Unique 均值和静态 ACP 选择如下，表编号从 0 开始，`U` 为 Unique，`R` 为 Reduction。Taobao 仅有四表；Kaggle 表 20 虽显示 Unique 候选 67,717.95，实际 ACP 采用 57,344 个向量的 Reduction。

| 表 | Kaggle：Unique 候选 / 选择 | Terabyte：Unique 候选 / 选择 | Taobao：Unique 候选 / 选择 |
| ---: | ---: | ---: | ---: |
| 0 | 350.26 / U | 2,120.88 / U | 3,776.06 / U |
| 1 | 513.64 / U | 2,039.55 / U | 537.62 / U |
| 2 | 2,469.38 / U | 2,067.62 / U | 450.60 / U |
| 3 | 2,980.05 / U | 927.20 / U | 1,064.82 / U |
| 4 | 241.20 / U | 4,958.33 / U | — |
| 5 | 2.41 / U | 2.01 / U | — |
| 6 | 1,878.62 / U | 1,440.16 / U | — |
| 7 | 68.09 / U | 303.79 / U | — |
| 8 | 2.28 / U | 13.27 / U | — |
| 9 | 11,893.50 / U | 1,365.98 / U | — |
| 10 | 4,312.97 / U | 1,625.12 / U | — |
| 11 | 15,728.76 / U | 14,138.23 / U | — |
| 12 | 8.75 / U | 4.38 / U | — |
| 13 | 105.61 / U | 1,171.70 / U | — |
| 14 | 6,573.19 / U | 4,654.30 / U | — |
| 15 | 135.62 / U | 79.10 / U | — |
| 16 | 3.50 / U | 2.62 / U | — |
| 17 | 480.43 / U | 116.12 / U | — |
| 18 | 12.25 / U | 6.12 / U | — |
| 19 | 42.00 / U | 1,540.66 / U | — |
| 20 | 67,717.95 / R | 19,679.92 / U | — |
| 21 | 178.66 / U | 4,669.63 / U | — |
| 22 | 109.84 / U | 5,268.13 / U | — |
| 23 | 3,461.91 / U | 4,116.50 / U | — |
| 24 | 25.06 / U | 20.42 / U | — |
| 25 | 31.50 / U | 16.18 / U | — |

## 稳态吞吐

单位 samples/s。Taobao 沿用相同 SHA256 输入及同一版本框架的三轮 RAM 实验；其他两组均是本轮新输入的新三轮结果。

| 模式 | Kaggle | Terabyte | Taobao |
| --- | ---: | ---: | ---: |
| HugeCTR hash 全 Unique | 1,051,579 | 1,062,032 | 7,133,196 |
| HugeCTR hash 全 Reduction | 599,779 | 606,180 | 不适用 |
| HugeCTR hash+ACP | 1,470,365 | 1,309,099 | 全 Unique，与原生行相同 |
| RAPID+HugeCTR 预物化 | 1,530,994 | 1,472,328 | 12,378,661 |
| RAPID+HugeCTR 在线映射 | 1,540,336 | 1,472,329 | 12,492,978 |
| TorchRec 锁页预取 | 1,216,698 | 1,177,296 | 3,918,016 |
| TorchRec 全部输入预放 GPU（诊断） | 1,222,782 | 1,193,731 | 3,957,228 |
| HET-GMP 普通 host 输入 | 661,833 | 690,174 | 9,862,499 |
| HET-GMP 稀疏 ID 预放 GPU（诊断） | 666,897 | 690,156 | 9,898,528 |

Kaggle 的 RAPID 在线相对 hash+ACP 高约 4.76%，且两者现在都在表 20 走 Reduction；Terabyte RAPID 在线相对 hash+ACP 高约 12.47%。同一输入下原生全 Unique 明显较慢。Taobao 的 HugeCTR 原生为全 Unique，RAPID 两模式均使用 CPS/ACP，因此不是只有 ACP 变化的单因素比较。

三次独立运行的逐轮 samples/s 如下，平均值在上一表；Taobao GMP GPU 行属于同输入的单独输入消融三轮。

| 模式 | Kaggle 第 1 / 2 / 3 轮 | Terabyte 第 1 / 2 / 3 轮 | Taobao 第 1 / 2 / 3 轮 |
| --- | --- | --- | --- |
| HugeCTR hash Unique | 1,054,252 / 1,051,810 / 1,048,676 | 1,054,015 / 1,065,986 / 1,066,095 | 7,113,813 / 7,142,888 / 7,142,888 |
| HugeCTR hash Reduction | 599,414 / 600,355 / 599,567 | 606,336 / 606,332 / 605,872 | — |
| HugeCTR hash ACP | 1,472,689 / 1,472,108 / 1,466,297 | 1,309,147 / 1,310,432 / 1,307,719 | — |
| RAPID 预物化 | 1,529,863 / 1,530,667 / 1,532,451 | 1,470,908 / 1,472,646 / 1,473,429 | 12,080,369 / 12,512,840 / 12,542,775 |
| RAPID 在线 | 1,546,265 / 1,529,522 / 1,545,222 | 1,484,434 / 1,465,544 / 1,467,008 | 12,512,840 / 12,483,048 / 12,483,048 |
| TorchRec 预取 | 1,226,330 / 1,212,183 / 1,211,580 | 1,171,432 / 1,163,089 / 1,197,366 | 3,911,250 / 3,932,924 / 3,909,874 |
| TorchRec GPU 预放 | 1,218,915 / 1,220,562 / 1,228,870 | 1,193,441 / 1,198,676 / 1,189,074 | 3,966,337 / 3,948,999 / 3,956,348 |
| GMP host | 666,601 / 652,830 / 666,067 | 689,816 / 690,825 / 689,880 | 9,890,360 / 9,838,394 / 9,858,744 |
| GMP GPU 稀疏 | 667,908 / 664,935 / 667,849 | 691,509 / 691,162 / 687,798 | 9,860,598 / 9,910,926 / 9,924,058 |

## 输入读取与初始化

所有新 Criteo 训练输入通过 `stage_checked.py` 逐文件校验 SHA256 后进入 `/dev/shm`。TorchRec 使用锁页内存预取，GMP 先把输入加载到 host 内存；HugeCTR 启用缓冲 reader。所有测量窗口的已采集进程/rank `read_bytes` 增量均为零。这只说明**本轮内存驻留样本**没有物理磁盘读取瓶颈，不说明冷盘或大于内存的训练无瓶颈，也不自动排除 CPU 整理与 H2D 拷贝。

另外，在 hash+ACP 的 HugeCTR 上分别运行一次 `SKIP_H2D=1` 诊断：Kaggle 约 1,501,484、Terabyte 约 1,327,214 samples/s，分别比正常三轮均值高 2.12% 和 1.38%。这个开关首批之后重复使用同一 GPU 输入，改变了 batch 序列，因而只能作为 reader/H2D 可节约时间的粗上界，不能把差值严格归因于读取。综合 tmpfs 零物理读取、输入预放对照和这一上界，当前内存驻留设置下输入不是主导瓶颈，但 CPU/H2D 的小幅开销仍存在。

| 数据 | TorchRec 预取相对 GPU 预放吞吐差 | GMP host 相对 GPU 稀疏输入差 | RAPID 预物化与在线稳态差 | HugeCTR 输入初始化：预物化 / 在线 |
| --- | ---: | ---: | ---: | ---: |
| Kaggle | 0.50% | 0.76% | 在线快约 0.61% | 7.3–8.6 / 76.1–87.1 秒 |
| Terabyte | 1.38% | 0.003% | 基本相同 | 7.3–7.5 / 86.0–86.6 秒 |
| Taobao | 1.00% | 0.90% | 在线快约 0.92% | 约 5–6 / 8–10 秒 |

Taobao 的 GMP 输入消融是另一次同输入三轮对照：普通 host 均值 9,810,524、GPU 稀疏 ID 均值 9,898,528 samples/s，因此表中 host 主实验的 9,862,499 不应用来计算该 0.90% 差值。

Kaggle 在线首轮训练步约 42.38 ms，预物化约 42.84 ms；Terabyte 在线与预物化三轮稳态均约 44.51 ms/步。初始化计时包含设备创建、计划解析、reader 准备等，并非纯映射 CPU 时间。短训练需要计入这些启动和离线规划成本，不能用稳态吞吐替代端到端总用时。Kaggle 与 Terabyte 的 RAPID 预物化/在线每轮打印 loss 最大绝对差不超过 `1e-6`；TorchRec Kaggle 两输入模式 80 步完全一致，Terabyte 最大差 `5.96e-8`；GMP 两输入模式三轮各 11 个打印 loss 均一致。

## 局限与证据

Terabyte 只覆盖当前下载的 8 个分片，不是完整原始 Terabyte 全量；模型也不是论文的 Terabyte DLRM。跨框架模型参数、优化器、embedding 分片、初始权重和训练样本在各 rank 的批次顺序未完全配对。TorchRec/GMP 的通信是逻辑 trace/协议估算，未获得新一轮全框架 GPU PCIe/NCCL 实测字节。GMP 镜像和 TorchRec 的 data-parallel 梯度路径使单个前向向量指标无法代表端到端通信成本。Taobao 吞吐来自同一天较早的同输入测试，未在本次脚本中重复执行。

逐表统计在 `/tmp/rapid_shuffle2026_results/{kaggle,terabyte,taobao}/` 的 `hash_traffic.json`、`initial_cps_traffic.json`、`final_optimized_traffic.json`；GMP 请求计数在 `gmp_traffic.json`，TorchRec 训练实际规划及协议估算在 `torch_actual_plan.json` 与 `torchrec_actual_traffic.json`，三轮日志及汇总在同目录。Criteo 两组的实际 TorchRec 分片计划与独立规划脚本逐表、逐 shard 完全一致。最终 RAPID 映射在 `/tmp/rapid_shuffle2026_kaggle_final_plan_fast` 与 `/tmp/rapid_shuffle2026_terabyte_final_plan`，导出时按每个全局 batch 校验了完整 label+dense+sparse 记录一一对应。证据已归档为数据盘 `evidence_20261102/shuffle2026_comparison.tar`，相邻的 `.tar.json` 记录其 SHA256。没有读取 `Rapid/store` 中的文件。
