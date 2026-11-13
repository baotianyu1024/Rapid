# TorchRec 全 Row-wise：端到端吞吐与留出集 AUC

## 实验设计

本轮固定单机 8×V100 32 GiB、每卡 batch 8,192、全局 batch 65,536、embedding dim 64、FP32、SGD `lr=0.001`，关闭 NCCL GPU P2P 和 IB。TorchRec 使用 `--row-wise-only`，planner 逐表验证每张 embedding 表都按行切到 8 rank；不采用此前的 AutoShard 吞吐或通信数值。RAPID 使用对应的静态 CPS/ACP 和预物化输入；原生 HugeCTR 使用全 Unique 作第三个真实 GPU 训练框架。四种模型使用参考示例宽度，代码见 `code/hugectr_criteo.py`、`code/torchrec_criteo.py`。

具体网络：DCNv2 dense bottom 为 `512→256→128`、3 层秩 512 的 cross、top 为 `1024→1024→512→256→1`；DLRM bottom 为 `512→256→64`、点积交互、top 为 `1024→1024→512→256→1`；Wide&Deep 的 deep 分支为 `1024→1024→1` 并加一阶项；DeepFM 的 deep 分支为 `400→400→400→1` 并加一阶与二阶 FM 项。MovieLens NeuMF 使用 64 维 MLP embedding、8 维 GMF embedding、`256→256→128→64` MLP 与最终 1 维输出。三框架的网络宽度按上述配置实现；不同库的初始化和算子细节未做逐参数等价核验。

Criteo Kaggle、Terabyte 和 Taobao 各取已有**全局 shuffle** 的 1,048,576 行子集；前 15 个完整 global batch 用于训练，最后一个 65,536 行 global batch 独立留出作 AUC。RAPID 的 CPS 只在每个 global batch 内重排样本；已确认 Criteo 两份数据的训练首批和留出批，其标签加 dense 字节记录排序后的多重集合与 TorchRec 输入完全相同。真实 GPU 前向、反向和更新各跑 510 步；由于训练集仅 983,040 行，510 步重复访问这些行，**这不是完整数据集的收敛 AUC**。AUC 比较也没有对齐不同框架的每个逻辑 ID 初始权重，不能据此断言 RAPID 与参考实现精度等价或不等价。

留出集的正例数：Terabyte 2,183/65,536，Kaggle 16,940/65,536，Taobao 3,320/65,536。输入位于 `/dev/shm` 或 `/tmp`，没有读取 `Rapid/store`。拆分脚本在每个新数据文件写入前检查目标目录容量并保留 8 GiB 余量。TorchRec 在训练前将批次预格式化、锁页并异步 H2D；HugeCTR 使用内存驻留 RawAsync reader。完整训练循环时间与含模型初始化的进程总时间分列，评估在循环结束后执行，不计入训练吞吐。每组保留各 rank 的 `/proc/self/io` 增量；仅有 `read_bytes=0` 仍不足以证明 H2D 无开销，所以另用 TorchRec GPU 常驻输入做消融。

## 原生 HugeCTR 表大小审计（2026-11-03 补正）

逐表扫描了三个数据集的 `native_train.bin` 和 `native_eval.bin`，对每张表取实际最大 ID 加一。Kaggle 的旧 `native_sizes.json` 有 **13/26** 张表小于实际 ID 范围（表 20：旧上界 18,583,209，实测最大 ID 21,195,836）；Terabyte 有 **16/26** 张表不匹配；Taobao **0/4** 张表不匹配。旧 Kaggle 原生 HugeCTR 的崩溃不能归咎于显存，旧 Terabyte 原生 HugeCTR 的吞吐/AUC 即使生成了日志也视为无效。经过训练集与验证集联合扫描的大小配置见 `/tmp/rapid_rw_e2e_20261103/{kaggle,terabyte,taobao}/native_sizes_verified.json`；原生 HugeCTR 使用新配置重跑，新日志后缀为 `unique_verified` 或 `mixed20_verified`。其余 RAPID/TorchRec 结果不使用这些旧大小配置。

## 输入瓶颈消融

Terabyte/DCNv2、TorchRec 全 26 表 row-wise、dim 64 的 20 步诊断：锁页预取的完整循环为 **5.220 秒**，同一输入预置 GPU 为 **5.367 秒**；两者训练窗口磁盘读取均为 0，留出 AUC 均约 0.47005。短作业有计时噪声，但 GPU 常驻输入没有带来速度提升，当前看不到足以解释系统吞吐差距的读取瓶颈。原始文件为 `/tmp/rapid_shuffle2026_results/terabyte/torchrec_rw_d64_{,gpu_}probe.json`。

新增 Taobao/DCNv2 100 步、10 步 warmup、全 4 表 row-wise 的相同消融：锁页预取循环 **6.2523 秒**，GPU 常驻输入 **6.2972 秒**；训练窗口两个模式的 `read_bytes` 均为 0，显存峰值分别为 0.394/0.398 GiB。原始文件为 `/tmp/rapid_rw_e2e_20261103/taobao/torchrec_rowwise_taobao_{prefetch,gpu}_input_probe.json`。两数据集上 GPU 常驻均未提高吞吐，支持当前 TorchRec 输入预取未成为主要瓶颈；这是有限次数的短窗口消融，不等于对所有模型的完整输入性能证明。

Kaggle/DCNv2 再按相同 100 步与 10 步 warmup 对照：锁页预取 **26.0704 秒**，GPU 常驻 **25.9994 秒**，仅快 **0.27%**；两个模式训练窗口 `read_bytes` 均为 0，峰值已分配显存约 1.421/1.616 GiB。原始 JSON 位于 `/tmp/rapid_rw_e2e_20261103/kaggle/torchrec_rowwise_kaggle_{prefetch,gpu}_input_probe.json`。所以 Criteo 两份数据与 Taobao 的 TorchRec 代表组均未显示显著输入读取瓶颈。

MovieLens-25M 每用户时间划分的 DLRM 100 步、10 步 warmup 对照：锁页预取 **3.9644 秒**，GPU 常驻 **4.0616 秒**；两个模式训练窗口 `read_bytes` 均为 0。原始 JSON 位于 `/tmp/rapid_rw_e2e_20261103/movielens25m_user_temporal/torchrec_rowwise_movielens_{prefetch,gpu}_input_probe.json`。至此四个真实数据集均有 TorchRec 输入路径消融；GPU 常驻均未带来超过 0.3% 的提速。

HugeCTR/Terabyte/DCNv2 原生全 Unique、dim 64、510 步另测 `--inter-iteration-overlap`：默认完整 fit **58.2184 秒**，打开跨迭代重叠 **58.5268 秒**，两次训练窗口 `read_bytes` 均为 0，AUC 分别为 0.471301734/0.471301854。这个开关没有减少训练耗时，说明当前 RawAsync 输入路径未表现出可由跨迭代预取消除的明显等待；单次差异约 0.5%，不能解释框架吞吐差距。

## 结果

本节只填写真实完成的配置；吞吐按 `510×65,536÷完整训练循环秒数` 计算。TorchRec 的 AUC/LogLoss 来自留出的 65,536 行；HugeCTR `AUC` 来自同一批留出行。HugeCTR 本地 `AverageLoss` 指标在后置单批评估中返回 0，不把它当成有效 LogLoss。

| 数据集／模型 | RAPID 吞吐 | RAPID AUC | TorchRec Row-wise 吞吐 | TorchRec AUC | HugeCTR 全 Unique 吞吐 | HugeCTR AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Terabyte／DCNv2 | 796,326 | 0.556838 | 281,600 | 0.515116 | 574,103 | 0.471302 |
| Terabyte／DLRM | 1,331,094 | 0.505952 | 325,247 | 0.498414 | 808,011 | 0.572901 |
| Terabyte／Wide&Deep | 1,296,082 | 0.521411 | 307,242 | 0.617974 | 779,562 | 0.521436 |
| Terabyte／DeepFM | 1,541,795 | 0.517824 | 318,132 | 0.577153 | 860,521 | 0.513538 |
| Kaggle／DCNv2 | 883,210 | 0.522270 | 280,745 | 0.518233 | 569,452 | 0.532189 |
| Kaggle／DLRM | 1,593,360 | 0.564089 | 324,420 | 0.529761 | 800,233 | 0.504997 |
| Kaggle／Wide&Deep | 1,591,193 | 0.496154 | 306,885 | 0.516437 | 773,374 | 0.553144 |
| Kaggle／DeepFM | 1,959,789 | 0.546409 | 317,725 | 0.583613 | 852,505 | 0.583022 |
| Taobao／DCNv2 | 2,973,398 | 0.505542 | 1,255,384 | 0.513935 | 2,722,611 | 0.508391 |
| Taobao／DLRM | 4,645,938 | 0.511270 | 1,489,347 | 0.507030 | 4,072,070 | 0.510062 |
| Taobao／Wide&Deep | 6,437,283 | 0.511615 | 1,605,616 | 0.511920 | 5,356,297 | 0.512417 |
| Taobao／DeepFM | 11,144,003 | 0.511635 | 1,766,859 | 0.487460 | 8,280,524 | 0.512487 |

TorchRec 在 Terabyte 四组的 LogLoss 依次为 0.557995、0.589845、0.151972、0.172130；Kaggle 四组为 0.654591、0.670050、0.589094、0.648206。Wide&Deep 与 DeepFM 的 AUC/Loss 和 RAPID 差异较大；这些模型尚无逻辑 ID 初始化配对，不将差异解释为 RAPID 协议造成的精度变化。两份 Criteo 的 TorchRec 逐模型计划均通过 26/26 表 row-wise 的机器校验。逐运行原始路径、进程时间、训练窗口读字节和 PyTorch 峰值已分配显存收录在 `rowwise_e2e_results_20261103.csv`。

Taobao 四组 TorchRec 计划均验证为 4/4 表 row-wise；其中 DLRM 的训练窗口 `read_bytes` 最多 1,847,296 字节，其余三组为 0。RAPID 和 HugeCTR Unique 的 Taobao DLRM 分别为 0 和 1,310,720 字节；与超过 33.4M 训练样本相比很小，但仍需输入消融进一步确认主机到 GPU 的传输开销。以上 Taobao AUC 均在 510 步短训后接近 0.5，不能视为最终精度。

### 训练进程总耗时

上表吞吐只按完整训练循环计，下面给出同一配置从进程启动到评估结束的墙钟秒数，含框架初始化、TorchRec 输入预格式化、训练和留出评估。RAPID 的离线 CPS/ACP 计划与物化时间**未**计入进程，因此这列仍不是含全部离线预处理的工作流总时长。

| 数据集／模型 | RAPID 进程秒数 | 原生 HugeCTR 进程秒数 | TorchRec 进程秒数 |
| --- | ---: | ---: | ---: |
| Terabyte／DCNv2 | 98.740 | 114.786 | 132.981 |
| Terabyte／DLRM | 48.530 | 64.911 | 116.991 |
| Terabyte／Wide&Deep | 51.731 | 69.604 | 122.621 |
| Terabyte／DeepFM | 39.372 | 56.371 | 119.594 |
| Kaggle／DCNv2 | 94.024 | 114.708 | 133.121 |
| Kaggle／DLRM | 44.634 | 65.304 | 117.547 |
| Kaggle／Wide&Deep | 46.778 | 68.475 | 122.903 |
| Kaggle／DeepFM | 35.083 | 57.842 | 118.610 |
| Taobao／DCNv2 | 40.190 | 39.493 | 37.838 |
| Taobao／DLRM | 26.830 | 27.643 | 34.277 |
| Taobao／Wide&Deep | 22.457 | 21.685 | 32.045 |
| Taobao／DeepFM | 15.987 | 16.692 | 30.209 |

Taobao/DCNv2 展示了区分两个口径的必要性：RAPID 训练循环吞吐高于 TorchRec，但包含启动和单批评估的进程总时间为 40.190 对 37.838 秒。短任务中初始化占比高，不能仅凭训练循环吞吐声称整个流程更快。

按主表相同 510 步训练循环计算，RAPID 在 12 个 Criteo/Taobao 组合上为 TorchRec 全 row-wise 的约 **2.37–6.31 倍**；原生 HugeCTR 全 Unique 为约 **2.03–4.69 倍**。这是强制全表 row-wise 的 TorchRec 结果，不能代替正常 AutoShard 的排名。当前没有对这 12 组逐一采集同口径 NCCL payload 与 kernel 时间，吞吐差不能只归因于通信量。

## MovieLens-25M 独立时间验证

从 GroupLens 官方 `ml-25m.zip` 重新下载 261,978,986 字节，ZIP 全文件 CRC 校验通过，SHA-256 为 `8b21cfb7eb1706b4ec0aac894368d90acf26ebdfb6aced3ebd4ad5bd1eb9c6aa`。以 `rating >= 4` 为正例，按时间将 25,000,095 条记录分为 19,988,480 条训练和 4,980,736 条验证（末尾不足完整 global batch 的 30,879 条不参与），训练集固定种子 42 shuffle，验证集保持时间顺序。原始数据与转换后的数据分别放 `/tmp`、`/dev/shm`，写入前检查目标剩余容量；划分清单见 `/dev/shm/rapid_movielens25m_temporal_20261103/manifest.json`。

| 模型 | HugeCTR 全 Unique 吞吐 | HugeCTR AUC | TorchRec 全 Row-wise 吞吐 | TorchRec AUC | TorchRec LogLoss |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLRM | 4,728,107 | 0.501049 | 2,149,182 | 0.494821 | 0.693788 |
| NeuMF | 17,592,483 | 0.505313 | 2,961,907 | 0.505924 | 0.693246 |

这里每组仍为 510 步、每卡 8,192、dim 64，训练吞吐只计训练循环，AUC 在完整独立验证集上计算。TorchRec 两个计划均逐表验证为 2/2 row-wise。HugeCTR 训练窗口 `read_bytes=0`；TorchRec DLRM 最大为 204,800 字节、NeuMF 为 0。四组 AUC 都接近 0.5，表明固定 `lr=0.001` 的 510 步短训尚未达到有意义的收敛精度。MovieLens 尚无 RAPID 的 CPS/ACP 离线计划与物化输入，不能把 HugeCTR Unique 结果写成 RAPID。

MovieLens DLRM 的 HugeCTR/TorchRec 训练进程总耗时为 26.330/29.761 秒；NeuMF 为 11.451/24.794 秒，均含完整时间验证但不含官方数据下载和预处理。

为检查 510 步是否只是学习率偏小，另用相同全局时间划分、`lr=0.01` 训练 5,000 步：DLRM 的 HugeCTR/TorchRec AUC 为 **0.499318/0.496076**，NeuMF 为 **0.519277/0.511039**；对应训练循环为 67.910/144.724 秒和 17.220/101.983 秒。它们仍未收敛。检查划分后发现验证集中 **4,222,838/4,980,736（84.78%）** 条记录的用户在训练集从未出现，而训练集电影正例率的平滑先验在该验证集的 AUC 为 **0.681722**。因此全局时间截断使大量用户处于冷启动状态；我们另做每用户按时间划分，不把两种划分的 AUC 混为一组。

### 每用户时间划分补充

对每个用户分别按时间将前 80% 放入训练、后 20% 放入验证，再固定种子 shuffle 训练记录并随机裁掉不足完整 global batch 的尾行。最终训练 **19,922,944**、验证 **5,046,272** 行；验证中未见用户 **0** 条、未见电影 **12,093** 条，电影先验 AUC **0.705449**。生成脚本 `code/prepare_movielens25m_temporal.py --per-user`，清单 `/dev/shm/rapid_movielens25m_user_temporal_20261103/manifest.json`。

| 模型 | HugeCTR 全 Unique 吞吐 | HugeCTR AUC | TorchRec 全 Row-wise 吞吐 | TorchRec AUC | TorchRec LogLoss |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLRM | 4,723,588 | 0.500136 | 2,153,213 | 0.492454 | 0.695298 |
| NeuMF | 17,515,914 | 0.500696 | 3,039,412 | 0.500081 | 0.691820 |

这里沿用 `lr=0.001`、SGD、510 步；row-wise 计划均验证为 2/2 表。虽然消除了大规模新用户问题，AUC 仍接近 0.5，表明当前优化配置在这一任务上不充分。两种时间划分的结果不能合并排序。

HugeCTR 官方 NeuMF 示例使用自适应优化器；为核查优化器因素，我们在同一每用户时间划分上改用两框架均支持的 AdaGrad（`lr=0.01`、epsilon `1e-8`；HugeCTR 初始累加器设 0，TorchRec 使用 FBGEMM 默认值），其余 batch、dim、网络宽度和 510 步不变。TorchRec 仍逐表验证 2/2 row-wise：

HugeCTR 本地 EBC 训练路径在 Adam 的 Global/Local 两种更新配置下均于首步中止，未生成可用结果；错误日志为 `/tmp/rapid_rw_e2e_20261103/movielens25m_user_temporal/hugectr_unique_dlrm_b8192_d64_510_adam.log` 与 `hugectr_unique_dlrm_adam_local_10.log`。AdaGrad 已经通过两框架的 10 步探针与完整运行。

| 模型 | HugeCTR 训练秒数 | HugeCTR AUC | HugeCTR 平均损失 | TorchRec 训练秒数 | TorchRec AUC | TorchRec LogLoss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DLRM | 7.095 | 0.811470 | 0.521456 | 15.656 | 0.814306 | 0.525867 |
| NeuMF | 1.918 | 0.807830 | 0.524841 | 11.470 | 0.807836 | 0.531561 |

这表明低 AUC 主要来自原 SGD 配置不足；同一数据下 AdaGrad 在两个框架均达到 AUC 0.8 以上。HugeCTR 这里的 `AverageLoss` 在 77 个验证批后为非零，但其汇总定义与 TorchRec 的逐样本 LogLoss 未做数值等价核对，因此仅作各自训练诊断。NeuMF 两框架 AUC 差约 `0.000006`；DLRM 差约 `0.002836`，随机初值尚未配对，不能据此宣称精度等价。

同一 AdaGrad 配置在 1,000 和 5,000 步后的独立验证如下；每个检查点均从头训练，不是从前一个 checkpoint 接续：

| 模型 | HugeCTR AUC 510→1,000→5,000 | TorchRec AUC 510→1,000→5,000 | 1,000 步训练秒数 HugeCTR/TorchRec | 5,000 步训练秒数 HugeCTR/TorchRec |
| --- | ---: | ---: | ---: | ---: |
| DLRM | 0.811470→0.815669→0.768198 | 0.814306→0.813855→0.763025 | 13.872 / 29.796 | 68.309 / 145.828 |
| NeuMF | 0.807830→0.807591→0.797096 | 0.807836→0.808471→0.799490 | 3.628 / 21.047 | 17.332 / 102.971 |

DLRM 两框架的验证损失从约 0.52 上升到 HugeCTR 0.832、TorchRec 0.869；NeuMF 也轻度退化。该学习率下训练更久并没有提高验证 AUC，应使用早停或学习率日程；510 步结果是已测检查点，不称为最终收敛值。

MovieLens 两种时间划分、SGD/AdaGrad 和 510/1,000/5,000 步共 24 份逐运行结果见 `rowwise_movielens_optimization_20261103.csv`。其中记录了每组学习率、训练时间、AUC、损失指标名称、验证行数、训练窗口读字节及原始日志路径。

## RAPID 与原生 HugeCTR 的 DLRM 配对核对

为分辨默认随机 embedding 初值的影响，在相同 shuffle 数据、global batch 序列、SGD 设置和验证集上，将两边所有逻辑 embedding ID 初始化为零，再分别训练 510 步；原生 HugeCTR 使用经验证的表大小和全 Unique。结果：

| 数据集 | RAPID AUC | 原生 HugeCTR AUC | 绝对差 | RAPID 训练秒数 | 原生训练秒数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Terabyte | 0.532595158 | 0.532595575 | 0.000000417 | 25.092 | 41.345 |
| Kaggle | 0.571184695 | 0.571184397 | 0.000000298 | 21.003 | 41.760 |

逐次日志为 `/tmp/rapid_rw_e2e_20261103/{terabyte,kaggle}/{rapid,native}_dlrm_paired_zero_b8192_d64_510.log`。本配对支持 CPS/ACP 在这两组 DLRM 短训中保持几乎相同的 AUC，也说明默认随机初始化不能作为协议精度差异的证据。零初始化只有单种子、单模型，且没有配对 TorchRec 的逻辑 ID 初值与 dense 权重；不能据此填写 Response R5 所要求的五种随机种子收敛等价区间。

## HET-GMP 容量回退与输入消融

此前 HET-GMP dim 32 的 CUDA ID 越界断言追溯到运行脚本写死旧词表大小：Terabyte 传入 **21,459,313**，而当前分区真实长度为 **21,877,455**；随后的 NCCL shared-memory 报错是设备断言后的连带现象。`code/run_reference_het_dcnv2_b8192.sh` 已改为从分区文件读取词表长度，并在新日志名中标记 `verified`。dim 64、每卡 8,192 的 all-to-all 查询/返回缓冲分配仍 CUDA OOM，因此下面的 dim 32 结果**不与主表 dim 64 吞吐直接比较**。

Terabyte/DCNv2 在 dim 32、每卡 8,192、`store_rate=0.25`、510 步下，去掉前三个 10 步块后，主机输入稳定块平均 **0.947000 秒/10 步**，折算 **692,038 samples/s**；GPU 常驻输入为 **0.944894 秒/10 步**、**693,581 samples/s**，仅快约 **0.22%**。8 个 rank 的训练窗口磁盘 `read_bytes` 均为 0；GPU 输入模式逐 rank 明确预载 224,395,264 字节。这支持该配置的数据读取不是主瓶颈。HET-GMP 本次使用 `--benchmark` 只计算训练损失，不计算留出 AUC，不能填精度栏；逐运行日志和 I/O 快照位于 `/tmp/rapid_shuffle2026_results/terabyte/het_reference_verified_dcnv2_b8192_d32_store025{,_gpu}_510.*`。

Kaggle/DCNv2 同配置的主机输入为 **0.977194 秒/10 步、670,655 samples/s**；GPU 常驻输入为 **0.974385 秒/10 步、672,588 samples/s**，快约 **0.29%**。8 个 rank 的训练窗口 `read_bytes` 同样均为 0；原始日志位于 `/tmp/rapid_shuffle2026_results/kaggle/het_reference_verified_dcnv2_b8192_d32_store025{,_gpu}_510.*`。两数据集的 GPU 常驻输入提升均小于 0.3%，支持已修复配置中的 HET-GMP 输入读取不是主要瓶颈。

## 完成度与边界

| Response 组合 | 本轮真实 GPU 训练与验证 | 未完成项 |
| --- | --- | --- |
| Kaggle/Terabyte/Taobao × 4 模型（12 组） | RAPID、HugeCTR Unique、TorchRec 全 row-wise 各 1 次；另有 Criteo DLRM 零初始化配对 | 完整数据收敛、三次独立运行、统一有效 LogLoss、其他方法 |
| MovieLens-25M × DLRM/NeuMF（2 组） | 官方数据；HugeCTR、TorchRec 全 row-wise，两个时间划分与优化器检查点 | RAPID CPS/ACP 离线计划、其他方法、随机初值配对 |
| Kaggle/Terabyte × DCNv2 HET-GMP | dim 32、镜像比例 0.25 的 510 步吞吐与输入消融 | dim 64 OOM、留出 AUC、与主表同维排名 |
| T180/T200/T510 | 官方生成与通信诊断另见 `官方T系列生成与通信诊断_20261103.md` | 官方全词表在本机 8×32 GiB 上的端到端训练不可容纳 |

Response R1/R5 列出的正式比较还要求每配置三次独立运行、完整数据训练、配对初始化、全量独立验证以及 EMBark、HET-GMP、Herald、BagPipe。当前 Criteo/Taobao 子集短训与 MovieLens 的两种时间划分、优化器补测均不冒充这些正式结果。HET-GMP 在 dim 64、batch 8,192 初始化 all-to-all 查询缓冲时 OOM；dim 32 的修正后吞吐和输入消融已单列，尚无同配置 AUC。官方完整 T180/T200/T510 的词表权重也超过本机 8×32 GiB 容量。MovieLens-25M 已完成官方数据下载及 HugeCTR/TorchRec 两模型实测，但缺少 RAPID 离线计划；EMBark、Herald、BagPipe 仍无统一模型和输入下可复核的 GPU 端到端训练适配。
