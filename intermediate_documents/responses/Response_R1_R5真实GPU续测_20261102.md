# Response R1/R5 真实 GPU 续测（2026-11-02）

> **2026-11-03 计时纠错：** 本文 HugeCTR/RAPID 表格中的吞吐使用 HugeCTR 内部 `Time(10 iters)`，而 TorchRec 使用完整训练步墙钟；内部计时漏掉打印/取 loss 后到下一块训练前的时间，不能据此对两框架排名。按同一日志第 30 至 110 步时间戳重新计算，Terabyte、本地 batch 2048 的 RAPID 为 **298,502 samples/s**，低于 TorchRec **329,287**，原文“高 0.24%”错误；Kaggle 的 RAPID 为 **314,674**，高于 TorchRec **300,618** 约 **4.7%**，原文“高 15.77%”错误。完整逐轮纠错和 GPU 诊断见《多日吞吐排序与计时口径排查_20261103.md》。以下旧表的 HugeCTR 数字仅保留为历史内部计时值。

## 测量范围与口径

使用 8 张 V100-SXM2 32 GiB、单机 FP32，设置 `NCCL_P2P_DISABLE=1`，使 NCCL 不走 GPU P2P/NVLink。Criteo embedding 主维度 128；Taobao DeepFM 主交互维度 128，HugeCTR/TorchRec 的表物理宽度 132（1 维一阶项、128 维交互项、3 维 padding），GMP 的表宽度 129。全局 batch 65,536，每卡 8,192；各输入为全范围 shuffle 后的 1,048,576 条子集，循环 16 个完整 global batch。Kaggle、Terabyte 为 DCNv2 风格，Taobao 为 DeepFM 风格。**它们只对应 response R1 的三个数据集/模型行中的 Kaggle/DCNv2 与 Taobao/DeepFM；Terabyte/DCNv2 当前显存不足。其他 R1 模型尚未实施，不应填数。**

吞吐是每配置三个独立进程/作业，预热 30 步后测 80 步真实 GPU 前向、反向、优化器和当前输入路径的稳态训练步耗时；HugeCTR 按打印的 10 步墙钟块累计第 40–110 步，TorchRec 取各 rank 每步最大值，GMP 取每 10 步跨 rank 最大时间。输入在 `/dev/shm`，HugeCTR 使用 buffered reader 和 RAPID 在线映射，TorchRec 使用锁页预取，GMP 先加载 host 内存。Criteo/Taobao 三框架计时边界仍存在上述实现差异；这些是真实 GPU 训练步结果，不是完整数据下载、离线 CPS/ACP、模型初始化及评估的总耗时。

## 8,192 本地 batch：历史内部计时与墙钟纠错

| 数据集/模型 | RAPID+HugeCTR 三轮内部计时 samples/s | TorchRec 三轮 samples/s | HET-GMP 三轮 samples/s | 结论 |
| --- | --- | --- | --- | --- |
| Kaggle/DCNv2，128 维 | 579,879 / 580,357 / 580,037；均值 **580,091** | 520,850 / 519,522 / 520,367；均值 **520,246** | CUDA OOM，未得到有效吞吐 | RAPID 墙钟复核约 523,974，与 TorchRec 接近；旧内部计时的 11.50% 优势无效 |
| Terabyte/DCNv2，128 维 | 编译 CUDA OOM（在线及预物化均失败） | 512,256 / 510,446 / 511,247；均值 **511,316** | CUDA OOM | 此 batch/显存下不能完成三框架对比 |
| Taobao/DeepFM，128 维主交互 | 13,604,442 / 13,598,126 / 13,597,879；均值 **13,600,149** | 3,889,404 / 3,935,212 / 3,888,543；均值 **3,904,387** | 约 9,825,487 / 9,834,703 / 9,829,171；均值 **9,829,787** | RAPID 墙钟未逐轮复核，暂不作严格跨框架排名 |

旧表按各自原始计时口径换算的平均 step latency：Kaggle RAPID 内部计时 113.0 ms、TorchRec 126.0 ms；Terabyte TorchRec 128.2 ms；Taobao RAPID 内部计时 4.82 ms、GMP 6.67 ms、TorchRec 16.79 ms。Taobao GMP 日志的 `speed` 为每 rank 吞吐，表中乘 8 并以第 40–110 步的跨 rank 最大块时间重新计算。

Kaggle RAPID 在线映射三轮 `INPUT_SETUP_SECONDS` 为 77.59/77.19/76.47 秒，模型编译 79.70/79.53/79.63 秒。按日志创建到最后写入的近似短作业时长，RAPID 为 174/173/172 秒，TorchRec 为 27/28/27 秒；这项近似包括模型启动和本轮 110 步训练，但不包括离线 CPS/ACP 和数据预处理。**短作业的完整墙钟时间 RAPID 更长，Kaggle 8192 的稳态墙钟吞吐与 TorchRec 接近**；必须同时报告，不能用稳态吞吐代替总训练时间。Taobao 同法近似 RAPID 为 17/16/16 秒、TorchRec 为 11/11/11 秒、GMP 为 6/7/6 秒。

## 输入路径诊断

上述 Kaggle RAPID、Taobao RAPID 与 Taobao GMP 的测量区间 `/proc/*/io` `read_bytes` 差值均为 0；TorchRec Kaggle、Terabyte、Taobao 三轮各 rank 的差值也均为 0。TorchRec Kaggle 128 维将 16 个 batch 预先放入 GPU 的单次诊断为 521,071 samples/s，相对锁页预取三轮均值 520,246 只高 0.16%。这支持本次内存驻留输入没有物理磁盘瓶颈，TorchRec 的 H2D/预取也不是主导耗时。`read_bytes=0` 不能单独排除 CPU 整理、在线映射或 H2D 成本；HugeCTR 128 维在线映射的约 77 秒属于初始化成本，已单列。Taobao GMP 的此前 host/GPU 输入消融亦见《全量Shuffle与三框架公平性复测报告.md》。

## 独立留出集精度：当前证据

在 1,048,576 行内将最后一个完整 global batch 的 65,536 条留出，前 15 批训练。Kaggle 原生 HugeCTR 与 RAPID 的留出批标签和 dense 特征校验一致。AUC/LogLoss 均来自 GPU 训练后独立验证批；HugeCTR 的 `AverageLoss` 指标在当前构建中打印为 0，不能将 0 当成有效 LogLoss。GMP 验证汇总了 8 rank 全部预测后计算全局 AUC，训练前缀为保持固定形状仅移动了 361 条样本的 rank，未改变 embedding 根节点与镜像。

| 数据集/模型 | 训练预算 | RAPID AUC | TorchRec AUC / LogLoss | GMP AUC / 验证 loss | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| Kaggle/DCNv2 | 75 步、SGD lr=0.001 | 0.4950 | 0.4486 / 0.6788 | OOM | 原生 HugeCTR AUC 0.4809；均未收敛 |
| Kaggle/DCNv2 | 500 步、SGD lr=0.01 | 0.5766 | 0.4930 / 0.5758 | OOM | 两实现初始 embedding 未按逻辑 ID 对齐 |
| Taobao/DeepFM | 75 步、SGD lr=0.001 | 未测 | 0.4864 / 0.6823 | 未测 | 仅训练 5 个子集 epoch |
| Taobao/DeepFM | 约 500 步、SGD lr=0.01 | 0.4878 | 0.5025 / 0.2309 | 0.5100 / 0.2005 | GMP 为 510 步，验证在第 510 步 |
| Terabyte/DCNv2，本地 batch 2,048 | 500 步、SGD lr=0.01 | **0.6001** | **0.5273 / 0.1737** | 未测 | 原生 HugeCTR AUC **0.5333**；最后 16,384 条独立留出 |

Terabyte 追加了相同验证批、500 步、SGD lr=0.01 的五个整数 seed（42–46），每个 seed 各自重新启动真实 8 卡训练。训练前缀为 63 个完整 global batch，训练时循环使用；验证是独立的第 64 批。下表的 seed 仅对齐整数值，**未对齐按逻辑 ID 的初始 embedding/optimizer state，也未保证 TorchRec 与 HugeCTR 的逐参数实现一致**，因此“RAPID−原生”只是描述性差值，不是 response R5 的配对等价性统计。

| Seed | 原生 HugeCTR AUC | RAPID+HugeCTR AUC | 描述性 AUC 差 | TorchRec AUC | TorchRec LogLoss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.533292 | 0.600074 | +0.066782 | 0.527273 | 0.173678 |
| 43 | 0.517156 | 0.587845 | +0.070689 | 0.527463 | 0.169703 |
| 44 | 0.507359 | 0.586084 | +0.078725 | 0.533511 | 0.176452 |
| 45 | 0.560977 | 0.619252 | +0.058275 | 0.549867 | 0.155402 |
| 46 | 0.537097 | 0.582780 | +0.045683 | 0.532981 | 0.164310 |
| 均值 | **0.531176** | **0.595207** | **+0.064031** | **0.534219** | **0.167909** |

### 后续诊断：配对零初始化

为定位上表相同整数 seed 却不同 AUC 的原因，检查 HugeCTR EBC 默认初始化：初始化幅度随物理表容量改变，而 RAPID 重新映射 ID 并改变物理容量，因此同 seed 不保证同一逻辑 ID 的初始 embedding 一致。新增 `--zero-embedding-init` 诊断选项并重编译 HugeCTR：两种运行均把全部 embedding 初始化为零，使用同一 seed 对齐 dense 初始化，保持同一训练/验证批、500 步、SGD lr=0.01。`--initial-snapshot-prefix` 保存的初始 dense 权重逐 seed SHA256 完全相同。零初始化消除了这项起始权重差异，但不是论文默认初始化配置，也不能代替随机初值下的逻辑 ID 权重搬运与配对。

| Seed | 原生 HugeCTR AUC | RAPID+HugeCTR AUC | RAPID 减原生 |
| ---: | ---: | ---: | ---: |
| 42 | 0.618096 | 0.618097 | +0.000001 |
| 43 | 0.607152 | 0.607153 | +0.000001 |
| 44 | 0.605623 | 0.605624 | +0.000001 |
| 45 | 0.622455 | 0.622456 | +0.000001 |
| 46 | 0.606525 | 0.606525 | 0.000000 |
| 均值 | **0.6119702** | **0.6119710** | **+0.0000008** |

这项诊断表明，消除未配对 embedding 初始值后，五组 RAPID 与原生的短训练 AUC 几乎一致；它不能把原先约 +0.064 的 AUC 差解释为算法带来的提升。五组初始 dense 快照逐 seed SHA256 全部相同。日志为 `/tmp/rapid_r5_zero_{native,rapid}_accuracy*.log`，初始 dense 快照为 `/tmp/rapid_r5_zero_{native,rapid}_seed*_dense_0.model`。两种训练仍使用 1M 子集、单批验证；正式 R5 的完整数据、完整训练、LogLoss 与随机初值配对结论仍待补。

Kaggle 的 dense-only 诊断模型在同一留出集 AUC 约 0.693，说明该输入并非完全不可学习。TorchRec 的两步审计显示 dense 各层和抽样的 embedding 权重都有实际更新。Taobao 子集正例仅 3,320/65,536；单表历史点击率诊断最高约 0.569 AUC，而三种 DeepFM 当前训练 AUC 仍接近 0.5。Terabyte 追加试验的 RAPID 与原生 HugeCTR 验证批均为 16,384 条、566 个正例；标签与 dense 记录多重集核对一致，RAPID 导出器也逐批审计了完整原始记录。此试验 RAPID 的 AUC 高于两种参考实现，但训练顺序、逻辑 ID 对应的初始 embedding 和 optimizer state 尚未配对，不能据此判断方法本身提高或保持精度。HugeCTR 当前 `AverageLoss` 输出为 0，不是有效 LogLoss。**这组 1M 子集与短训练结果不能支持 response R5 的“精度保持”主张**：缺少完整训练数据、五组配对 seed、逻辑 ID 对齐的初始权重及 optimizer state、全量验证集和预先确定的等价界限。三框架之间模型实现及表宽也未严格逐参数配对。response 的 R5 空格应继续保留。

## 可运行性与下一组公平 batch 实验

### 2,048 本地 batch 的追加实测

为让 128 维 Criteo 的 HET-GMP 能在 8 张 V100 上运行，重新从同一份全局 shuffle 输入规划每卡 2,048、全局 16,384、64 个完整 global batch。Kaggle 与 Terabyte 的 RAPID 离线规划和运行映射均已完成，导出的 64 批逐条记录与原始输入核对一致；两组 TorchRec 输入与规划输入的 SHA256 也一致。该组 batch 与上表的 8,192 不同，不填入 response R1 对应单元格。

四种实现共享 26 张表的 pooling factor、128 维 embedding、底层 MLP 512/256/128、三层宽 512 的 cross 投影及顶层 MLP 1024/1024/512/256/1；GMP 自定义 cross 层另有显式 bias，TorchRec 的两个 cross 线性投影设为无 bias，初始化和 embedding 优化器也未逐参数统一。因此这是同形态网络的吞吐对照，尚不是严格逐参数/逐 FLOP 相同模型的最终 R1 比较。

| 数据集/DCNv2 风格、128 维 | HugeCTR 原生全 Unique 三轮内部计时 samples/s | TorchRec 三轮 samples/s | HET-GMP 三轮 samples/s | RAPID+HugeCTR 三轮内部计时 samples/s |
| --- | --- | --- | --- | --- |
| Kaggle | 233,216 / 233,215 / 233,369；均值 **233,266** | 300,433 / 302,016 / 299,406；均值 **300,618** | 166,046 / 165,832 / 152,930；均值 **161,603** | 347,892 / 348,035 / 348,116；均值 **348,014** |
| Terabyte | 233,779 / 233,592 / 233,644；均值 **233,671** | 329,649 / 328,388 / 329,823；均值 **329,287** | 173,954 / 173,912 / 174,047；均值 **173,971** | 329,563 / 330,233 / 330,413；均值 **330,070** |

HugeCTR 与 RAPID 从第 40 至 110 步、GMP 同样从第 40 至 110 步按每 10 步区块耗时计算全局吞吐；TorchRec 预热 30 步、计时 80 步。四个方法均完成真实 GPU 前向、反向及优化器更新。GMP 日志的 `speed` 是单 rank 值，上表乘以 8 并以实际区块耗时重算。所有有效轮次、所有可观测 rank 的训练窗口 `read_bytes` 增量均为 0。Kaggle TorchRec 将循环输入预放 GPU 后为 **305,046 samples/s**，相对锁页预取三轮均值高 **1.47%**；Terabyte 同样消融为 **332,849 samples/s**，高于锁页预取均值 **1.08%**。GMP 将 sparse ID 预放 GPU 后，Kaggle 为 **165,524 samples/s**，与正常前两轮 166,046/165,832 接近；Terabyte 为 **174,131 samples/s**，比常规均值高 **0.09%**。这些测试排除了明显的物理读取瓶颈，也显示 H2D/预取不是本配置的主要耗时。GMP Kaggle 第三轮比前两轮低约 8%，保留原值而不挑选最快轮次。

新 RAPID 静态 ACP 在训练前确定：Kaggle **25 张 Unique、表 20 为 Reduction**；Terabyte **26 张 Unique、0 张 Reduction**。该选择来自本组 2,048 batch 的 CPS 后通信剖面，不能套用 8,192 batch 的 ACP 文件。旧内部计时得到的跨框架百分比 **15.77% 和 0.24% 无效**。按日志墙钟复核，Kaggle RAPID **314,674**、TorchRec **300,618** samples/s（约高 **4.7%**）；Terabyte RAPID **298,502**、TorchRec **329,287**（约低 **9.4%**）。原生 HugeCTR 同口径墙钟分别为 Kaggle **210,817**、Terabyte **211,134**；旧表对原生的 **49.19%/41.25%** 也是内部计时值。在线输入映射 Kaggle 每轮约 74–75 秒、Terabyte 约 82–83 秒，模型编译另约 28–30 秒；原生 HugeCTR 输入设置仅约 6–7 秒，因此短作业完整墙钟的比较不能由稳态吞吐代替。原生 HugeCTR 诊断曾错误地对原始大 ID 使用活跃 ID 数量作为表容量，引发首步越界和 `CUDNN_STATUS_MAPPING_ERROR`；该诊断已剔除。有效的原生对照从本组同一 shuffle 输入生成紧凑 ID，逐表检查容量，标签与 dense 列逐条一致。使用与本组规划不同 SHA256 的旧 `native.bin` 所做启动试验也已剔除。日志和导出位于 `/tmp/rapid_r1_*b2048*`，输入及 GMP 数组在 `/dev/shm`，未读取 `Rapid/store`。

以日志创建到最后写入的时间估算 110 步短作业墙钟（包含进程启动、模型编译与输入设置，不含离线 CPS/ACP 和数据预处理）：Kaggle 原生 HugeCTR 三轮约 **47/48/47 秒**、RAPID **127/129/125 秒**、GMP **20/20/20 秒**；Terabyte 原生 HugeCTR **47/52/48 秒**、RAPID **138/138/137 秒**、GMP **19/20/21 秒**。TorchRec 两组有效日志约 **19–20 秒**；Kaggle 第一轮的日志文件曾用于失败的环境诊断，不能用其创建时间估算墙钟。这些数值是日志时间戳近似，不是严格的预处理加总训练时间；RAPID 的稳态训练吞吐不能抵消该短作业的在线映射成本。

本次 2,048 batch 的 RAPID 离线规划从进程日志创建到 `complete.json` 写入，Kaggle 约 **26.9 分钟**、Terabyte 约 **23.6 分钟**。两项规划同时运行且期间还有分区和 GPU 基线任务，CPU/存储资源并非独占；这些是本轮实际观察到的准备时间，不能作为公平的独占规划基准。首次运行若包含规划，端到端总时间还要加上这一成本；上表的稳态训练吞吐并未包含它。

Kaggle 另用相同紧凑原生输入、只应用静态 ACP（表 20 Reduction，不使用 RAPID 样本调度和在线映射）跑三轮：**329,908 / 330,266 / 330,172 samples/s，均值 330,115**，计时窗口 `read_bytes` 均为 0。相对原生全 Unique 提升 **41.52%**；完整 RAPID 再比 ACP-only 高 **5.42%**。因此在该配置下，大部分 HugeCTR 吞吐提升来自表 20 的通信策略选择，其余来自 RAPID 的放置/样本调度等组合；这项消融不能把剩余 5.42% 单独归因于 CPS，因为规划还改变了物理 ID 和拥有者分布。

- GMP Criteo 128 维、本地 batch 8192：Kaggle 和 Terabyte 均在 CUDA 设备内存分配失败。Kaggle 降至 4096 仍失败，2048 完成 10 步启动测试；该诊断不能与 8192 的其他框架直接比较。上游 `hetuCTR/csrc/core/table_initialize.cu` 的辅助缓冲按 `batch_size × nrank × embedding_width` 为查询和返回分别预留双缓冲，multi-hot 输入放大了内存需求。
- Terabyte RAPID+HugeCTR 128 维、本地 batch 8192：在线映射和预物化输入均在 `model.compile()` 处 OOM。其最终计划有 26,250,363 个物理 embedding 行，8 卡 V100 32 GiB 不足以支持此 dense 网络/缓冲组合；不能把失败写成低吞吐。
- 已在 `/tmp/rapid_r1_{kaggle,terabyte}_b2048_plan` 从相同 shuffle 输入重新完成离线规划（每卡 batch 2048、64 global batch），导出并核对 runtime 映射和静态 ACP；四框架真实 GPU 吞吐和读取诊断见本节。此配置与上面的 8192 结果独立报告。

本报告不修改 `中文Response.md` 的 R1/R5 待填表格：目前只具备两个模型、三个数据集子集的部分真实 GPU 结果；尚缺 DLRM、Wide&Deep、MovieLens/NeuMF、T 系列及外部 EMBark/Herald/BagPipe 的严格模型适配。仓库中 `code/Rapid/training/run_hugectr_dlrm.py` 实际使用 MultiCross（DCNv2 风格），不能作为 DLRM 的实测入口。Response 引用的 `训练实验方案.md` 当前不存在，完整模型宽度、训练终止条件和验证划分仍需在正式填表前固定。

主要日志：`/tmp/rapid_r1_*`。本轮只使用 `Rapid/code`、`Rapid/code/upstream_fresh`、`/tmp` 和 `/dev/shm` 的代码与数据，没有读取 `Rapid/store` 中的文件。新训练/验证文件写入 `/dev/shm/rapid_r1_split2026` 和 `/dev/shm/rapid_r1_terabyte_b2048_*` 前逐次检查可用容量；代码与文档留在 home。
