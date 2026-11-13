# 内存驻留输入、公平性与 ACP 对照实验

> 后续排查：这里的“CPS 后”使用了额外 owner/sample 优化，且此前流程未启用论文 CPS 的子图聚类初始化；初始 CPS 的 Kaggle 表 20 实际选择 Reduction，规划顺序偏置明显。详见《Kaggle批间通信量与CPS顺序偏置排查.md》。先前吞吐不能作为论文完整 CPS 的严格复现。

> 更正：本次 ACP 的 `paper` 导出选项将远端 Unique 数与包含本地项的 `B×G` Reduction 数直接比较，统计范围并不一致。只比较远端通信应使用 `B×(G−1)`；本次三个 CPS 后计划在该阈值下仍全选 Unique，但这不能复现论文 Table 2 中 8 GPU Kaggle 表 20 选择 Reduction 的结果。下文的 `B×G` 是本次实际运行使用的阈值，不能作为已验证的远端通信最优规则。

日期：2026-11-02。本文记录重新下载的数据上的实验，不读取 `store`。历史实验与此前 response 的核查见《端到端对照与ACP排查报告》和相关验证报告；本文使用更严格的内存驻留输入与预热设置，历史吞吐不能混入本轮均值。

## 配置与实验边界

8 张 V100-SXM2 32 GiB，全局 batch 65,536，FP32，NCCL_P2P_DISABLE=1；预热 30 步，测量 80 步，独立运行三轮。Kaggle、Terabyte、Taobao 各选取 1,048,576 样本，16 个不同 batch 循环。前两者用 DCNv2 风格，后者用 DeepFM 风格；Terabyte 并非论文的 DLRM。训练吞吐排除数据预处理、规划、进程启动、模型构建、评估与 checkpoint。

输入文件逐一检查内存及目标容量后复制到 `/dev/shm/rapid_read_fair` 并逐文件 SHA256 校验；这避免源盘读取成为测试期瓶颈，而不代表数据超过 RAM 或冷盘训练的表现。HugeCTR 使用 buffered reader；TorchRec 把 16 个不同 batch 预格式化到锁页内存并双缓冲异步预取，另外把全部输入放 GPU 作诊断；GMP 启动时读入内存，以普通 host 输入为主，并比较锁页与全部稀疏 ID 放 GPU。后两类 GPU 预放模式消耗额外显存，只用于输入上界诊断。

表中的“原生 HugeCTR”指没有 RAPID 重排和在线映射的 hash 放置基线，但使用同一份修复 Unique 正确性问题的 HugeCTR 构建和 buffered I/O；它不是未打补丁的原始上游二进制。

GMP 的原有 rank 行数略不齐，`drop_last` 会丢弃整批。新分区仅移动 Kaggle 15、Terabyte 3、Taobao 24 条样本，8 rank 均有 131,072 条，根表和镜像逐元素核验不变。跨原始 global batch 的重新分配仍会影响样本组成，不应声称各框架训练轨迹逐项配对。各系统计时边界、初始权重、优化器状态和 embedding 布局也未完全对齐；这是控制读取后的工程吞吐对照，不是可直接写进 response 的严格公平排名。

## HugeCTR 改动及其作用

1. `HugeCTR/src/data_readers/multi_hot/detail/batch_file_reader.cpp` 增加可选 `HUGECTR_BUFFERED_IO=1`，默认仍为 `O_DIRECT`。此前本机 direct I/O 测得约 138 MB/s，约 59.8 MB 的 Criteo batch 估算 0.43 s，与旧版约 0.42 s/步相近。切换缓冲读取和内存驻留消除这一实验设置的存储等待；该数字不是一般硬盘性能结论。
2. `HugeCTR/include/data_readers/multi_hot/detail/rapid_host_plan.hpp` 对合适表采用至多 1 GiB/计划实例的直接索引，否则回退哈希；复用每线程缓冲并并行回写，`RAPID_MAPPING_THREADS=16`。Kaggle 的 reader 分阶段诊断，准备、映射、回写由约 49.20/23.45/16.27 ms 降至 2.74/21.98/3.02 ms。三段 CPU 时间可与 GPU 重叠，不能相加后当作训练步额外时间。
3. `embedding_collection.cpp` 修复 `DenseModelParallelWithReduction` 的梯度初始化分支，避免误进入全表 AllReduce 的初始化逻辑；`model_pipeline.cpp` 和 `embedding_collection.cpp` 同时补齐该类型的参数更新。旧库四表各有零个权重变化，修复后每表 1024 个变化。此项首先是训练正确性修复，不能把遗漏更新的旧结果与正确训练直接比较。
4. 使用 Release 构建。原生 direct I/O 测试从 423.55 降到 420.58 ms/步，编译优化本身不是原生慢的主要原因；在线映射则受 CPU 优化影响。
5. 新增 ACP 离线逐表选择及 HugeCTR 加载、映射计划与增量更新；ACP 改变实际通信策略，在线映射在 reader 执行。CPS/ACP 规划和输入重排并非免费，主表未计规划与初始化时间。

Kaggle 测得在线 RAPID 初始化输入阶段约 69–73 秒，原生/ACP-only 约 7.5–8 秒，另有模型 compile 约 32 秒；这些阶段口径包含设备初始化及计划解析，并非纯磁盘读取。短训练时应计入完整启动和规划，不能以稳态吞吐声称端到端训练总时长同样提高。

## ACP 实际选择

论文规则用每表平均每 rank 远端 Unique 数与 `B×G=8192×8=65536` 比较：小于阈值用 Unique，否则 Reduction；训练前静态选择。表编号从 0 开始。

| 数据 | 原始 hash 放置、前 4 batch | CPS 后、全部 16 batch |
| --- | --- | --- |
| Kaggle | 25 Unique，1 Reduction（表 20） | 26 Unique，0 Reduction |
| Terabyte | 24 Unique，2 Reduction（表 20、21） | 26 Unique，0 Reduction |
| Taobao | 4 Unique，0 Reduction | 4 Unique，0 Reduction |

这是这些输入、profile 与阈值下的结果，不能推断 ACP 总是选择 Unique；两组 profile 的 batch 数也不同。人工构造含重复 ID 的四表混合策略（2 Unique、2 Reduction）训练 3 步，与相同初始化的全 Unique 对照 loss 一致，更新后的 embedding 最大绝对差 `2.98e-8`，是短程验证而非完整精度验证。训练过程中观测的 AllReduce 对应 dense 网络梯度约 581 万 FP32 元素及对齐，而非 Unique 表的全表 embedding AllReduce。

关键表的平均远端 Unique 数：Kaggle 表 20 在 hash 前 4 批为 368,911.2，CPS 后全部 16 批为 51,158.3；Terabyte 表 20/21 在 hash 前 4 批为 244,845.4/79,675.7，CPS 后全部 16 批为 19,200.8/4,585.3。表 20 的 CPS 前四批均值是 35,641.2，不能与其 16 批决策值混淆。每张表的精确值和选择均保存在对应的 ACP JSON 证据中。

## Kaggle：ACP-only 为什么略快？

本轮 hash+ACP-only 为 1,348,131 samples/s，在线完整 RAPID 为 1,327,540，相差约 1.55%。这一结果不支持声称该子集的 CPS 带来额外吞吐提升。hash+ACP 对表 20 选择 Reduction；CPS 后全部 Unique，因而内核、去重和通信路径不同，样本分组、物理容量及初始权重也不同。前四批协议模型中，hash+ACP 估算 133,386.94、CPS+ACP 估算 81,248.69 个远端向量/步/rank，后者减少 39.09%；这是模型估算，不是 GPU 实测字节。更少通信量不必然等于更短墙钟，因为计算、通信和读入会重叠；现有测量不能确定哪项造成 1.55% 差异。

先前另对 hash 全 Reduction 与 hash+ACP 捕获预热后 20 步：NCCL Send/Recv 的 8 卡累计核函数时间分别为 13.196 秒和 1.624 秒，dense AllReduce 均约 1.37 秒。这解释了同一 hash 放置下策略对训练耗时的重要性；它不是 ACP-only 对完整 RAPID 的直接比较。跨 GPU 和 stream 的核函数会重叠，累计时间不能直接相减为墙钟节省。

本轮使用同样 RAM 输入配置，在第 30–50 步通过 Nsight Systems 追踪 ACP-only 与在线 RAPID，核函数八卡累计如下；两者的通信策略和样本/表布局不同，不是严格单因素消融：

| GPU 核函数 | hash + ACP-only | 在线 RAPID |
| --- | ---: | ---: |
| NCCL Send/Recv | 1.632 s / 1,280 次 | 1.321 s / 640 次 |
| dense FP32 AllReduce | 1.371 s / 160 次 | 1.387 s / 160 次 |
| embedding backward one-to-one atomic | 0.217 s / 160 次 | 0.307 s / 160 次 |
| embedding forward reset | 0.106 s / 160 次 | 0.199 s / 160 次 |
| partition-and-unique：identity + shard | 0.038 s / 320 次 | 0.239 s / 320 次 |

跟踪显示完整 RAPID 的 Send/Recv 累计降低，而这些去重和 embedding 处理核函数的累计增加；这与“通信减少不一定缩短关键路径”的解释一致，但既不能证明这些核函数是全部原因，也不能直接把累计差相加为墙钟差。需进一步固定逻辑权重、样本及布局，做逐表策略配对和关键路径分析才能确定因果。追踪会引入开销，不作为主表吞吐。

## 输入消融：Kaggle

主对照中所有被测进程在测量区间 `/proc/self/io` 的物理 `read_bytes` 增量为零；这不排除内存复制、格式化、页缺失、H2D 等开销。TorchRec 曾在相同数据进行单轮试探：原始数组驻留 CPU 但每步转换约 99.90 ms/步；预格式化锁页约 55.33 ms，异步预取约 53.59 ms，16 个不同 batch 全部预放 GPU 约 52.73 ms。说明单纯消除读盘仍可能被输入处理拖慢，预格式化和预取是这次公平对照的必要设置。

正式三轮 TorchRec 预取 53.17/53.36/53.23 ms/步，GPU 预放 52.92/52.67/52.75 ms/步；平均吞吐分别为 1,230,607 和 1,241,681 samples/s，差约 0.9%。这些数据支持当前预取输入路径对稳态吞吐的影响较小，不能推广至其他数据规模或冷盘。

GMP 分别以普通 host 内存、锁页 host 内存、稀疏 ID 全预放 GPU 运行，三轮分别为 97.09/97.18/97.13、98.05/98.35/98.07、97.09/96.98/97.16 ms/步；普通和 GPU 模式平均吞吐为 674,687 和 675,096 samples/s，仅差 0.06%。GPU 模式仍从 host 提供 dense/label，额外显存约 224 MiB/rank，结果只排除稀疏 ID 传输的主要影响。普通输入的 loader 计时约 0.21–0.23 ms/步/rank；异步 SharedTable 内 GPU 排序、去重和通信不应被统称为数据读取。三种模式的 11 个打印 loss 一致。

HugeCTR 在线和相同计划预物化输入分别为 49.31/49.29/49.50 与 49.83/49.84/49.83 ms/步；在线映射在 CPU 上确实耗时，但此工作负载中可与 GPU 流水线重叠，两者差异不能当作映射成本的精确测量。先前热缓存、固定复用首批的 `SKIP_H2D` 控制中，原生 Reduction 为 118.54 对正常输入 119.70 ms/步、Unique 为 66.11 对 66.57 ms/步；该控制改变 batch 序列，只是诊断上界。

## 核验和局限

Kaggle 在线与预物化每次 11 个打印 loss 一致；TorchRec 预取与 GPU 预放每次 80 个 loss 最大差不超过 `5.96e-8`。先前 8 卡测得四种框架各独立进程的 NVLink payload 计数增量为零；禁用 NCCL P2P 后经主机/PCIe，并不等同启用 GPU 间 PCIe P2P。ACP schema 测试、边界测试和代码语法检查通过。

该实验未完成全量数据、跨框架逐参数配对、完整 AUC/LogLoss、多 seed、完整 DLRM、全程总耗时与冷盘/超内存测试，不能据此证明论文所有 response 数值或者最终排名。输入从 RAM 取数时无物理读盘也不意味着 CPU 变换、内存带宽和 H2D 自动无开销，需要看下列对应消融。

TorchRec 运行时 FBGEMM 发出 autograd kernel 注册警告；抽样 embedding 权重确实有更新，但尚未逐项验证完整梯度与 AUC，必须保留这一正确性限制。

## 三数据集稳态吞吐

单位 samples/s，三轮吞吐算术平均；GPU 预放仅为诊断。具体逐轮值和测量区间 I/O 在归档 JSON 中。

| 配置 | Kaggle | Terabyte | Taobao |
| --- | ---: | ---: | ---: |
| HugeCTR 全 Reduction | 547,388 | 547,217 | — |
| HugeCTR 全 Unique | 979,506 | 972,525 | — |
| HugeCTR Taobao 全 Unique | — | — | 7,133,196 |
| hash + ACP-only | 1,348,131 | 1,190,752 | — |
| RAPID 预物化 | 1,315,214 | 1,346,515 | 12,378,661 |
| RAPID 在线 | 1,327,540 | 1,355,228 | 12,492,978 |
| TorchRec 预取 | 1,230,607 | 1,196,591 | 3,918,016 |
| TorchRec GPU 预放（诊断） | 1,241,681 | 1,213,859 | 3,957,228 |
| GMP 普通 host 输入 | 673,041 | 689,602 | 9,862,499 |

各配置逐轮 ms/步：

| 配置 | Kaggle | Terabyte | Taobao |
| --- | --- | --- | --- |
| HugeCTR 全 Reduction | 119.60 / 119.74 / 119.84 | 119.67 / 119.81 / 119.80 | — |
| HugeCTR 全 Unique | 66.66 / 66.76 / 67.30 | 67.41 / 67.33 / 67.42 | — |
| HugeCTR Taobao 全 Unique | — | — | 9.21 / 9.17 / 9.17 |
| hash + ACP-only | 48.61 / 48.61 / 48.61 | 55.06 / 55.04 / 55.01 | — |
| RAPID 预物化 | 49.83 / 49.84 / 49.83 | 48.68 / 48.68 / 48.66 | 5.42 / 5.24 / 5.22 |
| RAPID 在线 | 49.31 / 49.29 / 49.50 | 48.33 / 48.20 / 48.55 | 5.24 / 5.25 / 5.25 |
| TorchRec 预取 | 53.17 / 53.36 / 53.23 | 55.21 / 54.36 / 54.74 | 16.76 / 16.66 / 16.76 |
| TorchRec GPU 预放（诊断） | 52.92 / 52.67 / 52.75 | 53.87 / 53.81 / 54.29 | 16.52 / 16.60 / 16.56 |
| GMP 普通 host 输入 | 97.75 / 97.16 / 97.21 | 95.09 / 94.94 / 95.08 | 6.63 / 6.66 / 6.65 |

## GMP 输入三种模式对照

三轮 ms/步；GPU 模式只预放全部稀疏 ID，dense/label 仍在 host。

| 数据 | 普通 host | 锁页 host | GPU 稀疏 ID | 普通 host loader 最慢 rank，ms/步 |
| --- | --- | --- | --- | ---: |
| kaggle | 97.09 / 97.18 / 97.13 | 98.05 / 98.35 / 98.07 | 97.09 / 96.98 / 97.16 | 0.231 |
| terabyte | 95.62 / 94.93 / 94.88 | 95.89 / 95.76 / 95.83 | 94.77 / 95.15 / 94.65 | 0.236 |
| taobao | 6.74 / 6.65 / 6.65 | 6.69 / 6.69 / 6.65 | 6.65 / 6.61 / 6.60 | 0.097 |

每个数据集三种输入模式的 11 个打印 loss 均一致。loader 时间来自启用分段计时的首轮，是 host 侧调用时间，不代表 GPU 上排序、去重、通信或整个异步流水线的完成时间。

## 输入、启动与结果核对

- kaggle：TorchRec 预取与 GPU 预放吞吐差 0.90%；GMP 普通与 GPU 稀疏输入吞吐差 0.06%。
  主对照及 GMP 输入消融各进程/各 rank 测量窗口物理 read_bytes 增量均为零。
  torch_prefetch 各 rank 启动输入加载 1.48–1.51 s，输入准备 0.53–0.59 s。
  het 各 rank 启动输入加载 0.78–0.79 s，输入准备 0.47–0.54 s。
  unique 输入 setup 8.05–8.05 s（含计划等初始化）。
  acp_only 输入 setup 7.54–7.83 s（含计划等初始化）。
  materialized 输入 setup 7.60–7.69 s（含计划等初始化）。
  online 输入 setup 69.73–72.83 s（含计划等初始化）。
  materialized_vs_online_1：11 个 loss，最大绝对差 0。
  torch_prefetch_vs_torch_gpu_1：80 个 loss，最大绝对差 0。
  materialized_vs_online_2：11 个 loss，最大绝对差 0。
  torch_prefetch_vs_torch_gpu_2：80 个 loss，最大绝对差 0。
  materialized_vs_online_3：11 个 loss，最大绝对差 0。
  torch_prefetch_vs_torch_gpu_3：80 个 loss，最大绝对差 5.96e-08。
- terabyte：TorchRec 预取与 GPU 预放吞吐差 1.44%；GMP 普通与 GPU 稀疏输入吞吐差 0.30%。
  主对照及 GMP 输入消融各进程/各 rank 测量窗口物理 read_bytes 增量均为零。
  torch_prefetch 各 rank 启动输入加载 1.49–1.55 s，输入准备 0.53–0.63 s。
  het 各 rank 启动输入加载 0.76–0.83 s，输入准备 0.48–0.55 s。
  reduction 输入 setup 7.84–12.30 s（含计划等初始化）。
  unique 输入 setup 7.45–8.05 s（含计划等初始化）。
  acp_only 输入 setup 7.41–9.54 s（含计划等初始化）。
  materialized 输入 setup 7.43–8.37 s（含计划等初始化）。
  online 输入 setup 84.47–87.51 s（含计划等初始化）。
  materialized_vs_online_1：11 个 loss，最大绝对差 0。
  torch_prefetch_vs_torch_gpu_1：80 个 loss，最大绝对差 5.96e-08。
  materialized_vs_online_2：11 个 loss，最大绝对差 0。
  torch_prefetch_vs_torch_gpu_2：80 个 loss，最大绝对差 5.96e-08。
  materialized_vs_online_3：11 个 loss，最大绝对差 0。
  torch_prefetch_vs_torch_gpu_3：80 个 loss，最大绝对差 5.96e-08。
- taobao：TorchRec 预取与 GPU 预放吞吐差 1.00%；GMP 普通与 GPU 稀疏输入吞吐差 0.90%。
  主对照及 GMP 输入消融各进程/各 rank 测量窗口物理 read_bytes 增量均为零。
  torch_prefetch 各 rank 启动输入加载 0.03–0.04 s，输入准备 0.01–0.02 s。
  het 各 rank 启动输入加载 0.02–0.02 s，输入准备 0.02–0.02 s。
  native 输入 setup 5.07–5.49 s（含计划等初始化）。
  materialized 输入 setup 5.14–6.26 s（含计划等初始化）。
  online 输入 setup 7.64–10.17 s（含计划等初始化）。
  materialized_vs_online_1：11 个 loss，最大绝对差 0。
  torch_prefetch_vs_torch_gpu_1：80 个 loss，最大绝对差 0。
  materialized_vs_online_2：11 个 loss，最大绝对差 0。
  torch_prefetch_vs_torch_gpu_2：80 个 loss，最大绝对差 0。
  materialized_vs_online_3：11 个 loss，最大绝对差 0。
  torch_prefetch_vs_torch_gpu_3：80 个 loss，最大绝对差 0。

## 证据与复现

代码位于 `Rapid/code`：`run_ram_fair.sh` 执行主对照，`run_het_input_diagnostics.sh` 执行 GMP 输入消融，`profile_ram_kaggle.sh` 采集追踪；`summarize_ram_fair.py` 和 `summarize_het_input.py` 汇总逐轮日志。ACP 导出器在 `code/Rapid/experiments/acp/`，HugeCTR 上游修改在 `code/upstream_fresh/HugeCTR/`。这些脚本引用本机临时路径和匹配的 CUDA/NCCL 运行环境，重启前应先恢复构建、数据与计划，不能直接假定 `/tmp` 和 `/dev/shm` 永久存在。

本轮日志、JSON、Nsight 报告、平衡分区、ACP 清单、SHA256 manifest 和本文归档在数据盘 `rapid_experiment_data/evidence_20261102/ram_input_comparison_20261102.tar`；原始公开数据的下载及 response 逐项验证参见《实验数据与Response逐项核对.md》。本轮的数据、上游与基线均来自此前重新下载的公开来源，没有读取 `store`。
