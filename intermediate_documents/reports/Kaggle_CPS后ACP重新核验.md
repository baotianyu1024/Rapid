# Kaggle CPS 后 ACP 重新核验

> 后续排查更正：本文的 51,158.2578125 是 CPS 后再经额外 owner/sample 优化的同数据 profile，不是初始 CPS 的直接结果；初始 CPS 为 60,427.875，会选择表 20 Reduction。此前流程没有启用论文的子图聚类初始化，并存在显著规划顺序偏置。完整证据见《Kaggle批间通信量与CPS顺序偏置排查.md》。

本次只针对前述 Kaggle 表 20 和静态 ACP 选择重新实验。数据来自此前重新下载的 1,048,576 行输入，8 GPU，每 rank batch 为 8,192，16 个不同 global batch 循环。CPS 放置与样本调度固定；ACP 在训练之前逐表决定一次，训练期间不按 batch 切换。此数据是用于计划和统计的同一 16 个 batch，无法代表未来数据的分布。

## 统一口径的通信计数

直接从 CPS 后物化记录重算每个 batch、每个 rank 表 20 的远端去重 ID 数：物理 ID 的 owner 是 `ID % 8`，排除 owner 为当前 rank 的 ID，然后计算剩余不同 ID 的数量。物化记录按 batch、rank 和 rank 内 8,192 行排列。重新计算的 128 个结果与现有 CPS profile 的 `receive[:,20,:]` 逐项完全相同；这是独立从输入重新计数，不是 GPU 链路实测。脚本为 `code/recheck_cps_acp.py`，完整 16×8 值在 `/tmp/rapid_acp_recheck/table20_recount.json`。

只计算远端通信时，固定长度 Reduction 为 `8192×(8−1)=57,344` 向量/步/rank，Unique 为远端不同 ID 的接收向量数。按 16 batch、8 rank 的平均，Unique 是 **51,158.2578125**，低于 57,344，因此此有限 profile 下静态 ACP 选择 Unique；不等于每个 batch 都应在线变更策略。所有 batch 和 rank 的最小/最大值为 24,995 / 72,181。

| batch 编号（从 0 开始） | 8 rank 平均远端 Unique 向量数 |
| ---: | ---: |
| 0 | 32,843.125 |
| 1 | 32,640.375 |
| 2 | 37,543.250 |
| 3 | 39,538.000 |
| 4 | 42,653.875 |
| 5 | 45,172.875 |
| 6 | 48,278.750 |
| 7 | 50,586.125 |
| 8 | 52,975.500 |
| 9 | 55,571.500 |
| 10 | 58,205.125 |
| 11 | 60,580.500 |
| 12 | 62,357.250 |
| 13 | 64,690.250 |
| 14 | 66,189.625 |
| 15 | 68,706.000 |

16 批中后 6 批的 rank 平均超过远端 Reduction 阈值；128 个 batch/rank 组合中有 44 个超过阈值。前 4 批均值为 35,641.1875，后 8 批为 61,159.46875，后 4 批为 65,485.78125。最后一批的 68,706 与用户给出的 68,493.41 数值接近，但没有证据证明来自相同数据或 profile 窗口。

论文写明先 CPS、后 ACP，并在 Table 2 报告 8 GPU Kaggle 表 20 为 Reduction。本次 16 batch 重下载切片选择 Unique，**未复现论文表的策略**；可能涉及输入切片、multi-hot 构造、CPS 结果或统计窗口，当前没有证据区分原因。用户给出的 68,493.41 高于 57,344，按远端口径确实应选择 Reduction。

旧实现把远端 Unique 与包含本地项的 `B×G` 直接比较，不是同一口径。已将最终计划和独立导出器的默认改为 `B×(G−1)`；旧 `paper` 选项仅供复核历史结果。该修正没有改变本次 16 batch 的选择，因为 51,158.26 同时小于 57,344 和 65,536。

## 同一 CPS 放置的固定策略训练对照

以下比较只改表 20 的 Unique/Reduction；其他 25 张表始终为 Unique，使用同一 CPS 样本序列、表容量、初始 seed 和数据。各模式三个独立进程交替运行，预热 30 步、测量 80 步，全局 batch 65,536、8 张 V100、NCCL P2P 禁用。预物化用于隔离策略本身，在线版本从原始数据和同一 CPS 映射读取；两者都使用 RAM 数据和 buffered reader。逐轮吞吐为对应 80 步墙钟的样本数除以时间，平均是三轮吞吐算术平均。

| 输入方式及固定策略 | 三轮 ms/步 | 平均 samples/s |
| --- | --- | ---: |
| 预物化，全 Unique | 49.7625 / 49.7750 / 49.8250 | 1,316,315 |
| 预物化，表 20 Reduction | 46.9625 / 46.9625 / 47.2500 | 1,392,666 |
| 在线映射，全 Unique | 50.3875 / 49.2625 / 49.3000 | 1,320,104 |
| 在线映射，表 20 Reduction | 46.5625 / 46.3375 / 47.0625 | 1,404,778 |

固定 Reduction 在预物化和在线模式的三轮平均吞吐分别高 5.80% 和 6.41%。因此，在本次机器与 workload 下，通信向量平均数较少的 Unique 并不是测得的吞吐最快策略；这不能单独证明具体是哪一个 GPU kernel 或网络阶段造成差异，或推广到论文使用的数据及其他硬件。本次没有动态逐 batch 切换，也没有更改 ACP 规则去使用吞吐反馈。

12 次运行的测量窗口进程物理 `read_bytes` 增量全部为 0。同一输入方式和重复轮次下，两种策略各打印 11 个 loss，显示值完全一致；这是有限精度的短程检查，不是逐参数梯度或最终 AUC 校验。预物化输入 setup 约 7.18–7.96 秒，在线映射输入 setup 约 69.53–76.12 秒；setup 包含计划解析等初始化，不在稳态吞吐中。CPS 规划、数据转换、模型启动、评估和 checkpoint 同样未计入。

## 代码、验证与证据

`code/Rapid/experiments/acp/export_hugectr_acp.py` 和 `code/Rapid/experiments/export_runtime_plan.py` 的默认已修正为远端口径；原 `paper` 选项保留用于核对旧结果。`code/recheck_cps_acp.py` 重新计数，`code/run_cps_acp_recheck.sh` 与 `code/run_cps_acp_online_recheck.sh` 执行两类训练，`code/summarize_cps_acp_recheck.py` 校验 loss、物理读取及汇总时间。ACP 导出和读取的单测、Python 与 shell 语法检查通过。

本次报告、全部日志、静态计划、逐 batch/rank 计数及汇总位于数据盘 `rapid_experiment_data/evidence_20261102/kaggle_cps_acp_recheck_20261102.tar`。本轮没有访问 `store`。
