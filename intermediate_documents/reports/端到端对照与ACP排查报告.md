# HugeCTR、RAPID、TorchRec 和 HET-GMP 对照与 ACP 排查

后续内存驻留输入对照见《内存驻留输入公平性与ACP实验报告.md》；本报告中的吞吐是历史测量，不与后续数据合并。

日期：2026-11-02。本报告替代此前把未优化 HugeCTR 与 TorchRec 直接比较的性能结论。所有数据、上游代码均按原要求重新下载，本轮没有读取 `store`。

## 1. 直接结论

- 原生 HugeCTR 不存在“必然比 TorchRec 慢”的结论。多 hot 的 Criteo 数据上，论文将原生全 Reduction 作为主基线；它在本次 Kaggle 上为 54.7 万样本/秒，而同一 HugeCTR 的全 Unique 为 98.4 万、hash 放置 ACP-only 为 135.0 万。相同数据和模型中的通信策略影响很大。
- 完整 RAPID 在本次 Kaggle、Terabyte 的在线吞吐分别约 131.3 万、134.0 万样本/秒，与不做在线映射的预物化对照相近。此前约 6 万/秒的结果受到 `O_DIRECT` 冷/直读、无优化编译、每 batch 大缓冲清零及 Unique 参数更新遗漏等问题影响，不能视为有效训练的性能结论。
- 新增 ACP 离线逐表决策清单及 HugeCTR 加载。根据论文第 6 节，若平均每 rank 每 batch 的远端 Unique 数小于 `B×G`，表使用 Unique，否则使用 Reduction。决策在训练初始化时固定；不做每步重规划。
- 这仍是工程对照，不能证明论文全规模训练、AUC、跨框架精度和所有 batch 下的公平排名。尤其本次 Terabyte 用 DCNv2 风格网络，论文该数据集的代表网络是 DLRM；HET 改变 batch 成员，三框架的权重及优化器状态未逐项对齐。

## 2. 实验配置与计时口径

重新下载的 Kaggle 和 Terabyte 各取 1,048,576 行，13 个 dense 特征、26 张 embedding 表，固定 multi-hot 总长度 214；Taobao 使用新下载 CSV 合并后的 1,048,576 行，4 个 one-hot 表和 1 个 dense 特征。Criteo 使用 DCNv2 风格模型、embedding dim 32，Taobao 使用 DeepFM 风格模型、主 embedding dim 128。8 张 V100-SXM2 32 GiB、全局 batch 65,536、SGD 0.001、FP32，禁用 NCCL P2P；模型不代表完整数据集或完整训练。

每种配置独立进程三轮，HugeCTR 以第 10 至第 90 步的日志墙钟时间相减除以 80，TorchRec 每步同步并计入 CPU 输入切片、构造 KJT、H2D 和训练，取各 rank 每步最大时间；HET-GMP 每 10 步按 rank 最大时间计时。均为热缓存或已加载输入，排除离线数据转换、CPS/ACP 规划、进程启动、模型初始化、评估和 checkpoint；数据按 16 个 batch 循环使用。HugeCTR 每 10 步打印一次 loss，TorchRec 每步同步，HET 的数据已重分区并由自身 loader 预备，因此计时边界并非完全相同。以下表为实际工程运行对照，不是完全统一计时的基准发布结果。

原始 HugeCTR 使用相同逻辑输入的逐表紧凑 ID 与 hash 放置；RAPID 使用重排后的物理 ID 和样本序号，表容量按物理 ID 上界预留。预物化版本把同一 RAPID 计划写成临时输入，用于隔离在线映射，不是无需改变数据文件的方案。在线版从原始文件读取，host 执行样本调度与 ID 映射；本次为 Release、`HUGECTR_BUFFERED_IO=1`、`RAPID_MAPPING_THREADS=16`。buffered 模式的热缓存吞吐不代表冷盘或超过内存的数据集。

## 3. 三轮吞吐对照

单位为全体八卡 samples/s，均值是三轮吞吐的算术平均。HugeCTR 的“不在线映射”在训练之前物化了 RAPID 计划，额外转换时间和空间没有包含在表中。

| 配置 | Kaggle / DCNv2 | Terabyte / DCNv2 风格 | Taobao / DeepFM |
| --- | ---: | ---: | ---: |
| 原始 HugeCTR，全 Reduction | 547,483 | 547,426 | 不作为一热论文主基线 |
| 原始 HugeCTR，全 Unique | 984,456 | 971,924 | 7,172,238 |
| 原始 hash + ACP-only | 1,349,635 | 1,190,481 | 一热情况下等同全 Unique |
| RAPID+HugeCTR，预物化、无在线映射 | 1,313,567 | 1,344,558 | 12,532,845 |
| RAPID+HugeCTR，在线映射 | 1,312,632 | 1,339,762 | 12,503,004 |
| TorchRec | 784,350 | 786,719 | 3,836,770 |
| HET-GMP | 673,230 | 689,474 | 9,840,928 |

逐轮值：

| 配置 | Kaggle | Terabyte | Taobao |
| --- | --- | --- | --- |
| 原始 Reduction | 547,502 / 547,445 / 547,502 | 547,102 / 547,731 / 547,445 | — |
| 原始 Unique | 983,470 / 985,133 / 984,763 | 970,724 / 972,344 / 972,705 | 7,152,633 / 7,172,202 / 7,191,879 |
| hash + ACP-only | 1,348,824 / 1,349,866 / 1,350,214 | 1,190,211 / 1,190,481 / 1,190,752 | — |
| RAPID 预物化 | 1,313,018 / 1,314,664 / 1,313,018 | 1,344,673 / 1,344,328 / 1,344,673 | 12,512,840 / 12,512,840 / 12,572,854 |
| RAPID 在线 | 1,293,580 / 1,315,324 / 1,328,994 | 1,356,853 / 1,324,963 / 1,337,469 | 12,453,397 / 12,512,840 / 12,542,775 |
| TorchRec | 610,109 / 880,139 / 862,800 | 616,860 / 870,414 / 872,883 | 3,842,445 / 3,845,294 / 3,822,571 |
| HET-GMP | 673,883 / 673,753 / 672,052 | 689,753 / 687,898 / 690,771 | 9,836,548 / 9,875,457 / 9,810,778 |

两个 Criteo 的 TorchRec 首轮明显较慢，未删除；Kaggle 追加第四、第五轮为 849,253、872,603 samples/s，五轮平均 814,981、中位数 862,800。10 步预热不足以遍历 16 个 batch，是待核查的可能因素，不足以单凭时间确定原因。额外以 32 步预热各跑一轮，Kaggle 为 874,857、Terabyte 为 850,452 samples/s；这些额外轮次不是主表三轮的一部分。完整结果在归档 JSON 中。

在这些数据中，Kaggle 的 ACP-only（134.96 万）与完整 RAPID（131.26 万）很接近，不能声称 CPS 在该子集有额外吞吐提升；Terabyte 的在线 RAPID（133.98 万）高于 ACP-only（119.05 万），但两者的初始化、表容量和有效数值状态未按逻辑 ID 配对，不能把差值直接作为严格的 CPS 因果效应。两组均超过本次的 TorchRec 数字，但这不是跨系统最终公平排名。

只按上表算术，在线 RAPID 相对论文所选原始 HugeCTR 主基线的吞吐比分别约为 Kaggle 2.40 倍、Terabyte 2.45 倍、Taobao 1.74 倍。论文报告的对应倍数为 3.1、2.4、1.3；本次子集、硬件、模型、初始化及测量边界不同，特别是 Terabyte 模型不同，不能将数值接近解释为论文结果逐值复现。

## 4. ACP 的实际选择与适配

`experiments/acp/export_hugectr_acp.py` 从每表 profile 导出静态 `rapid-hugectr-acp-v1`；最终 `experiments/export_runtime_plan.py` 将 CPS 最终 profile 和物理映射一起导出为 `acp.json`。HugeCTR 入口在创建 embedding collection 之前读取计划，验证 schema、GPU 数、global/local batch、表顺序和 hotness，映射到 HugeCTR 原生 `CompressionStrategy.Unique` 与 `CompressionStrategy.Reduction`。输出记录选择、测得的远端 Unique 数、Reduction 阈值、profile 范围和摘要。另有显式的 `remote` 口径使用 `B×(G−1)`，不能与论文 `B×G` 混称。

- 原始 hash 放置取前 4 个 global batch profile，Kaggle 第 20 表选择 Reduction，其余 Unique；Terabyte 第 20、21 表选择 Reduction，其余 Unique。
- 完整 CPS 后，新下载的三个输入均选择全 Unique。Kaggle 第 20 表前四批平均远端 Unique 数，hash 约 368,911，CPS 后约 35,641；`B×G=65,536`。这是同一子集的固定 profile，不是未来分布保证。完整 CPS 计划使用所有 16 批的 in-sample profile，hash ACP-only 使用前 4 批，两者 profile 范围不同。
- 用户所述 Unique 梯度初始化问题已在 `embedding_collection.cpp` 修复；另修了同类型在 `model_pipeline.cpp` 和通用更新列表中遗漏参数更新的问题。详见 `性能排查与修复报告.md`。
- 实际 GPU 混合策略单元实验：人为指定两张表 Unique、两张表 Reduction，multi-hot 包含重复 ID；八卡运行 3 步，与全 Unique、相同初始化对照的三步 loss 显示值一致，训练后 embedding 最大绝对差 `2.98e-8`。这是短程数值检查，非完整 AUC 验证。真实 Kaggle/Terabyte ACP-only 三轮也实际采用混合配置并完成训练。

## 5. 时间到底花在哪里？

### 数据读取

HugeCTR 默认 RawAsync 以 `O_DIRECT` 打开文件，绕过页缓存。本机对 956,301,312 字节 Kaggle 文件直接读的测量约 138 MB/s；每 global batch 约 59.8 MB，按该带宽估算约 0.43 秒，接近此前默认 direct I/O 的原生 HugeCTR 约 0.42 秒/步。已增加可选 `HUGECTR_BUFFERED_IO=1`，默认 direct I/O 行为保留。这个近似解释不等同每步 I/O 分解，也不适用于所有磁盘和缓存状态。

同步热缓存下，使用 `SKIP_H2D=1` 的定位消融只读取首个 batch，后续复用 GPU 输入：原始全 Reduction 约 118.54 ms/步、全 Unique 约 66.11 ms/步，对应正常训练分别约 119.70、66.57 ms/步。这个开关改变训练输入，不是有效训练 benchmark；它说明当前热缓存运行中，后续读取/H2D 的墙钟影响可以与 GPU 工作重叠，主要差异不能归因于每步磁盘读取。

### 在线调度与映射

`RAPID_PROFILE_HOST=1` 记录 reader 中一批完整 global batch 的 host 阶段墙钟时间。16 线程下，去掉前 16 次观察后：

| 实现 | 缓冲准备与排列校验 | 样本复制与 ID 查找 | 回写 |
| --- | ---: | ---: | ---: |
| 原直接索引实现 | 49.20 ms | 23.45 ms | 16.27 ms |
| 线程局部缓冲复用及并行回写 | 2.74 ms | 21.98 ms | 3.02 ms |

旧实现每批分配并清零约 60 MB，且串行回写；新实现复用每 worker 缓冲并行回写。直接索引上限为 1 GiB/计划实例，过大或不合适的键域回退哈希；需要额外主机内存。线程局部缓冲还会按实际 worker 数占内存。三项为 CPU 阶段时间、带诊断日志，reader 可与 GPU 异步重叠，不能简单相加或直接等同训练步增量。

Kaggle 正式三轮中，在线 49.31–50.66 ms/步，预物化 49.85–49.91 ms/步；该工作负载的预取重叠使在线处理的额外墙钟开销很小，**不代表映射不耗 CPU**。全表文本映射解析、唯一性校验和建立直接索引仍是显著启动成本，早期记录的 reader 到 graph 日志约 67–78 秒，且训练/评估 reader 可能分别加载计划；还没有二进制计划加载或统一总时长对照。

### GPU 计算与通信

对 Kaggle 的原始全 Reduction 和 hash+ACP-only 使用相同 native 输入、正确 Unique 更新，通过 CUDA profiler API 捕获预热后的 20 步：

| 内核类 | 全 Reduction：八卡累计 | ACP-only：八卡累计 |
| --- | ---: | ---: |
| NCCL Send/Recv | 13.196 秒，640 次 | 1.624 秒，1,280 次 |
| FP32 NCCL AllReduce | 1.369 秒，160 次 | 1.370 秒，160 次 |

NCCL kernel 的累计时间跨 GPU/stream 可能重叠且包含等待，不是网络字节数，也不能直接当作 20 步墙钟时间相减；这两组 trace 的相同范围显示全 Reduction 的 Send/Recv 活动明显更多。AllReduce 的实际元素数 5,813,536 与 dense 模型梯度相符，详见 `性能排查与修复报告.md`，不能把它当作 Unique embedding 全表 AllReduce。矩阵乘法等 dense 计算依然占用大量 GPU 活动。

TorchRec 额外进行过同步分段诊断：输入及清梯度约 21.8 ms、前向约 20.7 ms、反向约 33.7 ms、优化器约 2.1 ms；各阶段取 rank 最大值再平均，单独同步会改变执行和重叠，四项不能作为无扰动基准。其 embedding 的抽样权重在各 rank 上均有更新；FBGEMM 发出 autograd 注册警告，仍需严格梯度对照。

## 6. 验证与限制

- 最终运行计划导出用 2 rank 合成 fixture 做了完整记录守恒及 `acp.json` 联合导出测试；ACP 决策边界、并列规则、无效 profile 的单元测试通过。混合 GPU 测试和真实多 hot 运行如第 4 节所述。
- 缓冲复用后的 C++ 计划转换与两批独立物化结果逐字节一致，增量扩展及大 ID/缺失 ID 回退测试通过。缓冲复用之后再次对同一小样本跑在线/物化各 10 步：训练前权重完全一致，训练后约 376 万个值中 5 个有浮点差异，最大误差 `5.82e-11`。
- Criteo 原生 all-Reduction、all-Unique 和 ACP-only 使用同一逻辑顺序且打印 loss 一致；RAPID 预物化与在线的 loss 日志一致。不同放置的初始 embedding 未按逻辑 ID 对齐，跨系统及不同放置的 loss 不可作为等价证明。
- TorchRec 用自己的 planner 和预紧凑 ID；HET 使用原生分区、镜像及可能不同的 global batch；Taobao 的 embedding padding 与实现分量布局也有差异。权重、embedding 容量、缓存预算、输入预处理、计时边界及 optimizer state 未完全统一。
- 原始 HugeCTR、在线 RAPID、TorchRec、HET-GMP 各自独立运行前后，NVLink 各链路 payload 计数器的 Tx/Rx 增量均为零；空闲对照同样为零。计数包括各测试进程启动和训练，未重置全局计数器；此结论限于本次设备、配置和被测运行。NCCL P2P 被禁用，通信经主机内存/PCIe，不等同开启 GPU 间 PCIe P2P。
- 未进行完整 DLRM/Terabyte 的三框架配对训练、完整 AUC/LogLoss、多 seed、冷缓存及大于 RAM 的训练，也未统一测量启动、规划、完整 epoch 和峰值显存。因此本表不应直接填入 response 的严格端到端表格。

## 7. 代码和证据

- 代码：`code/Rapid/experiments/acp/export_hugectr_acp.py`、`code/Rapid/experiments/export_runtime_plan.py`、`code/rapid_acp.py`、`code/hugectr_criteo.py`、`code/hugectr_smoke.py`；HugeCTR host reader 位于 `code/upstream_fresh/HugeCTR/HugeCTR/include/data_readers/multi_hot/detail/rapid_host_plan.hpp`。
- 运行和统计：`code/run_kaggle_controlled.sh`、`code/run_terabyte_controlled.sh`、`code/run_taobao_controlled.sh`、`code/summarize_controlled.py`。这些脚本依赖重新下载的 `/tmp/rapid_fresh_stage` 数据与 `/tmp` 中构建环境；临时目录清理后需重建。
- 日志、ACP plans、profile 和 CUDA traces 位于 `/mnt/nebula/cn-zhangjiakou/juicefs/longjiu.bty/rapid_experiment_data/evidence_20261102/acp_controlled_comparison.tar`；合成测试 fixture 与各 CPS profile 位于同目录 `acp_test_fixtures.tar`。输入数据也使用非 home 文件系统。任何重新下载或生成数据前都需检查目标磁盘余量。
