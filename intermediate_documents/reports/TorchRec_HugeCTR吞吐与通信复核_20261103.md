# TorchRec 与 HugeCTR 吞吐、表放置和通信复核（2026-11-03）

## 撤回先前的归因

前一份《多日吞吐排序与计时口径排查_20261103.md》正确指出旧报告把 HugeCTR 内部块计时与 TorchRec 墙钟混用，但把新排序主要解释为“GPU AllReduce 调度差异”，证据不足。此前我没有导出 TorchRec 自动 planner 的**实际逐表策略**，也把 Nsight 中单个 AllReduce kernel 的平均时长当成跨实现通信成本。随后第一次 NCCL 调用统计还把发送给本卡的 `peer==rank` 条目算成了跨卡通信。这三处判断方法错误已通过下面的对照和过滤修正。

本报告把三类数字分开：无 profiler 的完整训练循环吞吐；NCCL API 的逻辑 payload（按 communicator rank 去掉本卡 peer）；Nsight 的 30 步 GPU 时间线。NCCL API 逻辑字节数不是实际链路字节数：AllReduce 只计输入张量长度，不含 ring 算法放大；`send/recv` 每方向单列，也不含底层协议包头。

## 数据、模型和计划

- 8 张 V100 32 GiB，FP32，26 表、128 维、全局 batch 16,384，SGD lr=0.01，embedding 零初始化，510 步不做评估；禁用 NCCL P2P/IB，训练输入驻留 `/dev/shm`。TorchRec 逐表容量与原生 HugeCTR 的 `native_sizes.json` **完全相同**，总共 22,294,910 行。RAPID 改变物理 ID 和拥有者，逐条审计了原始记录。两框架的 dense 参数初值与部分算子尚未逐值配对，不能把 loss 近似当成严格数值等价。
- 预处理从三天分片中无放回抽样，再对整个 1,048,576 行训练文件 shuffle；HugeCTR reader 的 `shuffle=OFF` 仅表示读取时不二次打乱。原始与 `/dev/shm` 训练文件 SHA-256 均为 `ff5fd3c715bc222041d94c136844ee08749b14aa7cfda74f3767c1c786d42c6d`。训练窗口没有大规模物理磁盘读取；TorchRec 的 510 步默认计划各 rank `read_bytes=0`。本结论限于内存驻留的子集循环训练。
- TorchRec 默认 planner **13 张 data-parallel 小表**：3、5、6、7、8、12、13、15、16、17、18、24、25，权重共 13,850 行、7.0912 MB；另 13 张大表 table-wise。之前只看配置代码，没有确认实际计划，这是跨框架配置比较中的主要遗漏。强制 `--no-data-parallel` 后 26 表均为 table-wise。原先的 HugeCTR/RAPID 配置是 26 表全 MP，并非相同分片策略。

## 实际吞吐

| 方法与表策略 | 完整 510 步训练循环 samples/s | 另一个稳态墙钟口径 | 说明 |
| --- | ---: | ---: | --- |
| TorchRec 自动计划：13 DP + 13 table-wise | **324,349 / 325,784**（两轮） | 首轮逐 step 平均 324,508 | 先前排名中的 TorchRec |
| TorchRec 禁用 DP：26 table-wise | **300,113 / 304,219**（两轮） | 首轮逐 step 平均 299,811 | 同一数据/模型代码/学习率；末步 loss 与默认计划相差约 1e-6 |
| RAPID+HugeCTR：26 MP，在线映射 | `FIT_SECONDS` 换算约 **297,167** | 第 200–500 步墙钟 **299,269** | 在线映射初始化另约 73.53 秒 |
| RAPID+HugeCTR：26 MP，预物化 | `FIT_SECONDS` 换算约 **298,036** | 第 200–500 步墙钟 **299,342** | 与在线映射打印的 loss 一致 |
| 原生 HugeCTR：移植 TorchRec 的 13 DP + 13 table-wise 计划 | `FIT_SECONDS` 换算约 **132,378** | 第 200–500 步墙钟 **132,442** | 表放置相同，但 EBC 通信协议不同 |
| 原生 HugeCTR：26 MP | **210,293** | 第 200–500 步墙钟 **210,790** | 与 RAPID 同一紧凑 ID 样本；本轮 `FIT_READ_BYTES=0` |

上述长窗口均不带 NCCL 追踪；不能用 30 步追踪进程的耗时给无扰动吞吐排名。TorchRec 自动 planner 的小表 DP 在两轮中分别比全 table-wise 快约 **8.1% 和 7.1%**。关掉 DP 后它与 RAPID+HugeCTR 的完整训练循环吞吐接近；两轮全 table-wise 自身相差约 1.4%，所以不能据两者接近 300k 的单轮差值下严格排名。这也不能证明两个方法完全等价，因为一个是 table-wise，一个是经过 CPS 放置的 MP。

## NCCL 通信对照

使用 `code/trace_nccl_payload.c` 拦截 NCCL API，在进入 `fit`/训练循环前清零，取前 30 步；`ncclCommUserRank` 排除 `peer==rank`，8 卡总量除以 `8 × 30`。`send`、`recv` 是单方向的**每卡每步逻辑 payload**，AllReduce 一栏是每卡每步输入张量长度；它们不能直接相加称为物理网络流量。

| 方法与表策略 | 远端 send MB/卡/步 | 远端 recv MB/卡/步 | AllReduce 输入 MB/卡/步 |
| --- | ---: | ---: | ---: |
| 原生 HugeCTR，26 MP | **154.07** | **154.07** | **64.18** |
| RAPID+HugeCTR，26 MP | **26.56** | **26.56** | **64.18** |
| TorchRec，13 DP + 13 table-wise | **26.69** | **26.69** | **71.27** |
| TorchRec，26 table-wise | **50.96** | **50.96** | **64.18** |
| 原生 HugeCTR，照搬 TorchRec 的 13 DP + 13 table-wise | **151.57** | **151.57** | **71.27** |

两个实现的公共 dense AllReduce 输入几乎逐字节相同，约 64.18 MB/卡/步；13 张复制小表带来的额外梯度 AllReduce 约 **7.09 MB/卡/步**。因此全 Unique 仍有 AllReduce，因为 dense 梯度本来就要同步。RAPID 与 TorchRec 默认计划在这个 30 步窗口的远端 NCCL `send/recv` **逻辑 payload** 几乎相等；该计数包含路由数据，不能称作纯 embedding 向量字节。TorchRec 并非凭更少的总 `send/recv` 字节取胜。

原生 HugeCTR 照搬 TorchRec 的放置后仍慢，因为两者同名的 table-wise 放置**没有相同通信协议**。HugeCTR EBC 的 `DenseUniformModelParallelEmbedding::network_forward()` 先将 lookup 向量通过 `dense_communicate()` 发送，再在网络侧汇聚；TorchRec `TwPooledEmbeddingSharding` 分发的是 pooled embedding 张量。对 hotness 很高的表，前者可以发送远多于“一样本一表一向量”的数据；实测远端 `send` 为 151.57 对 26.69 MB/卡/步。HugeCTR 的 CPS 物理 ID 放置和样本调度把自己的远端发送从 154.07 降到 26.56 MB，证实 RAPID 在这组数据上确实有效。RAPID 离线回放约 26,976 个远端**前向接收**向量/卡/步，前后向合计约 53,951 个，折合约 27.62 MB；实测远端 `recv` 总逻辑 payload 为 26.56 MB/卡/步，两者量级接近，但一个只计 embedding，另一个还含路由等 NCCL 调用，且窗口分别是离线计划全窗口与训练前 30 步，不能当作逐字节相等的校验。

直接读取 RAPID 预物化训练文件的第一个 global batch，并按物理 ID `% 8` 还原拥有者，平均每卡有约 **236,502 个远端 ID 出现次数**，但只涉及约 **24,530 个远端 `(表,ID)`**。这说明当前 multi-hot 输入中重复访问很多，通信统计必须区分出现次数、Unique 向量数和池化后样本输出数；拿任何一个数字代称另一个都会误判。

**计数错误说明：**第一次追踪未过滤 `peer==rank`，RAPID 的 `send` 得到约 139.46 MB/卡/步；其中约 **112.90 MB/卡/步** 是发给本卡的逻辑调用。按 communicator rank 过滤后跨卡值是 26.56 MB。这不是证据显示 RAPID 偷偷产生了五倍网络流量；未过滤的值不得用于通信比较。

## GPU 时间线：相近 payload 下的剩余差距

已有 Nsight `fit` 内 30 步 trace，逐卡取均值：

| 方法 | NCCL kernel 时间/步 | 其他 kernel 时间/步 | NCCL 与其他 kernel 重叠/步 | GPU kernel 忙碌并集/步 |
| --- | ---: | ---: | ---: | ---: |
| RAPID+HugeCTR | 33.34 ms | 20.78 ms | **0.00 ms** | **54.11 ms** |
| TorchRec 默认计划 | 35.39 ms | 23.87 ms | **11.39 ms** | **47.86 ms** |

TorchRec 的 NCCL kernel **总时间并没有更短**；单个 AllReduce kernel 平均 7 ms 对 HugeCTR 约 23.6 ms 的对比忽略了 TorchRec 每步多个调用。更准确的观察是它把一部分通信和其他 kernel 放在两条 stream 上重叠，而 HugeCTR 这组运行只观察到一条执行 stream。该并行使 GPU 忙碌时间并集减少约 6.25 ms/步，与无 profiler 长窗口的每步差距量级一致。这是定位证据，不是证明单独改变 stream 就能获得同样加速；TorchRec/HugeCTR 的 kernel、调度和模型细节仍不完全相同。HugeCTR 的 `--intra-iteration-overlap` 只改变 embedding 路径，一轮墙钟提升约 0.5%，并未消除差距。

还单独打开了 HugeCTR 的 `use_cuda_graph`，保持 RAPID 预物化输入和其余设置不变：510 步 `FIT_SECONDS` 从 **28.036** 增至 **28.356** 秒，稳态第 200–500 步墙钟约 **299,488** samples/s（原配置 299,342），打印的五个 loss 检查点一致。CUDA Graph 在这组配置下没有消除吞吐差距；这是单轮消融，约 0.05% 的稳态差属于测量波动。

## 复核材料与限制

- TorchRec 计划：`/tmp/rapid_criteo_multiday_20261103/torchrec_dp_plan_audit.json`、`torchrec_mp_plan_audit.json`；510 步结果：`torchrec_ram_510_noeval_wall.json`、`torchrec_dp_510_repeat.json`、`torchrec_mp_510_audit.json`、`torchrec_mp_510_repeat.json`。
- HugeCTR 510 步：`hugectr_rapid_ram_510_noeval.log`、`hugectr_materialized_ram_510_noeval.log`、`hugectr_rapid_materialized_cuda_graph_510.log`、`hugectr_native_510_noeval.log`、`hugectr_native_torchrec_plan_510.log`；30 步 NCCL 日志在同目录，以 `*_30_nccl_remote.log` 结尾。
- GPU trace：`nsys_hugectr_fit_30.sqlite`、`nsys_torchrec_fit_30.sqlite`。新的可选 `--placement-plan` 仅用于 HugeCTR 控制实验；默认入口仍保持原 26 MP 策略。
- 没有读取 `Rapid/store`。没有写新数据集；已有数据位于 `/tmp` 和 `/dev/shm`。所有这里的结论限于这份 Criteo 多日分片子集、当前网络、模型和单机 8 卡。正式 Response 表格中的完整模型/数据集矩阵仍不能据此填满。
