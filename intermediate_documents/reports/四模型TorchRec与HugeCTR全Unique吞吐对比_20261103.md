# 四模型 TorchRec 与 HugeCTR 全 Unique 吞吐对比（2026-11-03）

## 配置与计时口径

单机 8×V100 32 GiB、每卡 batch 8,192、全局 batch 65,536、embedding dim 128、FP32、SGD lr=0.01，26 张 embedding 表。使用同一份已全局 shuffle 的多日 Criteo 子集，1,048,576 行，16 个 global batch 循环回放；每次训练 510 步，不做评估。HugeCTR 为 26 表 MP、**全部 Unique**；TorchRec 使用其自动规划，四模型均为 18 DP + 7 table-wise + 表 20 row-wise。`NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`；本机卡间 NCCL 日志此前核对为 SHM/direct/direct。此次未使用 `Rapid/store`，未写入新数据集。

两框架都使用原生 GPU 上的前向、反向和 SGD 更新。HugeCTR 使用多热 RawAsync reader，TorchRec 使用预格式化的锁页 host batch 与异步 H2D 预取。吞吐为完整训练循环的墙钟：`510 × 65,536 / 秒数`。HugeCTR 以 `FIT_SECONDS`、TorchRec 以 `loop_wall_seconds` 计算；不把 HugeCTR 内部的 `Time(100 iters)` 与 TorchRec 完整循环混用。两边的输入准备、模型构建和 HugeCTR 编译都不计入此训练吞吐。

模型结构：DLRM 用 13→512→256→128 的 bottom MLP、26 表与 dense 向量的二阶交互、1024→512→256→1 的 top MLP；DCNv2 沿用此前共同配置（bottom 13→512→256→128、3 层 rank 512 cross、top 1024→1024→512→256→1）；Wide&Deep 用池化 embedding 与 dense 拼接后分别进入 1024→512→256→1 的 deep 分支及线性 wide 分支；DeepFM 在同样的 deep 与线性分支外加 26 个 128 维 field 的二阶 FM 项。Wide 分支为该对照的简化实现，不另设原始 sparse ID 的独立一维表。四个模型在两边使用相同的表容量、hotness、维度、分支宽度和 batch；dense 初始参数及底层算子没有逐参数配对，因此这组数据用于吞吐对照，不构成严格数值等价或 AUC 结论。

## 完整训练循环结果

| 模型 | HugeCTR 全 Unique：510 步 / samples/s | TorchRec AutoShard：510 步 / samples/s | TorchRec / HugeCTR 吞吐 |
| --- | ---: | ---: | ---: |
| DLRM | 69.235 / 69.233 s；482,755 / 482,768 | 33.264 / 33.225 s；1,004,784 / 1,005,972 | 2.08× |
| DCNv2 | 103.925 s；321,611 | 65.866 / 65.931 s；507,444 / 506,945 | 1.58× |
| Wide&Deep | 72.390 s；461,711 | 38.506 s；867,997 | 1.88× |
| DeepFM | 73.648 s；453,823 | 39.479 s；846,618 | 1.87× |

DLRM 两边各重复两次，DCNv2 的 TorchRec 重复两次；其余为一次完整 510 步测量。上表比值取已有重复轮次均值。不同模型的模型计算量不同，比较应在同一行内进行；不把跨模型绝对吞吐当作框架优劣证据。

## 读取、显存与差距解释

HugeCTR 训练区间 `FIT_READ_BYTES`：DLRM 两轮 495,616 / 0 字节、DCNv2 0、Wide&Deep 8,192、DeepFM 20,480。TorchRec 三个新增模型各 rank 的训练区间 `read_bytes` 最大值依次约为 5.17 MB、0、1.09 MB，远小于循环回放数据量，且输入在计时前已预格式化。HugeCTR RawAsync 仍会在训练区间从 `/dev/shm` 读取，`read_bytes` 近零只能排除实质磁盘读取；本轮没有单独测其 CPU reader 的服务上限，因此不能宣称已严格证明所有模型都完全不受数据管线限制。

新增模型运行时 HugeCTR DLRM `nvidia-smi` 采样约 17.1 GiB/卡；TorchRec 三个新增模型的 rank 峰值 `max_memory_allocated` 分别最高约 3.97、4.02、4.18 GiB。两种显存统计口径不同，不用这些数直接计算显存节约比例。128 维全部成功运行，无须降维。

已有相同数据、batch 和 26 表全 Unique 的 DCNv2 Nsight 诊断显示，HugeCTR 每卡每步 NCCL kernel 忙碌约 124.43 ms、其他 kernel 77.92 ms，二者重叠为 0；TorchRec 对应约 75.14 / 75.31 ms，重叠约 22.86 ms。HugeCTR 全 Unique 的远端 send payload 约 487.79 MB/卡/步，TorchRec 自动计划的 NCCL 点对点 API payload 约 61.56 MB/卡/步，协议与表放置不同，逻辑 payload 不是物理 PCIe 字节。新增的 DLRM/Wide&Deep/DeepFM 没有逐模型 Nsight 时间线；它们的 HugeCTR 每步约 136–144 ms，接近 DCNv2 的通信量级，提示固定的全 Unique 通信路径可能限制不同模型的吞吐，但这是待进一步逐模型验证的解释，不应把全部差距都归到通信。

## Taobao / DeepFM 追加一次对照

按同一套 8 卡、每卡 batch 8,192、完整 510 步训练墙钟口径，使用已有的全局 shuffle Taobao 子集 1,048,576 行、4 张单 hot 表、1 个 dense 字段。HugeCTR 使用 `--unique --deepfm`，TorchRec 使用自动规划；两边均为 SGD lr=0.001、DeepFM 主交互维度 128、表物理宽度 132（128 维交互、1 维线性及 padding）。HugeCTR 的 `native_compact.bin` 从 TorchRec 所用同一 `input.bin` 按表减去 ID offset 得到，保持样本顺序。没有下载或写入新数据集。

| Taobao / DeepFM | 完整 510 步墙钟 | 每步平均 | 吞吐 |
| --- | ---: | ---: | ---: |
| HugeCTR，4 表全 Unique | **4.820 s** | **9.451 ms** | **6,934,216 samples/s** |
| TorchRec，4 表 table-wise | 9.301 s | 18.238 ms | 3,593,446 samples/s |

HugeCTR 为 TorchRec 的 **1.93×**。这里只做了一次有效测量，不提供方差或置信区间。HugeCTR 首轮训练区间曾发生 25,219,072 字节物理读取，故先读取现有输入文件预热页缓存，再重跑上表有效轮次；该轮 `FIT_READ_BYTES=0`。TorchRec 在计时前完成输入加载与锁页预格式化，训练区间各 rank 的 `read_bytes` 最大约 1.55 MB。两框架均没有显示实质磁盘瓶颈，但尚未单独测 CPU reader 的最大服务率。TorchRec 最高 rank 的 `max_memory_allocated` 为 317 MiB；HugeCTR 未采集同口径峰值，故不比较显存节约比例。

Taobao 仅 4 张单 hot 表，TorchRec planner 选择 4 表 table-wise；Criteo 是 26 张多热表，planner 为 18 DP、7 table-wise、1 row-wise。因此 Taobao 的 HugeCTR 优势与 Criteo 的 TorchRec 优势不矛盾；表数、hotness、放置和模型算子成本均不同，不能仅用数据集名称推断原因。Taobao 两框架未逐参数配对或测 AUC，本结果只回答吞吐。原始记录：`/tmp/rapid_taobao_deepfm_hugectr_unique_b8192_d128_510_cached.log`、`/tmp/rapid_taobao_deepfm_torchrec_b8192_d128_510.json`、同名 `.log` 与 `_plan.json`；首轮预热诊断保留在 `...hugectr_unique_b8192_d128_510.log`。

## 代码与记录

- 两框架的模型选择分别在 `code/hugectr_criteo.py`、`code/torchrec_criteo.py` 的 `--model` 参数中；`python -m py_compile` 均通过。HugeCTR 多卡全 Unique 继续使用由固定数据键数上界审计得到的 `DENSE_UNIQUE_RATIO=WGRAD_UNIQUE_RATIO=0.25`，避免默认缓冲在 8,192 batch 下编译 OOM；该容量系数仅适用于本固定数据。
- 新增模型的 HugeCTR 日志：`/tmp/rapid_criteo_multiday_20261103/model_compare_hugectr_{dlrm,wide_deep,deepfm}_510.log`；DLRM 复测追加 `_repeat.log`。TorchRec 同前缀 `model_compare_torchrec_{dlrm,wide_deep,deepfm}_510.json`、`.log`、`_plan.json`；DLRM 复测追加 `_repeat.json`、`_repeat.log`。
- DCNv2 既有日志：同目录 `hugectr_native_b8192_dense025_wgrad025_510.log`、`torchrec_b8192_510.json`、`torchrec_b8192_510_repeat.json`；时间线与通信详见 `本地Batch8192显存与吞吐复测_20261103.md`。
