# TorchRec 纯 Row-wise 分片约束与实测（2026-11-03）

## 实现

在 `code/torchrec_criteo.py` 和 `code/torchrec_deepfm.py` 增加 `--row-wise-only`。该开关给 TorchRec `EmbeddingShardingPlanner` 中**每张** embedding 表设置 `ParameterConstraints(sharding_types=['row_wise'])`，规划完成后验证表集合完整、每表确实为 `row_wise`、每表 ranks 覆盖全部 8 张卡；否则直接报错。默认不带开关时保留原自动混合计划。约束只改变 embedding 表的分片类型，dense 网络仍按原训练方式运行；它没有改变 TorchRec 的行分发、聚合、通信算子或优化器逻辑，也不是 RAPID 的 CPS/ACP 实现。

用法：在此前 TorchRec 训练命令中追加 `--row-wise-only`，例如 `torchrun --standalone --nproc-per-node=8 code/torchrec_criteo.py ... --row-wise-only --dump-plan /tmp/rowwise_plan.json`；Taobao 的入口是 `code/torchrec_deepfm.py`。`--dump-plan` 文件可逐表审计。Criteo 有 5 张极小词表在部分 rank 出现 0 行 shard；TorchRec 1.1.0 在本次 8 卡训练中正确执行，没有必要为这些表额外扩容或改变 ID 空间。

## 8 卡完整训练吞吐

沿用同一份已全局 shuffle 的 1,048,576 行输入，各 510 步、不做评估、每卡 batch 8,192、FP32、`NCCL_P2P_DISABLE=1`。Criteo 为 26 表多热、dim 128、DCNv2；Taobao 为 4 表单热、DeepFM 主交互 dim 128（物理表宽 132）。吞吐按完整训练循环墙钟 `510 × 65,536 / 秒数`，不含输入预格式化与模型初始化。没有读取 `Rapid/store`，也没有新写数据集。

| 数据集/模型 | TorchRec 原 AutoShard | TorchRec 纯 Row-wise | 原生 HugeCTR 全 Unique（参考） |
| --- | ---: | ---: | ---: |
| 多日 Criteo 子集 / DCNv2 | 507,444 / 506,945 samples/s（两轮） | **151,079 samples/s**；221.230 s / 510 步 | 321,611 samples/s |
| Taobao 子集 / DeepFM | 3,593,446 samples/s | **1,027,816 samples/s**；32.519 s / 510 步 | 6,934,216 samples/s |

纯 Row-wise 相比 TorchRec 原计划，Criteo 吞吐约为 **29.8%**，Taobao 约为 **28.6%**。本次纯 Row-wise 各跑一次，原 Criteo AutoShard 有两轮、其余参照为一次。两边数据和 batch 一致，但 HugeCTR 全 Unique 与 TorchRec Row-wise 使用各自原生通信协议；表中 HugeCTR 仅作定位参照，不能把差值直接解释为 RAPID 可获得的加速。

计划核验：Criteo **26/26 row-wise**，Taobao **4/4 row-wise**；每张表均列出 ranks 0–7，没有 data parallel 或 table-wise。Criteo 原计划是 18 DP、7 table-wise、1 row-wise；Taobao 原计划是 4 table-wise。纯 Row-wise 的训练区间 `/proc/self/io` `read_bytes` 在两组实验各 rank 均为 **0**，输入先预格式化到锁页 host 内存并异步 H2D；这排除了实质磁盘读取，但不能单靠 `read_bytes` 证明 CPU 与 H2D 完全不影响步骤时间。Criteo Row-wise 的最大 rank PyTorch 已分配显存约 2.60 GiB，Taobao 约 231 MiB，未触发 32 GiB 显存限制。

同种 TorchRec 模型间的训练数值抽查：Criteo AutoShard 和 Row-wise 首步 rank0 loss 都是 0.6753169894，末步分别为 0.1833064854 和 0.1833113134；Taobao 首步分别为 0.7044942379 和 0.7045191526，末步为 0.5735188723 和 0.5735125542。Taobao 两种计划的 embedding 随机初始化未逐逻辑 ID 对齐，因此这些近似值仅用于检查训练正常，不能证明严格等价。两组都没有本轮 AUC 测试。

## 如何解释排序

纯 Row-wise 将原先保持本地的 DP 表和整表放置的 table-wise 表全部改成按行跨 8 卡切分。它符合“只允许按 embedding 行切分”的比较约束，但会改变输入 ID 路由与 embedding 输出聚合的通信结构。当前观测显示这种强制约束在两份数据上显著降低 TorchRec 吞吐。已有 DCNv2 的 AutoShard/原生 HugeCTR 时间线不能直接解释纯 Row-wise 的全部差距；若需准确拆分原因，应对纯 Row-wise 另采 NCCL payload 与 GPU 时间线。不要把原 AutoShard 的通信量或性能当作这轮纯 Row-wise 的值。

## 原始记录

- Criteo 纯 Row-wise：`/tmp/rapid_criteo_multiday_20261103/torchrec_rowwise_dcnv2_510.json`、同名 `.log`、`torchrec_rowwise_dcnv2_510_plan.json`。2 步短训与其计划文件同目录，名称带 `_smoke`。
- Taobao 纯 Row-wise：`/tmp/rapid_taobao_deepfm_torchrec_rowwise_b8192_d128_510.json`、同名 `.log`、`/tmp/rapid_taobao_deepfm_torchrec_rowwise_b8192_d128_plan.json`。
- 原 AutoShard 和 HugeCTR 全 Unique 的日志索引见 `四模型TorchRec与HugeCTR全Unique吞吐对比_20261103.md`。两份 TorchRec 入口均已通过 `python -m py_compile`，并在 8 卡真实 GPU 完成前向、反向和更新。
