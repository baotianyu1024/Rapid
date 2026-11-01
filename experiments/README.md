# 实验代码

推荐入口均从仓库根目录执行，输出目录必须新建。源码和配置不依赖本机数据；数据与依赖库需要自行准备。默认方法名称统一为 RAPID。

## 1. 通信量、存储均衡与 batch size

`../run_all.sh` 对 Criteo、Kaggle、Taobao 和自定义表执行当前离线 RAPID。分别设置 `--ranks 8`、`--ranks 64`；使用不同 `--batch-size` 时必须控制总样本数，并为每种 batch size 重新计算其自身的 Unique 对照，不能复用另一 batch size 的数值。

纯 Unique 的计数入口为 `reproduction/scripts/measure.py`，`--help` 列出参数；计数使用未重排输入和 ID % rank 放置。RAPID 最终汇总为 `communication.json`，存储统计为 `optimized/peak_samples/storage.json`。历史 batch/order/参数敏感性脚本在 `archive/revision_v3/sensitivity_*.py`，以及 `archive/revision_v2/batch_unique.py`。历史参数并非适用于所有输入的性能保证。

输入顺序实验应在开始离线规划前对全局 batch 顺序/数据顺序建立明确的对照；训练调度阶段不允许跨原始全局 batch 移动 sample。要保存随机种子、行数、batch size、rank 数与输入哈希。

## 2. GMP / Herald / TorchRec / BagPipe

```bash
python scripts/fetch_dependencies.py HET-GMP Herald BagPipe torchrec-v1.1.0
python -m pip install pybind11==2.13.6
python experiments/build_baselines.py --sources sources
python experiments/run_baselines.py --case outputs/day6_8 \
  --methods gmp herald --sources sources --epochs 5
```

本次小规模比较验证使用独立 CPU 环境：PyTorch 2.6.0+cpu、FBGEMM-GPU 1.1.0+cpu、NumPy 2.2.6，以及锁定的 TorchRec v1.1.0 源码。不要用此环境替换核心离线流程的 NumPy 1.26.4 环境。额外配置好该环境后：

```bash
python experiments/run_baselines.py --case outputs/day6_8 \
  --methods torchrec bagpipe --sources sources --epochs 5 --lookahead 8
```

`run_baselines.py` 使用该 case 已生成的同一个 `data/`，输出到 `case/baselines/`。它不覆盖已有方法目录。CPU 核心环境只安装 NumPy，不自动安装可能与训练环境冲突的 PyTorch/FBGEMM 版本。TorchRec 和 BagPipe 须使用另行配置好的比较环境。

| 方法 | 执行内容与输出 |
|---|---|
| HET-GMP | 原生 `refine_data` / `refine_embed` 分区，无 batch 约束替代；strict s=0 缓存/梯度回放。`gmp/traffic/het_gmp_s0.json` 与 NPZ 保存方向计数。表 vocabulary 以 active IDs 紧凑映射。 |
| Herald | 官方 LAIA 调度、LRU 缓存和 CPU harness；`herald/result.json`、逐步 CSV、最终 pending flush。用于 sorted unique set 的 `boost::flat_set` 替换为 `std::set`，未改变评分与 tie-breaking。 |
| TorchRec | 官方自动分片 planner，meta tensors，逻辑单机拓扑；导出 `torchrec/plan.json`，按实际计划计算 `protocol_bytes.json`。HBM=16 GiB/rank，dim=128，SGD 无 momentum，dense 参数是存储保留代理，并非实际 forward。 |
| BagPipe | 执行官方 OracleCacher 的预取/依赖逻辑；计入 PS 拉取、写回、梯度同步和终端 flush，`bagpipe/result.json`；ring all-reduce 是显式字节模型，未执行 RPC/网络。 |

GMP 原生分区可能改变全局 batch 的组成，这与 RAPID 的 batch 保持约束不同。缓存、同步语义与 active-ID 工作集必须随数值一同披露。方法的总量不能直接解释为单 rank 峰值，也不能直接证明训练速度或收敛性。

`dependencies.json` 锁定官方来源。`het_gmp_safety.patch` 扩大 soft-count lookup 的边界，使 multi-hot 重复访问的度数不会越界；它不是分区启发式替代。Herald 的最后一版逐表字节插桩源码另见 `archive/official_tseries/` 的驱动和下述来源说明；统一 dim=128 的比较可直接用原始向量数 ×512。

## 3. 官方 T180 / T200 / T510

```bash
python scripts/fetch_dependencies.py HugeCTR-official
python experiments/datasets/generate_official.py t200 \
  --source sources/HugeCTR-official --out outputs/official_t200
```

生成器固定到 `61b4d539aa7044e9021896f0485be1a6697f4f44`；官方表数、hotness、vocabulary、Zipf alpha 不变；NumPy seed=20261027。T180 只修正上游输出变量名拼写。每次生成官方 part0，65,536 行，是 100 个官方分片中的一个，不能称为完整 100-part 数据集。T180 名称实际包含 177 张表。

manifest 位于 `outputs/official_t200/datasets/t200.json`，包含输入路径、每表 hotness、官方维度和哈希。将其中 `lengths` 作为 `run_all.sh --dataset custom --dense 13 --table-lengths ...`，local batch=512；8 rank 时 `--batches 16`，64 rank 时 `--batches 2`。统一比较采用 dim=128；官方 per-table 维度的补充统计须独立列出，不可混用。官方三种输入单个分片约占 3.53 / 1.11 / 0.66 GiB，并会生成更大的阶段副本。

## 4. 多轮增量，无周期微调

```bash
python experiments/run_incremental.py --input /data/criteo/day6.bin \
  --out outputs/incremental_8 --ranks 8 --batch-size 8192 \
  --initial-batches 32 --window-batches 1 --windows 32
```

先用 D0 构建初始状态，再按不相交的连续样本切片读取后续窗口。D0 是初始 planner，不等同于离线定价基线。一个窗口包含指定数量的完整全局 batch。宽 profile 支持最多 4,096 个 batch/窗口。

每轮执行发现新 ID / 新 hot ID、容量约束下放置新 ID、batch 内调度 sample，然后可在本轮训练前重新分配**本轮新 ID**并再次调度。不会迁移上轮已有 emb；旧 physical ID 同样固定。没有周期微调、权重迁移或旧样本重调度。每轮独立扫描原始记录复核通信计数和每 rank 样本容量。

`summary.json` 保存 D0 和每轮通信、存储、新增 emb 数与零迁移验证；`final_state/` 保存最终所有权。ACP 在 D0 上选择并固定。对 shuffled day6 的切片只代表数据量增加，不代表真实时间漂移。历史 frozen / full union+shuffle / pure Unique / 稳定性判定实验代码分别在 `archive/revision_v2/long_incremental.py`、`extend_incremental.py`、`finish_incremental_parallel.py`；这些历史调度脚本依赖原目录，需按注释恢复输入和阶段文件后运行。历史微调代码在 `reproduction/rapid_peak_optimization/stream_tune*.py` 等，仅保留追踪用途。

## 5. 全量冷 ID MCR

```bash
bash experiments/run_mcr.sh /data/criteo/day6.bin outputs/mcr
```

此入口复现既定 2,097,152 行、26 表、214 sparse slots 的 Criteo 实验。对每条样本内同表 ID 去重，再统计频率；冷 ID 阈值为出现次数 `<210`。对所有出现至少 10 次的冷 ID 计算最大共现比，报告支持数 10、20、50、100 下的结果。正式统计不抽样 ID；Python 审计会抽取少量目标独立复核 C++ 结果，抽样只用于实现检查。

该全量枚举可能使用较大的 CPU 内存与磁盘；不同记录格式不得直接套用本固定布局 kernel。完整源码在 `mcr/`。

## 6. 历史归档与训练

`archive/revision_v3/` 保存多数据集、参数敏感性、MCR、存储核对、报告生成和证据封存脚本；`archive/official_tseries/` 保存最终官方数据实验驱动；`archive/revision_v2/` 保存长期增量与宽 profile 实现。`reproduction/` 保存底层优化器与独立小测试。归档脚本中的个人工作区路径只是历史路径，不应被当作新机器的默认入口。

训练模板在根目录 `training/`。本发布不会把 CPU 回放说成 GPU 实测，也不包含重新填写的训练结果。正式端到端实验需要同数据划分、模型、优化器、全局 batch、epoch、seed 与评估周期，独立报告 AUC/loss/吞吐/显存。原模板的评估集设置需先改正。
