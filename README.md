# RAPID

RAPID 的当前离线基线、数据预处理、训练模板及通信量实验代码。本次整理以最后一轮审稿实验实际调用的源码为准，包含 owner 定价优化、存储容量修复、batch 内样本交换和峰值通信优化。`run_all.sh` 默认运行这一完整流程；旧 CPS 单阶段入口保留为 `run_cps.sh`。

**范围说明：** CPU 实验计算逻辑通信量；8 卡训练脚本与所用的五套比较框架源码现收录在 `revision/`。数据集、训练结果、二进制库和本机凭据不入库。请从 [revision 逐步复现指南](revision/README.md) 开始；其中区分可参数化入口与依赖原实验环境路径的 GPU 启动脚本。

## 目录

| 路径 | 内容 |
|---|---|
| `cps/` | hot ID 提取、emb 放置、batch 内 sample 调度与训练数据导出 |
| `acp/` | 每表 Unique/Reduction 通信策略选择 |
| `data_preparation/` | Criteo multi-hot、Taobao 原始数据处理 |
| `analysis/`, `evaluation/`, `plotting/` | 统计、消融和绘图工具 |
| `training/` | 原有 HugeCTR DLRM 训练模板 |
| `scripts/run_offline.py` | 当前 RAPID 离线默认入口，可指定输入和新输出目录 |
| `experiments/` | 独立实验目录：比较方法、长期增量、MCR 和历史实验源码 |
| `docs/source_manifest.json` | 打包前来源文件与 SHA-256，便于核对代码版本 |
| `docs/release_changes.json` | 打包后的新增文件、路径适配与文件校验记录 |
| `revision/benchmarks/` | 实际使用的 GPU 训练、基线、通信及审稿结果驱动脚本 |
| `revision/frameworks/` | HugeCTR、TorchRec、HET-GMP、Herald、BagPipe 的实验源码快照 |

## 安装与编译

Linux，Python 3.10+，支持 C++17/OpenMP 的 GCC（建议 GCC 11+），GNU `time`。CPU 回放核心只依赖 NumPy；GMP/Herald 的 Python 扩展另外需要 pybind11。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build.py --jobs 4
```

可通过 `CXX=/path/to/g++` 选择编译器。`RAPID_PYTHON` 指定 shell 入口使用的 Python。使用自装 GCC 时，应保证其运行时库可以被动态链接器找到。仓库不携带本地预编译 `.so`。

## 当前 RAPID 离线流程

输入必须是已准备好的固定宽度 binary，不能直接传 CSV/Parquet。需要一个尚不存在的输出目录：

```bash
bash run_all.sh --input /data/criteo/day6.bin \
  --dataset criteo --ranks 8 --batch-size 8192 --batches 16 \
  --out outputs/day6_8

bash run_all.sh --input /data/criteo/day6.bin \
  --dataset criteo --ranks 64 --batch-size 8192 --batches 4 \
  --out outputs/day6_64
```

Kaggle 使用 `--dataset kaggle`，Taobao 四表使用 `--dataset taobao`。自定义表使用 `--dataset custom --dense D --table-lengths L0 L1 ...`。`--plan-only` 只打印各阶段命令。`--batches` 是**全局 batch 数**，读取样本数为 `ranks × batch-size × batches`；当前离线 profile kernel 支持 1–64 个全局 batch，禁止静默截断。

执行顺序与最终审稿实验保持一致：hot ID / 初始放置与调度 → owner base → 四阶段定价 → symmetric → samples4 → cold repair → samples16 → batch soft → hot guard → hot budget → alternate2 → peak samples。最终目录为 `optimized/peak_samples/`。默认不添加额外镜像缓存，也不运行周期性 emb 微调。

emb 的离线统计可跨 batch；sample 的交换始终限定在原始全局 batch 内，每个 rank 每步恰好分到 `batch-size` 个样本。该约束由 kernel 和导出校验共同检查。

输出：

- `communication.json`：dim=128、FP32、5 epoch 的逻辑通信量；平均、逐步峰值、绝对峰值及峰均比。
- `optimized/peak_samples/scheduled/dataset_<rank>.bin`：最终各 rank 数据。
- `optimized/peak_samples/storage.json`：主 emb 数和存储峰均比。
- `cps_counts.npz`、各阶段日志、命令、源文件哈希：用于复核统计。

输出可能远大于输入：多个阶段保存重排数据及 profile。请预留磁盘空间；不要把它们提交到 Git。该版本的精确容量修复依赖过载 rank 存在足够多访问权重为 1 的冷 ID；不满足时会报错，不能把它当作对任意负载都保证成功的通用均衡器。

## 数据格式

每条记录由 32-bit label、D 个 float32 dense 值和 S 个 32-bit sparse ID 构成。ID 按表解释，不同表的同值 ID 不相同。

| 数据 | D | S | 字节/记录 |
|---|---:|---:|---:|
| Criteo / Kaggle multi-hot | 13 | 214 | 912 |
| Taobao 四表 | 1 | 4 | 24 |

Criteo/Kaggle hotness 为：

```text
3 2 1 2 6 1 1 1 1 7 3 8 1 6 9 5 1 1 1 12 100 27 10 3 1 1
```

预处理脚本的参数见各自 `--help`。原始下载数据和生成数据都应放在仓库外，或被忽略的 `data/`、`outputs/` 下。

## 实验入口

详见 [实验说明](experiments/README.md)。

```bash
# 在 RAPID 使用的同一份输入上比较 GMP / Herald
python scripts/fetch_dependencies.py HET-GMP Herald
python -m pip install pybind11==2.13.6
python experiments/build_baselines.py
python experiments/run_baselines.py --case outputs/day6_8 --methods gmp herald

# 连续增量：只加入新 emb、调度新 sample，旧 emb 不迁移
python experiments/run_incremental.py --input /data/criteo/day6.bin \
  --out outputs/incremental_8 --ranks 8 --initial-batches 32 \
  --window-batches 1 --windows 32

# 官方 T 系列；固定官方 commit，不使用自制替代分布
python scripts/fetch_dependencies.py HugeCTR-official
python experiments/datasets/generate_official.py t180 \
  --source sources/HugeCTR-official --out outputs/official_t180
```

增量入口的初始 D0 使用历史长期实验中的宽窗口 planner 从空状态构建；它与默认离线定价流水线是两套不同入口，不能把两者结果混写。每轮只在训练前调整该轮新 ID 的放置；已经存在的 owner 和 physical ID 均检查不变。

## 通信量定义

统一 dim=128、FP32 时，每个 embedding/gradient 向量为 512 字节。全 rank 前向量为各 rank 的远程接收量之和；同步反向梯度计数与前向发送量对应。**全 rank 前反向总量 = 2 × 全 rank 前向接收量**，不能再把全 rank 的发送与接收额外相加一次。

对单 rank，前向接收加前向发送可以表示该 rank 的前向接收与反向接收总量。报告同时区分“每步最大 rank 再平均”“全程单步绝对最大值”“各 rank 均值中的最大值”。ACP 按 profiling 数据为每表固定选择 Unique 或 Reduction，Reduction 的单 rank 前向接收数为 `local_batch × (ranks−1)`。

默认统计不包含 ID、路由元数据、dense 梯度、优化器状态和实际网络协议开销。GMP/Herald/BagPipe 的缓存、预取和最终写回须按各自协议单独统计。方法之间的训练同步语义、cache 容量和优化器设置并不自动相同。详见实验说明。

## GPU 训练

`training/run_hugectr_dlrm.py` 保留原 DLRM 实验模板。HugeCTR 可正常 import，或通过 `HUGECTR_PYTHONPATH` 指定编译产物路径。该模板仍包含历史 mode 对应的文件名、表容量和 GPU 配置，需要按真实数据修改；它不是这次 CPU 验证已经运行过的端到端实验。

可用 `cps/merge_for_hugectr.py` 将最终 per-rank 数据按全局 batch 交错合并。必须根据**最终映射**设置 embedding 表容量，并保持 batch 内的 rank 分片，不可再次任意 shuffle 已调度的数据。正式 AUC 实验须使用独立验证集；原模板默认复用训练文件作 eval 的设置不能作为无精度损失的证据。8 卡训练吞吐、AUC、loss、GPU 峰值显存与多机扩展性均需另行实测。

## 验证与源码追踪

```bash
python scripts/smoke_test.py
```

该测试仅生成临时小数据，覆盖 8/64 rank 离线全流程、连续三轮增量及样本交换的不变量，不用于论文性能结论。测试完成自动清理临时数据。正式数据此前经过清理，不随本仓库发布；重跑必须重新准备数据，不能只依赖旧的完成标记。

`experiments/archive/` 是历史实验脚本归档，保留了实验设置和统计逻辑，其中部分脚本依赖旧目录、生成结果或手动阶段衔接；它们不保证直接在新机器运行。请优先使用本 README 列出的可迁移入口。历史微调、缓存、跨 batch 的探索代码仅供追踪，不由默认入口调用。

## 凭据与提交

所有数据、编译产物、环境文件、私钥和日志都被排除在发布清单之外。本次发布使用新分支上的独立源码快照，不携带旧工作目录的 Git 历史。提交前扫描工作树与提交历史；发布校验记录位于 `docs/validation.json`。后续也应执行：

```bash
gitleaks dir . --redact=100
gitleaks git . --log-opts=--all --redact=100
```

只通过本机 Git 凭据管理器或 SSH agent 进行认证，不把任何访问密钥写入代码、URL、配置示例或提交说明。第三方源码使用与历史实验匹配的 commit，见 [依赖锁定文件](experiments/dependencies.json)。
