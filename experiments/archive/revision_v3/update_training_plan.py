from pathlib import Path
import csv,itertools,json
V=Path(__file__).resolve().parents[1];old=V.parent/'zh_revision_v2';s=(V/'inputs/prior_training_plan.md').read_text()
s=s.replace('原生系统端到端；另做保持原 global batch 的组件对照，并标注这是适配版本','原生系统端到端和通信比较，不修改其 sample 分配约束')
s=s.replace('当前 CPS 放置／组调度＋ACP','完整 RAPID 默认执行流程')
a=s.index('| ID | 数据集');b=s.index('\nMovieLens 的 user/item',a)
s=s[:a]+'''| ID | 数据集 | 划分 | 测试模型 |
|---|---|---|---|
| R1 | Criteo Kaggle | 固定训练／验证／测试划分 | DLRM、DLRM-DCNv2、Wide&Deep、DeepFM |
| R2 | Criteo Terabyte | 原始 day 划分 | DLRM、DLRM-DCNv2、Wide&Deep、DeepFM |
| R3 | Taobao | 保持原特征定义，按时间划分 | DLRM、DLRM-DCNv2、Wide&Deep、DeepFM |
| R4 | MovieLens-25M | rating≥4 为正，其余为负；时间 80%/10%/10% | DLRM、NeuMF |

选择依据：BagPipe 的 Table 2 和 Figure 10 使用 DLRM、Wide&Deep、Deep&Cross、DeepFM；EMBark 使用 DLRM-DCNv2 和 T180/T200/T510。这里用 DCNv2 覆盖 cross-network 结构，并增加适合 user-item 数据的 NeuMF；因此包括五种模型，而非只改 DLRM 的名称。[BagPipe 论文](https://arxiv.org/pdf/2202.12429)、[EMBark 作者说明](https://developer.nvidia.com/blog/boost-large-scale-recommendation-system-training-embedding-using-embark/)、[NCF 论文](https://arxiv.org/abs/1708.05031)。
''' +s[b:]
a=s.index('- **M1，');b=s.index('- 模型配置导出',a)
s=s[:a]+'''主比较的 embedding 统一 **dim=128、FP32、SUM pooling**；使用官方 T 系列数据生成器，不改词表或访问分布。官方逐表模型维度另作通信附表。以下是本轮冻结的比较配置，不声称逐参数复刻上述论文。

| ID | 模型 | 网络定义 |
|---|---|---|
| M1 | DLRM | bottom `[dense_dim,512,256,128]`；底部输出与各表向量做 dot interaction；top `[interaction_dim,512,256,1]` |
| M2 | DLRM-DCNv2 | bottom 同 M1；与表向量拼接；3 层低秩 CrossNet，投影维度 512；top `[concat_dim,1024,1024,512,256,1]` |
| M3 | Wide&Deep | deep：dense 与 pooled embedding 拼接后 `[concat_dim,512,256,128,1]`；wide：原始 dense 线性项及每个类别 ID 的一维权重；两支 logits 相加 |
| M4 | DeepFM | 一阶 dense／类别 ID 线性项；每个数值特征用可学习的 128 维向量乘其值，类别使用 pooled 向量；全部 field 的 FM 二阶交互；field 向量拼接后 deep `[fields×128,512,256,128,1]`；三支 logits 相加 |
| M5 | NeuMF | GMF 和 MLP 使用独立的 user/item embedding，共四张 128 维表；GMF 逐元素乘积；MLP `[256,256,128,64]`；两支拼接后线性输出 |

主实验无 BatchNorm、无 dropout，使用 BCE-with-logits。M3/M4 的一维线性权重是额外参数，不是把主 embedding 改成 dim=1；其通信在端到端总网络字节中另计。M5 的四张表不与 MovieLens/DLRM 的两张表混用。NeuMF 用于上述明确的评分二分类任务，不声称复刻 NCF 原论文的负采样和 leave-one-out 协议。通信回放表只描述实际输入的表与访问，不将同一组通信数复制成五种模型的实测结果。

''' +s[b:]
start=s.find('### 2.3')
if start>=0:
 end=s.find('\n## ',start)
 if end<0:end=len(s)
 s=s[:start]+'''### 2.3 官方 T 系列负载

使用 [HugeCTR 官方配置](https://github.com/NVIDIA-Merlin/HugeCTR/tree/61b4d539aa7044e9021896f0485be1a6697f4f44/benchmarks/embedding_collection)，不使用自定义 inspired 负载。生成器、逐表参数、补丁及 SHA256 保存在 `official_tseries`。完整数据按官方 generation.sh 由 100 个 65536 行分片组成。当前本地通信测试只使用一个完整官方分片；完整模型能否在单机 8 卡运行，须按实机显存与精度配置判断。各方法用相同官方数据，主比较统一dim=128；官方逐表模型维度另列通信附表，不能为了装入显存静默缩小词表。
''' +s[end:]
s=s.replace('使用固定 seed=20261026、按 global sample ID 可重建的流式生成器','后续扩大规模时使用固定 seed=20261027、按 global sample ID 可重建的流式生成器')
s=s.replace('另行使用 batch-preserving 适配版做固定批次机制对比。','')
s=s.replace('四个真实数据集的主模型先跑七个系统；三个 CTR 数据集再跑第二个模型的核心四系统（HugeCTR、TorchRec、EMBark、RAPID），其他系统适配完成后补齐。','按照 R1–R3 各四模型、R4 两模型的 14 个组合，逐一运行七个系统；先完成每个系统的兼容性检查。缺少同模型适配时明确记录，不用其他模型代替。')
s=s.replace('正确性对照至少包括：HugeCTR 原配置、RAPID-CPS only、RAPID-ACP only、RAPID-CPS+ACP。覆盖四个真实数据集主模型，以及三个 CTR 数据集的第二模型。','正确性对照只包含同模型的参考实现与完整 RAPID，覆盖上述 14 个真实数据集／模型组合。不要求将 CPS、ACP 拆开；本节验证完整方法是否影响精度。机制消融如需保留，放在独立实验中。')
s=s.replace('通信规模：四个真实数据集和三类合成负载，8/64 rank，至少 32 个完整 global batch（数据不足的 MovieLens 配置按实际数量注明）。','后续扩展目标：四个真实数据集和三类合成负载，8/64 rank，尽可能覆盖至少 32 个完整 global batch。当前已执行范围、行数和实际 batch 数以本目录实验证据为准，不能写成目标规模均已完成。')
s+='''
## 4. 本轮本地通信实验与后续训练的区别

本轮 real-data 子集各 1,048,576 行、8 rank、local batch=8192、5 epoch。T180/T200/T510 均直接使用固定版本 HugeCTR 官方生成脚本的完整单个分片：65536 行、seed=20261027、Zipf(1.1)，保留全部逐表词表大小、hotness、ID 随机置换；8 rank、local batch=512、5 epoch，FP32。主比较统一dim=128。通信附表逐表维度按官方 benchmark.sh，分别为 T180 的 32/64/128/256、T200 的 64/128、T510 的 128。官方 T180 脚本展开为 177 张表，不自行补表。官方完整数据为 100 个分片，本轮只测试其中一个，CPU 规划器只物化该分片活跃 ID，不能用于声称完整词表能装入单机显存。64 rank 额外复测 Criteo，相同 1,048,576 行只有 2 个完整 global batch/epoch。

RAPID 和 HET-GMP 运行实际规划及访问回放；Herald 执行上游 LAIA/LRU 内核；TorchRec 执行原生自动 planner 后按其通信协议计数；BagPipe 执行上游 oracle 后统计预取和梯度同步协议。均不是 GPU 实测网络。TorchRec 本次逻辑拓扑为单机 8 卡、每卡 16 GiB、带宽参数 14.66833055 GiB/s；实际训练时必须替换为实机拓扑，重新规划。

当前默认配置比较的缓存资源并不相同：RAPID 无额外镜像，GMP 镜像来自原生预算，Herald 每 worker 缓存为当前活跃 ID 数的 10%，BagPipe lookahead=8（默认 200 另存补充结果）。此表不能用于声称等显存预算下的优越性；端到端实验另外执行共同字节预算和原生默认配置两组。
'''
(V/'训练实验方案.md').write_text(s)
methods=['HugeCTR','TorchRec','EMBark','HET-GMP','Herald','BagPipe','RAPID'];pairs=[(d,m) for d in ['Kaggle','Terabyte','Taobao'] for m in ['DLRM','DLRM-DCNv2','Wide&Deep','DeepFM']]+[('MovieLens-25M',m) for m in ['DLRM','NeuMF']];perf=pairs+[(x,'DLRM-DCNv2') for x in ['T180（官方数据）','T200（官方数据）','T510（官方数据）']]
cols=['experiment','dataset','model','method','repeat_or_seed','dim','local_batch','status','samples_per_second','step_mean_ms','step_p95_ms','max_gpu_memory_MiB','total_network_bytes','preprocess_seconds','total_training_seconds','AUC','LogLoss','log_path','notes']
with (V/'训练结果待填.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
 for d,m in perf:
  for method,rep in itertools.product(methods,[1,2,3]):w.writerow(dict(experiment='performance',dataset=d,model=m,method=method,repeat_or_seed=rep,dim=128,local_batch=512 if d.startswith('T1') or d.startswith('T2') or d.startswith('T5') else 8192,status='待实测'))
 for d,m in pairs:
  for method,seed in itertools.product(['参考实现','RAPID'],[7,19,41,73,101]):w.writerow(dict(experiment='accuracy',dataset=d,model=m,method=method,repeat_or_seed=seed,dim=128,local_batch=8192,status='待实测'))
(V/'training_matrix.json').write_text(json.dumps(dict(model_count=5,real_dataset_model_pairs=pairs,methods=methods,performance_runs=len(perf)*7*3,accuracy_runs=len(pairs)*2*5,dim=128),ensure_ascii=False,indent=2))
