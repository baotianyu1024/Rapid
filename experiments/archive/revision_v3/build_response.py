from pathlib import Path
import json,re
V=Path(__file__).resolve().parents[1];old=V.parent/'zh_revision_v2';orig=(V/'inputs/prior_response.md').read_text()
def read(p):return json.loads(p.read_text())
def table(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
comp={ds:read(V/f'results/{ds}_8/comparison.json')['methods'] for ds in ['criteo','kaggle','taobao']}
for ds in ['t180','t200','t510']:
 p=V/f'official_tseries/results/{ds}_8/comparison.json';comp[ds]=read(p)['methods'] if p.exists() else {}
sens=read(V/'results/sensitivity_full/summary.json');by={r['name']:r for r in sens['results']};mcr=read(V/'results/mcr_all/summary.json')
names={'criteo':'Criteo Terabyte','kaggle':'Criteo Kaggle','taobao':'Taobao','t180':'T180（官方分片）','t200':'T200（官方分片）','t510':'T510（官方分片）'};methods=['RAPID','HET-GMP','Herald','TorchRec','BagPipe'];comm=table(['数据集／负载']+methods,[[names[d]]+[(f"{comp[d][m]['vector_GiB']:.3f}" if m in comp[d] else "【重跑中】") for m in methods] for d in comp])
pairs=read(V/'training_matrix.json')['real_dataset_model_pairs'];e2e=table(['数据集／模型','HugeCTR','TorchRec','EMBark','HET-GMP','Herald','BagPipe','RAPID'],[[f'{d}／{m}']+['【待填】']*7 for d,m in pairs]+[[d]+['【待填】']*7 for d in ['T180（官方数据）','T200（官方数据）','T510（官方数据）']])
text='''# Revision of VLDB Submission #1058

感谢各位审稿人的细致评阅与建设性意见。根据这些建议，我们补充了外部方法与通信实验，重新检查了共现统计和负载均衡，并明确了真实训练与通信分析的结论范围。以下先集中回应共同关注的问题，再逐条回复各位审稿人的意见。尚待单机 8 卡训练验证的结果保留为空。

## Meta Revision 1：补充外部基线、数据集与模型比较

### 1）基线选择

感谢审稿人指出比较不充分。我们将保留 HugeCTR，并补充 TorchRec 自动分片、EMBark、HET-GMP、Herald 和 BagPipe，覆盖自动分片、联合 embedding/sample 放置以及缓存预取与输入调度。我们已运行可在本地执行的规划和通信统计；端到端吞吐量、延迟、显存及完整训练时间由单机 8 卡实验补充。

### 2）数据集与模型

真实数据包括 Criteo Kaggle、Criteo Terabyte、Taobao 和 MovieLens-25M。前三者分别测试 DLRM、DLRM-DCNv2、Wide&Deep、DeepFM；MovieLens 测试 DLRM 和 NeuMF，共五种模型、14 个数据集／模型组合。MovieLens 使用 rating≥4 的二分类标签及时间划分，补充广告、电商以外的应用；两个 Criteo 数据集仍归为同一应用领域。

模型选择参考了 BagPipe 对 DLRM、Wide&Deep、Deep&Cross 和 DeepFM 的比较，以及 EMBark 的 DLRM-DCNv2 实验；NeuMF 用于覆盖 user-item 交互结构。[BagPipe](https://arxiv.org/pdf/2202.12429)、[EMBark](https://developer.nvidia.com/blog/boost-large-scale-recommendation-system-training-embedding-using-embark/)、[NCF](https://arxiv.org/abs/1708.05031)。各方法共享数据划分、网络结构、初始化和有效 batch，主比较的 embedding 统一为 dim=128；T 系列另列官方模型逐表维度的附表。本轮本地 Criteo 实验使用固定 multi-hot 扩展，各方法共享同一映射；原始单 hot 版本的训练对照另列，不将扩展后的共现直接归为原始单 hot 数据的性质。具体结构与完整执行步骤见[训练实验方案](训练实验方案.md)。

另加入 HugeCTR 官方 T180、T200、T510 负载，直接执行官方数据生成器，保留逐表词表大小、hotness、ID 随机置换及 Zipf(1.1) 采样。固定代码版本 `61b4d539aa7044e9021896f0485be1a6697f4f44`；T180 仅修复输出文件变量名笔误。数据生成不修改官方参数；主比较统一 dim=128。另按同版本 benchmark 的模型维度报告补充结果：T180 为 32/64/128/256，T200 为 64/128，T510 为 128。官方 T180 名称对应的脚本实际展开为 177 张表，我们保持原配置，不自行补表。这些合成数据仅用于通信与性能测试，不用于精度验证。[官方生成器与运行说明](https://github.com/NVIDIA-Merlin/HugeCTR/tree/61b4d539aa7044e9021896f0485be1a6697f4f44/benchmarks/embedding_collection)。

### 3）端到端比较与统一测量

端到端实验各配置独立运行三次，报告吞吐量、step latency、峰值显存及包含预处理的总训练时间。模型适配后先检查计算一致性，不支持的组合注明具体原因。吞吐量结果如下，单位 samples/s：

'''+e2e+'''

表 R1：单机 8 卡训练结果待填；逐次测量项见[训练结果待填.csv](训练结果待填.csv)。

我们比较 RAPID、原生 HET-GMP、Herald、TorchRec 自动分片和 BagPipe 的 embedding 通信。真实数据各取 1,048,576 条、local batch=8,192；T 系列分别使用官方生成器的一个完整分片，65,536 条、local batch=512。统一采用 8 rank、5 epoch、dim=128、FP32。结果为全体 rank 的前向与反向 payload 总量（GiB）：

'''+comm+'''

表 R2：主表统一 dim=128；按官方模型逐表维度统计的 T 系列结果另见[官方维度通信量对比.csv](官方维度通信量对比.csv)，TorchRec 对两种维度配置分别重新规划。一次跨端传输只计一次，包含缓存初始拉取与结束时尚未回写的梯度，排除 ID、metadata 和 dense 梯度通信。RAPID、HET-GMP 使用实际放置和访问回放；Herald 执行上游 LAIA/LRU 内核；TorchRec 执行原生自动 planner 后统计对应协议；BagPipe 执行上游 oracle 后统计预取、回写及 embedding 梯度 AllReduce。以上是 CPU 回放／协议计数，不是 GPU 实测网络或训练加速比。

T 系列官方完整 benchmark 包含 100 个分片；本地结果仅覆盖一个分片，并以该分片的活跃 ID 作为各 CPU 规划器的工作集，不代表完整词表的显存可行性或全规模训练。后续端到端实验使用官方完整配置，并根据硬件容量报告可运行范围；不缩小词表后仍宣称原规模。

各方法使用的缓存资源同时公开：RAPID 无额外镜像，HET-GMP 使用原生镜像预算，Herald 每 worker 缓存为活跃 ID 数的 10%，BagPipe lookahead=8（真实数据另存默认 200 的补充结果）。本表比较这些明确配置，不将其解释为等显存预算的性能排名；端到端另补共同缓存字节预算实验。原生 HET-GMP 可能改变 global batch，RAPID 的 sample 调度始终保留原 global batch。

'''
cr=comp['criteo'];ka=comp['kaggle'];ta=comp['taobao']
text+=f"在本次子集上，RAPID 相对 HET-GMP 在 Terabyte、Kaggle 的通信分别减少 {100*(1-cr['RAPID']['vector_GiB']/cr['HET-GMP']['vector_GiB']):.2f}%、{100*(1-ka['RAPID']['vector_GiB']/ka['HET-GMP']['vector_GiB']):.2f}%；Taobao 上则高出 {100*(ta['RAPID']['vector_GiB']/ta['HET-GMP']['vector_GiB']-1):.2f}%。我们保留收益与不利结果，并据此讨论适用范围。\n\n"
if all(set(comp[d])==set(methods) for d in ['t180','t200','t510']):
 winners=[min(comp[d],key=lambda m:comp[d][m]['vector_GiB']) for d in ['t180','t200','t510']]
 text+=f"在官方 T180、T200、T510 分片的统一 128 维比较中，最低 embedding 通信量分别由 {winners[0]}、{winners[1]}、{winners[2]} 获得。我们将据此讨论不同表数和 pooling 配置下的适用范围。\n\n"
text+='''## Meta Revision 2：明确物理扩展性结论的范围

我们同意，通信量下降不能证明真实训练的扩展性。回放不反映网络拓扑、链路竞争、同步等待及通信计算重叠。因此，我们将修改摘要、Section 7.6 和结论：物理训练结论限定于实际测量的单机 8 卡；16–128 rank 仅称为逻辑 rank 通信分析。由于目前没有多机实验条件，多机训练扩展性将明确列为尚未验证的限制。

## Meta Revision 3：扩大初始离线数据并评估长期增量更新

感谢审稿人关于离线计划有效期的建议。我们将初始数据扩大至 2,097,152 条，随后加入 56 个互不重叠的窗口，每个窗口 65,536 条。每轮只统计当前新增窗口，比较纯 Unique、不更新原放置、增量更新和全量重算。增量仅放置新 emb、在原 global batch 内调度新 sample，旧 emb 不移动；全量重算将截至该轮的全部数据合并、shuffle 后重新规划。

不更新原放置的对照将新 ID 按容量循环放置，仍调度新 sample；增量则利用新窗口访问关系选择新 ID 的位置。四种策略使用相同输入和统计范围，固定初始方案与协议配置。该增量专项实验采用同一窗口规划路径，与表 R2 的完整离线流程分开统计。

| 策略 | W49–W56 平均通信（向量／8,192 条新样本） | 相对纯 Unique 减少 |
| --- | --- | --- |
| 纯 Unique | 960,689.50 | — |
| 不更新原放置 | 356,736.38 | 62.87% |
| 增量更新 | 250,160.16 | 73.96% |
| 全量重算 | 207,912.91 | 78.36% |

表 R3：前向＋反向通信，仅统计新窗口。增量通信为全量重算的 120.32%，比不更新原放置减少 29.88%；旧 emb 迁移量为零。

最后两个八窗口区间的均值变化为 1.42%，最后八窗口 CV 为 0.56%，末十六窗口相对趋势为 3.08%，满足预设的 2%、5%、5% 判据。全量重算在 W1 及 W49–W56 执行，完整逐窗口结果见[逐窗口通信.csv](逐窗口通信.csv)。这些窗口来自已 shuffle 的同源 day6 数据，因此结果说明本次数据增长下的近似稳定性，不代表真实时间漂移下的有效期或数学收敛保证。我们将据此限定论文表述。

## Meta Revision 4：batch size、参数及负载均衡

### 1）每个 batch size 与对应纯 Unique 比较

我们在同一份 262,144 条 Criteo 输入上重新执行 RAPID，并在每个 local batch 下重新计算纯 Unique。纯 Unique 使用 hash 放置、原 sample 分配和去重传输。通信按每条样本的前向＋反向向量数归一化，减少比例为 1−RAPID 通信／同 batch 下的纯 Unique 通信。

'''
br=[]
for name in ['batch512','batch1024','batch2048','batch4096','default','batch16384']:
 r=by[name];br.append([r['batch'],f"{r['unique_vectors_per_sample']:.2f}",f"{r['vectors_per_sample']:.2f}",f"{100*r['saving']:.2f}%"])
text+=table(['local batch','纯 Unique 向量／样本','RAPID 向量／样本','通信减少'],br)+'\n\n表 R4：各行均为完整 RAPID 的独立运行。实际吞吐量与梯度累积结果将在单机训练后补充；不允许跨 microbatch 或原 global batch 调度样本。\n\n### 2）具体的平均通信、峰值和峰均比\n\n我们将通信负载定义为每个 rank 每步接收的前向与反向向量之和。为使原生 HET-GMP 的短尾步不改变比较的样本量，平均通信与平均峰值均按处理相同数量样本归一化：每处理 rank 数×local batch 条样本，计一个等效 global batch。平均通信为所有 rank 的累计通信均值除以等效 batch 数；平均峰值为每个实际步的最忙 rank 通信相加后除以同一分母。峰均比为两者之比，同时报告实际单步绝对峰值。每个向量含 128 个 FP32 分量，即 512 字节。\n\n'
bal=[];storage=[]
for g in [8,64]:
 c=read(V/f'results/criteo_{g}/comparison.json')['methods']
 for method in ['RAPID','HET-GMP']:
  r=c[method];bal.append([g,method,f"{r['mean_vectors_per_rank_step']:,.2f}",f"{r['mean_step_peak']:,.2f}",f"{r['peak_over_mean']:.3f}",f"{r['absolute_peak']:,}"]);storage.append([g,method,f"{r['primary_min']:,}–{r['primary_max']:,}",f"{r['primary_max_mean']:.6f}",f"{r['weight_MiB_max']:.2f}"])
text+=table(['rank','方法','平均通信','平均峰值','峰均比','单步绝对峰值'],bal)+'''\n
表 R5：1,048,576 条 Criteo、local batch=8,192、5 epoch，单位为向量数。所有实际步骤及 HET-GMP 的尾步均计入；平均通信与平均峰值的共同分母为 8 rank 下 80 个、64 rank 下 10 个等效 global batch，绝对峰值取实际单步。64 rank 的 RAPID 每 epoch 只有两个完整 global batch，本表用于负载诊断，不证明物理扩展性。

'''+table(['rank','方法','每 rank 主 emb 数量范围','主存储峰均比','含镜像峰值权重 MiB'],storage)+'''\n
表 R6：权重统一按 dim=128、FP32 换算，HET-GMP 包含镜像；不含梯度、optimizer state 和通信 buffer，不能代替峰值显存实测。RAPID 各 rank 的主 emb 数量最多相差一个，因此存储峰均比接近 1；这不意味着访问与通信也完全均衡。

### 3）参数和输入顺序的当前实验结果

感谢审稿人指出参数说明不足。我们在 262,144 条 Criteo、8 rank 上重新执行完整 RAPID，逐项改变参数，其余保持默认：β=0.0001、γ=0.2、全局 hot Top-K=40,000，默认不启用 METIS 初始化。profiling 顺序实验仅改变 profiling batch 的处理顺序，不改变训练 global batch 的成员。

'''
labels={'default':'默认','beta0':'β=0','beta0.001':'β=0.001','gamma0':'γ=0','gamma1':'γ=1','topk10000':'Top-K=10,000','topk100000':'Top-K=100,000','metis1e-05':'METIS 3 行','metis0.0001':'METIS 26 行','metis0.001':'METIS 262 行','orderreverse':'profiling batch 逆序','order11':'profiling batch shuffle，seed=11','order29':'profiling batch shuffle，seed=29'}
text+=table(['设置','前反向向量／样本','本地命中率','主 emb 存储最大／平均'],[[lab,f"{by[n]['vectors_per_sample']:.3f}",f"{100*by[n]['local_hit']:.2f}%",f"{by[n]['primary_max_mean']:.6f}"] for n,lab in labels.items()])+'''\n
表 R7：统计完成全部放置和调度步骤后的结果。各设置的最终主 emb 数量最多相差一个，因此主存储峰均比均约为 1。hot Top-K 是固定数量，不是固定百分比；METIS 的实际初始化行数随本次样本量明确列出。我们将报告参数与顺序变化带来的实际差异，并承认该启发式方法尚无相对最优解的形式化近似保证。

## Meta Revision 5：共现统计的支持次数与精度保持

### 1）在出现次数达到 x 的冷 ID 上重新计算 MCR

感谢审稿人指出低支持次数可能影响共现统计。我们使用 2,097,152 条固定 multi-hot 预处理后的 Criteo 样本，对最低出现次数为 10、20、50、100 的全部符合条件的冷 ID 重新计算 MCR，没有抽样目标 ID。冷 ID 仍定义为出现频率低于 0.01%，本次对应支持次数不超过 209；同一 ID 在同一样本中重复出现只记一次。

$$
\\mathrm{MCR}(u)=\\max_{v\\in S_{\\mathrm{cold}},\\ v\\ne u}\\frac{\\mathrm{Count}(u\\cap v)}{\\mathrm{Count}(u)}.
$$

最小次数限制作用于目标 ID，partner 保持原公式中的冷 ID 集合，不额外限制为跨表。

'''
text+=table(['最小支持次数','符合条件的全部冷 ID 数','MCR>95% 的 ID 数','比例'],[[r['min_support'],f"{r['eligible_ids']:,}",f"{r['mcr_above_95_ids']:,}",f"{r['mcr_above_95_percent']:.4f}%"] for r in mcr['statistics']])+'''\n
表 R8：固定输入窗口的全量统计，不涉及目标 ID 抽样置信区间。排除低支持 ID 后仍有较高比例的共现关系。我们将在 Figure 6(c) 中补充最低支持次数、冷 ID 定义和候选数量，以此替换含糊的单一百分比；不将窗口内共现直接推断为未来稳定性，也不将 95% 解释为置信水平。

'''
a=orig.index('### 2）为什么在所述条件下保持模型计算');b=orig.index('### 3）补充完整训练验证',a);text+=orig[a:b]
text+='''### 3）补充完整训练验证

我们将对上述 14 个真实数据集／模型组合，使用五组配对种子比较同模型参考实现与完整 RAPID。每组共享逻辑 ID 对齐的初始权重、global batch 序列和完整评估集，报告 AUC、LogLoss、训练曲线及配对差值区间。训练前先核对前向、梯度和 optimizer state；训练后依据预先设定的 AUC 等价界限判断，而不以“差异不显著”代替等价证据。

'''+table(['数据集／模型','参考实现 AUC','RAPID AUC','AUC 配对差及区间','两者 LogLoss'],[[f'{d}／{m}']+['【待填】']*4 for d,m in pairs])+'\n\n'
a=orig.index('## Meta Revision 6：');text+=orig[a:]
text=text.replace('实验覆盖四个真实数据集、DLRM 与 DLRM-DCNv2，以及表结构不同的合成压力负载。','实验覆盖四个真实数据集、五种模型，以及表结构不同的合成压力负载。')
text=text.replace('我们在出现次数至少为 1、5、10、20、50 的冷 ID 中重新抽样并计算原定义的 MCR，结果显示排除极低支持 ID 后仍存在明显的高共现比例。回复公开候选总体、抽样数量和跨表补充统计，不将新的筛选口径混写成原 79.3% 统计。','我们对出现次数至少为 10、20、50、100 的全部符合条件的冷 ID 计算原定义的 MCR，对应 MCR>95% 的比例为 70.5613%、80.7066%、84.7580%、85.7500%。回复公开每个阈值的完整分母与计数，不再使用抽样 ID 的估计，也不将新统计混写成原 79.3% 的口径。')
text=text.replace('并采用两类模型','并采用五种模型')
# Remove stale intermediate ablation numbers from reviewer-facing final paragraph.
a=text.index('当前重新执行的核心消融中，');b=text.index('\n',a) if '\n' in text[a:] else len(text)
text=text[:a]+'我们将通过表 R2、R5 和 R6 分别展示通信总量、通信峰值与存储分布，说明这些目标之间的取舍，同时明确承认已有联合放置工作的贡献。'+text[b:]
# Make reviewer prose concise without dropping request coverage.
a=text.index('请参见 Meta Revision 3。我们重新扩大');b=text.index('\n\n### 4.',a)
text=text[:a]+'''请参见 Meta Revision 3。初始数据增至 2,097,152 条，随后加入 56 个窗口。增量只放置新 emb、调度新 sample，旧 emb 不迁移。末八窗口的增量通信为全量重算的 120.32%，相对纯 Unique 减少 73.96%。该结论限于本次同源数据增长；真实时间漂移与物理更新耗时仍需另行验证。'''+text[b:]
text=text.replace('我们扩大 batch 实验数据，','我们重新执行 batch size 实验，').replace('平均每步峰值','按同等样本量统计的平均峰值')
(V/'中文Response.md').write_text(text)
# Retain the already completed incremental evidence with explicit provenance.
import shutil
shutil.copy2(old/'逐窗口通信.csv',V/'逐窗口通信.csv')
print('response characters',len(text))
