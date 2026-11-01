"""Summarize fresh runs; refuse to silently fall back to historical baselines."""
from pathlib import Path
import csv, hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
case=ROOT/'results/default8'
traffic=case/'fresh_baselines/traffic'
latest=case/'latest_rapid'
herald=json.loads((case/'herald/result.json').read_text())
assert herald['forward_vectors']==herald['backward_vectors']+herald['terminal_flush_vectors']
assert json.loads((latest/'complete.json').read_text())['success']
rapid=json.loads((latest/'traffic/rapid_acp_original.json').read_text())
gmp=json.loads((traffic/'het_gmp_s0.json').read_text())
assert herald['processed_rows']==rapid['processed_rows']==gmp['processed_rows']
assert gmp['pending_embedding_vectors']==0
audit=json.loads((latest/'rapid_schedule/sample_audit.json').read_text())
assert audit['matching_128bit_record_multiset'] and audit['rows']==herald['rows']
assert json.loads((latest/'rapid_guard_p5/complete.json').read_text())['all_records_inverse_verified']
for name,parent in [('herald_replay',case),('materialize',latest),('replay',latest),('het_preprocess',case/'fresh_baselines'),('gmp_replay',case/'fresh_baselines')]:
    assert json.loads((parent/(name+'.status.json')).read_text())['exit_code']==0,name

rows=[]
def add(name,fwd,bwd,flush):
    total=int(fwd+bwd+flush)
    rows.append(dict(method=name,forward_vectors=int(fwd),backward_vectors=int(bwd),terminal_flush_vectors=int(flush),
      total_vectors=total,million_vectors=total/1e6,vectors_per_sample=total/herald['processed_rows'],
      forward_vectors_per_8192_samples=float(fwd)*8192/herald['processed_rows'],
      GiB_FP32_dim32=total*128/2**30,GiB_FP32_dim128=total*512/2**30))
add('RAPID latest full pipeline',rapid['forward_receive_vectors']['total'],rapid['backward_receive_vectors']['total'],0)
add('Herald default LAIA, cache 10%',herald['forward_vectors'],herald['backward_vectors'],herald['terminal_flush_vectors'])
add('HET-GMP s=0',gmp['forward_receive_vectors']['total'],gmp['backward_receive_vectors']['total'],0)
for row in rows:row['relative_to_rapid']=row['total_vectors']/rows[0]['total_vectors']
rapid_vs_herald=1-rows[0]['total_vectors']/rows[1]['total_vectors']
gmp_vs_rapid=1-rows[2]['relative_to_rapid']
with (case/'comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=dict(rows=rows,processed_rows=herald['processed_rows'],herald_epoch_counts=herald['per_epoch'],
    herald_final_flush_vectors=herald['terminal_flush_vectors'],herald_without_terminal_flush=herald['total_vectors_without_flush'],
    rapid_strategies=rapid['strategies'],gmp_steps_per_epoch=gmp['steps_per_epoch'],
    gmp_primary_counts=gmp['primary_counts'],gmp_mirror_counts=gmp['mirror_counts'],
    herald_cache_entries_per_worker=herald['cache_entries_per_worker'],herald_assignment_sha256=herald['assignment_sha256'],
    rapid_source_manifest=str(latest/'source_manifest.json'),rapid_recipe=json.loads((latest/'complete.json').read_text()),
    rapid_reduction_vs_herald=rapid_vs_herald,gmp_reduction_vs_latest_rapid=gmp_vs_rapid)
(case/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
text=['Herald / 最新 RAPID / HET-GMP 通信量比较（2026-10-25）','',
'测量性质：实际运行 CPU 调度/划分与通信回放，统计逻辑 embedding/gradient 向量载荷；不是 8-GPU 训练或网卡字节实测。',
'机器只有一张 V100S，未运行三套系统的 CUDA 数值训练，不据此推导吞吐或准确率。','',
'配置：同一份 shuffled Criteo day6 前 4,194,304 行，26 表/214 多热槽位，8 个逻辑 worker，每卡 batch=8192，5 epoch，共 20,971,520 个样本访问。',
'RAPID: 当前 reproduction/working + reproduction/scripts 源码快照重新编译；采用 PARALLEL_RAPID_RUNBOOK 最近记录的完整流程。',
'topk=40000、gamma=0.2、ratio swap（screen=20000000、evaluations=500000）、16个独立batch worker，随后owner_base与price_f2/b8/b32/guard_p5固定精修链。',
'最新流程的精确参数和源码哈希见 latest_rapid/*.command.json、source_manifest.json；不是旧 code/Rapid 或较早的 code/update_Rapid 快照。',
'Herald: 默认非 local-shared 的全槽位 LaiaScheduler；每卡缓存=子集中活跃 embedding 的10%；空缓存开始，跨 epoch 保留。',
'HET-GMP: 原图划分10轮、store_rate=0.01、严格同步 s=0。每个样本每epoch恰好处理一次，尾批不补样本。',
'本次 latest_rapid 从热ID统计开始重新执行最新完整流程，fresh_baselines 重新执行 GMP 划分及回放；不复用旧数值作为主结果。',
'fresh_baselines 中的旧版 RAPID 只作历史参考，已被用户要求的最新代码结果替代。','',
'计量：前向+反向，全系统每次传输计一次，排除自通信。Herald 的 PS-worker 传输均计入；RAPID/GMP 计跨分片传输。',
'不含 ID/版本元数据、NCCL/网络包头、dense 梯度同步、训练样本分发及初始分区物化。不同后端元数据无法统一，因此只比较向量载荷。',
'Herald 按源码 MiniLRUCache 计 miss/refresh pull、有效缓存淘汰 push、跨 worker 依赖 push；结尾额外单列并计入全部剩余有效缓存贡献的 flush。',
'flush 是明确的补充通信模型，不声称作者提供了该 API。未把额外预取批当成训练批。','',
'方法 | 总向量(百万) | 向量/样本 | FP32 dim32 GiB | FP32 dim128 GiB | 相对RAPID']
for r in rows:text.append(f"{r['method']} | {r['million_vectors']:.6f} | {r['vectors_per_sample']:.4f} | {r['GiB_FP32_dim32']:.3f} | {r['GiB_FP32_dim128']:.3f} | {r['relative_to_rapid']:.4f}")
text+=['',f'本次最新RAPID比Herald少{rapid_vs_herald*100:.4f}%载荷；GMP比最新RAPID少{gmp_vs_rapid*100:.4f}%载荷。仅限本输入和上述内存/执行配置。']
text+=['',f"Herald 运行中向量数={herald['total_vectors_without_flush']}，终止flush={herald['terminal_flush_vectors']}。",'Herald 按epoch（终止flush不重复摊入各epoch）：']
for e in herald['per_epoch']:text.append(str(e))
text+=['','可比性和实现边界：',
'1. Herald 的 CPU PS 保存全表，GPU各有10%缓存；RAPID按GPU分片；GMP按GPU分片加副本。这不是等总存储资源对比。',
'2. 仅day6子集，规划和运行使用同一数据。不能代表整日、留出数据、多节点运行或其他数据集。',
'3. GMP每epoch的步数可能因分区不均衡多于64，因此主表按相同总样本归一化，不以每步均值掩盖尾批差异。',
'4. Herald以原始整数(table,id)紧凑映射输入保留214个访问槽位；没有经过上游Python loader的float32中转，避免大ID精度损失。未复现原始单热神经网络模型。',
'   LAIA打分沿用源码逐槽位计数，同一样本内重复ID重复计分；通信和缓存访问仍按ID去重。这是多热访问trace适配，不是原论文单热端到端实验。',
'5. Herald直接编译提取的原始 get_dist 函数，唯一函数体替换为 boost::flat_set -> std::set（相同有序集合语义）；使用原始Array、ThreadPool、MiniLRUCache。',
'   批次数据窗口化只平移样本索引；batch_id保持原tie-break。启用原始缓存已有bitmap查找，独立测试验证与哈希查找一致。',
'6. 这是默认静态 LAIA，不是论文自适应 Top-K 的端到端调度时间复现。原默认 local_shared=False 路径本身未使用 top_K_table。',
'7. 最新RAPID的ACP用前50批固定逐表选择，不改为逐批oracle。本次全部表为Unique，核验后采用同一去重向量计数；前向每个远端unique ID一个向量，反向按ID合并梯度。',
'8. 未报告预处理速度优劣：本次为通信量比较，长任务有并发，墙钟日志只用于复现和诊断。','',
'9. 主表为一次完整运行。原始异步调度/并行图划分可能产生不同分区或输出顺序；不声称具有统计显著性。','',
'验证：独立Python oracle覆盖72个小批次（两种cache表示、冷启动、共享ID、重复ID、淘汰、刷新、跨epoch tie-break及最终pending）；',
'每个Herald批次断言样本不丢不重且每卡正好8192；每epoch首批及前3批独立验证依赖集合；GMP回放断言更新次数守恒。',
'RAPID重新审计标签+dense+sparse完整记录多重集；所有完成状态要求exit_code=0。','',
'详细结果：comparison.json / comparison.csv / herald/steps.csv / herald/counts.npz。',
'所有命令、标准输出、time及状态文件保存在本目录、latest_rapid和fresh_baselines。Herald构建和源码哈希见../../build/provenance.json。']
(ROOT/'REPORT.txt').write_text('\n'.join(text)+'\n')
print('\n'.join(text[:24]))
