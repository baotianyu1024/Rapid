import argparse,csv
import numpy as np
from common import *
from replay_het import summary as traffic_summary

p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args();case=Path(a.case).resolve()
def read(path):return json.loads(path.read_text())
cfg=read(case/'data/input.json');hp=read(case/'het/partition.json');audit=read(case/'rapid/sample_audit.json')
assert audit['matching_128bit_record_multiset'] and audit['rows']==cfg['rows']
timings=[]
for name,folder,stages in [('RAPID',case/'rapid',['hot','placement','schedule','acp']),('HET-GMP',case/'het',['partition'])]:
    for stage in stages:
        v=read(folder/f'{stage}.status.json');assert v['exit_code']==0
        timings.append(dict(algorithm=name,stage=stage,seconds=v['seconds'],cpu_seconds=v['user_seconds']+v['system_seconds'],peak_rss_gib=v['max_rss_kib']/1024**2))
v=read(case/'het_materialize.status.json');timings.append(dict(algorithm='HET-GMP',stage='materialize',seconds=v['seconds'],cpu_seconds=v['user_seconds']+v['system_seconds'],peak_rss_gib=v['max_rss_kib']/1024**2))
for algo in ['rapid','het']:
    status=read(case/f'{algo}_pipeline.status.json');assert status['exit_code']==0
rapid_time=sum(t['seconds'] for t in timings if t['algorithm']=='RAPID')
het_raw_time=sum(t['seconds'] for t in timings if t['algorithm']=='HET-GMP')
het_time=het_raw_time-hp['guard_seconds']
lines=['# HET-GMP / RAPID 本地对比结果','',f"输入：Criteo day6 前 {cfg['rows']:,} 行；8 个逻辑 GPU，local batch 8192，26 表/214 槽位，128 维 FP32，按 RAPID 训练默认值回放 {EPOCHS} 个 epoch。输入 SHA-256：`{cfg['input_sha256']}`。",'',
       '**预处理为本机实测，通信为 CPU trace 回放/模型计数。当前每算法一次预处理运行，没有重复计时误差条；结果不是全日、多机或 GPU 训练测量。**','',
       '## 预处理时间','', '| 算法 | 阶段 | 墙钟秒 | CPU 秒 | 峰值 RSS GiB |','|---|---|---:|---:|---:|']
for t in timings:lines.append(f"| {t['algorithm']} | {t['stage']} | {t['seconds']:.2f} | {t['cpu_seconds']:.2f} | {t['peak_rss_gib']:.2f} |")
lines += ['',f"RAPID 原流程阶段合计 **{rapid_time:.2f} s**；HET 图划分/导出加分片物化合计 **{het_time:.2f} s**（从上表原进程时间扣除了额外安全检查）。",f"HET 专用无损 ID 适配另需 **{cfg['het_adapter_seconds']:.2f} s**；计入后为 **{het_time+cfg['het_adapter_seconds']:.2f} s**。公共输入切片复制 {cfg['copy_seconds']:.2f} s 不计入任一算法。",f"HET 图构建 {hp['graph_seconds']:.2f} s，10 轮 refinement 合计 {sum(x['data_seconds']+x['embedding_seconds'] for x in hp['rounds']):.2f} s，优先级排序及导出 {hp['export_priority_seconds']:.2f} s；输入边界安全检查另 {hp['guard_seconds']:.2f} s（包含在 partition 原进程时间，但不纳入算法合计）。",'',
          'RAPID ACP 阶段包含原脚本的 Hash 对照 profile；HET 额外物化与图划分分开计时。两者文件输出内容不同，不能将所有秒数都解释成纯算法计算。峰值 RSS 不等于 GPU 显存。','',
          '## 可比向量通信量','',
          '全局传输只计一次，排除本地；forward + backward 为 embedding 与 gradient 向量载荷。HET 保留所有行，尾批可变，主比较用同一样本集合的总流量。','',
          '| 方案 | 前向 GiB | 反向 GiB | 合计 GiB（128 维） | 合计 GiB（32 维） | 每样本 KiB（128 维） |',
          '|---|---:|---:|---:|---:|---:|']
tags=['hash_unique','hash_acp_original','rapid_unique','rapid_acp_original','rapid_acp_remote','het_1d_s0','het_gmp_s0','het_gmp_s10','het_gmp_s100']
rows=[]
for tag in tags:
    r=read(case/'traffic'/f'{tag}.json');assert r['rows']==cfg['rows']
    f=r['forward_receive_vectors']['total'];g=r['backward_receive_vectors']['total']
    assert f==r['forward_send_vectors']['total'] and g==r['backward_send_vectors']['total']
    assert r['vector_bytes_128']==(f+g)*512
    assert r['epochs']==EPOCHS and r['processed_rows']==cfg['rows']*EPOCHS
    row=dict(variant=tag,rows=cfg['rows'],epochs=EPOCHS,steps=r['steps'],forward_vectors=int(f),backward_vectors=int(g),total_gib_128=r['vector_bytes_128']/1024**3,total_gib_32=r['vector_bytes_32']/1024**3,kib_per_sample_128=r['vector_bytes_128']/r['processed_rows']/1024,forward_mean_rank_step=r['forward_receive_vectors']['mean_per_rank_step'],forward_mean_batch_peak=r['forward_receive_vectors']['mean_batch_peak'],forward_absolute_peak=r['forward_receive_vectors']['absolute_peak'])
    rows.append(row)
    row.update(send_mean_batch_peak=r['forward_send_vectors']['mean_batch_peak'],send_absolute_peak=r['forward_send_vectors']['absolute_peak'],recv_max_peak_mean_ratio=r['forward_receive_vectors']['max_peak_mean_ratio'],send_max_peak_mean_ratio=r['forward_send_vectors']['max_peak_mean_ratio'])
    if tag.startswith('het_'):
        raw=np.load(case/'traffic'/f'{tag}.npz')['forward']
        full=min(r['rank_rows'])//B
        mask=np.arange(len(raw))%r['steps_per_epoch']<full
        rs=traffic_summary(raw[mask].sum(axis=2));ss=traffic_summary(raw[mask].sum(axis=1))
    else:
        rs=r['forward_receive_vectors'];ss=r['forward_send_vectors']
    row.update(full_batch_recv_peak_mean_ratio=rs['max_peak_mean_ratio'],full_batch_send_peak_mean_ratio=ss['max_peak_mean_ratio'])
    lines.append(f"| {tag} | {f*512/1024**3:.3f} | {g*512/1024**3:.3f} | {row['total_gib_128']:.3f} | {row['total_gib_32']:.3f} | {row['kib_per_sample_128']:.3f} |")
base=next(x for x in rows if x['variant']=='het_gmp_s0');rapid=next(x for x in rows if x['variant']=='rapid_acp_original')
change=100*(rapid['total_gib_128']/base['total_gib_128']-1)
lines += ['',f"主对照：RAPID 原 ACP 策略相对 HET-GMP s=0 的向量载荷变化为 **{change:+.2f}%**（负值为减少）。s=10/100 的一致性条件不同，不能当作相同训练语义下的胜负。",'',
          '`acp_original` 保留源代码 B*P 选策略阈值，传输时剔除 self；`acp_remote` 是 B*(P-1) 选策略的显式修正变体。Reduction 采用密集远端池化向量模型，反向是假设梯度按同一通信结构返回的计数，不是 HugeCTR 抓包。','',
          '## 峰值与样本调度','', '| 方案 | 回放步数 | 前向每 rank 每步平均向量 | 逐步最忙 rank 平均向量 | 单步绝对峰值 |','|---|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['variant']} | {r['steps']} | {r['forward_mean_rank_step']:.1f} | {r['forward_mean_batch_peak']:.1f} | {r['forward_absolute_peak']:.0f} |")
lines += ['','发送侧不能由接收侧均衡性代替；以下仍为前向向量流量。尾部残批可能放大峰均比，均保留而不隐藏。','', '| 方案 | 逐步最忙发送 rank 平均向量 | 最坏接收峰均比 | 最坏发送峰均比 |','|---|---:|---:|---:|']
for r in rows:lines.append(f"| {r['variant']} | {r['send_mean_batch_peak']:.1f} | {r['recv_max_peak_mean_ratio']:.3f} | {r['send_max_peak_mean_ratio']:.3f} |")
lines += ['','为剔除 HET 尾部空 rank 的影响，另看每个 rank 都有完整 8192 条样本的步骤；这仅用于均衡性诊断，主通信总量仍保留所有样本。','', '| 方案 | 完整 batch 最坏接收峰均比 | 完整 batch 最坏发送峰均比 |','|---|---:|---:|']
for r in rows:lines.append(f"| {r['variant']} | {r['full_batch_recv_peak_mean_ratio']:.3f} | {r['full_batch_send_peak_mean_ratio']:.3f} |")
h=read(case/'traffic/het_gmp_s0.json')
first_1d=read(case/'traffic/epoch1_het_1d_s0.json')['forward_receive_vectors']['total']
predicted_1d=hp['rounds'][-1]['weighted_cost']/0.1*cfg['global_batches']
lines += ['',f"辅助检查：HET 内部 1D 代价模型换算到一轮输入约 {predicted_1d:,.0f} 个远端向量，独立实际 ID 回放为 {first_1d:,.0f}，差 {100*(first_1d/predicted_1d-1):+.3f}%。前者是 batch 访问概率估计，不能替代后者。"]
lines += ['',f"HET rank 样本数：{h['rank_rows']}；RAPID rank 样本数：{audit['rank_rows']}。不同步数的平均值仅用于诊断，不替代每输入样本/同一样本总量比较。",'',
          '## HET 元数据、延后更新与存储代价','',
          '| 方案 | 元数据 GiB | 尚未回传更新次数 | 尚未回传向量数 | 假设最终刷新的额外 MiB（128 维） |',
          '|---|---:|---:|---:|---:|']
for tag in ['het_1d_s0','het_gmp_s0','het_gmp_s10','het_gmp_s100']:
    v=read(case/'traffic'/f'{tag}.json')
    assert v['updates']==cfg['rows']*sum(LENGTHS)*EPOCHS
    lines.append(f"| {tag} | {v['metadata_receive_bytes']['total']/1024**3:.3f} | {v['pending_updates']:,} | {v['pending_embedding_vectors']:,} | {v['hypothetical_final_flush_vector_bytes_128']/1024**2:.3f} |")
lines += ['',f"每个 HET 变体另有末批管线排空 size 交换 {h['drain_round_shape_bytes']:,} B；假设最终 flush 的 ID/计数为每向量 16 B，详见 JSON。未计 NCCL/网络包头。RAPID 元数据缺少后端实证，未补造估计，也不拿其纯向量字节与 HET 总协议字节直接比较。",'',
          f"HET 主副本行数：{h['primary_counts']}；secondary 行数：{h['mirror_counts']}。",f"全体 secondary 向量加累积梯度缓冲区的额外 FP32 存储下界：**{sum(h['mirror_counts'])*128*4*2/1024**3:.3f} GiB**（128 维）；还不含哈希表、版本、临时区等。RAPID 无此 secondary 缓冲区，owner 行数为 {audit['owner_counts']}。",'',
          '## 验证与局限','',
          '- 原始 CPU 分区和 RAPID 工具实际执行，源代码快照与哈希保留。',
          '- RAPID 调度前后全部记录通过双 64 位整行指纹多重集守恒校验；HET assignment 为全部输入行唯一分配。',
          '- 通信发送/接收总量一致；HET 更新次数 + pending 严格守恒；小例子验证严格阈值、冷启动、重复 ID、更新合并和 metadata。',
          '- ID 压缩只作用于活跃访问图；未出现的参数、全表容量与真实模型显存未被模拟。',
          '- 图划分输出、RAPID 异步 chunk 顺序可能因并发而变；本次单次运行不代表稳定性能区间。',
          '- 分区及 ACP profile 使用同一输入范围；不主张对未见数据泛化。',
          '- 未验证 AUC、训练收敛、链路带宽、网络延迟或端到端 GPU 时间。',
          '- 安装、命令、计量定义、尾批与源码差异详见 README.md。','']
(case/'REPORT.md').write_text('\n'.join(lines))
with (case/'comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
save(case/'summary.json',dict(timings=timings,traffic=rows,rapid_seconds=rapid_time,het_seconds=het_time,het_adapter_seconds=cfg['het_adapter_seconds'],rapid_vs_het_s0_percent=change))
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print('Report:',case/'REPORT.md')
