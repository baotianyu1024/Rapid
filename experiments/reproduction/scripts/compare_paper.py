"""Compare measured artifacts with paper references without modifying measurements."""
import csv
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
refs=json.loads((ROOT/'paper_reference_targets.json').read_text())
rows=[];lines=['论文数值对照（未达到的目标保留为未复现）',
'参考值分两类：paper_text_targets 为论文正文；Figure 11 数字来自随附绘图脚本常数，仅作为参考，非独立图像数字化。',
'每行注明数据范围与 batch 数；只纳入 complete.json 已提交的结果。完整下载/全局打乱不等于对全量样本执行了 CPS。',
'通信口径：远程接收的前向 embedding 向量数/rank/batch，本地不计；Reduction 每表 B*(G-1)。CPS 另列发送峰值用于诊断，但不混入通信总量。','']
for case in sorted((ROOT/'results').iterdir()):
 if not (case/'complete.json').exists():continue
 cfg=json.loads((case/'config.json').read_text());g=cfg['ranks'];ds=cfg['dataset']
 h=json.loads((case/'hash_metrics.json').read_text());c=json.loads((case/'cps_metrics.json').read_text())
 ref=refs[ds][str(g)]
 scope=cfg.get('input_metadata',{}).get('scope',cfg['input'])
 for key,value,target in [('Figure11 Hash Unique mean',h['unique_receive']['mean'],ref['uniq']),
   ('Figure11 Hash ACP mean',h['acp_receive']['mean'],ref['acp']),
   ('Figure11 CPS ACP receive batch peak',c['acp_receive']['mean_batch_peak'],ref['cps']),
   ('Figure11 CPS Unique receive batch peak',c['unique_receive']['mean_batch_peak'],ref['uniq_cps'])]:
  rows.append(dict(case=case.name,dataset=ds,ranks=g,batches=cfg['batches'],evaluated_rows=cfg['rows'],input_scope=scope,metric=key,measured=value,paper_reference=target,relative_error_pct=100*(value/target-1)))
 if ds=='criteo' and str(g) in refs['paper_text_targets']['fig12_locality']:
  target=100*refs['paper_text_targets']['fig12_locality'][str(g)]
  for label,value in [('Figure12 unique locality pct',100*c['unique_local_hit_rate']),('Figure12 raw locality pct (alternative denominator)',100*c['local_hit_rate'])]:
   rows.append(dict(case=case.name,dataset=ds,ranks=g,batches=cfg['batches'],evaluated_rows=cfg['rows'],input_scope=scope,metric=label,measured=value,paper_reference=target,relative_error_pct=100*(value/target-1)))
 expected={'criteo':{8:2,16:1,32:0,64:0,128:0},'kaggle':{8:1,16:1,32:1,64:0,128:0},'taobao':{8:0,16:0,32:0,64:0,128:0}}[ds][g]
 actual=[t['table'] for t in h['per_table'] if t['strategy']=='Reduction']
 total_h=sum(t['unique_count_mean'] for t in h['per_table']);total_c=sum(t['unique_count_mean'] for t in c['per_table'])
 lines += [case.name,f"  输入范围={scope}; 实验行数={cfg['rows']:,}; G={g}, batches={cfg['batches']}",
 f"  Figure11 Hash Unique: 实测 {h['unique_receive']['mean']:.2f}, 参考 {ref['uniq']}, 偏差 {100*(h['unique_receive']['mean']/ref['uniq']-1):+.2f}%",
 f"  Figure11 CPS ACP 接收峰值: 实测 {c['acp_receive']['mean_batch_peak']:.2f}, 参考 {ref['cps']}, 偏差 {100*(c['acp_receive']['mean_batch_peak']/ref['cps']-1):+.2f}%",
 f"  Unique 每批接收 max/mean 平均={c['unique_receive']['mean_batch_peak_over_mean']:.6f}, 最坏={c['unique_receive']['max_batch_peak_over_mean']:.6f}; 最坏<=1.20: {c['unique_receive']['max_batch_peak_over_mean']<=1.200001}",
 f"  Unique 每批发送 max/mean 平均={c['unique_send']['mean_batch_peak_over_mean']:.6f}, 最坏={c['unique_send']['max_batch_peak_over_mean']:.6f}; 最坏<=1.20: {c['unique_send']['max_batch_peak_over_mean']<=1.200001}",
 f"  Table2 Hash Reduction 表={actual}, 数量={len(actual)}, 参考数量={expected}; {'一致' if len(actual)==expected else '不一致'}",
 f"  Figure12 总 unique IDs（含本地）Hash={total_h:.2f}, CPS={total_c:.2f}; remote-only CPS={c['unique_receive']['mean']:.2f}",
 f"  raw 命中率={100*c['local_hit_rate']:.4f}%; unique 命中率={100*c['unique_local_hit_rate']:.4f}%"]
 lines.append(f"  Figure10 Eq4 Hash Unique η={h['remote_deduplication_rate']:.6f}; Hash ACP effective η={1-h['acp_receive']['mean']/(8192*sum(cfg['table_lengths'])):.6f}; CPS ACP peak-derived effective η={1-c['acp_receive']['mean_batch_peak']/(8192*sum(cfg['table_lengths'])):.6f}")
 if ds=='criteo' and g==8:
  lines.append(f"  dense remote receive 成本口径 table20 Unique/Reduction={h['per_table'][20]['unique_receive_mean']/(8192*(g-1)):.4f}; Hash ACP/Unique={h['acp_receive']['mean']/h['unique_receive']['mean']:.4f}。图4非空peer口径与公式(5)/图11冲突。")
 lines.append('')
with (ROOT/'PAPER_COMPARISON.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for p in sorted((ROOT/'results').glob('*cold*/statistics.json')):
 d=json.loads(p.read_text());lines += ['Figure6 '+p.parent.name,
 f"  frequency rows={d['rows']:,}; <0.01% IDs={100*d['frequency_bin_shares'][0]:.4f}%（论文96.62%）",
 f"  MCR>95%={100*d['mcr_over_95_share']:.4f}%（论文79.3%）; 频率与共现窗口={d.get('mcr_rows',d['rows']):,} 行，均匀抽取 {d['mcr_sample_size']} 个冷 ID，非全 ID 枚举；95% CI={d.get('mcr_over_95_wilson_95ci','旧实验未保存')}",
 f"  阈值 cold definition: raw cold={100*d['traffic_cold_share_raw_then_unique'][0]:.4f}%, unique cold={100*d['traffic_cold_share_raw_then_unique'][1]:.4f}%",
 '  注意论文/代码 Top40000 hot 与 <0.01% sample-frequency cold 不是互补定义，不能把两个口径混用。','']
for p in sorted((ROOT/'results').glob('*hot*split.json')):
 lines+=['Figure6 Top40000 traffic split '+p.name,p.read_text(),'']
for p in sorted((ROOT/'results').glob('*ablation*/metrics.json')):
 d=json.loads(p.read_text());lines+=['Table3 九格消融 '+p.parent.name,'  同一 placement、样本和 batch；两个分母分别输出。']
 for k,v in d.items():lines.append(f"  {k}: raw={100*v['local_hit_rate']:.3f}%, unique={100*v['unique_local_hit_rate']:.3f}%")
 lines.append('  论文参考 Hash[3.13,5.49,5.44], FreqRR[3.12,4.77,5.61], Ours[3.26,24.75,56.5]，仅 Criteo G=32 可对照。\n')
for p in sorted((ROOT/'results').glob('*figure4.json')):
 d=json.loads(p.read_text());r=d['batch_size']*(d['ranks']-1)
 unique=np.asarray(d['per_table_unique']);acp=np.minimum(unique,r)
 lines += ['Figure4 统一远程接收口径 '+p.stem,
 f"  All Reduction={r*len(unique):.2f}；All Unique={unique.sum():.2f}；ACP={acp.sum():.2f}（逐表 Unique/Reduction 取较小值）",
 f"  Table20 Unique/Reduction={unique[20]/r:.6f}；ACP/Unique={acp.sum()/unique.sum():.6f}",
 f"  原图非空远端 peer 值仅作错误口径存档：Reduction={d['total_reduction']:.2f}（图示567000），ACP={d['total_acp']:.2f}（图示228000）；与公式(5)/图11不兼容。",'']
storage=ROOT/'storage_balance.json'
if storage.exists():
 lines += ['Section7.8 存储均衡（ID 数量，独立于通信量）']
 for name,d in json.loads(storage.read_text()).items():
  lines.append(f"  {name}: max/mean={d['max_over_mean']:.6f}; 各rank均<=ceil(total/G): {d['all_within_ceil_capacity']}")
 lines.append('')
lines += ['覆盖与尚未复现项',
'已执行 CPU 逻辑指标：Figure4 按统一公式重新计算的 ACP、Figure6 冷热/共现统计（MCR 抽样）、Figure10 有效通信缩减率、Figure11 通信与 ACP 策略、Figure12 两种 locality/unique 数量、Table2、Table3、接收/发送 max/mean、swap 和存储约束额外消融。',
'Figure2/8/9 GPU 训练性能、端到端时间、V100 8rank 速度、Figure12a AUC 及训练收敛未测，不能由本 CPU trace 推断。Figure1/3/5/7 是架构或流程示意图，无待复现测量值。',
'原默认多线程 Taobao trace 全部样本守恒，但四个 global batch 都不守恒；修正为单线程后逐 batch 守恒。',
'未解决：论文 seed graph/METIS 未随代码提供；发送端负载在高 rank 数明显不平衡；若数值存在差距，不能通过改常数、混分母或只挑选最小 pilot 宣称复现。',
'详见 RUNBOOK.txt、discrepancies.txt。']
(ROOT/'PAPER_COMPARISON.txt').write_text('\n'.join(lines)+'\n')
print('comparison rows',len(rows))
