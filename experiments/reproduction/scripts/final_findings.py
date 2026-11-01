"""Compact decision ledger over matched-scope completed real-data experiments."""
import csv,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];refs=json.loads((R/'paper_reference_targets.json').read_text());cases={};variants={};allcases=[]
for p in sorted((R/'results').iterdir()):
 if not (p/'complete.json').exists():continue
 cfg=json.loads((p/'config.json').read_text());m=json.loads((p/'cps_metrics.json').read_text());e=(p,cfg,m);allcases.append(e)
 if cfg.get('base_case'):variants.setdefault(Path(cfg['base_case']).name,[]).append(e)
 if not cfg['repo'].endswith('update_Rapid'):continue
 key=cfg['dataset'],cfg['ranks'];priority=('day6' in p.name,cfg['batches'])
 if key not in cases or priority>cases[key][0]:cases[key]=(priority,e)
rows=[]
for (ds,g),(_,base) in sorted(cases.items()):
 pending=[base];candidates=[];seen=set()
 while pending:
  e=pending.pop()
  if e[0].name in seen:continue
  seen.add(e[0].name)
  if e[1]['rows']!=base[1]['rows']:continue
  candidates.append(e);pending.extend(variants.get(e[0].name,[]))
 balanced=[e for e in candidates if e[2]['acp_receive']['max_batch_peak_over_mean']<=1.200001]
 e=min(balanced or candidates,key=lambda e:(e[2]['acp_receive']['mean_batch_peak'],e[2]['acp_send']['mean_batch_peak_over_mean']))
 p,c,m=e;ref=refs[ds][str(g)]['cps'];peak=m['acp_receive']['mean_batch_peak'];before=base[2]['acp_receive']['mean_batch_peak']
 hashmean=json.loads((base[0]/'hash_metrics.json').read_text())['unique_receive']['mean'];paperhash=refs[ds][str(g)]['uniq']
 mintraffic=min(candidates,key=lambda e:e[2]['acp_receive']['mean_batch_peak'])
 rows.append(dict(dataset=ds,ranks=g,rows=c['rows'],paper_peak=ref,paper_hash_unique=paperhash,measured_hash_unique=hashmean,paper_reduction_from_hash_percent=100*(1-ref/paperhash),selected_reduction_from_hash_percent=100*(1-peak/hashmean),baseline_peak=before,selected_peak=peak,delta_percent=100*(peak/ref-1),improvement_percent=100*(1-peak/before),selected_mean=m['acp_receive']['mean'],receive_mean_batch_ratio=m['acp_receive']['mean_batch_peak_over_mean'],receive_worst_ratio=m['acp_receive']['max_batch_peak_over_mean'],send_worst_ratio=m['acp_send']['max_batch_peak_over_mean'],sum_worst_ratio=m['acp_send_plus_receive']['max_batch_peak_over_mean'],unique_locality_percent=100*m['unique_local_hit_rate'],raw_locality_percent=100*m['local_hit_rate'],traffic_meets_reference=peak<=ref,receive_balance_meets_120=m['acp_receive']['max_batch_peak_over_mean']<=1.200001,send_balance_meets_120=m['acp_send']['max_batch_peak_over_mean']<=1.200001,baseline_case=base[0].name,selected_case=p.name,minimum_traffic_peak=mintraffic[2]['acp_receive']['mean_batch_peak'],minimum_traffic_case=mintraffic[0].name,minimum_traffic_receive_mean_batch_ratio=mintraffic[2]['acp_receive']['mean_batch_peak_over_mean'],minimum_traffic_receive_worst_ratio=mintraffic[2]['acp_receive']['max_batch_peak_over_mean']))
with (R/'final_selected_results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['论文复现实测结果说明','',f'已完成并保留 {len(allcases)} 组主实验/改进实验（另有九格消融与数据特征统计）。',f'图11：15个数据集×rank配置中，{sum(r["traffic_meets_reference"] for r in rows)}/{len(rows)} 个接收峰值达到或优于论文参考；未达到者仍列出，不改写实验结果。','', '完整对比：paper_vs_measured_report.pdf；逐图索引：figure_comparison/INDEX.txt；逐项数据：figure_comparison/all_numeric_comparisons.csv。',
'主要基线是用户提供的update_Rapid原始行为；Rapid另有8rank对照实验。working是本次修改后的副本，未覆盖用户原始代码。','样本范围：Criteo 为完整 2015-02-21（204,846,845行）初始化与打乱后的前4,194,304行实验，并非全部24天；Kaggle完整训练集初始化后取相同4,194,304行；淘宝按代码采用展示广告数据，8rank使用全部405个完整batch，其余rank各20batch。','每rank batch size=8192；通信量单位为远程接收的前向embedding向量数，本地命中不计，不包含ID元数据、反向梯度或实际传输耗时。GPU rank由CPU精确计数模拟。','', '图11所选结果（接收峰值=各batch最大rank接收量的平均；偏差为相对论文，负值更好）：','数据集 ranks 论文值 原始值 改进/所选值 偏差% RX最坏max/mean TX最坏max/mean']
for r in rows:lines.append(f'{r["dataset"]:7} {r["ranks"]:3} {r["paper_peak"]:9.2f} {r["baseline_peak"]:11.2f} {r["selected_peak"]:11.2f} {r["delta_percent"]:+7.2f}% {r["receive_worst_ratio"]:.4f} {r["send_worst_ratio"]:.4f}')
lines+=['','选择规则：相同样本的候选中，先选择每个batch接收峰均比<=1.20者，再选择平均接收峰值最低者；若没有满足均衡者，选择接收峰值最低者。初始化、搜索预算、目标函数及owner迁移的变化均保存在各case配置中。每个batch都满足1.20是本复现的保守筛选条件，论文未明确声明最坏batch上限；同时给出平均batch比例和最坏batch比例。此规则不代表发送侧已均衡。','原始与全部改进结果均保留在results_summary.csv；不隐藏负向结果。已经优于论文的结果不调差。final_selected_results.csv 另列不限制均衡时的最小通信量，避免将负载均衡的代价隐藏。', '', '主要口径与未复现项：','1. 端到端图2/8/9和需要实际训练的图12(a)未执行；图1/3/5/7及图4(a)为示意图，无数值复现。','2. 图4(b)(c)的Reduction柱使用非空远端peer模型，与公式(5)及图11的B*G模型冲突。当前所有实测通信量统一只计远程接收，Reduction为B*(G-1)，Unique按(table_id,fid)去重；图4的Reduction、ACP绝对值及60%结论不再作为有效复现目标。','3. 图6频率、冷热占比是数据分布特征，不能把数值变化简单当作算法提升；频率遍历整日，MCR用1,000,000行窗口、512个冷ID抽样，置信区间见统计JSON。','4. 图12(b)左图标为remote IDs，但数值乘以非本地比例才得到图11通信量，说明柱值实际是总unique IDs；右图与表3使用unique-ID locality，而正文“embedding requests”定义容易被理解为原始请求口径。两种分母均保留。','5. 论文图12的unique数×非本地比例与图11的CPS数几乎相同，但正文称图11为瓶颈rank峰值，存在均值/峰值聚合口径疑点；仍按正文的最繁忙rank严格评判，均值只作附加核对。','6. 图10对ACP/RAPID曲线使用1-通信payload/(B*sumL)，其中payload可含Reduction成本；严格说是有效通信缩减率，不全是公式(4)定义的unique-ID去重率。','7. 接收侧满足10%–20%上限，并不证明发送侧或收发合计满足该上限。论文未说明方向；未达标的发送侧偏斜明确保留。','8. 分区与策略在当前离线trace上生成；这些是in-sample算法计数，不是独立留出集泛化结论。METIS为依据论文描述补充的初始化，不能声称恢复了作者未提供的全部参数。','9. 原始两份code未修改；所有修复和实验性改进位于reproduction/working，原始文件SHA校验在final_audit.json。working_changes.patch可审阅8个源码文件的改动。',
'10. Criteo实测Hash Unique基线本身高于论文约18%，因此同时提供相对Hash的降幅；它帮助区分输入分布差异和算法收益，但不替代绝对通信量复现。具体日期/数据范围未在论文中明确，不能据此认定差异的唯一原因。',
'11. METIS的通信改善伴有存储取舍：Criteo128导出ID数最大/平均从1.241变为1.309。StorageBalance展示各分区计数，不宣称实际GPU分配内存均衡。','12. Criteo8/16/32/64最终达标结果使用新增的离线交替优化：固定样本时选择最佳ID owner，固定owner时重新分配样本，并以等收益owner并列项压平接收峰值。Rapid和update_Rapid原样运行均没有这一步，不能把改进结果归为原始算法复现。每个owner改进case都保存可逆ID映射并逐记录逆验证；该方法拟合同一评测trace，尚未证明对未见trace泛化。','', '论文内部口径审计见 PAPER_INTERNAL_CONSISTENCY_AUDIT.md；代码改动与负向实验详见discrepancies.txt；数据来源、运行命令及环境见RUNBOOK.txt。']
(R/'FINAL_FINDINGS.txt').write_text('\n'.join(lines)+'\n');print('Selected results:',sum(r['traffic_meets_reference'] for r in rows),'/',len(rows),'traffic targets met')
