"""Build result tables only from completed measurement artifacts."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
rows=[]
for case in sorted((root/'results').iterdir()):
    if not (case/'complete.json').exists():continue
    cfg=json.loads((case/'config.json').read_text())
    h=json.loads((case/'hash_metrics.json').read_text())
    c=json.loads((case/'cps_metrics.json').read_text())
    rows.append(dict(case=case.name,dataset=cfg['dataset'],ranks=cfg['ranks'],batches=cfg['batches'],
        rows=cfg['rows'],repo=cfg['repo'],raw_locality_pct=100*c['local_hit_rate'],
        unique_locality_pct=100*c['unique_local_hit_rate'],hash_receive=h['unique_receive']['mean'],
        cps_receive=c['unique_receive']['mean'],cps_receive_mean_batch_peak=c['unique_receive']['mean_batch_peak'],
        cps_send_mean_batch_peak=c['unique_send']['mean_batch_peak'],
        cps_receive_peak_fp32_bytes=c['unique_receive']['mean_batch_peak']*(128 if cfg['dataset']=='taobao' else 32)*4,
        receive_peak_over_mean=c['unique_receive']['mean_batch_peak_over_mean'],
        receive_worst_batch_ratio=c['unique_receive']['max_batch_peak_over_mean'],
        send_peak_over_mean=c['unique_send']['mean_batch_peak_over_mean'],
        send_worst_batch_ratio=c['unique_send']['max_batch_peak_over_mean'],
        acp_receive_peak_over_mean=c['acp_receive']['mean_batch_peak_over_mean'],
        acp_receive_worst_batch_ratio=c['acp_receive']['max_batch_peak_over_mean'],
        acp_send_peak_over_mean=c['acp_send']['mean_batch_peak_over_mean'],
        acp_send_worst_batch_ratio=c['acp_send']['max_batch_peak_over_mean'],
        sum_io_peak_over_mean=c['unique_send_plus_receive']['mean_batch_peak_over_mean'],
        hash_acp=h['acp_receive']['mean'],cps_acp_peak=c['acp_receive']['mean_batch_peak'],
        hash_unique_dedup_rate=h['remote_deduplication_rate'],
        hash_acp_effective_dedup_rate=1-h['acp_receive']['mean']/(8192*sum(cfg['table_lengths'])),
        cps_acp_effective_dedup_rate=1-c['acp_receive']['mean_batch_peak']/(8192*sum(cfg['table_lengths'])),
        cps_acp_mean_effective_dedup_rate=1-c['acp_receive']['mean']/(8192*sum(cfg['table_lengths'])),
        hash_reduction_tables=sum(x['strategy']=='Reduction' for x in h['per_table']),
        comm_reduction_pct=100*(1-c['acp_receive']['mean_batch_peak']/h['unique_receive']['mean'])))
if rows:
    with (root/'results_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (root/'results_summary.json').write_text(json.dumps(rows,indent=2))
lines=['论文复现实验记录（仅列出已完成的真实数据实验）',
       '所有通信量单位：每 rank、每 batch 的前向 embedding 向量数；不含元数据和反向传播。',
       'Reduction通信量按每表每rank接收B*(G-1)计算，本地rank不计；B*G仅是论文公式(5)的参考值。',
       'locality 同时报告原始访问次数分母与去重 ID 分母，不混用。',
       '接收、发送及二者之和分别统计。逐批峰值的平均、最坏批次、rank 时间均值的峰值分别保留。',
       '非 GPU 耗时实验；未复现 V100 端到端吞吐、训练时延及 AUC。',
       'Criteo 日期 2015-02-21，由于论文未交代通信实验日期，本次自主选定。',
       '小规模试跑不等同于全数据结论；训练/分区与评估使用同一离线 trace。','']
for r in rows:
    lines += [r['case'],f"  ranks={r['ranks']}, batches={r['batches']}, rows={r['rows']:,}",
        f"  原始访问命中率={r['raw_locality_pct']:.3f}%; 去重 ID 命中率={r['unique_locality_pct']:.3f}%",
        f"  Hash 平均接收={r['hash_receive']:.2f}; CPS 平均接收={r['cps_receive']:.2f}; CPS 逐批接收峰值平均={r['cps_receive_mean_batch_peak']:.2f}",
        f"  接收 peak/mean={r['receive_peak_over_mean']:.4f}, 最坏批次={r['receive_worst_batch_ratio']:.4f}; 发送 peak/mean={r['send_peak_over_mean']:.4f}",
        f"  ACP+CPS 接收峰值相对 Hash Unique 均值降幅={r['comm_reduction_pct']:.3f}%",'']
for p in sorted((root/'results').glob('*ablation*/metrics.json')):
    lines += ['消融实验 '+p.parent.name,'  placement__schedule: raw locality %, unique locality %']
    for k,v in json.loads(p.read_text()).items():
        lines.append(f"  {k}: {100*v['local_hit_rate']:.3f}, {100*v['unique_local_hit_rate']:.3f}")
    lines.append('')
lines += ['审计与不一致记录见 discrepancies.txt；原始配置、命令、日志、逐 rank 计数保存在 results/ 各组目录。',
          '论文绘图脚本内写死的参考值没有被当作本次实测数据。']
(root/'REPORT.txt').write_text('\n'.join(lines)+'\n')
if rows:
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for dataset in ['taobao','criteo','kaggle']:
        selected=[r for r in rows if r['dataset']==dataset and 'update_Rapid' in r['repo']]
        best={}
        for r in selected:
            key=lambda v: ('day6' in v['case'],v['batches'])
            if r['ranks'] not in best or key(r)>key(best[r['ranks']]):best[r['ranks']]=r
        values=[best[g] for g in sorted(best)]
        if not values:continue
        x=[v['ranks'] for v in values]
        axes[0].plot(x,[v['cps_receive_mean_batch_peak'] for v in values],'o-',label=dataset+' receive peak')
        axes[0].plot(x,[v['hash_receive'] for v in values],'o--',label=dataset+' hash mean')
        axes[1].plot(x,[v['receive_peak_over_mean'] for v in values],'o-',label=dataset+' receive')
        axes[1].plot(x,[v['send_peak_over_mean'] for v in values],'s--',label=dataset+' send')
    axes[1].axhspan(1.1,1.2,color='green',alpha=.12,label='paper 1.10-1.20')
    axes[0].set_ylabel('Forward embedding vectors / rank / batch')
    axes[1].set_ylabel('Mean of per-batch max(rank) / mean(rank)')
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks([8,16,32,64,128],[8,16,32,64,128]);ax.set_xlabel('Simulated ranks');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Measured traces: largest completed updated-code run per configuration')
    fig.tight_layout();fig.savefig(root/'measured_scaling.png',dpi=160);plt.close(fig)
print('report updated:',len(rows),'completed cases')
