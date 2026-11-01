from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
V=Path(__file__).resolve().parents[1];E=V/'results/incremental_extension';d=json.loads((E/'summary.json').read_text());s=json.loads((E/'stability.json').read_text());td={(r['policy'],r['window']):r['metrics']['vectors_per_8192'] for r in d['results']};fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained');colors=['#747474','#d98d27','#2468b4','#26945b']
for p,label,c in zip(['pure_unique','frozen','incremental','full_rebuild'],['Pure Unique','Frozen old placement','Incremental (no migration)','Full rebuild'],colors):
 keys=sorted(k for policy,k in td if policy==p);axes[0].plot(keys,[td[p,k]/1000 for k in keys],label=label,color=c,marker='o' if p=='full_rebuild' else None,linestyle='None' if p=='full_rebuild' else '-',markersize=4)
axes[0].set(xlabel='Arrival window',ylabel='F+B vectors per 8,192 samples (thousands)');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
ks=[k for k in range(1,d['windows']+1) if ('full_rebuild',k) in td];axes[1].plot(ks,[td['incremental',k]/td['full_rebuild',k] for k in ks],linestyle='None',marker='o',color='#2468b4');axes[1].axhline(1,color='gray',linestyle='--',linewidth=1);axes[1].axvspan(s['tail_start']-.5,s['tail_end']+.5,color='#2468b4',alpha=.10,label='Tail satisfying empirical criterion' if s['stable'] else 'Evaluated tail (not stable)');axes[1].set(xlabel='Arrival window',ylabel='Incremental / full rebuild traffic');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
(V/'figures').mkdir(exist_ok=True)
for suffix in ['png','pdf']:fig.savefig(V/f'figures/incremental_no_migration.{suffix}',dpi=180)
