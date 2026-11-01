"""Measured counterparts to Figures 10/11/12. References never feed computations."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'results_summary.json').read_text())
fig,axes=plt.subplots(3,3,figsize=(15,12))
for column,dataset in enumerate(['criteo','kaggle','taobao']):
 selected={}
 for r in rows:
  if r['dataset']!=dataset or 'update_Rapid' not in r['repo']:continue
  key=lambda v:('day6' in v['case'],v['batches'])
  if r['ranks'] not in selected or key(r)>key(selected[r['ranks']]):selected[r['ranks']]=r
 values=[selected[g] for g in sorted(selected)];x=[r['ranks'] for r in values]
 for field,label in [('hash_receive','Hash Unique mean'),('hash_acp','Hash ACP mean'),('cps_acp_peak','CPS ACP receive peak')]:axes[0,column].plot(x,[r[field] for r in values],'o-',label=label)
 for field,label in [('hash_unique_dedup_rate','Unique eta'),('hash_acp_effective_dedup_rate','ACP effective eta'),('cps_acp_effective_dedup_rate','CPS ACP effective eta')]:axes[1,column].plot(x,[r[field] for r in values],'o-',label=label)
 for field,label in [('raw_locality_pct','Raw-access denominator'),('unique_locality_pct','Unique-ID denominator')]:axes[2,column].plot(x,[r[field] for r in values],'o-',label=label)
 axes[0,column].set_title(dataset)
 for ax in axes[:,column]:
  ax.set_xscale('log',base=2);ax.set_xticks([8,16,32,64,128],[8,16,32,64,128]);ax.set_xlabel('Logical ranks');ax.legend(fontsize=8);ax.grid(alpha=.2)
axes[0,0].set_ylabel('Forward vectors / rank / batch')
axes[1,0].set_ylabel('1 - transferred vectors / raw accesses')
axes[2,0].set_ylabel('Local hit rate (%)')
fig.suptitle('Real-data measurements; full-day Criteo preferred when complete\nCPU traces, not GPU training; exact case scopes in results_summary.csv')
fig.tight_layout();fig.savefig(root/'measured_paper_panels.png',dpi=160)
