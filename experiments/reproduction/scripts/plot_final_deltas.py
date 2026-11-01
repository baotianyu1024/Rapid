"""Make remaining communication gaps legible beside the log-scale paper plot."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];rows=list(csv.DictReader((R/'final_selected_results.csv').open()))
fig,axes=plt.subplots(1,3,figsize=(14,4.5))
for ax,ds in zip(axes,['criteo','kaggle','taobao']):
 rr=sorted([r for r in rows if r['dataset']==ds],key=lambda r:int(r['ranks']));x=np.arange(len(rr));before=[100*(float(r['baseline_peak'])/float(r['paper_peak'])-1) for r in rr];after=[float(r['delta_percent']) for r in rr]
 ax.bar(x-.19,before,.38,color='lightgray',label='Supplied-code baseline')
 ax.bar(x+.19,after,.38,color=['#bc3b35' if v>0 else '#368552' for v in after],label='Selected measured result')
 for i,v in enumerate(after):ax.annotate(f'{v:+.1f}%',(i+.19,v),xytext=(0,4 if v>=0 else -12),textcoords='offset points',ha='center',fontsize=8)
 ax.axhline(0,color='black',lw=1);ax.set_xticks(x,[r['ranks'] for r in rr]);ax.set_xlabel('Logical ranks');ax.set_title(ds);ax.grid(axis='y',alpha=.2);ax.margins(y=.2);ax.legend(fontsize=7)
axes[0].set_ylabel('Receive peak relative to paper (%)')
fig.suptitle('Figure11 deviations: below zero is better; same measured scopes as Figure11')
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(R/'figure_comparison'/f'Figure11Deviation.{ext}',dpi=180)
