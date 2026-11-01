"""Show receive and send balance separately for the controlled 32-rank pilot."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
names=['criteo_32_update_Rapid_4b','criteo_32_ratio_extended_4b']
fig,axes=plt.subplots(1,2,figsize=(12,4))
for name,label in zip(names,['Original default schedule','Extended ratio search']):
 d=json.loads((root/'results'/name/'cps_metrics.json').read_text())
 for ax,key in zip(axes,['unique_receive','unique_send']):
  counts=np.array(d[key]['rank_means']);ax.plot(np.arange(len(counts)),counts/counts.mean(),'o-',markersize=3,label=label)
for ax,title in zip(axes,['Receive vectors: rank mean / cluster mean','Send vectors: rank mean / cluster mean']):
 ax.axhline(1.2,color='red',ls=':',label='1.20 bound');ax.set_title(title);ax.set_xlabel('Rank');ax.set_ylabel('Relative load');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Criteo pilot, G=32, 4 global batches; same placement and samples')
fig.tight_layout();fig.savefig(root/'rank_balance_32.png',dpi=180)
