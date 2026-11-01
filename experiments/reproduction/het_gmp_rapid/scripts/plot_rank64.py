from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[1]/'results/rank64_balance');a=p.parse_args();O=a.case.resolve()
s=json.loads((O/'comparison.json').read_text());names=['RAPID_initial','RAPID_balanced','GMP_s0'];labels=['RAPID initial','RAPID latest full pipeline','GMP s=0'];colors=['#b56b2b','#276cad','#278764'];x=np.arange(64)
fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
for name,label,color in zip(names,labels,colors):
 d=s['storage'][name]['primary'];axs[0,0].plot(x,np.array(d['values'])/d['mean'],label=label,color=color,lw=1.5)
 for ax,key in [(axs[0,1],'forward_receive'),(axs[1,0],'forward_send')]:
  d=s['traffic'][name]['full_batches'][key]['rank_means'];ax.plot(x,np.array(d['values'])/d['mean'],label=label,color=color,lw=1.5)
for ax,title in zip([axs[0,0],axs[0,1],axs[1,0]],['Primary embedding rows by rank','Forward receive: rank means / overall mean','Forward send: rank means / overall mean']):
 ax.axhline(1,color='gray',ls='--',lw=.8);ax.set(xlabel='Logical rank',ylabel='Ratio to mean',title=title);ax.grid(alpha=.2)
axs[0,0].legend(fontsize=9)
pos=np.arange(2)
for j,(name,label,color) in enumerate(zip(names,labels,colors)):
 vals=[s['traffic'][name]['full_batches'][k]['mean_step_max_over_mean'] for k in ['forward_receive','forward_send']]
 bars=axs[1,1].bar(pos+(j-1)*.25,vals,.25,label=label,color=color)
 axs[1,1].bar_label(bars,fmt='%.3f',fontsize=9)
axs[1,1].set_xticks(pos,['Receive','Send']);axs[1,1].set(ylabel='Mean of per-step max / mean',title='Instantaneous imbalance: full batches only');axs[1,1].axhline(1,color='gray',ls='--',lw=.8);axs[1,1].grid(axis='y',alpha=.2)
fig.suptitle('64 logical ranks | Same 4,194,304 Day6 samples | CPU trace replay',fontsize=14)
fig.savefig(O/'balance.png',dpi=170);fig.savefig(O/'balance.svg');plt.close(fig)
# Absolute storage includes mirror parameters; not optimizer/version/gradient bytes.
fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for name,label,color in zip(names,labels,colors):
 d=s['storage'][name]['resident_parameter_rows'];axs[0].plot(x,np.array(d['values'])/1e6,label=label,color=color)
axs[0].set(xlabel='Logical rank',ylabel='Parameter vectors (millions)',title='Resident primary + mirror parameter rows');axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
xx=np.arange(3)
f=[s['traffic'][n]['normalized_forward_per_8192']/1000 for n in names];b=[(s['traffic'][n]['normalized_forward_backward_per_8192']-s['traffic'][n]['normalized_forward_per_8192'])/1000 for n in names]
axs[1].bar(xx,f,label='Forward',color='#628eb5');axs[1].bar(xx,b,bottom=f,label='Backward',color='#a8bfd3');axs[1].set_xticks(xx,['RAPID initial','RAPID latest','GMP s=0']);axs[1].set(ylabel='Vectors per 8192 samples (thousands)',title='All rows and partial tails included');axs[1].legend();axs[1].grid(axis='y',alpha=.2)
fig.savefig(O/'storage_and_volume.png',dpi=170);fig.savefig(O/'storage_and_volume.svg');plt.close(fig)
