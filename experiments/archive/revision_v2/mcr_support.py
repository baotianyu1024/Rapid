from pathlib import Path
from collections import Counter
import json,numpy as np
V=Path(__file__).resolve().parents[1];H=V.parent;W=H.parents[1];out=V/'results/mcr_support';out.mkdir(exist_ok=False)
L=[3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1];off=np.cumsum([0]+L);N=1000000;a=np.memmap(W/'data/processed/criteo/day6.bin',dtype='<u4',mode='r').reshape(-1,228)[:N,14:];freq=[]
for t in range(26):
 x=np.sort(a[:,off[t]:off[t+1]],axis=1);mask=np.ones(x.shape,bool);mask[:,1:]=x[:,1:]!=x[:,:-1];freq.append(np.bincount(x[mask]));print('freq',t,flush=True)
rng=np.random.default_rng(20261026);cohorts={};population={};selected=set()
for threshold in [1,5,10,20,50]:
 ids=[np.flatnonzero((f>=threshold)&(f<100)) for f in freq];starts=np.cumsum([0]+[len(x) for x in ids]);population[threshold]=int(starts[-1]);picked=[]
 for ix in rng.choice(starts[-1],min(512,starts[-1]),replace=False):
  t=int(np.searchsorted(starts,ix,side='right')-1);pair=(t,int(ids[t][ix-starts[t]]));picked.append(pair);selected.add(pair)
 cohorts[threshold]=picked
records={}
for t in range(26):
 ids=np.array(sorted(f for u,f in selected if u==t),np.uint32)
 if not len(ids):continue
 block=a[:,off[t]:off[t+1]];r,c=np.nonzero(np.isin(block,ids));f=block[r,c]
 for fid in ids:
  rows=np.unique(r[f==fid]);co=Counter();cross=Counter();key=(t<<32)|int(fid)
  for row in rows:
   for u in range(26):
    partners=np.unique(a[row,off[u]:off[u+1]]);partners=partners[freq[u][partners]<100];keys=[(u<<32)|int(v) for v in partners if ((u<<32)|int(v))!=key];co.update(keys)
    if u!=t:cross.update(keys)
  best=max(co.values(),default=0);bc=max(cross.values(),default=0);assert len(rows)==freq[t][fid];records[(t,int(fid))]=dict(table=t,id=int(fid),support=len(rows),max_pair_count=best,mcr=best/len(rows),cross_table_mcr=bc/len(rows))
 print('partners',t,len(records),flush=True)
def ci(k,n):
 z=1.95996398454;p=k/n;d=1+z*z/n;m=(p+z*z/(2*n))/d;q=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [m-q,m+q]
summary=[]
for x,picks in cohorts.items():
 z=[records[v] for v in picks];k=sum(v['mcr']>.95 for v in z);cross=sum(v['cross_table_mcr']>.95 for v in z);summary.append(dict(min_support=x,max_support=99,population=population[x],sampled=len(z),over_95=k,share=k/len(z),wilson_fraction_ci=ci(k,len(z)),cross_table_over_95=cross,cross_table_share=cross/len(z),mean_mcr=float(np.mean([v['mcr'] for v in z])),cohort_ids=picks))
result=dict(rows=N,cold='positive per-sample support <0.01%; support<100',candidate_partner='any other cold ID; minimum x applies only to target IDs',sampling='512 uniform IDs independently per cumulative threshold; exact in-profile MCR for each sampled ID; not full-ID enumeration',seed=20261026,summary=summary,records=list(records.values()),success=True)
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps([{k:v for k,v in z.items() if k!='cohort_ids'} for z in summary],indent=2))
