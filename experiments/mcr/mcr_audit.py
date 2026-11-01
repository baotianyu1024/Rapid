from pathlib import Path
import argparse
from collections import Counter
import numpy as np,json
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out;m=json.loads((out/'manifest.json').read_text());f=np.memmap(out/'frequency.u32',dtype=np.uint32,mode='r');r=np.fromfile(out/'all_targets.u32',np.uint32).reshape(-1,4);eligible=np.flatnonzero((f>=10)&(f<m['cold_upper_exclusive']));assert np.array_equal(np.sort(r[:,0].astype(np.int64)),eligible);assert np.array_equal(f[r[:,0]],r[:,1]);assert np.all(r[:,2]<=r[:,1]);rows=[]
for t in [10,20,50,100]:
 z=r[r[:,1]>=t];rows.append(dict(min_support=t,eligible_ids=len(z),mcr_above_95_ids=int(np.count_nonzero(z[:,2].astype(np.uint64)*100>z[:,1].astype(np.uint64)*95)),mcr_above_95_percent=float(np.mean(z[:,2].astype(np.uint64)*100>z[:,1].astype(np.uint64)*95)*100),mean_mcr=float(np.mean(z[:,2]/z[:,1]))))
a=np.memmap(m['source'],dtype='<u4',mode='r',shape=(m['rows'],228));base=np.array(m['base']);cols=np.cumsum([14]+m['lengths']);rng=np.random.default_rng(20261027);chosen=[]
for lo,hi in [(10,20),(20,50),(50,100),(100,210)]:
 for full in [True,False]:
  q=np.flatnonzero((r[:,1]>=lo)&(r[:,1]<hi)&((r[:,1]==r[:,2])==full));chosen.extend(rng.choice(q,min(2,len(q)),replace=False).tolist())
audit=[]
for j in chosen:
 uid,n,best,witness=map(int,r[j]);t=int(np.searchsorted(base,uid,side='right')-1);raw=uid-int(base[t]);ix=np.flatnonzero(np.any(a[:,cols[t]:cols[t+1]]==raw,axis=1));assert len(ix)==n
 c=Counter()
 for row in a[ix]:
  for k in range(len(m['lengths'])):
   ids=np.unique(row[cols[k]:cols[k+1]].astype(np.uint64)+base[k]);ids=ids[(f[ids]>0)&(f[ids]<m['cold_upper_exclusive'])];c.update(int(i) for i in ids if i!=uid)
 exact=max(c.values(),default=0);assert exact==best,(uid,best,exact);assert c[witness]==best
 audit.append(dict(id=uid,support=n,best=best,exact_python_counter=exact));print(audit[-1],flush=True)
(out/'summary.json').write_text(json.dumps(dict(**m,statistics=rows,audit=dict(all_eligible_ids_enumerated=True,frequency_and_bounds_checked_for_all=True,independent_counter_checks=audit,sampling_only_for_code_validation=True)),indent=2)+'\n')
print(json.dumps(rows,indent=2))
