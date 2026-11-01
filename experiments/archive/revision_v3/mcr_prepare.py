from pathlib import Path
import numpy as np,json
V=Path(__file__).resolve().parents[1];W=V.parents[2];out=V/'results/mcr_all';out.mkdir(exist_ok=False);n=2097152;L=[3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1];src=W/'data/processed/criteo/day6.bin';a=np.memmap(src,dtype='<u4',mode='r',shape=(n,228));base=[0];col=14
with open(out/'frequency.u32','wb') as f:
 for t,l in enumerate(L):
  mx=int(a[:,col:col+l].max());counts=np.zeros(mx+1,np.uint32)
  for k in range(0,n,65536):
   x=np.sort(a[k:k+65536,col:col+l],axis=1);m=np.ones(x.shape,bool);m[:,1:]=x[:,1:]!=x[:,:-1];u,c=np.unique(x[m],return_counts=True);counts[u]+=c.astype(np.uint32)
  counts.tofile(f);base.append(base[-1]+len(counts));col+=l;print(t,base[-1],flush=True)
np.array(base,np.uint64).tofile(out/'table_offsets.u64');(out/'manifest.json').write_text(json.dumps(dict(rows=n,cold_upper_exclusive=n*.0001,min_supports=[10,20,50,100],source=str(src),lengths=L,base=base,all_eligible_ids=True,frequency_per_sample_deduplicated=True),indent=2))
