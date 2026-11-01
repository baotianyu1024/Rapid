"""Figure 6 trace statistics; MCR is an explicitly labelled uniform-ID sample."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from run_case import LENGTHS
from prepare_criteo_parquet import original

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--rows',type=int,default=1048576)
p.add_argument('--output',required=True);p.add_argument('--mcr-sample',type=int,default=512)
p.add_argument('--mcr-rows',type=int,default=1000000,help='Paper Section 5.1.2 uses one million samples')
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
raw=np.memmap(a.input,dtype='<u4',mode='r').reshape(-1,228)
n=len(raw) if a.rows==0 else min(a.rows,len(raw));data=raw[:n,14:];offsets=np.cumsum([0]+LENGTHS)
frequencies=[np.zeros(size,dtype=np.int64) for size in original.NUM_EMBEDDINGS]
cold=[];hist=np.zeros(4,dtype=np.int64)
for start in range(0,n,1000000):
    block=data[start:start+1000000].copy()
    for t,L in enumerate(LENGTHS):
        ids=np.sort(block[:,offsets[t]:offsets[t+1]],axis=1)
        mask=np.ones(ids.shape,dtype=bool);mask[:,1:]=ids[:,1:]!=ids[:,:-1]
        frequencies[t]+=np.bincount(ids[mask].astype(np.int64),minlength=len(frequencies[t]))
    print('frequency rows',min(n,start+1000000),'/',n,flush=True)
for t,L in enumerate(LENGTHS):
    counts=frequencies[t]
    observed=np.flatnonzero(counts)
    hist+=np.histogram(counts[observed]/n,bins=[0,.0001,.01,.1,1.0000001])[0]
    c=observed[counts[observed]<n*.0001]
    cold.append((np.uint64(t)<<np.uint64(32))|c.astype(np.uint64))
    print('sample-frequency table',t,'observed',len(observed),'cold',len(c),flush=True)
full_frequencies=frequencies
mcr_n=min(n,a.mcr_rows)
if mcr_n<n:
    frequencies=[];cold=[]
    for t,L in enumerate(LENGTHS):
        ids=np.sort(data[:mcr_n,offsets[t]:offsets[t+1]],axis=1)
        mask=np.ones(ids.shape,dtype=bool);mask[:,1:]=ids[:,1:]!=ids[:,:-1]
        counts=np.bincount(ids[mask].astype(np.int64),minlength=original.NUM_EMBEDDINGS[t])
        frequencies.append(counts)
        c=np.flatnonzero((counts>0)&(counts<mcr_n*.0001))
        cold.append((np.uint64(t)<<np.uint64(32))|c.astype(np.uint64))
all_cold=np.concatenate(cold)
rng=np.random.default_rng(42)
selected=all_cold[rng.choice(len(all_cold),min(a.mcr_sample,len(all_cold)),replace=False)]
selected.sort();mcr=[];records=[]
for t,L in enumerate(LENGTHS):
    chosen=selected[(selected>>np.uint64(32))==t]&np.uint64(0xffffffff)
    if not len(chosen):continue
    block=data[:mcr_n,offsets[t]:offsets[t+1]]
    postings={int(fid):[] for fid in chosen}
    for start in range(0,mcr_n,1000000):
        chunk=block[start:start+1000000]
        locations=np.argwhere(np.isin(chunk,chosen))
        for row,col in locations:postings[int(chunk[row,col])].append(int(row)+start)
    for fid in chosen:
        rows=np.unique(postings[int(fid)]);counter=Counter();own=(t<<32)|int(fid)
        assert len(rows)==frequencies[t][fid]
        for row in rows:
            keys=[]
            for u,U in enumerate(LENGTHS):
                ids=np.unique(data[row,offsets[u]:offsets[u+1]])
                keep=ids[frequencies[u][ids]<mcr_n*.0001]
                keys.extend((u<<32)|int(x) for x in keep if ((u<<32)|int(x))!=own)
            counter.update(keys)
        ratio=max(counter.values(),default=0)/len(rows)
        mcr.append(ratio);records.append(dict(table=t,id=int(fid),sample_frequency=len(rows),mcr=ratio))
    print('MCR sampled table',t,'done',len(records),flush=True)
# Traffic split before/after Unique on hash ownership, G=32, B=8192.
frequencies=full_frequencies
G=32;B=8192;nb=min(20,n//(G*B));traffic=np.zeros((2,2),dtype=np.int64)
for b in range(nb):
    for r in range(G):
        block=data[(b*G+r)*B:(b*G+r+1)*B]
        for t,L in enumerate(LENGTHS):
            ids=block[:,offsets[t]:offsets[t+1]].ravel()
            remote=ids[ids%G!=r]
            for kind,values in [(0,remote),(1,np.unique(remote))]:
                is_cold=frequencies[t][values]<n*.0001
                traffic[kind,0]+=np.count_nonzero(~is_cold)
                traffic[kind,1]+=np.count_nonzero(is_cold)
phat=float(np.mean(np.array(mcr)>.95));sample_n=len(mcr);z=1.959963984540054
center=(phat+z*z/(2*sample_n))/(1+z*z/sample_n)
half=z*np.sqrt(phat*(1-phat)/sample_n+z*z/(4*sample_n**2))/(1+z*z/sample_n)
results=dict(input=str(Path(a.input).resolve()),rows=n,scope='entire supplied trace' if n==len(raw) else 'prefix of shuffled trace',
    cold_definition='ID occurs in fewer than 0.01% of samples; per-sample duplicates removed',
    frequency_bins=['<0.01%','0.01%-1%','1%-10%','10%-100%'],frequency_bin_counts=hist.tolist(),
    frequency_bin_shares=(hist/hist.sum()).tolist(),mcr_scope='uniform sample of observed cold IDs in the specified MCR sample trace; not full-ID enumeration',
    mcr_rows=mcr_n,
    mcr_sample_size=len(mcr),mcr_seed=42,mcr_over_95_share=phat,
    mcr_over_95_wilson_95ci=[float(center-half),float(center+half)],
    mcr_singleton_share=float(np.mean([x['sample_frequency']==1 for x in records])),
    mcr_records=records,traffic_scope=dict(ranks=G,batch=B,batches=nb),
    traffic_counts_raw_then_unique_hot_then_cold=traffic.tolist(),
    traffic_cold_share_raw_then_unique=(traffic[:,1]/traffic.sum(axis=1)).tolist())
(out/'statistics.json').write_text(json.dumps(results,indent=2))
print('complete',out,flush=True)
