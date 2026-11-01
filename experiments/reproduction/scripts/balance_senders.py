"""Balance sender traffic by bijective same-table ID-owner swaps.

Only swap IDs with identical request-presence profiles on the two ranks.
Thus each rank's Unique receive count is exactly unchanged, as are total
traffic, table cardinality, physical ID range and per-table storage counts.
This is a new offline placement refinement, not part of the paper baseline.
The independent trace evaluator verifies all predicted communication counts.
"""
import argparse
import json
import shutil
from pathlib import Path
import time
import numpy as np
from measure import measure

p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',required=True)
p.add_argument('--candidates',type=int,default=40000);p.add_argument('--rounds',type=int,default=200)
p.add_argument('--target',type=float,default=1.2)
a=p.parse_args();case=Path(a.case).resolve();out=Path(a.output).resolve()
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True);cfg=json.loads((case/'config.json').read_text());G=cfg['ranks'];NB=cfg['batches'];B=8192
lengths=cfg['table_lengths'];dense=cfg['dense'];width=1+dense+sum(lengths);offsets=np.cumsum([0]+lengths)
if NB>20:raise ValueError('Use a bounded profiling trace (<=20 batches)')
arrays=[np.memmap(case/'scheduled'/f'dataset_{r}.bin',dtype='<u4',mode='r').reshape(-1,width) for r in range(G)]
baseline=np.load(case/'cps_counts.npz');send=baseline['send'].sum(axis=1).copy()
before=send.copy();tables=[];rng=np.random.default_rng(42)
for t in range(len(lengths)):
    lists=[]
    for b in range(NB):
        for r in range(G):lists.append(np.unique(arrays[r][b*B:(b+1)*B,1+dense+offsets[t]:1+dense+offsets[t+1]]))
    ids,counts=np.unique(np.concatenate(lists),return_counts=True)
    keep=np.flatnonzero(counts>=2)
    if len(keep)>a.candidates:
        high=keep[np.argsort(-counts[keep],kind='stable')[:a.candidates//2]]
        rest=np.setdiff1d(keep,high,assume_unique=False)
        keep=np.concatenate([high,rng.choice(rest,a.candidates-len(high),replace=False)])
    ids=np.sort(ids[keep]);presence=np.zeros((len(ids),NB,G),dtype=bool)
    if len(ids):
        for q,values in enumerate(lists):
            pos=np.searchsorted(ids,values);valid=pos<len(ids);pos=pos[valid];values=values[valid]
            pos=pos[ids[pos]==values];presence[pos,q//G,q%G]=True
    owners=(ids%G).astype(np.int32)
    powers=(np.uint64(1)<<np.arange(NB,dtype=np.uint64))[None,:,None]
    profiles=(presence*powers).sum(axis=1)
    weight=presence.sum(axis=2).astype(np.int64)
    tables.append(dict(ids=ids,newids=ids.copy(),owners=owners,
                       profiles=profiles,weight=weight))
    del lists,presence
    print('sender candidates table',t,len(tables[-1]['ids']),flush=True)
swaps=0;history=[]
for iteration in range(a.rounds):
    if np.all(send.max(axis=1)<=a.target*send.mean(axis=1)):break
    sources=np.argsort(-send.mean(axis=0))[:min(4,G)]
    destinations=np.argsort(send.mean(axis=0))[:min(12,G)]
    peak=send.max(axis=1);best=None;best_gain=0
    for source in sources:
        for dest in destinations:
            if source==dest:continue
            old_a=send[:,source];old_b=send[:,dest]
            for t,table in enumerate(tables):
                hot=np.flatnonzero(table['owners']==source)
                low=np.flatnonzero(table['owners']==dest)
                if not len(hot) or not len(low):continue
                weight=table['weight'];hot=hot[np.argsort(-weight[hot].mean(axis=1),kind='stable')]
                low=low[np.argsort(weight[low].mean(axis=1),kind='stable')]
                profiles=table['profiles']
                hot_codes=profiles[hot,source]|(profiles[hot,dest]<<np.uint64(NB))
                low_codes=profiles[low,source]|(profiles[low,dest]<<np.uint64(NB))
                for code in np.unique(hot_codes[:64]):
                    hi=hot[hot_codes==code];lo=low[low_codes==code]
                    m=min(len(hi),len(lo),1024)
                    if not m:continue
                    hi=hi[:m];lo=lo[:m]
                    change=np.cumsum(weight[hi]-weight[lo],axis=0)
                    new_a=old_a[None,:]-change;new_b=old_b[None,:]+change
                    feasible=((new_a>=0)&(new_b>=0)&(new_a<=peak)&(new_b<=peak)).all(axis=1)
                    gain=((old_a.astype(float)**2+old_b.astype(float)**2)[None,:]-new_a.astype(float)**2-new_b.astype(float)**2).sum(axis=1)
                    gain[~feasible]=-np.inf;k=int(gain.argmax())
                    if gain[k]>best_gain:
                        best_gain=float(gain[k]);best=(t,int(source),int(dest),hi[:k+1].copy(),lo[:k+1].copy(),change[k].copy())
    if best is None:break
    t,source,dest,hot,low,change=best;table=tables[t]
    saved=table['newids'][hot].copy();table['newids'][hot]=table['newids'][low];table['newids'][low]=saved
    table['owners'][hot]=dest;table['owners'][low]=source
    send[:,source]-=change;send[:,dest]+=change;swaps+=len(hot)
    assert np.array_equal(send.sum(axis=1),before.sum(axis=1))
    assert np.all(send.max(axis=1)<=peak)
    ratios=send.max(axis=1)/send.mean(axis=1)
    history.append(dict(iteration=iteration,table=t,source=source,destination=dest,swaps=len(hot),ratios=ratios.tolist()))
    print('sender refinement',iteration,'swaps',swaps,'ratios',ratios.tolist(),flush=True)
(out/'refinement.json').write_text(json.dumps(dict(candidate_limit=a.candidates,max_rounds=a.rounds,
    swaps=swaps,target=a.target,history=history,predicted_send=send.tolist(),
    guarantees='per-batch per-rank Unique receive counts unchanged; total traffic and storage unchanged',
    fit_scope='same offline trace, not held-out'),indent=2))
destdir=out/'scheduled';destdir.mkdir()
for t,table in enumerate(tables):
    assert np.array_equal(np.sort(table['ids']),np.sort(table['newids']))
    np.savez_compressed(out/f'table_{t}_remap.npz',old=table['ids'],new=table['newids'])
for r in range(G):
    with (destdir/f'dataset_{r}.bin').open('wb') as f:
        for start in range(0,NB*B,65536):
            records=arrays[r][start:min(start+65536,NB*B)].copy()
            for t,table in enumerate(tables):
                moved=table['ids']!=table['newids'];old=table['ids'][moved];new=table['newids'][moved]
                if not len(old):continue
                cols=records[:,1+dense+offsets[t]:1+dense+offsets[t+1]]
                pos=np.searchsorted(old,cols);valid=pos<len(old);safe=np.minimum(pos,len(old)-1)
                valid&=old[safe]==cols;cols[valid]=new[safe[valid]]
            records.tofile(f)
metrics,actual=measure([destdir/f'dataset_{r}.bin' for r in range(G)],G,B,NB,lengths,dense,True)
assert np.array_equal(actual['receive'],baseline['receive']), 'Receive counts changed'
assert np.array_equal(actual['unique'],baseline['unique']), 'Unique counts changed'
assert np.array_equal(actual['send'].sum(axis=1),send), 'Predicted sender counts disagree'
(out/'cps_metrics.json').write_text(json.dumps(metrics,indent=2));np.savez_compressed(out/'cps_counts.npz',**actual)
shutil.copyfile(case/'hash_metrics.json',out/'hash_metrics.json')
cfg.update(repo=str(Path(__file__).resolve().parents[1]/'working'),base_case=str(case),sender_refinement=True)
(out/'config.json').write_text(json.dumps(cfg,indent=2));(out/'complete.json').write_text(json.dumps(dict(completed=time.time())))
print('sender refinement verified complete',out,flush=True)
