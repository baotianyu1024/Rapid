"""Sender-aware ID migration with nonincreasing per-batch total traffic.

Migrate an ID only if the new owner consumes it whenever the old owner does.
This changes per-rank receive counts, unlike the overly restrictive swap-only
experiment, but never increases their original per-batch maximum. No replicas
or dropped IDs are introduced. Fixed offline trace, not held-out inference.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from measure import measure
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',required=True)
p.add_argument('--per-owner',type=int,default=40000);p.add_argument('--rounds',type=int,default=300);p.add_argument('--target',type=float,default=1.18)
p.add_argument('--receive-target',type=float,default=1.20)
a=p.parse_args();case=Path(a.case).resolve();out=Path(a.output).resolve()
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True);cfg=json.loads((case/'config.json').read_text());G=cfg['ranks'];NB=cfg['batches'];B=cfg.get('batch_size',8192)
lengths=cfg['table_lengths'];dense=cfg['dense'];width=1+dense+sum(lengths);offsets=np.cumsum([0]+lengths)
if NB>63:raise ValueError('Presence profiles support at most63 batches')
arrays=[np.memmap(case/'scheduled'/f'dataset_{r}.bin',dtype='<u4',mode='r').reshape(-1,width) for r in range(G)]
base=np.load(case/'cps_counts.npz');send_detail=base['send'].copy();recv_detail=base['receive'].copy()
send=send_detail.sum(axis=1);recv=recv_detail.sum(axis=1);before_send=send.copy();before_recv=recv.copy()
receive_limit=recv.max(axis=1);send_target=send.mean(axis=1)*a.target
receive_ratio_limit=np.maximum(a.receive_target,receive_limit/recv.mean(axis=1))
storage=np.zeros(G,dtype=np.int64);tables=[];powers=np.uint64(1)<<np.arange(NB,dtype=np.uint64)
for t in range(len(lengths)):
 lists=[]
 for b in range(NB):
  for r in range(G):lists.append(np.unique(arrays[r][b*B:(b+1)*B,1+dense+offsets[t]:1+dense+offsets[t+1]]))
 all_ids,counts=np.unique(np.concatenate(lists),return_counts=True);owners=(all_ids%G).astype(np.int32)
 storage+=np.bincount(owners,minlength=G)
 next_local=np.array([int(all_ids[owners==r].max())//G+1 if np.any(owners==r) else 0 for r in range(G)],dtype=np.uint64)
 keep=[]
 for r in range(G):
  available=np.flatnonzero((owners==r)&(counts>1));order=np.argsort(-counts[available],kind='stable');keep.extend(available[order[:a.per_owner]])
 ids=np.sort(all_ids[np.array(keep,dtype=np.int64)]);profiles=np.zeros((len(ids),G),dtype=np.uint64);weight=np.zeros((len(ids),NB),dtype=np.int64)
 for q,values in enumerate(lists):
  if not len(ids):break
  pos=np.searchsorted(ids,values);valid=pos<len(ids);pos=pos[valid];values=values[valid];pos=pos[ids[pos]==values]
  profiles[pos,q%G]|=powers[q//G];weight[pos,q//G]+=1
 tables.append(dict(ids=ids,newids=ids.copy(),owners=(ids%G).astype(np.int32),profiles=profiles,weight=weight,next_local=next_local,mean_weight=weight.mean(axis=1)))
 print('migration candidates table',t,len(ids),flush=True)
 del lists,all_ids,counts,owners
before_storage=storage.copy();storage_limit=int(storage.max());history=[];moves=0
for iteration in range(a.rounds):
 if np.all(send.max(axis=1)<=send_target):break
 best=None;best_gain=0.;sources=np.argsort(-(send/send.mean(axis=1)[:,None]).max(axis=0))[:min(2,G)];destinations=np.argsort(send.mean(axis=0))[:min(12,G)]
 for source in sources:
  candidate_tables=np.argsort(-send_detail[:,:,source].mean(axis=0))[:min(5,len(tables))]
  for t in candidate_tables:
   table=tables[t];hi=np.flatnonzero(table['owners']==source)
   hi=hi[np.argsort(-table['mean_weight'][hi],kind='stable')]
   for dest in destinations:
    if source==dest or storage[dest]>=storage_limit:continue
    indices=hi[(table['profiles'][hi,source]&~table['profiles'][hi,dest])==0]
    indices=indices[:min(8192,storage_limit-storage[dest])]
    if not len(indices):continue
    source_presence=((table['profiles'][indices,source,None]&powers[None,:])!=0).astype(np.int64)
    dest_presence=((table['profiles'][indices,dest,None]&powers[None,:])!=0).astype(np.int64)
    source_delta=table['weight'][indices]-source_presence;dest_delta=table['weight'][indices]-dest_presence
    individual=(dest_delta<=np.maximum(send_target-send[:,dest],0)).all(axis=1)
    indices=indices[individual];source_presence=source_presence[individual];dest_presence=dest_presence[individual]
    if not len(indices):continue
    delta=np.cumsum(source_delta[individual],axis=0);add=np.cumsum(dest_delta[individual],axis=0)
    local=np.cumsum(source_presence,axis=0);new_local=np.cumsum(dest_presence,axis=0)
    ns=send[:,source][None,:]-delta;nd=send[:,dest][None,:]+add
    rs=recv[:,source][None,:]+local;rd=recv[:,dest][None,:]-new_local
    new_mean=(recv.sum(axis=1)[None,:]+local-new_local)/G
    others=[r for r in range(G) if r not in [source,dest]]
    other_peak=recv[:,others].max(axis=1) if others else np.zeros(NB)
    new_recv_peak=np.maximum(np.maximum(rs,rd),other_peak)
    feasible=((ns>=0)&(nd<=np.maximum(send[:,dest][None,:],new_mean*a.target))&(new_recv_peak<=receive_limit)&(new_recv_peak<=new_mean*receive_ratio_limit)&(rd>=0)).all(axis=1)
    gain=((send[:,source].astype(float)**2+send[:,dest].astype(float)**2)[None,:]-ns.astype(float)**2-nd.astype(float)**2).sum(axis=1);gain[~feasible]=-np.inf
    k=int(gain.argmax())
    if gain[k]>best_gain:
     best_gain=float(gain[k]);best=(int(t),int(source),int(dest),indices[:k+1].copy(),delta[k].copy(),add[k].copy(),local[k].copy(),new_local[k].copy())
 if best is None:break
 t,source,dest,indices,delta,add,local,new_local=best;table=tables[t];n=len(indices)
 fresh=(table['next_local'][dest]+np.arange(n,dtype=np.uint64))*G+dest
 if fresh[-1]>np.iinfo(np.uint32).max:raise OverflowError('New physical IDs exceed uint32')
 table['newids'][indices]=fresh.astype(np.uint32);table['next_local'][dest]+=n;table['owners'][indices]=dest
 send[:,source]-=delta;send[:,dest]+=add;send_detail[:,t,source]-=delta;send_detail[:,t,dest]+=add
 recv[:,source]+=local;recv[:,dest]-=new_local;recv_detail[:,t,source]+=local;recv_detail[:,t,dest]-=new_local
 send_target=send.mean(axis=1)*a.target
 storage[source]-=n;storage[dest]+=n;moves+=n
 assert np.all(send.sum(axis=1)<=before_send.sum(axis=1)) and np.all(recv.sum(axis=1)<=before_recv.sum(axis=1))
 assert np.array_equal(send.sum(axis=1),recv.sum(axis=1))
 assert np.all(recv.max(axis=1)<=receive_limit) and storage.max()<=storage_limit
 ratios=send.max(axis=1)/send.mean(axis=1)
 history.append(dict(iteration=iteration,table=t,source=source,destination=dest,moved=n,send_ratios=ratios.tolist()))
 print('migrate',iteration,'moves',moves,'send ratios',ratios.tolist(),flush=True)
metadata=dict(per_owner_candidates=a.per_owner,target=a.target,max_rounds=a.rounds,migration_events=moves,history=history,
 before_storage=before_storage.tolist(),after_storage=storage.tolist(),predicted_send=send.tolist(),predicted_receive=recv.tolist(),
 guarantees='Batch/table total traffic never increases; unique counts preserved; receiving peak and observed-ID storage peak do not increase; no replication',
 limitation='Physical ID space can acquire unused holes; rank storage here counts observed IDs, not padded allocated memory. Offline fit on this same trace.')
(out/'refinement.json').write_text(json.dumps(metadata,indent=2));destdir=out/'scheduled';destdir.mkdir()
for t,table in enumerate(tables):
 assert len(np.unique(table['newids']))==len(table['newids'])
 np.savez_compressed(out/f'table_{t}_remap.npz',old=table['ids'],new=table['newids'])
for r in range(G):
 with (destdir/f'dataset_{r}.bin').open('wb') as f:
  for start in range(0,NB*B,65536):
   records=arrays[r][start:min(start+65536,NB*B)].copy()
   for t,table in enumerate(tables):
    moved=table['ids']!=table['newids'];old=table['ids'][moved];new=table['newids'][moved]
    if not len(old):continue
    cols=records[:,1+dense+offsets[t]:1+dense+offsets[t+1]];pos=np.searchsorted(old,cols);valid=pos<len(old);safe=np.minimum(pos,len(old)-1);valid&=old[safe]==cols;cols[valid]=new[safe[valid]]
   records.tofile(f)
metrics,actual=measure([destdir/f'dataset_{r}.bin' for r in range(G)],G,B,NB,lengths,dense,True)
assert np.array_equal(actual['unique'],base['unique'])
assert np.array_equal(actual['receive'],recv_detail)
assert np.array_equal(actual['send'],send_detail)
assert np.all(actual['receive'].sum(axis=2)<=base['receive'].sum(axis=2))
(out/'cps_metrics.json').write_text(json.dumps(metrics,indent=2));np.savez_compressed(out/'cps_counts.npz',**actual)
shutil.copyfile(case/'hash_metrics.json',out/'hash_metrics.json');cfg.update(repo=str(Path(__file__).resolve().parents[1]/'working'),base_case=str(case),sender_migration=True)
(out/'config.json').write_text(json.dumps(cfg,indent=2));(out/'code_sha256.json').write_text(json.dumps({str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}));(out/'complete.json').write_text(json.dumps(dict(completed=time.time())))
print('migration independently verified',out,flush=True)
