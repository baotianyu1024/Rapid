"""Alternate new-ID-only placement with fixed-global-batch sample swaps, before training."""
import json,time,gc,ctypes,os
from pathlib import Path
import numpy as np
from wide_stream_window import place,profiles,peak,ptr
from stream_tune import stats
from compact_new_slots import compact_new_slots,storage_accounting
from targeted_scheduler import targeted
from zero_scheduler import targeted_zero
from loadaware_scheduler import loadaware
from group_scheduler import grouped
_balanced=ctypes.CDLL(str(Path('/mnt/hugui/longjiu.bty/reproduction/revision_response_20261026/code/reproduction/rapid_peak_optimization')/'batch_balanced_samples.so')).batch_balanced_samples
_balanced.argtypes=peak.argtypes;_balanced.restype=peak.restype
_auction=ctypes.CDLL(str(Path('/mnt/hugui/longjiu.bty/reproduction/revision_response_20261026/code/reproduction/rapid_peak_optimization')/'batch_auction_samples.so')).batch_auction_samples
_auction.argtypes=peak.argtypes;_auction.restype=peak.restype

wide=ctypes.CDLL('/mnt/hugui/longjiu.bty/reproduction/revision_response_20261026/zh_revision_v2/code/wide_profiles.so');old_stats=stats;stats=wide.stats;stats.argtypes=old_stats.argtypes;stats.restype=old_stats.restype

def relocate_new(state,original,planned,old_n):
 before=np.zeros(state.ranks,np.int64);after=before.copy();moved=0
 for t,(ids,own) in enumerate(zip(original,planned)):
  changed=state.owner[t][ids]!=own;ix=ids[changed];dest=own[changed];assert np.all(state.gid[t][ix]>=old_n);old=state.owner[t][ix].copy();physical=state.physical[t][ix].copy();before+=np.bincount(old,minlength=state.ranks);after+=np.bincount(dest,minlength=state.ranks);moved+=len(ix)
  for r in range(state.ranks):state.free.setdefault((t,r),[]).extend(physical[old==r].tolist())
  for r in range(state.ranks):
   selected=ix[dest==r];count=len(selected)
   if not count:continue
   slots=state.free[(t,r)];take=min(count,len(slots))
   if take:state.physical[t][selected[:take]]=np.array(slots[-take:][::-1],np.uint32);del slots[-take:]
   first=int(state.next[t,r]);state.physical[t][selected[take:]]=(np.arange(first,first+count-take,dtype=np.uint64)*state.ranks+r).astype(np.uint32);state.next[t,r]+=count-take
  state.owner[t][ix]=dest
 assert np.array_equal(before,after);compact_new_slots(state,old_n);return moved

def alternate(state,case,data,out,rounds=4,balance=4,reallocate=True,scheduler="peak"):
 case=Path(case);out=Path(out);out.mkdir(exist_ok=False);(out/'profile_cache').mkdir();cfg=json.load(open(case/'config.json'));cfg['base_case']=str(case);(out/'config.json').write_text(json.dumps(cfg,indent=2));placement=json.load(open(case/'placement.json'));old_n=placement['old_embeddings'];g=state.ranks;nb=cfg['batches'];n=cfg['rows'];b=cfg['batch_size'];lengths=np.array(cfg['table_lengths'],np.int32);nt=len(lengths);dense=cfg['dense'];start=time.perf_counter();original=[];planned=[];newpr=[];tables=[];masks=[];rd=np.zeros((nb,nt,g),np.int64);sd=rd.copy();cap=np.zeros(g,np.int64)
 if reallocate:
  for t in range(nt):
   ids=np.load(case/f'{t}.original_ids.npy');p=np.load(case/f'profile_cache/{t}.profiles.npy',mmap_mode='r');o=state.owner[t][ids].copy();mask=state.gid[t][ids]>=old_n
   if (case/f'{t}.new_hot_ids.npy').exists():mask&=~np.isin(ids,np.load(case/f'{t}.new_hot_ids.npy'))
   elif placement['new_hot_embeddings']:raise ValueError('Need saved new-hot identity set')
   fixed=np.asarray(p[~mask],np.uint64);fo=o[~mask];w=np.zeros((len(fo),nb),np.uint16);r=np.zeros((nb,g),np.int64);s=r.copy();stats(len(fo),g,nb,*map(ptr,[fixed,fo,w,r,s]));rd[:,t]=r;sd[:,t]=s;newpr.append(np.asarray(p[mask],np.uint64));tables.append(np.full(int(mask.sum()),t,np.uint8));cap+=np.bincount(o[mask],minlength=g);original.append(ids);planned.append(o);masks.append(mask)
   print('new-only prep',t,flush=True)
  pr=np.concatenate(newpr);del newpr;tabs=np.concatenate(tables);newown=np.full(len(tabs),-1,np.int16);assert place(len(tabs),g,nb,nt,*map(ptr,[tabs,pr,cap,newown,rd,sd]))==0 and not cap.any();off=0
  for t,mask in enumerate(masks):planned[t][mask]=newown[off:off+int(mask.sum())];off+=int(mask.sum())
  before=state.global_owners()[:old_n].copy();moved=relocate_new(state,original,planned,old_n);assert np.array_equal(state.global_owners()[:old_n],before);del pr,tabs,newown,masks,before;gc.collect();np.savez_compressed(out/'post_placement_counts.npz',receive=rd,send=sd)
 else:
  original=[np.load(case/f"{t}.original_ids.npy") for t in range(nt)];moved=0
 if rounds==0:
  for t,ids in enumerate(original):
   os.link(case/f'profile_cache/{t}.profiles.npy',out/f'profile_cache/{t}.profiles.npy');np.save(out/f'profile_cache/{t}.ids.npy',state.physical[t][ids]);np.save(out/f'{t}.original_ids.npy',ids);np.save(out/f'{t}.owners.npy',state.owner[t][ids]);np.save(out/f'{t}.logical_gids.npy',state.gid[t][ids]);np.save(out/f'{t}.new_hot_ids.npy',np.load(case/f'{t}.new_hot_ids.npy') if (case/f'{t}.new_hot_ids.npy').exists() else np.empty(0,np.uint32))
  os.link(case/'assignment.npy',out/'assignment.npy');placement.update(storage_accounting=storage_accounting(state),new_ids_relocated_before_training=moved,additional_sample_moves=0,alternating_pass=placement.get('alternating_pass',0)+1,alternating_seconds=time.perf_counter()-start,scheduler='unchanged',old_owners_unchanged_after_alternation=True);(out/'placement.json').write_text(json.dumps(placement,indent=2));return placement
 assignment=np.load(case/'assignment.npy');order=np.argsort(assignment,kind='stable');nr=n//g;assert np.array_equal((order//(g*b)),np.tile(np.repeat(np.arange(nb),b),g));compact=np.empty((n,int(lengths.sum())),np.uint32);owners=[];off=0;col=dense+1;dst=0
 for t,(ids,l) in enumerate(zip(original,lengths)):
  lut=np.full(len(state.owner[t]),np.iinfo(np.uint32).max,np.uint32);lut[ids]=np.arange(off,off+len(ids),dtype=np.uint32);compact[:,dst:dst+l]=lut[data[order,col:col+l]];owners.append(state.owner[t][ids]);off+=len(ids);col+=l;dst+=l
 compact.sort(axis=1);own=np.concatenate(owners);assign=np.repeat(np.arange(g,dtype=np.int32),nr)
 if scheduler=="group":
  main=int(np.argmax(lengths));col=dense+1+int(lengths[:main].sum());keys=(data[order,col].astype(np.uint64)<<np.uint64(32))|data[order,col+1].astype(np.uint64);sample_moves=grouped(ptr(compact),n,int(lengths.sum()),off,g,b,ptr(own),ptr(assign),rounds,balance,keys)
 else:sample_moves=(loadaware if scheduler=="loadaware" else targeted_zero if scheduler=="zero" else targeted if scheduler=="targeted" else _auction if scheduler=="auction" else _balanced if scheduler=="balanced" else peak)(ptr(compact),n,int(lengths.sum()),off,g,b,ptr(own),ptr(assign),rounds,balance)
 assignment[order]=assign;del compact,own,assign,order
 if scheduler=='group':
  from group_scheduler import LAST_COUNTS
  np.savez_compressed(out/'predicted_group_counts.npz',**LAST_COUNTS)
 for k in range(nb):assert np.array_equal(np.bincount(assignment[k*g*b:(k+1)*g*b],minlength=g),np.full(g,b))
 np.save(out/'assignment.npy',assignment)
 for t,ids in enumerate(original):
  lut=np.full(len(state.owner[t]),-1,np.int32);lut[ids]=np.arange(len(ids));lptr=np.array([lut.ctypes.data],np.uintp);sz=np.array([len(lut)],np.int64);ls=np.array([lengths[t]],np.int32);col=dense+1+int(lengths[:t].sum());rec=np.empty((n,1+int(lengths[t])),np.uint32);rec[:,0]=0;rec[:,1:]=data[:,col:col+lengths[t]];pr=np.zeros((len(ids),g,(nb+63)//64),np.uint64);profiles(ptr(rec),n,g,b,rec.shape[1],0,1,*map(ptr,[ls,lptr,sz,assignment,pr]));np.save(out/f'profile_cache/{t}.profiles.npy',pr.astype(np.uint8 if nb<=8 else np.uint16 if nb<=16 else np.uint32 if nb<=32 else np.uint64));np.save(out/f'profile_cache/{t}.ids.npy',state.physical[t][ids]);np.save(out/f'{t}.original_ids.npy',ids);np.save(out/f'{t}.owners.npy',state.owner[t][ids]);np.save(out/f'{t}.logical_gids.npy',state.gid[t][ids]);np.save(out/f'{t}.new_hot_ids.npy',np.load(case/f'{t}.new_hot_ids.npy') if (case/f'{t}.new_hot_ids.npy').exists() else np.empty(0,np.uint32));print('alternate profiles',t,flush=True)
 placement.update(storage_accounting=storage_accounting(state),new_ids_relocated_before_training=moved,additional_sample_moves=sample_moves,alternating_pass=placement.get('alternating_pass',0)+1,alternating_seconds=time.perf_counter()-start,scheduler=scheduler,old_owners_unchanged_after_alternation=True);(out/'placement.json').write_text(json.dumps(placement,indent=2));return placement
