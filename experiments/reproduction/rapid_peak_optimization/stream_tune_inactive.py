import ctypes,json,time
from pathlib import Path
import numpy as np
from stream_tune import stats,ptr
ROOT=Path(__file__).resolve().parent;lib=ctypes.CDLL(str(ROOT/'stream_inactive_swaps.so'));fn=lib.swaps;fn.argtypes=[ctypes.c_long]+[ctypes.c_int]*3+[ctypes.c_void_p]*14+[ctypes.c_long];fn.restype=ctypes.c_long

def tune(state,case,fraction,out,min_mass=2):
 case=Path(case);out=Path(out);out.mkdir(exist_ok=False);cfg=json.load(open(case/'config.json'));g=state.ranks;nb=cfg['batches'];nt=len(cfg['table_lengths']);max_moved=int(state.next_gid*fraction);start=time.perf_counter();original=[];owners=[];pr=[];ws=[];offsets=[0];hot=[];heat=[];rd=np.zeros((nb,nt,g),np.int64);sd=rd.copy();pool_t=[[] for _ in range(g)];pool_id=[[] for _ in range(g)];pool_limit=max(64,int(np.ceil(max_moved/g))*4)
 for t in range(nt):
  ids=np.load(case/f'{t}.original_ids.npy');o=state.owner[t][ids].copy();p=np.asarray(np.load(case/f'profile_cache/{t}.profiles.npy',mmap_mode='r'),dtype=np.uint64);w=np.zeros((len(ids),nb),np.uint16);r=np.zeros((nb,g),np.int64);s=r.copy();stats(len(ids),g,nb,*map(ptr,[p,o,w,r,s]));rd[:,t]=r;sd[:,t]=s;mass=w.sum(1);ix=np.flatnonzero(mass>=min_mass);hot.append(ix+offsets[-1]);heat.append(mass[ix]);offsets.append(offsets[-1]+len(ids));original.append(ids);owners.append(o);pr.append(p);ws.append(w)
  inactive=state.owner[t].copy();inactive[ids]=-1
  for rank in range(g):
   need=pool_limit-sum(map(len,pool_id[rank]))
   if need>0:
    candidates=np.flatnonzero(inactive==rank)[:need];pool_id[rank].append(candidates);pool_t[rank].append(np.full(len(candidates),t,np.uint8))
 pools_id=[np.concatenate(x) if x else np.empty(0,np.int64) for x in pool_id];pools_t=[np.concatenate(x) if x else np.empty(0,np.uint8) for x in pool_t];available=np.array(list(map(len,pools_id)),np.int64);initial_available=available.copy();offsets=np.array(offsets,np.int64);hot=np.concatenate(hot).astype(np.int64);heat=np.concatenate(heat);hot=hot[np.argsort(-heat.astype(np.int64),kind='stable')];recv=rd.sum(1);send=sd.sum(1);norm=recv.mean(1);before_peak=float((recv+send).max(1).mean());before_total=int(recv.sum()*2);pp=np.array([x.ctypes.data for x in pr],np.uintp);pw=np.array([x.ctypes.data for x in ws],np.uintp);po=np.array([x.ctypes.data for x in owners],np.uintp);mh=np.zeros(max_moved//2,np.int64);mf=np.zeros(len(mh),np.int16);mt=mf.copy()
 count=fn(len(hot),nt,g,nb,*map(ptr,[offsets,hot,pp,pw,po,rd,sd,recv,send,norm,available,mh,mf,mt]),max_moved);tables=[];ids=[];dest=[];taken=np.zeros(g,np.int64)
 for h,s,d in zip(mh[:count],mf[:count],mt[:count]):
  t=int(np.searchsorted(offsets,h,side='right')-1);fid=int(original[t][h-offsets[t]]);j=int(taken[d]);taken[d]+=1;ct=int(pools_t[d][j]);ci=int(pools_id[d][j]);tables.extend([t,ct]);ids.extend([fid,ci]);dest.extend([int(d),int(s)])
 assert np.array_equal(initial_available-available,taken)
 for t in range(nt):
  w=np.zeros_like(ws[t]);r=np.zeros((nb,g),np.int64);s=r.copy();stats(len(owners[t]),g,nb,*map(ptr,[pr[t],owners[t],w,r,s]));assert np.array_equal(r,rd[:,t]) and np.array_equal(s,sd[:,t])
 ledger=state.apply_changes(tables,ids,dest,allowed=True,max_moved=max_moved);ledger.update(fraction=fraction,max_moved=max_moved,min_mass=min_mass,inactive_counterparts=True,information_source=str(case),future_window_inspected=False,historical_uncached_total_before=before_total,historical_uncached_total_after=int(recv.sum()*2),historical_uncached_peak_before=before_peak,historical_uncached_peak_after=float((recv+send).max(1).mean()),weight_bytes=len(ids)*512,adam_bytes=len(ids)*1536,routing_record_bytes_assumption=16,fully_replicated_routing_bytes=len(ids)*16*(g-1),adam_plus_replicated_routes_bytes=len(ids)*(1536+16*(g-1)),routing_cost_is_scenario_not_measured=True,seconds=time.perf_counter()-start)
 (out/'migration.json').write_text(json.dumps(ledger,indent=2));print('INACTIVE TUNING',len(ids),'past peak',before_peak,'->',ledger['historical_uncached_peak_after'],flush=True);return ledger
