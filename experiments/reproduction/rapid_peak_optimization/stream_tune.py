import ctypes,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;ptr=lambda x:x.ctypes.data_as(ctypes.c_void_p)
lib=ctypes.CDLL(str(ROOT/'profile_stats.so'));stats=lib.stats;stats.argtypes=[ctypes.c_long,ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*5;stats.restype=None
sw=ctypes.CDLL(str(ROOT/'stream_swaps.so'));swap=sw.owner_swaps;swap.restype=ctypes.c_long;swap.argtypes=[ctypes.c_long]*2+[ctypes.c_int]*3+[ctypes.c_void_p]*11+[ctypes.c_int,ctypes.c_long]
def tune(state,case,fraction,out):
 case=Path(case);out=Path(out);out.mkdir(exist_ok=False);cfg=json.load(open(case/'config.json'));g=state.ranks;nb=cfg['batches'];nt=len(cfg['table_lengths']);max_moved=int(state.next_gid*fraction);start=time.perf_counter();pr=[];ws=[];owners=[];original=[];rd=np.zeros((nb,nt,g),np.int64);sd=rd.copy();offsets=[0];cold=[];hot=[];heat=[]
 for t in range(nt):
  ids=np.load(case/f'{t}.original_ids.npy');o=state.owner[t][ids].copy();p=np.asarray(np.load(case/f'profile_cache/{t}.profiles.npy',mmap_mode='r'),dtype=np.uint64);w=np.zeros((len(ids),nb),np.uint16);r=np.zeros((nb,g),np.int64);s=r.copy();stats(len(ids),g,nb,*map(ptr,[p,o,w,r,s]));rd[:,t]=r;sd[:,t]=s;mass=w.sum(1);cold.append(np.flatnonzero(mass==1)+offsets[-1]);ix=np.flatnonzero(mass>=16);hot.append(ix+offsets[-1]);heat.append(mass[ix]);offsets.append(offsets[-1]+len(ids));original.append(ids);owners.append(o);ws.append(w);pr.append(p)
 order=np.concatenate(cold).astype(np.int64);hotorder=np.concatenate(hot).astype(np.int64);heat=np.concatenate(heat);hotorder=hotorder[np.argsort(-heat.astype(np.int64),kind='stable')];offsets=np.array(offsets,np.int64);recv=rd.sum(1);send=sd.sum(1);norm=recv.mean(1);before_peak=float((recv+send).max(1).mean());before_total=int(recv.sum()*2);pp=np.array([x.ctypes.data for x in pr],np.uintp);pw=np.array([x.ctypes.data for x in ws],np.uintp);po=np.array([x.ctypes.data for x in owners],np.uintp)
 moved=swap(len(order),len(hotorder),nt,g,nb,*map(ptr,[offsets,order,hotorder,pp,pw,po,rd,sd,recv,send,norm]),1,max_moved)
 tables=[];ids=[];new=[]
 for t,o in enumerate(owners):
  ix=np.flatnonzero(o!=state.owner[t][original[t]]);tables.extend([t]*len(ix));ids.extend(original[t][ix].tolist());new.extend(o[ix].tolist())
 assert len(ids)==2*moved and len(ids)<=max_moved
 # Recount independently after swaps, then execute sparse metadata migration.
 for t in range(nt):
  w=np.zeros_like(ws[t]);r=np.zeros((nb,g),np.int64);s=r.copy();stats(len(owners[t]),g,nb,*map(ptr,[pr[t],owners[t],w,r,s]));assert np.array_equal(r,rd[:,t]) and np.array_equal(s,sd[:,t])
 ledger=state.apply_changes(tables,ids,new,allowed=True,max_moved=max_moved,state_tensors=1);ledger.update(fraction=fraction,max_moved=max_moved,information_source=str(case),future_window_inspected=False,historical_uncached_total_before=before_total,historical_uncached_total_after=int(recv.sum()*2),historical_uncached_peak_before=before_peak,historical_uncached_peak_after=float((recv+send).max(1).mean()),weight_bytes=len(ids)*512,adam_bytes=len(ids)*1536,routing_record_bytes_assumption=16,fully_replicated_routing_bytes=len(ids)*16*(g-1),adam_plus_replicated_routes_bytes=len(ids)*(1536+16*(g-1)),routing_cost_is_scenario_not_measured=True,seconds=time.perf_counter()-start)
 (out/'migration.json').write_text(json.dumps(ledger,indent=2));print('TUNING',len(ids),'past peak',before_peak,'->',ledger['historical_uncached_peak_after'],flush=True);return ledger
