"""Exact strict-s0 clocks, processed embedding-by-embedding to bound replay cost."""
from persistent_replay import *
lib=ctypes.CDLL(str(ROOT/'persistent_oracle_sparse.so'));fn=lib.replay;fn.argtypes=[ctypes.c_long,ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*10+[ctypes.c_long];fn.restype=None

def replay_case(state,case,expected_saved=None,old_n=0):
 case=Path(case);cfg=json.load(open(case/'config.json'));g=state.p;nb=cfg['batches'];n=len(state.owner);assert state.bound==0 and not state.history;start=time.perf_counter();cached=np.zeros(n,np.uint64);valid=np.zeros(n,np.uint64)
 for r,ids in enumerate(state.mirrors):
  bit=np.uint64(1)<<np.uint64(r);cached[ids]|=bit;valid[ids[state.version[r]==state.master[ids]]]|=bit
 finalvalid=valid.copy();f=np.zeros((5*nb,g,g),np.int64);grad=f.copy();oldnew=np.zeros(2,np.int64);before=int(state.master.sum())
 for t in range(len(cfg['table_lengths'])):
  prof=np.asarray(np.load(case/f'profile_cache/{t}.profiles.npy',mmap_mode='r'),dtype=np.uint64);gids=np.load(case/f'{t}.logical_gids.npy');owners=np.asarray(state.owner[gids],np.int16);assert np.array_equal(owners,np.load(case/f'{t}.owners.npy'));fn(len(gids),g,nb,*map(ptr,[prof,owners,gids,cached,valid,state.master,finalvalid,f,grad,oldnew]),old_n);print('persistent oracle table',t,flush=True)
 state.updates+=int(state.master.sum())-before
 for r,ids in enumerate(state.mirrors):state.version[r]=np.where((finalvalid[ids]&(np.uint64(1)<<np.uint64(r)))!=0,state.master[ids],-1)
 state.history=list(zip(f,grad,grad));result,arrays=state.finish();state.history=[];saved=(grad-f).sum((0,2))
 if expected_saved is not None:assert np.array_equal(saved,expected_saved),(saved,expected_saved)
 result.update(persistent_cache=True,exact_s0_version_oracle=True,update_counts_normalized_to_presence=True,epochs=5,steps_per_epoch=nb,saved_forward_vectors_by_rank=saved.tolist(),raw_old_id_remote_queries_per_epoch=int(oldnew[0]),raw_new_id_remote_queries_per_epoch=int(oldnew[1]),replay_seconds=time.perf_counter()-start)
 return result,arrays
