"""Replay successive windows: add new IDs, schedule samples, never migrate existing IDs."""
import argparse,ctypes,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];X=ROOT/'experiments/reproduction/rapid_peak_optimization'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--ranks',type=int,choices=[8,64],default=8);p.add_argument('--batch-size',type=int,default=8192);p.add_argument('--initial-batches',type=int,default=32);p.add_argument('--window-batches',type=int,default=1);p.add_argument('--windows',type=int,default=32);p.add_argument('--dense',type=int,default=13);p.add_argument('--table-lengths',type=int,nargs='+',default=[3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1]);p.add_argument('--dim',type=int,default=128);a=p.parse_args()
 if not 1<=a.initial_batches<=4096 or not 1<=a.window_batches<=4096:p.error('Each window must contain 1..4096 complete global batches')
 if a.windows<1 or a.batch_size<1 or a.dense<0 or min(a.table_lengths)<1:p.error('Invalid layout or window count')
 if max(a.table_lengths)<2:p.error('This historical grouped incremental recipe requires at least one multi-hot table')
 sys.path[:0]=[str(ROOT/'experiments/incremental'),str(X)]
 from stream_owner_state import OwnerState
 from wide_stream_window import window
 from wide_stream_alternate import alternate
 import group_scheduler as group
 lib=ctypes.CDLL(str(X/'sample_group_peak_phase.so'));fn=lib.sample_group_peak;fn.argtypes=group.fn.argtypes;fn.restype=group.fn.restype;group.fn=fn
 width=1+a.dense+sum(a.table_lengths);n0=a.initial_batches*a.ranks*a.batch_size;nw=a.window_batches*a.ranks*a.batch_size
 if a.input.stat().st_size%(4*width):p.error('Input contains a partial record')
 raw=np.memmap(a.input,dtype='<u4',mode='r').reshape(-1,width)
 if len(raw)<n0+a.windows*nw:p.error('Not enough records for the requested initial profile and disjoint windows')
 a.out.mkdir(parents=True,exist_ok=False);state=OwnerState(a.ranks,len(a.table_lengths));cfg=dict(batch_size=a.batch_size,dense=a.dense,table_lengths=a.table_lengths);rows=[]
 for k in range(a.windows+1):
  data=np.ascontiguousarray(raw[:n0] if k==0 else raw[n0+(k-1)*nw:n0+k*nw]);out=a.out/f'window_{k:03d}';out.mkdir()
  before=[(np.flatnonzero(o>=0),o[o>=0].copy(),q[o>=0].copy()) for o,q in zip(state.owner,state.physical)];old=state.next_gid
  window(state,data,cfg,out/'initial',rounds=4,balance=4,schedule='grouped')
  alternate(state,out/'initial',data,out/'final',rounds=4,balance=4,reallocate=True,scheduler='group')
  for t,(ids,owners,physical) in enumerate(before):
   assert np.array_equal(state.owner[t][ids],owners);assert np.array_equal(state.physical[t][ids],physical)
  assignment=np.load(out/'final/assignment.npy');nb=len(data)//(a.ranks*a.batch_size);recv=np.zeros((nb,len(a.table_lengths),a.ranks),np.int64);send=recv.copy()
  for batch in range(nb):
   part=assignment[batch*a.ranks*a.batch_size:(batch+1)*a.ranks*a.batch_size];records=data[batch*a.ranks*a.batch_size:(batch+1)*a.ranks*a.batch_size]
   assert np.array_equal(np.bincount(part,minlength=a.ranks),np.full(a.ranks,a.batch_size))
   for r in range(a.ranks):
    col=1+a.dense
    for t,l in enumerate(a.table_lengths):
     ids=np.unique(records[part==r,col:col+l]);col+=l;o=state.owner[t][ids];assert np.all(o>=0);remote=o[o!=r];recv[batch,t,r]=len(remote);send[batch,t]+=np.bincount(remote,minlength=a.ranks)
  pred=np.load(out/'final/predicted_group_counts.npz');assert np.array_equal(recv.sum(1),pred['receive']) and np.array_equal(send.sum(1),pred['send'])
  cap=a.batch_size*(a.ranks-1)
  if k==0:choice=recv.mean((0,2))<cap
  f=np.where(choice[None,:,None],recv,cap).sum(1);s=np.where(choice[None,:,None],send,cap).sum(1)
  result=dict(window=k,rows=len(data),new_embeddings=state.next_gid-old,known_embeddings=state.next_gid,old_owner_changes=0,old_physical_id_changes=0,migration_bytes=0,storage=state.storage.tolist(),forward_backward_GiB_one_epoch=float(2*f.sum()*a.dim*4/2**30),mean_rank_vectors_per_step=float((f+s).mean()),mean_step_peak_vectors=float((f+s).max(1).mean()))
  np.savez_compressed(out/'independent_counts.npz',receive=recv,send=send);rows.append(result)
  (a.out/'summary.json').write_text(json.dumps(dict(dim=a.dim,epochs=1,gpu_execution=False,periodic_tuning=False,fixed_acp=choice.tolist(),initial_algorithm='phase-guarded wide-window incremental planner from empty state; distinct from offline recipe',results=rows),indent=2));print(json.dumps(result),flush=True)
 state.save(a.out/'final_state');state.audit()
if __name__=='__main__':main()
