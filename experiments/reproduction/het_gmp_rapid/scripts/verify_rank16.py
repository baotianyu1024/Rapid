"""Independent real-output and bound-zero traffic checks, outside preprocessing timing."""
import json,time
from pathlib import Path
import numpy as np
from common import ROOT,WIDTH,save
CASE=ROOT/'results/rank16_fast';P=16;B=8192

def main():
 start=time.perf_counter();data=CASE/'data';part=CASE/'gmp';final=CASE/'rapid_guard_p5'
 done=json.loads((final/'complete.json').read_text());assert done['independent_counts_verified'] and done['all_records_inverse_verified']
 cfg=json.loads((data/'input.json').read_text());assert cfg['rows']==4194304 and cfg['ranks']==P
 raw=np.memmap(data/'input.bin',dtype='<u4',mode='r').reshape(-1,WIDTH);sparse=np.load(data/'sparse.npy',mmap_mode='r')
 assignment=np.load(part/'assignment.npy');owner=np.load(part/'owner.npy');rows=[np.flatnonzero(assignment==r) for r in range(P)]
 assert sum(map(len,rows))==len(raw) and np.all((owner>=0)&(owner<P))
 steps=max((len(v)+B-1)//B for v in rows)
 expected=np.zeros((steps,P,P),dtype=np.int64)
 for r in range(P):
  written=np.memmap(part/f'scheduled/dataset_{r}.bin',dtype='<u4',mode='r').reshape(-1,WIDTH);ix=rows[r];assert len(written)==len(ix)
  for off in range(0,len(ix),32768):assert np.array_equal(written[off:off+32768],raw[ix[off:off+32768]])
  for b in range(steps):
   ids=np.unique(sparse[ix[b*B:(b+1)*B]].ravel());expected[b,r]=np.bincount(owner[ids],minlength=P);expected[b,r,r]=0
  mirrors=np.load(part/f'mirrors_{r}.npy');assert np.all(owner[mirrors]!=r) and np.all(mirrors[1:]>mirrors[:-1])
  print('GMP records and independent remote-ID counts verified rank',r,flush=True)
 with np.load(CASE/'traffic/het_gmp_s0.npz') as z:
  f,g=z['forward'],z['backward'];assert f.shape==g.shape==(steps*5,P,P)
  assert np.array_equal(g,np.tile(expected,(5,1,1)))
  assert np.array_equal(f[0],g[0]) and np.all(f<=g)
  assert np.all(np.diagonal(f,axis1=1,axis2=2)==0) and not np.any(z['pending_flush'])
 meta=json.loads((CASE/'traffic/het_gmp_s0.json').read_text());assert meta['pending_updates']==0 and meta['processed_rows']==len(raw)*5
 # Independently undo the initial owner remap; the final remap has already
 # checked every record against this base inside materialize().
 base=CASE/'rapid_owner_base';inverses=[]
 for t in range(26):
  with np.load(base/f'table_{t}_remap.npz') as m:
   inv=np.zeros(int(m['new'].max())+1,dtype=np.uint32);inv[m['new']]=m['old'];inverses.append(inv)
 offsets=np.cumsum([14]+cfg['lengths'])
 for r in range(P):
  before=np.memmap(CASE/f'rapid_schedule/scheduled/dataset_{r}.bin',dtype='<u4',mode='r').reshape(-1,WIDTH)
  after=np.memmap(base/f'scheduled/dataset_{r}.bin',dtype='<u4',mode='r').reshape(-1,WIDTH);assert len(before)==len(after)==32*B
  for off in range(0,len(before),32768):
   restored=after[off:off+32768].copy()
   for t,inv in enumerate(inverses):restored[:,offsets[t]:offsets[t+1]]=inv[restored[:,offsets[t]:offsets[t+1]]]
   assert np.array_equal(restored,before[off:off+32768])
 save(CASE/'verification.json',dict(success=True,gmp_all_records_match_source=True,gmp_backward_s0_independently_recounted=True,gmp_cold_first_step_and_forward_subset_verified=True,gmp_no_pending_gradients=True,rapid_initial_and_final_owner_maps_inverse_verified=True,rapid_final_traffic_independently_verified=True,seconds=time.perf_counter()-start))
 print('ALL OUTPUT AND TRAFFIC CHECKS PASS',flush=True)
if __name__=='__main__':main()
