"""Offline RAPID strict-s0 read-cache extension; equal or smaller per-rank GMP mirror budget.
Counts exact cache hits from presence bitmaps, independently checked against State.
No delayed gradients; primary placement and sample schedule stay unchanged.
"""
import argparse,json,sys,time,ctypes
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def masks_python(profile,weights):
 n,g=profile.shape;nb=weights.shape[1];last=np.full(n,-1,dtype=np.int16);result=[]
 for epoch in range(2):
  hit=np.zeros((n,g),dtype=np.uint64)
  for b in range(nb):
   ix=np.flatnonzero((last>=0)&(weights[:,b]>0));r=last[ix]
   ok=((profile[ix,r]>>np.uint64(b))&np.uint64(1))!=0
   hit[ix[ok],r[ok]]|=np.uint64(1)<<np.uint64(b)
   last[weights[:,b]>1]=-1
   ix=np.flatnonzero(weights[:,b]==1)
   if len(ix):last[ix]=np.argmax(((profile[ix]>>np.uint64(b))&np.uint64(1)),axis=1)
  result.append(hit)
 return result
def masks(profile,weights):
 library=ROOT/'cache_masks.so'
 if not library.exists():return masks_python(profile,weights)
 profile=np.ascontiguousarray(profile,dtype=np.uint64);n,g=profile.shape;nb=weights.shape[1]
 cold=np.zeros_like(profile);warm=np.zeros_like(profile)
 lib=ctypes.CDLL(str(library));fn=lib.cache_masks;fn.argtypes=[ctypes.c_long,ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*3;fn.restype=None
 fn(n,g,nb,*[x.ctypes.data_as(ctypes.c_void_p) for x in [profile,cold,warm]])
 return cold,warm
LUT=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
def pop(x):return LUT[np.ascontiguousarray(x).view(np.uint8).reshape(x.shape+(8,))].sum(-1,dtype=np.uint16)

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--gmp-meta',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
 cfg=json.load(open(a.case/'config.json'));g=cfg['ranks'];nb=cfg['batches'];epochs=5
 meta=json.load(open(a.gmp_meta));budgets=np.array(meta['mirror_counts'],dtype=int);assert len(budgets)==g
 selected=[(np.empty(0,np.uint64),np.empty(0,np.uint16)) for _ in range(g)]
 offsets=[0];start=time.perf_counter()
 for t in range(len(cfg['table_lengths'])):
  prof=np.asarray(np.load(a.cache/f'{t}.profiles.npy',mmap_mode='r'),dtype=np.uint64);w=np.load(a.cache/f'{t}.weights.npy',mmap_mode='r');own=np.load(a.case/f'{t}.owners.npy');n=len(own)
  h0,h1=masks(prof,w);scores=pop(h0)+4*pop(h1);scores[np.arange(n),own]=0
  np.save(a.out/f'{t}.cold_hits.npy',h0);np.save(a.out/f'{t}.warm_hits.npy',h1)
  for r in range(g):
   ix=np.flatnonzero(scores[:,r]);ids=np.concatenate([selected[r][0],ix.astype(np.uint64)+offsets[-1]]);sc=np.concatenate([selected[r][1],scores[ix,r]])
   if len(ids)>budgets[r]:
    # Deterministic highest benefit, then lowest global ID. Integer score range <=320.
    cutoff=np.partition(sc,len(sc)-budgets[r])[len(sc)-budgets[r]]
    hi=np.flatnonzero(sc>cutoff);eq=np.flatnonzero(sc==cutoff)[:budgets[r]-len(hi)];keep=np.concatenate([hi,eq]);ids,sc=ids[keep],sc[keep]
   order=np.argsort(ids);selected[r]=(ids[order],sc[order])
  offsets.append(offsets[-1]+n);print('cache selection table',t,'seconds',round(time.perf_counter()-start,1),flush=True)
 z=np.load(a.case/'predicted_counts.npz');rd=z['receive'].sum(1);sd=z['send'].sum(1)
 fr=np.tile(rd,(epochs,1));fs=np.tile(sd,(epochs,1));br=fs.copy();bs=fr.copy();saved_total=0
 for r,(ids,sc) in enumerate(selected):np.save(a.out/f'mirrors_{r}.npy',ids);saved_total+=int(sc.sum())
 for t in range(len(cfg['table_lengths'])):
  own=np.load(a.case/f'{t}.owners.npy');h0=np.load(a.out/f'{t}.cold_hits.npy',mmap_mode='r');h1=np.load(a.out/f'{t}.warm_hits.npy',mmap_mode='r')
  for r,(ids,sc) in enumerate(selected):
   ix=(ids[(ids>=offsets[t])&(ids<offsets[t+1])]-offsets[t]).astype(np.int64)
   for e in range(epochs):
    h=h0 if e==0 else h1;hm=h[ix,r]
    for b in range(nb):
     kept=ix[((hm>>np.uint64(b))&np.uint64(1))!=0];fr[e*nb+b,r]-=len(kept);fs[e*nb+b]-=np.bincount(own[kept],minlength=g)
  # Regenerable large hit matrices are removed after exact aggregate reconstruction.
  del h0,h1
  (a.out/f'{t}.cold_hits.npy').unlink();(a.out/f'{t}.warm_hits.npy').unlink()
 assert np.all(fr>=0) and np.all(fs>=0) and np.array_equal(fr.sum(1),fs.sum(1))
 assert int(br.sum()-fr.sum())==saved_total
 np.savez_compressed(a.out/'counts.npz',fr=fr,fs=fs,br=br,bs=bs)
 def summary(x):return dict(mean=float(x.mean()),total=int(x.sum()),mean_step_peak=float(x.max(1).mean()),absolute_peak=int(x.max()),rank_totals=x.sum(0).tolist())
 result=dict(case=str(a.case.resolve()),cache=str(a.cache.resolve()),gmp_meta=str(a.gmp_meta.resolve()),ranks=g,epochs=epochs,batches=nb,rows=cfg['rows'],mirror_budget=budgets.tolist(),mirror_counts=[len(x[0]) for x in selected],saved_vectors=saved_total,strict_s=0,gradient_delaying=False,primary_changed=False,samples_changed=False,selection='offline exact avoided-fetch benefit on same trace; top beneficial remote IDs within equal GMP per-rank budgets',seconds=time.perf_counter()-start,metrics={k:summary(v) for k,v in dict(forward_receive=fr,forward_send=fs,backward_receive=br,backward_send=bs,combined_receive=fr+br,combined_send=fs+bs).items()},phase_bottleneck_mean=float((np.maximum(fr.max(1),fs.max(1))+np.maximum(br.max(1),bs.max(1))).mean()))
 (a.out/'result.json').write_text(json.dumps(result,indent=2));print('DONE',json.dumps({k:v['mean_step_peak'] for k,v in result['metrics'].items()}),flush=True)
if __name__=='__main__':main()
