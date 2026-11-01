"""Run upstream LAIA kernel and its cache state on RAPID multi-hot accesses."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'build'))
from herald_native import Scheduler

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=5)
    ap.add_argument('--cache-ratio',type=float,default=.1)
    ap.add_argument('--max-batches',type=int)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((a.data/'input.json').read_text())
    sparse=np.load(a.data/'sparse.npy',mmap_mode='r')
    p=cfg['ranks'];b=cfg['local_batch']; n=cfg['global_batches']
    if a.max_batches:n=min(n,a.max_batches)
    cap=max(1,int(cfg['active_embeddings']*a.cache_ratio))
    assert sparse.shape==(cfg['rows'],sum(cfg['lengths']))
    scheduler=Scheduler(p,b,sparse.shape[1],cap,cfg['active_embeddings'])
    history=[]; digest=hashlib.sha256();start=time.perf_counter()
    with (a.out/'steps.csv').open('w') as f:
        f.write('epoch,batch,rank,miss_pull,update_pull,eviction_push,dependency_push,unique_accesses\n')
        for epoch in range(a.epochs):
            for batch in range(n):
                x=np.asarray(sparse[batch*p*b:(batch+1)*p*b],dtype=np.uint64)
                v=scheduler.step(x,batch,batch==0 or (epoch==0 and batch<3))
                counts=np.asarray(v['counts'],dtype=np.int64);history.append(counts)
                assignment=np.asarray(v['assignment'],dtype=np.uint8)
                assert np.array_equal(np.bincount(assignment,minlength=p),np.full(p,b))
                digest.update(assignment.tobytes())
                for r,c in enumerate(counts):f.write(','.join(map(str,[epoch+1,batch,r,*c]))+'\n')
                f.flush()
                print(f'epoch={epoch+1} batch={batch+1}/{n} pulls={counts[:,:2].sum()} pushes={counts[:,2:4].sum()} elapsed={time.perf_counter()-start:.1f}',flush=True)
    pending=scheduler.pending();arr=np.array(history)
    np.savez_compressed(a.out/'counts.npz',counts=arr,pending=pending)
    pulls=int(arr[:,:,:2].sum());pushes=int(arr[:,:,2:4].sum());flush=sum(pending)
    result=dict(method='Herald default static LAIA native kernel + upstream LRU cache trace',
      measurement='CPU access trace, not GPU training or network measurement',
      data=str(a.data.resolve()),input_sha256=cfg['input_sha256'],rows=p*b*n,epochs=a.epochs,
      processed_rows=p*b*n*a.epochs,ranks=p,local_batch=b,steps_per_epoch=n,
      cache_ratio=a.cache_ratio,cache_entries_per_worker=cap,cache_basis='active distinct table-ID pairs in this input subset',
      slots=sparse.shape[1],tables=len(cfg['lengths']),forward_vectors=pulls,backward_vectors=pushes,
      terminal_flush_vectors=flush,total_vectors_without_flush=pulls+pushes,total_vectors_with_flush=pulls+pushes+flush,
      vector_bytes_128=(pulls+pushes+flush)*512,vector_bytes_32=(pulls+pushes+flush)*128,
      per_epoch=[{'epoch':e+1,'pulls':int(arr[e*n:(e+1)*n,:,:2].sum()),'pushes':int(arr[e*n:(e+1)*n,:,2:4].sum())} for e in range(a.epochs)],
      assignment_sha256=digest.hexdigest(),seconds=time.perf_counter()-start,
      conventions=['Counts each PS-worker vector transfer once, both directions; no dense-gradient or metadata traffic.',
        'Cache starts empty and persists between epochs; original sorted-unique LRU access order. Upstream bitmap cache lookup enabled (same semantics).',
        'Every original multi-hot slot is retained; shared table-ID keys remain shared across slots.',
        'Full flush of remaining valid cache contributions is an explicitly modeled terminal cost, not an upstream API.',
        'No extra future training batch is counted. The last gradient state is covered by terminal flush.',
        'No ID roundtrip through float32 (upstream Python loader does that); exact int32 keys retained.'])
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
