"""CPU control-flow replay of HET-GMP; no embedding math/GPU timing claimed."""
import argparse, time
import numpy as np
from common import *

def summary(a):
    a=np.asarray(a,dtype=np.float64)
    means=a.mean(axis=1); peaks=a.max(axis=1)
    return dict(total=float(a.sum()),mean_per_rank_step=float(a.mean()),mean_batch_peak=float(peaks.mean()),absolute_peak=float(a.max()),rank_totals=a.sum(axis=0).tolist(),max_peak_mean_ratio=float(np.max(np.divide(peaks,means,out=np.ones_like(peaks),where=means!=0))))

class State:
    def __init__(self,owner,mirrors,bound,p):
        self.owner=owner; self.mirrors=mirrors; self.bound=bound; self.p=p
        self.master=np.ones(len(owner),dtype=np.int64)
        self.version=[np.full(len(m),-1,dtype=np.int64) for m in mirrors]
        self.pending=[np.zeros(len(m),dtype=np.int64) for m in mirrors]
        self.history=[]; self.updates=0

    def prepare(self,batches):
        result=[]
        for r,(ids,counts) in enumerate(batches):
            owners=self.owner[ids]; remote=owners!=r; mirror=self.mirrors[r]
            pos=np.searchsorted(mirror,ids)
            cached=np.zeros(len(ids),dtype=bool)
            if len(mirror):cached=(pos<len(mirror)) & (mirror[np.minimum(pos,len(mirror)-1)]==ids)
            result.append((owners,remote,cached,pos))
        return result

    def step(self,batches,prepared=None):
        # One synchronous protocol round: all reads, then all local and remote
        # updates. This matches pushPull(previous gradients,current queries):
        # mirror versions are captured after local update, and primary versions
        # used for outdated checks include all previous remote gradients.
        p=self.p
        forward=np.zeros((p,p),dtype=np.int64) # [requester, owner]
        gradients=np.zeros((p,p),dtype=np.int64) # [sender, owner]
        queries=np.zeros((p,p),dtype=np.int64)
        cached_batches=[]
        if prepared is None:prepared=self.prepare(batches)
        for r,(ids,counts) in enumerate(batches):
            owners,remote,cached,pos=prepared[r]
            stale=remote & ~cached
            ci=np.flatnonzero(cached); cp=pos[ci]
            if len(ci):
                cv=self.version[r][cp]
                stale[ci]=(cv<0) | (cv+self.bound<self.master[ids[ci]])
            refresh=np.flatnonzero(stale & cached)
            self.version[r][pos[refresh]]=self.master[ids[refresh]]
            queries[r]=np.bincount(owners[remote],minlength=p)
            forward[r]=np.bincount(owners[stale],minlength=p)
            cached_batches.append((ids,counts,owners,remote,cached,pos))
        # Exact gradient-count rules in preprocess.cu / generate_query.cu.
        for r,(ids,counts,owners,remote,cached,pos) in enumerate(cached_batches):
            self.updates+=int(counts.sum())
            self.master[ids[~remote]]+=counts[~remote]
            uncached=remote & ~cached
            self.master[ids[uncached]]+=counts[uncached]
            ci=np.flatnonzero(cached); cp=pos[ci]
            self.pending[r][cp]+=counts[ci]
            flush=self.pending[r][cp]>self.bound
            fi=ci[flush]; fp=cp[flush]
            amounts=self.pending[r][fp].copy()
            self.version[r][fp]+=amounts
            self.master[ids[fi]]+=amounts
            self.pending[r][fp]=0
            gids=np.concatenate((ids[uncached],ids[fi]))
            gradients[r]=np.bincount(self.owner[gids],minlength=p)
        assert np.trace(forward)==np.trace(gradients)==np.trace(queries)==0
        self.history.append((forward,gradients,queries))

    def finish(self):
        f=np.array([x[0] for x in self.history]); g=np.array([x[1] for x in self.history]); q=np.array([x[2] for x in self.history])
        assert f.sum()==f.sum(axis=1).sum()==f.sum(axis=2).sum()
        pending_updates=sum(int(x.sum()) for x in self.pending)
        assert int(self.master.sum())-len(self.master)+pending_updates==self.updates
        flush=np.zeros((self.p,self.p),dtype=np.int64)
        for r,m in enumerate(self.mirrors):
            flush[r]=np.bincount(self.owner[m[self.pending[r]>0]],minlength=self.p)
        # Indices/clocks 16B request, flags 8B reply, refreshed clock 8B,
        # gradient index/update count 16B. Three size exchanges / round.
        metadata_recv=16*q.sum(axis=1)+8*q.sum(axis=2)+8*f.sum(axis=2)+16*g.sum(axis=1)+3*8*(self.p-1)
        metadata_send=16*q.sum(axis=2)+8*q.sum(axis=1)+8*f.sum(axis=1)+16*g.sum(axis=2)+3*8*(self.p-1)
        assert metadata_recv.sum()==metadata_send.sum()
        fr=f.sum(axis=2); fs=f.sum(axis=1); gr=g.sum(axis=1); gs=g.sum(axis=2)
        result=dict(bound=self.bound,steps=len(f),updates=self.updates,forward_receive_vectors=summary(fr),forward_send_vectors=summary(fs),backward_receive_vectors=summary(gr),backward_send_vectors=summary(gs),metadata_receive_bytes=summary(metadata_recv),metadata_send_bytes=summary(metadata_send),vector_bytes_128=int((f.sum()+g.sum())*128*4),vector_bytes_32=int((f.sum()+g.sum())*32*4),pending_updates=pending_updates,pending_embedding_vectors=int(flush.sum()),hypothetical_final_flush_vector_bytes_128=int(flush.sum()*128*4),hypothetical_final_flush_metadata_bytes=int(flush.sum()*16),drain_round_shape_bytes=3*8*self.p*(self.p-1),sample_normalization='all assigned rows, ragged last batches, no dropped or repeated samples; not upstream dataloader wraparound')
        return result,dict(forward=f,backward=g,queries=q,pending_flush=flush)

def main():
    global P, B, EPOCHS
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--partition',required=True); p.add_argument('--out',required=True)
    p.add_argument('--ranks',type=int,default=P); p.add_argument('--batch-size',type=int,default=B)
    p.add_argument('--epochs',type=int,default=EPOCHS); p.add_argument('--bounds',type=int,nargs='+',default=[0,10,100])
    p.add_argument('--skip-1d',action='store_true')
    a=p.parse_args(); P=a.ranks; B=a.batch_size; EPOCHS=a.epochs
    assert P > 0 and B > 0 and EPOCHS > 0 and 0 in a.bounds
    data=Path(a.data); part=Path(a.partition); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    owner=np.load(part/'owner.npy'); assignment=np.load(part/'assignment.npy'); sparse=np.load(data/'sparse.npy',mmap_mode='r')
    assert len(sparse)==len(assignment)
    rows=[np.flatnonzero(assignment==r) for r in range(P)]
    mirrors=[np.load(part/f'mirrors_{r}.npy') for r in range(P)]
    assert np.all((owner >= 0) & (owner < P)) and np.all((assignment >= 0) & (assignment < P))
    states={} if a.skip_1d else {'het_1d_s0':State(owner,[np.array([],dtype=np.int32) for _ in range(P)],0,P)}
    for b in a.bounds: states[f'het_gmp_s{b}']=State(owner,mirrors,b,P)
    steps=max((len(v)+B-1)//B for v in rows); start=time.perf_counter();trace=[]
    for s in range(steps):
        batches=[]
        for r in range(P):
            ids,cnt=np.unique(sparse[rows[r][s*B:(s+1)*B]].ravel(),return_counts=True)
            batches.append((ids,cnt))
        prepared=states['het_gmp_s0'].prepare(batches)
        trace.append((batches,prepared))
        for tag,state in states.items():state.step(batches,prepared if tag!='het_1d_s0' else None)
        print('replayed epoch 1',s+1,'/',steps,'seconds',round(time.perf_counter()-start,1),flush=True)
    for tag,state in states.items():
        first,_=state.finish();save(out/f'epoch1_{tag}.json',first)
    for epoch in range(1,EPOCHS):
        for s,(batches,prepared) in enumerate(trace):
            for tag,state in states.items():state.step(batches,prepared if tag!='het_1d_s0' else None)
            if (s+1)%8==0:print('replayed epoch',epoch+1,s+1,'/',steps,'seconds',round(time.perf_counter()-start,1),flush=True)
    for tag,state in states.items():
        result,arrays=state.finish()
        result.update(rows=len(sparse),epochs=EPOCHS,processed_rows=len(sparse)*EPOCHS,steps_per_epoch=steps,rank_rows=[len(v) for v in rows],mirror_counts=[len(m) for m in state.mirrors],primary_counts=np.bincount(owner,minlength=P).tolist(),active_embeddings=len(owner))
        save(out/f'{tag}.json',result); np.savez_compressed(out/f'{tag}.npz',**arrays)
    save(out/'complete.json',dict(seconds=time.perf_counter()-start,rows=len(sparse),epochs=EPOCHS,steps=steps*EPOCHS))

if __name__=='__main__':main()
