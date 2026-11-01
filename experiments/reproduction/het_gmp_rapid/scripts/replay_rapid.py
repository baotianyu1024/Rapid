"""RAPID fixed ACP choices, measured output ID trace, explicit vector-only model."""
import argparse
import numpy as np
from common import *
from replay_het import summary

def count(paths,cps,nb):
    arrays=[np.memmap(path,dtype=np.int32,mode='r').reshape(-1,WIDTH) for path in paths]
    mats=np.zeros((nb,len(LENGTHS),P,P),dtype=np.int64)
    offsets=np.cumsum([0]+LENGTHS)
    for b in range(nb):
        for r in range(P):
            row=b*B if cps else (b*P+r)*B
            batch=arrays[r if cps else 0][row:row+B,1+DENSE:]
            assert len(batch)==B
            for t in range(len(LENGTHS)):
                ids=np.unique(batch[:,offsets[t]:offsets[t+1]].astype(np.uint32))
                owners=np.bincount((ids%P).astype(np.int64),minlength=P); owners[r]=0
                mats[b,t,r]=owners
        print('rapid count',cps,b+1,'/',nb,flush=True)
    return mats

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--rapid',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    data=Path(a.data); rapid=Path(a.rapid); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((data/'input.json').read_text()); nb=cfg['global_batches']; profile=min(nb,50)
    original=json.loads((rapid/'acp.json').read_text())[str(P)]
    for source,paths,cps in [('hash',[data/'input.bin'],False),('rapid',[rapid/'scheduled'/f'dataset_{r}.bin' for r in range(P)],True)]:
        mats=count(paths,cps,nb)
        costs=mats[:profile].sum(axis=3).mean(axis=(0,2))
        original_tables=original['acp_cps_per_table' if cps else 'per_table']
        assert np.allclose(costs,[t['c_unique'] for t in original_tables])
        for variant,threshold in [('unique',None),('acp_original',B*P),('acp_remote',B*(P-1))]:
            selected=np.ones(len(LENGTHS),dtype=bool) if threshold is None else costs<threshold
            mixed=mats.copy()
            for t in np.flatnonzero(~selected):
                mixed[:,t]=B
                for r in range(P):mixed[:,t,r,r]=0
            m=mixed.sum(axis=1); fr=m.sum(axis=2); fs=m.sum(axis=1)
            m=np.tile(m,(EPOCHS,1,1));fr=m.sum(axis=2);fs=m.sum(axis=1)
            assert fr.sum()==fs.sum()
            tag=f'{source}_{variant}'
            result=dict(rows=cfg['rows'],epochs=EPOCHS,processed_rows=cfg['rows']*EPOCHS,steps=nb*EPOCHS,steps_per_epoch=nb,profile_batches=profile,strategies=['Unique' if u else 'Reduction' for u in selected],selection_threshold=threshold,forward_receive_vectors=summary(fr),forward_send_vectors=summary(fs),backward_receive_vectors=summary(fs),backward_send_vectors=summary(fr),vector_bytes_128=int(2*m.sum()*DIM*4),vector_bytes_32=int(2*m.sum()*32*4),metadata='Not modeled: RAPID delegates kernels to external HugeCTR; do not compare with HET metadata-inclusive totals',model='Unique: one vector/remote unique(table,id); Reduction: dense B per ordered rank pair/table; backward is reverse traffic assuming local gradient aggregation, no compression',selection='fixed using first up to 50 output batches; in-sample, not oracle per batch')
            save(out/f'{tag}.json',result); np.savez_compressed(out/f'{tag}.npz',forward=m,per_table_single_epoch=mixed)
    save(out/'complete.json',dict(rows=cfg['rows'],epochs=EPOCHS,steps=nb*EPOCHS))

if __name__=='__main__':main()
