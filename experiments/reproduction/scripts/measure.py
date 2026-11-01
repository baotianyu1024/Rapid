"""Trace measurements under receive-only traffic, excluding local vectors.

Unique deduplicates within each table, so its key is (table_id, fid).
Reduction uses the Eq.(5) dense all-to-all model with the self-rank removed:
B*(G-1) received vectors/table. Paper's B*G is retained as a reference-only
field. Metadata/backward traffic is excluded. Sum(send)==sum(receive).
"""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(a):
    a = np.asarray(a, dtype=np.float64)
    means = a.mean(axis=1)
    peaks = a.max(axis=1)
    return dict(mean=float(means.mean()), mean_batch_peak=float(peaks.mean()),
                absolute_peak=float(peaks.max()),
                peak_of_rank_means=float(a.mean(axis=0).max()),
                rank_means=a.mean(axis=0).tolist(),
                mean_batch_peak_over_mean=float(np.divide(peaks, means,
                    out=np.ones_like(means), where=means != 0).mean()),
                max_batch_peak_over_mean=float(np.divide(peaks, means,
                    out=np.ones_like(means), where=means != 0).max()))


def summarize_counts(recv, send, unique, hits, ranks, batch, lengths, cps=False, paths=None):
    batches=recv.shape[0]
    assert recv.shape==send.shape==unique.shape==hits.shape==(batches,len(lengths),ranks)
    assert np.array_equal(send.sum(axis=(1,2)),recv.sum(axis=(1,2)))
    reduction_receive=batch*(ranks-1)
    choose_unique=recv.mean(axis=(0,2))<reduction_receive
    acp_recv=np.where(choose_unique[None,:,None],recv,reduction_receive)
    acp_send=np.where(choose_unique[None,:,None],send,reduction_receive)
    result=dict(ranks=ranks,batch_size=batch,batches=batches,cps=cps,
        paths=[str(Path(p).resolve()) for p in (paths or [])],
        metric='received remote forward embedding vectors; no local, metadata, or backward traffic',
        acp_selection='fixed per table from measured batches (in-sample); Reduction B*(G-1) excludes self',
        local_hit_rate=float(hits.sum()/(batches*ranks*batch*sum(lengths))),
        unique_local_hit_rate=float(1-recv.sum()/unique.sum()),
        remote_deduplication_rate=float(1-recv.sum()/(batches*ranks*batch*sum(lengths))),
        all_reduction_paper_per_rank=batch*ranks*len(lengths),
        all_reduction_excluding_self_per_rank=reduction_receive*len(lengths),
        all_reduction_receive_per_rank=reduction_receive*len(lengths),
        unique_receive=summarize(recv.sum(axis=1)),
        unique_send=summarize(send.sum(axis=1)),
        unique_send_plus_receive=summarize((recv+send).sum(axis=1)),
        acp_receive=summarize(acp_recv.sum(axis=1)),
        acp_send=summarize(acp_send.sum(axis=1)),
        acp_send_plus_receive=summarize((acp_recv+acp_send).sum(axis=1)),
        per_table=[dict(table=t,pooling=lengths[t],
            unique_receive_mean=float(recv[:,t].mean()),
            unique_count_mean=float(unique[:,t].mean()),
            local_hit_rate=float(hits[:,t].sum()/(batches*ranks*batch*lengths[t])),
            unique_local_hit_rate=float(1-recv[:,t].sum()/unique[:,t].sum()),
            reduction_receive_per_rank=reduction_receive,
            strategy='Unique' if choose_unique[t] else 'Reduction') for t in range(len(lengths))])
    return result


def measure(paths, ranks, batch, batches, lengths, dense, cps=False):
    width = 1 + dense + sum(lengths)
    arrays = []
    for p in paths:
        if Path(p).stat().st_size % (4 * width):
            raise ValueError(f'Truncated record: {p}')
        arrays.append(np.memmap(p, dtype='<u4', mode='r').reshape(-1, width))
    available = min(len(a) // batch for a in arrays) if cps else len(arrays[0]) // (batch * ranks)
    if available < batches or batches < 1:
        raise ValueError(f'Requested {batches} batches, available {available}')
    offsets = np.cumsum([0] + lengths)
    recv = np.zeros((batches, len(lengths), ranks), dtype=np.int64)
    send = np.zeros_like(recv)
    unique = np.zeros_like(recv)
    hits = np.zeros_like(recv)
    for b in range(batches):
        for r in range(ranks):
            start = b * batch if cps else (b * ranks + r) * batch
            rows = arrays[r if cps else 0][start:start + batch, 1 + dense:]
            for t in range(len(lengths)):
                ids = rows[:, offsets[t]:offsets[t+1]]
                u = np.unique(ids)
                owners = u % ranks
                owner_counts = np.bincount(owners.astype(np.int64), minlength=ranks)
                unique[b,t,r] = len(u)
                hits[b,t,r] = np.count_nonzero(ids % ranks == r)
                owner_counts[r] = 0
                recv[b,t,r] = owner_counts.sum()
                send[b,t] += owner_counts
        assert np.array_equal(send[b].sum(axis=1), recv[b].sum(axis=1))
        print(f'measured batch {b+1}/{batches}', flush=True)
    result=summarize_counts(recv,send,unique,hits,ranks,batch,lengths,cps,paths)
    return result, dict(receive=recv,send=send,unique=unique,hits=hits)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--cps',action='store_true')
    p.add_argument('--ranks',type=int,required=True)
    p.add_argument('--batch',type=int,default=8192)
    p.add_argument('--batches',type=int,default=10)
    p.add_argument('--lengths',nargs='+',type=int,required=True)
    p.add_argument('--dense',type=int,default=13)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    paths=[str(Path(a.input)/f'dataset_{r}.bin') for r in range(a.ranks)] if a.cps else [a.input]
    result,raw=measure(paths,a.ranks,a.batch,a.batches,a.lengths,a.dense,a.cps)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),**raw)
