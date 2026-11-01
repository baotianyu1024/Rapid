"""Recount Criteo day6 with table-scoped Unique and receive-only traffic.

Each observation is one global batch of G*B records. Reduction follows the
dense Eq.(5) model, excluding the self rank: B*(G-1) received vectors/table.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_case import LENGTHS


def recount(data, batch_ids, ranks, batch_size):
    offsets = np.cumsum([0] + LENGTHS)
    shape = (len(batch_ids), len(LENGTHS), ranks)
    unique_count = np.zeros(shape, np.int64)
    unique_remote = np.zeros(shape, np.int64)
    fid_only_count = np.zeros((len(batch_ids), ranks), np.int64)
    fid_only_remote = np.zeros_like(fid_only_count)
    fid_only = []
    for j, b in enumerate(batch_ids):
        for r in range(ranks):
            start = (int(b) * ranks + r) * batch_size
            sparse = data[start:start + batch_size, 14:]
            for t in range(len(LENGTHS)):
                ids = np.unique(sparse[:, offsets[t]:offsets[t+1]])
                unique_count[j, t, r] = len(ids)
                unique_remote[j, t, r] = np.count_nonzero(ids % ranks != r)
            # Diagnostic only: this intentionally merges equal FIDs across tables.
            plain_ids = np.unique(sparse)
            fid_only_count[j, r] = len(plain_ids)
            fid_only_remote[j, r] = np.count_nonzero(plain_ids % ranks != r)
            if j < 8:
                fid_only.append((int(b), r, int(fid_only_count[j,r]),
                                 int(unique_count[j, :, r].sum())))
        print(f"{j+1}/{len(batch_ids)} global batches", flush=True)
    return unique_count, unique_remote, fid_only_count, fid_only_remote, fid_only


def summarize(count, remote, fid_only_count, fid_only_remote, batch_ids, ranks, batch_size, fid_only):
    red_per_table = batch_size * (ranks - 1)
    per_table_unique = remote.mean(axis=(0, 2))
    choose_reduction = per_table_unique >= red_per_table
    combined = np.where(choose_reduction[None, :, None], red_per_table, remote)
    raw = batch_size * sum(LENGTHS)
    rows = []
    for t, length in enumerate(LENGTHS):
        rows.append(dict(table=t, pooling=length, raw_requests=batch_size*length,
                         unique_after_dedup=float(count[:,t,:].mean()),
                         unique_remote_receive=float(remote[:,t,:].mean()),
                         local_unique=float((count[:,t,:]-remote[:,t,:]).mean()),
                         reduction_remote_receive=red_per_table,
                         rapid_strategy='Reduction' if choose_reduction[t] else 'Unique',
                         rapid_remote_receive=float(combined[:,t,:].mean())))
    mean_count=float(count.sum(axis=1).mean())
    mean_remote=float(remote.sum(axis=1).mean())
    mean_combined=float(combined.sum(axis=1).mean())
    result=dict(dataset='Criteo Terabyte day6, 2015-02-21',
                rows_measured=len(batch_ids)*ranks*batch_size,
                global_batches=len(batch_ids), first_batch=int(batch_ids[0]),
                last_batch=int(batch_ids[-1]), ranks=ranks, batch_size_per_rank=batch_size,
                raw_requests_per_rank_batch=raw,
                table_scoped_unique_per_rank_batch=mean_count,
                duplicate_elimination_rate=1-mean_count/raw,
                local_unique_per_rank_batch=mean_count-mean_remote,
                all_unique_remote_receive_per_rank_batch=mean_remote,
                fid_only_unique_including_local_per_rank_batch=float(fid_only_count.mean()),
                fid_only_remote_receive_per_rank_batch=float(fid_only_remote.mean()),
                cross_table_false_merge_including_local_per_rank_batch=mean_count-float(fid_only_count.mean()),
                cross_table_false_merge_remote_per_rank_batch=mean_remote-float(fid_only_remote.mean()),
                remote_effective_reduction_rate=1-mean_remote/raw,
                all_reduction_remote_receive_per_rank_batch=red_per_table*len(LENGTHS),
                rapid_unique_tables=[int(t) for t in np.where(~choose_reduction)[0]],
                rapid_reduction_tables=[int(t) for t in np.where(choose_reduction)[0]],
                rapid_unique_remote_receive_per_rank_batch=float(remote[:,~choose_reduction,:].sum(axis=1).mean()),
                rapid_reduction_remote_receive_per_rank_batch=int(red_per_table*choose_reduction.sum()),
                rapid_total_remote_receive_per_rank_batch=mean_combined,
                rapid_mean_batch_receive_peak=float(combined.sum(axis=1).max(axis=1).mean()),
                rapid_absolute_receive_peak=int(combined.sum(axis=1).max()),
                all_unique_mean_batch_receive_peak=float(remote.sum(axis=1).max(axis=1).mean()),
                all_unique_absolute_receive_peak=int(remote.sum(axis=1).max()),
                fid_only_diagnostic_first_8_batches=fid_only,
                per_table=rows)
    assert abs(result['rapid_unique_remote_receive_per_rank_batch']+
               result['rapid_reduction_remote_receive_per_rank_batch']-mean_combined)<1e-8
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--ranks',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=8192)
    p.add_argument('--batches',type=int,default=64)
    p.add_argument('--spread',action='store_true')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    width=1+13+sum(LENGTHS)
    data=np.memmap(a.input,dtype='<u4',mode='r').reshape(-1,width)
    available=len(data)//(a.ranks*a.batch_size)
    if a.batches>available:raise ValueError('not enough complete batches')
    ids=(np.linspace(0,available-1,a.batches,dtype=np.int64)
         if a.spread else np.arange(a.batches,dtype=np.int64))
    count,remote,fid_only_count,fid_only_remote,fid_only=recount(data,ids,a.ranks,a.batch_size)
    result=summarize(count,remote,fid_only_count,fid_only_remote,ids,a.ranks,a.batch_size,fid_only)
    result['input']=str(Path(a.input).resolve())
    result['total_day_rows']=len(data)
    result['complete_global_batches_in_day']=available
    result['batch_selection']='evenly spaced across complete day' if a.spread else 'first complete batches after global shuffle'
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    with out.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(result['per_table'][0]));writer.writeheader();writer.writerows(result['per_table'])
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_table','fid_only_diagnostic_first_8_batches')},indent=2))


if __name__=='__main__':main()
