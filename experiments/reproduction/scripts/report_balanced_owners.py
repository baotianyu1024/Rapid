"""Compare independently verified joint-owner refinement with its RAPID base."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args()
    case=Path(a.case).resolve()
    refinement=json.loads((case/'refinement.json').read_text())
    base=Path(refinement['base_case'])
    assert json.loads((case/'complete.json').read_text())['all_records_inverse_verified']
    cfg=json.loads((case/'config.json').read_text());g=cfg['ranks']
    summaries={}
    for name,path in [('baseline',base),('joint_balanced',case)]:
        metrics=json.loads((path/'cps_metrics.json').read_text())
        assert all(t['strategy']=='Unique' for t in metrics['per_table'])
        counts=np.load(path/'cps_counts.npz')
        storage=np.array(refinement['before_storage' if name=='baseline' else 'after_storage'])
        summaries[name]=dict(forward=metrics['acp_receive'],backward=metrics['acp_send'],
            total_forward_vectors=int(counts['receive'].sum()),
            total_forward_backward_vectors=int(counts['receive'].sum()+counts['send'].sum()),
            storage=dict(counts=storage.tolist(),mean=float(storage.mean()),minimum=int(storage.min()),
                         maximum=int(storage.max()),max_over_mean=float(storage.max()/storage.mean()),
                         min_over_mean=float(storage.min()/storage.mean()),mirrors=0))
    observed=np.zeros(g,dtype=np.int64)
    for t in range(len(cfg['table_lengths'])):
        with np.load(case/f'table_{t}_remap.npz') as m:
            assert len(np.unique(m['new']))==len(m['old'])
            assert np.array_equal(m['new']%g,m['owner'])
            observed+=np.bincount(m['owner'],minlength=g)
    assert observed.tolist()==refinement['after_storage']
    before,after=summaries['baseline'],summaries['joint_balanced']
    traffic_change=after['total_forward_vectors']/before['total_forward_vectors']-1
    result=dict(base_case=str(base),case=str(case),summaries=summaries,
        traffic_change=traffic_change,records_inverse_verified=True,
        constraints='same samples, rank assignments, per-rank batches, no mirrors; all-Unique vector model',
        limitation='optimized and verified on the same offline trace; no held-out generalization or GPU/AUC measurement')
    (case/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['RAPID joint forward/backward/storage owner refinement','',
        f"Ranks: {g}; batches: {cfg['batches']}; local batch: {cfg['batch_size']}",
        f'Baseline: {base}',f'Output: {case}','',
        'Metric | baseline | refined']
    for direction in ['forward','backward']:
        for metric in ['mean','mean_batch_peak','absolute_peak','max_batch_peak_over_mean']:
            lines.append(f"{direction}/{metric}: {before[direction][metric]:.6f} -> {after[direction][metric]:.6f}")
    for metric in ['minimum','mean','maximum','max_over_mean','min_over_mean']:
        lines.append(f"storage/{metric}: {before['storage'][metric]:.6f} -> {after['storage'][metric]:.6f}")
    lines += [f'Forward plus backward vector count: {before["total_forward_backward_vectors"]} -> {after["total_forward_backward_vectors"]}',
        f'Traffic change: {traffic_change*100:.6f}%', '',
        'Verified: complete inverse mapping of every written record, unchanged labels/dense/samples/order/batches;',
        'per-table bijective ID maps, unique ID counts, exact predicted per-batch forward and backward traffic;',
        'no mirror copies; compact per-table/per-owner IDs (not a GPU allocation measurement).',
        'GMP-inspired owner service-load and storage balancing, exact RAPID batch-presence costs.',
        'This is an additional offline optimizer; not original RAPID and not a claim of theoretical optimality.',
        'In-sample trace optimization only. No training throughput, accuracy or unseen-trace claim.']
    (case/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
