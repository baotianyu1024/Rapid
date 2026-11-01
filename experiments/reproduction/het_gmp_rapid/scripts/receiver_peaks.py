"""Mean over steps of the busiest receiving rank, from saved traffic traces."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=Path, default=Path(__file__).resolve().parents[1]/'results/main64')
args = parser.parse_args()
case = args.case
cfg = json.loads((case/'data/input.json').read_text())
out = case/'receiver_peaks'
out.mkdir(exist_ok=True)
results = {}
for tag in ['rapid_acp_original', 'het_gmp_s0', 'het_gmp_s10', 'het_gmp_s100']:
    meta = json.loads((case/'traffic'/f'{tag}.json').read_text())
    with np.load(case/'traffic'/f'{tag}.npz') as trace:
        # f[step, requester, owner]; g[step, gradient_sender, owner].
        f = trace['forward']
        g = trace['backward'] if 'backward' in trace else f
        fr = f.sum(axis=2)
        br = g.sum(axis=1)
    assert fr.shape == br.shape == (meta['steps'], cfg['ranks'])
    for values, field in [(fr, 'forward_receive_vectors'), (br, 'backward_receive_vectors')]:
        assert int(values.sum()) == meta[field]['total']
        assert np.isclose(values.max(axis=1).mean(), meta[field]['mean_batch_peak'])
    full_steps = (min(meta['rank_rows'])//cfg['local_batch'] if 'rank_rows' in meta
                  else meta['steps_per_epoch'])
    full = np.arange(meta['steps']) % meta['steps_per_epoch'] < full_steps
    entry = {'steps': meta['steps'], 'epochs': meta['epochs'], 'processed_rows': meta['processed_rows']}
    for scope, mask in [('all_steps', np.ones(meta['steps'], dtype=bool)), ('all_ranks_full_batch', full)]:
        entry[scope] = {'steps': int(mask.sum())}
        for name, values in [('forward', fr), ('backward', br), ('forward_plus_backward', fr+br)]:
            a = values[mask]
            peaks = a.max(axis=1)
            entry[scope][name] = {
                'mean_step_max_vectors': float(peaks.mean()),
                'mean_step_max_mib': float(peaks.mean()*cfg['dim']*4/2**20),
                'absolute_max_mib': float(peaks.max()*cfg['dim']*4/2**20),
                'rank_mean_mib': (a.mean(axis=0)*cfg['dim']*4/2**20).tolist(),
                'max_rank_mean_mib': float(a.mean(axis=0).max()*cfg['dim']*4/2**20),
            }
    np.savetxt(out/f'{tag}_per_step.csv',
        np.column_stack([np.arange(meta['steps']), fr.max(axis=1), br.max(axis=1), (fr+br).max(axis=1)]),
        delimiter=',', fmt='%d', header='step,forward_max_vectors,backward_max_vectors,combined_max_vectors', comments='')
    results[tag] = entry
report = dict(definition='mean_over_steps(max_over_8_ranks(received_bytes)); combined max taken after per-rank forward+backward sum',
              scope='embedding/gradient vectors only, excludes self, metadata and dense allreduce; logical trace replay',
              dim=cfg['dim'], dtype='FP32', input_rows=cfg['rows'], results=results)
(out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
for tag, entry in results.items():
    for scope in ['all_steps', 'all_ranks_full_batch']:
        v = entry[scope]
        print(tag, scope, v['steps'], {k:round(v[k]['mean_step_max_mib'],6) for k in ['forward','backward','forward_plus_backward']})
