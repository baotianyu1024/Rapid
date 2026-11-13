"""Reread scheduled records and independently enumerate remote payload and hit rates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'response_audit_20261108'
CASES = {name: Path('/tmp/rapid_fresh_stage') / f'{name}_8_v2'
         for name in ['kaggle', 'terabyte', 'taobao']}
CASES.update({name: Path('/tmp/rapid_tseries_d32_20261104') / sub
              for name, sub in [('t180', 't180'), ('t200', 't200'), ('t510', 't510_verified')]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=CASES, default=list(CASES))
    p.add_argument('--output-dir', type=Path, default=OUT)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.datasets:
        case = CASES[name]
        folder = case / 'optimized/peak_samples'
        cfg = json.loads((folder / 'config.json').read_text())
        g, b, nb = cfg['ranks'], cfg['batch_size'], cfg['batches']
        lengths, dense = cfg['table_lengths'], cfg['dense']
        cols = np.cumsum([1+dense] + lengths)
        width = int(cols[-1])
        receive = np.zeros((nb, len(lengths), g), dtype=np.int64)
        send = np.zeros_like(receive)
        request_hits = 0
        unique_requests = 0
        for rank in range(g):
            shard = folder / 'scheduled' / f'dataset_{rank}.bin'
            rows = np.memmap(shard, dtype='<u4', mode='r').reshape(-1, width)
            assert len(rows) == nb*b
            for step in range(nb):
                block = rows[step*b:(step+1)*b]
                for t in range(len(lengths)):
                    ids = block[:, cols[t]:cols[t+1]]
                    # Sorting the raw occurrences gives the table-specific request set.
                    ordered = np.sort(ids.ravel())
                    mask = np.r_[True, ordered[1:] != ordered[:-1]]
                    keys = ordered[mask]
                    owners = keys % g
                    remote_owners = owners[owners != rank]
                    receive[step, t, rank] = len(remote_owners)
                    send[step, t] += np.bincount(remote_owners.astype(np.int64), minlength=g)
                    request_hits += int(np.count_nonzero(ids % g == rank))
                    unique_requests += len(keys)
            print('REREAD', name, 'rank', rank, flush=True)
        assert receive.sum() == send.sum()
        with np.load(folder / 'cps_counts.npz') as old:
            assert np.array_equal(receive, old['receive'])
            assert np.array_equal(send, old['send'])
        unique_tables = receive.mean(axis=(0, 2)) < b*(g-1)
        forward = np.where(unique_tables[None, :, None], receive, b*(g-1)).sum(axis=1)
        backward = np.where(unique_tables[None, :, None], send, b*(g-1)).sum(axis=1)
        dim = 32 if name in ('t180', 't200', 't510') else 128
        total_gib = float((forward.sum()+backward.sum())*dim*4*5/2**30)
        prior = json.loads((case / 'communication.json').read_text())
        scaled_prior = prior['all_rank_forward_backward_GiB'] * dim / prior['dim']
        assert abs(total_gib - scaled_prior) < 1e-9
        metric = json.loads((folder / 'cps_metrics.json').read_text())
        request_rate = request_hits/(nb*g*b*sum(lengths))
        unique_rate = 1-receive.sum()/unique_requests
        assert abs(request_rate-metric['local_hit_rate']) < 1e-12
        assert abs(unique_rate-metric['unique_local_hit_rate']) < 1e-12
        result = dict(dataset=name, dim=dim, ranks=g, local_batch=b, batches=nb, epochs=5,
                      total_embedding_payload_gib=total_gib, request_local_hit_rate=request_rate,
                      deduplicated_id_local_hit_rate=float(unique_rate),
                      mean_rank_vectors_per_step=float((forward+backward).mean()),
                      mean_step_peak=float((forward+backward).max(axis=1).mean()),
                      absolute_peak=int((forward+backward).max()),
                      cached_counts_exact_match=True, case=str(case))
        results.append(result)
        (args.output_dir / 'traffic_recount.json').write_text(json.dumps(results, indent=2)+'\n')
        print('RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
