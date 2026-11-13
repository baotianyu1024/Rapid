"""Count T-series payload on the exact compact IDs consumed by HugeCTR."""
import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan-root', type=Path, default=Path('/tmp/rapid_tseries_d32_20261104'))
parser.add_argument('--gpu-root', type=Path, default=Path('/tmp/rapid_tseries_d32_gpu_20261104'))
parser.add_argument('--official-root', type=Path, default=Path('/tmp'))
parser.add_argument('--t510-case', default='t510_verified')
args = parser.parse_args()
ROOT = args.plan_root
GPU = args.gpu_root
for name, case in [('t180', 't180'), ('t200', 't200'), ('t510', args.t510_case)]:
    audit = json.loads((args.official_root / f'rapid_official_{name}_20261103/hash_acp_audit_b512.json').read_text())
    meta = json.loads((ROOT / case / 'data/input.json').read_text())
    rapid = json.loads((ROOT / case / 'communication.json').read_text())
    if (audit['ranks'], audit['local_batch'], audit['batches']) != (8, 512, 16) or meta['rows'] != 65536:
        raise ValueError(f'{name}: unexpected HugeCTR window')
    if (rapid['dim'], rapid['epochs']) != (128, 5):
        raise ValueError(f'{name}: unexpected RAPID replay window')
    lengths = meta['lengths']
    width = 1 + meta['dense'] + sum(lengths)
    source = GPU / f'{name}_native_compact.bin'
    if source.stat().st_size != meta['rows'] * width * 4:
        raise ValueError(f'{name}: compact input size mismatch')
    data = np.memmap(source, dtype='<u4', mode='r', shape=(meta['rows'], width))
    reduction = ({t['table'] for t in audit['tables'] if t['strategy'] == 'Reduction'}
                 if name == 't180' else set())
    remote = 0
    left = 1 + meta['dense']
    for table, hotness in enumerate(lengths):
        right = left + hotness
        if table in reduction:
            remote += 512 * 7 * 8 * 16
        else:
            for batch in range(16):
                for rank in range(8):
                    start = batch * 4096 + rank * 512
                    unique = np.unique(data[start:start + 512, left:right])
                    remote += int(np.count_nonzero(unique % 8 != rank))
        left = right
    hugectr_gib = remote * 32 * 4 * 2 * 5 / 2**30
    rapid_gib = rapid['all_rank_forward_backward_GiB'] * 32 / 128
    print(f'{name.upper()}: HugeCTR={hugectr_gib:.6f} GiB '
          f'RAPID={rapid_gib:.6f} GiB reduction={1 - rapid_gib / hugectr_gib:.2%}')
