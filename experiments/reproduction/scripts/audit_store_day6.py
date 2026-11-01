"""Count table-scoped Unique on the materialized readable prefix of store/day_6.gz."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_day6_receive import recount, summarize
from run_case import LENGTHS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ranks', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=8192)
    ap.add_argument('--batches', type=int, default=64)
    ap.add_argument('--spread', action='store_true')
    args = ap.parse_args()
    meta = json.loads(args.metadata.read_text())
    width = 1 + 13 + sum(LENGTHS)
    if args.input.stat().st_size != meta['rows']*meta['record_bytes']:
        raise ValueError('File/metadata size mismatch')
    data = np.memmap(args.input, dtype='<u4', mode='r').reshape(-1, width)
    available = len(data)//(args.ranks*args.batch_size)
    if args.batches == 0:
        args.batches = available
    if args.batches > available:
        raise ValueError(f'Only {available} complete global batches available')
    ids = (np.linspace(0, available-1, args.batches, dtype=np.int64)
           if args.spread else np.arange(args.batches, dtype=np.int64))
    count, remote, fid_count, fid_remote, diag = recount(
        data, ids, args.ranks, args.batch_size)
    result = summarize(count, remote, fid_count, fid_remote,
                       ids, args.ranks, args.batch_size, diag)
    result['dataset'] = 'readable prefix of store/day_6.gz (truncated gzip)'
    result['source_gzip_sha256'] = meta['input_sha256']
    result['source_gzip_integrity'] = meta['input_gzip_integrity']
    result['input_rows'] = len(data)
    result['complete_global_batches_in_input'] = available
    result['batch_selection'] = ('evenly spaced across readable prefix'
                                 if args.spread else 'first batches after global shuffle')
    result['input'] = str(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    with args.output.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result['per_table'][0]))
        writer.writeheader()
        writer.writerows(result['per_table'])
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ('per_table','fid_only_diagnostic_first_8_batches')},
                     indent=2))


if __name__ == '__main__':
    main()
