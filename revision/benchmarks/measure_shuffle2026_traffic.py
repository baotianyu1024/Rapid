"""Measure remote Unique vectors and static ACP choices on one trace layout."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'Rapid/acp'))
from compute_acp import compute_unique_per_table_cps, compute_unique_per_table_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--cps-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lengths', type=int, nargs='+', required=True)
    parser.add_argument('--dense', type=int, required=True)
    parser.add_argument('--ranks', type=int, default=8)
    parser.add_argument('--local-batch', type=int, default=8192)
    parser.add_argument('--batches', type=int, default=16)
    args = parser.parse_args()
    if args.cps_dir:
        remote, _, _ = compute_unique_per_table_cps(str(args.cps_dir), args.ranks,
            args.local_batch, args.batches, args.lengths, args.dense)
        layout = 'CPS scheduled; owner = physical ID modulo ranks'
    else:
        remote, _ = compute_unique_per_table_hash(str(args.input), args.ranks,
            args.local_batch, args.batches, args.lengths, args.dense)
        layout = 'source order; owner = original ID modulo ranks'
    reduction = args.local_batch * (args.ranks - 1)
    tables = [dict(table=i, pooling=int(length), unique_remote_vectors=float(value),
        reduction_remote_vectors=reduction, strategy='Unique' if value < reduction else 'Reduction')
        for i, (length, value) in enumerate(zip(args.lengths, remote))]
    result = dict(input=str(args.input), layout=layout, ranks=args.ranks,
        local_batch=args.local_batch, batches=args.batches,
        unit='vectors per rank per global batch, forward remote payload only',
        acp_timing='one static choice per table before training', tables=tables,
        all_unique_vectors=sum(t['unique_remote_vectors'] for t in tables),
        all_reduction_vectors=reduction * len(tables),
        acp_vectors=sum(min(t['unique_remote_vectors'], reduction) for t in tables))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Unique:', [t['table'] for t in tables if t['strategy'] == 'Unique'])
    print('Reduction:', [t['table'] for t in tables if t['strategy'] == 'Reduction'])
    print('ACP vectors:', result['acp_vectors'])


if __name__ == '__main__':
    main()
