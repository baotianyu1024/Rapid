"""Count logical remote bytes from a real TorchRec plan, not measured network traffic."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def table_bytes(table, p, b, steps):
    """Global cross-rank FP32 bytes; IDs and KJT lengths int64."""
    kind, d, k = table['sharding_type'], table['dim'], table['pooling']
    unit = (p - 1) * b * d * 4 * steps
    if kind in ('table_wise', 'column_wise', 'table_column_wise'):
        copies = len(table['shards'])
        assert sum(s['size'][1] for s in table['shards']) == d
        return dict(forward_bytes=unit, backward_bytes=unit,
                    id_bytes=(p - 1)*b*k*8*steps*copies,
                    length_bytes=(p - 1)*b*8*steps*copies)
    if kind in ('row_wise', 'table_row_wise'):
        assert len(table['shards']) == p, 'Only single-host full-row partition supported'
        # Each rank moves (p - 1) output shards in both collective phases.
        return dict(forward_bytes=p*unit, backward_bytes=p*unit,
                    id_bytes=None, length_bytes=p*(p - 1)*b*8*steps)
    if kind == 'data_parallel':
        return dict(forward_bytes=0, backward_bytes=2*(p - 1)*table['rows']*d*4*steps,
                    id_bytes=0, length_bytes=0)
    raise ValueError(f'Unsupported plan: {kind}')

# Import table_bytes from report_comparisons.py; historical CLI defaults were removed.
