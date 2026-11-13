"""Split the final complete global batch into an independent validation file."""
import argparse
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--record-bytes', type=int, required=True)
p.add_argument('--global-batch', type=int, default=65536)
a = p.parse_args()

size = a.source.stat().st_size
batch_bytes = a.global_batch * a.record_bytes
if size < 2 * batch_bytes or size % batch_bytes:
    raise ValueError('source must contain at least two complete global batches')
a.out.mkdir(parents=True, exist_ok=True)
free = os.statvfs(a.out).f_bavail * os.statvfs(a.out).f_frsize
if free < size + 2**30:
    raise OSError(f'insufficient free space: need {size + 2**30}, have {free}')

for name, begin, end in [('train.bin', 0, size - batch_bytes),
                         ('eval.bin', size - batch_bytes, size)]:
    target = a.out / name
    if target.exists():
        raise FileExistsError(target)
    temporary = a.out / f'{name}.part'
    with a.source.open('rb') as src, temporary.open('xb') as dst:
        src.seek(begin)
        remaining = end - begin
        while remaining:
            block = src.read(min(8 * 2**20, remaining))
            if not block:
                raise EOFError(a.source)
            dst.write(block)
            remaining -= len(block)
    temporary.rename(target)
print(f'train_rows={(size - batch_bytes) // a.record_bytes} eval_rows={a.global_batch}')
