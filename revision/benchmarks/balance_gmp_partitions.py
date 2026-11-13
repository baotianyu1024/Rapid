"""Make GMP data ranks exactly balanced while retaining embedding placement."""
import json
import shutil
from pathlib import Path
import numpy as np

root = Path('/dev/shm/rapid_read_fair')
results = {}
for ds in ('kaggle', 'terabyte', 'taobao'):
    inp = root / ds / 'het/partition.npz'
    out = root / ds / 'het/partition_balanced.npz'
    if out.exists():
        raise RuntimeError(f'{out} already exists')
    if shutil.disk_usage(root).free < inp.stat().st_size + 1024**3:
        raise RuntimeError(f'insufficient tmpfs space before writing {out}')
    available = next(int(x.split()[1]) * 1024 for x in Path('/proc/meminfo').read_text().splitlines()
                     if x.startswith('MemAvailable:'))
    if available < 2 * inp.stat().st_size + 4 * 1024**3:
        raise RuntimeError(f'insufficient memory before writing {out}')
    with np.load(inp) as z:
        arrays = {k: z[k] for k in z.files}
    old = arrays['data_partition']
    data = old.copy()
    counts = np.bincount(data, minlength=8)
    if len(data) % 8 or len(counts) != 8:
        raise ValueError('expected eight ranks with divisible sample count')
    target = len(data) // 8
    for donor in range(8):
        excess = int(counts[donor] - target)
        if excess <= 0:
            continue
        candidates = np.flatnonzero(data == donor)[-excess:]
        offset = 0
        for recipient in range(8):
            n = min(excess - offset, int(target - counts[recipient]))
            if n > 0:
                data[candidates[offset:offset+n]] = recipient
                counts[recipient] += n
                counts[donor] -= n
                offset += n
            if offset == excess:
                break
        if offset != excess:
            raise ValueError('could not balance data ranks')
    if not np.all(np.bincount(data, minlength=8) == target):
        raise ValueError('unbalanced result')
    arrays['data_partition'] = data
    np.savez_compressed(out, **arrays)
    with np.load(out) as verify:
        if not np.array_equal(verify['data_partition'], data):
            raise ValueError('written data partition mismatch')
        for name in arrays:
            if name != 'data_partition' and not np.array_equal(verify[name], arrays[name]):
                raise ValueError(f'embedding partition changed: {name}')
    results[ds] = dict(target=target, original_counts=np.bincount(old, minlength=8).tolist(),
                       moved=int(np.count_nonzero(old != data)),
                       note='embedding roots and mirrors byte-for-byte unchanged')
    print(ds, results[ds], flush=True)
(root / 'gmp_balance_audit.json').write_text(json.dumps(results, indent=2) + '\n')
