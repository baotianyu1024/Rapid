"""Independently verify original-prefix seed-42 shuffle via labels and IDs."""
import importlib.util
import json
from pathlib import Path
import numpy as np

script = Path('/home/longjiu.bty/Rapid/code/Rapid/data_preparation/criteo_singlehot_to_multihot.py')
spec = importlib.util.spec_from_file_location('criteo_preprocess', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
n = 1048576
labels = np.empty(n, dtype=np.int32)
ids = np.empty(n, dtype=np.int32)
raw = Path('/mnt/nebula/cn-zhangjiakou/juicefs/longjiu.bty/rapid_experiment_data/fresh_datasets/criteo_kaggle/dac/train.txt')
with raw.open('rb') as f:
    for i in range(n):
        columns = f.readline().rstrip(b'\n').split(b'\t')
        if len(columns) != 40:
            raise ValueError(f'malformed source row {i}')
        labels[i] = int(columns[0])
        ids[i] = int(columns[34], 16) % module.NUM_EMBEDDINGS[20] if columns[34] else 0
perm = np.random.RandomState(42).permutation(n)
binary = np.memmap('/tmp/rapid_fresh_stage/kaggle.bin', dtype=module.RECORD_DTYPE, mode='r')
start = sum(module.MULTI_HOT_SIZES[:20])
label_errors = int(np.count_nonzero(labels[perm] != binary['label']))
id_errors = int(np.count_nonzero(ids[perm] != binary['sparse'][:, start]))
result = dict(source=str(raw), rows_checked=n, seed=42, label_errors=label_errors,
              first_table20_id_errors=id_errors, raw_rows_used='first 1048576 rows only')
Path('/tmp/rapid_kaggle_shuffle_source_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
