"""
Extract 4 large tables from Taobao 15-sparse dataset.

Input format (68 bytes/sample):
  clk(i4) + price(f4) + 15 sparse(i4)

Output format (24 bytes/sample):
  clk(i4) + price(f4) + 4 sparse(i4)

Tables extracted: user(col 0), adgroup_id(col 9), campaign_id(col 11), customer(col 12)
"""
import numpy as np
import os
import shutil
import sys
import time

SPARSE_COLS = [0, 9, 11, 12]  # user, adgroup_id, campaign_id, customer
NUM_SPARSE_IN = 15
NUM_SPARSE_OUT = len(SPARSE_COLS)

input_path = sys.argv[1] if len(sys.argv) > 1 else 'taobao/taobao_dataset_shuffled.bin'
output_path = sys.argv[2] if len(sys.argv) > 2 else 'taobao/taobao_4tables.bin'

dt_in = np.dtype([
    ('label', 'i4'),
    ('dense', 'f4'),
    ('sparse', 'i4', (NUM_SPARSE_IN,))
])

dt_out = np.dtype([
    ('label', 'i4'),
    ('dense', 'f4'),
    ('sparse', 'i4', (NUM_SPARSE_OUT,))
])

print(f"Reading {input_path}...")
t0 = time.time()
data = np.fromfile(input_path, dtype=dt_in)
n = len(data)
print(f"  {n:,} samples, {time.time()-t0:.1f}s")

print(f"Extracting columns {SPARSE_COLS}...")
free_bytes = shutil.disk_usage(os.path.dirname(os.path.abspath(output_path))).free
needed_bytes = n * dt_out.itemsize
if free_bytes < needed_bytes:
    raise OSError(f"Insufficient output space: need {needed_bytes:,}, free {free_bytes:,} bytes")
out = np.zeros(n, dtype=dt_out)
out['label'] = data['label']
out['dense'] = data['dense']
for i, col in enumerate(SPARSE_COLS):
    out['sparse'][:, i] = data['sparse'][:, col]

print(f"Writing {output_path}...")
out.tofile(output_path)
elapsed = time.time() - t0
fsize = os.path.getsize(output_path)
print(f"Done: {n:,} samples, {fsize:,} bytes ({fsize/1e9:.2f} GB), {elapsed:.1f}s")
print(f"Record size: {dt_out.itemsize} bytes")
print(f"Vocab sizes: user={data['sparse'][:,0].max()+1}, "
      f"adgroup={data['sparse'][:,9].max()+1}, "
      f"campaign={data['sparse'][:,11].max()+1}, "
      f"customer={data['sparse'][:,12].max()+1}")
