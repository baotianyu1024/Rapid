"""
Taobao Display Ad dataset preprocessor.
Reads 3 CSV tables -> merge -> clean -> encode -> shuffle -> export binary + metadata.

Binary format per sample (68 bytes = 17 x int32/float32):
  clk(int32) | price(float32) | user(int32) | cms_segid(int32) | ... | pid(int32)

Usage:
  python3 preprocess.py [--raw_dir ./raw] [--output_dir .] [--seed 42]
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

SPARSE_FEATURES = [
    'user', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
    'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
    'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'pid',
]
DENSE_FEATURES = ['price']


def load_and_merge(raw_dir):
    print(">>> 1. Reading CSVs...")
    df_raw = pd.read_csv(os.path.join(raw_dir, 'raw_sample.csv'))
    df_user = pd.read_csv(os.path.join(raw_dir, 'user_profile.csv'))
    df_ad = pd.read_csv(os.path.join(raw_dir, 'ad_feature.csv'))

    for df in [df_raw, df_user, df_ad]:
        df.columns = df.columns.str.strip()

    df_raw['user'] = df_raw['user'].astype(str)
    df_raw['adgroup_id'] = df_raw['adgroup_id'].astype(str)
    if 'userid' in df_user.columns:
        df_user['userid'] = df_user['userid'].astype(str)
    if 'adgroup_id' in df_ad.columns:
        df_ad['adgroup_id'] = df_ad['adgroup_id'].astype(str)

    print(">>> 2. Merging tables...")
    data = pd.merge(df_raw, df_user, left_on='user', right_on='userid', how='left')
    data = pd.merge(data, df_ad, on='adgroup_id', how='left')
    print(f"    Merged shape: {data.shape}")
    return data


def clean_and_encode(data):
    print(">>> 3. Cleaning and encoding...")

    if 'nonclk' in data.columns:
        data['clk'] = 1 - data['nonclk']
    elif 'clk' not in data.columns:
        raise ValueError("Need 'nonclk' or 'clk' column")

    for feat in SPARSE_FEATURES:
        if feat in data.columns:
            data[feat] = data[feat].astype(str).replace(
                ['nan', 'None', 'NULL', '<NA>', ''], '-1'
            )
        else:
            print(f"    Warning: '{feat}' not found, filling with -1")
            data[feat] = '-1'

    for feat in DENSE_FEATURES:
        if feat in data.columns:
            data[feat] = pd.to_numeric(data[feat], errors='coerce').fillna(0)
            data[feat] = np.log1p(data[feat])
        else:
            data[feat] = 0

    vocab_sizes = {}
    for feat in SPARSE_FEATURES:
        # Same sorted string vocabulary as sklearn LabelEncoder; avoids an
        # unnecessary SciPy dependency for categorical integer encoding.
        data[feat] = pd.factorize(data[feat], sort=True)[0]
        vocab_sizes[feat] = int(data[feat].max()) + 1

    for feat in DENSE_FEATURES:
        low = data[feat].min()
        span = data[feat].max() - low
        # sklearn treats near-constant float64 ranges as one.
        if span < 10 * np.finfo(np.float64).eps:
            span = 1.0
        scale = 1.0 / span
        data[feat] = data[feat] * scale - low * scale

    return data, vocab_sizes


def export_binary(data, output_path, seed):
    print(f">>> 4. Shuffling (seed={seed}) and exporting binary...")

    dtype_list = [('clk', 'i4'), ('price', 'f4')]
    for feat in SPARSE_FEATURES:
        dtype_list.append((feat, 'i4'))
    binary_dtype = np.dtype(dtype_list)

    num_samples = len(data)
    required_bytes = num_samples * binary_dtype.itemsize
    free_bytes = shutil.disk_usage(os.path.dirname(os.path.abspath(output_path))).free
    if free_bytes < required_bytes:
        raise OSError(f"Insufficient output space: need {required_bytes:,}, free {free_bytes:,} bytes")
    structured_arr = np.zeros(num_samples, dtype=binary_dtype)
    structured_arr['clk'] = data['clk'].values.astype(np.int32)
    structured_arr['price'] = data['price'].values.astype(np.float32)
    for feat in SPARSE_FEATURES:
        structured_arr[feat] = data[feat].values.astype(np.int32)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    structured_arr = structured_arr[indices]

    structured_arr.tofile(output_path)
    return structured_arr, binary_dtype, num_samples


def write_metadata(output_dir, num_samples, vocab_sizes, record_size):
    metadata = {
        'num_samples': num_samples,
        'record_size_bytes': record_size,
        'label_dim': 1,
        'dense_features': DENSE_FEATURES,
        'num_dense': len(DENSE_FEATURES),
        'sparse_features': SPARSE_FEATURES,
        'num_sparse': len(SPARSE_FEATURES),
        'vocab_sizes': vocab_sizes,
        'format': 'clk(int32) + price(float32) + 15 sparse(int32)',
        'sampling': 'shuffle entire merged dataset before any benchmark subset',
    }
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"    Metadata saved to: {meta_path}")
    return metadata


def verify(output_path, binary_dtype, num_samples):
    expected_size = num_samples * binary_dtype.itemsize
    actual_size = os.path.getsize(output_path)
    assert actual_size == expected_size, \
        f"Size mismatch: {actual_size} vs expected {expected_size}"

    arr = np.fromfile(output_path, dtype=binary_dtype, count=5)
    print("\n>>> Verification — first 5 samples:")
    for i in range(len(arr)):
        print(f"  Sample {i}: clk={arr[i]['clk']}, price={arr[i]['price']:.4f}, "
              f"sparse={[int(arr[i][f]) for f in SPARSE_FEATURES[:5]]}...")
    print(f"\n    File size OK: {actual_size:,} bytes = {num_samples:,} x {binary_dtype.itemsize}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess Taobao Display Ad dataset')
    parser.add_argument('--raw_dir', default='./raw', help='Directory with CSV files')
    parser.add_argument('--output_dir', default='.', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Shuffle seed')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, 'taobao_dataset_shuffled.bin')

    data = load_and_merge(args.raw_dir)
    data, vocab_sizes = clean_and_encode(data)
    structured_arr, binary_dtype, num_samples = export_binary(data, output_path, args.seed)
    write_metadata(args.output_dir, num_samples, vocab_sizes, binary_dtype.itemsize)
    verify(output_path, binary_dtype, num_samples)

    print(f"\n>>> Done! Output: {output_path}")
    print(f"    {num_samples:,} samples, {binary_dtype.itemsize} bytes each, shuffled.")


if __name__ == '__main__':
    main()
