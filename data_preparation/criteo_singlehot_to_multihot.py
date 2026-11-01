"""
Convert a Criteo day_X TSV file to the MLPerf DLRM v2 multi-hot binary format.

The parsing and feature transforms intentionally mirror the facebookresearch/dlrm
data path:
  1. torchrec BinaryCriteoUtils.tsv_to_npys-style TSV parsing
     - label: decimal int32
     - dense: empty string -> 0, otherwise decimal integer
     - sparse: empty string -> 0, otherwise hexadecimal integer modulo table size
     - dense transform: dense_np -= dense_np.min() - 2, then np.log(..., float32)
  2. torchrec_dlrm multi_hot.py-style one-hot to multi-hot conversion
     - np.random.seed(0) is set once before building all lookup tables
     - the first lookup column is the original ID
     - remaining lookup columns are np.random.randint synthetic IDs

The only intentional differences from facebookresearch's materialization scripts are:
  - input is a single TSV/TSV.GZ file such as day_6.gz
  - output is one pure binary file with records laid out as:
    label(int32) + 13 dense(float32) + 214 sparse(int32) = 912 bytes/sample
"""
import argparse
import gzip
import os
import time
from contextlib import contextmanager

import numpy as np


MULTI_HOT_SIZES = [
    3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1,
    6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1,
]

NUM_EMBEDDINGS = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
    40000000, 3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14,
    40000000, 40000000, 40000000, 590152, 12973, 108, 36,
]

NUM_LABELS = 1
NUM_DENSE = 13
NUM_SPARSE_TABLES = 26
NUM_TSV_COLUMNS = NUM_LABELS + NUM_DENSE + NUM_SPARSE_TABLES
TOTAL_MULTI_HOT = sum(MULTI_HOT_SIZES)
RECORD_SIZE_BYTES = 4 + NUM_DENSE * 4 + TOTAL_MULTI_HOT * 4

RECORD_DTYPE = np.dtype([
    ("label", "i4"),
    ("dense", "f4", (NUM_DENSE,)),
    ("sparse", "i4", (TOTAL_MULTI_HOT,)),
])


@contextmanager
def open_text_input(input_path):
    if input_path.endswith(".gz"):
        with gzip.open(input_path, "rt", errors="ignore") as input_file:
            yield input_file
        return

    with open(input_path, "r") as input_file:
        yield input_file


def read_line_chunk(input_file, max_rows, processed_rows, chunk_size):
    if max_rows is not None and processed_rows >= max_rows:
        return []

    remaining_rows = None
    if max_rows is not None:
        remaining_rows = max_rows - processed_rows

    lines = []
    target_chunk_size = chunk_size if remaining_rows is None else min(chunk_size, remaining_rows)

    for _ in range(target_chunk_size):
        try:
            line = input_file.readline()
        except (EOFError, OSError) as exc:
            raise IOError("Truncated or corrupt input; refusing partial dataset") from exc
        if not line:
            break
        # Validate line has 40 columns (Criteo TSV format)
        if line.rstrip("\n").count("\t") != 39:
            raise ValueError("Malformed Criteo record: expected 40 tab-separated columns")
        lines.append(line)

    return lines


def parse_dense_value(raw_value):
    if raw_value == "":
        return 0
    return int(raw_value)


def parse_sparse_value(raw_value, table_idx):
    if raw_value == "":
        return 0
    return int(raw_value, 16) % NUM_EMBEDDINGS[table_idx]


def parse_chunk(lines, first_row_idx):
    labels = np.zeros(len(lines), dtype=np.int32)
    dense = np.zeros((len(lines), NUM_DENSE), dtype=np.int32)
    sparse = np.zeros((len(lines), NUM_SPARSE_TABLES), dtype=np.int32)

    for row_idx, line in enumerate(lines):
        columns = line.rstrip("\n").split("\t")
        if len(columns) != NUM_TSV_COLUMNS:
            absolute_row_idx = first_row_idx + row_idx
            raise ValueError(
                f"Expected {NUM_TSV_COLUMNS} columns, got {len(columns)} "
                f"at row {absolute_row_idx}"
            )

        labels[row_idx] = int(columns[0])

        for dense_idx in range(NUM_DENSE):
            dense[row_idx, dense_idx] = parse_dense_value(columns[1 + dense_idx])

        sparse_start_idx = NUM_LABELS + NUM_DENSE
        for table_idx in range(NUM_SPARSE_TABLES):
            sparse[row_idx, table_idx] = parse_sparse_value(
                columns[sparse_start_idx + table_idx],
                table_idx,
            )

    return labels, dense, sparse


def scan_dense_min(input_path, max_rows, chunk_size):
    processed_rows = 0
    dense_min = None
    start_time = time.time()

    with open_text_input(input_path) as input_file:
        while True:
            lines = read_line_chunk(input_file, max_rows, processed_rows, chunk_size)
            if not lines:
                break

            _, dense, _ = parse_chunk(lines, processed_rows)
            chunk_dense_min = dense.min()
            dense_min = chunk_dense_min if dense_min is None else min(dense_min, chunk_dense_min)

            processed_rows += len(lines)
            elapsed = time.time() - start_time
            speed = processed_rows / elapsed if elapsed > 0 else 0
            print(
                f"  Scanned {processed_rows} rows for dense min... "
                f"({speed:.0f} rows/s)",
                end="\r",
            )

    if dense_min is None:
        raise ValueError("Input file contains no rows")

    print(f"\nDense min: {dense_min}")
    return dense_min, processed_rows


def build_multi_hot_tables(multi_hot_sizes, num_embeddings_per_feature):
    np.random.seed(0)
    multi_hot_tables = []

    for table_idx, (num_embeddings, multi_hot_size) in enumerate(
        zip(num_embeddings_per_feature, multi_hot_sizes)
    ):
        embedding_ids = np.arange(num_embeddings)[:, np.newaxis]
        synthetic_sparse_ids = np.random.randint(
            0,
            num_embeddings,
            size=(num_embeddings, multi_hot_size - 1),
        )
        multi_hot_table = np.concatenate(
            (embedding_ids, synthetic_sparse_ids),
            axis=-1,
        ).astype(np.int32, copy=False)
        multi_hot_tables.append(multi_hot_table)

        print(
            f"  Table {table_idx}: embs={num_embeddings}, "
            f"multi_hot={multi_hot_size}, lookup shape={multi_hot_table.shape}"
        )

    return multi_hot_tables


def build_sparse_offsets():
    sparse_offsets = []
    sparse_offset = 0

    for multi_hot_size in MULTI_HOT_SIZES:
        sparse_offsets.append((sparse_offset, sparse_offset + multi_hot_size))
        sparse_offset += multi_hot_size

    return sparse_offsets


def transform_dense(dense, dense_min):
    dense -= dense_min - 2
    return np.log(dense, dtype=np.float32)


def expand_sparse_to_multi_hot(sparse, lookup_tables, sparse_offsets):
    sparse_out = np.zeros((sparse.shape[0], TOTAL_MULTI_HOT), dtype=np.int32)

    for table_idx in range(NUM_SPARSE_TABLES):
        sparse_start, sparse_end = sparse_offsets[table_idx]
        sparse_ids = sparse[:, table_idx]
        sparse_out[:, sparse_start:sparse_end] = lookup_tables[table_idx][sparse_ids]

    return sparse_out


def write_binary_records(output_file, labels, dense, sparse):
    records = np.zeros(labels.shape[0], dtype=RECORD_DTYPE)
    records["label"] = labels
    records["dense"] = dense
    records["sparse"] = sparse
    records.tofile(output_file)


def shuffle_binary(file_path, num_samples, seed=42):
    """Shuffle binary file by generating a permutation and writing to a temp file, then replacing."""
    import mmap

    print(f"\nShuffling {num_samples:,} samples (seed={seed})...")
    shuffle_start = time.time()

    # Generate random permutation
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_samples)

    # mmap source, write shuffled to temp file
    tmp_path = file_path + ".shuffle_tmp"
    file_size = num_samples * RECORD_SIZE_BYTES

    with open(file_path, "rb") as f_in:
        mm = mmap.mmap(f_in.fileno(), file_size, access=mmap.ACCESS_READ)

        with open(tmp_path, "wb", buffering=1 << 24) as f_out:
            for i, idx in enumerate(perm):
                offset = int(idx) * RECORD_SIZE_BYTES
                f_out.write(mm[offset:offset + RECORD_SIZE_BYTES])
                if (i + 1) % 5000000 == 0:
                    elapsed = time.time() - shuffle_start
                    print(f"  Written {i + 1:,} / {num_samples:,} ({elapsed:.0f}s)")

        mm.close()

    # Replace original with shuffled
    os.replace(tmp_path, file_path)

    elapsed = time.time() - shuffle_start
    print(f"Shuffle done in {elapsed:.1f}s")


def convert(input_path, output_path, max_rows=None, chunk_size=100000, shuffle=True, shuffle_seed=42):
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Max rows: {max_rows or 'all'}")
    print(f"Shuffle: {shuffle} (seed={shuffle_seed})")
    print(f"Multi-hot sizes: {MULTI_HOT_SIZES} (total={TOTAL_MULTI_HOT})")
    print(f"Record size: {RECORD_SIZE_BYTES} bytes")
    assert RECORD_SIZE_BYTES == 912

    print("\nPass 1/2: scanning dense global minimum...")
    dense_min, scanned_rows = scan_dense_min(input_path, max_rows, chunk_size)
    print(f"Rows to convert: {scanned_rows}")

    print("\nBuilding facebookresearch/dlrm multi-hot lookup tables...")
    lookup_start_time = time.time()
    lookup_tables = build_multi_hot_tables(MULTI_HOT_SIZES, NUM_EMBEDDINGS)
    print(f"Lookup tables built in {time.time() - lookup_start_time:.1f}s")

    sparse_offsets = build_sparse_offsets()

    print("\nPass 2/2: writing binary multi-hot records...")
    converted_rows = 0
    convert_start_time = time.time()

    with open_text_input(input_path) as input_file, open(output_path, "wb") as output_file:
        while True:
            lines = read_line_chunk(input_file, max_rows, converted_rows, chunk_size)
            if not lines:
                break

            labels, dense, sparse = parse_chunk(lines, converted_rows)
            dense_out = transform_dense(dense, dense_min)
            sparse_out = expand_sparse_to_multi_hot(sparse, lookup_tables, sparse_offsets)
            write_binary_records(output_file, labels, dense_out, sparse_out)

            converted_rows += len(lines)
            elapsed = time.time() - convert_start_time
            speed = converted_rows / elapsed if elapsed > 0 else 0
            print(
                f"  Converted {converted_rows} rows... ({speed:.0f} rows/s)",
                end="\r",
            )

    elapsed = time.time() - convert_start_time
    file_size = os.path.getsize(output_path)
    expected_size = converted_rows * RECORD_SIZE_BYTES

    print(f"\nConversion done. Total rows: {converted_rows}")
    print(f"Time: {elapsed:.1f}s ({converted_rows / elapsed:.0f} rows/s)")
    print(f"Output: {output_path} ({file_size / 1024 / 1024 / 1024:.2f} GB)")
    print(f"Expected: {expected_size} bytes, Actual: {file_size} bytes, Match: {expected_size == file_size}")

    if shuffle:
        shuffle_binary(output_path, converted_rows, seed=shuffle_seed)

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Criteo TSV/TSV.GZ to facebookresearch-compatible multi-hot binary"
    )
    parser.add_argument("--input", required=True, help="Input Criteo day_X TSV or TSV.GZ file")
    parser.add_argument("--output", required=True, help="Output binary file")
    parser.add_argument("--max_rows", type=int, default=None, help="Max rows to convert (default: all)")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Rows per processing chunk")
    parser.add_argument("--no_shuffle", action="store_true", help="Skip shuffle step")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Shuffle random seed (default: 42)")
    args = parser.parse_args()

    convert(args.input, args.output, args.max_rows, args.chunk_size,
            shuffle=not args.no_shuffle, shuffle_seed=args.shuffle_seed)