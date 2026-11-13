"""Generate smaller T-series shards with the upstream truncated-Zipf recipe."""

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


SOURCES = {
    "T180": "180table_70B_hotness80.py",
    "T200": "200table_100B_hotness20.py",
    "T510": "510table_110B_hotness5.py",
}
SOURCE_DIR = Path(__file__).resolve().parent / "upstream_fresh/HugeCTR/benchmarks/embedding_collection/dataset"
RESERVE = 8 * 1024**3


def check_space(directory, bytes_to_write):
    free = shutil.disk_usage(directory).free
    if free < bytes_to_write + RESERVE:
        raise OSError(f"insufficient disk space: free={free:,}, required={bytes_to_write + RESERVE:,}")


def source_groups(path):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "dataset_info" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("dataset_info missing from upstream generator")


def scale_vocab(size, factor, maximum):
    if size <= 10_000:
        return size
    return min(size, max(10_000, (size + factor - 1) // factor), maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SOURCES)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--vocab-factor", type=int, default=8)
    parser.add_argument("--max-vocab", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    if args.rows <= 0 or args.rows % 4096:
        parser.error("rows must be positive and divisible by the eight-GPU global batch of 4096")
    if args.vocab_factor < 1 or args.max_vocab < 10_000:
        parser.error("invalid vocabulary scale")
    if args.out.exists():
        parser.error("output directory already exists")
    if not args.out.parent.is_dir():
        parser.error("output parent must exist")

    source = SOURCE_DIR / SOURCES[args.dataset]
    groups = source_groups(source)
    lengths = [hot for count, hot, _ in groups for _ in range(count)]
    original_sizes = [size for count, _, size in groups for _ in range(count)]
    sizes = [scale_vocab(size, args.vocab_factor, args.max_vocab) for size in original_sizes]
    width = 14 + sum(lengths)
    data_bytes = args.rows * width * 4
    check_space(args.out.parent, data_bytes)
    args.out.mkdir()
    output = args.out / "input.bin"
    check_space(args.out, data_bytes)
    raw = np.memmap(output, dtype=np.int32, mode="w+", shape=(args.rows, width))
    rng = np.random.default_rng(args.seed)
    shuffle = rng.permutation(args.rows)
    raw[shuffle, 0] = rng.integers(0, 2, size=args.rows, dtype=np.int32)
    raw[shuffle, 1:14] = rng.random((args.rows, 13), dtype=np.float32).view(np.int32)

    offset = 14
    for table, (hot, size) in enumerate(zip(lengths, sizes)):
        ranks = np.arange(1, size + 1, dtype=np.float64)
        probability = np.power(ranks, -1.1)
        probability /= probability.sum()
        permuted_ids = rng.permutation(size).astype(np.int32)
        values = rng.choice(permuted_ids, size=(args.rows, hot), p=probability)
        raw[shuffle, offset:offset + hot] = values
        offset += hot
        if (table + 1) % 25 == 0:
            raw.flush()
            print(f"generated {table + 1}/{len(lengths)} tables", flush=True)
    raw.flush()
    del raw

    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024**2), b""):
            digest.update(chunk)
    manifest = {
        "dataset": args.dataset,
        "rows": args.rows,
        "dense": 13,
        "lengths": lengths,
        "original_sizes": original_sizes,
        "sizes": sizes,
        "ranks": 8,
        "local_batch": 512,
        "zipf_alpha": 1.1,
        "seed": args.seed,
        "vocab_factor": args.vocab_factor,
        "max_vocab": args.max_vocab,
        "source_generator": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_bytes": data_bytes,
        "input_sha256": digest.hexdigest(),
        "scope": "scaled-vocabulary synthetic workload, not the official full-size benchmark",
    }
    metadata = json.dumps(manifest, indent=2) + "\n"
    check_space(args.out, len(metadata.encode()))
    (args.out / "manifest.json").write_text(metadata)
    check_space(args.out, len(json.dumps(sizes).encode()) + 1)
    (args.out / "sizes.json").write_text(json.dumps(sizes) + "\n")
    print(json.dumps({k: manifest[k] for k in ("dataset", "rows", "input_bytes", "input_sha256")}), flush=True)


if __name__ == "__main__":
    main()
