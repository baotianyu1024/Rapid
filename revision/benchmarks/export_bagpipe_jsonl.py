"""Export an already-shuffled HugeCTR raw split to BagPipe's ragged JSONL input."""
import argparse
import shutil
from pathlib import Path

import numpy as np
import orjson


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dense-count', type=int, required=True)
    parser.add_argument('--hotness', required=True)
    args = parser.parse_args()
    hotness = [int(value) for value in args.hotness.split(',')]
    width = 1 + args.dense_count + sum(hotness)
    input_bytes = args.input.stat().st_size
    if input_bytes % (width * 4):
        raise ValueError('raw input size does not match record layout')
    existing_parent = args.output_dir.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < input_bytes * 4 + 2 * 1024**3:
        raise OSError('insufficient space for BagPipe JSONL export')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = np.memmap(args.input, dtype=np.int32, mode='r').reshape(-1, width)
    floats = rows.view(np.float32).reshape(-1, width)
    boundaries = np.cumsum([0] + hotness)
    output = args.output_dir / 'part_0.txt'
    pending = output.with_suffix('.tmp')
    with pending.open('wb') as stream:
        buffer = bytearray()
        for index in range(len(rows)):
            ids = rows[index, 1 + args.dense_count:]
            sparse = [ids[boundaries[i]:boundaries[i + 1]].tolist()
                      for i in range(len(hotness))]
            record = {'label': float(rows[index, 0]),
                      'dense': floats[index, 1:1 + args.dense_count].tolist(),
                      'sparse': sparse}
            buffer.extend(orjson.dumps(record) + b'\n')
            if (index + 1) % 8192 == 0:
                if shutil.disk_usage(args.output_dir).free < len(buffer) + 1024**3:
                    raise OSError('insufficient space before writing next BagPipe chunk')
                stream.write(buffer)
                buffer.clear()
        if buffer:
            if shutil.disk_usage(args.output_dir).free < len(buffer) + 1024**3:
                raise OSError('insufficient space before writing final BagPipe chunk')
            stream.write(buffer)
    pending.replace(output)
    print(f'BAGPIPE_INPUT={output} ROWS={len(rows)} BYTES={output.stat().st_size}')


if __name__ == '__main__':
    main()
