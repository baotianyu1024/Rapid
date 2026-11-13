"""Stage experiment inputs in tmpfs with a per-file capacity check and checksum."""
import argparse
import hashlib
import shutil
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def stage(source, target):
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    needed = source.stat().st_size
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if shutil.disk_usage(target.parent).free < needed + 8 * 1024**3 or available < needed + 8 * 1024**3:
        raise OSError(f'Insufficient tmpfs or RAM capacity before writing {target}')
    shutil.copyfile(source, target)
    source_hash, target_hash = sha256(source), sha256(target)
    if source_hash != target_hash:
        raise IOError(f'Checksum mismatch: {source} -> {target}')
    print(f'{target} bytes={needed} sha256={target_hash}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    stage(args.source, args.target)
