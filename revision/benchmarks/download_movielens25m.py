"""Download and verify the official MovieLens 25M ratings source."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


URL = 'https://files.grouplens.org/datasets/movielens/ml-25m.zip'
MIN_HEADROOM = 8 * 1024**3


def require_space(path, needed):
    free = shutil.disk_usage(path).free
    if free < needed + MIN_HEADROOM:
        raise OSError(f'insufficient space at {path}: free={free}, need={needed + MIN_HEADROOM}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / 'ml-25m.zip'
    if archive.exists():
        parser.error(f'archive already exists: {archive}')
    with urllib.request.urlopen(urllib.request.Request(URL, method='HEAD'), timeout=30) as response:
        expected = int(response.headers['Content-Length'])
    require_space(args.out, expected)
    temporary = archive.with_suffix('.zip.part')
    if temporary.exists():
        parser.error(f'partial download exists: {temporary}')
    digest = hashlib.sha256()
    with urllib.request.urlopen(URL, timeout=60) as source, temporary.open('wb') as target:
        while chunk := source.read(8 * 1024**2):
            target.write(chunk)
            digest.update(chunk)
    if temporary.stat().st_size != expected:
        raise ValueError('download size differs from official Content-Length')
    with zipfile.ZipFile(temporary) as zipped:
        selected = [zipped.getinfo(f'ml-25m/{name}') for name in ('ratings.csv', 'movies.csv')]
        if zipped.testzip() is not None:
            raise ValueError('MovieLens archive failed CRC verification')
        require_space(args.out, sum(item.file_size for item in selected))
        for item in selected:
            zipped.extract(item, args.out)
    temporary.replace(archive)
    manifest = dict(url=URL, archive_bytes=expected, archive_sha256=digest.hexdigest(),
                    extracted=[dict(path=item.filename, bytes=item.file_size) for item in selected])
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
