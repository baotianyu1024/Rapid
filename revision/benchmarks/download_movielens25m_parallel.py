"""Download the official MovieLens archive in verified HTTP byte ranges."""
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


URL = 'https://files.grouplens.org/datasets/movielens/ml-25m.zip'
OUT = Path('/tmp/rapid_movielens25m_parallel_20261103')
CHUNK = 8 * 1024**2
HEADROOM = 8 * 1024**3


def require_space(needed):
    free = shutil.disk_usage(OUT).free
    if free < needed + HEADROOM:
        raise OSError(f'insufficient space at {OUT}: free={free}, need={needed + HEADROOM}')


def download_part(part, start, end):
    expected = end - start + 1
    destination = OUT / f'part_{part:03}.bin'
    if destination.exists() and destination.stat().st_size == expected:
        return destination
    for attempt in range(12):
        have = destination.stat().st_size if destination.exists() else 0
        if have == expected:
            break
        if have > expected:
            raise ValueError(f'part {part} exceeds its byte range: {have} > {expected}')
        require_space(expected - have)
        with destination.open('ab') as target:
            result = subprocess.run(
                ['curl', '-fLsS', '--connect-timeout', '15', '--max-time', '300',
                 '--range', f'{start + have}-{end}', '-o', '-', URL],
                stdout=target, stderr=subprocess.PIPE, text=False)
        if destination.stat().st_size > expected:
            raise ValueError(f'part {part} received more bytes than requested')
        if result.returncode and destination.stat().st_size == have:
            raise RuntimeError(f'part {part} stalled: {result.stderr.decode(errors="replace")}')
        print(f'part {part} attempt {attempt + 1}: {destination.stat().st_size}/{expected}', flush=True)
    if destination.stat().st_size != expected:
        raise ValueError(f'incomplete range {part}: {destination.stat().st_size}/{expected}')
    print(f'part {part} {expected}', flush=True)
    return destination


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(URL, method='HEAD'), timeout=30) as response:
        size = int(response.headers['Content-Length'])
    require_space(size * 2)
    ranges = [(i, i * CHUNK, min(size - 1, (i + 1) * CHUNK - 1))
              for i in range((size + CHUNK - 1) // CHUNK)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(download_part, *item) for item in ranges]
        parts = [future.result() for future in futures]
    archive = OUT / 'ml-25m.zip'
    require_space(size)
    digest = hashlib.sha256()
    with archive.open('wb') as target:
        for part in parts:
            with part.open('rb') as source:
                while chunk := source.read(8 * 1024**2):
                    target.write(chunk)
                    digest.update(chunk)
    if archive.stat().st_size != size:
        raise ValueError('assembled archive size mismatch')
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('archive CRC mismatch')
        files = [zipped.getinfo(f'ml-25m/{name}') for name in ('ratings.csv', 'movies.csv')]
        require_space(sum(item.file_size for item in files))
        for item in files:
            zipped.extract(item, OUT)
    manifest = dict(url=URL, archive_bytes=size, sha256=digest.hexdigest(),
                    extracted=[dict(path=item.filename, bytes=item.file_size) for item in files])
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
