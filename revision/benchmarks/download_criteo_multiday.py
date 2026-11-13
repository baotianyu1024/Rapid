"""Download verified CriteoClickLogs shards outside the home filesystem."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote


BASE = 'https://hf-mirror.com'
REPO = 'criteo/CriteoClickLogs'


def curl_bytes(url):
    return subprocess.check_output(['curl', '-fLsS', '--retry', '3', '--connect-timeout', '20',
                                    '--max-time', '120', url])


def digest(path):
    hash_value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            hash_value.update(chunk)
    return hash_value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dates', nargs='+', default=['2015-02-15', '2015-02-16',
                                                     '2015-02-17', '2015-02-18'])
    parser.add_argument('--files-per-day', type=int, default=4)
    args = parser.parse_args()
    if args.files_per_day < 1:
        parser.error('files-per-day must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for date in args.dates:
        url = f'{BASE}/api/datasets/{REPO}/tree/main/data/day%3D{date}'
        entries = sorted((x for x in json.loads(curl_bytes(url)) if x['type'] == 'file'),
                         key=lambda x: x['path'])
        if len(entries) < args.files_per_day:
            raise RuntimeError(f'only {len(entries)} files listed for {date}')
        indexes = [i * len(entries) // args.files_per_day for i in range(args.files_per_day)]
        for index in indexes:
            item = entries[index]
            dest_dir = args.out / date
            dest_dir.mkdir(exist_ok=True)
            dest = dest_dir / Path(item['path']).name
            size, sha = item['size'], item['lfs']['oid']
            if not (dest.exists() and dest.stat().st_size == size and digest(dest) == sha):
                free = shutil.disk_usage(dest_dir).free
                if free < size + 8 * 1024**3:
                    raise OSError(f'insufficient space before {dest}: free={free}, need={size + 8 * 1024**3}')
                temp = dest.with_suffix(dest.suffix + '.download')
                file_url = f'{BASE}/datasets/{REPO}/resolve/main/{quote(item["path"], safe="/=")}'
                subprocess.run(['curl', '-fLsS', '--retry', '4', '--retry-delay', '3',
                                '--connect-timeout', '30', '--max-time', '1800',
                                file_url, '-o', str(temp)], check=True)
                if temp.stat().st_size != size or digest(temp) != sha:
                    raise ValueError(f'official size/SHA mismatch: {temp}')
                temp.replace(dest)
            manifest.append(dict(date=date, path=item['path'], local=str(dest),
                                 bytes=size, sha256=sha))
            print(f'verified {date} {dest.name} {size}', flush=True)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
