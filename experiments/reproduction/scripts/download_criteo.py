"""Bounded concurrent downloads with official size/SHA256 verification."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[2]
manifest=json.loads((root/'reproduction/criteo_day6_metadata.json').read_text())
def download(x):
    path=root/'data/raw/criteo_day6'/Path(x['path']).name
    def valid(p):
        if not p.exists() or p.stat().st_size != x['size']:return False
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
        return h.hexdigest()==x['lfs']['oid']
    if valid(path):return str(path.name)+' verified existing'
    temp=path.with_suffix('.download')
    for attempt in range(4):
        r=subprocess.run(['curl','-fLsS','-C','-','--connect-timeout','30',
            '--max-time','600','--speed-limit','1024','--speed-time','60',
            'https://huggingface.co/datasets/criteo/CriteoClickLogs/resolve/main/'+x['path'],
            '-o',str(temp)])
        if r.returncode==0 and valid(temp):
            os.replace(temp,path)
            return str(path.name)+' downloaded and SHA256 verified'
    raise RuntimeError('Download failed: '+str(path))
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    for result in pool.map(download,reversed(manifest)):
        print(result,flush=True)
