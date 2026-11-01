"""Download fixed HTTP ranges; validate Content-Range before assembling."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import re
import subprocess
import time
import argparse

root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser()
p.add_argument('--url',default='https://ndownloader.figshare.com/files/10082655')
p.add_argument('--size',type=int,default=4576820670)
p.add_argument('--chunk-mb',type=int,default=32)
p.add_argument('--parts',default=str(root/'data/raw/kaggle_parts'))
p.add_argument('--output',default=str(root/'data/raw/kaggle_dac.assembled.tar.gz'))
a=p.parse_args()
parts=Path(a.parts);parts.mkdir(parents=True,exist_ok=True)
size=a.size
chunk=a.chunk_mb<<20
url=a.url
def get(start):
    end=min(size,start+chunk)-1
    dest=parts/f'{start:012d}.part';header=dest.with_suffix('.headers')
    if dest.exists() and dest.stat().st_size==end-start+1 and header.exists():
        text=header.read_text(errors='replace').lower()
        if f'content-range: bytes {start}-{end}/{size}' in text:return dest
    temp=dest.with_suffix('.downloading')
    for attempt in range(6):
        result=subprocess.run(['curl','-fLsS','--range',f'{start}-{end}',
            '--connect-timeout','30','--max-time','300','--speed-limit','1024','--speed-time','45',
            '-D',str(header),url,'-o',str(temp)])
        if (result.returncode==0 and temp.stat().st_size==end-start+1 and
            f'content-range: bytes {start}-{end}/{size}' in header.read_text(errors='replace').lower()):
            os.replace(temp,dest);print('verified range',start,end,flush=True);return dest
        time.sleep(min(5,attempt+1))
    raise RuntimeError(f'failed range {start}-{end}')
with ThreadPoolExecutor(max_workers=6) as pool:
    files=list(pool.map(get,range(0,size,chunk)))
target=Path(a.output)
with target.open('wb') as out:
    for p in files:
        with p.open('rb') as src:
            for block in iter(lambda:src.read(4<<20),b''):out.write(block)
assert target.stat().st_size==size
print('assembled',target,flush=True)
