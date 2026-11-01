"""Hash the exact processed files used by every reported experiment."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,time
root=Path(__file__).resolve().parents[2]
paths=['data/processed/criteo/day6.bin','data/processed/kaggle/multihot.bin','data/processed/taobao/taobao_4tables.bin']
def digest(rel):
 p=root/rel;before=p.stat();h=hashlib.sha256();start=time.time()
 with p.open('rb') as f:
  while data:=f.read(16*1024*1024):h.update(data)
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 r=dict(path=str(p),bytes=after.st_size,sha256=h.hexdigest(),seconds=time.time()-start)
 (root/'reproduction'/f'{p.parent.name}_processed_sha256.json').write_text(json.dumps(r,indent=2));print(r,flush=True);return r
with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(digest,paths))
(root/'reproduction/processed_inputs_sha256.json').write_text(json.dumps(results,indent=2))
