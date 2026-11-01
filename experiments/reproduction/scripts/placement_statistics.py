"""Storage-count balance from exported placement files; independent of traffic."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
root=Path(__file__).resolve().parents[1]
def count(path):
 n=0
 with path.open('rb') as f:
  while block:=f.read(8<<20):n+=block.count(b'\n')
 return n
results={}
for case in sorted((root/'results').iterdir()):
 if not (case/'complete.json').exists() or not (case/'placement').exists():continue
 cfg=json.loads((case/'config.json').read_text());paths=[case/'placement'/f'table_{r}' for r in range(cfg['ranks'])]
 with ThreadPoolExecutor(max_workers=8) as pool:counts=list(pool.map(count,paths))
 a=np.array(counts,dtype=np.int64);cap=int((a.sum()+len(a)-1)//len(a))
 results[case.name]=dict(total_ids=int(a.sum()),rank_counts=counts,mean=float(a.mean()),min=int(a.min()),max=int(a.max()),max_over_mean=float(a.max()/a.mean()),ceil_uniform_capacity=cap,all_within_ceil_capacity=bool(np.all(a<=cap)))
 print(case.name,'max/mean',results[case.name]['max_over_mean'],flush=True)
(root/'storage_balance.json').write_text(json.dumps(results,indent=2))
