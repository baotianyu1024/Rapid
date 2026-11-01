"""GMP ablation: minimal row-count repair to preserve original global batches.
Unmodified saved GMP owners/mirrors; no RAPID optimization or cache reselection.
"""
import argparse,json,sys,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;R=ROOT.parent
p=argparse.ArgumentParser();p.add_argument('--ranks',type=int,choices=[8,64],required=True);p.add_argument('--source',type=Path);p.add_argument('--data',type=Path);p.add_argument('--out',type=Path);a=p.parse_args();g=a.ranks;b=8192
src=R/'herald_rapid_gmp/results/default8/fresh_baselines/het' if g==8 else R/'het_gmp_rapid/results/rank64_balance/gmp'
src=a.source.resolve() if a.source else src
if not (src/'assignment.npy').exists():
 print('Missing',src);sys.exit(1)
data=a.data.resolve() if a.data else R/'het_gmp_rapid/results/main64/data';out=a.out.resolve() if a.out else ROOT/f'gmp_fixed_batch_{g}';out.mkdir(exist_ok=False);part=out/'partition';part.mkdir()
old=np.load(src/'assignment.npy');assignment=old.copy();owner=np.load(src/'owner.npy');sparse=np.load(data/'sparse.npy',mmap_mode='r');assert len(old)==len(sparse);n=len(old);changes=[];rng=np.random.default_rng(42)
for k in range(n//(g*b)):
 lo=k*g*b;v=assignment[lo:lo+g*b];counts=np.bincount(v,minlength=g);lower_bound=int(np.maximum(counts-b,0).sum());moved=0
 for r in np.flatnonzero(counts>b):
  candidates=np.flatnonzero(v==r);rng.shuffle(candidates)
  for i in candidates[:counts[r]-b]:
   under=np.flatnonzero(counts<b);ids=np.unique(sparse[lo+i]);hits=np.bincount(owner[ids],minlength=g);dest=int(under[np.argmax(hits[under])]);v[i]=dest;counts[r]-=1;counts[dest]+=1;moved+=1
 assert np.array_equal(counts,np.full(g,b)) and moved==lower_bound;changes.append(moved)
np.save(part/'assignment.npy',assignment)
for f in ['owner.npy']+[f'mirrors_{r}.npy' for r in range(g)]: (part/f).symlink_to((src/f).resolve())
(out/'constraint.json').write_text(json.dumps(dict(source=str(src),global_batch_membership_preserved=True,local_batch=b,changed_rows_by_batch=changes,changed_fraction=sum(changes)/n,primary_and_mirrors_unchanged=True,repair='minimum number of moved rows; deterministic source subset; underfull destination with highest unique owner hits',seed=42),indent=2));print('REPAIR',sum(changes),sum(changes)/n,flush=True)
cmd=[sys.executable,str(R/'het_gmp_rapid/scripts/replay_het.py'),'--data',str(data),'--partition',str(part),'--out',str(out/'traffic'),'--ranks',str(g),'--bounds','0','--skip-1d']
subprocess.run(cmd,check=True)
