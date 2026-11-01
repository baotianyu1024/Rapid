"""Independent strict-s0 remote-ID oracle, reusable by final validation."""
from common import *
import argparse,hashlib,numpy as np
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,default=ROOT/'results/rank64_balance');a=p.parse_args();O=a.case.resolve();P=64;B=8192
s=np.load(O/'data/sparse.npy',mmap_mode='r');assignment=np.load(O/'gmp/assignment.npy');owner=np.load(O/'gmp/owner.npy');assert len(s)==len(assignment)
assert np.all((assignment>=0)&(assignment<P)) and np.all((owner>=0)&(owner<P))
rows=[np.flatnonzero(assignment==r) for r in range(P)];steps=max((len(v)+B-1)//B for v in rows);m=np.zeros((steps,P,P),np.int64)
for r in range(P):
 for b in range(steps):
  ids=np.unique(s[rows[r][b*B:(b+1)*B]].ravel());m[b,r]=np.bincount(owner[ids],minlength=P);m[b,r,r]=0
 print('oracle rank',r,flush=True)
np.savez_compressed(O/'gmp_remote_oracle.npz',remote=m)
save(O/'gmp_remote_oracle.json',dict(sparse_sha256=hashlib.sha256(memoryview(s)).hexdigest(),assignment_sha256=hashlib.sha256(memoryview(assignment)).hexdigest(),owner_sha256=hashlib.sha256(memoryview(owner)).hexdigest(),rows=len(s),rank_rows=list(map(len,rows)),steps=steps))
full=np.arange(steps)<min(map(len,rows))//B
for label,x in [('receive',m.sum(2)),('send',m.sum(1))]:
 z=x[full];print(label,'mean',z.mean(),'mean_step_peak/mean',np.mean(z.max(1)/z.mean(1)),'worst',np.max(z.max(1)/z.mean(1)),flush=True)
