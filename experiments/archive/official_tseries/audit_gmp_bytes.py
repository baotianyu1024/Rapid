"""Independent table-by-table unique-gradient byte count, native assignment."""
from pathlib import Path
import argparse,json,numpy as np
O=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('dataset');a=p.parse_args();c=O/f'results/{a.dataset}_8';m=json.loads((c/'data/input.json').read_text());sp=np.load(c/'data/sparse.npy',mmap_mode='r');own=np.load(c/'gmp/owner.npy');assign=np.load(c/'gmp/assignment.npy');z=np.load(c/'gmp_bytes/het_gmp_s0.npz');g=m['ranks'];b=m['local_batch'];rows=[np.flatnonzero(assign==r) for r in range(g)];steps=max((len(x)+b-1)//b for x in rows);expected=np.zeros((steps,g),np.int64);cols=np.cumsum([0]+m['lengths'])
for k in range(steps):
 for r in range(g):
  batch=sp[rows[r][k*b:(k+1)*b]]
  for t,d in enumerate(m['dimensions']):
   ids=np.unique(batch[:,cols[t]:cols[t+1]]);expected[k,r]+=np.count_nonzero(own[ids]!=r)*int(d)*4
assert np.array_equal(z['backward_bytes'].sum(2),np.tile(expected,(5,1)))
assert np.array_equal(z['forward_bytes'][0].sum(1),expected[0]);assert z['pending_flush'].sum()==0
out=dict(success=True,independent_tablewise_unique_gradients=True,all_five_epochs_checked=True,first_forward_checked=True,pending_flush=0,total_backward_bytes=int(expected.sum()*5))
(c/'gmp_bytes/independent_audit.json').write_text(json.dumps(out,indent=2));print(a.dataset,out)
