from pathlib import Path
from collections import OrderedDict
import sys,json,numpy as np
O=Path(__file__).resolve().parents[1];sys.path.insert(0,str(O/'herald/build'));from herald_native import Scheduler
p,b,cols,cap,n=2,4,3,4,16;weights=np.array([128,256,512,1024]*4);state=Scheduler(p,b,cols,cap,n);state.set_weights(weights.tolist());cache=[OrderedDict() for _ in range(p)];rng=np.random.default_rng(41)
for step in range(12):
 x=rng.integers(n,size=(p*b,cols),dtype=np.uint64);actual=state.step(x,step,True);assign=np.array(actual['assignment']);keys=[set(map(int,x[assign==r].ravel())) for r in range(p)];expect=np.zeros((p,5),np.uint64);byte=np.zeros_like(expect)
 plans=[set(k for other in range(p) if other!=r for k in keys[other] if cache[r].get(k,False)) for r in range(p)]
 for r in range(p):
  expect[r,3]=len(plans[r]);byte[r,3]=sum(weights[k] for k in plans[r]);expect[r,4]=len(keys[r]);byte[r,4]=sum(weights[k] for k in keys[r])
  for k in plans[r]:cache[r][k]=False
  for k in sorted(keys[r]):
   if k in cache[r]:
    if not cache[r][k]:expect[r,1]+=1;byte[r,1]+=weights[k]
    del cache[r][k]
   else:
    expect[r,0]+=1;byte[r,0]+=weights[k]
    if len(cache[r])==cap:
     ev,valid=cache[r].popitem(last=False)
     if valid:expect[r,2]+=1;byte[r,2]+=weights[ev]
   cache[r][k]=True
 assert np.array_equal(actual['counts'],expect),(step,actual['counts'],expect)
 assert np.array_equal(actual['bytes'],byte),(step,actual['bytes'],byte)
assert state.pending_bytes()==[sum(int(weights[k]) for k,v in c.items() if v) for c in cache]
(O/'validation/herald_mixed_audit.json').write_text(json.dumps(dict(success=True,steps=12,independent_ordered_dict_lru=True,all_four_transfer_types_and_terminal_checked=True),indent=2));print('PASS mixed-dimension Herald independent LRU')
