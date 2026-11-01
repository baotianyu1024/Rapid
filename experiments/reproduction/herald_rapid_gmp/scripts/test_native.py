"""Independent, tiny Python oracle for sample allocation and cache traffic."""
import sys
from pathlib import Path
from collections import OrderedDict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'build'))
from herald_native import Scheduler

def reference(x,caches,cap,batch_id):
    p=len(caches); b=len(x)//p; allocation=[]; loads=[0]*p
    for row in x:
        scores=[sum(c.get(int(k),False) for k in row) for c in caches]
        order=[(j+batch_id)%p for j in range(p)]
        r=max((r for r in order if loads[r]<b),key=lambda r:scores[r])
        allocation.append(r);loads[r]+=1
    keys=[set(map(int,x[np.array(allocation)==r].ravel())) for r in range(p)]
    plans=[{k for s in range(p) if s!=r for k in keys[s] if caches[r].get(k,False)} for r in range(p)]
    counts=[]
    for r,c in enumerate(caches):
        n=[0,0,0,len(plans[r]),len(keys[r])]
        for k in plans[r]:c[k]=False
        for k in sorted(keys[r]):
            if k in c:
                if not c[k]:n[1]+=1
                del c[k]
            else:
                n[0]+=1
                if len(c)==cap:
                    _,valid=c.popitem(last=False)
                    n[2]+=valid
            c[k]=True
        counts.append(n)
    return counts,allocation

for cap,universe in [(c,u) for c in [2,8,50] for u in [0,15]]:
    native=Scheduler(2,4,3,cap,universe); caches=[OrderedDict(),OrderedDict()]
    rng=np.random.default_rng(739)
    for step in range(12):
        x=rng.integers(0,15,size=(8,3),dtype=np.uint64)
        got=native.step(x,step%4,True)
        counts,assignment=reference(x,caches,cap,step%4)
        assert got['counts']==counts,(cap,step,got,counts)
        assert got['assignment']==assignment
    assert native.pending()==[sum(c.values()) for c in caches]
print('PASS: 72 batches, both upstream cache lookup representations; assignment, dependencies, miss/refresh pulls, eviction/dependency pushes, terminal state match independent oracle')
