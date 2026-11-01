import importlib.util,json,sys
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parents[1];V=O.parent
sys.path.insert(0,str(O/'herald/build'));from herald_native import Scheduler
p,b,cols,n=2,4,3,16
x=np.array([[0,1,2],[1,2,3],[4,5,6],[0,7,8],[9,10,11],[12,13,14],[14,15,1],[0,4,8]],np.uint64)
s=Scheduler(p,b,cols,4,n);s.set_weights([512]*n)
for i in range(3):
 v=s.step(x,i,True);assert np.array_equal(np.array(v['bytes']),np.array(v['counts'])*512)
assert np.array_equal(np.array(s.pending_bytes()),np.array(s.pending())*512)
spec=importlib.util.spec_from_file_location('weighted',O/'code/replay_het_bytes.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
own=np.arange(n)%p;mir=[np.array([1,3,5]),np.array([0,2,4])];state=mod.State(own,mir,0,p)
for _ in range(3):state.step([(np.array([0,1,3]),np.array([1,2,1])),(np.array([0,2,5]),np.array([1,1,3]))])
_,arr=state.finish();assert np.array_equal(arr['forward_bytes'],arr['forward']*512);assert np.array_equal(arr['backward_bytes'],arr['backward']*512)
state=mod.State(own,[np.array([],dtype=int)]*p,0,p);state.payload_bytes=np.arange(1,n+1)*4;state.step([(np.array([0,1,3]),np.ones(3,dtype=int)),(np.array([0,2,5]),np.ones(3,dtype=int))]);_,arr=state.finish();assert arr['forward_bytes'].sum()==8+16+4+12;assert arr['backward_bytes'].sum()==40
(O/'instrumentation_check.json').write_text(json.dumps(dict(herald_uniform_matches_counts=True,gmp_uniform_matches_counts=True,gmp_mixed_manual_bytes_each_direction=40),indent=2));print('PASS')
