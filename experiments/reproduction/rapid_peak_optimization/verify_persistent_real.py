import argparse,json,pickle
from persistent_fast import *
p=argparse.ArgumentParser();p.add_argument('--ranks',type=int,required=True);a=p.parse_args();g=a.ranks;case=ROOT/f'r{g}_peak_soft';cache=ROOT/f'r{g}_peak_soft_cache'
with open(ROOT/f'stream_init_{g}/state.pkl','rb') as f:reference=pickle.load(f)
state=State(reference.owner.copy(),reference.mirrors,0,g);result,arr=replay_case(state,case);counts=np.load(cache/'counts.npz')
for key,actual in [('fr',arr['forward'].sum(2)),('fs',arr['forward'].sum(1)),('br',arr['backward'].sum(1)),('bs',arr['backward'].sum(2))]:assert np.array_equal(actual,counts[key]),key
assert np.array_equal(state.master,reference.master)
for r,ids in enumerate(state.mirrors):assert np.array_equal(state.version[r]==state.master[ids],reference.version[r]==reference.master[ids])
(ROOT/f'stream_init_{g}/native_verified.json').write_text(json.dumps(dict(success=True,all_actual_traffic_identical=True,all_master_versions_identical=True,all_cache_validity_identical=True,replay_seconds=result['replay_seconds']),indent=2));print('REAL NATIVE ORACLE VERIFIED',g,flush=True)
