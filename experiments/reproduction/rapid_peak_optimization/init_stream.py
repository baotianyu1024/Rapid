import argparse,pickle,json,time
import numpy as np
from persistent_replay import *
from stream_owner_state import OwnerState
p=argparse.ArgumentParser();p.add_argument('--ranks',type=int,required=True);a=p.parse_args();g=a.ranks;case=ROOT/f'r{g}_peak_soft';cache=ROOT/f'r{g}_peak_soft_cache';out=ROOT/f'stream_init_{g}';out.mkdir(exist_ok=True);start=time.perf_counter()
own=OwnerState.from_export(ROOT/f'r{g}_guard_budget_original_owners');trace=trace_from_case(case,own);gids=np.concatenate([np.load(case/f'{t}.logical_gids.npy') for t in range(26)]);mirrors=[np.sort(gids[np.load(cache/f'mirrors_{r}.npy')]) for r in range(g)];state=State(own.global_owners(),mirrors,0,g);result,arr=replay_window(state,trace);ref=np.load(cache/'counts.npz');print('REFERENCE KEYS',ref.files,flush=True)
for key,actual in [('forward_receive',arr['forward'].sum(2)),('forward_send',arr['forward'].sum(1)),('backward_receive',arr['backward'].sum(1)),('backward_send',arr['backward'].sum(2))]:
 aliases={'forward_receive':'fr','forward_send':'fs','backward_receive':'br','backward_send':'bs'}
 assert np.array_equal(actual,ref[aliases[key]]),(key,actual.shape)
with open(out/'state.pkl','wb') as f:pickle.dump(state,f,protocol=5)
(out/'verified.json').write_text(json.dumps(dict(success=True,all_offline_traffic_arrays_identical=True,seconds=time.perf_counter()-start),indent=2));print('D0 PERSISTENT STATE VERIFIED',g,flush=True)
