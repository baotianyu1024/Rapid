import json,pickle,os,shutil,ctypes
from pathlib import Path
from persistent_fast import *
from stream_alternate import alternate
import group_scheduler as scheduler
source=ROOT/'stream64_group_i4_f01_pilot';out=ROOT/'group_wide_probe64';out.mkdir(exist_ok=False)
os.link(source/'checkpoint.pkl',out/'owner_snapshot.pkl')
with open(out/'owner_snapshot.pkl','rb') as f:owner,_,k=pickle.load(f)
assert k==7,k
case=out/'baseline';shutil.copytree(source/f'window_{k:02d}',case,copy_function=lambda s,d:shutil.copy2(s,d) if Path(s).suffix=='.json' else os.link(s,d));assert (case/'profile_cache').exists()
lib=ctypes.CDLL(str(ROOT/'sample_group_peak_wide.so'));fn=lib.sample_group_peak;fn.argtypes=scheduler.fn.argtypes;fn.restype=scheduler.fn.restype;scheduler.fn=fn
cfg=json.load(open(case/'config.json'));metrics=json.load(open(case/'metrics.json'));g=64;n=cfg['rows'];width=1+cfg['dense']+sum(cfg['table_lengths']);data=np.memmap('/mnt/hugui/longjiu.bty/data/processed/criteo/day6.bin',dtype='<u4',mode='r',offset=metrics['source_row_start']*width*4,shape=(n,width));budgets=json.load(open(ROOT/'r64_peak_soft_cache/result.json'))['mirror_budget'];results=[]
candidate=out/'candidate';alternate(owner,case,data,candidate,reallocate=False,scheduler='group',rounds=8)
for tag,current in [('baseline',case),('wide',candidate)]:
 state=State(owner.global_owners(),[np.empty(0,np.uint32) for _ in range(g)],0,g);mirrors,saved=select_mirrors(current,state,budgets,current/'cold_probe_mirrors');set_mirrors(state,mirrors);result,arrays=replay_case(state,current,expected_saved=saved);fr=arrays['forward'].sum(2);br=arrays['backward'].sum(1);load=fr+br
 if tag=='wide':
  counts=np.load(current/'predicted_group_counts.npz');assert np.array_equal(counts['receive'],arrays['backward'][:cfg['batches']].sum(2));assert np.array_equal(counts['send'],arrays['backward'][:cfg['batches']].sum(1))
 d=dict(tag=tag,window=k,cache_initialization='matched COLD diagnostic, not production persistent traffic',total=float(load.mean()),peak=float(load.max(1).mean()),cv=float(load.mean(0).std()/load.mean()));results.append(d);print('WIDE PROBE',d,flush=True);(out/'results.json').write_text(json.dumps(results,indent=2))
