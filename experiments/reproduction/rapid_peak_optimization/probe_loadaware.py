import json,pickle,os,shutil,time
from pathlib import Path
from persistent_fast import *
from stream_alternate import alternate
source=ROOT/'stream64_i4_f001';out=ROOT/'loadaware_probe64';out.mkdir(exist_ok=False)
while True:
 try:
  k=json.load(open(source/'metrics.json'))[-1]['window']
  if (source/'checkpoint.pkl').stat().st_mtime_ns<(source/'metrics.json').stat().st_mtime_ns:time.sleep(1);continue
  os.link(source/'checkpoint.pkl',out/'owner_snapshot.pkl')
  with open(out/'owner_snapshot.pkl','rb') as f:owner,_,completed=pickle.load(f)
  if completed!=k:(out/'owner_snapshot.pkl').unlink();continue
  case=out/'baseline';shutil.copytree(source/f'window_{k:02d}',case,copy_function=lambda s,d:shutil.copy2(s,d) if Path(s).suffix=='.json' else os.link(s,d));break
 except FileNotFoundError:
  if (out/'owner_snapshot.pkl').exists():(out/'owner_snapshot.pkl').unlink()
  if (out/'baseline').exists():shutil.rmtree(out/'baseline')
  time.sleep(1)
cfg=json.load(open(case/'config.json'));metrics=json.load(open(case/'metrics.json'));g=64;n=cfg['rows'];width=1+cfg['dense']+sum(cfg['table_lengths']);data=np.memmap('/mnt/hugui/longjiu.bty/data/processed/criteo/day6.bin',dtype='<u4',mode='r',offset=metrics['source_row_start']*width*4,shape=(n,width));budgets=json.load(open(ROOT/'r64_peak_soft_cache/result.json'))['mirror_budget'];results=[]
for tag in ['baseline','candidate']:
 current=case if tag=='baseline' else out/'candidate'
 if tag!='baseline':alternate(owner,case,data,current,reallocate=False,scheduler='loadaware',rounds=32)
 state=State(owner.global_owners(),[np.empty(0,np.uint32) for _ in range(g)],0,g);mirror_out=current/'cold_probe_mirrors';mirrors,saved=select_mirrors(current,state,budgets,mirror_out);set_mirrors(state,mirrors);result,arrays=replay_case(state,current,expected_saved=saved);fr=arrays['forward'].sum(2);br=arrays['backward'].sum(1);load=fr+br;d=dict(tag=tag,window=k,cache_initialization='matched cold start diagnostic, not the persistent production result',total=float(load.mean()),peak=float(load.max(1).mean()),cv=float(load.mean(0).std()/load.mean()));results.append(d);print('LOADAWARE PROBE',d,flush=True);(out/'results.json').write_text(json.dumps(results,indent=2))
