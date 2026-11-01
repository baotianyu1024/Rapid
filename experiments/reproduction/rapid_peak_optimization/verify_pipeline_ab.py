import json,pickle,time,gc
from pathlib import Path
import numpy as np
from persistent_replay import State
root=Path(__file__).resolve().parent
names=[f'pipeline_ab_{g}_{m}' for g in [8,64] for m in ['serial','pipeline']]
while not all((root/n/'complete.json').exists() for n in names):
 for n in names:
  log=root/(n+'.log')
  if log.exists() and ('Traceback' in log.read_text() or 'Aborted' in log.read_text()):raise RuntimeError('A/B trial failed: '+n)
 time.sleep(5)
for g in [8,64]:
 a=root/f'pipeline_ab_{g}_serial';b=root/f'pipeline_ab_{g}_pipeline';ma=json.load(open(a/'metrics.json'));mb=json.load(open(b/'metrics.json'));assert len(ma)==len(mb)==4
 for k in range(1,5):
  ca=a/f'window_{k:02d}';cb=b/f'window_{k:02d}';assert np.array_equal(np.load(ca/'assignment.npy'),np.load(cb/'assignment.npy'))
  za=np.load(ca/'traffic.npz');zb=np.load(cb/'traffic.npz')
  for key in za.files:assert np.array_equal(za[key],zb[key]),(g,k,key)
  for key in ['training_vectors','training_metadata_bytes','peak_combined_receive','phase_bottleneck','migration_moved_rows','migration_adam_plus_routes_bytes']:assert ma[k-1][key]==mb[k-1][key],(g,k,key)
 with open(a/'checkpoint.pkl','rb') as f:oa,sa,ka=pickle.load(f)
 with open(b/'checkpoint.pkl','rb') as f:ob,sb,kb=pickle.load(f)
 assert ka==kb==4 and oa.next_gid==ob.next_gid and np.array_equal(oa.storage,ob.storage) and np.array_equal(oa.next,ob.next) and oa.free==ob.free
 for field in ['owner','physical','gid']:
  for x,y in zip(getattr(oa,field),getattr(ob,field)):assert np.array_equal(x,y),(g,field)
 assert np.array_equal(sa.owner,sb.owner) and np.array_equal(sa.master,sb.master)
 for field in ['mirrors','version','pending']:
  for x,y in zip(getattr(sa,field),getattr(sb,field)):assert np.array_equal(x,y),(g,field)
 la=json.load(open(a/'migration_before_03/migration.json'));lb=json.load(open(b/'migration_before_03/migration.json'))
 for key in ['logical_gids','old_owners','new_owners','old_physical_ids','new_physical_ids','routing_rows']:assert la[key]==lb[key],(g,key)
 del oa,ob,sa,sb;gc.collect();print('PIPELINE EXACT A/B PASS',g,flush=True)
(root/'pipeline_ab_verified.json').write_text(json.dumps(dict(success=True,ranks=[8,64],rows_per_window=524288,windows=4,checkpoint_before_window=3,all_assignments_and_traffic_identical=True,all_final_primary_physical_logical_addresses_identical=True,all_final_cache_clocks_and_mirrors_identical=True,migration_ledger_identical=True,prefetch_does_not_cross_checkpoint=True),indent=2))
