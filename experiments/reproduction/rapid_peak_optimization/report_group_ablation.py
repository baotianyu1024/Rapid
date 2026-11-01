"""Matched persistent-cache ablation of the new within-batch group correction."""
import json,csv
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent;old=root/'stream64_control';new=root/'stream64_group_control';a={d['window']:d for d in json.load(open(old/'metrics.json'))};b={d['window']:d for d in json.load(open(new/'metrics.json'))};rows=[]
for k in sorted(set(a)&set(b)):
 x=a[k];y=b[k];assert x['source_row_start']==y['source_row_start'] and x['rows']==y['rows'] and not x['migration_moved_rows'] and not y['migration_moved_rows']
 for t in range(26):
  for field in ['original_ids','owners','logical_gids']:
   assert np.array_equal(np.load(old/f'window_{k:02d}/{t}.{field}.npy'),np.load(new/f'window_{k:02d}/{t}.{field}.npy')),(k,t,field)
 rows.append(dict(window=k,primary_owners_and_logical_ids_identical=True,before_mean_vectors=x['total_vectors_per_8192'],after_mean_vectors=y['total_vectors_per_8192'],before_peak_vectors=x['peak_combined_receive'],after_peak_vectors=y['peak_combined_receive'],peak_reduction_percent=100*(1-y['peak_combined_receive']/x['peak_combined_receive']),before_cv_percent=x['rank_communication_cv']*100,after_cv_percent=y['rank_communication_cv']*100))
out=root/'final_incremental_report';out.mkdir(exist_ok=True);(out/'group_ablation.json').write_text(json.dumps(rows,indent=2))
with open(out/'group_ablation.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Verified identical primary ownership in',len(rows),'matched windows')
for r in rows:print(r['window'],round(r['before_peak_vectors'],2),'->',round(r['after_peak_vectors'],2))
