from pathlib import Path
import json,numpy as np
V=Path(__file__).resolve().parents[1];cases=[V/f'results/{n}/optimized/peak_samples' for n in ['criteo_8','criteo_64','kaggle_8','taobao_8']]+[V/f'official_tseries/results/{n}/optimized/peak_samples' for n in ['t180_8','t200_8','t510_8']]+[p/'optimized/peak_samples' for p in (V/'results/sensitivity_full').iterdir() if p.is_dir()];rows=[]
for d in cases:
 cfg=json.loads((d/'config.json').read_text());g=cfg['ranks'];storage=np.zeros(g,np.int64)
 for t in range(len(cfg['table_lengths'])):
  owners=np.load(d/f'{t}.owners.npy',mmap_mode='r');ids=np.load(d/f'profile_cache/{t}.ids.npy',mmap_mode='r');assert np.array_equal(owners,ids%g);storage+=np.bincount(owners,minlength=g)
 stated=json.loads((d/'storage.json').read_text())['primary_counts'];assert storage.tolist()==stated;assert storage.max()-storage.min()<=1
 rows.append(dict(case=str(d),actual_final_owner_histogram=storage.tolist(),maximum_over_mean=float(storage.max()/storage.mean()),success=True))
(V/'storage_audit.json').write_text(json.dumps(dict(success=True,cases=rows),indent=2)+'\n');print('storage verified',len(rows))
