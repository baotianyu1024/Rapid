import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
checks={}
for g in [8,64]:
 owner=ROOT/f'r{g}_symmetric';samples=ROOT/f'r{g}_samples';cache=ROOT/f'r{g}_samples_cache'
 o=json.load(open(owner/'complete.json'));s=json.load(open(samples/'complete.json'));v=json.load(open(cache/'verification.json'));m=json.load(open(cache/'result.json'))
 assert o['independent_counts_verified'] and o['all_records_inverse_verified'] and s['success'] and s['all_records_permutation_verified'] and s['independent_counts_verified'] and not s['subset'] and v['success']
 cfg=json.load(open(samples/'config.json'));assert cfg['rows']==4194304 and cfg['ranks']==g
 counts=np.array(json.load(open(samples/'storage.json'))['primary_counts']);assert np.array_equal(counts,np.array(json.load(open(owner/'refinement.json'))['after_storage']))
 assert np.all(np.array(m['mirror_counts'])<=np.array(m['mirror_budget']))
 z=np.load(cache/'counts.npz');assert np.array_equal(z['fr'][0],z['bs'][0]) and np.array_equal(z['fs'][0],z['br'][0]);assert np.all(z['fr']<=z['bs']) and np.all(z['fs']<=z['br']);assert np.array_equal(z['fr'].sum(1),z['fs'].sum(1)) and np.array_equal(z['br'].sum(1),z['bs'].sum(1))
 checks[g]=dict(owner_remap_inverse_verified=True,all_samples_conserved=True,all_real_counts_verified=True,all_cache_hits_independently_verified=True,primary_storage_unchanged_by_samples=True,mirror_budget_respected=True,cold_start_verified=True)
frozen=json.load(open(ROOT/'frozen_recipe.json'))
for p,h in frozen['sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
assert json.load(open(ROOT/'version_audit.json'))['source_hashes_match']
(ROOT/'verification.json').write_text(json.dumps(dict(success=True,checks=checks,frozen_recipe_hashes_match=True),indent=2));print('FINAL ALL CHECKS PASS')
