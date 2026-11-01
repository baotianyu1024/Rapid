"""Remove only explicitly listed regenerable datasets from the verified 64-rank run."""
from common import *
CASE=ROOT/'results/rank64_balance'
assert json.loads((CASE/'verification.json').read_text())['success']
assert (CASE/'comparison.json').exists() and (CASE/'balance.png').exists()
paths=[CASE/'data/input.bin',CASE/'data/sparse.npy']
for folder in ['gmp/scheduled','rapid_schedule/scheduled','rapid_owner_base/scheduled','rapid_guard_p5/scheduled','latest_rapid/rapid_schedule/scheduled','latest_rapid/rapid_owner_base/scheduled','latest_rapid/rapid_guard_p5/scheduled']:
 paths+=list((CASE/folder).glob('dataset_*.bin'))
for folder in ['rapid_profile_cache','latest_rapid/profile_cache']:
 for pattern in ['*.npz','*.profiles.npy','*.weights.npy','*.done']:
  paths+=list((CASE/folder).glob(pattern))
removed=[]
for p in paths:
 assert p.is_relative_to(CASE) and not p.is_symlink()
 if p.is_file():removed.append(dict(path=str(p),bytes=p.stat().st_size));p.unlink()
save(CASE/'cleanup.json',dict(removed=removed,bytes=sum(x['bytes'] for x in removed),original_sources_preserved=True,plans_and_results_preserved=True))
print('Removed derived bytes:',sum(x['bytes'] for x in removed))
