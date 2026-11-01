"""Portable results/source bundle, excluding raw datasets and large trace shards."""
from pathlib import Path
import tarfile,json,hashlib,time
R=Path(__file__).resolve().parents[1];files=set();inventory=[]
for p in R.iterdir():
 if p.is_file() and p.suffix in ['.txt','.json','.csv','.pdf','.png','.patch']:files.add(p)
for sub in ['scripts','figure_comparison','logs']:
 for p in (R/sub).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.cpp','.h','.sh','.json','.csv','.pdf','.png','.txt','.log']:files.add(p)
for p in (R/'working').rglob('*'):
 if p.is_file() and '__pycache__' not in p.parts and (p.suffix in ['.py','.cpp','.h','.sh','.md'] or '.pre_' in p.name or p.name in ['arrange_ids_to_devices','generate_local_dataset','verify_hit_rate']):files.add(p)
for p in (R/'metis_seed_day6').glob('*'):
 if p.is_file() and p.suffix in ['.txt','.json','.npz']:files.add(p)
for case in (R/'results').iterdir():
 if case.is_file() and case.suffix in ['.json','.npz']:files.add(case)
 if not case.is_dir():continue
 for p in case.iterdir():
  if p.is_file() and p.suffix in ['.json','.npz','.log'] and not p.name.endswith('_remap.npz'):files.add(p)
 cfg=case/'config.json'
 if cfg.exists():
  status='complete' if (case/'complete.json').exists() else 'interrupted' if (case/'interrupted.json').exists() else 'failed' if (case/'failed.json').exists() else 'ablation complete' if (case/'metrics.json').exists() else 'incomplete/failed'
  inventory.append(dict(case=case.name,status=status))
(R/'experiment_inventory.json').write_text(json.dumps(inventory,indent=2));files.add(R/'experiment_inventory.json')
for name in ['Rapid','update_Rapid']:
 for p in (R.parent/'code'/name).rglob('*'):
  if p.is_file() and p.suffix in ['.cpp','.h','.py','.sh','.md'] and '.git' not in p.parts and '__pycache__' not in p.parts:files.add(p)
manifest={str(p.relative_to(R.parent)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files) if p.name not in ['bundle_manifest.json']}
(R/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2));files.add(R/'bundle_manifest.json')
with tarfile.open(R/'reproduction_results_bundle.tar.gz','w:gz') as tar:
 for p in sorted(files):tar.add(p,arcname=str(p.relative_to(R.parent)),recursive=False)
print('Bundle:',len(files),'files,',(R/'reproduction_results_bundle.tar.gz').stat().st_size,'bytes')
