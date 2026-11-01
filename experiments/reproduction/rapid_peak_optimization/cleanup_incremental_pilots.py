"""Remove regenerable engineering-pilot caches only; preserve original data and results."""
import json,shutil,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
check=subprocess.run(['pgrep','-af',r'^/home/longjiu.bty/miniconda3/envs/het-gmp-rapid/bin/python .*pilot_alternate.py'],capture_output=True,text=True)
assert check.returncode==1,check.stdout
folders=[p for p in root.glob('stream*_pilot*') if p.is_dir()]+[root/n for n in ['stream8_frozen','stream8_i4_b001','stream64_frozen','stream64_i4_b001']]
removed=[]
for base in folders:
 assert base.parent==root and 'final_prefix' not in base.name and 'control' not in base.name
 for directory in list(base.rglob('profile_cache'))+list(base.rglob('mirrors')):
  if not directory.exists():continue
  selection=directory/'selection.json'
  if selection.exists():shutil.copy2(selection,directory.parent/'mirror_selection_archived.json')
  size=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file());removed.append(dict(path=str(directory),logical_bytes=size));shutil.rmtree(directory)
 for file in base.rglob('checkpoint.pkl'):
  removed.append(dict(path=str(file),logical_bytes=file.stat().st_size));file.unlink()
(root/'pilot_cleanup.json').write_text(json.dumps(dict(removed=removed,logical_bytes_removed=sum(x['logical_bytes'] for x in removed),original_inputs_untouched=True,numerical_metrics_and_traffic_arrays_preserved=True,selected_offline_and_final_incremental_cases_untouched=True),indent=2));print('Removed pilot cache logical GiB',sum(x['logical_bytes'] for x in removed)/2**30)
