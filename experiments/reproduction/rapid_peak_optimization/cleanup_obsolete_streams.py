"""Remove regenerable V1 caches/checkpoints only after corrected runs verify."""
import json,shutil
from pathlib import Path
root=Path(__file__).resolve().parent
assert (root/'final_incremental_report/verification_complete.json').exists(), 'Corrected streams must finish and verify first'
names=['stream8_control','stream8_i4_f0001','stream8_i4_f001','stream8_i4_f01','stream8_inactive_f001','stream8_inactive_f01','stream64_control','stream64_i4_f001','stream64_i8_f001']
names += ['stream8_group_control','stream8_group_i4_f001','stream64_group_control','stream64_group_i4_f001','stream64_group_i8_f001','stream64_group_i4_f01_pilot'] + [f'pipeline_ab_{g}_{m}' for g in [8,64] for m in ['serial','pipeline']]
removed=[];logical_bytes=0
for name in names:
 exp=root/name;assert (exp/'complete.json').exists() or (exp/'stopped_for_revision.json').exists()
 for win in exp.glob('window_*'):
  for sub in ['profile_cache','mirrors']:
   p=win/sub
   if p.is_dir():
    selection=p/'selection.json'
    if selection.exists():shutil.copy2(selection,win/'retained_mirror_selection.json')
    logical_bytes+=sum(f.stat().st_size for f in p.rglob('*') if f.is_file());shutil.rmtree(p);removed.append(str(p))
 for f in [exp/'checkpoint.pkl',exp/'checkpoint.tmp',*exp.glob('window_*/incoming_cache.pkl')]:
  if f.exists():logical_bytes+=f.stat().st_size;f.unlink();removed.append(str(f))
(root/'obsolete_stream_cleanup.json').write_text(json.dumps(dict(removed=removed,logical_bytes_removed=logical_bytes,logical_size_can_double_count_hardlinks=True,original_datasets_untouched=True,metrics_traffic_assignments_and_owner_maps_preserved=True,corrected_final_streams_untouched=True),indent=2))
print('Removed obsolete generated cache/checkpoint paths:',len(removed),'logical GiB:',logical_bytes/2**30)
