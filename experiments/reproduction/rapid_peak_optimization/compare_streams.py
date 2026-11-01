import argparse,json,csv
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--ranks',type=int,required=True);p.add_argument('--out',type=Path);p.add_argument('--group',action='store_true');p.add_argument('--phase',action='store_true');a=p.parse_args();g=a.ranks;control=root/f'stream{g}_{"phase_" if a.phase else "group_" if a.group else ""}control';out=a.out or root/f'incremental_comparison_{"phase_" if a.phase else "group_" if a.group else ""}{g}';out.mkdir(exist_ok=True)
if not (control/'metrics.json').exists():raise SystemExit('Control not yet available')
base={d['window']:d for d in json.load(open(control/'metrics.json'))};experiments=[]
for path in root.glob(f'stream{g}_*'):
 if ('_phase_' in path.name)!=a.phase or ('_group_' in path.name)!=a.group:continue
 if not path.is_dir() or path==control or any(x in path.name for x in ['pilot','prefix','frozen','_b001']):continue
 if (path/'metrics.json').exists() and (path/'branch.json').exists():experiments.append(path)
rows=[];summaries=[]
for path in sorted(experiments):
 manifest=json.load(open(path/'manifest.json'));data={d['window']:d for d in json.load(open(path/'metrics.json'))};common=sorted(set(base)&set(data));cum=0.;netweight=0.;netadam=0.;gross=0.;migrated=0;first_payback=None;segments=[];segment=None
 for k in common:
  b=base[k];d=data[k]
  if k<=manifest['fine_tuning_interval_windows']:
   for field in ['training_vectors','training_metadata_bytes','peak_combined_receive','phase_bottleneck']:assert b[field]==d[field],('Unexpected change before first checkpoint',path.name,k,field)
   if a.group or a.phase:
    ba=np.load(control/f'window_{k:02d}/traffic.npz');da=np.load(path/f'window_{k:02d}/traffic.npz')
    for field in ['fr','fs','br','bs']:assert np.array_equal(ba[field],da[field]),('Pre-checkpoint traffic mismatch',path.name,k,field)
    assert np.array_equal(np.load(control/f'window_{k:02d}/assignment.npy'),np.load(path/f'window_{k:02d}/assignment.npy')),('Pre-checkpoint sample assignment mismatch',path.name,k)

  assert b['rows']==d['rows'] and b['source_row_start']==d['source_row_start'] and b['source_row_end']==d['source_row_end'];assert d['placement']['old_owners_and_physical_ids_unchanged'];assert max(d['placement']['storage'])-min(d['placement']['storage'])<=1
  saved=(b['training_weight_bytes']+b['training_metadata_bytes'])-(d['training_weight_bytes']+d['training_metadata_bytes']);weights=d['migration_weight_bytes'];adam=d['migration_adam_bytes'];conservative=d.get('migration_adam_plus_routes_bytes',adam+d['migration_moved_rows']*16*(g-1));cum+=saved-conservative;netweight+=saved-weights;netadam+=saved-adam;gross+=saved;migrated+=d['migration_moved_rows']
  if d['migration_moved_rows']:
   if segment:segments.append(segment)
   segment=dict(checkpoint_before_window=k,moved_embeddings=d['migration_moved_rows'],adam_plus_routes_bytes=conservative,cumulative_net_bytes=-conservative,break_even_window=None)
  if segment:
   segment['cumulative_net_bytes']+=saved
   if segment['break_even_window'] is None and segment['cumulative_net_bytes']>=0:segment['break_even_window']=k
  if k>4 and first_payback is None and cum>0:first_payback=k
  rows.append(dict(ranks=g,method=path.name,window=k,source_row_start=d['source_row_start'],rows=d['rows'],baseline_total_vectors_per_rank_step=b['total_vectors_per_8192'],total_vectors_per_rank_step=d['total_vectors_per_8192'],baseline_mean_step_peak=b['peak_combined_receive'],mean_step_peak=d['peak_combined_receive'],rank_communication_cv=d['rank_communication_cv'],phase_bottleneck=d['phase_bottleneck'],new_embeddings=d['placement']['new_embeddings'],known_embeddings=d['placement']['known_embeddings'],primary_rows_min=min(d['placement']['storage']),primary_rows_max=max(d['placement']['storage']),moved_embeddings=d['migration_moved_rows'],migration_weights_bytes=weights,migration_adam_bytes=adam,migration_adam_plus_replicated_routes_bytes=conservative,window_gross_training_savings_bytes=saved,window_net_savings_bytes=saved-conservative,cumulative_net_savings_bytes=cum))
 if segment:segments.append(segment)
 validation=[k for k in common if k>4];bp=np.mean([base[k]['peak_combined_receive'] for k in validation]) if validation else 0;mp=np.mean([data[k]['peak_combined_receive'] for k in validation]) if validation else 0;training_base=sum(base[k]['training_weight_bytes']+base[k]['training_metadata_bytes'] for k in validation)
 summary=dict(method=path.name,tune_mode=manifest.get('tune_mode','active'),interval_windows=manifest['fine_tuning_interval_windows'],budget_fraction=manifest['fine_tuning_fraction'],complete=(path/'complete.json').exists(),common_windows=len(common),validation_windows=validation,baseline_mean_peak=bp,mean_peak=mp,peak_reduction_pct=(1-mp/bp)*100 if bp else None,worst_validation_window_peak=max((data[k]['peak_combined_receive'] for k in validation),default=None),baseline_worst_validation_window_peak=max((base[k]['peak_combined_receive'] for k in validation),default=None),gross_training_savings_bytes=gross,net_savings_weights_bytes=netweight,net_savings_adam_bytes=netadam,net_savings_adam_plus_routes_bytes=cum,net_savings_pct_of_validation_training=100*cum/training_base if training_base else None,moved_embeddings=migrated,first_cumulative_payback_window=first_payback,checkpoint_segments=segments,later_segments_include_effects_of_prior_migrations=True,segment_payback_is_not_marginal_causal_attribution=True,pre_checkpoint_control_identity_verified=True)
 summaries.append(summary)
(out/'summary.json').write_text(json.dumps(summaries,indent=2));(out/'rounds.json').write_text(json.dumps(rows,indent=2))
if rows:
 with open(out/'rounds.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for s in summaries:print(s['method'],'windows',s['common_windows'],'peak%',round(s['peak_reduction_pct'] or 0,3),'net GiB',round(s['net_savings_adam_plus_routes_bytes']/2**30,3),'moved',s['moved_embeddings'],'complete',s['complete'])
