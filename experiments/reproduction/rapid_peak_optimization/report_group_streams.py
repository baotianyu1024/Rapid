"""Create an auditable result bundle from completed, corrected experiments."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent;out=root/'final_incremental_report';out.mkdir(exist_ok=True)
allrows=[];policies=[];ledgers=[]
for g in [8,64]:
 control=root/f'stream{g}_group_control';base=json.load(open(control/'metrics.json'));assert (control/'complete.json').exists()
 for suffix in ['control','i4_f001']+(['i8_f001'] if g==64 else []):
  exp=root/f'stream{g}_group_{suffix}';data=json.load(open(exp/'metrics.json'));manifest=json.load(open(exp/'manifest.json'));assert (exp/'complete.json').exists() and len(data)==len(base)==manifest['windows']
  cumulative=0;primary_cv=[]
  for b,d in zip(base,data):
   traffic=np.load(exp/f"window_{d['window']:02d}/traffic.npz");step_load=traffic['fr']+traffic['br'];pl=d['placement'];assert d['group_exact_counts_independently_verified'] and pl['old_owners_and_physical_ids_unchanged'];assert max(pl['storage'])-min(pl['storage'])<=1;assert d['source_row_start']==b['source_row_start'] and d['source_row_end']==b['source_row_end'];assert np.all(np.array(d['mirror_rows'])<=manifest['mirror_capacity_fixed'])
   training_saved=b['training_weight_bytes']+b['training_metadata_bytes']-d['training_weight_bytes']-d['training_metadata_bytes'];net=training_saved-d['migration_adam_plus_routes_bytes'];cumulative+=net
   row=dict(ranks=g,policy=suffix,window=d['window'],new_window_rows=d['rows'],training_steps=d['steps'],mean_rank_forward_backward_vectors=d['total_vectors_per_8192'],mean_step_max_rank_forward_backward_vectors=d['peak_combined_receive'],mean_step_max_rank_forward_backward_send_vectors=d['peak_combined_send'],phase_bottleneck_vectors=d['phase_bottleneck'],rank_communication_cv_percent=d['rank_communication_cv']*100,mean_step_rank_cv_percent=float(np.mean(step_load.std(1)/step_load.mean(1))*100),mean_step_rank_max_over_mean=float(np.mean(step_load.max(1)/step_load.mean(1))),absolute_step_rank_peak_vectors=int(step_load.max()),primary_rows_min=min(pl['storage']),primary_rows_max=max(pl['storage']),allocated_slot_max_over_mean=pl['storage_accounting']['allocated_slot_max_over_mean'],allocated_plus_active_mirrors_max_over_mean=float(np.max(np.array(pl['storage_accounting']['allocated_physical_slots'])+d['mirror_rows'])/np.mean(np.array(pl['storage_accounting']['allocated_physical_slots'])+d['mirror_rows'])),active_mirror_rows_min=min(d['mirror_rows']),active_mirror_rows_max=max(d['mirror_rows']),known_embedding_rows=pl['known_embeddings'],new_embedding_rows=pl['new_embeddings'],new_hot_embedding_rows=pl['new_hot_embeddings'],old_id_remote_queries_per_epoch=d['raw_old_id_remote_queries_per_epoch'],new_id_remote_queries_per_epoch=d['raw_new_id_remote_queries_per_epoch'],moved_rows=d['migration_moved_rows'],migration_weight_MiB=d['migration_weight_bytes']/2**20,migration_Adam_MiB=d['migration_adam_bytes']/2**20,migration_Adam_routes_MiB=d['migration_adam_plus_routes_bytes']/2**20,training_saved_GiB=training_saved/2**30,net_saved_GiB=net/2**30,cumulative_net_saved_GiB=cumulative/2**30,planning_and_replay_seconds=d['seconds'])
   allrows.append(row)
   if d['migration_moved_rows']:
    ledger=json.load(open(exp/f"migration_before_{d['window']:02d}/migration.json"));assert ledger['future_window_inspected']==False and ledger['moved_embeddings']<=ledger['max_moved'];assert sum(ledger['send_rows'])==d['migration_moved_rows'];ledgers.append(dict(ranks=g,policy=suffix,before_window=d['window'],moved_rows=d['migration_moved_rows'],known_rows_before_migration=pl['old_embeddings'],actual_moved_percent=100*d['migration_moved_rows']/pl['old_embeddings'],Adam_routes_MiB=d['migration_adam_plus_routes_bytes']/2**20,peak_rank_send_Adam_MiB=max(ledger['send_rows'])*1536/2**20,migration_planning_seconds=ledger['seconds']))
  selected=[x for x in allrows if x['ranks']==g and x['policy']==suffix];validation=[x for x in selected if x['window']>4];ref=base[4:]
  policies.append(dict(ranks=g,policy=suffix,windows=len(data),interval_training_steps=manifest['fine_tuning_interval_windows']*data[0]['steps'],mean_peak_after_window4=float(np.mean([x['mean_step_max_rank_forward_backward_vectors'] for x in validation])),peak_reduction_after_window4_percent=100*(1-np.mean([x['mean_step_max_rank_forward_backward_vectors'] for x in validation])/np.mean([x['peak_combined_receive'] for x in ref])),mean_total_after_window4=float(np.mean([x['mean_rank_forward_backward_vectors'] for x in validation])),total_reduction_after_window4_percent=100*(1-np.mean([x['mean_rank_forward_backward_vectors'] for x in validation])/np.mean([x['total_vectors_per_8192'] for x in ref])),mean_phase_after_window4=float(np.mean([x['phase_bottleneck_vectors'] for x in validation])),max_window_cv_percent=max(x['rank_communication_cv_percent'] for x in selected),last_peak=data[-1]['peak_combined_receive'],last_mean=data[-1]['total_vectors_per_8192'],last_cv_percent=data[-1]['rank_communication_cv']*100,total_moved_rows=sum(x['moved_rows'] for x in selected),total_Adam_routes_MiB=sum(x['migration_Adam_routes_MiB'] for x in selected),net_saved_GiB=cumulative/2**30,raw_validation=(exp/f"window_{len(data):02d}/raw_verification.json").exists()))
for name,rows in [('rounds',allrows),('policies',policies),('migration',ledgers)]:
 (out/f'{name}.json').write_text(json.dumps(rows,indent=2))
 with open(out/f'{name}.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for suffix,label in [('control','No old-owner migration'),('i4_f001','Swap <=0.1%, every 80 steps'),('i8_f001','Swap <=0.1%, every 160 steps')]:
 rows=[x for x in allrows if x['ranks']==64 and x['policy']==suffix];x=[r['window'] for r in rows]
 for ax,key,scale in [(axes[0,0],'mean_step_max_rank_forward_backward_vectors',1e-3),(axes[0,1],'mean_rank_forward_backward_vectors',1e-3),(axes[1,0],'rank_communication_cv_percent',1),(axes[1,1],'cumulative_net_saved_GiB',1)]:ax.plot(x,[r[key]*scale for r in rows],marker='.',label=label)
for ax,title,ylabel in zip(axes.flat,['64 ranks: mean of step-wise peak','64 ranks: mean communication per rank','64 ranks: communication imbalance','64 ranks: net savings vs no migration'],['Thousand vectors (forward + backward)','Thousand vectors (forward + backward)','CV (%)','GiB, after Adam + replicated routes']):ax.set(title=title,xlabel='Incremental window',ylabel=ylabel);ax.grid(alpha=.25);ax.set_xticks(range(1,21,2))
axes[0,0].legend(fontsize=8);fig.savefig(out/'incremental_64.png',dpi=180);fig.savefig(out/'incremental_64.pdf');plt.close(fig)
notes='''Scope: CPU exact communication replay, not a multi-GPU throughput measurement.
Initial day6 prefix: 4,194,304 rows. Every new window: next 2,097,152 original rows, five epochs.
The 8-rank experiment has 10 windows; the 64-rank experiment has 20 windows.
All sample moves preserve original global batches and 8192 rows/rank/step.
Ordinary increments freeze old primary owners and physical IDs. Only checkpoints may swap old rows.
Both swap endpoints count toward the 0.1% budget. Decisions read the preceding window only.
Primary storage differs by at most one live row per rank. Mirror capacities remain at the original GMP caps; active mirror counts need not be identical.
Strict-s0 mirror state persists across windows. Cache fills and invalidation losses enter actual training traffic.
Peak: average over steps of max_rank(forward_receive + backward_receive), not sum of separate phase maxima.
Volume: vectors; each vector is 128 FP32 = 512 bytes. Total normalized mean is per rank per step (8192 samples).
Modeled migration costs: weights 512 B/row; Adam weights+m+v 1536 B/row; conservative replicated routing adds 16*(ranks-1) B/row.
Migration transport is not executed. GPU transfer time, optimizer compute, input-data redistribution/I/O, and routing implementation runtime are not measured. Communication tables cover embedding/gradient payloads and modeled protocol metadata.
Net savings include training protocol metadata and subtract Adam plus conservative routing migration bytes.
Policy gains after later checkpoints include effects of all prior checkpoints; they are not marginal causal gains of only the latest swap.
First four owner plans reuse deterministic plans from the earlier experiment; preprocessing wall time is not a fair end-to-end speed benchmark.
The group-exchange correction and mirror extension are optimized RAPID extensions, not claims about unmodified published RAPID.
'''
(out/'methodology.txt').write_text(notes)
print(json.dumps(policies,indent=2))
