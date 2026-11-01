import argparse,json,pickle,time,shutil,gc
import numpy as np
from stream_window import window
from stream_owner_state import OwnerState
from stream_tune import tune
from stream_tune_inactive import tune as tune_inactive
from stream_alternate import alternate
from persistent_replay import *
from persistent_fast import replay_case
p=argparse.ArgumentParser();p.add_argument('--ranks',type=int,required=True);p.add_argument('--windows',type=int,default=16);p.add_argument('--rows',type=int,default=4194304);p.add_argument('--interval',type=int,default=0);p.add_argument('--fraction',type=float,default=.001);p.add_argument('--rounds',type=int,default=4);p.add_argument('--balance',type=float,default=4);p.add_argument('--seed-schedule',default='affinity');p.add_argument('--refit-new',action='store_true');p.add_argument('--tune-mode',choices=['active','inactive'],default='active');p.add_argument('--reuse-owner-plans',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--keep-profiles',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--recover-first',action='store_true');p.add_argument('--stop-after',type=int,default=0);a=p.parse_args();g=a.ranks;a.out.mkdir(exist_ok=a.resume or a.recover_first);cfg=json.load(open(ROOT/f'r{g}_peak_soft/config.json'));source=Path('/mnt/hugui/longjiu.bty/data/processed/criteo/day6.bin');width=1+cfg['dense']+sum(cfg['table_lengths']);state=OwnerState.from_export(ROOT/f'r{g}_guard_budget_original_owners');budgets=json.load(open(ROOT/f'r{g}_peak_soft_cache/result.json'))['mirror_budget'];cache_state=None
manifest=dict(group_post_correction=True,group_rounds=8,group_total_growth_budget=0,owner_plan_reuse_source=str(a.reuse_owner_plans) if a.reuse_owner_plans else None,ranks=g,windows=a.windows,rows_per_window=a.rows,source=str(source),initial_rows=4194304,epochs=5,sample_refinement_rounds=a.rounds,balance=a.balance,seed_schedule=a.seed_schedule,refit_new=a.refit_new,tune_mode=a.tune_mode,fine_tuning_interval_windows=a.interval,fine_tuning_fraction=a.fraction,mirror_capacity_fixed=budgets,global_batch_membership_preserved=True,gpu_execution=False)
if a.resume:
 previous_manifest=json.load(open(a.out/'manifest.json'))
 for key in ['ranks','rows_per_window','source','initial_rows','epochs','sample_refinement_rounds','balance','seed_schedule','refit_new','group_post_correction']:
  assert previous_manifest[key]==manifest[key],('Resume cannot change workload/algorithm',key)
(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));allmetrics=[];last=None;total_moved=0
start_window=1
if a.resume:
 with open(a.out/'checkpoint.pkl','rb') as f:state,cache_state,completed=pickle.load(f)
 allmetrics=json.load(open(a.out/'metrics.json'));assert len(allmetrics)==completed;last=a.out/f'window_{completed:02d}';total_moved=allmetrics[-1]['cumulative_moved_rows'];start_window=completed+1
for k in range(start_window,a.windows+1):
 start=time.perf_counter();migration=None
 if a.interval and k>1 and (k-1)%a.interval==0:
  migration=(tune_inactive if a.tune_mode=="inactive" else tune)(state,last,a.fraction,a.out/f'migration_before_{k:02d}');total_moved+=migration['moved_embeddings'];inv=update_owners(cache_state,state.global_owners(),migration['logical_gids']);migration['invalidated_cached_copies_by_rank']=inv;(a.out/f'migration_before_{k:02d}/migration.json').write_text(json.dumps(migration,indent=2))
 offset=4194304+(k-1)*a.rows;case=a.out/f'window_{k:02d}'
 if a.recover_first and k==1:
  placement=json.load(open(case/'placement.json'))
  for t in range(len(state.owner)):
   ids=np.load(case/f'{t}.original_ids.npy');state._grow(t,int(ids.max())+1);state.owner[t][ids]=np.load(case/f'{t}.owners.npy');state.physical[t][ids]=np.load(case/f'profile_cache/{t}.ids.npy');state.gid[t][ids]=np.load(case/f'{t}.logical_gids.npy')
   for r in range(g):
    physical=state.physical[t][state.owner[t]==r];state.next[t,r]=int((physical//g).max())+1 if len(physical) else 0
  state.next_gid=placement['known_embeddings'];state.storage=np.array(placement['storage'],np.int64)
  if (case/'mirrors').exists():shutil.rmtree(case/'mirrors')
 else:
  data=np.memmap(source,dtype='<u4',mode='r',offset=offset*width*4,shape=(a.rows,width))
  if a.reuse_owner_plans:
   planned=a.reuse_owner_plans/f'window_{k:02d}';plan=json.load(open(planned/'placement.json'));assert plan['old_embeddings']==state.next_gid
   for t in range(len(state.owner)):
    ids=np.load(planned/f'{t}.original_ids.npy');owners=np.load(planned/f'{t}.owners.npy');gids=np.load(planned/f'{t}.logical_gids.npy');existing=state.lookup_owner(t,ids)>=0
    assert np.array_equal(state.lookup_owner(t,ids[existing]),owners[existing]);assert np.array_equal(state.gid[t][ids[existing]],gids[existing])
    state.add_new(t,ids[~existing],owners[~existing]);state.gid[t][ids[~existing]]=gids[~existing]
   assert state.next_gid==plan['known_embeddings'] and np.array_equal(state.storage,plan['storage']);state.global_owners()
   placement=alternate(state,planned,data,case,reallocate=False,scheduler='group',rounds=8);placement['owner_plan_reuse_source']=str(planned);placement['old_owner_identity_audited']=True;(case/'placement.json').write_text(json.dumps(placement,indent=2))
  else:
   seed_case=a.out/f'window_{k:02d}_seed';owner_case=a.out/f'window_{k:02d}_owner';placement=window(state,data,cfg,seed_case,rounds=a.rounds,balance=a.balance,schedule=a.seed_schedule);placement=alternate(state,seed_case,data,owner_case,rounds=0);placement=alternate(state,owner_case,data,case,reallocate=False,scheduler='group',rounds=8)
   if not a.keep_profiles:
    shutil.rmtree(seed_case/'profile_cache');shutil.rmtree(owner_case/'profile_cache')
  del data;gc.collect()
 if cache_state is None:
  with open(ROOT/f'stream_init_{g}/state.pkl','rb') as f:cache_state=pickle.load(f)
 update_owners(cache_state,state.global_owners());mirrors,saved=select_mirrors(case,cache_state,budgets,case/'mirrors');set_mirrors(cache_state,mirrors)
 if k==a.windows:
  with open(case/'incoming_cache.pkl','wb') as f:pickle.dump(cache_state,f,protocol=5)
 result,arrays=replay_case(cache_state,case,expected_saved=saved,old_n=placement['old_embeddings'])
 predicted=np.load(case/'predicted_group_counts.npz');nb_case=json.load(open(case/'config.json'))['batches'];assert np.array_equal(predicted['receive'],arrays['backward'][:nb_case].sum(2));assert np.array_equal(predicted['send'],arrays['backward'][:nb_case].sum(1))
 fr=arrays['forward'].sum(2);fs=arrays['forward'].sum(1);br=arrays['backward'].sum(1);bs=arrays['backward'].sum(2);load=fr+br;means=load.mean(0);steps=len(fr);old_forward=0;old_backward=0
 old_forward=result['raw_old_id_remote_queries_per_epoch'];old_backward=result['raw_new_id_remote_queries_per_epoch'];nb=cfg['batches'] if a.rows==cfg['rows'] else a.rows//(g*cfg['batch_size'])
 metrics=dict(window=k,source_row_start=offset,source_row_end=offset+a.rows,rows=a.rows,steps=steps,total_vectors_per_8192=float(load.sum()/steps/g),peak_combined_receive=float(load.max(1).mean()),peak_combined_send=float((fs+bs).max(1).mean()),phase_bottleneck=float((np.maximum(fr.max(1),fs.max(1))+np.maximum(br.max(1),bs.max(1))).mean()),rank_communication_cv=float(means.std()/means.mean()),rank_max_over_mean=float(means.max()/means.mean()),first_epoch_peak=float(load[:nb].max(1).mean()),first_epoch_total=float(load[:nb].mean()),training_vectors=int(load.sum()),training_weight_bytes=int(load.sum())*512,training_metadata_bytes=int(result['metadata_receive_bytes']['total']),raw_old_id_remote_queries_per_epoch=old_forward,raw_new_id_remote_queries_per_epoch=old_backward,migration_moved_rows=0 if migration is None else migration['moved_embeddings'],cumulative_moved_rows=total_moved,migration_weight_bytes=0 if migration is None else migration['weight_bytes'],migration_adam_bytes=0 if migration is None else migration['adam_bytes'],migration_adam_plus_routes_bytes=0 if migration is None else migration['adam_plus_replicated_routes_bytes'],mirror_rows=[len(x) for x in cache_state.mirrors],group_exact_counts_independently_verified=True,placement=placement,seconds=time.perf_counter()-start)
 np.savez_compressed(case/'traffic.npz',fr=fr,fs=fs,br=br,bs=bs);(case/'traffic.json').write_text(json.dumps(result,indent=2));(case/'metrics.json').write_text(json.dumps(metrics,indent=2));allmetrics.append(metrics);(a.out/'metrics.json').write_text(json.dumps(allmetrics,indent=2));print('WINDOW COMPLETE',json.dumps({key:metrics[key] for key in ['window','total_vectors_per_8192','peak_combined_receive','migration_moved_rows','seconds']}),flush=True)
 if last is not None and not a.keep_profiles:
  shutil.rmtree(last/'profile_cache');shutil.rmtree(last/'mirrors')
 with open(a.out/'checkpoint.tmp','wb') as f:pickle.dump((state,cache_state,k),f,protocol=5)
 (a.out/'checkpoint.tmp').replace(a.out/'checkpoint.pkl')
 if a.stop_after and k>=a.stop_after:print('CHECKPOINT STOP',k,flush=True);raise SystemExit(0)
 last=case;del arrays,fr,fs,br,bs,load,mirrors;gc.collect()
state.save(a.out/'final_owners');(a.out/'complete.json').write_text(json.dumps(dict(success=True,windows=len(allmetrics),cumulative_moved_rows=total_moved),indent=2))
