import argparse,json,pickle,time,shutil,gc
import numpy as np
from stream_window import window
from stream_owner_state import OwnerState
from stream_tune import tune
from persistent_replay import *
p=argparse.ArgumentParser();p.add_argument('--ranks',type=int,required=True);p.add_argument('--windows',type=int,default=16);p.add_argument('--rows',type=int,default=4194304);p.add_argument('--interval',type=int,default=0);p.add_argument('--fraction',type=float,default=.001);p.add_argument('--rounds',type=int,default=4);p.add_argument('--balance',type=float,default=4);p.add_argument('--out',type=Path,required=True);p.add_argument('--keep-profiles',action='store_true');a=p.parse_args();g=a.ranks;a.out.mkdir(exist_ok=False);cfg=json.load(open(ROOT/f'r{g}_peak_soft/config.json'));source=Path('/mnt/hugui/longjiu.bty/data/processed/criteo/day6.bin');width=1+cfg['dense']+sum(cfg['table_lengths']);state=OwnerState.from_export(ROOT/f'r{g}_guard_budget_original_owners');budgets=json.load(open(ROOT/f'r{g}_peak_soft_cache/result.json'))['mirror_budget'];cache_state=None
manifest=dict(ranks=g,windows=a.windows,rows_per_window=a.rows,source=str(source),initial_rows=4194304,epochs=5,sample_refinement_rounds=a.rounds,balance=a.balance,fine_tuning_interval_windows=a.interval,fine_tuning_fraction=a.fraction,mirror_capacity_fixed=budgets,global_batch_membership_preserved=True,gpu_execution=False)
(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));allmetrics=[];last=None;total_moved=0
for k in range(1,a.windows+1):
 start=time.perf_counter();migration=None
 if a.interval and k>1 and (k-1)%a.interval==0:
  migration=tune(state,last,a.fraction,a.out/f'migration_before_{k:02d}');total_moved+=migration['moved_embeddings'];inv=update_owners(cache_state,state.global_owners(),migration['logical_gids']);migration['invalidated_cached_copies_by_rank']=inv
 offset=4194304+(k-1)*a.rows;data=np.memmap(source,dtype='<u4',mode='r',offset=offset*width*4,shape=(a.rows,width));case=a.out/f'window_{k:02d}';placement=window(state,data,cfg,case,rounds=a.rounds,balance=a.balance);del data;gc.collect()
 if cache_state is None:
  with open(ROOT/f'stream_init_{g}/state.pkl','rb') as f:cache_state=pickle.load(f)
 update_owners(cache_state,state.global_owners());trace=trace_from_case(case,state);mirrors,saved=select_mirrors(case,cache_state,budgets,case/'mirrors');set_mirrors(cache_state,mirrors);result,arrays=replay_window(cache_state,trace,expected_saved=saved)
 fr=arrays['forward'].sum(2);fs=arrays['forward'].sum(1);br=arrays['backward'].sum(1);bs=arrays['backward'].sum(2);load=fr+br;means=load.mean(0);steps=len(fr);old_forward=0;old_backward=0
 # Raw old/new query split is exact; caching split omitted unless explicitly measured.
 for batch in trace:
  for r,(ids,c) in enumerate(batch):
   remote=cache_state.owner[ids]!=r;old_forward+=int((remote&(ids<placement['old_embeddings'])).sum());old_backward+=int((remote&(ids>=placement['old_embeddings'])).sum())
 metrics=dict(window=k,source_row_start=offset,source_row_end=offset+a.rows,rows=a.rows,steps=steps,total_vectors_per_8192=float(load.sum()/steps/g),peak_combined_receive=float(load.max(1).mean()),peak_combined_send=float((fs+bs).max(1).mean()),phase_bottleneck=float((np.maximum(fr.max(1),fs.max(1))+np.maximum(br.max(1),bs.max(1))).mean()),rank_communication_cv=float(means.std()/means.mean()),rank_max_over_mean=float(means.max()/means.mean()),first_epoch_peak=float(load[:len(trace)].max(1).mean()),first_epoch_total=float(load[:len(trace)].mean()),training_vectors=int(load.sum()),training_weight_bytes=int(load.sum())*512,training_metadata_bytes=int(result['metadata_receive_bytes']['total']),raw_old_id_remote_queries_per_epoch=old_forward,raw_new_id_remote_queries_per_epoch=old_backward,migration_moved_rows=0 if migration is None else migration['moved_embeddings'],cumulative_moved_rows=total_moved,migration_weight_bytes=0 if migration is None else migration['weight_bytes'],migration_adam_bytes=0 if migration is None else migration['adam_bytes'],placement=placement,seconds=time.perf_counter()-start)
 np.savez_compressed(case/'traffic.npz',fr=fr,fs=fs,br=br,bs=bs);(case/'traffic.json').write_text(json.dumps(result,indent=2));(case/'metrics.json').write_text(json.dumps(metrics,indent=2));allmetrics.append(metrics);(a.out/'metrics.json').write_text(json.dumps(allmetrics,indent=2));print('WINDOW COMPLETE',json.dumps({key:metrics[key] for key in ['window','total_vectors_per_8192','peak_combined_receive','migration_moved_rows','seconds']}),flush=True)
 if last is not None and not a.keep_profiles:
  shutil.rmtree(last/'profile_cache');shutil.rmtree(last/'mirrors')
 last=case;del trace,arrays,fr,fs,br,bs,load,mirrors;gc.collect()
state.save(a.out/'final_owners');(a.out/'complete.json').write_text(json.dumps(dict(success=True,windows=len(allmetrics),cumulative_moved_rows=total_moved),indent=2))
