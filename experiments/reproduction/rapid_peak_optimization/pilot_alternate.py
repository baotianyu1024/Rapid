import argparse,pickle,json,gc
from stream_alternate import alternate
from persistent_fast import *
p=argparse.ArgumentParser();p.add_argument('--ranks',type=int,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--passes',type=int,default=2);p.add_argument('--sample-only',action='store_true');p.add_argument('--scheduler',default='peak');p.add_argument('--tag',default='alternate');p.add_argument('--rounds',type=int,default=4);a=p.parse_args();g=a.ranks
with open(a.source/'checkpoint.pkl','rb') as f:state,unused,k=pickle.load(f)
del unused;assert k==1;case=a.source/'window_01';cfg=json.load(open(case/'config.json'));n=cfg['rows'];width=1+cfg['dense']+sum(cfg['table_lengths']);data=np.memmap('/mnt/hugui/longjiu.bty/data/processed/criteo/day6.bin',dtype='<u4',mode='r',offset=4194304*width*4,shape=(n,width));budgets=json.load(open(ROOT/f'r{g}_peak_soft_cache/result.json'))['mirror_budget'];results=[]
for j in range(1,a.passes+1):
 out=ROOT/f'stream{g}_{a.tag}_pilot_{j}';placement=alternate(state,case,data,out,reallocate=not a.sample_only,scheduler=a.scheduler,rounds=a.rounds)
 with open(ROOT/f'stream_init_{g}/state.pkl','rb') as f:cache=pickle.load(f)
 update_owners(cache,state.global_owners());mirrors,saved=select_mirrors(out,cache,budgets,out/'mirrors');set_mirrors(cache,mirrors);result,arrays=replay_case(cache,out,expected_saved=saved,old_n=placement['old_embeddings']);fr=arrays['forward'].sum(2);fs=arrays['forward'].sum(1);br=arrays['backward'].sum(1);bs=arrays['backward'].sum(2);load=fr+br;metrics=dict(pass_number=j,total_vectors_per_8192=float(load.mean()),peak_combined_receive=float(load.max(1).mean()),rank_cv=float(load.mean(0).std()/load.mean()),raw_old_id_remote_queries_per_epoch=result['raw_old_id_remote_queries_per_epoch'],raw_new_id_remote_queries_per_epoch=result['raw_new_id_remote_queries_per_epoch'],placement=placement);(out/'metrics.json').write_text(json.dumps(metrics,indent=2));np.savez_compressed(out/'traffic.npz',fr=fr,fs=fs,br=br,bs=bs)
 with open(out/'checkpoint.pkl','wb') as f:pickle.dump((state,cache,1),f,protocol=5)
 results.append(metrics);(ROOT/f'{a.tag}_pilot_{g}.json').write_text(json.dumps(results,indent=2));print('ALTERNATE RESULT',json.dumps({k:v for k,v in metrics.items() if k!='placement'}),flush=True);case=out;del arrays,cache;gc.collect()
