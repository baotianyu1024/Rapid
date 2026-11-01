import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;R=ROOT.parent
p=argparse.ArgumentParser();p.add_argument('--cache-result',type=Path,required=True);a=p.parse_args();m=json.load(open(a.cache_result/'result.json'));g=m['ranks'];case=Path(m['case']);z=np.load(a.cache_result/'counts.npz');storage=np.array(json.load(open(case/'storage.json'))['primary_counts'])

def stats(x):
 x=np.asarray(x,dtype=np.float64);mean=x.mean();return dict(min=float(x.min()),max=float(x.max()),mean=float(mean),cv=float(x.std()/mean),max_over_mean=float(x.max()/mean))

def traffic(fr,fs,br,bs):
 arrays=dict(forward_receive=fr,forward_send=fs,backward_receive=br,backward_send=bs,combined_receive=fr+br,combined_send=fs+bs)
 return {k:dict(mean_step_peak=float(v.max(1).mean()),mean_step_max_over_mean=float((v.max(1)/v.mean(1)).mean()),mean_step_cv=float((v.std(1)/v.mean(1)).mean()),rank_means=v.mean(0).tolist(),rank_mean_balance=stats(v.mean(0))) for k,v in arrays.items()}
result={'optimized':dict(primary=stats(storage),primary_counts=storage.tolist(),active_primary_and_mirrors=stats(storage+np.array(m['mirror_counts'])),primary_and_reserved_mirror_capacity=stats(storage+np.array(m['mirror_budget'])),traffic=traffic(z['fr'],z['fs'],z['br'],z['bs']))}
q=ROOT/f'gmp_fixed_batch_{g}/traffic';gm=json.load(open(q/'het_gmp_s0.json'));z=np.load(q/'het_gmp_s0.npz');f=z['forward'];b=z['backward'];result['gmp_fixed_batch']=dict(primary=stats(gm['primary_counts']),primary_counts=gm['primary_counts'],active_primary_and_mirrors=stats(np.array(gm['primary_counts'])+np.array(gm['mirror_counts'])),traffic=traffic(f.sum(2),f.sum(1),b.sum(1),b.sum(2)))
original=R/'herald_rapid_gmp/results/default8/fresh_baselines/traffic' if g==8 else R/'het_gmp_rapid/results/rank64_balance/traffic';om=json.load(open(original/'het_gmp_s0.json'));oz=np.load(original/'het_gmp_s0.npz');full=np.tile(np.arange(om['steps_per_epoch'])<min(om['rank_rows'])//8192,5);of=oz['forward'][full];ob=oz['backward'][full];result['gmp_original_full_steps']=dict(primary=stats(om['primary_counts']),primary_counts=om['primary_counts'],traffic=traffic(of.sum(2),of.sum(1),ob.sum(1),ob.sum(2)))
latest=R/'herald_rapid_gmp/results/default8/latest_rapid' if g==8 else R/'het_gmp_rapid/results/rank64_balance/latest_rapid';q=latest/'rapid_guard_p5';meta=json.load(open(q/'refinement.json'));z=np.load(q/'predicted_counts.npz');r=z['receive'].sum(1);s=z['send'].sum(1);result['rapid_original']=dict(primary=stats(meta['after_storage']),primary_counts=meta['after_storage'],traffic=traffic(r,s,s,r))
result['scope']=dict(ranks=g,rows=m['rows'],epochs=m['epochs'],original_global_batch_membership_preserved=True,communication_unit='embedding vectors',hardware_measured=False,cache_model='strict s=0 read cache; equal per-rank GMP mirror capacity upper bound; unused capacity distinguished from active cached IDs')
(a.cache_result/'balance.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:{'primary':v['primary'],'combined_receive':v['traffic']['combined_receive']['rank_mean_balance']} for k,v in result.items() if k!='scope'},indent=2))
