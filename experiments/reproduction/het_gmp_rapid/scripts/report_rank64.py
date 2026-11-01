"""Balance at 64 logical ranks: primary/mirror storage and directed traffic."""
from common import *
import numpy as np,hashlib,csv
import argparse
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,default=ROOT/'results/rank64_balance');a=p.parse_args();CASE=a.case.resolve();P=64;B=8192;E=5
read=lambda p:json.loads(p.read_text())
def distribution(v):
 v=np.asarray(v,dtype=float);m=v.mean()
 return dict(min=float(v.min()),max=float(v.max()),mean=float(m),std=float(v.std()),cv=float(v.std()/m) if m else 0,max_over_mean=float(v.max()/m) if m else 1,min_over_mean=float(v.min()/m) if m else 1,values=v.tolist())
def traffic(a):
 a=np.asarray(a,dtype=float);means=a.mean(1);ratio=np.divide(a.max(1),means,out=np.ones(len(a)),where=means>0);cv=np.divide(a.std(1),means,out=np.zeros(len(a)),where=means>0)
 return dict(steps=len(a),mean=float(a.mean()),total=int(a.sum()),mean_step_peak=float(a.max(1).mean()),absolute_peak=int(a.max()),mean_step_max_over_mean=float(ratio.mean()),worst_step_max_over_mean=float(ratio.max()),mean_step_cv=float(cv.mean()),worst_step_cv=float(cv.max()),rank_means=distribution(a.mean(0)),step_peak_ratios=ratio.tolist())
def main():
 from run_latest_rapid64 import validate_manifest
 validate_manifest(CASE/'latest_rapid/source_manifest.json')
 assert read(CASE/'verification.json')['success']
 assert read(CASE/'latest_baseline_complete.json')['success']
 audit=read(CASE/'latest_baseline_audit.json');assert audit['global_batch_membership_preserved'] and audit['matching_128bit_record_multiset']
 cfg=read(CASE/'data/input.json');meta=read(CASE/'traffic/het_gmp_s0.json');final=CASE/'latest_rapid/rapid_guard_p5'
 assert cfg['rows']==4194304 and cfg['ranks']==64
 owner=np.load(CASE/'gmp/owner.npy');gp=np.bincount(owner,minlength=P);gm=np.array([len(np.load(CASE/f'gmp/mirrors_{r}.npy')) for r in range(P)])
 rp=np.array(read(final/'refinement.json')['after_storage']);rawp=np.array(audit['owner_counts'])
 assert rp.sum()==gp.sum()==rawp.sum()==55874235
 storage={}
 for name,primary,mirror in [('RAPID_initial',rawp,np.zeros(P,int)),('RAPID_balanced',rp,np.zeros(P,int)),('GMP_s0',gp,gm)]:
  storage[name]=dict(primary=distribution(primary),mirrors=distribution(mirror),resident_parameter_rows=distribution(primary+mirror),mirror_parameter_plus_gradient_rows=int(2*mirror.sum()),mirror_over_primary=float(mirror.sum()/primary.sum()),resident_parameter_max_over_primary_mean=float((primary+mirror).max()/primary.mean()))
 with np.load(CASE/'traffic/het_gmp_s0.npz') as z:
  f=z['forward'];g=z['backward'];assert f.shape==g.shape and np.all(f<=g)
  gs=meta['steps_per_epoch'];full=np.arange(len(f))%gs < min(meta['rank_rows'])//B
  arrays={'GMP_s0':dict(forward_receive=f.sum(2),forward_send=f.sum(1),backward_receive=g.sum(1),backward_send=g.sum(2))}
  # With strict s=0 and no replicas, every remote unique ID is fetched and returned.
  arrays['GMP_1d_derived']=dict(forward_receive=g.sum(2),forward_send=g.sum(1),backward_receive=g.sum(1),backward_send=g.sum(2))
 masks={'GMP_s0':full,'GMP_1d_derived':full};steps={'GMP_s0':gs,'GMP_1d_derived':gs}
 for name,path in [('RAPID_initial',CASE/'latest_rapid/rapid_schedule'),('RAPID_balanced',final)]:
  with np.load(path/'cps_counts.npz') as z:
   assert z['receive'].shape==(8,26,P);assert np.array_equal(z['receive'].sum(2),z['send'].sum(2))
   assert np.all(z['receive'].mean(axis=(0,2))<B*(P-1)), 'ACP selects Reduction: update direction model'
   a=np.tile(z['receive'].sum(1),(E,1));b=np.tile(z['send'].sum(1),(E,1))
  arrays[name]=dict(forward_receive=a,forward_send=b,backward_receive=b,backward_send=a);masks[name]=np.ones(len(a),bool);steps[name]=8
 results={}
 with (CASE/'traffic_by_rank_step.csv').open('w') as csvf:
  w=csv.writer(csvf);w.writerow(['algorithm','epoch','step_in_epoch','rank','full_batch_step','forward_receive','forward_send','backward_receive','backward_send'])
  for name,a in arrays.items():
   assert np.array_equal(a['forward_receive'].sum(1),a['forward_send'].sum(1));assert np.array_equal(a['backward_receive'].sum(1),a['backward_send'].sum(1))
   n=len(a['forward_receive']);first=np.arange(n)<steps[name]
   a['combined_receive']=a['forward_receive']+a['backward_receive'];a['combined_send']=a['forward_send']+a['backward_send']
   results[name]={scope:{key:traffic(v[mask]) for key,v in a.items()} for scope,mask in [('all_steps',np.ones(n,bool)),('full_batches',masks[name]),('first_epoch_full_batches',first&masks[name])]}
   results[name]['normalized_forward_per_8192']=int(a['forward_receive'].sum())/(cfg['rows']*E)*B
   results[name]['normalized_forward_backward_per_8192']=int(a['combined_receive'].sum())/(cfg['rows']*E)*B
   np.savez_compressed(CASE/(name+'_counts.npz'),**a,full_batch_mask=masks[name])
   for i in range(n):
    for r in range(P):w.writerow([name,i//steps[name]+1,i%steps[name],r,int(masks[name][i]),*[int(a[k][i,r]) for k in ['forward_receive','forward_send','backward_receive','backward_send']]])
 with (CASE/'storage_by_rank.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['algorithm','rank','primary','mirror','resident_parameters'])
  for name,s in storage.items():
   for r in range(P):w.writerow([name,r,int(s['primary']['values'][r]),int(s['mirrors']['values'][r]),int(s['resident_parameter_rows']['values'][r])])
 summary=dict(primary_comparison=['RAPID_balanced','GMP_s0'],rapid_source_manifest=read(CASE/'latest_rapid/source_manifest.json'),superseded_mixed_source_results_excluded=True,config=read(CASE/'benchmark_config.json'),storage=storage,traffic=results,gmp_rank_rows=meta['rank_rows'],gmp_full_steps_per_epoch=int(full.sum()/E),verification=read(CASE/'verification.json'),baseline_audit=audit,limitations=['CPU replay, no actual 64-GPU training or network measurements','Same day6 prefix 4194304 rows; only eight distinct global batches per epoch; five epochs do not give forty independent data batches','RAPID_initial is the intermediate placement/schedule of the same latest-source run, without owner optimization; primary comparison uses RAPID_balanced, the complete latest pipeline; old mixed-source run is excluded','RAPID_balanced transfers existing fixed parameters to 64 ranks; no retuning','Full-batch steps exclude ragged/empty rank tails for balance diagnosis; all rows retained in normalized volume totals','GMP uses global sample grouping; RAPID preserves global batch membership','Only observed table/ID primary rows; not full declared table capacities or measured GPU memory','Storage and traffic units are vectors, equal dimensions assumed; excludes metadata, dense allreduce and preprocessing','GMP shared-state OpenMP heuristic one run; no confidence intervals','Algorithms generated concurrently: stage wall times are not a speed comparison','GMP_1d_derived is a no-mirror strict-s0 control derived from independently verified remote unique-ID matrices'])
 save(CASE/'comparison.json',summary)
 lines=['64-rank RAPID / GMP 均衡性实验','',f"相同 day6 {cfg['rows']:,} 样本，64 rank，每rank batch8192，5 epoch；8个独立global batch，不是40个独立数据批次。",'GMP 10轮划分、s=0、1%远端候选mirror；RAPID全部来自当前working/scripts源码快照，初始阶段仅作诊断，完整最新流程为主对照。','均衡性以所有rank都有完整batch的步骤统计；总量包含全部尾批并按实际样本归一。','不是64张GPU实测，未测GPU时间或实际网络流量。','', '主副本存储（活跃embedding ID数；最大/平均越接近1越均衡）：']
 for name,s in storage.items():
  d=s['primary'];lines.append(f"{name}: min={d['min']:.0f}, max={d['max']:.0f}, mean={d['mean']:.3f}, max/mean={d['max_over_mean']:.6f}, CV={100*d['cv']:.4f}%, mirror/primary={100*s['mirror_over_primary']:.4f}%")
 lines+=['','通信（满batch，5epoch）：算法 | 方向 | 平均向量 | 每步最忙rank均值 | 平均step峰均比 | 最差step峰均比 | 平均step CV']
 for name in ['RAPID_initial','RAPID_balanced','GMP_s0','GMP_1d_derived']:
  for k in ['forward_receive','forward_send','backward_receive','backward_send']:
   d=results[name]['full_batches'][k];lines.append(f"{name} | {k} | {d['mean']:.3f} | {d['mean_step_peak']:.3f} | {d['mean_step_max_over_mean']:.6f} | {d['worst_step_max_over_mean']:.6f} | {100*d['mean_step_cv']:.4f}%")
 lines+=['','同样本总量归一，每8192样本：']
 for name,d in results.items():lines.append(f"{name}: forward={d['normalized_forward_per_8192']:.6f}, forward+backward={d['normalized_forward_backward_per_8192']:.6f}")
 lines+=['',f"GMP 每epoch步骤={gs}，所有rank均满batch的步骤={int(full.sum()/E)}；rank样本数 min={min(meta['rank_rows'])}, max={max(meta['rank_rows'])}",'GMP mirror数量='+str(int(gm.sum()))+'；按128维FP32，mirror参数+梯度额外存储下界GiB='+str(int(2*gm.sum())*128*4/1024**3),'','版本和局限：']+summary['limitations']
 (CASE/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
 files=[Path(__file__),ROOT/'scripts/run64_comparison.py',ROOT/'scripts/verify_rank64.py',ROOT/'scripts/baseline64.py',ROOT/'scripts/audit_samples64.cpp',ROOT/'scripts/plot_rank64.py',ROOT/'scripts/run_latest_rapid64.py',ROOT/'scripts/oracle_rank64.py']
 save(CASE/'analysis_source_sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
 import subprocess
 subprocess.run(['/home/longjiu.bty/miniconda3/bin/python',str(ROOT/'scripts/plot_rank64.py'),'--case',str(CASE)],check=True)
if __name__=='__main__':main()
