"""16-rank timings and remote-vector counts with explicit replay and tail scopes."""
import json
from pathlib import Path
import numpy as np
from common import ROOT,save
CASE=ROOT/'results/rank16_fast';P=16;B=8192;E=5
read=lambda p:json.loads(p.read_text())
def stats(a):
 means=a.mean(1);peaks=a.max(1)
 return dict(steps=len(a),mean=float(a.mean()),mean_batch_peak=float(peaks.mean()),absolute_peak=int(a.max()),max_peak_mean_ratio=float(np.divide(peaks,means,out=np.ones_like(peaks,dtype=float),where=means!=0).max()),rank_means=a.mean(0).tolist(),total_vectors=int(a.sum()))
def main():
 assert read(CASE/'preprocess_complete.json')['success'] and read(CASE/'verification.json')['success']
 names=['input_and_gmp_adapter','gmp_partition','gmp_materialize','rapid_hot','rapid_placement','rapid_schedule','rapid_owner_base','rapid_price_f2','rapid_price_b8','rapid_price_b32','rapid_guard_p5','rapid_final_materialize']
 timing={n:read(CASE/(n+'.status.json')) for n in names};assert all(v['exit_code']==0 for v in timing.values())
 data=read(CASE/'data/input.json');gmeta=read(CASE/'traffic/het_gmp_s0.json');final=CASE/'rapid_guard_p5'
 rapid_time=sum(v['seconds'] for n,v in timing.items() if n.startswith('rapid_'))
 gmp_time=sum(timing[n]['seconds'] for n in ['input_and_gmp_adapter','gmp_partition','gmp_materialize'])-data['copy_seconds']
 with np.load(final/'cps_counts.npz') as z:
  assert z['receive'].shape==(32,26,16)
  rf=np.tile(z['receive'].sum(1),(E,1));rb=np.tile(z['send'].sum(1),(E,1))
  assert np.all(z['receive'].mean(axis=(0,2))<B*(P-1))
 with np.load(CASE/'traffic/het_gmp_s0.npz') as z:
  gf=z['forward'].sum(2);gb=z['backward'].sum(1)
 assert gmeta['processed_rows']==data['rows']*E and gmeta['pending_updates']==0
 assert int(gf.sum())==gmeta['forward_receive_vectors']['total'] and int(gb.sum())==gmeta['backward_receive_vectors']['total']
 fullg=np.arange(len(gf))%gmeta['steps_per_epoch']<min(gmeta['rank_rows'])//B
 results={}
 for name,f,b,full,steps in [('rapid',rf,rb,np.ones(len(rf),bool),32),('gmp_s0',gf,gb,fullg,gmeta['steps_per_epoch'])]:
  results[name]={}
  for scope,mask in [('all_steps',np.ones(len(f),bool)),('full_batches',full),('first_epoch',np.arange(len(f))<steps)]:
   results[name][scope]={label:stats(a[mask]) for label,a in [('forward',f),('backward',b),('combined',f+b)]}
  np.savez_compressed(CASE/(name+'_receive_counts.npz'),forward=f,backward=b,full_batch_mask=full)
  np.savetxt(CASE/(name+'_step_peaks.csv'),np.column_stack([np.arange(len(f)),f.max(1),b.max(1),(f+b).max(1)]),fmt='%d',delimiter=',',header='step,forward_max,backward_max,combined_max',comments='')
 rapid_owner=np.array(read(final/'refinement.json')['after_storage']);gmp_owner=np.array(gmeta['primary_counts']);mirrors=np.array(gmeta['mirror_counts'])
 storage=dict(rapid_primary=rapid_owner.tolist(),rapid_primary_max_over_mean=float(rapid_owner.max()/rapid_owner.mean()),gmp_primary=gmp_owner.tolist(),gmp_primary_max_over_mean=float(gmp_owner.max()/gmp_owner.mean()),gmp_mirrors=mirrors.tolist(),gmp_mirror_over_primary=float(mirrors.sum()/gmp_owner.sum()))
 pct={k:100*(results['rapid']['full_batches'][k]['mean_batch_peak']/results['gmp_s0']['full_batches'][k]['mean_batch_peak']-1) for k in ['forward','backward']}
 pct['total_vectors']=100*(results['rapid']['all_steps']['combined']['total_vectors']/results['gmp_s0']['all_steps']['combined']['total_vectors']-1)
 report=dict(ranks=P,rows=data['rows'],local_batch=B,epochs=E,input_sha256=data['input_sha256'],rapid_preprocess_seconds=rapid_time,gmp_preprocess_seconds=gmp_time,rapid_over_gmp_time=rapid_time/gmp_time,common_input_copy_seconds=data['copy_seconds'],preprocessing_stages=timing,traffic=results,rapid_change_vs_gmp_percent=pct,storage=storage,gmp_rank_rows=gmeta['rank_rows'],verification=read(CASE/'verification.json'),scope=read(CASE/'benchmark_config.json'))
 save(CASE/'comparison.json',report)
 lines=['16-rank RAPID / GMP：完整预处理耗时与通信量','',
 '相同4194304样本，每rank batch8192，RAPID每epoch32步，通信统计5epoch。',
 'RAPID 为最新精确加速版本，8rank固定参数直接迁移16rank；未重新搜索16rank参数，不声称是16rank全局最优。',
 'GMP 为原生代码10轮分区、1% remote-candidate mirrors、s=0协议回放。',
 '预处理由CPU实际运行计时；通信由真实分区/数据上的CPU协议回放统计，不是GPU/NCCL测量。',
 '同机顺序执行，OMP16/BLAS1，RAPID16个batch worker。全新生成，无旧placement/owner/profile缓存。','',
 f'RAPID 完整预处理：{rapid_time:.3f} 秒，{rapid_time/60:.3f} 分钟。',f'GMP 完整预处理：{gmp_time:.3f} 秒，{gmp_time/60:.3f} 分钟。',f'RAPID/GMP 耗时比：{rapid_time/gmp_time:.4f}。',
 f'公共输入切片复制 {data["copy_seconds"]:.3f} 秒单列；回放及额外验收不计入预处理。各一次、正常文件缓存。','',
 '通信单位：embedding/梯度向量个数。只含远端向量，排除自身、索引/时钟等元数据、稠密参数AllReduce。',
 '满batch口径：每一步所有rank均有8192样本；均值=rank×step平均；峰值均值=先每步取16rank最大值再平均。',
 '方案 | 满batch步数 | 方向 | rank×step均值 | 每步最大接收量均值 | 单步单rank绝对峰值 | 最差max/mean']
 for name in results:
  for kind in ['forward','backward']:
   v=results[name]['full_batches'][kind];lines.append(f'{name} | {v["steps"]} | {kind} | {v["mean"]:.3f} | {v["mean_batch_peak"]:.3f} | {v["absolute_peak"]} | {v["max_peak_mean_ratio"]:.6f}')
 lines += ['',f'RAPID 相对GMP：前向峰值均值 {pct["forward"]:+.3f}%；反向峰值均值 {pct["backward"]:+.3f}%。',
 '同样样本数的5epoch总通信（包含全部尾批，前向+反向）：']
 for name in results:lines.append(f'{name}: {results[name]["all_steps"]["combined"]["total_vectors"]} 个向量。')
 lines += [f'RAPID 总通信相对GMP：{pct["total_vectors"]:+.3f}%。','',
 f'GMP各rank样本数：{gmeta["rank_rows"]}；每epoch {gmeta["steps_per_epoch"]} 步；完整batch均值排除不等长尾批。',
 '完整batch均值用于消除小尾批稀释，不代表双方每一步是同一组样本；同样样本总量比较必须包含所有尾批。',
 '首epoch冷启动和包含尾批的均值另存comparison.json。mirror版本跨epoch保留。',
 f'RAPID owner最大/平均={storage["rapid_primary_max_over_mean"]:.6f}；GMP primary最大/平均={storage["gmp_primary_max_over_mean"]:.6f}。',
 f'GMP mirror数量={int(mirrors.sum())}；占唯一primary数={storage["gmp_mirror_over_primary"]:.4%}；RAPID无mirror。','',
 '验证：GMP全部输出记录与源数据匹配；s=0反向通信独立逐rank逐batch去重重算，5epoch矩阵一致；首步冷读取和前向子集约束通过；无待发送梯度。',
 'RAPID两次owner映射均完成完整逆映射核对；最终通信独立重扫一致。',
 '回放假设：全体rank按同步协议轮次执行，完整消费样本并使用不重复/不丢弃的尾批；不模拟实际GPU异步到达顺序和原生dataloader循环补齐。','', '预处理分项（秒）：']
 lines += [f'{n}: {v["seconds"]:.3f}' for n,v in timing.items()]
 (CASE/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
