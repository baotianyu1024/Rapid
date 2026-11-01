"""One-to-one paper/measurement panels; never use references in measurements."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figure_comparison';OUT.mkdir(exist_ok=True)
refs=json.loads((ROOT/'paper_reference_targets.json').read_text());records=[]
plt.rcParams.update({'font.size':9})
cases={};variants={}
for p in sorted((ROOT/'results').iterdir()):
 if not (p/'complete.json').exists():continue
 cfg=json.loads((p/'config.json').read_text());c=json.loads((p/'cps_metrics.json').read_text());h=json.loads((p/'hash_metrics.json').read_text())
 entry=dict(path=p,cfg=cfg,c=c,h=h)
 if cfg.get('base_case'):variants.setdefault(Path(cfg['base_case']).name,[]).append(entry)
 if not cfg['repo'].endswith('update_Rapid'):continue
 key=(cfg['dataset'],cfg['ranks']);priority=('day6' in p.name,cfg['batches'])
 if key not in cases or priority>cases[key]['priority']:cases[key]=entry|dict(priority=priority)
def add(fig,panel,metric,paper,value,scope,direction='descriptive',note='',tolerance=0.0):
 if paper is None:status='仅实测/原图未给出可比数值';delta=None
 else:
  delta=value-paper
  if direction=='descriptive':status='参考值精度内一致' if abs(delta)<=max(tolerance,1e-10) else '数据特征或口径差异，非性能达标判定'
  else:status='达到或优于参考值' if (value<=paper+tolerance if direction=='lower' else value>=paper-tolerance) else '未达到参考值'
 records.append(dict(figure=fig,panel=panel,metric=metric,paper=paper,measured=float(value),delta=delta,direction=direction,status=status,scope=scope,note=note,reference_rounding_tolerance=tolerance))
def save(fig,name):
 fig.tight_layout();fig.savefig(OUT/f'{name}.png',dpi=180);fig.savefig(OUT/f'{name}.pdf');plt.close(fig)
def scope(e):return f"{e['path'].name}; {e['cfg']['rows']} rows"
def best(e, require_balance=True):
 eligible=[];pending=[e];seen=set()
 while pending:
  current=pending.pop()
  if current['path'].name in seen:continue
  seen.add(current['path'].name)
  for v in variants.get(current['path'].name,[]):
   if v['cfg']['rows']==e['cfg']['rows']:
    eligible.append(v);pending.append(v)
 candidates=[e]+eligible
 balanced=[v for v in candidates if v['c']['acp_receive']['max_batch_peak_over_mean']<=1.200001] if require_balance else []
 return min(balanced or candidates,key=lambda v:(v['c']['acp_receive']['mean_batch_peak'],v['c']['acp_send']['mean_batch_peak_over_mean']))
# Figure4 audited against the Eq.(5)/Figure11 dense Reduction model,
# excluding self-rank payload under the user's receive-only convention.
# The printed Figure4 Reduction bars use a peer-sparse model and conflict
# with the paper's stated B*G cost, so they are references to an invalidated
# alternative rather than a second valid communication metric.
f4=ROOT/'results/criteo_day6_figure4.json'
if not f4.exists():f4=ROOT/'results/criteo_pilot_figure4.json'
if f4.exists():
 d=json.loads(f4.read_text());selected=[0,4,9,19,20,21];pu=np.array([7,6,16,30,198,65])*1000;pr=np.array([17,30,36,45,57,56])*1000
 canonical_per_table=d['batch_size']*(d['ranks']-1)
 canonical_reduction=np.full(26,canonical_per_table,dtype=float)
 canonical_acp=np.minimum(np.array(d['per_table_unique']),canonical_reduction)
 canonical_total_reduction=float(canonical_reduction.sum());canonical_total_acp=float(canonical_acp.sum())
 fig,ax=plt.subplots(1,2,figsize=(13,4));x=np.arange(6);w=.2
 for off,vals,label,color,hatch in [(-1.5,pu,'Paper Unique','C0','//'),(-.5,np.array(d['per_table_unique'])[selected],'Measured Unique','C0',None),(.5,pr,'Paper Figure4 Reduction','C1','//'),(1.5,canonical_reduction[selected],'Dense receive, excluding self','C3',None)]:ax[0].bar(x+off*w,vals/1000,w,label=label,color=color,hatch=hatch,alpha=.7 if hatch else 1)
 ax[0].set_xticks(x,[f'T{t}' for t in selected]);ax[0].set_ylabel('1000 forward vectors/rank/batch');ax[0].set_title('Figure4(b): Reduction conflicts with Eq.(5)/Figure11');ax[0].legend(fontsize=7)
 pvals=[567000,377000,228000];mvals=[canonical_total_reduction,d['total_unique'],canonical_total_acp]
 ax[1].bar(np.arange(3)-.18,np.array(pvals)/1000,.36,label='Paper Figure4',hatch='//');ax[1].bar(np.arange(3)+.18,np.array(mvals)/1000,.36,label='Dense remote receive');ax[1].set_xticks(range(3),['Reduction','Unique','ACP']);ax[1].legend();ax[1].set_title('Figure4(c): receive-only recount');fig.suptitle(f'{f4.stem}; {d["rows"]:,} rows')
 save(fig,'Figure4')
 for t,u,r in zip(selected,pu,pr):
  add('4','b',f'T{t} Unique',int(u),d['per_table_unique'][t],f4.stem,note='Paper bar labels rounded to 1000 vectors')
  add('4','b',f'T{t} Reduction remote receive',int(r),canonical_per_table,f4.stem,note='Dense B*(G-1) receive excluding self; Figure4 uses peer-sparse values')
 for key,pv,mv in zip(['Reduction remote receive','Unique','ACP remote receive'],pvals,mvals):add('4','c',key,pv,mv,f4.stem,note='Paper Figure4 uses peer-sparse Reduction rather than dense receive-only model')
 add('4','c','ACP / Unique remote receive',.60,canonical_total_acp/d['total_unique'],f4.stem,note='Descriptive only: paper numerator uses a different Reduction model')
 add('4','b','T20 Unique / Reduction remote receive',3.5,d['per_table_unique'][20]/canonical_per_table,f4.stem,note='Paper ratio uses its peer-sparse Figure4 Reduction bar')
# Figure6: all printed histogram bins, MCR estimates labelled as samples.
f6=ROOT/'results/criteo_day6_cold_statistics/statistics.json'
if not f6.exists():f6=ROOT/'results/criteo_pilot_cold_exact1m_mcr/statistics.json'
hot=ROOT/'results/criteo_day6_hot_top40k_split.json'
if not hot.exists():hot=ROOT/'results/criteo_hot_top40k_split.json'
if f6.exists():
 d=json.loads(f6.read_text());ph=np.array([96.62,3.35,.03,.01]);mh=np.array(d['frequency_bin_shares'])*100
 mcr=np.array([x['mcr'] for x in d['mcr_records']]);pm=np.array([11.,7.7,1.9,79.3]);mm=np.array([np.mean(mcr<=.5),np.mean((mcr>.5)&(mcr<=.8)),np.mean((mcr>.8)&(mcr<=.95)),np.mean(mcr>.95)])*100
 fig,ax=plt.subplots(1,3,figsize=(15,4))
 # Top40000 split is included only if its real-data artifact is available.
 if hot.exists():
  hd=json.loads(hot.read_text());raw=hd['hot_share_raw_then_unique'][0];uq=hd['cold_share_raw_then_unique'][1];mp=np.array([100*raw,100*uq]);pp=np.array([71.,91.5])
  ax[0].bar(np.arange(2)-.18,pp,.36,label='Paper',hatch='//');ax[0].bar(np.arange(2)+.18,mp,.36,label='Measured Top40000');ax[0].set_xticks(range(2),['Raw hot share','Unique cold share']);ax[0].legend(fontsize=7)
  for label,pv,mv in zip(['raw hot %','unique cold %'],pp,mp):add('6','a',label,pv,mv,hot.stem)
 else:ax[0].text(.1,.5,'Top40000 measurement pending')
 for axis,pvals,mvals,labels,title in [(ax[1],ph,mh,['<0.01%','0.01-1%','1-10%','10-100%'],'(b) Observed-ID frequency'),(ax[2],pm,mm,['0-50%','50-80%','80-95%','>95%'],'(c) MCR: sampled cold IDs')]:
  x=np.arange(4);axis.bar(x-.18,pvals,.36,label='Paper',hatch='//');axis.bar(x+.18,mvals,.36,label='Measured');axis.set_xticks(x,labels,rotation=15);axis.legend(fontsize=7);axis.set_title(title)
  for label,pv,mv in zip(labels,pvals,mvals):add('6','b' if axis is ax[1] else 'c',label+' share %',pv,mv,f6.parent.name,note=f'MCR uses {d.get("mcr_rows",d["rows"])} rows, {d["mcr_sample_size"]} sampled IDs; frequency uses {d["rows"]} rows')
 n=len(mcr);z=1.959963984540054;phat=mm/100
 center=(phat+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/(1+z*z/n)
 ax[2].errorbar(np.arange(4)+.18,mm,yerr=np.vstack([mm-100*(center-half),100*(center+half)-mm]),fmt='none',ecolor='black',capsize=3,label='95% Wilson CI')
 ax[2].set_title(f'(c) MCR: {n} cold IDs; 95% CI');ax[2].legend(fontsize=6)
 ax[1].set_yscale('log');ax[0].set_title('(a) Traffic composition');fig.suptitle('Figure6: dataset statistics; larger values do not imply a faster algorithm')
 save(fig,'Figure6')
# Figures10/11: paper script constants vs measured baseline and genuine improvements.
for figure in [10,11]:
 fig,axes=plt.subplots(1,3,figsize=(16,4.5))
 for ax,ds in zip(axes,['criteo','kaggle','taobao']):
  gs=[g for g in [8,16,32,64,128] if (ds,g) in cases];denom=8192*(4 if ds=='taobao' else 214)
  fields=['uniq','acp','cps'] if figure==10 else ['red','uniq','acp','cps']
  for i,key in enumerate(fields):
   paper=[];measured=[];improved=[]
   for g in gs:
    e=cases[ds,g];v=best(e);r=refs[ds][str(g)][key]
    val={'red':e['h']['all_reduction_receive_per_rank'],'uniq':e['h']['unique_receive']['mean'],'acp':e['h']['acp_receive']['mean'],'cps':e['c']['acp_receive']['mean_batch_peak']}[key]
    opt=v['c']['acp_receive']['mean_batch_peak'] if key=='cps' else val
    if figure==10:r,val,opt=1-r/denom,1-val/denom,1-opt/denom
    paper.append(r);measured.append(val);improved.append(opt)
    add(str(figure),ds,f'{key} G{g}',r,val,scope(e),'higher' if figure==10 and key=='cps' else ('lower' if figure==11 and key=='cps' else 'descriptive'),note='Figure10 uses the same CPS receive-peak payload as supplied paper plotting script')
    if v is not e and key=='cps':add(str(figure),ds,f'{key} G{g} improved',r,opt,scope(v),'higher' if figure==10 else 'lower')
   ax.plot(gs,paper,'--',color=f'C{i}',label=f'Paper {key}');ax.plot(gs,measured,'o-',color=f'C{i}',label=f'Measured {key}',markersize=4)
   if key=='cps' and not np.array_equal(improved,measured):ax.plot(gs,improved,'s:',color='black',label='Improved cps',markersize=4)
  unconstrained=[best(cases[ds,g],False) for g in gs]
  if any(v['path']!=best(cases[ds,g])['path'] for g,v in zip(gs,unconstrained)):
   peaks=[v['c']['acp_receive']['mean_batch_peak'] for v in unconstrained]
   ax.plot(gs,[1-v/denom for v in peaks] if figure==10 else peaks,'^:',color='purple',label='Min traffic (may fail balance)',markersize=4)
   for g,v,mv in zip(gs,unconstrained,peaks):
    if v['path']==best(cases[ds,g])['path']:continue
    pv=refs[ds][str(g)]['cps']
    add(str(figure),ds,f'cps G{g} min-traffic alternative',1-pv/denom if figure==10 else pv,1-mv/denom if figure==10 else mv,scope(v),'higher' if figure==10 else 'lower',note=f"Alternative may fail balance; receive worst ratio={v['c']['acp_receive']['max_batch_peak_over_mean']}")
  if figure==11:
   for g in gs:
    e=cases[ds,g];v=best(e);pv=100*(1-refs[ds][str(g)]['cps']/refs[ds][str(g)]['uniq']);mv=100*(1-v['c']['acp_receive']['mean_batch_peak']/e['h']['unique_receive']['mean'])
    add('11-derived',ds,f'CPS peak reduction vs Hash mean % G{g}',pv,mv,scope(v),'higher',note='Derived normalization for different input diversity; does not replace absolute Figure11 comparison')
  means=[cases[ds,g]['c']['acp_receive']['mean'] for g in gs]
  ax.plot(gs,[1-v/denom for v in means] if figure==10 else means,'x:',color='grey',label='CPS mean (alternative)',markersize=4)
  for g,mv in zip(gs,means):
   pv=refs[ds][str(g)]['cps']
   add(str(figure),ds,f'cps mean alternative G{g}',1-pv/denom if figure==10 else pv,1-mv/denom if figure==10 else mv,scope(cases[ds,g]),note='Alternative aggregation only; does not replace the required bottleneck-rank peak')
  ax.set_xscale('log',base=2);ax.set_xticks(gs,gs);ax.set_title(ds);ax.set_xlabel('Logical ranks');ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2)
  if figure==11:ax.set_yscale('log')
 axes[0].set_ylabel('1 - transferred / raw requests' if figure==10 else 'Forward vectors/rank/batch (log)')
 fig.suptitle(f'Figure{figure}: paper dashed, real-data measured solid; CPU trace simulation')
 save(fig,f'Figure{figure}')
# Figure12b: keep both locality denominators and disclose the remote-ID label inconsistency.
gs=[g for g in [8,32,128] if ('criteo',g) in cases]
if gs:
 fig,ax=plt.subplots(1,2,figsize=(12,4));hash_ref={8:467000,32:467000,128:467000};cps_ref={8:348000,32:286000,128:235000}
 for scheme,color in [('hash','C0'),('cps','C1')]:
  paper=[(hash_ref if scheme=='hash' else cps_ref)[g] for g in gs];meas=[]
  for g,pv in zip(gs,paper):
   e=cases['criteo',g];m=e['h' if scheme=='hash' else 'c'];mv=sum(t['unique_count_mean'] for t in m['per_table']);meas.append(mv)
   add('12','b-left',f'{scheme} total unique G{g}',pv,mv,scope(e),'descriptive',note='Paper calls these remote IDs, but constant Hash value includes local IDs; true remote separately reported')
   add('12','b-left',f'{scheme} actual remote unique G{g}',None,m['unique_receive']['mean'],scope(e))
  ax[0].plot(gs,np.array(paper)/1000,'--',color=color,label=f'Paper {scheme}');ax[0].plot(gs,np.array(meas)/1000,'o-',color=color,label=f'Measured total {scheme}')
 local_ref=refs['paper_text_targets']['fig12_locality'];ax[1].plot(gs,[100*local_ref[str(g)] for g in gs],'k--',label='Paper CPS')
 for key,label in [('unique_local_hit_rate','Measured CPS unique denominator'),('local_hit_rate','Measured CPS raw denominator')]:
  vals=[]
  for g in gs:
   e=cases['criteo',g];mv=100*e['c'][key];vals.append(mv);add('12','b-right',f'{key} G{g}',100*local_ref[str(g)],mv,scope(e),'higher',note='Both denominators shown; do not choose whichever matches better')
  ax[1].plot(gs,vals,'o-',label=label)
 for key,label in [('unique_local_hit_rate','Selected improved: unique'),('local_hit_rate','Selected improved: raw')]:
  vals=[]
  for g in gs:
   e=cases['criteo',g];v=best(e);mv=100*v['c'][key];vals.append(mv)
   if v is not e:add('12','b-right',f'{key} G{g} improved',100*local_ref[str(g)],mv,scope(v),'higher',note='Same candidate as Figure11; both locality denominators retained')
  ax[1].plot(gs,vals,'s:',label=label)
 totals=[]
 for g in gs:
  e=cases['criteo',g];v=best(e);mv=sum(t['unique_count_mean'] for t in v['c']['per_table']);totals.append(mv)
  if v is not e:
   add('12','b-left',f'cps total unique G{g} improved',cps_ref[g],mv,scope(v),'descriptive')
   add('12','b-left',f'cps actual remote unique G{g} improved',None,v['c']['unique_receive']['mean'],scope(v))
 ax[0].plot(gs,np.array(totals)/1000,'s:',color='black',label='Selected improved CPS total')
 ax[1].plot(gs,[100/g for g in gs],'s:',label='Hash 1/G');ax[0].set_ylabel('1000 unique IDs/rank/batch');ax[1].set_ylabel('Locality (%)')
 for axis in ax:axis.set_xscale('log',base=2);axis.set_xticks(gs,gs);axis.set_xlabel('Logical ranks');axis.legend(fontsize=7);axis.grid(alpha=.2)
 fig.suptitle('Figure12(b); AUC panel (a) excluded: requires actual training')
 save(fig,'Figure12b')
# Table3: paper, measured raw, measured unique, identical nine-cell order.
ab=ROOT/'results/criteo_day6_32_ablation_16b/metrics.json'
if not ab.exists():ab=ROOT/'results/criteo_32_ablation_4b/metrics.json'
if ab.exists():
 d=json.loads(ab.read_text());keys=[f'{p}__{s}' for p in ['hash','freq_rr','ours'] for s in ['sequential','greedy_repair','ours']]
 paper=np.array(refs['paper_text_targets']['table3_raw_or_unique_unspecified']);raw=np.array([100*d[k]['local_hit_rate'] for k in keys]).reshape(3,3);uniq=np.array([100*d[k]['unique_local_hit_rate'] for k in keys]).reshape(3,3)
 fig,axes=plt.subplots(1,3,figsize=(12,4))
 for ax,values,title in zip(axes,[paper,raw,uniq],['Paper','Measured raw requests','Measured unique IDs']):
  ax.imshow(values,vmin=0,vmax=max(70,uniq.max()),cmap='Blues');ax.set_xticks(range(3),['Sequential','Greedy+Repair','Ours'],rotation=15);ax.set_yticks(range(3),['Hash','FreqRR','Ours placement']);ax.set_title(title)
  for i in range(3):
   for j in range(3):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]>40 else 'black')
 for k,pv,rv,uv in zip(keys,paper.ravel(),raw.ravel(),uniq.ravel()):
  add('Table3','raw',k,float(pv),rv,ab.parent.name,'higher',tolerance=.005,note='Paper rounded to0.01 percentage points');add('Table3','unique',k,float(pv),uv,ab.parent.name,'higher',tolerance=.005,note='Paper rounded to0.01 percentage points')
 fig.suptitle(f'Table3 locality (%) — {ab.parent.name}');save(fig,'Table3')
fig,axes=plt.subplots(2,3,figsize=(15,8))
for col,ds in enumerate(['criteo','kaggle','taobao']):
 gs=[g for g in [8,16,32,64,128] if (ds,g) in cases]
 for row,stat in enumerate(['mean_batch_peak_over_mean','max_batch_peak_over_mean']):
  ax=axes[row,col]
  for key,color in [('acp_receive','C0'),('acp_send','C1'),('acp_send_plus_receive','C2')]:
   vals=[];opts=[]
   for g in gs:
    e=cases[ds,g];v=best(e);value=e['c'][key][stat];opt=v['c'][key][stat];vals.append(value);opts.append(opt)
    add('Section7.8',ds,f'{key} {stat} G{g}',1.2,value,scope(e),'lower',note='Values below1.10 are better, not failures. Worst-batch<=1.20 is an additional conservative check; paper does not specify a worst-batch guarantee.')
    if v is not e:add('Section7.8',ds,f'{key} {stat} G{g} improved',1.2,opt,scope(v),'lower')
   ax.plot(gs,vals,'o-',color=color,label=key.replace('acp_','')+' baseline')
   if not np.array_equal(vals,opts):ax.plot(gs,opts,'s:',color=color,label=key.replace('acp_','')+' improved')
  ax.axhline(1.2,color='red',ls='--',label='Paper upper bound1.20');ax.set_title(ds+(' batch-average ratio' if row==0 else ' worst-batch ratio'));ax.set_xscale('log',base=2);ax.set_xticks(gs,gs);ax.set_yscale('log');ax.set_xlabel('Logical ranks');ax.set_ylabel('max(rank) / mean(rank)');ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.suptitle('Section7.8: actual ACP receive, send and sum; direction is not explicit in the paper');save(fig,'LoadBalance')
storage=json.loads((ROOT/'storage_balance.json').read_text())
def storage_of(e):
 path=e['path'];seen=set()
 while path.name not in storage:
  if path.name in seen:return None
  seen.add(path.name);cfg=json.loads((path/'config.json').read_text())
  if not cfg.get('base_case'):return None
  path=Path(cfg['base_case'])
 return storage[path.name]
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,ds in zip(axes,['criteo','kaggle','taobao']):
 gs=[g for g in [8,16,32,64,128] if (ds,g) in cases];x=np.arange(len(gs));before=[];after=[]
 for g in gs:
  e=cases[ds,g];v=best(e);a=storage_of(e);b=storage_of(v);before.append(a['max_over_mean']);after.append(b['max_over_mean'])
  add('Section7.8-storage',ds,f'exported ID count max/mean G{g}',None,a['max_over_mean'],scope(e),note='Observed embedding rows, not allocated GPU bytes; paper gives no numerical storage bound')
  if v is not e:add('Section7.8-storage',ds,f'exported ID count max/mean G{g} improved',None,b['max_over_mean'],scope(v),note='Inherited placement for scheduling-only variants; METIS may change storage balance')
 ax.bar(x-.18,before,.36,label='Baseline');ax.bar(x+.18,after,.36,label='Selected result');ax.axhline(1,color='black',ls='--',label='Ideal equal row counts');ax.set_xticks(x,gs);ax.set_xlabel('Logical ranks');ax.set_title(ds);ax.legend(fontsize=7);ax.set_ylim(bottom=0.95)
axes[0].set_ylabel('Largest exported row count / average')
fig.suptitle('Storage balance: exported IDs; not an allocated GPU memory measurement');save(fig,'StorageBalance')
for ds,relative,gb,dense,tables,emb,sparse in [
 ('criteo','data/processed/criteo/day6.bin',169,13,26,32,214),
 ('kaggle','data/processed/kaggle/multihot.bin',39,13,26,32,214),
 ('taobao','data/processed/taobao/taobao_4tables.bin',2,1,4,128,4)]:
 p=ROOT.parent/relative
 if p.exists():
  add('Table1',ds,'processed size GB (decimal)',gb,p.stat().st_size/1e9,relative,note='Paper Size unit/file format unspecified; processed four-table Taobao differs from full fifteen-table file')
  add('Table1',ds,'processed size GiB (binary)',gb,p.stat().st_size/2**30,relative,note='Alternative unit interpretation, not a separate data size')
 for label,value in [('dense features',dense),('embedding tables',tables),('embedding dimension',emb),('sparse IDs per sample',sparse)]:add('Table1',ds,label,value,value,relative,note='Matches code configuration; not a trained-model measurement')
full_taobao=ROOT.parent/'data/processed/taobao/taobao_dataset_shuffled.bin'
if full_taobao.exists():
 add('Table1','taobao','full 15-feature preprocessing size GB',2,full_taobao.stat().st_size/1e9,str(full_taobao.relative_to(ROOT.parent)),note='Intermediate full-feature file before selecting four model tables; paper file format unspecified')
for ds in ['criteo','kaggle']:
 for g in [8,16,32,64]:
  if (ds,g) not in cases:continue
  e=cases[ds,g];expected=({8:2,16:1,32:0,64:0} if ds=='criteo' else {8:1,16:1,32:1,64:0})[g];actual=sum(t['strategy']=='Reduction' for t in e['h']['per_table']);add('Table2',ds,f'Reduction count G{g}',expected,actual,scope(e))
with (OUT/'all_numeric_comparisons.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(OUT/'all_numeric_comparisons.json').write_text(json.dumps(records,indent=2))
lines=['逐图对照索引','MainResults.png/pdf：15个配置的论文、基线、改进及均衡性总表。','图1：训练工作流示意图，无实测数字。','图2：端到端延时分解，按用户要求排除。','图3：Unique/Reduction 工作流示意图，无实测数字。','图4(a)：示意图；图4(b)(c)：Figure4.png/pdf。论文图4的非空peer Reduction与公式(5)/图11冲突；实测按远程接收B*(G-1)重算。','图5：整体架构示意图，无实测数字。','图6(a)(b)(c)：Figure6.png/pdf。频率/共现是数据特征，不把“更大”简单判为算法更好；MCR 仅抽样估计。','图7：CPS 工作流示意图，无实测数字。','图8：端到端吞吐，按用户要求排除。','图9：训练迭代时间分解，按用户要求排除。','图10：Figure10.png/pdf；沿论文绘图脚本用接收峰值计算有效通信缩减率，包含Reduction时不等于纯Unique去重率。','图11：Figure11.png/pdf；完整 3 数据集、5 rank 数、Reduction/Unique/ACP/CPS 曲线；Figure11Deviation另展示相对论文偏差。','图12(a)：AUC 收敛需实际训练，未测；图12(b)：Figure12b.png/pdf，左右两子图均对比。','Section7.8：LoadBalance.png/pdf；接收与发送、均值与最坏批次分别与1.20上限对比。','StorageBalance.png/pdf：补充比较分区导出ID数的均衡性，原文未给具体数值上限；不是GPU实际分配内存测量。','表1：all_numeric_comparisons.csv 中 Table1 行；预处理大小同时给出GB/GiB，模型维度是代码配置而非已训练模型。','表2：all_numeric_comparisons.csv 中 Table2 行；策略数量精确对比。','表3：Table3.png/pdf；分别给出两种分母的九格消融。','', '判定：通信量越低越好；locality/有效去重率越高越好。优于论文保留，不为了数值相等调差。数据分布统计与算法收益分开判断。','改进曲线选择规则：在相同样本的后续实验中（分区初始化变体会改变placement，并单独记录），优先取最坏批次接收max/mean<=1.20的候选，再选接收峰值最低者；每个batch的1.20上限是额外保守筛选，原文未明确保证最坏batch；发送侧未达标仍如实列出。所有其他尝试保留于results_summary.csv与REPORT.txt。', '图12的unique数乘非本地比例与图11的CPS值几乎重合，存在均值/峰值口径疑点；paper_internal_consistency.json给出算式。图10/11加灰色均值曲线供核对，但严格评判仍用峰值。','原文中取整标注的参考值保留其精度；曲线参考值来自随附绘图脚本常数，绝不进入实测计算。','所有数值逐项列于 all_numeric_comparisons.csv/json，包含输入范围、偏差、方向与结论。','每图使用的实测 case 以 CSV scope 为准，完整日数据结果完成后自动优先替换 pilot。']
for r in records:
 if r['direction']!='descriptive':lines.append(f"图{r['figure']} {r['panel']} {r['metric']}: 论文={r['paper']}, 实测={r['measured']:.6f}; {r['status']} [{r['scope']}]")
(OUT/'INDEX.txt').write_text('\n'.join(lines)+'\n')
print('Figure comparisons:',len(records),'numeric rows')
