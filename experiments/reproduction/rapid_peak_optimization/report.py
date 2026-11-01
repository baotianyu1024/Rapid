import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;R=ROOT.parent

def summarize(fr,fs,br,bs,full,rows,g):
 processed=rows*5;equiv=processed/(8192*g)
 arrays=dict(forward_receive=fr,forward_send=fs,backward_receive=br,backward_send=bs,combined_receive=fr+br,combined_send=fs+bs)
 result={k:dict(full_mean_peak=float(x[full].max(1).mean()),all_sample_normalized_peak=float(x.max(1).sum()/equiv),absolute_peak=int(x.max())) for k,x in arrays.items()}
 phase=np.maximum(fr.max(1),fs.max(1))+np.maximum(br.max(1),bs.max(1))
 result['phase_bottleneck']=dict(full_mean_peak=float(phase[full].mean()),all_sample_normalized_peak=float(phase.sum()/equiv))
 result['total_per_8192']=float((fr.sum()+br.sum())/(processed/8192));result['total_vectors']=int(fr.sum()+br.sum());result['full_steps']=int(full.sum());result['all_steps']=len(fr);first=full.copy();first[len(fr)//5:]=False;result['cold_epoch_combined_receive_peak']=float((fr+br)[first].max(1).mean());result['cold_epoch_phase_bottleneck']=float(phase[first].mean());result['cold_epoch_total_per_8192']=float((fr[:len(fr)//5].sum()+br[:len(fr)//5].sum())/(rows/8192))
 return result

def main():
 report={}
 for g in [8,64]:
  latest=(R/'herald_rapid_gmp/results/default8/latest_rapid') if g==8 else (R/'het_gmp_rapid/results/rank64_balance/latest_rapid')
  traf=(R/'herald_rapid_gmp/results/default8/fresh_baselines/traffic') if g==8 else (R/'het_gmp_rapid/results/rank64_balance/traffic')
  gm=json.load(open(traf/'het_gmp_s0.json'));z=np.load(traf/'het_gmp_s0.npz');f=z['forward'];b=z['backward'];full=np.tile(np.arange(gm['steps_per_epoch'])<min(gm['rank_rows'])//8192,5)
  methods={'GMP':summarize(f.sum(2),f.sum(1),b.sum(1),b.sum(2),full,gm['rows'],g)}
  for name,path in [('RAPID_original',latest/'rapid_guard_p5')]+[(x.name,x) for x in sorted(ROOT.glob(f'r{g}_*')) if x.is_dir() and (x/'predicted_counts.npz').exists()]:
   cfg=json.load(open(path/'config.json'))
   if cfg['rows']!=gm['rows']:continue
   file=path/'predicted_counts.npz';z=np.load(file);r=np.tile(z['receive'].sum(1),(5,1));s=np.tile(z['send'].sum(1),(5,1));methods[name]=summarize(r,s,s,r,np.ones(len(r),bool),gm['rows'],g);methods[name]['verified_actual_trace']=(path/'complete.json').exists() and bool(json.load(open(path/'complete.json')).get('independent_counts_verified',False))
  for path in sorted(ROOT.glob(f'r{g}_*cache')):
   if not (path/'result.json').exists():continue
   z=np.load(path/'counts.npz');methods[path.name]=summarize(z['fr'],z['fs'],z['br'],z['bs'],np.ones(len(z['fr']),bool),gm['rows'],g)
   m=json.load(open(path/'result.json'));methods[path.name]['mirrors']=sum(m['mirror_counts']);methods[path.name]['verified_actual_trace']=(path/'verification.json').exists() and bool(json.load(open(path/'verification.json')).get('success',False))
  report[str(g)]=methods
 (ROOT/'comparison.json').write_text(json.dumps(report,indent=2))
 for g,methods in report.items():
  print(g,'rank: name,total/8192,combined_recv_peak,combined_send_peak,phase_bottleneck,all_rows_combined_recv_peak')
  for k,v in methods.items():print(k,*[round(x,2) for x in [v['total_per_8192'],v['combined_receive']['full_mean_peak'],v['combined_send']['full_mean_peak'],v['phase_bottleneck']['full_mean_peak'],v['combined_receive']['all_sample_normalized_peak']]])
if __name__=='__main__':main()
