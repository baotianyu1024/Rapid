"""Collect all owner trials and GMP full-batch / equal-sample traffic comparisons."""
import argparse,json
from pathlib import Path
import numpy as np
from measure import summarize

def main():
 p=argparse.ArgumentParser();p.add_argument('--selected8',required=True);p.add_argument('--selected32',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];out=Path(a.output).resolve();out.mkdir(exist_ok=True)
 result={};lines=['Joint RAPID owner optimization: 8-rank search followed by 32-rank validation','',
 'Unit: transmitted embedding / gradient vectors. No byte conversion.',
 'Dataset: the same 4,194,304 Criteo day6 samples; local batch 8192.',
 'RAPID: exact 8192 samples/rank/step, no mirrors, synchronous Unique traffic.',
 'GMP: s=0, original graph sample assignment and 1% remote-candidate mirrors.',
 'GMP complete batches only for peak/mean diagnostics; ALL ragged steps retained in equal-sample totals.',
 'GMP replay spans 5 epochs with cache reuse; RAPID static trace is repeated 5 times for totals.',
 'These are CPU offline communication calculations, not GPU end-to-end or accuracy experiments.','']
 for g,selected,gmpname,base in [(8,a.selected8,'main64','criteo_day6_8_owner_target7_f10_64b'),(32,a.selected32,'rank32_best','criteo_day6_32_owner_balanced_16b')]:
  cases=[root/'results'/base,Path(selected).resolve()]
  extra=sorted((root/'results').glob(f'criteo_day6_{g}_price_*'))+sorted((root/'results').glob(f'criteo_day6_{g}_joint_owner_trial'))+sorted((root/'results').glob(f'criteo_day6_{g}_reschedule_trial'))
  if g==32:extra.append(root/'results/criteo_day6_32_joint_balance_v7')
  cases=list(dict.fromkeys(cases+extra));trials={}
  for case in cases:
   f=case/'cps_counts.npz' if (case/'cps_counts.npz').exists() else case/'predicted_counts.npz'
   if not f.exists():continue
   counts=np.load(f);entry=dict(path=str(case),forward=summarize(counts['receive'].sum(1)),backward=summarize(counts['send'].sum(1)),total_vectors_5epochs=int(5*(counts['receive'].sum()+counts['send'].sum())),independent_dataset_scan=(case/'cps_counts.npz').exists())
   if (case/'refinement.json').exists():
    j=json.loads((case/'refinement.json').read_text());s=np.array(j['after_storage']);entry.update(storage=dict(counts=s.tolist(),mean=float(s.mean()),max_over_mean=float(s.max()/s.mean()),min_over_mean=float(s.min()/s.mean())),parameters=j['parameters'],optimization_seconds=j['optimization_seconds'])
   trials[case.name]=entry
  # Use the winner's input storage audit for the baseline.
  win=Path(selected).resolve();verification=json.loads((win/'complete.json').read_text())
  assert verification['independent_counts_verified'] and verification['all_records_inverse_verified']
  j=json.loads((win/'refinement.json').read_text());s=np.array(j['before_storage']);trials[base]['storage']=dict(counts=s.tolist(),mean=float(s.mean()),max_over_mean=float(s.max()/s.mean()),min_over_mean=float(s.min()/s.mean()))
  path=root/'het_gmp_rapid/results'/gmpname/'traffic/het_gmp_s0.npz';arrays=np.load(path);meta=json.loads(path.with_suffix('.json').read_text())
  full=min(meta['rank_rows'])//8192;mask=np.arange(meta['steps'])%meta['steps_per_epoch']<full
  primary=np.array(meta['primary_counts']);mirrors=np.array(meta['mirror_counts']);resident=primary+mirrors
  gmp=dict(forward=summarize(arrays['forward'][mask].sum(2)),backward=summarize(arrays['backward'][mask].sum(1)),full_steps=int(mask.sum()),all_steps=meta['steps'],total_vectors_5epochs=int(arrays['forward'].sum()+arrays['backward'].sum()),primary_storage=dict(counts=primary.tolist(),max_over_mean=float(primary.max()/primary.mean())),mirrors=mirrors.tolist(),resident_max_over_mean=float(resident.max()/resident.mean()))
  result[g]=dict(selected=win.name,trials=trials,gmp=gmp)
  lines.extend([f'{g} ranks:', 'case | forward mean | forward mean step max | backward mean | backward mean step max | worst F ratio | worst B ratio | owner max/mean | total change vs baseline'])
  base_total=trials[base]['total_vectors_5epochs']
  for name,e in trials.items():
   f,b=e['forward'],e['backward'];s=e.get('storage',{}).get('max_over_mean',float('nan'))
   lines.append(f"{name} | {f['mean']:.3f} | {f['mean_batch_peak']:.3f} | {b['mean']:.3f} | {b['mean_batch_peak']:.3f} | {f['max_batch_peak_over_mean']:.6f} | {b['max_batch_peak_over_mean']:.6f} | {s:.6f} | {100*(e['total_vectors_5epochs']/base_total-1):+.3f}%")
  f,b=gmp['forward'],gmp['backward'];lines.append(f"GMP full batches | {f['mean']:.3f} | {f['mean_batch_peak']:.3f} | {b['mean']:.3f} | {b['mean_batch_peak']:.3f} | {f['max_batch_peak_over_mean']:.6f} | {b['max_batch_peak_over_mean']:.6f} | {gmp['primary_storage']['max_over_mean']:.6f} | {100*(gmp['total_vectors_5epochs']/base_total-1):+.3f}%")
  lines.extend([f'Selected: {win}',f'GMP full/all steps: {gmp["full_steps"]}/{gmp["all_steps"]}; total mirrors: {mirrors.sum()}',f'Equal-sample totals (all 5 epochs): baseline {base_total}, selected {trials[win.name]["total_vectors_5epochs"]}, GMP {gmp["total_vectors_5epochs"]}',''])
 lines.extend(['Method: squared per-batch forward/backward congestion + squared owner cardinality imbalance + total communication penalty.',
 'Fast aggregate proposals for cold IDs; batch-exact owner proposals for hot IDs. Every accepted move decreases the exact joint objective.',
 'Hard per-batch total-communication budget. Budget is an upper bound, not a target; absolute counts are always reported.',
 'Final winners are materialized and independently rescanned. Every record is restored through the inverse ID map and compared with its source.',
 'The selected method changes only embedding ownership/remapping, preserving samples, rank assignments, batch boundaries and labels.',
 'No replica, staleness, compression, dropped sample or changed numerical training rule is introduced. GPU accuracy/throughput remain unmeasured.',
 'All optimization and validation use this same offline trace. No held-out or theoretical optimality claim.',
 'Rejected alternative: max-owner-hit sample rescheduling worsens forward peak despite improving backward load.',
 'Fixed sample ownership can impose a locality/balance tradeoff; further joint sample-embedding graph refinement remains a separate direction.'])
 (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');(out/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
