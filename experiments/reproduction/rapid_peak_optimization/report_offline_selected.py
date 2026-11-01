import json,csv
from pathlib import Path
root=Path(__file__).resolve().parent;selected=json.load(open(root/'offline_selection.json'));fixed=json.load(open(root/'gmp_fixed_batch_summary.json'));rows=[]
for g in [8,64]:
 bal=json.load(open(root/f'r{g}_peak_soft_cache/balance.json'));s=selected['selections'][str(g)];base=root.parent/('herald_rapid_gmp/results/default8/fresh_baselines/traffic' if g==8 else 'het_gmp_rapid/results/rank64_balance/traffic');gm=json.load(open(base/'het_gmp_s0.json'));original=dict(rapid=(bal['rapid_original']['traffic']['combined_receive']['rank_mean_balance']['mean'],bal['rapid_original']['traffic']['combined_receive']['mean_step_peak']),gmp=((gm['forward_receive_vectors']['total']+gm['backward_receive_vectors']['total'])/gm['processed_rows']*8192,bal['gmp_original_full_steps']['traffic']['combined_receive']['mean_step_peak']))
 for method,key,vol,peak in [('Original RAPID','rapid_original',*original['rapid']),('Original GMP reference','gmp_original_full_steps',*original['gmp']),('GMP with original-batch capacity repair','gmp_fixed_batch',fixed[str(g)]['total_vectors_per_8192'],fixed[str(g)]['peak_combined_receive']),('Optimized RAPID + strict-s0 mirrors','optimized',s['metrics']['total_per_8192'],s['metrics']['combined_receive']['full_mean_peak'])]:
  b=bal[key];rows.append(dict(ranks=g,method=method,mean_rank_vectors=vol,mean_step_peak_vectors=peak,communication_rank_mean_cv_percent=100*b['traffic']['combined_receive']['rank_mean_balance']['cv'],primary_rows_min=b['primary']['min'],primary_rows_max=b['primary']['max']))
out=root/'final_incremental_report';out.mkdir(exist_ok=True);(out/'offline.json').write_text(json.dumps(rows,indent=2))
with open(out/'offline.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
