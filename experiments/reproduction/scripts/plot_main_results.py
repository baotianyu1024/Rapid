"""A compact 15-configuration results page before the full figure ledger."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];rows=list(csv.DictReader((R/'final_selected_results.csv').open()));cells=[];colors=[]
for r in rows:
 nums=[float(r[k]) for k in ['paper_peak','baseline_peak','selected_peak','minimum_traffic_peak','receive_worst_ratio','minimum_traffic_receive_worst_ratio','send_worst_ratio']]
 cells.append([r['dataset'],r['ranks'],*[f'{v/1000:.2f}' for v in nums[:4]],*[f'{v:.3f}' for v in nums[4:]]])
 cc=['white']*9
 for idx,val in [(4,nums[2]),(5,nums[3])]:cc[idx]='#e3f2e5' if val<=nums[0] else '#ffe5e2'
 for idx,val in [(6,nums[4]),(7,nums[5]),(8,nums[6])]:cc[idx]='#e3f2e5' if val<=1.200001 else '#fff0d9'
 colors.append(cc)
fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off')
fig.suptitle('Main communication results: paper, baseline and improvements',fontsize=15,y=.95)
fig.text(.05,.89,'Traffic columns: mean of per-batch maximum receive vectors, in thousands (K).\nBalanced pick prioritizes worst-batch RX/mean <=1.20 where feasible, then minimizes traffic.\nLowest traffic has no balance filter. The paper does not specify a worst-batch guarantee.',fontsize=9,va='top')
t=ax.table(cellText=cells,colLabels=['Dataset','Ranks','Paper K','Baseline K','Balanced\npick K','Lowest\ntraffic K','RX worst\nbalanced','RX worst\nlowest','TX worst\nbalanced'],cellColours=colors,loc='center',colWidths=[.10,.06,.10,.11,.11,.11,.11,.11,.11]);t.auto_set_font_size(False);t.set_fontsize(8);t.scale(1,1.85)
fig.text(.05,.12,'Green: at/below traffic reference or ratio1.20. Red: traffic above reference. Orange: ratio above1.20.\nRX=receive, TX=send. Receive balance is not evidence of send or total network balance.\nExact values, case identities, rows and locality: final_selected_results.csv. All attempts: results_summary.csv.\nCPU replay; forward vectors only; no GPU runtime, backward traffic or ID metadata measured.',fontsize=9,va='top')
for ext in ['png','pdf']:fig.savefig(R/'figure_comparison'/f'MainResults.{ext}',dpi=180)
