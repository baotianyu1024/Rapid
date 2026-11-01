from pathlib import Path
import json,sys
import numpy as np
from report_comparisons import rapid
from evidence_utils import save,read
V=Path(__file__).resolve().parents[1];sys.path.insert(0,str(V/'code/reproduction/scripts'));from measure import measure
names=['default','beta0','beta0.001','gamma0','gamma1','topk10000','topk100000','metis1e-05','metis0.0001','metis0.001','orderreverse','order11','order29','batch512','batch1024','batch2048','batch4096','batch16384'];rows=[]
for name in names:
 d=V/'results/sensitivity_full'/name/'optimized/peak_samples';assert (d.parent/'complete.json').exists(),name
 v,audit=rapid(d,1);v['audit']=audit;v['name']=name;v['initial_settings']=read(V/'results/sensitivity'/name/'initial_metrics.json');cfg=read(d/'config.json')
 if name=='default' or name.startswith('batch'):
  m,z=measure([V/'results/sensitivity/input.bin'],8,cfg['batch_size'],cfg['batches'],cfg['table_lengths'],13,False);v['unique_vectors_per_sample']=float(2*z['receive'].sum()/cfg['rows']);v['saving']=1-v['vectors_per_sample']/v['unique_vectors_per_sample'];v['batch']=cfg['batch_size']
 rows.append(v);save(V/'results/sensitivity_full'/name/'result.json',v);print(name,round(v['vectors_per_sample'],5),v['primary_max_mean'],flush=True)
save(V/'results/sensitivity_full/summary.json',dict(rows=262144,ranks=8,dim=128,recipe='complete RAPID default recipe; optional cache excluded',results=rows))
