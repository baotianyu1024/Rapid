"""Publish only after complete, audited official runs exist."""
import json,os,subprocess,time
from pathlib import Path
O=Path(__file__).resolve().parents[1];V=O.parent;P='/home/longjiu.bty/miniconda3/envs/het-gmp-rapid/bin/python';done=set()
while len(done)<3:
 for ds in ['t180','t200','t510']:
  if ds in done:continue
  c=O/f'results/{ds}_8'
  needed=['complete.json','optimized/peak_samples/cps_counts.npz','herald/result.json','bagpipe_la8/steps.json','torchrec/plan.json','gmp_bytes/complete.json']
  if all((c/f).exists() for f in needed):
   subprocess.run([P,str(O/'code/report.py'),'--dataset',ds],check=True);done.add(ds)
 if len(done)<3:time.sleep(10)
for script in ['finalize_tables.py','build_response.py','record_sources.py','seal_evidence.py','validate_deliverables.py']:
 subprocess.run([P,str(V/'code'/script)],check=True)
result=dict(success=True,datasets=sorted(done),rows_per_dataset=65536,official_full_dataset_rows=6553600,methods=['RAPID','HET-GMP','Herald','TorchRec','BagPipe'],dimensions='main uniform128; supplemental official per-table',epochs=5,ranks=8,local_batch=512,source_commit='61b4d539aa7044e9021896f0485be1a6697f4f44',historical_failure='T510 file limit raised to65536; T180 samples16 intentionally restarted with16 independent workers; both resolved',finished=time.time())
(O/'complete.json').write_text(json.dumps(result,indent=2));print('ALL OFFICIAL RUNS AUDITED AND PUBLISHED',flush=True)
