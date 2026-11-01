import subprocess,sys,time,json
from pathlib import Path
root=Path(__file__).resolve().parent
names=[f'stream{g}_phase_{s}' for g in [8,64] for s in (['control','i4_f001'] if g==8 else ['control','i4_f001','i8_f001'])];pending=set(names);jobs={};done=set()
while pending or jobs:
 for name in sorted(pending.copy()):
  exp=root/name
  if (exp/'complete.json').exists():
   metrics=json.load(open(exp/'metrics.json'));verification=exp/f"window_{metrics[-1]['window']:02d}/raw_verification.json"
   if verification.exists() and json.load(open(verification)).get('success'):
    pending.remove(name);done.add(name);print('ALREADY VERIFIED',name,flush=True);continue
   with open(root/(name+'_raw_verify.log'),'w') as log:jobs[name]=subprocess.Popen([sys.executable,str(root/'verify_stream_raw.py'),str(exp)],stdout=log,stderr=subprocess.STDOUT)
   pending.remove(name);print('START',name,jobs[name].pid,flush=True)
 for name,job in list(jobs.items()):
  code=job.poll()
  if code is not None:
   if code:raise RuntimeError('Raw verification failed: '+name)
   done.add(name);del jobs[name];print('VERIFIED',name,flush=True)
 time.sleep(10)
subprocess.run(['/home/longjiu.bty/miniconda3/bin/python',str(root/'report_phase_streams.py')],check=True)
(root/'final_incremental_report/verification_complete.json').write_text(json.dumps(dict(all_independent_raw_replays_passed=True,experiments=sorted(done)),indent=2))
