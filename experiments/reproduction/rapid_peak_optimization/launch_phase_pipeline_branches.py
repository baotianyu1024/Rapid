"""Launch matched corrected streams only after the immutable prefix exits cleanly."""
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--ranks',type=int,required=True);a=p.parse_args();root=Path(__file__).resolve().parent;g=a.ranks;prefix=root/f'stream{g}_phase_prefix';jobs=[]
while True:
 log=prefix.with_suffix('.log').read_text() if prefix.with_suffix('.log').exists() else ''
 if 'Traceback' in log:raise RuntimeError('Prefix failed; inspect '+str(prefix))
 if 'CHECKPOINT STOP 4' in log and (root/'pipeline_ab_verified.json').exists():break
 time.sleep(10)
sources=['run_stream_phase.py','prepare_stream_owner.py','run_stream_phase_pipeline.py','stream_window.py','stream_alternate.py','sample_group_peak_phase.cpp','sample_group_peak_phase.so','group_scheduler.py','stream_tune.py','stream_swaps.so','persistent_fast.py','persistent_oracle_sparse.so','stream_owner_state.py']
snapshot={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in sources}
(root/f'phase_recipe_{g}.json').write_text(json.dumps(snapshot,indent=2))
for suffix,interval in [('control',0),('i4_f001',4)]+([('i8_f001',8)] if g==64 else []):
 out=root/f'stream{g}_phase_{suffix}';subprocess.run([sys.executable,str(root/'clone_stream.py'),str(prefix),str(out)],check=True)
 cmd=[sys.executable,str(root/'run_stream_phase_pipeline.py'),'--ranks',str(g),'--rows','2097152','--windows',str(20 if g==64 else 10),'--seed-schedule','grouped','--refit-new','--interval',str(interval),'--fraction','.001','--resume','--out',str(out)]
 with open(out.with_suffix('.log'),'w') as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 jobs.append((proc,out));print('STARTED',proc.pid,out,flush=True)
(root/f'phase_jobs_{g}.json').write_text(json.dumps([dict(pid=p.pid,out=str(o)) for p,o in jobs],indent=2))
for proc,out in jobs:
 code=proc.wait();print('FINISHED',code,out,flush=True)
 if code:raise RuntimeError(str(out)+' failed')
subprocess.run([sys.executable,str(root/'compare_streams.py'),'--ranks',str(g),'--phase'],check=True)
