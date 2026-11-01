import subprocess,sys,time,json
from pathlib import Path
root=Path(__file__).resolve().parent;prefix=root/'stream8_final_prefix';log=root/'stream8_final_prefix.log'
while 'CHECKPOINT STOP 4' not in log.read_text():time.sleep(1)
jobs=[]
for name,interval,fraction in [('stream8_control',0,0),('stream8_i4_f0001',4,.0001),('stream8_i4_f001',4,.001),('stream8_i4_f01',4,.01)]:
 out=root/name;subprocess.run([sys.executable,str(root/'clone_stream.py'),str(prefix),str(out)],check=True);f=open(root/(name+'.log'),'w');cmd=[sys.executable,str(root/'run_stream_fast.py'),'--ranks','8','--rows','2097152','--windows','10','--seed-schedule','grouped','--refit-new','--resume','--interval',str(interval),'--fraction',str(fraction),'--out',str(out)];p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);jobs.append((name,p,f));print('START',name,p.pid,flush=True)
results={}
for name,p,f in jobs:results[name]=p.wait();f.close();print('FINISHED',name,results[name],flush=True)
(root/'screen8_jobs.json').write_text(json.dumps(results,indent=2));assert all(v==0 for v in results.values())
