import subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent;log=root/'stream64_final_prefix.log'
while 'CHECKPOINT STOP 4' not in log.read_text():time.sleep(1)
out=root/'stream64_control';subprocess.run([sys.executable,str(root/'clone_stream.py'),str(root/'stream64_final_prefix'),str(out)],check=True)
with open(root/'stream64_control.log','w') as f:subprocess.run([sys.executable,str(root/'run_stream_fast.py'),'--ranks','64','--rows','2097152','--windows','20','--seed-schedule','grouped','--refit-new','--resume','--out',str(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
