"""64-rank balance experiment. RAPID defaults exclusively to current working sources."""
import argparse,json,os,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from common import measured,save
from run_latest_rapid64 import validate_manifest
EXP=Path(__file__).resolve().parents[1];REP=EXP.parent;S=EXP/'scripts';PY=sys.executable

def running(script,scope=None):
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  except (FileNotFoundError,PermissionError,UnicodeError):continue
  if len(argv)<2 or 'python' not in Path(argv[0]).name:continue
  if not any(Path(x).name==script for x in argv[1:3]):continue
  if script=='run_latest_rapid64.py' and scope is not None:
   actual=Path(argv[argv.index('--case')+1]).resolve() if '--case' in argv else EXP/'results/rank64_balance'
   if actual==Path(scope):return True
  elif scope is None or any(str(scope)==x or x.startswith(str(scope)+'/') for x in argv):return True
 return False

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=EXP/'results/rank64_balance',help='Fresh experiment directory, or resume same-source results');a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);data=out/'data'
 save(out/'benchmark_config.json',dict(rows=4194304,ranks=64,local_batch=8192,batches=8,epochs=5,rapid='DEFAULT: complete current reproduction/working + reproduction/scripts source snapshot, fresh compilation, fixed full owner refinement chain',gmp='Native 10 rounds, 1% remote-candidate mirrors, s=0',topology='Uniform logical-rank off-diagonal cost .1',timing='Independent algorithm jobs may run concurrently; no speed comparison',traffic='CPU vector trace replay, not GPU/NCCL measurements; full batches for balance, all rows for total volume'))
 def stage(name,cmd,script=None):
  status=out/(name+'.status.json')
  if script:
   while not status.exists() and running(script,out):time.sleep(5)
  if status.exists():
   assert json.loads(status.read_text())['exit_code']==0,f'Previous stage failed: {status}'
   return
  measured(name,cmd,out,threads=16)
 stage('input_and_gmp_adapter',[PY,S/'prepare.py','--input',REP.parent/'data/processed/criteo/day6.bin','--out',data,'--batches',8,'--ranks',64],'prepare.py')
 assert json.loads((data/'input.json').read_text())['input_sha256']=='44ba74ba68f60c13d36d3724cbb7dd95b30d997fc9d489cbd67347dc3fe70317'
 # Existing completed results must match current sources. Never silently fall back
 # to an old placement executable or an old completed RAPID result.
 latest=out/'latest_rapid';manifest=latest/'source_manifest.json'
 if manifest.exists():validate_manifest(manifest)
 def gmp_work():
  stage('gmp_partition',[PY,S/'partition_het.py','--data',data,'--out',out/'gmp','--ranks',64],'partition_het.py')
  stage('gmp_materialize',[PY,S/'materialize_het.py','--data',data,'--partition',out/'gmp','--ranks',64],'materialize_het.py')
 with ThreadPoolExecutor(max_workers=1) as pool:
  gmp_job=pool.submit(gmp_work)
  if not (latest/'complete.json').exists():
   # The current session started the fresh latest workflow separately. Await it
   # when it is active; a new run invokes the identical default workflow below.
   if running('run_latest_rapid64.py',out):
    while running('run_latest_rapid64.py',out) and not (latest/'complete.json').exists():time.sleep(5)
    assert (latest/'complete.json').exists(),'Latest RAPID process stopped before completion'
   else:stage('latest_rapid_pipeline',[PY,S/'run_latest_rapid64.py','--case',out])
  assert json.loads((latest/'complete.json').read_text())['success'];validate_manifest(manifest)
  stage('latest_initial_measure',[PY,S/'baseline64.py','--case',out])
  gmp_job.result()
 save(out/'preprocess_complete.json',dict(success=True,rapid_source='latest_source',completed=time.time()))
 stage('gmp_replay',[PY,S/'replay_het.py','--data',data,'--partition',out/'gmp','--out',out/'traffic','--ranks',64,'--epochs',5,'--bounds',0,'--skip-1d'],'replay_het.py')
 stage('verify_outputs',[PY,S/'verify_rank64.py','--case',out])
 stage('report',[PY,S/'report_rank64.py','--case',out])
 print('64-RANK LATEST RAPID / GMP COMPLETE',flush=True)
if __name__=='__main__':main()
