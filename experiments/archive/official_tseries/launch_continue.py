import json,os,resource,sys,time
from pathlib import Path
O=Path(__file__).resolve().parents[1];V=O.parent;sys.path.insert(0,str(V/'code/reproduction/het_gmp_rapid/scripts'));from common import measured,save
soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard),hard));os.environ['LD_LIBRARY_PATH']='/mnt/hugui/longjiu.bty/reproduction/toolchain/lib';case=O/'results/t180_8'
while not (case/'optimized.status.json').exists():time.sleep(1)
assert json.loads((case/'optimized.status.json').read_text())['exit_code']!=0
for suffix in ['log','status.json','command.json','time']:
 f=case/f'optimized.{suffix}'
 if f.exists():f.rename(case/f'interrupted_workers8_optimized.{suffix}')
measured('optimized',[sys.executable,O/'code/continue_optimized.py','--dataset','t180','--stage','samples16','--workers','16'],case,threads=8)
save(case/'complete.json',dict(success=True,finished=time.time(),runtime_parallelism_only=True,resumed_stage='samples16'))
