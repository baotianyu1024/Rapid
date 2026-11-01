"""Resume after the all-Unique guard; charge the fresh profile-building attempt."""
import json,os,sys,time,hashlib
from pathlib import Path
from common import measured,save
EXP=Path(__file__).resolve().parents[1];REP=EXP.parent;OUT=EXP/'results/kaggle8_fast';S=REP/'scripts';PY=sys.executable
os.environ['LD_LIBRARY_PATH']=str(REP/'toolchain/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
for suffix in ['status.json','command.json','time','log']:(OUT/('rapid_price_f2.'+suffix)).rename(OUT/('rapid_profile_build_failed.'+suffix))
(OUT/'rapid_price_f2').rename(OUT/'rapid_price_f2_failed')
save(OUT/'acp_resume_sources.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),S/'joint_price_owners.py',S/'refine_balanced_owners.py',EXP/'scripts/report_dataset.py']})
base=OUT/'rapid_owner_base';previous=None
for suffix in ['price_f2','price_b8','price_b32','guard_p5']:
 params=json.loads((EXP/'results/timing8_rerun'/('rapid_'+suffix)/'refinement.json').read_text())['parameters'];dest=OUT/('rapid_'+suffix)
 cmd=[PY,S/'joint_price_owners.py','--case',base,'--cache',OUT/'rapid_profile_cache','--output',dest,'--allow-reduction-tables']
 for key in ['forward','backward','storage','penalty','budget','passes','hot_threshold']:
  if key in params:cmd+=['--'+key.replace('_','-'),params[key]]
 if previous:cmd+=['--resume-from',previous]
 if suffix=='guard_p5':cmd+=['--guard-case',previous]
 measured('rapid_'+suffix,cmd,OUT);previous=dest
measured('rapid_final_materialize',[PY,S/'refine_balanced_owners.py','--case',base,'--cache',OUT/'rapid_profile_cache','--output',previous,'--materialize-existing','--uncompressed-maps'],OUT)
save(OUT/'preprocess_complete.json',dict(success=True,completed=time.time()))
for name,cmd in [('gmp_replay',[PY,EXP/'scripts/replay_het.py','--data',OUT/'data','--partition',OUT/'gmp','--out',OUT/'traffic','--ranks',8,'--epochs',5,'--bounds',0,'--skip-1d']),('verify_outputs',[PY,EXP/'scripts/verify_dataset.py','--case',OUT]),('report',[PY,EXP/'scripts/report_dataset.py','--case',OUT])]:measured(name,cmd,OUT)
print('ALL COMPLETE',flush=True)
