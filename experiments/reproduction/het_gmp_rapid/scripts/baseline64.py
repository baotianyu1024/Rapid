"""Diagnostic initial stage from the SAME current-source RAPID run; never old baseline."""
from common import *
import numpy as np,argparse,shutil
REP=ROOT.parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=ROOT/'results/rank64_balance');a=p.parse_args();CASE=a.case.resolve()
sys.path.insert(0,str(REP/'scripts'))
from measure import measure
from run_latest_rapid64 import validate_manifest
validate_manifest(CASE/'latest_rapid/source_manifest.json')
c=CASE/'latest_rapid/rapid_schedule'
assert json.loads((CASE/'latest_rapid/schedule.status.json').read_text())['exit_code']==0
m,z=measure([c/f'scheduled/dataset_{r}.bin' for r in range(64)],64,8192,8,LENGTHS,13,True)
save(c/'cps_metrics.json',m);np.savez_compressed(c/'cps_counts.npz',**z)
j=json.loads((c/'sample_audit.json').read_text());assert j['matching_128bit_record_multiset'] and j['global_batch_membership_preserved']
save(CASE/'latest_baseline_audit.json',j)
save(CASE/'latest_baseline_complete.json',dict(success=True,source='latest_source'))
print('LATEST INITIAL-STAGE MEASUREMENT COMPLETE',flush=True)
