"""All timed preprocessors run sequentially to avoid cross-experiment contention."""
import argparse
from common import *
p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args()
case=Path(a.case).resolve();data=case/'data'
assert (data/'input.json').exists()
measured('rapid_pipeline',[sys.executable,ROOT/'scripts/run_preprocess.py','--data',data,'--out',case/'rapid','--algorithm','rapid'],case)
measured('het_pipeline',[sys.executable,ROOT/'scripts/run_preprocess.py','--data',data,'--out',case/'het','--algorithm','het'],case)
measured('het_materialize',[sys.executable,ROOT/'scripts/materialize_het.py','--data',data,'--partition',case/'het'],case)
measured('sample_audit',[ROOT/'scripts/audit_samples',data/'input.bin',case/'rapid',case/'rapid/sample_audit.json'],case)
measured('replay_het',[sys.executable,ROOT/'scripts/replay_het.py','--data',data,'--partition',case/'het','--out',case/'traffic'],case)
measured('replay_rapid',[sys.executable,ROOT/'scripts/replay_rapid.py','--data',data,'--rapid',case/'rapid','--out',case/'traffic'],case)
save(case/'complete.json',dict(completed=True))
