import argparse,json,subprocess,sys,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;R=ROOT.parent
p=argparse.ArgumentParser();p.add_argument('--ranks',type=int,choices=[8,64],required=True);p.add_argument('--passes',type=int,default=6);a=p.parse_args();g=a.ranks;case=ROOT/f'r{g}_samples';cache=case/'profile_cache';out=ROOT/f'r{g}_strict'
z=np.load(case/'cps_counts.npz');norm=float(z['receive'].sum(1).mean());n=sum(len(np.load(cache/f'{t}.ids.npy',mmap_mode='r')) for t in range(26));sw=128*(n/g)/(norm*.0002)
cmd=[sys.executable,str(R/'scripts/joint_price_owners.py'),'--case',str(case),'--cache',str(cache),'--output',str(out),'--library',str(ROOT/'owner_kernel.so'),'--forward','128','--backward','128','--storage',str(sw),'--penalty','10','--budget','.2','--passes',str(a.passes),'--hot-threshold','16','--hot-first']
(ROOT/f'r{g}_strict.command.json').write_text(json.dumps(cmd,indent=2))
print('adaptive storage weight',sw,flush=True);subprocess.run(cmd,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
