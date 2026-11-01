import json, subprocess, os, sys
from pathlib import Path
import numpy as np
R=Path('/mnt/hugui/longjiu.bty/reproduction'); old=R/'het_gmp_rapid/results/rank64_balance/latest_rapid';out=R/'rapid_peak_optimization'
scheduled=out/'schedule64'
cmd=json.load(open(old/'schedule.command.json'))
cmd[cmd.index('--input_file')+1]=str(R/'het_gmp_rapid/results/main64/data/input.bin')
cmd[cmd.index('--output_dir')+1]=str(scheduled)
subprocess.run(cmd,env=dict(os.environ,OMP_NUM_THREADS='16'),check=True)
base=old/'rapid_owner_base';dest=base/'scheduled';dest.mkdir(exist_ok=True)
cfg=json.load(open(base/'config.json'));width=1+cfg['dense']+sum(cfg['table_lengths']);offsets=np.cumsum([1+cfg['dense']]+cfg['table_lengths']);luts=[]
for t in range(26):
 z=np.load(base/f'table_{t}_remap.npz');lut=np.full(int(z['old'].max())+1,-1,dtype=np.int64);lut[z['old']]=z['new'];luts.append(lut)
for r in range(64):
 x=np.memmap(scheduled/f'dataset_{r}.bin',dtype='<u4',mode='r').reshape(-1,width)
 with open(dest/f'dataset_{r}.bin','wb') as f:
  for i in range(0,len(x),8192):
   y=x[i:i+8192].copy()
   for t,lut in enumerate(luts):
    v=lut[y[:,offsets[t]:offsets[t+1]]];assert np.all(v>=0);y[:,offsets[t]:offsets[t+1]]=v
   y.tofile(f)
 print('restored rank',r,flush=True)
sys.path.insert(0,str(R/'scripts'))
from measure import measure
_,a=measure([dest/f'dataset_{r}.bin' for r in range(64)],64,8192,8,cfg['table_lengths'],cfg['dense'],True)
z=np.load(base/'cps_counts.npz');assert all(np.array_equal(a[k],z[k]) for k in a)
from refine_balanced_owners import build_cache
build_cache(base,old/'profile_cache',cfg)
print('RESTORED AND COUNTS VERIFIED',flush=True)
