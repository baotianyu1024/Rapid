"""Prepare only the next window's new owners; never read or change training cache state."""
import argparse,pickle,json
from pathlib import Path
import numpy as np
from stream_window import window
from stream_alternate import alternate
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--state-in',type=Path,required=True);p.add_argument('--state-out',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--ranks',type=int,required=True);p.add_argument('--rows',type=int,required=True);p.add_argument('--window',type=int,required=True);p.add_argument('--rounds',type=int,required=True);p.add_argument('--balance',type=float,required=True);p.add_argument('--seed-schedule',required=True);a=p.parse_args()
with open(a.state_in,'rb') as f:state=pickle.load(f)
cfg=json.load(open(ROOT/f'r{a.ranks}_peak_soft/config.json'));width=1+cfg['dense']+sum(cfg['table_lengths']);offset=4194304+(a.window-1)*a.rows;data=np.memmap('/mnt/hugui/longjiu.bty/data/processed/criteo/day6.bin',dtype='<u4',mode='r',offset=offset*width*4,shape=(a.rows,width));seed=a.out/f'window_{a.window:02d}_seed';owner=a.out/f'window_{a.window:02d}_owner';window(state,data,cfg,seed,rounds=a.rounds,balance=a.balance,schedule=a.seed_schedule);alternate(state,seed,data,owner,rounds=0)
with open(a.state_out.with_suffix('.tmp'),'wb') as f:pickle.dump(state,f,protocol=5)
a.state_out.with_suffix('.tmp').replace(a.state_out);print('OWNER PREPARATION COMPLETE',a.window,flush=True)
