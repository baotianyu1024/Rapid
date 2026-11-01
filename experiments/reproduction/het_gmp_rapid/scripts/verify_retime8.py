"""Post-timing output checks, excluded from generation wall time."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from common import WIDTH,save
ROOT=Path(__file__).resolve().parents[3];CASE=ROOT/'reproduction/het_gmp_rapid/results/timing8_rerun';OLD=ROOT/'reproduction/results'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def main():
 assert json.loads((CASE/'preprocess_complete.json').read_text())['success']
 start=time.perf_counter()
 raw=np.memmap(CASE/'data/input.bin',dtype='<u4',mode='r').reshape(-1,WIDTH)
 assignment=np.load(CASE/'gmp/assignment.npy',mmap_mode='r')
 assert len(assignment)==len(raw) and np.all((assignment>=0)&(assignment<8))
 rank_rows=[]
 for r in range(8):
  ix=np.flatnonzero(assignment==r);rank_rows.append(len(ix))
  written=np.memmap(CASE/f'gmp/scheduled/dataset_{r}.bin',dtype='<u4',mode='r').reshape(-1,WIDTH)
  assert len(written)==len(ix)
  for off in range(0,len(ix),32768):assert np.array_equal(written[off:off+32768],raw[ix[off:off+32768]])
 assert sum(rank_rows)==4194304
 same={}
 for current,old in [('rapid_schedule','criteo_day6_8_updated_64b_improved'),('rapid_owner_base','criteo_day6_8_owner_target7_f10_64b'),('rapid_guard_p5','criteo_day6_8_guard_p5')]:
  matches=[]
  for r in range(8):
   new=CASE/current/f'scheduled/dataset_{r}.bin';prior=OLD/old/f'scheduled/dataset_{r}.bin'
   assert new.stat().st_size==524288*WIDTH*4
   matches.append(digest(new)==digest(prior))
  same[current]=matches
  print(current,'historical exact matches',sum(matches),'/8',flush=True)
 final=json.loads((CASE/'rapid_guard_p5/complete.json').read_text())
 assert final['independent_counts_verified'] and final['all_records_inverse_verified']
 save(CASE/'postrun_verification.json',dict(success=True,gmp_full_records_equal_source=True,gmp_rank_rows=rank_rows,rapid_historical_shard_matches=same,rapid_independent_counts_verified=True,rapid_inverse_records_verified=True,seconds=time.perf_counter()-start,note='Excluded from reported generation time; historical matching is diagnostic because upstream parallel placement can vary.'))
if __name__=='__main__':main()
