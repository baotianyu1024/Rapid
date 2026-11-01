"""Small CPU correctness checks; synthetic fixtures are not performance datasets."""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path);a=p.parse_args();checks=[]
 with tempfile.TemporaryDirectory(prefix='rapid-smoke-') as tmp:
  tmp=Path(tmp)
  def run(name,args):
   with (tmp/(name+'.log')).open('w') as log:r=subprocess.run([sys.executable,*map(str,args)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
   if r.returncode:
    print((tmp/(name+'.log')).read_text()[-5000:]);raise RuntimeError(name)
   checks.append(name);print('PASS',name,flush=True)
  for g,rows,card,b in [(8,512,2000,32),(64,2560,20000,8)]:
   rng=np.random.default_rng(101 if g==8 else 202);x=rng.integers(0,card,(rows,7),dtype=np.uint32);x[:,0]=np.arange(rows);inp=tmp/f'input{g}.bin';x.tofile(inp)
   run(f'offline{g}',[ROOT/'scripts/run_offline.py','--input',inp,'--out',tmp/f'offline{g}','--ranks',g,'--dataset','custom','--dense',1,'--table-lengths',3,2,'--batch-size',b,'--batches',2,'--topk',16])
   cfg=json.loads((tmp/f'offline{g}/optimized/peak_samples/storage.json').read_text());assert max(cfg['primary_counts'])-min(cfg['primary_counts'])<=1
   run(f'incremental{g}',[ROOT/'experiments/run_incremental.py','--input',inp,'--out',tmp/f'inc{g}','--ranks',g,'--dense',1,'--table-lengths',3,2,'--batch-size',8,'--initial-batches',2,'--window-batches',1,'--windows',3])
   summary=json.loads((tmp/f'inc{g}/summary.json').read_text());assert len(summary['results'])==4
   for row in summary['results']:
    assert row['old_owner_changes']==row['old_physical_id_changes']==row['migration_bytes']==0
  X=ROOT/'experiments/reproduction/rapid_peak_optimization'
  run('exact_sample_swaps_8_64',[X/'test_samples.py']);run('exact_group_swaps_8_64',[X/'test_group_scheduler.py'])
 report=dict(success=True,checks=checks,gpu_execution=False,scope='small synthetic correctness fixtures, not performance or paper evidence')
 if a.report:a.report.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
