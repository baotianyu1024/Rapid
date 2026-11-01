"""Run comparison algorithms on the exact data prepared by run_all.sh (CPU replay)."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'experiments/reproduction';B=ROOT/'experiments/baselines'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--sources',type=Path,default=ROOT/'sources');p.add_argument('--methods',nargs='+',choices=['gmp','herald','bagpipe','torchrec'],default=['gmp','herald']);p.add_argument('--epochs',type=int,default=5);p.add_argument('--lookahead',type=int,default=8);a=p.parse_args()
 case=a.case.resolve();data=case/'data';meta=json.loads((data/'input.json').read_text());g=meta['ranks'];b=meta['local_batch'];out=case/'baselines';out.mkdir(exist_ok=True)
 def run(name,args):
  cmd=[sys.executable,*map(str,args)];(out/(name+'.command.json')).write_text(json.dumps(cmd,indent=2))
  with (out/(name+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',name,flush=True)
 for method in a.methods:
  target=out/method
  if target.exists():p.error(f'Existing baseline outputs: {target}; use a fresh case')
  if method=='gmp':
   H=R/'het_gmp_rapid/scripts'
   run('gmp_partition',[H/'partition_het.py','--data',data,'--out',target/'partition','--ranks',g,'--batch-size',b])
   run('gmp_replay',[H/'replay_het.py','--data',data,'--partition',target/'partition','--out',target/'traffic','--ranks',g,'--batch-size',b,'--epochs',a.epochs,'--bounds',0,'--skip-1d'])
  elif method=='herald':run(method,[R/'herald_rapid_gmp/scripts/run_herald.py','--data',data,'--out',target,'--epochs',a.epochs])
  elif method=='bagpipe':run(method,[B/'bagpipe_count.py','--source',a.sources/'BagPipe','--data',data,'--out',target,'--epochs',a.epochs,'--lookahead',a.lookahead])
  else:
   target.mkdir();run(method,[B/'torchrec_plan.py','--source',a.sources/'torchrec-v1.1.0','--data',data,'--out',target/'plan.json','--dim',128,'--hbm-gib',16])
   sys.path.insert(0,str(B));from torchrec_count import table_bytes
   plan=json.loads((target/'plan.json').read_text());parts=[table_bytes(t,g,b,meta['global_batches']*a.epochs) for t in plan['tables']]
   result=dict(dim=128,epochs=a.epochs,all_rank_forward_backward_GiB=sum(x['forward_bytes']+x['backward_bytes'] for x in parts)/2**30,per_table=parts,measurement='protocol payload model from upstream plan, not measured network; row-wise ID bytes not counted')
   (target/'protocol_bytes.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
