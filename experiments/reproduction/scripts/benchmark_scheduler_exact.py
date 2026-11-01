"""Sequential scheduler A/B on identical real batches, verify every output byte."""
import json,os,subprocess,time,hashlib,re
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results/speed_exact';out.mkdir(exist_ok=True);baseline=root/'het_gmp_rapid/results/timing32_rerun';basecmd=json.loads((baseline/'rapid_schedule.command.json').read_text());env=dict(os.environ,OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
results=[]
for version in ['reference','fast']:
 target=out/f'schedule_{version}_2b';cmd=basecmd.copy();cmd[0]=str(root/f'speed_work/{version}/scheduler');cmd[cmd.index('--num_chunks')+1]='2';cmd[cmd.index('--output_dir')+1]=str(target)
 (out/f'schedule_{version}.command.json').write_text(json.dumps(cmd,indent=2));t=time.perf_counter()
 with (out/f'schedule_{version}.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 elapsed=time.perf_counter()-t;log=(out/f'schedule_{version}.log').read_text();phases={k:sum(map(float,re.findall(k+r'=([\d.]+)s',log))) for k in ['parse','hits','balance','write']}
 for r in range(32):
  path=target/f'dataset_{r}.bin';data=path.read_bytes();gold=baseline/'rapid_base/scheduled'/f'dataset_{r}.bin'
  with gold.open('rb') as f:expected=f.read(len(data))
  assert data==expected, f'{version} rank {r} differs from historical baseline'
 row=dict(version=version,wall_seconds=elapsed,phases=phases,all_output_bytes_equal_historical_baseline=True,rows=2*32*8192,binary_sha256=hashlib.sha256(Path(cmd[0]).read_bytes()).hexdigest());results.append(row);print(row,flush=True)
 (out/'scheduler_benchmark.json').write_text(json.dumps(dict(results=results,scope='first two real 32-rank global batches, sequential, OMP16, exact original scheduling parameters; file cache uncontrolled'),indent=2))
