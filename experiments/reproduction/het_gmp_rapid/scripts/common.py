import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = [3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1]
P, B, DENSE, DIM, EPOCHS = 8, 8192, 13, 128, 5
WIDTH = 1 + DENSE + sum(LENGTHS)

def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')

def measured(name, cmd, out, threads=16):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cmd = [str(x) for x in cmd]
    save(out / (name+'.command.json'), cmd)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    start = time.perf_counter()
    with (out / (name+'.log')).open('w') as f:
        r = subprocess.run(['/usr/bin/time','-f','%e %M %U %S','-o',str(out/(name+'.time')), *cmd], stdout=f, stderr=subprocess.STDOUT, env=env)
    parts = (out/(name+'.time')).read_text().strip().splitlines()[-1].split()
    v = dict(seconds=time.perf_counter()-start, max_rss_kib=int(parts[1]), user_seconds=float(parts[2]), system_seconds=float(parts[3]), exit_code=r.returncode, threads=threads)
    save(out/(name+'.status.json'),v)
    print(name, v, flush=True)
    if r.returncode: raise RuntimeError(f'{name} failed: {out/name}.log')
    return v
