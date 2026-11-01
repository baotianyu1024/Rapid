"""Validate seeded cold ownership, malformed inputs, and unchanged default output."""
import tempfile,subprocess,os
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];exe=root/'reproduction/working/cps/arrange_ids_to_devices'
with tempfile.TemporaryDirectory(dir=root/'reproduction') as d:
 d=Path(d);(d/'hot').mkdir();(d/'hot/0.txt').write_text('9:1\n')
 np.array([[0,0,1],[0,0,2],[0,0,1],[0,0,2]],dtype=np.int32).tofile(d/'input.bin')
 cmd=['--input_bin',str(d/'input.bin'),'--hot_ids_dir',str(d/'hot'),'--batch_size','2','--num_devices','2','--n_batches','1','--dense_count','1','--strategy','3','--migrate_threshold','1000000','--table_lengths','1']
 def run(binary,out,extra=[]):return subprocess.run([str(binary),*cmd,'--output_dir',str(d/out),*extra],env=dict(os.environ,OMP_NUM_THREADS='1'),capture_output=True)
 for name,b in [('old',str(exe)+'.pre_metis'),('new',exe)]:
  r=run(b,name);assert r.returncode==0,r.stderr
 for f in (d/'old').iterdir():assert f.read_bytes()==(d/'new'/f.name).read_bytes()
 (d/'seed').write_text('0 1 1\n0 2 0\n');r=run(exe,'seeded',['--seed_assignment',str(d/'seed')]);assert r.returncode==0,r.stderr
 assert '0:1' in (d/'seeded/table_1').read_text();assert '0:2' in (d/'seeded/table_0').read_text()
 for text in ['0 9 0\n','0 1 0\n0 1 1\n','0 1 2\n','3 1 0\n']:
  (d/'bad').write_text(text);r=run(exe,'badout',['--seed_assignment',str(d/'bad')]);assert r.returncode!=0
 print('PASS default byte parity, seed ownership, hot/duplicate/rank/table rejection')
