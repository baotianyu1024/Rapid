from pathlib import Path
import numpy as np,json,subprocess,sys
V=Path(__file__).resolve().parents[1];d=V/'results/validation/bagpipe_tiny/data';d.mkdir(parents=True,exist_ok=True)
x=np.column_stack([np.zeros(16,np.int64),np.arange(1,17,dtype=np.int64)]);np.save(d/'sparse.npy',x)
(d/'input.json').write_text(json.dumps(dict(active_id_offsets=[0,1,17],lengths=[1,1],ranks=8,local_batch=1,global_batches=2,rows=16,active_embeddings=17)))
o=d.parent/'trace';subprocess.run([sys.executable,str(V/'code/bagpipe_count.py'),'--data',str(d),'--out',str(o),'--epochs','1','--lookahead','2'],check=True)
r=json.loads((o/'result.json').read_text());assert r['ps_pull_vectors']==24;assert r['ps_writeback_vectors_including_terminal_flush']==17;assert r['gradient_allreduce_vectors']==28;assert r['total_vectors_with_flush']==69
(d.parent/'audit.json').write_text(json.dumps(dict(success=True,hand_computed=dict(pulls=24,writebacks=17,allreduce=28,total=69)),indent=2))
