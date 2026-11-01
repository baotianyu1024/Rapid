from pathlib import Path
import numpy as np,json,subprocess,sys
O=Path(__file__).resolve().parents[1];d=O/'validation/bagpipe/data';d.mkdir(parents=True,exist_ok=True)
x=np.column_stack([np.zeros(16,np.int64),np.arange(1,17,dtype=np.int64)]);np.save(d/'sparse.npy',x)
(d/'input.json').write_text(json.dumps(dict(active_id_offsets=[0,1,17],lengths=[1,1],dimensions=[32,128],ranks=8,local_batch=1,global_batches=2,rows=16,active_embeddings=17)))
o=d.parent/'trace';subprocess.run([sys.executable,str(O/'code/bagpipe_count.py'),'--data',str(d),'--out',str(o),'--epochs','1','--lookahead','2'],check=True)
r=json.loads((o/'result.json').read_text());assert r['ps_pull_vectors']==24 and r['ps_writeback_vectors_including_terminal_flush']==17 and r['gradient_allreduce_vectors']==28
assert r['total_embedding_bytes']==8*32*4+16*128*4+32*4+16*128*4+28*32*4
(d.parent/'audit.json').write_text(json.dumps(dict(success=True,hand_computed_bytes=21120),indent=2));print('PASS mixed-dimension BagPipe bytes',r['total_embedding_bytes'])
