import tempfile,json
from pathlib import Path
import numpy as np
from stream_owner_state import OwnerState
from stream_tune import tune
rng=np.random.default_rng(3)
for g,nb in [(8,64),(64,8)]:
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp);c=p/'profile_cache';c.mkdir();n=g*100;prof=np.zeros((n,g),np.uint64)
  for i in range(n):
   for j in range(1 if i<n//2 else 32):prof[i,rng.integers(g)]|=np.uint64(1)<<np.uint64(rng.integers(nb))
  ids=np.arange(n,dtype=np.uint32);np.save(p/'0.original_ids.npy',ids);np.save(c/'0.profiles.npy',prof);(p/'config.json').write_text(json.dumps(dict(ranks=g,batches=nb,table_lengths=[1])))
  state=OwnerState(g,1);state.add_new(0,ids,ids%g);old=state.owner[0].copy();storage=state.storage.copy();result=tune(state,p,.05,p/'out');assert np.array_equal(state.storage,storage);assert result['moved_embeddings']<=int(n*.05);assert result['moved_embeddings']==int((state.owner[0]!=old).sum());assert result['historical_uncached_total_after']<=result['historical_uncached_total_before'];assert result['historical_uncached_peak_after']<=result['historical_uncached_peak_before'];state.audit()
print('BOUNDED CAUSAL SWAPS PASS')
