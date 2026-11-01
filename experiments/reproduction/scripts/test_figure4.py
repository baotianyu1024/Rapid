import numpy as np
from measure_figure4 import peer_counts
rng=np.random.default_rng(17)
for g in [2,8,32,64]:
 ids=rng.integers(0,1000,(100,19),dtype=np.uint32)
 for r in range(g):
  expected=[len({int(x)%g for x in row}-{r}) for row in ids]
  assert np.array_equal(peer_counts(ids,g,r),expected)
print('PASS: vectorized Figure4 counts equal original per-sample remote-owner sets')
