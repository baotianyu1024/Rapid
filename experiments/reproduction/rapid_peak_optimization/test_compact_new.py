import numpy as np
from stream_owner_state import OwnerState
from stream_alternate import relocate_new
from compact_new_slots import storage_accounting
rng=np.random.default_rng(9)
for g in [8,64]:
 s=OwnerState(g,2)
 for t in range(2):s.add_new(t,np.arange(g*2),np.arange(g*2)%g)
 old_n=s.next_gid;original=[x.copy() for x in s.physical];ids=[];planned=[]
 for t in range(2):
  newids=np.arange(g*2,g*6);s.add_new(t,newids,newids%g);allids=np.arange(g*6);owners=s.owner[t].copy();owners[newids]=rng.permutation(owners[newids]);ids.append(allids);planned.append(owners)
 relocate_new(s,ids,planned,old_n);s.audit()
 for t in range(2):assert np.array_equal(s.physical[t][:g*2],original[t])
 account=storage_accounting(s);assert not any(account['free_physical_slots'])
print('NEW-ONLY ADDRESS COMPACTION PASS')
