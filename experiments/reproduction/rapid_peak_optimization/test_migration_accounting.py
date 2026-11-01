from migration_accounting import migration_ledger,checkpoint_due
import json
from pathlib import Path
x=migration_ledger([0,1,2,3],[1,0,3,2],4,True,4,state_tensors=3)
assert x['moved_embeddings']==4 and x['total_network_tensor_bytes']==6144
assert x['storage_count_delta']==[0,0,0,0]
assert x['peak_rank_send_tensor_bytes']==1536
for allowed,budget in [(False,4),(True,3)]:
 try:migration_ledger([0,1,2,3],[1,0,3,2],4,allowed,budget)
 except ValueError:pass
 else:raise AssertionError('Missing budget/freeze rejection')
assert not checkpoint_due(127,0,128) and checkpoint_due(128,0,128)
assert migration_ledger([0,1],[0,1],2,False,0)['total_network_tensor_bytes']==0
Path(__file__).with_name('migration_accounting_test.json').write_text(json.dumps(dict(success=True,scope='accounting only; no incremental experimental results yet'),indent=2));print('MIGRATION ACCOUNTING PASS')
