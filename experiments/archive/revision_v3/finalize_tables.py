"""Current tables exclude all retired inspired workloads."""
from pathlib import Path
import json,csv
V=Path(__file__).resolve().parents[1];rows=[];bag=[];official_rows=[]
for ds in ['criteo','kaggle','taobao','t180','t200','t510']:
 official=ds.startswith('t') and ds!='taobao';case=(V/'official_tseries' if official else V)/f'results/{ds}_8';p=case/'comparison.json'
 if not p.exists():
  for method in ['RAPID','HET-GMP','Herald','TorchRec','BagPipe']:rows.append(dict(dataset=ds,method=method,rows=65536,ranks=8,dimensions='128',local_batch=512,epochs=5,total_embedding_bytes='',GiB='',bytes_per_sample='',status='重跑中',evidence=''))
  continue
 c=json.loads(p.read_text());n=c['config']['rows']
 if official:
  for method,m in c.get('official_dimension_methods',{}).items():official_rows.append(dict(dataset=ds,method=method,rows=n,ranks=8,dimensions='/'.join(map(str,sorted(set(c['config']['dimensions'])))),local_batch=512,epochs=5,total_embedding_bytes=m['total_embedding_bytes'],GiB=m['vector_GiB']))
 for method,m in c['methods'].items():
  total=m.get('total_embedding_bytes',round(m['vector_GiB']*2**30));rows.append(dict(dataset=ds,method=method,rows=n,ranks=8,dimensions='128' if official else '128',local_batch=c['config']['local_batch'],epochs=5,total_embedding_bytes=total,GiB=m['vector_GiB'],bytes_per_sample=total/(n*5),status='完成',evidence=str(p)))
 for folder in (['bagpipe_la8'] if official else ['bagpipe_la8','bagpipe']):
  b=json.loads((case/folder/'result.json').read_text())
  if official:
   steps=json.loads((case/folder/'steps.json').read_text());pull=sum(x['ps_pull_bytes'] for x in steps);push=sum(x['ps_writeback_bytes'] for x in steps);sync=sum(x['gradient_allreduce_bytes'] for x in steps)
  else:pull=b['ps_pull_vectors']*512;push=b['ps_writeback_vectors_including_terminal_flush']*512;sync=b['gradient_allreduce_vectors']*512
  bag.append(dict(dataset=ds,lookahead=b['lookahead'],ps_pull_GiB=pull/2**30,ps_writeback_GiB=push/2**30,embedding_allreduce_GiB=sync/2**30,total_GiB=(pull+push+sync)/2**30))
for name,r in [('通信量对比.csv',rows),('BagPipe窗口与通信分解.csv',bag)]:
 with (V/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
print('Published real datasets and official T-series only')

with (V/'官方维度通信量对比.csv').open('w',newline='') as f:
 fields=['dataset','method','rows','ranks','dimensions','local_batch','epochs','total_embedding_bytes','GiB'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(official_rows)
