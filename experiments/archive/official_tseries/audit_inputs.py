import ast,hashlib,json,re
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parents[1];results={}
for ds in ['t180','t200','t510']:
 m=json.loads((O/f'datasets/{ds}.json').read_text());assert m['complete'];source=Path(m['generator']);text=source.read_text();assert hashlib.sha256(text.encode()).hexdigest()==m['original_sha256'];tree=ast.parse(text)
 info=next(ast.literal_eval(z.value) for z in tree.body if isinstance(z,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dataset_info' for t in z.targets))
 assert m['lengths']==[h for nt,h,v in info for _ in range(nt)];assert m['cardinalities']==[v for nt,h,v in info for _ in range(nt)]
 x=np.memmap(m['input'],dtype='<u4',mode='r').reshape(m['rows'],-1);assert np.all(x[:,0]<=1);assert np.all((x[:,1:14].view('<f4')>=0)&(x[:,1:14].view('<f4')<1));col=14
 for length,card in zip(m['lengths'],m['cardinalities']):
  ids=x[:,col:col+length];assert int(ids.min())>=0 and int(ids.max())<card;col+=length
 assert col==x.shape[1]
 results[ds]=dict(all_id_ranges_checked=True,all_label_dense_checked=True,source_config_matches=True,rows=m['rows'],tables=len(info),expanded_tables=len(m['lengths']),input_sha256=m['data_sha256'])
(O/'input_audit.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
