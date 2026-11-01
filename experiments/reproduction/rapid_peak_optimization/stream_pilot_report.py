import json
from pathlib import Path
root=Path(__file__).resolve().parent;rows=[]
for g,folder in [(8,'stream8_frozen'),(64,'stream64_i4_b001'),(64,'stream64_random_pilot')]:
 p=root/folder/'window_01/metrics.json'
 if p.exists():
  d=json.load(open(p));rows.append(dict(ranks=g,method=folder,total=d['total_vectors_per_8192'],peak=d['peak_combined_receive'],cv=d['rank_communication_cv'],source=str(p)))
for p in root.glob('*_pilot_*.json'):
 try:
  g=int(p.stem.rsplit('_',1)[1]);data=json.load(open(p))
  if not isinstance(data,list):continue
  for d in data:rows.append(dict(ranks=g,method=p.stem+f"_pass{d['pass_number']}",total=d['total_vectors_per_8192'],peak=d['peak_combined_receive'],cv=d['rank_cv'],source=str(p)))
 except (ValueError,KeyError):continue
rows.sort(key=lambda x:(x['ranks'],x['peak']));(root/'incremental_pilot_comparison.json').write_text(json.dumps(rows,indent=2))
for d in rows:print(d['ranks'],d['method'],round(d['total'],2),round(d['peak'],2),round(d['cv']*100,3))
