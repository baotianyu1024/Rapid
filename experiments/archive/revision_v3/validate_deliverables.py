from pathlib import Path
import json,re,csv,hashlib
V=Path(__file__).resolve().parents[1];s=(V/'中文Response.md').read_text();plan=(V/'训练实验方案.md').read_text()
assert s.count('$$')%2==0
for line in s.splitlines():
 if '$$' in line:assert line.strip()=='$$',line
assert not any(w in s for w in ['优化后 RAPID','GMP batch 约束版','核心 CPS','抽取 512','CPS only AUC','dim=32'])
assert 'Response to Reviewer 2' in s and 'Response to Reviewer 3' in s and 'Response to Reviewer 5' in s
# Every Markdown table row has a matching number of columns.
prev=None
for line in s.splitlines():
 if line.startswith('|'):
  cols=line.count('|')
  if prev is not None:assert cols==prev,(cols,prev,line)
  prev=cols
 else:prev=None
for x in re.findall(r'\]\(([^)]+)\)',s):
 if not x.startswith(('http://','https://')):assert (V/x).exists(),x
for d in ['criteo_8','kaggle_8','taobao_8','criteo_64']:
 c=json.loads((V/f'results/{d}/comparison.json').read_text());assert c['config']['dim']==128
 if d!='criteo_64':assert set(c['methods'])=={'RAPID','HET-GMP','Herald','TorchRec','BagPipe'}
 for method,r in c['methods'].items():assert abs(r['vector_GiB']-r['total_vectors']*512/2**30)<1e-9
 for method in ['RAPID','HET-GMP']:
  r=c['methods'][method];assert abs(r['peak_over_mean']-r['mean_step_peak']/r['mean_vectors_per_rank_step'])<1e-12
 r=c['methods']['RAPID'];assert r['primary_max']-r['primary_min']<=1
m=json.loads((V/'results/mcr_all/summary.json').read_text());assert [r['min_support'] for r in m['statistics']]==[10,20,50,100];assert m['all_eligible_ids'] and m['audit']['all_eligible_ids_enumerated'];assert len(m['audit']['independent_counter_checks'])==16
r=json.loads((V/'results/sensitivity_full/summary.json').read_text());assert len(r['results'])==18
for q in r['results']:assert q['audit']['success'] and q['primary_max']-q['primary_min']<=1
rows=list(csv.DictReader((V/'训练结果待填.csv').open()));assert len(rows)==497
assert set(x['method'] for x in rows if x['experiment']=='accuracy')=={'参考实现','RAPID'}
assert all(not x['AUC'] and not x['samples_per_second'] and x['status']=='待实测' and x['dim']=='128' for x in rows)
storage=json.loads((V/'storage_audit.json').read_text())
if len(storage['cases'])!=25 or sum('official_tseries' in x['case'] for x in storage['cases'])!=3:
 import subprocess
 subprocess.run(['/home/longjiu.bty/miniconda3/envs/het-gmp-rapid/bin/python',str(V/'code/audit_storage.py')],check=True)
assert json.loads((V/'storage_audit.json').read_text())['success']
assert len(json.loads((V/'storage_audit.json').read_text())['cases'])==25
assert json.loads((V/'math_validation.json').read_text())['success']
assert json.loads((V/'results/sensitivity_full/resolved_status.json').read_text())['success']
assert all(rc==0 for ds,method,rc in json.loads((V/'results/baseline_status.json').read_text()))
assert all(rc==0 for ds,rc in json.loads((V/'results/bagpipe_la8_status.json').read_text()))
coverage=json.loads((V/'review_coverage.json').read_text());assert len(coverage)==37
assert {x['reviewer'] for x in coverage}=={2,3,5}
for ds in ['t180','t200','t510']:
 c=json.loads((V/f'official_tseries/results/{ds}_8/comparison.json').read_text());assert set(c['methods'])=={'RAPID','HET-GMP','Herald','TorchRec','BagPipe'}
 assert c['config']['rows']==65536 and c['config']['dim']==128 and c['verification']['RAPID']['success']
 assert set(c['official_dimension_methods'])==set(c['methods'])
 for m in c['methods'].values():assert abs(m['vector_GiB']-m['total_embedding_bytes']/2**30)<1e-9
assert '重跑中' not in s and 'Zipf(1.2)' not in s and 'Zipf(1.2)' not in plan
assert len(list(csv.DictReader((V/'官方维度通信量对比.csv').open())))==15
comm=list(csv.DictReader((V/'通信量对比.csv').open()));assert len(comm)==30 and all(x['status']=='完成' for x in comm)
result=dict(review_items=37,success=True,communication_cases=7,parameter_and_batch_cases=18,training_rows=497,mcr_supports=[10,20,50,100],math_blocks=s.count('$$')//2,table_widths_consistent=True,links_exist=True)
(V/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
files={str(p.relative_to(V)):hashlib.sha256(p.read_bytes()).hexdigest() for p in V.rglob('*') if p.is_file() and (p.suffix in ['.json','.csv','.md'] and 'sources' not in p.parts and 'code' not in p.parts and p.name!='evidence_manifest.json')}
(V/'evidence_manifest.json').write_text(json.dumps(files,indent=2)+'\n')
