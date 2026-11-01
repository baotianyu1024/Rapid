"""Cross-file final audit of complete real-data measurement artifacts."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];checks=[]
for case in sorted((ROOT/'results').iterdir()):
 if not (case/'complete.json').exists():continue
 cfg=json.loads((case/'config.json').read_text());G=cfg['ranks'];NB=cfg['batches'];T=len(cfg['table_lengths']);width=4*(1+cfg['dense']+sum(cfg['table_lengths']))
 for tag in ['hash','cps']:
  p=case/f'{tag}_counts.npz'
  if not p.exists():continue
  a=np.load(p);recv=a['receive'];send=a['send'];uniq=a['unique'];hits=a['hits']
  assert recv.shape==(NB,T,G),(case,tag,recv.shape)
  assert np.array_equal(recv.sum(axis=2),send.sum(axis=2)),(case,tag,'traffic conservation')
  assert np.all(recv>=0) and np.all(uniq>=recv) and np.all(hits>=0)
  m=json.loads((case/f'{tag}_metrics.json').read_text())
  for key,counts in [('unique_receive',recv),('unique_send',send)]:
   per=counts.sum(axis=1);np.testing.assert_allclose(m[key]['mean'],per.mean());np.testing.assert_allclose(m[key]['mean_batch_peak'],per.max(axis=1).mean());np.testing.assert_allclose(m[key]['rank_means'],per.mean(axis=0))
  np.testing.assert_allclose(m['unique_local_hit_rate'],1-recv.sum()/uniq.sum())
  np.testing.assert_allclose(m['local_hit_rate'],hits.sum()/(NB*G*8192*sum(cfg['table_lengths'])))
 for rank in range(G):
  p=case/'scheduled'/f'dataset_{rank}.bin'
  assert p.stat().st_size==NB*8192*width,(case,rank,'row count')
 checks.append(case.name)
manifest=json.loads((ROOT/'baseline_manifest.json').read_text())
for name,files in manifest.items():
 for rel,digest in files.items():
  p=ROOT.parent/'code'/name/rel
  assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,('original changed',p)
for name in ['criteo','kaggle']:
 p=ROOT/f'{name}_preprocessing_record_validation.json'
 if p.exists():assert json.loads(p.read_text())['all_checks_pass']
# Independent Figure4 counter must exactly match corresponding Hash count tensors.
for p in (ROOT/'results').glob('*figure4.json'):
 d=json.loads(p.read_text());raw=np.load(p.with_suffix('.npz'))['unique']
 for case in (ROOT/'results').iterdir():
  if not (case/'complete.json').exists() or not (case/'hash_counts.npz').exists():continue
  cfg=json.loads((case/'config.json').read_text())
  if cfg['input']==d['input'] and cfg['ranks']==8 and cfg['batches']>=d['batches']:
   independent=np.load(case/'hash_counts.npz')['receive'][:d['batches']]
   assert np.array_equal(raw,independent),(p,case,'Figure4 independent parity')
result=dict(status='PASS',completed_cases_checked=len(checks),cases=checks,original_code_unchanged=True,
 checks=['send/receive conservation every table and batch','metrics agree with stored counts','per-rank full record counts','both locality denominators','original source SHA256 unchanged','preprocessing spot checks','Figure4 independent hash-counter parity'])
(ROOT/'final_audit.json').write_text(json.dumps(result,indent=2));print('PASS final audit:',len(checks),'completed cases')
