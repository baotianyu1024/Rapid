from pathlib import Path
import hashlib,json
V=Path(__file__).resolve().parents[1]
def sha(p,limit=None):
 h=hashlib.sha256();left=limit
 with p.open('rb') as f:
  while left is None or left>0:
   z=f.read(16*1024*1024 if left is None else min(left,16*1024*1024))
   if not z:break
   h.update(z)
   if left is not None:left-=len(z)
 if left is not None:assert left==0
 return h.hexdigest()
files=[]
for d in ['criteo_8','criteo_64','kaggle_8','taobao_8']:
 c=V/'results'/d;files.extend([c/'optimized/peak_samples/cps_counts.npz',c/'gmp_traffic/het_gmp_s0.npz'])
 if d!='criteo_64':files.extend([c/'herald/counts.npz',c/'bagpipe_la8/steps.json',c/'torchrec/plan.json'])
for d in ['t180_8','t200_8','t510_8']:
 c=V/'official_tseries/results'/d;files.extend([c/'optimized/peak_samples/cps_counts.npz',c/'gmp_bytes/het_gmp_s0.npz',c/'herald/counts.npz',c/'bagpipe_la8/steps.json',c/'torchrec/plan.json'])
for d in (V/'results/sensitivity_full').iterdir():
 if d.is_dir():files.append(d/'optimized/peak_samples/cps_counts.npz')
files.extend([V/'results/mcr_all/all_targets.u32',V/'results/mcr_all/frequency.u32'])
m=json.loads((V/'results/mcr_all/manifest.json').read_text());n=m['rows']*228*4
result=dict(files={str(p.relative_to(V)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},mcr_source_prefix=dict(path=m['source'],bytes=n,sha256=sha(Path(m['source']),n)))
(V/'raw_evidence_hashes.json').write_text(json.dumps(result,indent=2)+'\n');print('sealed',len(files),'evidence files')
