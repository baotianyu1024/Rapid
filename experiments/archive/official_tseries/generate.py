"""Execute pinned upstream generator, with RNG seeding and T180 filename fix only."""
import argparse,ast,hashlib,json,re,subprocess,sys
from pathlib import Path
import numpy as np
V=Path(__file__).resolve().parents[2];O=V/'official_tseries';S=V/'sources/HugeCTR-official';p=argparse.ArgumentParser();p.add_argument('dataset',choices=['t180','t200','t510']);a=p.parse_args()
existing=O/f'datasets/{a.dataset}.json'
if existing.exists():
 prior=json.loads(existing.read_text())
 if prior.get('complete'):
  assert hashlib.sha256(Path(prior['generator']).read_bytes()).hexdigest()==prior['original_sha256']
  digest=hashlib.sha256()
  with Path(prior['input']).open('rb') as stream:
   for chunk in iter(lambda:stream.read(64*1024**2),b''):digest.update(chunk)
  assert digest.hexdigest()==prior['data_sha256']
  print('REUSE verified official data',a.dataset,flush=True);sys.exit(0)

names={'t180':'180table_70B_hotness80','t200':'200table_100B_hotness20','t510':'510table_110B_hotness5'};name=names[a.dataset];src=S/f'benchmarks/embedding_collection/dataset/{name}.py';original=src.read_text();code=original.replace('output_file =','output_filename =') if a.dataset=='t180' else original
info=next(ast.literal_eval(x.value) for x in ast.parse(original).body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dataset_info' for t in x.targets))
bench=(S/'benchmarks/embedding_collection/benchmark.sh').read_text();block=bench.split('dataset_'+name+'="')[1].split('"')[0];group_dims=list(map(int,re.search(r'--ev_size_per_table ([0-9,]+)',block)[1].split(',')));dims=[d for (nt,_,_),d in zip(info,group_dims) for _ in range(nt)];lengths=[h for nt,h,v in info for _ in range(nt)];cards=[v for nt,h,v in info for _ in range(nt)]
meta=dict(dataset=a.dataset,source_url='https://github.com/NVIDIA-Merlin/HugeCTR',commit=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip(),generator=str(src),original_sha256=hashlib.sha256(original.encode()).hexdigest(),executed_sha256=hashlib.sha256(code.encode()).hexdigest(),patch='T180 output_file renamed output_filename' if a.dataset=='t180' else None,numpy_version=np.__version__,python_version=sys.version,byteorder=sys.byteorder,seed=20261027,alpha=1.1,rows=65536,parts=1,official_full_parts=100,scope='one complete official generated part; not full 100-part benchmark dataset',lengths=lengths,dimensions=dims,cardinalities=cards,record_bytes=4*(14+sum(lengths)),input=str(O/f'datasets/{name}_synthetic_alpha1.1.bin.part0'))
(O/f'datasets/{a.dataset}.json').write_text(json.dumps(meta,indent=2));(O/f'code/{name}.py').write_text(code)
np.random.seed(meta['seed']);sys.argv=[str(src),'0',str(O/'datasets')];exec(compile(code,str(src),'exec'),{'__name__':'__main__'})
f=Path(meta['input']);assert f.stat().st_size==meta['rows']*meta['record_bytes'];digest=hashlib.sha256()
with f.open('rb') as stream:
 for chunk in iter(lambda:stream.read(64*1024**2),b''):digest.update(chunk)
meta['data_sha256']=digest.hexdigest();meta['complete']=True;(O/f'datasets/{a.dataset}.json').write_text(json.dumps(meta,indent=2));print('COMPLETE',a.dataset,meta['data_sha256'],flush=True)
