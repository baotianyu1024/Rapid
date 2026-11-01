from pathlib import Path
import subprocess,json,hashlib
V=Path(__file__).resolve().parents[1];out={}
for d in (V/'sources').iterdir():
 if not d.is_dir():continue
 files={str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for p in d.rglob('*') if p.is_file() and '.git' not in p.parts and p.suffix in ['.py','.cc','.cpp','.h','.hpp','.md','.cmake','.sh']}
 git={}
 if (d/'.git').exists():
  for k,cmd in [('commit',['rev-parse','HEAD']),('remote',['remote','get-url','origin']),('status',['status','--porcelain'])]:
   z=subprocess.run(['git','-C',str(d),*cmd],text=True,capture_output=True);git[k]=z.stdout.strip() if z.returncode==0 else None
 out[d.name]=dict(path=str(d),**git,files=files)
code={str(p.relative_to(V)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (V/'code').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.cpp','.h','.so']}
for directory in ['official_tseries/code','official_tseries/herald']:
 for p in (V/directory).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.cpp','.h','.so','.sh']:code[str(p.relative_to(V))]=hashlib.sha256(p.read_bytes()).hexdigest()
(V/'source_manifest.json').write_text(json.dumps(dict(sources=out,executed_code=code),indent=2)+'\n')
print({k:dict(commit=v.get('commit'),files=len(v['files'])) for k,v in out.items()})
