"""Create portable environment, code/results, and the exact common input subset."""
import argparse, hashlib, tarfile
import conda_pack
from common import *
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--out',required=True);p.add_argument('--reuse-env',action='store_true');a=p.parse_args()
case=Path(a.case).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
assert (case/'complete.json').exists() and (case/'REPORT.md').exists()
env=out/'het-gmp-rapid-env-linux64.tar.gz'
if a.reuse_env:
    assert env.is_file()
else:
    conda_pack.pack(prefix=sys.prefix,output=str(env),force=True,n_threads=1,verbose=True)
archive=out/'het-gmp-rapid-code-results.tar.gz'
def filt(info):
    parts=Path(info.name).parts
    if '__pycache__' in parts:return None
    if info.name.endswith('.pyc'):return None
    if 'scheduled' in parts:return None
    if 'results' in parts and 'data' in parts and not (info.isdir() or info.name.endswith('input.json')):return None
    return info
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
    for name in ['README.md','REPORT.md','requirements.txt','environment.yml','conda-explicit-linux-64.txt','provenance.json','hardware.txt','scripts','src','logs','results']:
        tar.add(ROOT/name,arcname='het_gmp_rapid/'+name,filter=filt)
data_archive=out/'criteo-day6-prefix-4194304.tar.gz'
with tarfile.open(data_archive,'w:gz',compresslevel=1) as tar:
    tar.add(case/'data/input.bin',arcname='input/input.bin')
    tar.add(case/'data/input.json',arcname='input/input.json')
checks=[]
for path in [env,archive,data_archive]:
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024**2),b''):digest.update(chunk)
    checks.append(f'{digest.hexdigest()}  {path.name}')
(out/'SHA256SUMS').write_text('\n'.join(checks)+'\n')
save(out/'manifest.json',dict(created=time.time(),python=sys.version,archives=[dict(name=x.name,bytes=x.stat().st_size) for x in [env,archive,data_archive]],scope='Environment, source + measured results (excluding large derived traces), exact input subset. Rebuild and rerun to regenerate derived files.'))
print('Packages:',out,flush=True)
