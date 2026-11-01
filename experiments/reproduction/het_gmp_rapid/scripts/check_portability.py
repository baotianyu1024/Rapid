"""Actually relocate the packed env and rebuild tools in an independent copy."""
import argparse, hashlib, shutil, tarfile
from common import *
p=argparse.ArgumentParser();p.add_argument('--artifacts',required=True);a=p.parse_args()
artifacts=Path(a.artifacts).resolve();dest=artifacts/'relocation-check'
assert not dest.exists(), 'Use an unused relocation-check directory'
env=dest/'env';project=dest/'project';env.mkdir(parents=True);project.mkdir()
with tarfile.open(artifacts/'het-gmp-rapid-env-linux64.tar.gz') as tar:tar.extractall(env)
runenv=dict(os.environ,PATH=str(env/'bin')+os.pathsep+os.environ.get('PATH',''))
subprocess.run([str(env/'bin/conda-unpack')],env=runenv,check=True)
for folder in ['src','scripts']:
    shutil.copytree(ROOT/folder,project/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
commands=[['-m','pip','check'],[str(project/'scripts/build.py')],[str(project/'scripts/test_replay.py')]]
results=[]
for args in commands:
    proc=subprocess.run([str(env/'bin/python'),*args],env=runenv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    results.append(dict(args=args,exit_code=proc.returncode,output=proc.stdout))
    if proc.returncode:
        save(artifacts/'portability-validation.json',dict(passed=False,results=results));raise RuntimeError(proc.stdout)
so=next((project/'scripts').glob('hetuCTR_partition*.so'))
dynamic=subprocess.check_output(['readelf','-d',str(so)],text=True)
assert str(env/'lib') in dynamic
save(artifacts/'portability-validation.json',dict(passed=True,relocated_prefix=str(env),results=results,native_extension_rpath_verified=True,note='Same host, different absolute prefix; does not claim testing another physical machine. Rebuild on target Linux x86_64.'))
shutil.rmtree(dest)
print('PASS relocated environment, dependency check, native rebuild/import, replay checks')
