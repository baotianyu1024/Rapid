"""Build the three CPU tools needed by run_case.py with a C++17 compiler."""
import argparse,os,shlex,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--repo',default=str(root/'working'));p.add_argument('--cxx',default=os.environ.get('CXX','g++'));a=p.parse_args();repo=Path(a.repo).resolve()
for relative in ['cps/arrange_ids_to_devices','cps/generate_local_dataset','analysis/verify_hit_rate']:
 dest=repo/relative;tmp=dest.with_name(dest.name+'.build-new')
 cmd=[*shlex.split(a.cxx),'-O3','-std=c++17','-fopenmp','-pthread',str(dest.with_suffix('.cpp')),'-o',str(tmp)]
 print(shlex.join(cmd),flush=True);subprocess.run(cmd,check=True);tmp.replace(dest)
