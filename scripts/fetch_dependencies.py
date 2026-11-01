"""Fetch explicitly requested, pinned public dependencies into an ignored directory."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 lock=json.loads((ROOT/'experiments/dependencies.json').read_text());p=argparse.ArgumentParser(description=__doc__);p.add_argument('names',nargs='+',choices=list(lock));p.add_argument('--out',type=Path,default=ROOT/'sources');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 for name in a.names:
  dest=a.out/name;info=lock[name]
  if dest.exists():
   rev=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip();assert rev==info['commit'],f'Wrong revision: {dest}'
  else:
   subprocess.run(['git','clone','--no-checkout',info['url'],str(dest)],check=True);subprocess.run(['git','-C',str(dest),'checkout','--detach',info['commit']],check=True)
  if name=='HET-GMP':
   patch=ROOT/'experiments/het_gmp_safety.patch'
   applied=subprocess.run(['git','-C',str(dest),'apply','--reverse','--check',str(patch)],capture_output=True).returncode==0
   if not applied:subprocess.run(['git','-C',str(dest),'apply',str(patch)],check=True)
  print('READY',name,info['commit'])
if __name__=='__main__':main()
