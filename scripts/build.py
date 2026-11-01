"""Build CPU tools and replay kernels from source; no prebuilt binaries shipped."""
import argparse,concurrent.futures,os,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
 subprocess.run(['bash',str(ROOT/'compile.sh')],check=True)
 tasks=[]
 for sub in ['rapid_peak_optimization','scripts']:
  for src in sorted((ROOT/'experiments/reproduction'/sub).glob('*.cpp')):
   if re.search(r'\bint\s+main\s*\(',src.read_text()):continue
   tasks.append(src)
 tasks.append(ROOT/'experiments/incremental/wide_profiles.cpp')
 def build(src):
  cmd=[os.environ.get('CXX','g++'),'-O3','-std=c++17','-fopenmp','-fPIC','-shared',str(src),'-o',str(src.with_suffix('.so'))]
  subprocess.run(cmd,check=True);print('BUILT',src.relative_to(ROOT),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:list(pool.map(build,tasks))
if __name__=='__main__':main()
