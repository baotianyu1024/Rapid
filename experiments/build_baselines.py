"""Build native comparison harnesses against pinned public checkouts."""
import argparse,os,subprocess,sysconfig
from pathlib import Path
import pybind11
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',type=Path,default=ROOT/'sources');p.add_argument('--methods',nargs='+',choices=['gmp','herald'],default=['gmp','herald']);a=p.parse_args()
 flags=[os.environ.get('CXX','g++'),'-O3','-std=c++17','-fopenmp','-pthread','-shared','-fPIC','-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include')]
 if 'gmp' in a.methods:
  src=a.sources/'HET-GMP/hetuCTR/csrc';out=ROOT/'experiments/reproduction/het_gmp_rapid/scripts'/('hetuCTR_partition'+sysconfig.get_config_var('EXT_SUFFIX'))
  subprocess.run(flags+['-I'+str(src),str(src/'partition/partition.cc'),str(src/'partition/pybind.cc'),'-o',str(out)],check=True)
 if 'herald' in a.methods:
  src=a.sources/'Herald/laia';build=ROOT/'experiments/reproduction/herald_rapid_gmp/build'
  subprocess.run(flags+['-I'+str(src/'include'),str(build/'herald_native.cpp'),str(src/'src/array.cc'),str(src/'src/thread_pool.cc'),'-o',str(build/('herald_native'+sysconfig.get_config_var('EXT_SUFFIX')))],check=True)
if __name__=='__main__':main()
