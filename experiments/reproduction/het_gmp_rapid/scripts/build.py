import os, subprocess, sys, sysconfig
from pathlib import Path
import pybind11
from common import ROOT

prefix = Path(sys.prefix)
cxx = prefix / 'bin/x86_64-conda-linux-gnu-g++'
if not cxx.exists(): cxx = Path(os.environ.get('CXX','g++'))
flags = ['-O3','-std=c++17','-fopenmp','-pthread']
for name in ['arrange_ids_to_devices','generate_local_dataset']:
    src = ROOT/'src/rapid/cps'/f'{name}.cpp'
    subprocess.run([str(cxx),*flags,str(src),'-o',str(src.with_suffix(''))],check=True)
src = ROOT/'src/het-gmp/hetuCTR/csrc'
dest = ROOT/'scripts'/('hetuCTR_partition'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd = [str(cxx),*flags,'-shared','-fPIC','-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),'-I'+str(src),str(src/'partition/partition.cc'),str(src/'partition/pybind.cc'),'-o',str(dest)]
print(' '.join(cmd),flush=True)
subprocess.run(cmd,check=True)
import hetuCTR_partition
print('Original HET-GMP CPU partition extension imported successfully')
subprocess.run([str(cxx),*flags,str(ROOT/'scripts/audit_samples.cpp'),'-o',str(ROOT/'scripts/audit_samples')],check=True)
