"""Compile upstream default LAIA get_dist and cache, with a CPU trace harness.

The only get_dist edit substitutes std::set for boost::container::flat_set:
both have identical sorted unique integer set semantics used by this function.
No changes to scoring, sample order, tie-breaking, capacity or dependencies.
"""
from pathlib import Path
import hashlib, json, subprocess, sys, sysconfig
import pybind11

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT.parents[2] / 'sources/Herald/laia'
build = ROOT/'build'
build.mkdir(exist_ok=True)
source = (UP/'src/laia_scheduler.cc').read_text()
fn = source[source.index('void LaiaScheduler::get_dist('):source.rindex('} // namespace laia_cache')]
fn = fn.replace('boost::container::flat_set<emb_key_t>', 'std::set<emb_key_t>')
prefix = r'''
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <stdexcept>
#include "array.h"
#include "thread_pool.h"
#include "mini_lru_cache.h"
namespace py = pybind11;
namespace laia_cache {
using emb_key_t = uint64_t;
using dist_t = Array2D;
using comm_plan_t = std::vector<std::set<emb_key_t>>;
class LaiaScheduler {
public:
 size_t num_worker_, mini_batch_size_, batch_size_, num_table_, num_sample_, num_thread_=16, batch_id_=0;
 Array2D sample_embs_, count_, scores_;
 Array3D sample_emb_dep_;
 std::vector<int> workers_workload_;
 std::vector<MiniLRUCache> cache_snaps_;
 LaiaScheduler(size_t p,size_t b,size_t cols,size_t cap,size_t universe):
 num_worker_(p),mini_batch_size_(b),batch_size_(p*b),num_table_(cols),num_sample_(p*b),
 sample_embs_(2,{p*b,cols},0),count_(2,{p*b,p+1},0),scores_(2,{p*b,p},0),
 sample_emb_dep_(3,{p*b,p+1,cols},0) {
  if(!p || !b || !cols || !cap) throw std::runtime_error("positive sizes required");
  cache_snaps_.reserve(p);
  for(size_t r=0;r<p;++r) {
   if(universe) cache_snaps_.emplace_back(universe);
   else cache_snaps_.emplace_back();
   cache_snaps_.back().set_cap(cap);
  }
 }
 void get_dist(comm_plan_t&,dist_t&);
 py::dict step(py::array_t<uint64_t,py::array::c_style|py::array::forcecast> input,size_t batch_id,bool audit) {
  auto x=input.unchecked<2>();
  if(size_t(x.shape(0))!=batch_size_ || size_t(x.shape(1))!=num_table_) throw std::runtime_error("shape");
  batch_id_=batch_id;
  for(size_t i=0;i<batch_size_;++i) for(size_t j=0;j<num_table_;++j) {
    if(x(i,j)>2147483647) throw std::runtime_error("cache uses int32 keys");
    sample_embs_[i][j]=x(i,j);
  }
  comm_plan_t plan(num_worker_); dist_t dist(2,{num_worker_,mini_batch_size_},0);
  get_dist(plan,dist);
  std::vector<std::vector<uint64_t>> keys(num_worker_);
  std::vector<int> seen(batch_size_,0);
  std::vector<uint64_t> assignment(batch_size_);
  for(size_t r=0;r<num_worker_;++r) {
   auto &k=keys[r];k.reserve(mini_batch_size_*num_table_);
   for(size_t i=0;i<mini_batch_size_;++i) {
    auto row=dist[r][i]; if(row>=batch_size_ || seen[row]++) throw std::runtime_error("sample lost/duplicated");
    assignment[row]=r;
    k.insert(k.end(),sample_embs_[row],sample_embs_[row]+num_table_);
   }
   std::sort(k.begin(),k.end());k.erase(std::unique(k.begin(),k.end()),k.end());
  }
  if(audit) for(size_t r=0;r<num_worker_;++r) {
   std::set<uint64_t> expected;
   for(size_t other=0;other<num_worker_;++other) if(other!=r)
    for(auto k:keys[other]) if(cache_snaps_[r].check(k)) expected.insert(k);
   if(expected!=plan[r]) throw std::runtime_error("dependency plan disagrees with Algorithm 2");
  }
  std::vector<std::vector<uint64_t>> counts(num_worker_,std::vector<uint64_t>(5));
  auto pool=ThreadPool::Get();
  std::vector<std::future<void>> futures;
  for(size_t r=0;r<num_worker_;++r) futures.push_back(pool->Enqueue([&,r] {
   auto &c=counts[r];c[3]=plan[r].size();c[4]=keys[r].size();
   for(auto k:plan[r]) cache_snaps_[r].outdate(k);
   for(auto k:keys[r]) {
    int status=cache_snaps_[r].get(k);
    if(status==-2) ++c[1];
    else if(status>=0) {++c[0]; if(status>0) ++c[2];}
   }
  }));
  pool->Wait(num_worker_);for(auto &f:futures) f.get();
  py::dict result;result["counts"]=counts;result["assignment"]=assignment;return result;
 }
 std::vector<uint64_t> pending() {
  std::vector<uint64_t> result;
  for(auto &c:cache_snaps_) result.push_back(c.get_keys().size());
  return result;
 }
};
'''
suffix = r'''
}
PYBIND11_MODULE(herald_native,m) {
 py::class_<laia_cache::LaiaScheduler>(m,"Scheduler")
 .def(py::init<size_t,size_t,size_t,size_t,size_t>(),py::arg("p"),py::arg("b"),py::arg("cols"),py::arg("cap"),py::arg("universe")=0)
 .def("step",&laia_cache::LaiaScheduler::step)
 .def("pending",&laia_cache::LaiaScheduler::pending);
}
'''
(build/'herald_native.cpp').write_text(prefix+fn+suffix)
cxx=Path(sys.prefix)/'bin/x86_64-conda-linux-gnu-g++'
cmd=[str(cxx),'-O3','-std=c++17','-shared','-fPIC','-pthread',
 '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),'-I'+str(UP/'include'),
 str(build/'herald_native.cpp'),str(UP/'src/array.cc'),str(UP/'src/thread_pool.cc'),
 '-o',str(build/('herald_native'+sysconfig.get_config_var('EXT_SUFFIX')))]
subprocess.run(cmd,check=True)
(build/'provenance.json').write_text(json.dumps({'command':cmd,'source_hashes':{
 str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [UP/'src/laia_scheduler.cc',UP/'include/mini_lru_cache.h',build/'herald_native.cpp']},
 'get_dist_change':'boost::container::flat_set -> std::set only',
 'mode':'default non-local-shared LaiaScheduler, all original access slots; original cache, arrays, threadpool'},indent=2))
print('Built',build)
