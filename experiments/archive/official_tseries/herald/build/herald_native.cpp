
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
 std::vector<uint64_t> payload_bytes_;
 void set_weights(std::vector<uint64_t> w) {payload_bytes_=std::move(w);}
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
  auto bytes=counts;
  auto pool=ThreadPool::Get();
  std::vector<std::future<void>> futures;
  for(size_t r=0;r<num_worker_;++r) futures.push_back(pool->Enqueue([&,r] {
   auto &c=counts[r];auto &bc=bytes[r];c[3]=plan[r].size();c[4]=keys[r].size();
   for(auto k:plan[r]) bc[3]+=payload_bytes_.at(k);
   for(auto k:keys[r]) bc[4]+=payload_bytes_.at(k);
   for(auto k:plan[r]) cache_snaps_[r].outdate(k);
   for(auto k:keys[r]) {
    int status=cache_snaps_[r].get(k);
    if(status==-2) {++c[1];bc[1]+=payload_bytes_.at(k);}
    else if(status>=0) {++c[0];bc[0]+=payload_bytes_.at(k); if(status>0) {++c[2];bc[2]+=payload_bytes_.at(cache_snaps_[r].last_evicted_key);}}
   }
  }));
  pool->Wait(num_worker_);for(auto &f:futures) f.get();
  py::dict result;result["counts"]=counts;result["bytes"]=bytes;result["assignment"]=assignment;return result;
 }
 std::vector<uint64_t> pending_bytes() {
  std::vector<uint64_t> result;for(auto &c:cache_snaps_) {uint64_t n=0;for(auto k:c.get_keys()) n+=payload_bytes_.at(k);result.push_back(n);}return result;
 }
 std::vector<uint64_t> pending() {
  std::vector<uint64_t> result;
  for(auto &c:cache_snaps_) result.push_back(c.get_keys().size());
  return result;
 }
};
void LaiaScheduler::get_dist(comm_plan_t &cplan, dist_t &dist) {
    auto thread_pool = ThreadPool::Get();

    // reset variables
    count_.reset(0);
    scores_.reset(0);
    sample_emb_dep_.reset(0);
    workers_workload_.clear();
    workers_workload_.resize(num_worker_);

    // current batch
    int batch_start_idx = (batch_id_ * batch_size_) % num_sample_;
    vector<emb_key_t *> batch_sample;
    for (size_t i = batch_start_idx; i < batch_start_idx + batch_size_; i++) {
        batch_sample.emplace_back(sample_embs_[i % num_sample_]);
    }

    // sort table as max frequecies
    // std::nth_element(table_frequency_.begin(),
    //                  table_frequency_.begin() + top_k_table_,
    //                  table_frequency_.end());

    // score
    for (size_t t = 0; t < num_thread_; t++) {
        thread_pool->Enqueue([&, t]() -> void {
            int x = batch_size_ / num_thread_;
            int y = batch_size_ % num_thread_;
            int start = (t == 0) ? 0 : (y + t * x);
            int end = (t == 0) ? (x + y) : (start + x);
            for (int i = start; i < end; i++) {
                for (size_t j = 0; j < num_table_; j++) {
                // for (int k = 0; k < top_k_table_; k++) {
                    // int j = table_frequency_[k].index;
                    auto emb = batch_sample[i][j];
                    // table_frequency_[k].record(emb);
                    bool find_flag = false;
                    for (size_t z = 0; z < num_worker_; z++) {
                        if (cache_snaps_[z].check(emb)) {
                            scores_[i][z] += 1;
                            sample_emb_dep_(i, z, count_[i][z]) = emb;
                            count_[i][z] += 1;
                            find_flag = true;
                        }
                    }
                    if (!find_flag) {
                        sample_emb_dep_(i, num_worker_,
                                        count_[i][num_worker_]) = emb;
                        count_[i][num_worker_] += 1;
                    }
                }
            }
        });
    }
    thread_pool->Wait(num_thread_);

    // distribution
    vector<unordered_set<emb_key_t>> dist_keys;
    dist_keys.resize(num_worker_);
    int max_workload = mini_batch_size_;
    dist.reset(0);
    for (size_t i = 0; i < batch_size_; i++) {
        int max_score = -1;
        int max_score_worker = -1;
        for (size_t j = 0; j < num_worker_; j++) {
            int worker = (j + batch_id_) % num_worker_;
            int score = scores_[i][worker];
            if (workers_workload_[worker] < max_workload) {
                if (max_score < score) {
                    max_score = score;
                    max_score_worker = worker;
                }
            }
        }
        dist[max_score_worker][workers_workload_[max_score_worker]] =
            (i + batch_start_idx) % num_sample_;
        dist_keys[max_score_worker].emplace((i + batch_start_idx)
                                            % num_sample_);
        workers_workload_[max_score_worker] += 1;
    }

    // communication plan
    for (size_t i = 0; i < num_worker_; i++) {
        std::set<emb_key_t> *plan_i;
        plan_i = &cplan[i];
        thread_pool->Enqueue([&, i, plan_i]() -> void {
            plan_i->clear();
            for (size_t s = 0; s < batch_size_; s++) {
                bool in_flag = false;
                auto position = (s + batch_start_idx) % num_sample_;
                in_flag = (dist_keys[i].find(position) != dist_keys[i].end());
                if (!in_flag) {
                    for (uint64_t j = 0; j < count_[s][i]; j++) {
                        auto emb = sample_emb_dep_(s, i, j);
                        plan_i->emplace(emb);
                    }
                }
            }
        });
    }
    thread_pool->Wait(num_worker_);
}


}
PYBIND11_MODULE(herald_native,m) {
 py::class_<laia_cache::LaiaScheduler>(m,"Scheduler")
 .def(py::init<size_t,size_t,size_t,size_t,size_t>(),py::arg("p"),py::arg("b"),py::arg("cols"),py::arg("cap"),py::arg("universe")=0)
 .def("step",&laia_cache::LaiaScheduler::step)
 .def("pending",&laia_cache::LaiaScheduler::pending)
 .def("set_weights",&laia_cache::LaiaScheduler::set_weights)
 .def("pending_bytes",&laia_cache::LaiaScheduler::pending_bytes);
}
