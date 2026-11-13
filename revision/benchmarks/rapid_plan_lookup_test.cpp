#include <data_readers/multi_hot/detail/rapid_host_plan.hpp>
#include <fstream>
#include <iostream>
#include <vector>
int main() {
  using HugeCTR::MultiHot::RapidHostPlan;
  const std::string root = "/tmp/rapid_fresh_stage/plan_smoke/";
  RapidHostPlan old(root+"plan",8,8192,24,1,"1,1,1,1");
  RapidHostPlan next(root+"plan_v2",8,8192,24,1,"1,1,1,1");
  if (!next.extends(old)) return 2;
  std::ifstream raw(root+"data_2.bin",std::ios::binary), expected(root+"transformed_2.bin",std::ios::binary);
  for(size_t batch=0;batch<2;++batch) {
    std::vector<uint8_t> data(8192*24), want(data.size());
    raw.read(reinterpret_cast<char*>(data.data()),data.size());
    expected.read(reinterpret_cast<char*>(want.data()),want.size());
    std::vector<uint8_t*> shards; std::vector<size_t> sizes(8,1024*24);
    for(size_t r=0;r<8;++r) shards.push_back(data.data()+r*sizes[r]);
    next.apply(batch,shards,sizes);
    if(data!=want) {std::cerr << "Mismatch batch " << batch << '\n';return 3;}
  }
  std::cout << "Both batches match; incremental extension accepted\n";
}
