#include "../working/cps/swap_optimizer.h"
#include <cassert>
#include <iostream>
int main() {
    cps_swap::Samples samples(8);
    for (auto& sample : samples)
        for (uint32_t id=0; id<4; ++id) sample.push_back({id,id});
    cps_swap::Buckets initial(4);
    for (uint32_t r=0;r<4;++r) {initial[r].push_back(2*r);initial[r].push_back(2*r+1);}
    cps_swap::Config config;
    config.algorithm=cps_swap::Algorithm::Ratio;
    config.stop_at_target=true;
    auto result=cps_swap::optimize(samples,initial,config);
    assert(result.before==std::vector<uint64_t>({3,3,3,3}));
    assert(result.after==result.before && result.buckets==initial);
    assert(result.stats.target_met && result.stats.stop_reason=="target_reached");
    assert(result.evaluations==0 && result.stats.screened_id_queries==0);
    std::cout << "PASS: an already balanced nonzero trace stops with zero search and unchanged samples\n";
}
