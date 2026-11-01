#include "../speed_work/fast/swap_optimizer_sample.h"
#include <random>
#include <cstdio>
using namespace cps_swap;
extern "C" int greedy_samples(const uint32_t* records,int g,int b,int width,int dense,int nt,const int* lengths,uint32_t* assignment,int seed,double loadweight,double balance,int64_t* final_f,int64_t* final_s){
 Samples samples(g*b);int off=1+dense;
 for(int t=0;t<nt;t++){for(int i=0;i<g*b;i++)for(int j=0;j<lengths[t];j++){uint32_t id=records[(size_t)i*width+off+j];samples[i].push_back({(uint64_t(t)<<32)|id,id%uint32_t(g)});}off+=lengths[t];}
 for(auto&v:samples)canonicalize(v);
 std::vector<FlatCounts>counts(g);std::vector<int>load(g,0);std::vector<int64_t>f(g,0),s(g,0);std::vector<uint32_t>order(g*b);std::iota(order.begin(),order.end(),0);std::mt19937 rng(seed);std::shuffle(order.begin(),order.end(),rng);
 for(uint32_t i:order){
  std::vector<int>hits(g,0),ranks(g);for(auto id:samples[i])hits[id.owner]++;std::iota(ranks.begin(),ranks.end(),0);
  std::stable_sort(ranks.begin(),ranks.end(),[&](int a,int d){return hits[a]>hits[d];});std::vector<int>cand;
  for(int k=0;k<std::min(g,8);k++)cand.push_back(ranks[k]);
  std::stable_sort(ranks.begin(),ranks.end(),[&](int a,int d){return load[a]<load[d];});for(int k=0;k<std::min(g,4);k++)cand.push_back(ranks[k]);cand.push_back(i/b);cand.push_back(rng()%g);
  std::sort(cand.begin(),cand.end());cand.erase(std::unique(cand.begin(),cand.end()),cand.end());
  double norm=std::max(1.0,std::accumulate(f.begin(),f.end(),0.0)/g);double best=1e99;int dest=-1;
  for(int r:cand)if(load[r]<b){double cost=loadweight*double(load[r])/b;for(auto id:samples[i])if(id.owner!=uint32_t(r)&&counts[r].find(id.key)==counts[r].end())cost+=1+balance*(f[r]+s[id.owner])/norm;
   if(cost<best){best=cost;dest=r;}}
  if(dest<0)for(int r=0;r<g;r++)if(load[r]<b){dest=r;break;}
  for(auto id:samples[i])if(id.owner!=uint32_t(dest)){auto&c=counts[dest][id.key];if(c++==0){f[dest]++;s[id.owner]++;}}
  assignment[dest*b+load[dest]++]=i;
 }
 for(int r=0;r<g;r++){final_f[r]=f[r];final_s[r]=s[r];}return g*b;
}
