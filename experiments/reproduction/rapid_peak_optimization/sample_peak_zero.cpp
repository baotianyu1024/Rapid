#include "../speed_work/fast/swap_optimizer_sample.h"
using namespace cps_swap;
extern "C" int sample_peak_zero(const uint32_t* records,int g,int b,int width,int dense,int nt,const int* lengths,uint32_t* assignment,int rounds,int64_t* final_f,int64_t* final_s){
 Samples samples(g*b);int off=1+dense;
 for(int t=0;t<nt;t++){for(int i=0;i<g*b;i++)for(int j=0;j<lengths[t];j++){uint32_t id=records[(size_t)i*width+off+j];samples[i].push_back({(uint64_t(t)<<32)|id,id%uint32_t(g)});}off+=lengths[t];}
 for(auto&v:samples)canonicalize(v);
 Buckets buckets(g);for(int r=0;r<g;r++)for(int j=0;j<b;j++)buckets[r].push_back(r*b+j);
 State state(samples,buckets);std::vector<int64_t>send(g,0);
 for(int r=0;r<g;r++){std::unordered_set<uint64_t>seen;for(auto i:buckets[r])for(auto id:samples[i])if(id.owner!=uint32_t(r)&&seen.insert(id.key).second)send[id.owner]++;}
 const int64_t fcap=*std::max_element(state.volumes().begin(),state.volumes().end()),scap=*std::max_element(send.begin(),send.end());int accepted=0;
 auto db=[&](int r,const Sample&o,const Sample&in,std::vector<int64_t>&delta){size_t i=0,j=0;while(i<o.size()||j<in.size()){if(j==in.size()||(i<o.size()&&o[i].key<in[j].key)){auto id=o[i++];if(id.owner!=uint32_t(r)&&state.count(r,id.key)==1)delta[id.owner]--;}else if(i==o.size()||in[j].key<o[i].key){auto id=in[j++];if(id.owner!=uint32_t(r)&&state.count(r,id.key)==0)delta[id.owner]++;}else{i++;j++;}}};
 for(int round=0;round<rounds;round++){
  std::vector<int>ranks(g);std::iota(ranks.begin(),ranks.end(),0);std::stable_sort(ranks.begin(),ranks.end(),[&](int a,int d){return state.volumes()[a]>state.volumes()[d];});
  for(int si=0;si<std::min(g,2);si++)for(int di=0;di<std::min(g,4);di++){
   int a=ranks[si],d=ranks[g-1-di];if(a==d)continue;
   std::vector<std::pair<int,int>>xs,ys;
   for(int k=0;k<std::min(b,4096);k++){
    int x=(k*7919+round*997+si*131+di*17)%b;
    xs.push_back({int(state.insertion_cost(d,samples[state.buckets()[a][x]]))-int(state.removable(a,x)),x});
    ys.push_back({int(state.insertion_cost(a,samples[state.buckets()[d][x]]))-int(state.removable(d,x)),x});
   }
   int nx=std::min(int(xs.size()),64),ny=std::min(int(ys.size()),32);
   std::partial_sort(xs.begin(),xs.begin()+nx,xs.end());std::partial_sort(ys.begin(),ys.begin()+ny,ys.end());
   for(int xi=0;xi<nx;xi++){
    int x=xs[xi].second,besty=-1;double bestcost=0;std::vector<int64_t>bestds;
    double norm=std::accumulate(state.volumes().begin(),state.volumes().end(),0.0)/g;
    int64_t ccap=0;for(int r=0;r<g;r++)ccap=std::max(ccap,int64_t(state.volumes()[r])+send[r]);
    for(int yi=0;yi<ny;yi++){
     int y=ys[yi].second;auto df=state.swap_delta(a,x,d,y);int64_t gain=-df.first-df.second;if(gain<0)continue;
     if(int64_t(state.volumes()[a])+df.first>fcap || int64_t(state.volumes()[d])+df.second>fcap)continue;
     std::vector<int64_t>ds(g,0);auto&left=samples[state.buckets()[a][x]];auto&right=samples[state.buckets()[d][y]];db(a,left,right,ds);db(d,right,left,ds);
     double cost=-double(gain);bool ok=true;
     for(int r=0;r<g;r++){
      int64_t dr=(r==a?df.first:0)+(r==d?df.second:0);double f=state.volumes()[r],s=send[r];
      if(s+ds[r]>scap || f+dr+s+ds[r]>ccap){ok=false;break;}
      cost+=8*((2*f*dr+dr*dr)+(2*s*ds[r]+ds[r]*ds[r]))/norm;
     }
     if(ok&&cost<bestcost){bestcost=cost;besty=y;bestds=ds;}
    }
    if(besty>=0){state.apply_swap(a,x,d,besty);for(int r=0;r<g;r++)send[r]+=bestds[r];accepted++;}
   }
  }
 }
 for(int r=0;r<g;r++){final_f[r]=state.volumes()[r];final_s[r]=send[r];for(int j=0;j<b;j++)assignment[r*b+j]=state.buckets()[r][j];}
 return accepted;
}
