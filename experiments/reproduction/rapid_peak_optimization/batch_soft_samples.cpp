#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <cstdio>
#include <omp.h>
// Fixed owners, exact sample capacity. Snapshot proposals committed as balanced
// pairs/three-cycles; counts rebuilt between rounds, with no shared-state races.
extern "C" int batch_soft_samples(const uint32_t* data,int n,int width,int m,int g,int b,const int16_t* owner,int32_t* assignment,int rounds,double balance){
 std::vector<uint32_t>cnt((size_t)m*g,0);std::vector<double>soft(n+1),delta(n+1);
 double cap=double(n)/g;for(int k=0;k<=n;k++)soft[k]=k>=cap?1:-std::expm1(b*std::log1p(-k/cap));
 for(int k=0;k<n;k++)delta[k]=soft[k+1]-soft[k];
 struct Move{int i;float gain;}; int allmoves=0;
 for(int pass=0;pass<rounds;pass++){
  std::fill(cnt.begin(),cnt.end(),0);std::vector<std::vector<int>>rows(g);
  for(int i=0;i<n;i++)rows[assignment[i]].push_back(i);
  #pragma omp parallel for num_threads(16)
  for(int r=0;r<g;r++)for(int i:rows[r])for(int j=0;j<width;j++){uint32_t id=data[(size_t)i*width+j];if(j && id==data[(size_t)i*width+j-1])continue;cnt[(size_t)id*g+r]++;}
  std::vector<double>f(g,0),s(g,0);
  for(int id=0;id<m;id++)for(int r=0;r<g;r++)if(r!=owner[id]){double v=soft[cnt[(size_t)id*g+r]];f[r]+=v;s[owner[id]]+=v;}
  double mean=std::accumulate(f.begin(),f.end(),0.0)/g;
  std::vector<int>target(n,-1);std::vector<float>gain(n,0);
  #pragma omp parallel for num_threads(16) schedule(static)
  for(int i=0;i<n;i++){
   int old=assignment[i];std::vector<double>cost(g,0);
   for(int j=0;j<width;j++){
    uint32_t id=data[(size_t)i*width+j];if(j && id==data[(size_t)i*width+j-1])continue;int o=owner[id];
    for(int r=0;r<g;r++)if(r!=o){uint32_t c=cnt[(size_t)id*g+r]-(r==old);cost[r]+=delta[c]*(1+balance*(f[r]+s[o])/mean);}
   }
   int dest=std::min_element(cost.begin(),cost.end())-cost.begin();if(cost[old]>cost[dest]+1e-7){target[i]=dest;gain[i]=cost[old]-cost[dest];}
  }
  int moves=0;int nb=n/(g*b),nr=n/g;
  std::vector<std::vector<Move>>q(nb*g*g);for(int i=0;i<n;i++)if(target[i]>=0){int batch=(i%nr)/b;q[(batch*g+assignment[i])*g+target[i]].push_back({i,gain[i]});}
  for(auto&v:q)std::sort(v.begin(),v.end(),[](Move a,Move b){return a.gain<b.gain;});
  for(int batch=0;batch<nb;batch++){
   auto queue=[&](int a,int d)->std::vector<Move>&{return q[(batch*g+a)*g+d];};
   auto take=[&](int a,int d){auto&v=queue(a,d);int i=v.back().i;v.pop_back();assignment[i]=d;moves++;};
   for(int a=0;a<g;a++)for(int d=a+1;d<g;d++)while(!queue(a,d).empty()&&!queue(d,a).empty()){take(a,d);take(d,a);}
   for(int a=0;a<g;a++)for(int d=0;d<g;d++)if(a!=d)for(int e=0;e<g;e++)if(e!=a&&e!=d)while(!queue(a,d).empty()&&!queue(d,e).empty()&&!queue(e,a).empty()){take(a,d);take(d,e);take(e,a);}
  }
  allmoves+=moves;printf("fixed-batch soft round %d moves %d estimated_mean %.2f recv_peak %.2f service_peak %.2f\n",pass,moves,mean,*std::max_element(f.begin(),f.end()),*std::max_element(s.begin(),s.end()));fflush(stdout);
 }
 return allmoves;
}
