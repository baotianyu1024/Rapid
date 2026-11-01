#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <cstdio>
#include <omp.h>
#include <queue>
// Fixed owners and original global batches. Greedy capacitated assignment with
// exact traffic acceptance guards; optional damping uses arbitrary-length cycles.
extern "C" int batch_auction_samples(const uint32_t* data,int n,int width,int m,int g,int b,const int16_t* owner,int32_t* assignment,int rounds,double balance){
 std::vector<uint32_t>cnt((size_t)m*g,0);std::vector<double>soft(n+1),delta(n+1);
 double cap=double(n)/g;for(int k=0;k<=n;k++)soft[k]=k>=cap?1:-std::expm1(b*std::log1p(-k/cap));
 for(int k=0;k<n;k++)delta[k]=soft[k+1]-soft[k];
 struct Move{int i;float gain;}; int allmoves=0;int nb=n/(g*b),nr=n/g;
 auto actual=[&](const int32_t* assign){
  std::vector<uint8_t>seen((size_t)m*g,0);std::vector<std::vector<int>>rows(nb*g);for(int i=0;i<n;i++)rows[((i%nr)/b)*g+assign[i]].push_back(i);
  std::vector<int64_t>fr(nb*g,0),services((size_t)nb*g*g,0);
  #pragma omp parallel for num_threads(16)
  for(int r=0;r<g;r++)for(int batch=0;batch<nb;batch++)for(int i:rows[batch*g+r])for(int j=0;j<width;j++){
   uint32_t id=data[(size_t)i*width+j];int o=owner[id];if(o==r)continue;auto&v=seen[(size_t)id*g+r];if(v!=batch+1){v=batch+1;fr[batch*g+r]++;services[((size_t)batch*g+r)*g+o]++;}
  }
  double total=0,peak=0;for(int batch=0;batch<nb;batch++){int64_t mx=0;for(int r=0;r<g;r++){int64_t service=0;for(int q=0;q<g;q++)service+=services[((size_t)batch*g+q)*g+r];mx=std::max(mx,fr[batch*g+r]+service);total+=2*fr[batch*g+r];}peak+=mx;}
  return std::pair<double,double>(total/(nb*g),peak/nb);
 };
 auto previous=actual(assignment);
 for(int pass=0;pass<rounds;pass++){
  std::fill(cnt.begin(),cnt.end(),0);std::vector<std::vector<int>>rows(g);
  for(int i=0;i<n;i++)rows[assignment[i]].push_back(i);
  #pragma omp parallel for num_threads(16)
  for(int r=0;r<g;r++)for(int i:rows[r])for(int j=0;j<width;j++){uint32_t id=data[(size_t)i*width+j];if(j && id==data[(size_t)i*width+j-1])continue;cnt[(size_t)id*g+r]++;}
  std::vector<double>f(g,0),s(g,0);
  for(int id=0;id<m;id++)for(int r=0;r<g;r++)if(r!=owner[id]){double v=soft[cnt[(size_t)id*g+r]];f[r]+=v;s[owner[id]]+=v;}
  double mean=std::accumulate(f.begin(),f.end(),0.0)/g;
  std::vector<double>price(g);for(int r=0;r<g;r++)price[r]=std::pow((f[r]+s[r])/(2*mean),4);std::vector<float>costs((size_t)n*g,0);
  #pragma omp parallel for num_threads(16) schedule(static)
  for(int i=0;i<n;i++){
   int old=assignment[i];std::vector<double>cost(g,0);
   for(int j=0;j<width;j++){
    uint32_t id=data[(size_t)i*width+j];if(j && id==data[(size_t)i*width+j-1])continue;int o=owner[id];
    for(int r=0;r<g;r++)if(r!=o){uint32_t c=cnt[(size_t)id*g+r]-(r==old);cost[r]+=delta[c]*(1+0.5*balance*(price[r]+price[o]));}
   }
   for(int r=0;r<g;r++)costs[(size_t)i*g+r]=cost[r];
  }

  std::vector<int32_t>candidate(n);int moves=0;
  #pragma omp parallel for num_threads(8)
  for(int batch=0;batch<nb;batch++){
   std::vector<int>rows;for(int source=0;source<g;source++)for(int j=0;j<b;j++)rows.push_back(source*nr+batch*b+j);
   struct Bid{float value;int i;bool operator<(const Bid&o)const{return value!=o.value?value>o.value:i>o.i;}};
   std::vector<std::priority_queue<Bid>>accepted(g);std::vector<float>price(g,0);std::vector<int>waiting=rows;size_t head=0;long bids=0;
   while(head<waiting.size() && bids<100L*rows.size()){
    int i=waiting[head++],best=-1;float first=-1e30,second=-1e30;auto costs_i=costs.data()+(size_t)i*g;
    for(int r=0;r<g;r++){float utility=-costs_i[r]-price[r];if(utility>first){second=first;first=utility;best=r;}else if(utility>second)second=utility;}
    float bid=price[best]+first-second+0.1f;accepted[best].push({bid,i});candidate[i]=best;bids++;
    if(accepted[best].size()>size_t(b)){int other=accepted[best].top().i;accepted[best].pop();waiting.push_back(other);}
    if(accepted[best].size()==size_t(b))price[best]=accepted[best].top().value;
   }
   if(head<waiting.size()){for(int i:rows)candidate[i]=assignment[i];printf("auction limit batch %d bids %ld\n",batch,bids);}
   else{for(int r=0;r<g;r++)if(accepted[r].size()!=size_t(b))std::abort();printf("auction batch %d bids %ld\n",batch,bids);}

  }
  auto score=actual(candidate.data());bool accept=score.first<=previous.first && score.second<previous.second;
  printf("full proposal mean %.2f peak %.2f accept %d\n",score.first,score.second,accept);fflush(stdout);
  if(!accept){
   auto full=candidate;
   for(int damp=2;damp<=4&&!accept;damp*=2){
    std::copy(assignment,assignment+n,candidate.begin());long cycle=0;
    for(int batch=0;batch<nb;batch++){
     std::vector<std::vector<int>>out(g);for(int source=0;source<g;source++)for(int j=0;j<b;j++){int i=source*nr+batch*b+j;if(assignment[i]!=full[i])out[assignment[i]].push_back(i);}
     for(int start=0;start<g;start++)while(!out[start].empty()){
      int r=start;std::vector<int>path;do{if(out[r].empty())std::abort();int i=out[r].back();out[r].pop_back();path.push_back(i);r=full[i];}while(r!=start);
      if((cycle++%damp)==0)for(int i:path)candidate[i]=full[i];
     }
    }
    score=actual(candidate.data());printf("damped %d mean %.2f peak %.2f\n",damp,score.first,score.second);fflush(stdout);accept=score.first<=previous.first && score.second<previous.second;
   }
  }
  printf("balanced round %d actual mean %.2f -> %.2f peak %.2f -> %.2f accept %d\n",pass,previous.first,score.first,previous.second,score.second,accept);fflush(stdout);
  if(!accept)break;
  for(int i=0;i<n;i++){moves+=candidate[i]!=assignment[i];assignment[i]=candidate[i];}allmoves+=moves;previous=score;
 }
 return allmoves;
}
