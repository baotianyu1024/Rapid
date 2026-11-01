#include "../speed_work/fast/swap_optimizer_sample.h"
#include <cstdio>
#include <cmath>
using namespace cps_swap;
struct GroupCount{uint64_t key;uint32_t owner;int count;};
extern "C" int sample_group_peak(const uint32_t* records,const uint64_t* keys,int g,int b,int width,int rounds,double budget,uint32_t* assignment,int64_t* final_f,int64_t* final_s){
 Samples samples(g*b);for(int i=0;i<g*b;i++)for(int j=1;j<width;j++){uint32_t id=records[(size_t)i*width+j];samples[i].push_back({id,id%uint32_t(g)});}for(auto&v:samples)canonicalize(v);
 Buckets buckets(g);for(int r=0;r<g;r++)for(int j=0;j<b;j++)buckets[r].push_back(r*b+j);State state(samples,buckets);std::vector<int64_t>send(g,0);
 for(int r=0;r<g;r++){std::unordered_set<uint64_t>seen;for(auto i:buckets[r])for(auto id:samples[i])if(id.owner!=uint32_t(r)&&seen.insert(id.key).second)send[id.owner]++;}
 int64_t fcap=*std::max_element(state.volumes().begin(),state.volumes().end()),scap=*std::max_element(send.begin(),send.end()),total0=std::accumulate(send.begin(),send.end(),int64_t(0));long accepted=0;
 auto aggregate=[&](int r,const std::vector<int>&pos){std::vector<Id>all;for(int x:pos){auto&v=samples[state.buckets()[r][x]];all.insert(all.end(),v.begin(),v.end());}std::sort(all.begin(),all.end(),[](Id a,Id d){return a.key<d.key;});std::vector<GroupCount>out;for(auto id:all){if(!out.empty()&&out.back().key==id.key)out.back().count++;else out.push_back({id.key,id.owner,1});}return out;};
 for(int round=0;round<rounds;round++){
  std::vector<int>ranks(g);std::iota(ranks.begin(),ranks.end(),0);std::stable_sort(ranks.begin(),ranks.end(),[&](int a,int d){return state.volumes()[a]+send[a]>state.volumes()[d]+send[d];});int before=accepted;
  double mean=2.*std::accumulate(send.begin(),send.end(),int64_t(0))/g;if(state.volumes()[ranks[0]]+send[ranks[0]]<=1.08*mean)break;
  for(int si=0;si<std::min(g,2);si++)for(int di=0;di<std::min(g,4);di++){
   int a=ranks[si],d=ranks[g-1-di];if(a==d)continue;std::unordered_map<uint64_t,std::vector<int>>ga,gd;
   for(int j=0;j<b;j++){ga[keys[state.buckets()[a][j]]].push_back(j);gd[keys[state.buckets()[d][j]]].push_back(j);}
   uint64_t hotkey=0;size_t hs=0;bool found=false;for(auto&kv:gd){size_t score=kv.second.size()*(ga.count(kv.first)?4:1);if(!found||score>hs||(score==hs&&kv.first<hotkey)){hotkey=kv.first;hs=score;found=true;}}
   if(!found)continue;double la=state.volumes()[a]+send[a],ld=state.volumes()[d]+send[d];std::vector<std::pair<double,int>>returns;
   for(int y:gd[hotkey])returns.push_back({la*state.insertion_cost(a,samples[state.buckets()[d][y]])-ld*state.removable(d,y),y});std::sort(returns.begin(),returns.end());size_t used=0;
   struct Proposal{double score;uint64_t key;};std::vector<Proposal>proposals;
   for(auto&kv:ga)if(kv.first!=hotkey && kv.second.size()<=32){auto ids=aggregate(a,kv.second);int removable=0,insertion=0;for(auto id:ids){if(id.owner!=uint32_t(a)&&state.count(a,id.key)==uint32_t(id.count))removable++;if(id.owner!=uint32_t(d)&&state.count(d,id.key)==0)insertion++;}proposals.push_back({(la*removable-ld*insertion)/std::sqrt(double(kv.second.size())),kv.first});}
   std::sort(proposals.begin(),proposals.end(),[](Proposal x,Proposal y){return x.score!=y.score?x.score>y.score:x.key<y.key;});
   for(int q=0;q<std::min(size_t(256),proposals.size());q++){
    auto&xp=ga[proposals[q].key];int k=xp.size();if(used+k>=returns.size())continue;std::vector<int>yp;for(int j=0;j<k;j++)yp.push_back(returns[used+j].second);auto out=aggregate(a,xp),in=aggregate(d,yp);std::vector<int64_t>ds(g,0);int64_t fa=0,fd=0;size_t x=0,y=0;
    while(x<out.size()||y<in.size()){
     uint64_t key;uint32_t owner;int delta;
     if(y==in.size()||(x<out.size()&&out[x].key<in[y].key)){key=out[x].key;owner=out[x].owner;delta=-out[x++].count;}
     else if(x==out.size()||in[y].key<out[x].key){key=in[y].key;owner=in[y].owner;delta=in[y++].count;}
     else{key=out[x].key;owner=out[x].owner;delta=in[y++].count-out[x++].count;}
     if(!delta)continue;int ca=state.count(a,key),cd=state.count(d,key);require((owner==uint32_t(a)||ca+delta>=0)&&(owner==uint32_t(d)||cd-delta>=0),"group remote count underflow");int da=(owner!=uint32_t(a))*((ca+delta>0)-(ca>0));int dd=(owner!=uint32_t(d))*((cd-delta>0)-(cd>0));fa+=da;fd+=dd;ds[owner]+=da+dd;
    }
    int64_t total=std::accumulate(send.begin(),send.end(),int64_t(0)),ccap=0,pcap=0;for(int r=0;r<g;r++){ccap=std::max(ccap,int64_t(state.volumes()[r])+send[r]);pcap=std::max(pcap,std::max(int64_t(state.volumes()[r]),send[r]));}
    if(total+fa+fd>std::floor(total0*(1+budget)))continue;double cost=fa+fd,norm=std::max(1.,double(total)/g);bool ok=true;
    for(int r=0;r<g;r++){int64_t df=(r==a?fa:0)+(r==d?fd:0);double f=state.volumes()[r],s=send[r];if(f+df>pcap||s+ds[r]>pcap||f+df+s+ds[r]>ccap){ok=false;break;}double change=df+ds[r];cost+=8*(2*(f+s)*change+change*change+2*f*df+df*df+2*s*ds[r]+ds[r]*ds[r])/norm;}
    if(!ok||cost>=0)continue;auto fbefore=state.volumes();for(int j=0;j<k;j++)state.apply_swap(a,xp[j],d,yp[j]);require(int64_t(state.volumes()[a])-fbefore[a]==fa&&int64_t(state.volumes()[d])-fbefore[d]==fd,"group delta mismatch");for(int r=0;r<g;r++)send[r]+=ds[r];used+=k;accepted+=k;
   }
  }
  int64_t peak=0;for(int r=0;r<g;r++)peak=std::max(peak,int64_t(state.volumes()[r])+send[r]);printf("group round %d swaps %ld peak %ld mean %.2f\n",round,accepted-before,peak,2.*std::accumulate(send.begin(),send.end(),int64_t(0))/g);fflush(stdout);if(accepted==before)break;
 }
 for(int r=0;r<g;r++){final_f[r]=state.volumes()[r];final_s[r]=send[r];for(int j=0;j<b;j++)assignment[r*b+j]=state.buckets()[r][j];}return accepted;
}
