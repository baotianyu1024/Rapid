#include "../speed_work/fast/flat_counts.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstdint>
#include <cmath>
// Existing owners (plus already placed new hot IDs) are read-only.
extern "C" void initial_schedule(const uint32_t* records,int n,int g,int b,int width,int dense,int nt,const int* lengths,const int16_t** owners,const int64_t* sizes,int seed,int32_t* assignment,int64_t* rd,int64_t* sd){
 int nb=n/(g*b);
 #pragma omp parallel for num_threads(8)
 for(int batch=0;batch<nb;batch++){
  int m=g*b,start=batch*m;std::vector<uint16_t>hits((size_t)m*g,0);int col=1+dense;
  for(int t=0;t<nt;t++){for(int i=0;i<m;i++)for(int j=0;j<lengths[t];j++){uint32_t id=records[(size_t)(start+i)*width+col+j];int r=id<uint64_t(sizes[t])?owners[t][id]:-1;if(r>=0)hits[(size_t)i*g+r]++;}col+=lengths[t];}
  std::vector<int>order(m),load(g,0);std::iota(order.begin(),order.end(),0);std::mt19937 rng(seed+batch);std::shuffle(order.begin(),order.end(),rng);
  for(int i:order){int best=-1;double score=-1e99;int rotate=rng()%g;for(int q=0;q<g;q++){int r=(q+rotate)%g;if(load[r]==b)continue;double v=hits[(size_t)i*g+r]-.2*load[r];if(v>score){score=v;best=r;}}assignment[start+i]=best;load[best]++;}
  std::vector<FlatCounts>seen(g);col=1+dense;
  for(int t=0;t<nt;t++){for(int i=0;i<m;i++){int r=assignment[start+i];for(int j=0;j<lengths[t];j++){uint32_t id=records[(size_t)(start+i)*width+col+j];int o=id<uint64_t(sizes[t])?owners[t][id]:-1;if(o<0||o==r)continue;uint64_t key=(uint64_t(t)<<32)|id;if(seen[r].find(key)==seen[r].end()){seen[r][key]=1;rd[(batch*nt+t)*g+r]++;sd[(batch*nt+t)*g+o]++;}}}col+=lengths[t];}
 }
}
extern "C" void new_profiles(const uint32_t* records,int n,int g,int b,int width,int dense,int nt,const int* lengths,const int32_t** lookup,const int64_t* sizes,const int32_t* assignment,uint64_t* profiles){
 std::vector<std::vector<int>>rows(g);for(int i=0;i<n;i++)rows[assignment[i]].push_back(i);
 #pragma omp parallel for num_threads(16)
 for(int r=0;r<g;r++)for(int i:rows[r]){uint64_t bit=uint64_t(1)<<(i/(g*b));int col=1+dense;for(int t=0;t<nt;t++){for(int j=0;j<lengths[t];j++){uint32_t id=records[(size_t)i*width+col+j];int k=id<uint64_t(sizes[t])?lookup[t][id]:-1;if(k>=0)profiles[(size_t)k*g+r]|=bit;}col+=lengths[t];}}
}
extern "C" int place_new(long n,int g,int nb,int nt,const uint8_t* tables,const uint64_t* profiles,int64_t* capacity,int16_t* owner,int64_t* rd,int64_t* sd){
 std::vector<int64_t>fr(nb*g,0),fs(nb*g,0);std::vector<double>norm(nb,0);for(int b=0;b<nb;b++){for(int t=0;t<nt;t++)for(int r=0;r<g;r++){fr[b*g+r]+=rd[(b*nt+t)*g+r];fs[b*g+r]+=sd[(b*nt+t)*g+r];}for(int r=0;r<g;r++)norm[b]+=fr[b*g+r]/double(g);norm[b]=std::max(1.,norm[b]);}
 std::vector<std::vector<long>>order(nb*g+1);for(long i=0;i<n;i++){int mass=0;for(int r=0;r<g;r++)mass+=__builtin_popcountll(profiles[i*g+r]);if(!mass)return -1;order[mass].push_back(i);}
 for(int mass=nb*g;mass>=1;mass--)for(long i:order[mass]){
  auto p=profiles+i*g;int w[64]={},active[64],na=0;for(int r=0;r<g;r++){auto bits=p[r];while(bits){int b=__builtin_ctzll(bits);w[b]++;bits&=bits-1;}}for(int b=0;b<nb;b++)if(w[b])active[na++]=b;
  int best=-1;double bc=1e99;
  if(mass==1)for(int r=0;r<g;r++)if(p[r]&&capacity[r]>0){best=r;break;}
  if(best<0)for(int r=0;r<g;r++)if(capacity[r]>0){double cost=0;for(int ab=0;ab<na;ab++){int b=active[ab],local=(p[r]>>b)&1,k=w[b],service=k-local,change=k-2*local;double f=fr[b*g+r],s=fs[b*g+r],z=norm[b];cost+=(128*(-2*f*local-local+2*s*service+service*service+2*(f+s+local)*change+change*change)/(z*z)+10.*service/z)/nb;}if(cost<bc){bc=cost;best=r;}}
  if(best<0)return -2;owner[i]=best;capacity[best]--;int t=tables[i];for(int r=0;r<g;r++)if(r!=best){auto bits=p[r];while(bits){int b=__builtin_ctzll(bits);fr[b*g+r]++;fs[b*g+best]++;rd[(b*nt+t)*g+r]++;sd[(b*nt+t)*g+best]++;bits&=bits-1;}}
 }
 for(int r=0;r<g;r++)if(capacity[r])return -3;return 0;
}
