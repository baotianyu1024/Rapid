#include <cstdint>
extern "C" void scores(long n,int g,int nb,const uint64_t* profiles,const int16_t* owners,uint16_t* score){
 for(long i=0;i<n;i++){
  uint64_t active[64]={};for(int r=0;r<g;r++){uint64_t bits=profiles[i*g+r];while(bits){int b=__builtin_ctzll(bits);active[b]|=uint64_t(1)<<r;bits&=bits-1;}}
  int last=-1;int nonempty[64],na=0;for(int b=0;b<nb;b++)if(active[b])nonempty[na++]=b;
  for(int e=0;e<2;e++)for(int j=0;j<na;j++){int b=nonempty[j];
   auto a=active[b];if(!a)continue;
   if(last>=0&&last!=owners[i]&&(a&(uint64_t(1)<<last)))score[(long)last*n+i]+=(e?4:1);
   last=__builtin_popcountll(a)==1?__builtin_ctzll(a):-1;
  }
 }
}
extern "C" void aggregate(long n,int g,int nb,const uint64_t* profiles,const int16_t* owners,const uint64_t* cached,int64_t* hr,int64_t* hs){
 for(long i=0;i<n;i++){
  uint64_t active[64]={};for(int r=0;r<g;r++){uint64_t bits=profiles[i*g+r];while(bits){int b=__builtin_ctzll(bits);active[b]|=uint64_t(1)<<r;bits&=bits-1;}}
  int last=-1;
  for(int e=0;e<5;e++)for(int b=0;b<nb;b++){
   auto a=active[b];if(!a)continue;
   if(last>=0&&(a&cached[i]&(uint64_t(1)<<last))){hr[(e*nb+b)*g+last]++;hs[(e*nb+b)*g+owners[i]]++;}
   last=__builtin_popcountll(a)==1?__builtin_ctzll(a):-1;
  }
 }
}
