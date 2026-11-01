#include <cstdint>
extern "C" void cache_masks(long n,int g,int nb,const uint64_t* profiles,uint64_t* cold,uint64_t* warm){
 for(long i=0;i<n;i++){
  uint64_t active[64]={};for(int r=0;r<g;r++){uint64_t bits=profiles[i*g+r];while(bits){int b=__builtin_ctzll(bits);active[b]|=uint64_t(1)<<r;bits&=bits-1;}}
  int last=-1;
  for(int e=0;e<2;e++)for(int b=0;b<nb;b++){
   auto a=active[b];if(!a)continue;
   if(last>=0 && (a&(uint64_t(1)<<last)))(e?warm:cold)[i*g+last]|=uint64_t(1)<<b;
   last=__builtin_popcountll(a)==1?__builtin_ctzll(a):-1;
  }
 }
}
