#include <cstdint>
extern "C" void oracle(long n,int g,int nb,const uint64_t* profiles,const int16_t* owners,const uint64_t* cached,int64_t* hr,int64_t* hs){
 for(long i=0;i<n;i++){
  uint64_t active[64]={};int version[64];for(int r=0;r<g;r++){version[r]=-1;uint64_t bits=profiles[i*g+r];while(bits){int b=__builtin_ctzll(bits);active[b]|=uint64_t(1)<<r;bits&=bits-1;}}
  int master=0;
  for(int e=0;e<5;e++)for(int b=0;b<nb;b++){
   uint64_t bits=active[b]&cached[i];while(bits){int r=__builtin_ctzll(bits);if(version[r]==master){hr[(e*nb+b)*g+r]++;hs[(e*nb+b)*g+owners[i]]++;}else version[r]=master;bits&=bits-1;}
   master+=__builtin_popcountll(active[b]);bits=active[b]&cached[i];while(bits){int r=__builtin_ctzll(bits);version[r]++;bits&=bits-1;}
  }
 }
}
