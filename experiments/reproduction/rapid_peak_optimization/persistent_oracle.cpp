#include <cstdint>
extern "C" void replay(long n,int g,int nb,const uint64_t* profiles,const int16_t* owners,const uint32_t* gids,const uint64_t* cached,const uint64_t* valid,int64_t* master,uint64_t* finalvalid,int64_t* forward,int64_t* backward,int64_t* oldnew,long oldn){
 for(long i=0;i<n;i++){
  uint32_t id=gids[i];uint64_t active[64]={};int64_t ver[64];auto mask=cached[id];auto initial=valid[id];auto clock=master[id];int o=owners[i];
  for(int r=0;r<g;r++){ver[r]=(initial&(uint64_t(1)<<r))?clock:-1;auto bits=profiles[i*g+r];while(bits){int b=__builtin_ctzll(bits);active[b]|=uint64_t(1)<<r;bits&=bits-1;}}
  for(int e=0;e<5;e++)for(int b=0;b<nb;b++){
   auto remote=active[b]&~(uint64_t(1)<<o);auto bits=remote;long step=e*nb+b;
   while(bits){int r=__builtin_ctzll(bits);auto bit=uint64_t(1)<<r;bool hit=(mask&bit)&&ver[r]==clock;backward[(step*g+r)*g+o]++;if(!hit)forward[(step*g+r)*g+o]++;if(e==0)oldnew[id>=oldn]++;if(mask&bit)ver[r]=clock;bits&=bits-1;}
   clock+=__builtin_popcountll(active[b]);bits=remote&mask;while(bits){int r=__builtin_ctzll(bits);ver[r]++;bits&=bits-1;}
  }
  master[id]=clock;auto bits=mask;uint64_t result=0;while(bits){int r=__builtin_ctzll(bits);if(ver[r]==clock)result|=uint64_t(1)<<r;bits&=bits-1;}finalvalid[id]=result;
 }
}
