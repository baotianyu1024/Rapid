#include <cstdint>
extern "C" void shared_bound(long n,int g,int nb,const uint64_t* p,const uint16_t*w,int64_t*out){for(long i=0;i<n;i++){uint64_t shared=0;for(int b=0;b<nb;b++)if(w[i*nb+b]>1)shared|=uint64_t(1)<<b;if(!shared)continue;for(int r=0;r<g;r++){uint64_t bits=p[i*g+r]&shared;while(bits){int b=__builtin_ctzll(bits);out[b*g+r]++;bits&=bits-1;}}}}
