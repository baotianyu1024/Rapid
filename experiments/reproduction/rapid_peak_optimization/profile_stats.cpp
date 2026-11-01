#include <cstdint>
extern "C" void stats(long n,int g,int nb,const uint64_t* p,const int16_t* own,uint16_t* weights,int64_t* rd,int64_t* sd){
 for(long i=0;i<n;i++)for(int r=0;r<g;r++){uint64_t bits=p[i*g+r];while(bits){int b=__builtin_ctzll(bits);bits&=bits-1;weights[i*nb+b]++;if(r!=own[i]){rd[b*g+r]++;sd[b*g+own[i]]++;}}}
}
