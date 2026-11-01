#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
extern "C" void* build_trace(long n,int g,int nb,const uint64_t* p,const uint32_t* gids,int64_t* counts){
 std::vector<std::vector<uint32_t>>q(g*nb);for(long i=0;i<n;i++)for(int r=0;r<g;r++){uint64_t bits=p[i*g+r];while(bits){int b=__builtin_ctzll(bits);q[b*g+r].push_back(gids[i]);bits&=bits-1;}}
 size_t total=0;for(int j=0;j<g*nb;j++){counts[j]=q[j].size();total+=q[j].size();}auto out=(uint32_t*)std::malloc(std::max(size_t(1),total*sizeof(uint32_t)));if(!out)return nullptr;size_t off=0;for(auto&v:q){if(!v.empty())std::memcpy(out+off,v.data(),v.size()*sizeof(uint32_t));off+=v.size();}return out;
}
extern "C" void free_trace(void*p){std::free(p);}
