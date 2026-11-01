#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cassert>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>
template<class T>struct Map{T* p;size_t n;Map(const char*path){int fd=open(path,O_RDONLY);assert(fd>=0);struct stat s;fstat(fd,&s);n=s.st_size/sizeof(T);p=(T*)mmap(0,s.st_size,PROT_READ,MAP_SHARED,fd,0);close(fd);assert(p!=MAP_FAILED);}};
int main(int argc,char**argv){assert(argc==5);Map<uint32_t>raw(argv[1]),freq(argv[2]);Map<uint64_t>base(argv[3]);std::string out=argv[4];const int N=2097152,W=228,cut=210;const int ls[]={3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1};size_t M=freq.n;assert(M<UINT32_MAX);std::vector<uint64_t> rp(N+1),pp(M+1);std::vector<uint32_t> ids;ids.reserve(size_t(N)*180);std::vector<uint32_t> targets;
for(size_t u=0;u<M;u++){pp[u+1]=pp[u]+(freq.p[u]>0&&freq.p[u]<cut?freq.p[u]:0);if(freq.p[u]>=10&&freq.p[u]<cut)targets.push_back(u);}std::vector<uint32_t> posts(pp.back());std::vector<uint64_t> cursor=pp;
for(int r=0;r<N;r++){rp[r]=ids.size();int col=14;for(int t=0;t<26;t++){uint32_t tmp[100];for(int j=0;j<ls[t];j++)tmp[j]=raw.p[size_t(r)*W+col+j];std::sort(tmp,tmp+ls[t]);auto end=std::unique(tmp,tmp+ls[t]);for(auto q=tmp;q!=end;q++){uint32_t u=base.p[t]+*q;if(freq.p[u]&&freq.p[u]<cut){ids.push_back(u);posts[cursor[u]++]=r;}}col+=ls[t];}if(r%262144==0)std::cout<<"rows "<<r<<std::endl;}rp[N]=ids.size();for(size_t u=0;u<M;u++)assert(cursor[u]==pp[u+1]);cursor.clear();cursor.shrink_to_fit();std::cout<<"index ready targets "<<targets.size()<<" edges "<<ids.size()<<std::endl;
struct R{uint32_t id,support,best,partner;};std::vector<R> result(targets.size());uint64_t completed=0;
#pragma omp parallel num_threads(16)
{std::vector<uint16_t> counts(M,0);std::vector<uint32_t> touched;touched.reserve(50000);
#pragma omp for schedule(dynamic,128)
for(size_t z=0;z<targets.size();z++){uint32_t u=targets[z],n=freq.p[u],best=0,partner=UINT32_MAX;auto begin=posts.data()+pp[u],end=posts.data()+pp[u+1];uint32_t r=*begin;
// Exact shortcut: a partner containing every target occurrence gives maximum MCR=1.
for(uint64_t j=rp[r];j<rp[r+1];j++){uint32_t v=ids[j];if(v==u||freq.p[v]<n)continue;auto vb=posts.data()+pp[v],ve=posts.data()+pp[v+1];bool all=true;for(auto p=begin+1;p!=end;p++)if(!std::binary_search(vb,ve,*p)){all=false;break;}if(all){best=n;partner=v;break;}}
if(best<n){touched.clear();for(auto p=begin;p!=end;p++){uint32_t row=*p;for(uint64_t j=rp[row];j<rp[row+1];j++){uint32_t v=ids[j];if(v==u||freq.p[v]<=best)continue;if(!counts[v])touched.push_back(v);uint32_t c=++counts[v];if(c>best){best=c;partner=v;}}}for(uint32_t v:touched)counts[v]=0;}result[z]={u,n,best,partner};
#pragma omp atomic update
completed++;
if(z%100000==0){
#pragma omp critical
std::cout<<"completed "<<completed<<" / "<<targets.size()<<std::endl;}}
}
std::ofstream f(out+"/all_targets.u32",std::ios::binary);f.write((char*)result.data(),result.size()*sizeof(R));f.close();for(uint32_t threshold:{10,20,50,100}){uint64_t total=0,high=0;double mean=0;for(auto r:result)if(r.support>=threshold){total++;high+=(uint64_t(r.best)*100>uint64_t(r.support)*95);mean+=double(r.best)/r.support;}std::cout<<"RESULT "<<threshold<<" "<<total<<" "<<high<<" "<<double(high)/total<<" "<<mean/total<<std::endl;}}
