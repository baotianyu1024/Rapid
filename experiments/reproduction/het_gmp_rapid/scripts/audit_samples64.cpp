// Verify conservation of complete records after RAPID's documented ID remap.
// Two independent 64-bit fingerprints, compared as sorted multisets.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using H=std::array<uint64_t,2>;
int main(int argc,char**argv){
 if(argc!=4)throw std::runtime_error("input.bin rapid_dir output.json");
 const int P=64, W=228;
 std::vector<int> lens={3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1};
 std::vector<int> table;for(int t=0;t<26;t++)for(int k=0;k<lens[t];k++)table.push_back(t);
 std::vector<std::vector<std::pair<uint32_t,int>>> entries(26);
 std::vector<uint64_t> owners(P,0);
 for(int r=0;r<P;r++){
  std::ifstream f(std::string(argv[2])+"/placement/table_"+std::to_string(r));if(!f)throw std::runtime_error("placement missing");
  std::string line;while(std::getline(f,line)){auto k=line.find(':');int t=std::stoi(line.substr(0,k));uint32_t id=std::stoul(line.substr(k+1));entries.at(t).push_back({id,r});owners[r]++;}
 }
 std::vector<std::vector<uint32_t>> mapping(26);
 for(int t=0;t<26;t++){
  auto&e=entries[t];std::sort(e.begin(),e.end());if(e.empty())throw std::runtime_error("empty table");
  mapping[t].resize(size_t(e.back().first)+1,UINT32_MAX);std::vector<uint32_t> counters(P,0);
  for(auto[id,r]:e){if(mapping[t][id]!=UINT32_MAX)throw std::runtime_error("duplicate placement");mapping[t][id]=counters[r]++*P+r;}
 }
 auto read=[&](std::string path,bool remap,std::vector<H>&out){
  std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("input missing");
  std::vector<uint32_t> buf(W*8192);uint64_t n=0;
  while(f.read((char*)buf.data(),buf.size()*4)||f.gcount()){
   auto bytes=f.gcount();if(bytes%(W*4))throw std::runtime_error("partial row");size_t rows=bytes/(W*4);
   for(size_t i=0;i<rows;i++){
    uint64_t a=1469598103934665603ULL,b=0x9e3779b97f4a7c15ULL;
    for(int c=0;c<W;c++){
     uint32_t v=buf[i*W+c];if(remap&&c>=14){auto&m=mapping[table[c-14]];if(v>=m.size()||m[v]==UINT32_MAX)throw std::runtime_error("unmapped input");v=m[v];}
     a=(a^v)*1099511628211ULL;b^=uint64_t(v)+0x9e3779b97f4a7c15ULL+(b<<6)+(b>>2);
    }out.push_back({a,b});
   }n+=rows;
  }return n;
 };
 std::vector<H> raw,mapped;auto n=read(argv[1],true,raw);std::vector<uint64_t> counts;
 for(int r=0;r<P;r++)counts.push_back(read(std::string(argv[2])+"/scheduled/dataset_"+std::to_string(r)+".bin",false,mapped));
 const size_t B=8192, NB=n/(P*B);bool batches_ok=true;
 if(n!=NB*P*B)throw std::runtime_error("incomplete global batch");
 for(size_t b=0;b<NB;++b){
  std::vector<H> lhs(raw.begin()+b*P*B,raw.begin()+(b+1)*P*B),rhs;
  for(int r=0;r<P;++r){auto beg=mapped.begin()+r*NB*B+b*B;rhs.insert(rhs.end(),beg,beg+B);}
  std::sort(lhs.begin(),lhs.end());std::sort(rhs.begin(),rhs.end());batches_ok &= lhs==rhs;
 }
 if(!batches_ok)throw std::runtime_error("global batch membership changed");
 std::sort(raw.begin(),raw.end());std::sort(mapped.begin(),mapped.end());bool ok=raw==mapped;
 std::ofstream out(argv[3]);out<<"{\"rows\":"<<n<<",\"global_batch_membership_preserved\":true,\"matching_128bit_record_multiset\":"<<(ok?"true":"false")<<",\"rank_rows\":[";
 for(int r=0;r<P;r++)out<<(r?",":"")<<counts[r];out<<"],\"owner_counts\":[";for(int r=0;r<P;r++)out<<(r?",":"")<<owners[r];out<<"]}\n";
 if(!ok)throw std::runtime_error("sample conservation FAILED");std::cout<<"PASS "<<n<<" records, all mapped, 128-bit multiset parity\n";
}
