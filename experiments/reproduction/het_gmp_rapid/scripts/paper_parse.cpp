#include <array>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("input.tsv output-directory required");
 std::ifstream in(argv[1]);std::string dir=argv[2];std::ofstream out(dir+"/categories_local.bin",std::ios::binary);
 if(!in||!out)throw std::runtime_error("open failed");
 std::array<std::unordered_map<uint64_t,int32_t>,26> maps;
 std::array<int32_t,26> row;std::string line;uint64_t n=0;
 while(std::getline(in,line)){
  size_t a=0;
  for(int col=0;col<40;col++){
   size_t b=line.find('\t',a);if(b==std::string::npos)b=line.size();
   if(col<39&&b==line.size())throw std::runtime_error("short row");
   if(col>=14){
    size_t len=b-a;if(len>8)throw std::runtime_error("hex longer than 8");uint64_t key=0;
    for(size_t j=a;j<b;j++){char c=line[j];int d=(c>='0'&&c<='9')?c-'0':(c>='a'&&c<='f')?c-'a'+10:-1;if(d<0)throw std::runtime_error("invalid hex");key=key*17+d+1;}
    for(size_t j=len;j<8;j++)key*=17;
    auto &m=maps[col-14];auto z=m.emplace(key,int32_t(m.size()));row[col-14]=z.first->second;
   }
   a=b+1;
  }
  out.write((char*)row.data(),104);if(!out)throw std::runtime_error("write failed");
  if(++n%1000000==0)std::cout<<"parsed "<<n<<std::endl;
 }
 if(in.bad())throw std::runtime_error("read failed");out.close();int64_t offset=0;
 std::ofstream meta(dir+"/vocabulary.json");meta<<"{\"rows\":"<<n<<",\"counts\":[";
 for(int t=0;t<26;t++){
  std::vector<std::pair<uint64_t,int32_t>> v(maps[t].begin(),maps[t].end());std::sort(v.begin(),v.end());
  std::vector<int32_t> lut(v.size());for(size_t i=0;i<v.size();i++)lut[v[i].second]=int32_t(offset+i);
  std::ofstream f(dir+"/lut_"+std::to_string(t)+".bin",std::ios::binary);f.write((char*)lut.data(),lut.size()*4);
  meta<<(t?",":"")<<v.size();offset+=v.size();
 }
 meta<<"],\"embeddings\":"<<offset<<"}\n";std::cout<<"DONE rows="<<n<<" embeddings="<<offset<<std::endl;
}catch(std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
