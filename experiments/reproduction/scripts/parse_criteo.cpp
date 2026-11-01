// Strict Criteo TSV parser matching original hex/modulo and empty->zero logic.
#include <array>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
int main(int argc,char** argv) {
    try {
        if(argc!=3) throw std::runtime_error("usage: parse_criteo input.tsv output.singlehot.bin");
        const std::array<uint64_t,26> sizes{{40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36}};
        std::ifstream in(argv[1]);std::ofstream out(argv[2],std::ios::binary);
        if(!in || !out) throw std::runtime_error("cannot open input/output");
        std::string line;uint64_t rows=0;int32_t minimum=std::numeric_limits<int32_t>::max();
        std::array<int32_t,40> rec{};
        while(std::getline(in,line)) {
            const char* begin=line.data();const char* end=begin+line.size();const char* start=begin;
            for(int col=0;col<40;++col) {
                const char* stop=start;while(stop<end && *stop!='\t')++stop;
                if((col<39 && stop==end) || (col==39 && stop!=end))
                    throw std::runtime_error("expected 40 columns at row "+std::to_string(rows));
                rec[col]=0;
                if(stop!=start) {
                    if(col<14) {
                        auto r=std::from_chars(start,stop,rec[col],10);
                        if(r.ec!=std::errc{} || r.ptr!=stop) throw std::runtime_error("invalid int32 at row "+std::to_string(rows));
                    } else {
                        uint64_t value;auto r=std::from_chars(start,stop,value,16);
                        if(r.ec!=std::errc{} || r.ptr!=stop) throw std::runtime_error("invalid hex at row "+std::to_string(rows));
                        rec[col]=value%sizes[col-14];
                    }
                } else if(col==0) throw std::runtime_error("empty label");
                if(col>0 && col<14)minimum=std::min(minimum,rec[col]);
                if(col<39)start=stop+1;
            }
            out.write(reinterpret_cast<const char*>(rec.data()),sizeof(rec));
            if(!out)throw std::runtime_error("output write failure");
            if(++rows%1000000==0)std::cerr<<"parsed "<<rows<<"\n";
        }
        if(in.bad() || rows==0)throw std::runtime_error("input read failure/empty input");
        out.close();if(!out)throw std::runtime_error("output flush failure");
        std::ofstream meta(std::string(argv[2])+".json");
        meta<<"{\"rows\":"<<rows<<",\"dense_min\":"<<minimum<<",\"record_bytes\":160}\n";
        std::cout<<"rows="<<rows<<" dense_min="<<minimum<<"\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
