#pragma once
#include <stddef.h>
#include <cstdint>
#include <vector>
#include <utility>
// Reference-count dictionary. Iteration order is deliberately not exposed.
// Zero-reference entries keep their key so deletion never breaks probe chains.
class FlatCounts {
public:
 struct Entry {uint64_t first=0;uint32_t second=0;bool used=false;};
 FlatCounts():cells_(16){}
 Entry* end(){return nullptr;} const Entry* end() const{return nullptr;}
 size_t size()const{return live_;}
 Entry* find(uint64_t key){auto &e=cells_[slot(key)];return e.used&&e.second?&e:nullptr;}
 const Entry* find(uint64_t key)const{const auto &e=cells_[slot(key)];return e.used&&e.second?&e:nullptr;}
 uint32_t& operator[](uint64_t key){
  if((occupied_+1)*10>=cells_.size()*7)grow();
  auto &e=cells_[slot(key)];if(!e.used){e.used=true;e.first=key;occupied_++;}
  if(!e.second)live_++;return e.second;
 }
 void erase(Entry* e){e->second=0;--live_;}
private:
 static uint64_t hash(uint64_t x){x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
 size_t slot(uint64_t key)const{size_t p=hash(key)&(cells_.size()-1);while(cells_[p].used&&cells_[p].first!=key)p=(p+1)&(cells_.size()-1);return p;}
 void grow(){auto old=std::move(cells_);cells_.resize(old.size()*2);occupied_=0;for(const auto&e:old)if(e.used&&e.second){cells_[slot(e.first)]=e;occupied_++;}}
 std::vector<Entry>cells_;size_t occupied_=0,live_=0;
};
