#include <cstdint>
#include <algorithm>
#include <limits>
extern "C" long repair(long n,int nt,int g,int nb,const int64_t* offsets,const int64_t* order,const uint64_t** profiles,const uint16_t** weights,int16_t** owners,int64_t* rd,int64_t* sd,int64_t* recv,int64_t* send,int64_t* storage,const int64_t* capacity,const double* norm){
 long moved=0;
 for(int phase=0;phase<3;phase++)for(long q=0;q<n;q++){
  long gid=order[q];int t=int(std::upper_bound(offsets,offsets+nt+1,gid)-offsets)-1;long i=gid-offsets[t];int s=owners[t][i];if(storage[s]<=capacity[s])continue;
  auto prof=profiles[t]+i*g;int home=-1,b=-1;for(int r=0;r<g;r++)if(prof[r]){home=r;b=__builtin_ctzll(prof[r]);break;}
  if(phase==0&&(home==s||storage[home]>=capacity[home]))continue;
  if(phase==1&&home==s)continue;
  int best=-1;double bc=std::numeric_limits<double>::infinity();
  for(int d=0;d<g;d++)if(storage[d]<capacity[d]){
   if(phase==0&&d!=home)continue;
   int rs=home==s,rdst=home==d,ss=1-rs,sdst=1-rdst;double z=norm[b];
   double cost=(128*(2.*recv[b*g+s]*rs+rs-2.*recv[b*g+d]*rdst+rdst-2.*send[b*g+s]*ss+ss+2.*send[b*g+d]*sdst+sdst))/(z*z)+10.*(rs-rdst)/z;
   if(cost<bc){bc=cost;best=d;}
  }
  if(best<0)continue;int d=best,rs=home==s,rdst=home==d,ss=1-rs,sdst=1-rdst;
  recv[b*g+s]+=rs;recv[b*g+d]-=rdst;send[b*g+s]-=ss;send[b*g+d]+=sdst;
  rd[(b*nt+t)*g+s]+=rs;rd[(b*nt+t)*g+d]-=rdst;sd[(b*nt+t)*g+s]-=ss;sd[(b*nt+t)*g+d]+=sdst;
  storage[s]--;storage[d]++;owners[t][i]=d;moved++;
 }
 return moved;
}
