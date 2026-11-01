#include <cstdint>
#include <algorithm>
#include <limits>
extern "C" void capacity_owner(long n,int nt,int g,int nb,const int64_t* offsets,const int64_t* order,const uint64_t** profiles,const uint16_t** weights,int16_t** owners,int64_t* rd,int64_t* sd,int64_t* recv,int64_t* send,int64_t* storage,const int64_t* capacity,int64_t* reserve,const double* norm,double fw,double bw,double penalty){
 for(long q=0;q<n;q++){
  long gid=order[q];int t=int(std::upper_bound(offsets,offsets+nt+1,gid)-offsets)-1;long i=gid-offsets[t];auto prof=profiles[t]+i*g;auto w=weights[t]+i*nb;
  int mass=0;for(int b=0;b<nb;b++)mass+=w[b];int home=-1;
  if(mass==1){for(int r=0;r<g;r++)if(prof[r]){home=r;break;}if(home>=0&&reserve[home]>0)reserve[home]--;}
  int best=-1;double bc=std::numeric_limits<double>::infinity();
  for(int attempt=0;attempt<2&&best<0;attempt++)for(int d=0;d<g;d++){
   if(storage[d]>=capacity[d])continue;
   if(!attempt && storage[d]>=capacity[d]-reserve[d])continue;
   double cost=0;uint64_t bits=prof[d];
   for(int b=0;b<nb;b++){
    if(!w[b])continue;int local=(bits>>b)&1,remote=w[b]-local;double z=norm[b];
    cost+=(fw*(-2.*recv[b*g+d]*local+local)+bw*(2.*send[b*g+d]*remote+remote*remote))/(z*z)-penalty*local/z;
   }
   if(cost<bc || (cost==bc&&best>=0&&storage[d]<storage[best])){bc=cost;best=d;}
  }
  int d=best;owners[t][i]=d;storage[d]++;
  for(int b=0;b<nb;b++){
   int local=(prof[d]>>b)&1,remote=w[b]-local;
   recv[b*g+d]-=local;send[b*g+d]+=remote;rd[(b*nt+t)*g+d]-=local;sd[(b*nt+t)*g+d]+=remote;
  }
 }
}
