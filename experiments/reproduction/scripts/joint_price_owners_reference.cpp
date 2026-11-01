#include <cstdint>
#include <cmath>
#include <algorithm>
// Sequential coordinate descent: cheap aggregate proposal, exact batch objective acceptance.
extern "C" long refine(long n,int g,int nb,int16_t* owner,const uint64_t* profile,
 const uint16_t* weight,int64_t* rd,int64_t* sd,int64_t* recv,int64_t* send,
 int64_t* storage,const double* norm,const int64_t* total0,double fw,double bw,
 double sw,double penalty,double budget,int reverse,int hot_threshold) {
 double ar[64]={},as[64]={}; double sm=0,avg=0;
 for(int r=0;r<g;r++){sm+=storage[r];for(int b=0;b<nb;b++){ar[r]+=double(recv[b*g+r])/nb;as[r]+=double(send[b*g+r])/nb;}}
 sm/=g;for(int b=0;b<nb;b++)avg+=norm[b]/nb;
 long moved=0;
 for(long ii=0;ii<n;ii++){
  long i=reverse?n-1-ii:ii;int s=owner[i];double mass=0;int hits[64];
  for(int r=0;r<g;r++)hits[r]=__builtin_popcountll(profile[i*g+r]);
  for(int b=0;b<nb;b++)mass+=weight[i*nb+b];
  int best=-1;double bestcost=0;
  for(int d=0;d<g;d++)if(d!=s){
   double ls=double(hits[s])/nb,ld=double(hits[d])/nb,ds=mass/nb-ls,dd=mass/nb-ld;
   double cost=fw*(2*(ar[s]-avg)*ls+ls*ls-2*(ar[d]-avg)*ld+ld*ld)/(avg*avg)
    +bw*(-2*(as[s]-avg)*ds+ds*ds+2*(as[d]-avg)*dd+dd*dd)/(avg*avg)
    +sw*(2.*(storage[d]-storage[s])+2)/(sm*sm)+penalty*(ls-ld)/avg;
   if(cost<bestcost){bestcost=cost;best=d;}
  }
  if(hot_threshold>0 && mass>=hot_threshold){
   best=-1;bestcost=0;
   for(int d=0;d<g;d++)if(d!=s){
    double candidate=sw*(2.*(storage[d]-storage[s])+2)/(sm*sm);
    for(int b=0;b<nb;b++){
     int ls=(profile[i*g+s]>>b)&1,ld=(profile[i*g+d]>>b)&1;
     int ds=weight[i*nb+b]-ls,dd=weight[i*nb+b]-ld;double z=norm[b];
     candidate+=(fw*(2*(recv[b*g+s]-z)*ls+ls*ls-2*(recv[b*g+d]-z)*ld+ld*ld)/(z*z)
       +bw*(-2*(send[b*g+s]-z)*ds+ds*ds+2*(send[b*g+d]-z)*dd+dd*dd)/(z*z)
       +penalty*(ls-ld)/z)/nb;
    }
    if(candidate<bestcost){bestcost=candidate;best=d;}
   }
  }
  if(best<0)continue;int d=best;double cost=sw*(2.*(storage[d]-storage[s])+2)/(sm*sm);bool ok=true;
  for(int b=0;b<nb;b++){
   int ls=(profile[i*g+s]>>b)&1,ld=(profile[i*g+d]>>b)&1;
   int ds=weight[i*nb+b]-ls,dd=weight[i*nb+b]-ld;
   int64_t tot=0;for(int r=0;r<g;r++)tot+=recv[b*g+r];
   if(tot+ls-ld>total0[b]*(1+budget)){ok=false;break;}
   double z=norm[b];
   cost+=(fw*(2*(recv[b*g+s]-z)*ls+ls*ls-2*(recv[b*g+d]-z)*ld+ld*ld)/(z*z)
    +bw*(-2*(send[b*g+s]-z)*ds+ds*ds+2*(send[b*g+d]-z)*dd+dd*dd)/(z*z)
    +penalty*(ls-ld)/z)/nb;
  }
  if(!ok||cost>=-1e-16)continue;
  for(int b=0;b<nb;b++){
   int ls=(profile[i*g+s]>>b)&1,ld=(profile[i*g+d]>>b)&1;
   int ds=weight[i*nb+b]-ls,dd=weight[i*nb+b]-ld;
   recv[b*g+s]+=ls;recv[b*g+d]-=ld;send[b*g+s]-=ds;send[b*g+d]+=dd;
   rd[b*g+s]+=ls;rd[b*g+d]-=ld;sd[b*g+s]-=ds;sd[b*g+d]+=dd;
  }
  ar[s]+=double(hits[s])/nb;ar[d]-=double(hits[d])/nb;
  as[s]-=(mass-hits[s])/nb;as[d]+=(mass-hits[d])/nb;
  storage[s]--;storage[d]++;owner[i]=d;moved++;
 }
 return moved;
}
