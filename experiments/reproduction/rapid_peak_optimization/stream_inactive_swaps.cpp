#include <cstdint>
#include <algorithm>
#include <vector>
#include <cmath>
extern "C" long swaps(long nh,int nt,int g,int nb,const int64_t* offsets,const int64_t* hot,const uint64_t** profiles,const uint16_t** weights,int16_t** owners,int64_t* rd,int64_t* sd,int64_t* recv,int64_t* send,const double* norm,int64_t* available,int64_t* moved_hot,int16_t* moved_from,int16_t* moved_to,long max_moved){
 auto table=[&](long gid){return int(std::upper_bound(offsets,offsets+nt+1,gid)-offsets)-1;};
 int64_t totals[64]={},caps[64]={},fcap[64]={},scap[64]={};for(int b=0;b<nb;b++)for(int r=0;r<g;r++){fcap[b]=std::max(fcap[b],recv[b*g+r]);scap[b]=std::max(scap[b],send[b*g+r]);totals[b]+=recv[b*g+r];}std::copy(totals,totals+nb,caps);
 long moved=0;
 for(long q=0;q<nh && 2*moved+2<=max_moved;q++){
  long gid=hot[q];int t=table(gid);long i=gid-offsets[t];int s=owners[t][i];auto p=profiles[t]+i*g;auto w=weights[t]+i*nb;int d=-1;double best=0;int64_t currentcap[64]={};for(int b=0;b<nb;b++)for(int r=0;r<g;r++)currentcap[b]=std::max(currentcap[b],recv[b*g+r]+send[b*g+r]);
  for(int k=0;k<g;k++)if(k!=s&&available[k]>0){
   double cost=0;bool ok=true;
   for(int b=0;b<nb;b++)if(w[b]){
    int ls=(p[s]>>b)&1,ld=(p[k]>>b)&1;double df1=ls,df2=-ld,ds1=-int(w[b])+ls,ds2=int(w[b])-ld;int64_t f1=recv[b*g+s],f2=recv[b*g+k],s1=send[b*g+s],s2=send[b*g+k],ccap=0;
    ccap=currentcap[b];
    if(totals[b]+ls-ld>caps[b]||f1+df1>fcap[b]||f2+df2>fcap[b]||s1+ds1>scap[b]||s2+ds2>scap[b]||f1+df1+s1+ds1>ccap||f2+df2+s2+ds2>ccap){ok=false;break;}
    double z=norm[b],a=df1+ds1,c=df2+ds2;cost+=(128*(2*(f1+s1)*a+a*a+2*(f2+s2)*c+c*c+2*f1*df1+df1*df1+2*f2*df2+df2*df2+2*s1*ds1+ds1*ds1+2*s2*ds2+ds2*ds2)/(z*z)+10*(df1+df2)/z)/nb;
   }
   if(ok&&cost<best){best=cost;d=k;}
  }
  if(d<0)continue;
  for(int b=0;b<nb;b++){int ls=(p[s]>>b)&1,ld=(p[d]>>b)&1;recv[b*g+s]+=ls;recv[b*g+d]-=ld;send[b*g+s]-=w[b]-ls;send[b*g+d]+=w[b]-ld;rd[(b*nt+t)*g+s]+=ls;rd[(b*nt+t)*g+d]-=ld;sd[(b*nt+t)*g+s]-=w[b]-ls;sd[(b*nt+t)*g+d]+=w[b]-ld;totals[b]+=ls-ld;}
  owners[t][i]=d;available[d]--;moved_hot[moved]=gid;moved_from[moved]=s;moved_to[moved]=d;moved++;
 }
 return moved;
}
