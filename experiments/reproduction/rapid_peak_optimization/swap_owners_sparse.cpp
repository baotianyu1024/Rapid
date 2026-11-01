#include <cstdint>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
extern "C" long owner_swaps(long nc,long nh,int nt,int g,int nb,const int64_t* offsets,const int64_t* cold,const int64_t* hot,const uint64_t** profiles,const uint16_t** weights,int16_t** owners,int64_t* rd,int64_t* sd,int64_t* recv,int64_t* send,const double* norm,int passes){
 auto table=[&](long gid){return int(std::upper_bound(offsets,offsets+nt+1,gid)-offsets)-1;};
 std::vector<std::vector<long>>pool(g);for(long q=0;q<nc;q++){long gid=cold[q];int t=table(gid);pool[owners[t][gid-offsets[t]]].push_back(gid);}
 int64_t totals[64]={},caps[64]={};for(int b=0;b<nb;b++)for(int r=0;r<g;r++)totals[b]+=recv[b*g+r];std::copy(totals,totals+nb,caps);
 auto cost=[&](int b,int s,int d,double r1,double r2,double s1,double s2,double rf1,double rf2,double sf1,double sf2){double z=norm[b];return (128*(2*r1*rf1+rf1*rf1+2*r2*rf2+rf2*rf2+2*s1*sf1+sf1*sf1+2*s2*sf2+sf2*sf2)/(z*z)+10*(rf1+rf2)/z)/nb;};
 long moved=0;
 for(int pass=0;pass<passes;pass++)for(long q=0;q<nh;q++){
  long gid=hot[q];int t=table(gid);long i=gid-offsets[t];int s=owners[t][i];auto p=profiles[t]+i*g;auto w=weights[t]+i*nb;
  int active[64],na=0;for(int b=0;b<nb;b++)if(w[b])active[na++]=b;int d=-1;double hc=1e99;
  for(int k=0;k<g;k++)if(k!=s&&!pool[k].empty()){
   double c=0;for(int ab=0;ab<na;ab++){int b=active[ab];int ls=(p[s]>>b)&1,ld=(p[k]>>b)&1;c+=cost(b,s,k,recv[b*g+s],recv[b*g+k],send[b*g+s],send[b*g+k],ls,-ld,-int(w[b])+ls,int(w[b])-ld);}
   if(c<hc){hc=c;d=k;}
  }
  if(d<0)continue;long best=-1;double bc=0;
  for(int j=0;j<std::min(size_t(32),pool[d].size());j++){
   long pos=(q*7919+pass*997+j*131)%pool[d].size(),cid=pool[d][pos];int ct=table(cid);long ci=cid-offsets[ct];auto cp=profiles[ct]+ci*g;int home=-1,cb=-1;for(int r=0;r<g;r++)if(cp[r]){home=r;cb=__builtin_ctzll(cp[r]);break;}
   int ls=(p[s]>>cb)&1,ld=(p[d]>>cb)&1;int crs=-(home==s),crd=home==d,css=home!=s,csd=-(home!=d);
   double c=hc+cost(cb,s,d,recv[cb*g+s]+ls,recv[cb*g+d]-ld,send[cb*g+s]-w[cb]+ls,send[cb*g+d]+w[cb]-ld,crs,crd,css,csd);
   if(c>=bc)continue;bool ok=true;for(int b=0;b<nb;b++){int v=int((p[s]>>b)&1)-int((p[d]>>b)&1)+(b==cb?crs+crd:0);if(totals[b]+v>caps[b]){ok=false;break;}}
   if(ok){bc=c;best=pos;}
  }
  if(best<0)continue;
  long cid=pool[d][best];pool[d][best]=pool[d].back();pool[d].pop_back();pool[s].push_back(cid);int ct=table(cid);long ci=cid-offsets[ct];auto cp=profiles[ct]+ci*g;int home=-1,cb=-1;for(int r=0;r<g;r++)if(cp[r]){home=r;cb=__builtin_ctzll(cp[r]);break;}
  for(int b=0;b<nb;b++){
   int ls=(p[s]>>b)&1,ld=(p[d]>>b)&1,ds=w[b]-ls,dd=w[b]-ld;recv[b*g+s]+=ls;recv[b*g+d]-=ld;send[b*g+s]-=ds;send[b*g+d]+=dd;rd[(b*nt+t)*g+s]+=ls;rd[(b*nt+t)*g+d]-=ld;sd[(b*nt+t)*g+s]-=ds;sd[(b*nt+t)*g+d]+=dd;totals[b]+=ls-ld;
  }
  int rs=-(home==s),rr=home==d,ss=home!=s,sdst=-(home!=d);recv[cb*g+s]+=rs;recv[cb*g+d]+=rr;send[cb*g+s]+=ss;send[cb*g+d]+=sdst;rd[(cb*nt+ct)*g+s]+=rs;rd[(cb*nt+ct)*g+d]+=rr;sd[(cb*nt+ct)*g+s]+=ss;sd[(cb*nt+ct)*g+d]+=sdst;totals[cb]+=rs+rr;owners[t][i]=d;owners[ct][ci]=s;moved++;
 }
 return moved;
}
