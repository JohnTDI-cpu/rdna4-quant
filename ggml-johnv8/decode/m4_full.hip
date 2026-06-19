// M4: FULL Bielik model forward + token loop. embed -> 50 layers -> output_norm -> output proj -> argmax.
// Real decode t/s of the custom engine. Reuses validated kernels from m4_layer.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#define CK(x) do{hipError_t e=(x);if(e){printf("err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef unsigned char u8; typedef unsigned short u16; typedef unsigned int u32; typedef unsigned long long u64;
#define NE 4096
#define NH 32
#define NKV 8
#define HD 128
#define GQA 4
#define NFF 14336
#define NL 50
#define VOCAB 32128
#define QK 32
#define MAXSEQ 1024
static const u8* gg; static size_t gp; static u64 g_data;
static u64 r64(){u64 v;memcpy(&v,gg+gp,8);gp+=8;return v;} static u32 r32(){u32 v;memcpy(&v,gg+gp,4);gp+=4;return v;}
static u64 rstr(const char**o){u64 n=r64();*o=(const char*)(gg+gp);gp+=n;return n;}
static const int tsz[13]={1,1,2,2,4,4,4,1,0,0,8,8,8};
static void skipv(u32 t){if(t==8){u64 n=r64();gp+=n;return;}if(t==9){u32 et=r32();u64 n=r64();if(et==8){for(u64 i=0;i<n;i++){u64 l=r64();gp+=l;}}else gp+=tsz[et]*n;return;}gp+=tsz[t];}
static u32 g_last_type=0;  // ggml_type of last tensor found by find() (12=Q4_K, 14=Q6_K, 13=Q5_K, 11=Q3_K)
static const u8* find(const char* w,u64* ne0,u64* ne1){ size_t sp=gp; gp=0; r32();r32();u64 nt=r64(),nkv=r64();u32 al=32;
  for(u64 i=0;i<nkv;i++){const char*k;u64 kl=rstr(&k);u32 t=r32();if(kl==17&&!strncmp(k,"general.alignment",17)){al=r32();continue;}skipv(t);}
  u64 foff=0,d0=1,d1=1;int fnd=0;
  for(u64 i=0;i<nt;i++){const char*nm;u64 nl=rstr(&nm);char nb[160];int cn=nl<159?nl:159;memcpy(nb,nm,cn);nb[cn]=0;u32 nd=r32();u64 dd[4]={1,1,1,1};for(u32 d=0;d<nd;d++)dd[d]=r64();u32 _tp=r32();u64 off=r64();if(!strcmp(nb,w)){foff=off;d0=dd[0];d1=dd[1];fnd=1;g_last_type=_tp;}}
  g_data=(gp+al-1)&~(u64)(al-1); gp=sp; if(!fnd){printf("missing %s\n",w);exit(1);} *ne0=d0;*ne1=d1; return gg+g_data+foff; }
static void loadW(const char* name,signed char** dW,__half** dWS,int* No,int* Ko){
  u64 e0,e1;const u8* wd=find(name,&e0,&e1); int k=e0,n=e1,nb=k/QK; if(No)*No=n; if(Ko)*Ko=k;
  std::vector<signed char> W((size_t)n*k); std::vector<__half> WS((size_t)n*nb);
  for(int r=0;r<n;r++)for(int b=0;b<nb;b++){const u8* blk=wd+((size_t)r*nb+b)*34;u16 d;memcpy(&d,blk,2);WS[(size_t)r*nb+b]=*(__half*)&d;memcpy(&W[(size_t)r*k+b*QK],blk+2,32);}
  CK(hipMalloc(dW,(size_t)n*k));CK(hipMalloc(dWS,(size_t)n*nb*2));
  CK(hipMemcpy(*dW,W.data(),(size_t)n*k,hipMemcpyHostToDevice));CK(hipMemcpy(*dWS,WS.data(),(size_t)n*nb*2,hipMemcpyHostToDevice)); }
// Q4_0 loader: block_q4_0 = 18 bytes {fp16 d; uint8 qs[16]} -> packed nibbles W4 (k/2 bytes/row) + fp16 scales
static void loadW_q4(const char* name,uint8_t** dW,__half** dWS,int* No,int* Ko){
  u64 e0,e1;const u8* wd=find(name,&e0,&e1); int k=e0,n=e1,nb=k/QK; if(No)*No=n; if(Ko)*Ko=k;
  std::vector<uint8_t> W((size_t)n*(k/2)); std::vector<__half> WS((size_t)n*nb);
  for(int r=0;r<n;r++)for(int b=0;b<nb;b++){const u8* blk=wd+((size_t)r*nb+b)*18;u16 d;memcpy(&d,blk,2);WS[(size_t)r*nb+b]=*(__half*)&d;memcpy(&W[(size_t)r*(k/2)+b*16],blk+2,16);}
  CK(hipMalloc(dW,(size_t)n*(k/2)));CK(hipMalloc(dWS,(size_t)n*nb*2));
  CK(hipMemcpy(*dW,W.data(),(size_t)n*(k/2),hipMemcpyHostToDevice));CK(hipMemcpy(*dWS,WS.data(),(size_t)n*nb*2,hipMemcpyHostToDevice)); }
static float* loadF32(const char* name){ u64 e0,e1;const u8* d=find(name,&e0,&e1); float* dd;CK(hipMalloc(&dd,e0*4));CK(hipMemcpy(dd,d,e0*4,hipMemcpyHostToDevice));return dd; }
// vocab: extract tokenizer.ggml.tokens (array<string>) for detokenization
static std::vector<std::string> g_vocab;
static void loadVocab(){ size_t sp=gp; gp=0; r32();r32(); u64 nt=r64(),nkv=r64(); (void)nt;
  for(u64 i=0;i<nkv;i++){ const char* k;u64 kl=rstr(&k);u32 t=r32();
    if(kl==21&&!strncmp(k,"tokenizer.ggml.tokens",21)){ u32 et=r32();(void)et;u64 n=r64();g_vocab.resize(n);
      for(u64 j=0;j<n;j++){const char* s;u64 sl=rstr(&s);g_vocab[j].assign(s,sl);} gp=sp; return; }
    skipv(t); }
  gp=sp; }
static void detok(int id,std::string& out){ if(id<0||id>=(int)g_vocab.size())return; const std::string& s=g_vocab[id];
  // byte-fallback token <0xHH>
  if(s.size()==6&&s[0]=='<'&&s[1]=='0'&&s[2]=='x'&&s[5]=='>'){ auto hx=[](char c){return c<='9'?c-'0':(c|32)-'a'+10;}; out+=(char)(hx(s[3])*16+hx(s[4])); return; }
  for(size_t i=0;i<s.size();){ if(i+2<s.size()&&(u8)s[i]==0xE2&&(u8)s[i+1]==0x96&&(u8)s[i+2]==0x81){out+=' ';i+=3;} else {out+=s[i];i++;} } }
// embed: dequant row `tok` of token_embd {NE, VOCAB} (NE contiguous) -> x[NE]
static const u8* g_embd;
static void embed_cpu(int tok,float* x){ int nb=NE/QK; const u8* row=g_embd+(size_t)tok*nb*34;
  for(int b=0;b<nb;b++){const u8* blk=row+b*34;u16 dh;memcpy(&dh,blk,2);
    u32 s=(dh>>15)&1,e=(dh>>10)&0x1f,m=dh&0x3ff,o; if(e==0){if(m==0)o=s<<31;else{e=127-15+1;while(!(m&0x400)){m<<=1;e--;}m&=0x3ff;o=(s<<31)|(e<<23)|(m<<13);}}else if(e==31)o=(s<<31)|0x7f800000|(m<<13);else o=(s<<31)|((e-15+127)<<23)|(m<<13);
    float d;memcpy(&d,&o,4); const signed char* q=(const signed char*)(blk+2); for(int i=0;i<QK;i++)x[b*QK+i]=d*(float)q[i]; } }
// kernels (from m4_layer)
__global__ void rmsnorm(const float* x,const float* w,float* y,int n,float eps){int i=threadIdx.x;__shared__ float ss;float p=0;for(int j=i;j<n;j+=blockDim.x)p+=x[j]*x[j];__shared__ float red[256];red[i]=p;__syncthreads();for(int s=blockDim.x/2;s>0;s>>=1){if(i<s)red[i]+=red[i+s];__syncthreads();}if(i==0)ss=rsqrtf(red[0]/n+eps);__syncthreads();for(int j=i;j<n;j+=blockDim.x)y[j]=x[j]*ss*w[j];}
__global__ void quant(const float* x,signed char* q,float* s,int nb){int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=nb)return;float am=0;for(int i=0;i<QK;i++){float a=fabsf(x[b*QK+i]);if(a>am)am=a;}float d=am/127.f,id=d>0?1/d:0;s[b]=d;for(int i=0;i<QK;i++){int v=(int)lrintf(x[b*QK+i]*id);if(v>127)v=127;if(v<-127)v=-127;q[b*QK+i]=v;}}
__global__ void gemv(const signed char* W,const __half* WS,const signed char* xq,const float* xs,float* y,int N,int K){int wpb=blockDim.x/warpSize,wid=blockIdx.x*wpb+threadIdx.x/warpSize,nw=gridDim.x*wpb,lane=threadIdx.x%warpSize,nb=K/QK;for(int row=wid;row<N;row+=nw){const signed char* w=W+(size_t)row*K;const __half* ws=WS+(size_t)row*nb;float acc=0;for(int b=lane;b<nb;b+=warpSize){const signed char* wp=w+b*QK;const signed char* xp=xq+b*QK;int s=0;
  #pragma unroll
  for(int i=0;i<QK;i++)s+=(int)wp[i]*(int)xp[i];acc+=(float)s*__half2float(ws[b])*xs[b];}for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);if(lane==0)y[row]=acc;}}
__global__ void rope(float* X,int heads,int pos,float base){int h=blockIdx.x;if(h>=heads)return;int i=threadIdx.x;if(i>=HD/2)return;float th=pos*powf(base,-2.0f*i/(float)HD),c=cosf(th),s=sinf(th);float* x=X+(size_t)h*HD;float a=x[2*i],b=x[2*i+1];x[2*i]=a*c-b*s;x[2*i+1]=a*s+b*c;}
__global__ void attn(const float* Q,const float* Kc,const float* Vc,float* O,int seq,float scale){int h=blockIdx.x;if(h>=NH)return;int kvh=h/GQA;int tid=threadIdx.x;extern __shared__ float sh[];__shared__ float qsh[HD];__shared__ float red[256];__shared__ float mx,sm;for(int d=tid;d<HD;d+=blockDim.x)qsh[d]=Q[(size_t)h*HD+d];__syncthreads();for(int t=tid;t<seq;t+=blockDim.x){const float* k=Kc+((size_t)kvh*MAXSEQ+t)*HD;float dot=0;
  #pragma unroll 4
  for(int d=0;d<HD;d++)dot+=qsh[d]*k[d];sh[t]=dot*scale;}__syncthreads();float lm=-1e30f;for(int t=tid;t<seq;t+=blockDim.x)lm=fmaxf(lm,sh[t]);red[tid]=lm;__syncthreads();for(int s=blockDim.x/2;s>0;s>>=1){if(tid<s)red[tid]=fmaxf(red[tid],red[tid+s]);__syncthreads();}if(tid==0)mx=red[0];__syncthreads();float ls=0;for(int t=tid;t<seq;t+=blockDim.x){float e=expf(sh[t]-mx);sh[t]=e;ls+=e;}red[tid]=ls;__syncthreads();for(int s=blockDim.x/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}if(tid==0)sm=red[0];__syncthreads();if(tid<HD){float acc=0,inv=1.f/sm;for(int t=0;t<seq;t++)acc+=sh[t]*inv*Vc[((size_t)kvh*MAXSEQ+t)*HD+tid];O[(size_t)h*HD+tid]=acc;}}
__global__ void silumul(const float* g,const float* u,float* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float x=g[i];y[i]=(x/(1.f+expf(-x)))*u[i];}}
__global__ void residual(float* x,const float* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=y[i];}
__global__ void kvappend(float* Kc,float* Vc,const float* kk,const float* vv,int pos){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NKV*HD)return;int kvh=i/HD,d=i%HD;size_t o=((size_t)kvh*MAXSEQ+pos)*HD+d;Kc[o]=kk[i];Vc[o]=vv[i];}
__global__ void argmax_k(const float* logits,int n,int* out){__shared__ float bv[256];__shared__ int bi[256];int t=threadIdx.x;float mv=-1e30f;int mi=0;for(int i=t;i<n;i+=blockDim.x)if(logits[i]>mv){mv=logits[i];mi=i;}bv[t]=mv;bi[t]=mi;__syncthreads();for(int s=blockDim.x/2;s>0;s>>=1){if(t<s&&bv[t+s]>bv[t]){bv[t]=bv[t+s];bi[t]=bi[t+s];}__syncthreads();}if(t==0)*out=bi[0];}
// ===== FUSED kernels (lossless: identical math, fewer launches -> cut dispatch idle) =====
// rmsnorm + quant fused (single block, n=NE): writes quantized q/s directly, no float intermediate
__global__ void rmsnorm_q(const float* x,const float* w,signed char* q,float* s,int n,float eps){
  int i=threadIdx.x;__shared__ float ss;__shared__ float red[256];float p=0;
  for(int j=i;j<n;j+=blockDim.x)p+=x[j]*x[j];red[i]=p;__syncthreads();
  for(int st=blockDim.x/2;st>0;st>>=1){if(i<st)red[i]+=red[i+st];__syncthreads();}
  if(i==0)ss=rsqrtf(red[0]/n+eps);__syncthreads();
  int nb=n/QK;
  for(int b=i;b<nb;b+=blockDim.x){float vals[QK];float am=0;
    for(int e=0;e<QK;e++){int idx=b*QK+e;float v=x[idx]*ss*w[idx];vals[e]=v;float a=fabsf(v);if(a>am)am=a;}
    float d=am/127.f,id=d>0?1/d:0;s[b]=d;
    for(int e=0;e<QK;e++){int vq=(int)lrintf(vals[e]*id);if(vq>127)vq=127;if(vq<-127)vq=-127;q[b*QK+e]=vq;}}
}
// gemv accumulating into y (fuses residual: y[row]+=acc). y must pre-hold the residual stream.
__global__ void gemv_acc(const signed char* W,const __half* WS,const signed char* xq,const float* xs,float* y,int N,int K){int wpb=blockDim.x/warpSize,wid=blockIdx.x*wpb+threadIdx.x/warpSize,nw=gridDim.x*wpb,lane=threadIdx.x%warpSize,nb=K/QK;for(int row=wid;row<N;row+=nw){const signed char* w=W+(size_t)row*K;const __half* ws=WS+(size_t)row*nb;float acc=0;for(int b=lane;b<nb;b+=warpSize){const signed char* wp=w+b*QK;const signed char* xp=xq+b*QK;int s=0;
  #pragma unroll
  for(int i=0;i<QK;i++)s+=(int)wp[i]*(int)xp[i];acc+=(float)s*__half2float(ws[b])*xs[b];}for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);if(lane==0)y[row]+=acc;}}
// silu(g)*u + quant fused (per 32-block)
__global__ void silumul_q(const float* g,const float* u,signed char* q,float* s,int nb){
  int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=nb)return;float vals[QK];float am=0;
  for(int e=0;e<QK;e++){int idx=b*QK+e;float x=g[idx];float v=(x/(1.f+expf(-x)))*u[idx];vals[e]=v;float a=fabsf(v);if(a>am)am=a;}
  float d=am/127.f,id=d>0?1/d:0;s[b]=d;
  for(int e=0;e<QK;e++){int vq=(int)lrintf(vals[e]*id);if(vq>127)vq=127;if(vq<-127)vq=-127;q[b*QK+e]=vq;}}
// fused rope for q (NH heads) and k (NKV heads) in one launch
__global__ void rope2(float* q,float* kk,int pos,float base){int h=blockIdx.x;int i=threadIdx.x;if(i>=HD/2)return;
  float* x=(h<NH)?(q+(size_t)h*HD):(kk+(size_t)(h-NH)*HD);
  float th=pos*powf(base,-2.0f*i/(float)HD),c=cosf(th),s=sinf(th);float a=x[2*i],b=x[2*i+1];x[2*i]=a*c-b*s;x[2*i+1]=a*s+b*c;}
// fused q,k,v GEMV: 3 weights share input xq/xs and K -> one launch, route warp by global row
__global__ void gemv_qkv(const signed char* Wq,const __half* Sq,const signed char* Wk,const __half* Sk,const signed char* Wv,const __half* Sv,
                         const signed char* xq,const float* xs,float* q,float* kk,float* vv,int K){
  int wpb=blockDim.x/warpSize,wid=blockIdx.x*wpb+threadIdx.x/warpSize,nw=gridDim.x*wpb,lane=threadIdx.x%warpSize,nb=K/QK;
  int total=NE+2*NKV*HD;
  for(int gr=wid;gr<total;gr+=nw){const signed char* W;const __half* WS;float* y;int row;
    if(gr<NE){W=Wq;WS=Sq;y=q;row=gr;} else if(gr<NE+NKV*HD){W=Wk;WS=Sk;y=kk;row=gr-NE;} else {W=Wv;WS=Sv;y=vv;row=gr-NE-NKV*HD;}
    const signed char* w=W+(size_t)row*K;const __half* ws=WS+(size_t)row*nb;float acc=0;
    for(int b=lane;b<nb;b+=warpSize){const signed char* wp=w+b*QK;const signed char* xp=xq+b*QK;int s=0;
      #pragma unroll
      for(int i=0;i<QK;i++)s+=(int)wp[i]*(int)xp[i];acc+=(float)s*__half2float(ws[b])*xs[b];}
    for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);if(lane==0)y[row]=acc;}}
// fused gate,up GEMV: both NFF rows, share input -> one launch
__global__ void gemv_gu(const signed char* Wg,const __half* Sg,const signed char* Wu,const __half* Su,
                        const signed char* xq,const float* xs,float* g,float* u,int K,int Nff){
  int wpb=blockDim.x/warpSize,wid=blockIdx.x*wpb+threadIdx.x/warpSize,nw=gridDim.x*wpb,lane=threadIdx.x%warpSize,nb=K/QK;
  int total=2*Nff;
  for(int gr=wid;gr<total;gr+=nw){const signed char* W;const __half* WS;float* y;int row;
    if(gr<Nff){W=Wg;WS=Sg;y=g;row=gr;} else {W=Wu;WS=Su;y=u;row=gr-Nff;}
    const signed char* w=W+(size_t)row*K;const __half* ws=WS+(size_t)row*nb;float acc=0;
    for(int b=lane;b<nb;b+=warpSize){const signed char* wp=w+b*QK;const signed char* xp=xq+b*QK;int s=0;
      #pragma unroll
      for(int i=0;i<QK;i++)s+=(int)wp[i]*(int)xp[i];acc+=(float)s*__half2float(ws[b])*xs[b];}
    for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);if(lane==0)y[row]=acc;}}
// ===== MEGAKERNEL: whole decoder layer in ONE cooperative launch (grid.sync replaces dispatch idle) =====
namespace cg = cooperative_groups;
struct MegaArgs {
  const signed char *Wq,*Wk,*Wv,*Wo,*Wg,*Wu,*Wd;
  const __half *Sq,*Sk,*Sv,*So,*Sg,*Su,*Sd;
  float *Kc,*Vc; const float *an,*fn;
  float *x; signed char *hq; float *hs;
  float *q,*kk,*vv,*ao; signed char *aoq; float *aos;
  float *g,*u; signed char *fhq; float *fhs;
  int pos; float eps,scale,base;
};
__global__ void __launch_bounds__(256,4) layer_mega(MegaArgs a){
  cg::grid_group grid = cg::this_grid();
  int bid=blockIdx.x,nbk=gridDim.x,tid=threadIdx.x,nth=blockDim.x;
  int gtid=bid*nth+tid,gnth=nbk*nth;
  int wpb=nth/warpSize,wid=bid*wpb+tid/warpSize,nw=nbk*wpb,lane=tid%warpSize;
  int nbE=NE/QK,nbF=NFF/QK; float* x=a.x;
  __shared__ float red[256]; __shared__ float shA[MAXSEQ]; __shared__ float qsh[HD]; __shared__ float mx,sm;
  // 1. attn-norm + quant (each block self-reduces sumsq over x; redundant but x is tiny/cached)
  { float p=0; for(int j=tid;j<NE;j+=nth)p+=x[j]*x[j]; red[tid]=p; __syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    float ss=rsqrtf(red[0]/NE+a.eps); __syncthreads();
    for(int b=gtid;b<nbE;b+=gnth){float vals[QK],am=0;
      for(int e=0;e<QK;e++){int i=b*QK+e;float v=x[i]*ss*a.an[i];vals[e]=v;float aa=fabsf(v);if(aa>am)am=aa;}
      float d=am/127.f,id=d>0?1/d:0;a.hs[b]=d;
      for(int e=0;e<QK;e++){int vq=(int)lrintf(vals[e]*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.hq[b*QK+e]=vq;}} }
  grid.sync();
  // 2. gemv q,k,v
  { int total=NE+2*NKV*HD;
    for(int r=wid;r<total;r+=nw){const signed char* W;const __half* WS;float* y;int row;
      if(r<NE){W=a.Wq;WS=a.Sq;y=a.q;row=r;}else if(r<NE+NKV*HD){W=a.Wk;WS=a.Sk;y=a.kk;row=r-NE;}else{W=a.Wv;WS=a.Sv;y=a.vv;row=r-NE-NKV*HD;}
      const signed char* w=W+(size_t)row*NE;const __half* ws=WS+(size_t)row*nbE;float acc=0;
      for(int b=lane;b<nbE;b+=warpSize){const signed char* wp=w+b*QK;const signed char* xp=a.hq+b*QK;int s=0;
        #pragma unroll
        for(int j=0;j<8;j++)s=__builtin_amdgcn_sudot4(true,((const int*)wp)[j],true,((const int*)xp)[j],s,false);acc+=(float)s*__half2float(ws[b])*a.hs[b];}
      for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);if(lane==0)y[row]=acc;} }
  grid.sync();
  // 3+4. rope q+k MERGED with kvappend (k-pairs rope-then-append in same thread -> 1 fewer grid.sync)
  { int totr=(NH+NKV)*(HD/2);
    for(int wk=gtid;wk<totr;wk+=gnth){int h=wk/(HD/2),i=wk%(HD/2);
      float th=a.pos*powf(a.base,-2.0f*i/(float)HD),c=cosf(th),s=sinf(th);
      if(h<NH){float* xr=a.q+(size_t)h*HD;float aa=xr[2*i],bb=xr[2*i+1];xr[2*i]=aa*c-bb*s;xr[2*i+1]=aa*s+bb*c;}
      else{int kvh=h-NH;float* xr=a.kk+(size_t)kvh*HD;float aa=xr[2*i],bb=xr[2*i+1];float r0=aa*c-bb*s,r1=aa*s+bb*c;xr[2*i]=r0;xr[2*i+1]=r1;
        size_t o=((size_t)kvh*MAXSEQ+a.pos)*HD;a.Kc[o+2*i]=r0;a.Kc[o+2*i+1]=r1;a.Vc[o+2*i]=a.vv[kvh*HD+2*i];a.Vc[o+2*i+1]=a.vv[kvh*HD+2*i+1];}} }
  grid.sync();
  // 5. attn (one block per head)
  if(bid<NH){int h=bid,kvh=h/GQA,seq=a.pos+1;
    for(int d=tid;d<HD;d+=nth)qsh[d]=a.q[(size_t)h*HD+d];__syncthreads();
    for(int t=tid;t<seq;t+=nth){const float* k=a.Kc+((size_t)kvh*MAXSEQ+t)*HD;float dot=0;
      #pragma unroll 4
      for(int d=0;d<HD;d++)dot+=qsh[d]*k[d];shA[t]=dot*a.scale;}__syncthreads();
    float lm=-1e30f;for(int t=tid;t<seq;t+=nth)lm=fmaxf(lm,shA[t]);red[tid]=lm;__syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]=fmaxf(red[tid],red[tid+s]);__syncthreads();}if(tid==0)mx=red[0];__syncthreads();
    float ls=0;for(int t=tid;t<seq;t+=nth){float e=expf(shA[t]-mx);shA[t]=e;ls+=e;}red[tid]=ls;__syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}if(tid==0)sm=red[0];__syncthreads();
    if(tid<HD){float acc=0,inv=1.f/sm;for(int t=0;t<seq;t++)acc+=shA[t]*inv*a.Vc[((size_t)kvh*MAXSEQ+t)*HD+tid];
      // quant fused: head's 128 outputs = 4 quant-blocks = 4 warps; warp max over 32 lanes (1 fewer grid.sync)
      float am=fabsf(acc); for(int o=16;o>0;o>>=1)am=fmaxf(am,__shfl_down(am,o,32)); am=__shfl(am,0,32);
      float d=am/127.f,id=d>0?1/d:0; if((tid&31)==0)a.aos[h*4+tid/32]=d;
      int vq=(int)lrintf(acc*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.aoq[(size_t)h*HD+tid]=vq;} }
  grid.sync();
  // 6. gemv o -> x += (reads fused aoq/aos)
  { for(int r=wid;r<NE;r+=nw){const signed char* w=a.Wo+(size_t)r*NE;const __half* ws=a.So+(size_t)r*nbE;float acc=0;
      for(int b=lane;b<nbE;b+=warpSize){const signed char* wp=w+b*QK;const signed char* xp=a.aoq+b*QK;int s=0;
        #pragma unroll
        for(int j=0;j<8;j++)s=__builtin_amdgcn_sudot4(true,((const int*)wp)[j],true,((const int*)xp)[j],s,false);acc+=(float)s*__half2float(ws[b])*a.aos[b];}
      for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);if(lane==0)x[r]+=acc;} }
  grid.sync();
  // 7. ffn-norm + quant
  { float p=0; for(int j=tid;j<NE;j+=nth)p+=x[j]*x[j]; red[tid]=p; __syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    float ss=rsqrtf(red[0]/NE+a.eps); __syncthreads();
    for(int b=gtid;b<nbE;b+=gnth){float vals[QK],am=0;
      for(int e=0;e<QK;e++){int i=b*QK+e;float v=x[i]*ss*a.fn[i];vals[e]=v;float aa=fabsf(v);if(aa>am)am=aa;}
      float d=am/127.f,id=d>0?1/d:0;a.hs[b]=d;
      for(int e=0;e<QK;e++){int vq=(int)lrintf(vals[e]*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.hq[b*QK+e]=vq;}} }
  grid.sync();
  // 8. gemv gate+up
  { int total=2*NFF;
    for(int r=wid;r<total;r+=nw){const signed char* W;const __half* WS;float* y;int row;
      if(r<NFF){W=a.Wg;WS=a.Sg;y=a.g;row=r;}else{W=a.Wu;WS=a.Su;y=a.u;row=r-NFF;}
      const signed char* w=W+(size_t)row*NE;const __half* ws=WS+(size_t)row*nbE;float acc=0;
      for(int b=lane;b<nbE;b+=warpSize){const signed char* wp=w+b*QK;const signed char* xp=a.hq+b*QK;int s=0;
        #pragma unroll
        for(int j=0;j<8;j++)s=__builtin_amdgcn_sudot4(true,((const int*)wp)[j],true,((const int*)xp)[j],s,false);acc+=(float)s*__half2float(ws[b])*a.hs[b];}
      for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);if(lane==0)y[row]=acc;} }
  grid.sync();
  // 9. silu*u + quant
  { for(int b=gtid;b<nbF;b+=gnth){float vals[QK],am=0;
      for(int e=0;e<QK;e++){int i=b*QK+e;float xv=a.g[i];float v=(xv/(1.f+expf(-xv)))*a.u[i];vals[e]=v;float aa=fabsf(v);if(aa>am)am=aa;}
      float d=am/127.f,id=d>0?1/d:0;a.fhs[b]=d;
      for(int e=0;e<QK;e++){int vq=(int)lrintf(vals[e]*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.fhq[b*QK+e]=vq;}} }
  grid.sync();
  // 10. gemv down -> x +=
  { for(int r=wid;r<NE;r+=nw){const signed char* w=a.Wd+(size_t)r*NFF;const __half* ws=a.Sd+(size_t)r*nbF;float acc=0;
      for(int b=lane;b<nbF;b+=warpSize){const signed char* wp=w+b*QK;const signed char* xp=a.fhq+b*QK;int s=0;
        #pragma unroll
        for(int j=0;j<8;j++)s=__builtin_amdgcn_sudot4(true,((const int*)wp)[j],true,((const int*)xp)[j],s,false);acc+=(float)s*__half2float(ws[b])*a.fhs[b];}
      for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);if(lane==0)x[r]+=acc;} }
}
// ===== Q4_0 path: embed, output GEMV, and Q4 megakernel (sudot4 on packed nibbles) =====
// fast Q4_0 block dot: sudot4 on packed nibbles, inline -8 bias via sudot4(0x08..,x). weights[0..15]=low nibbles, [16..31]=high.
#define Q4DOT(wp,xp) ({int dot_=0,bias_=0; const int* w32_=(const int*)(wp); const int* xl_=(const int*)(xp); const int* xh_=(const int*)((xp)+16); \
  _Pragma("unroll") for(int k_=0;k_<4;k_++){ unsigned ww_=w32_[k_]; int lo_=ww_&0x0F0F0F0F,hi_=(ww_>>4)&0x0F0F0F0F; \
    dot_=__builtin_amdgcn_sudot4(false,lo_,true,xl_[k_],dot_,false); dot_=__builtin_amdgcn_sudot4(false,hi_,true,xh_[k_],dot_,false); \
    bias_=__builtin_amdgcn_sudot4(false,0x08080808,true,xl_[k_],bias_,false); bias_=__builtin_amdgcn_sudot4(false,0x08080808,true,xh_[k_],bias_,false);} \
  dot_-bias_; })
__global__ void embed_gpu_q4(const uint8_t* We,const __half* Se,const int* tokp,float* x){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NE)return;int tok=*tokp;int b=i/QK,j=i%QK;
  const uint8_t* w=We+(size_t)tok*(NE/2)+b*16;float sc=__half2float(Se[(size_t)tok*(NE/QK)+b]);
  int nib=(j<16)?(w[j]&0xF):(w[j-16]>>4);x[i]=(float)(nib-8)*sc;}
__global__ void gemv_q4(const uint8_t* W,const __half* WS,const signed char* xq,const float* xs,float* y,int N,int K){
  int wpb=blockDim.x/warpSize,wid=blockIdx.x*wpb+threadIdx.x/warpSize,nw=gridDim.x*wpb,lane=threadIdx.x%warpSize,nb=K/QK;
  for(int row=wid;row<N;row+=nw){const uint8_t* w=W+(size_t)row*(K/2);const __half* ws=WS+(size_t)row*nb;float acc=0;
    for(int b=lane;b<nb;b+=warpSize){const uint8_t* wp=w+b*16;const signed char* xp=xq+b*QK;int s=Q4DOT(wp,xp);acc+=(float)s*__half2float(ws[b])*xs[b];}
    for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);if(lane==0)y[row]=acc;}}
// ===== Q4_K path (reuse proven q4kc kernel, f32 activation; --pure Q4_K model) =====
struct block_q4_K { __half d; __half dmin; uint8_t scales[12]; uint8_t qs[128]; };          // on-disk 144B
struct block_q4_K_r { __half d; __half dmin; int8_t sc[8]; int8_t m[8]; uint8_t qs[128]; };  // repacked 148B (scales pre-decoded)
__device__ static void gsm_d(int j,const uint8_t* q,uint8_t* d,uint8_t* m){ if(j<4){*d=q[j]&63;*m=q[j+4]&63;}else{*d=(q[j+4]&0xF)|((q[j-4]>>6)<<4);*m=(q[j+4]>>4)|((q[j]>>6)<<4);} }
__global__ void q4k_repack(const block_q4_K* src,block_q4_K_r* dst,long nbtot){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=nbtot)return; const block_q4_K* s=src+i; block_q4_K_r* d=dst+i;
  d->d=s->d; d->dmin=s->dmin; for(int j=0;j<8;j++){uint8_t a,b;gsm_d(j,s->scales,&a,&b);d->sc[j]=(int8_t)a;d->m[j]=(int8_t)b;}
  const uint32_t* sq=(const uint32_t*)s->qs; uint32_t* dq=(uint32_t*)d->qs; for(int j=0;j<32;j++)dq[j]=sq[j]; }
template<int BS,int NR>
__global__ void __launch_bounds__(BS) q4kc(const block_q4_K_r* __restrict__ W,const float* __restrict__ a,float* __restrict__ out,int N,int K){
  int row0=blockIdx.x*NR, tid=threadIdx.x; int nb=K/256, nunits=nb*32; float acc[NR];
  #pragma unroll
  for(int r=0;r<NR;r++)acc[r]=0;
  for(int u=tid;u<nunits;u+=BS){ int block=u>>5, within=u&31, group=within>>3, wb=(within&7)*4;
    const float* ag=a+block*256+group*64; float4 al=*(const float4*)(ag+wb), ah=*(const float4*)(ag+32+wb);
    #pragma unroll
    for(int r=0;r<NR;r++){ const block_q4_K_r* B=W+(size_t)(row0+r)*nb+block;
      uint32_t pk=((const uint32_t*)B->qs)[within]; float d=__half2float(B->d),dmin=__half2float(B->dmin);
      float d1=d*B->sc[2*group],mm1=dmin*B->m[2*group],d2=d*B->sc[2*group+1],mm2=dmin*B->m[2*group+1];
      acc[r]+=(d1*(pk&0xF)-mm1)*al.x+(d1*((pk>>8)&0xF)-mm1)*al.y+(d1*((pk>>16)&0xF)-mm1)*al.z+(d1*((pk>>24)&0xF)-mm1)*al.w
             +(d2*((pk>>4)&0xF)-mm2)*ah.x+(d2*((pk>>12)&0xF)-mm2)*ah.y+(d2*((pk>>20)&0xF)-mm2)*ah.z+(d2*((pk>>28)&0xF)-mm2)*ah.w; } }
  __shared__ float sh[NR][BS];
  #pragma unroll
  for(int r=0;r<NR;r++)sh[r][tid]=acc[r]; __syncthreads();
  for(int s=BS/2;s>0;s>>=1){ if(tid<s){ for(int r=0;r<NR;r++)sh[r][tid]+=sh[r][tid+s]; } __syncthreads(); }
  if(tid==0){ for(int r=0;r<NR;r++)out[row0+r]=sh[r][0]; } }
__global__ void embed_q4k(const block_q4_K_r* We,const int* tokp,float* x,int NEv){
  int u=blockIdx.x*blockDim.x+threadIdx.x; int nb=NEv/256; if(u>=nb*32)return; int tok=*tokp;
  const block_q4_K_r* B=We+(size_t)tok*nb+(u>>5); int within=u&31, group=within>>3, wb=(within&7)*4;
  uint32_t pk=((const uint32_t*)B->qs)[within]; float d=__half2float(B->d),dmin=__half2float(B->dmin);
  float d1=d*B->sc[2*group],m1=dmin*B->m[2*group],d2=d*B->sc[2*group+1],m2=dmin*B->m[2*group+1];
  int base=(u>>5)*256+group*64+wb;
  x[base]=d1*(pk&0xF)-m1; x[base+1]=d1*((pk>>8)&0xF)-m1; x[base+2]=d1*((pk>>16)&0xF)-m1; x[base+3]=d1*((pk>>24)&0xF)-m1;
  x[base+32]=d2*((pk>>4)&0xF)-m2; x[base+33]=d2*((pk>>12)&0xF)-m2; x[base+34]=d2*((pk>>20)&0xF)-m2; x[base+35]=d2*((pk>>28)&0xF)-m2; }
__global__ void silumul_f32(const float* g,const float* u,float* fh,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){float v=g[i]; fh[i]=(v/(1.f+expf(-v)))*u[i];} }
static void loadW_q4k(const char* name,block_q4_K_r** dW,int* No,int* Ko){
  u64 e0,e1; const u8* wd=find(name,&e0,&e1); int K=e0,N=e1,nb=K/256; if(No)*No=N; if(Ko)*Ko=K;
  long nbtot=(long)N*nb; block_q4_K* draw; CK(hipMalloc(&draw,(size_t)nbtot*sizeof(block_q4_K)));
  CK(hipMemcpy(draw,wd,(size_t)nbtot*sizeof(block_q4_K),hipMemcpyHostToDevice));
  CK(hipMalloc(dW,(size_t)nbtot*sizeof(block_q4_K_r)));
  q4k_repack<<<(nbtot+255)/256,256>>>(draw,*dW,nbtot); CK(hipDeviceSynchronize()); CK(hipFree(draw)); }
struct MegaArgsQ4 {
  const uint8_t *Wq,*Wk,*Wv,*Wo,*Wg,*Wu,*Wd;
  const __half *Sq,*Sk,*Sv,*So,*Sg,*Su,*Sd;
  float *Kc,*Vc; const float *an,*fn;
  float *x; signed char *hq; float *hs;
  float *q,*kk,*vv,*ao; signed char *aoq; float *aos;
  float *g,*u; signed char *fhq; float *fhs;
  int pos; float eps,scale,base;
};
__global__ void __launch_bounds__(256,4) layer_mega_q4(MegaArgsQ4 a){
  cg::grid_group grid = cg::this_grid();
  int bid=blockIdx.x,nbk=gridDim.x,tid=threadIdx.x,nth=blockDim.x;
  int gtid=bid*nth+tid,gnth=nbk*nth;
  int wpb=nth/warpSize,wid=bid*wpb+tid/warpSize,nw=nbk*wpb,lane=tid%warpSize;
  int nbE=NE/QK,nbF=NFF/QK; float* x=a.x;
  __shared__ float red[256]; __shared__ float shA[MAXSEQ]; __shared__ float qsh[HD]; __shared__ float mx,sm;
  // 1. attn-norm + quant
  { float p=0; for(int j=tid;j<NE;j+=nth)p+=x[j]*x[j]; red[tid]=p; __syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    float ss=rsqrtf(red[0]/NE+a.eps); __syncthreads();
    for(int b=gtid;b<nbE;b+=gnth){float vals[QK],am=0;
      for(int e=0;e<QK;e++){int i=b*QK+e;float v=x[i]*ss*a.an[i];vals[e]=v;float aa=fabsf(v);if(aa>am)am=aa;}
      float d=am/127.f,id=d>0?1/d:0;a.hs[b]=d;
      for(int e=0;e<QK;e++){int vq=(int)lrintf(vals[e]*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.hq[b*QK+e]=vq;}} }
  grid.sync();
  // 2. gemv q,k,v (Q4)
  { int total=NE+2*NKV*HD;
    for(int r=wid;r<total;r+=nw){const uint8_t* W;const __half* WS;float* y;int row;
      if(r<NE){W=a.Wq;WS=a.Sq;y=a.q;row=r;}else if(r<NE+NKV*HD){W=a.Wk;WS=a.Sk;y=a.kk;row=r-NE;}else{W=a.Wv;WS=a.Sv;y=a.vv;row=r-NE-NKV*HD;}
      const uint8_t* w=W+(size_t)row*(NE/2);const __half* ws=WS+(size_t)row*nbE;float acc=0;
      for(int b=lane;b<nbE;b+=warpSize){const uint8_t* wp=w+b*16;const signed char* xp=a.hq+b*QK;int s=Q4DOT(wp,xp);acc+=(float)s*__half2float(ws[b])*a.hs[b];}
      for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);if(lane==0)y[row]=acc;} }
  grid.sync();
  // 3+4. rope q+k MERGED with kvappend
  { int totr=(NH+NKV)*(HD/2);
    for(int wk=gtid;wk<totr;wk+=gnth){int h=wk/(HD/2),i=wk%(HD/2);
      float th=a.pos*powf(a.base,-2.0f*i/(float)HD),c=cosf(th),s=sinf(th);
      if(h<NH){float* xr=a.q+(size_t)h*HD;float aa=xr[2*i],bb=xr[2*i+1];xr[2*i]=aa*c-bb*s;xr[2*i+1]=aa*s+bb*c;}
      else{int kvh=h-NH;float* xr=a.kk+(size_t)kvh*HD;float aa=xr[2*i],bb=xr[2*i+1];float r0=aa*c-bb*s,r1=aa*s+bb*c;xr[2*i]=r0;xr[2*i+1]=r1;
        size_t o=((size_t)kvh*MAXSEQ+a.pos)*HD;a.Kc[o+2*i]=r0;a.Kc[o+2*i+1]=r1;a.Vc[o+2*i]=a.vv[kvh*HD+2*i];a.Vc[o+2*i+1]=a.vv[kvh*HD+2*i+1];}} }
  grid.sync();
  // 5. attn + fused quant
  if(bid<NH){int h=bid,kvh=h/GQA,seq=a.pos+1;
    for(int d=tid;d<HD;d+=nth)qsh[d]=a.q[(size_t)h*HD+d];__syncthreads();
    for(int t=tid;t<seq;t+=nth){const float* k=a.Kc+((size_t)kvh*MAXSEQ+t)*HD;float dot=0;
      #pragma unroll 4
      for(int d=0;d<HD;d++)dot+=qsh[d]*k[d];shA[t]=dot*a.scale;}__syncthreads();
    float lm=-1e30f;for(int t=tid;t<seq;t+=nth)lm=fmaxf(lm,shA[t]);red[tid]=lm;__syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]=fmaxf(red[tid],red[tid+s]);__syncthreads();}if(tid==0)mx=red[0];__syncthreads();
    float ls=0;for(int t=tid;t<seq;t+=nth){float e=expf(shA[t]-mx);shA[t]=e;ls+=e;}red[tid]=ls;__syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}if(tid==0)sm=red[0];__syncthreads();
    if(tid<HD){float acc=0,inv=1.f/sm;for(int t=0;t<seq;t++)acc+=shA[t]*inv*a.Vc[((size_t)kvh*MAXSEQ+t)*HD+tid];
      float am=fabsf(acc); for(int o=16;o>0;o>>=1)am=fmaxf(am,__shfl_down(am,o,32)); am=__shfl(am,0,32);
      float d=am/127.f,id=d>0?1/d:0; if((tid&31)==0)a.aos[h*4+tid/32]=d;
      int vq=(int)lrintf(acc*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.aoq[(size_t)h*HD+tid]=vq;} }
  grid.sync();
  // 6. gemv o (Q4) -> x +=
  { for(int r=wid;r<NE;r+=nw){const uint8_t* w=a.Wo+(size_t)r*(NE/2);const __half* ws=a.So+(size_t)r*nbE;float acc=0;
      for(int b=lane;b<nbE;b+=warpSize){const uint8_t* wp=w+b*16;const signed char* xp=a.aoq+b*QK;int s=Q4DOT(wp,xp);acc+=(float)s*__half2float(ws[b])*a.aos[b];}
      for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);if(lane==0)x[r]+=acc;} }
  grid.sync();
  // 7. ffn-norm + quant
  { float p=0; for(int j=tid;j<NE;j+=nth)p+=x[j]*x[j]; red[tid]=p; __syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    float ss=rsqrtf(red[0]/NE+a.eps); __syncthreads();
    for(int b=gtid;b<nbE;b+=gnth){float vals[QK],am=0;
      for(int e=0;e<QK;e++){int i=b*QK+e;float v=x[i]*ss*a.fn[i];vals[e]=v;float aa=fabsf(v);if(aa>am)am=aa;}
      float d=am/127.f,id=d>0?1/d:0;a.hs[b]=d;
      for(int e=0;e<QK;e++){int vq=(int)lrintf(vals[e]*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.hq[b*QK+e]=vq;}} }
  grid.sync();
  // 8. gemv gate+up (Q4)
  { int total=2*NFF;
    for(int r=wid;r<total;r+=nw){const uint8_t* W;const __half* WS;float* y;int row;
      if(r<NFF){W=a.Wg;WS=a.Sg;y=a.g;row=r;}else{W=a.Wu;WS=a.Su;y=a.u;row=r-NFF;}
      const uint8_t* w=W+(size_t)row*(NE/2);const __half* ws=WS+(size_t)row*nbE;float acc=0;
      for(int b=lane;b<nbE;b+=warpSize){const uint8_t* wp=w+b*16;const signed char* xp=a.hq+b*QK;int s=Q4DOT(wp,xp);acc+=(float)s*__half2float(ws[b])*a.hs[b];}
      for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);if(lane==0)y[row]=acc;} }
  grid.sync();
  // 9. silu*u + quant
  { for(int b=gtid;b<nbF;b+=gnth){float vals[QK],am=0;
      for(int e=0;e<QK;e++){int i=b*QK+e;float xv=a.g[i];float v=(xv/(1.f+expf(-xv)))*a.u[i];vals[e]=v;float aa=fabsf(v);if(aa>am)am=aa;}
      float d=am/127.f,id=d>0?1/d:0;a.fhs[b]=d;
      for(int e=0;e<QK;e++){int vq=(int)lrintf(vals[e]*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.fhq[b*QK+e]=vq;}} }
  grid.sync();
  // 10. gemv down (Q4) -> x +=
  { for(int r=wid;r<NE;r+=nw){const uint8_t* w=a.Wd+(size_t)r*(NFF/2);const __half* ws=a.Sd+(size_t)r*nbF;float acc=0;
      for(int b=lane;b<nbF;b+=warpSize){const uint8_t* wp=w+b*16;const signed char* xp=a.fhq+b*QK;int s=Q4DOT(wp,xp);acc+=(float)s*__half2float(ws[b])*a.fhs[b];}
      for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);if(lane==0)x[r]+=acc;} }
}
// ===== Q4_K megakernel (f32-activation, warp/row q4kc inline, grid.sync fused layer) =====
__device__ static float q4k_warprow(const block_q4_K_r* W,int row,const float* act,int lane,int K){
  int nb=K/256; const block_q4_K_r* Wr=W+(size_t)row*nb; int nunits=nb*32; float acc=0;
  for(int u=lane;u<nunits;u+=warpSize){ int block=u>>5, within=u&31, group=within>>3, wb=(within&7)*4;
    const float* ag=act+block*256+group*64; float4 al=*(const float4*)(ag+wb), ah=*(const float4*)(ag+32+wb);
    const block_q4_K_r* B=Wr+block; uint32_t pk=((const uint32_t*)B->qs)[within];
    float d=__half2float(B->d),dmin=__half2float(B->dmin);
    float d1=d*B->sc[2*group],mm1=dmin*B->m[2*group],d2=d*B->sc[2*group+1],mm2=dmin*B->m[2*group+1];
    acc+=(d1*(pk&0xF)-mm1)*al.x+(d1*((pk>>8)&0xF)-mm1)*al.y+(d1*((pk>>16)&0xF)-mm1)*al.z+(d1*((pk>>24)&0xF)-mm1)*al.w
        +(d2*((pk>>4)&0xF)-mm2)*ah.x+(d2*((pk>>12)&0xF)-mm2)*ah.y+(d2*((pk>>20)&0xF)-mm2)*ah.z+(d2*((pk>>28)&0xF)-mm2)*ah.w; }
  for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o); return acc; }
template<int NR>  // NR rows/warp: load activation float4 ONCE, reuse across NR weight rows (q4kc NR4=109% vs NR1=99%)
__device__ static void gemv_q4k_nr(const block_q4_K_r* W,const float* act,float* y,int N,int K,int wid,int nw,int lane,int accum){
  int nb=K/256, nunits=nb*32;
  for(int rb=wid*NR; rb<N; rb+=nw*NR){ int nr=(N-rb<NR)?(N-rb):NR; float acc[NR];
    #pragma unroll
    for(int r=0;r<NR;r++)acc[r]=0;
    for(int u=lane;u<nunits;u+=warpSize){ int block=u>>5,within=u&31,group=within>>3,wb=(within&7)*4;
      const float* ag=act+block*256+group*64; float4 al=*(const float4*)(ag+wb),ah=*(const float4*)(ag+32+wb);
      for(int r=0;r<nr;r++){ const block_q4_K_r* B=W+(size_t)(rb+r)*nb+block; uint32_t pk=((const uint32_t*)B->qs)[within];
        float d=__half2float(B->d),dmin=__half2float(B->dmin);
        float d1=d*B->sc[2*group],mm1=dmin*B->m[2*group],d2=d*B->sc[2*group+1],mm2=dmin*B->m[2*group+1];
        acc[r]+=(d1*(pk&0xF)-mm1)*al.x+(d1*((pk>>8)&0xF)-mm1)*al.y+(d1*((pk>>16)&0xF)-mm1)*al.z+(d1*((pk>>24)&0xF)-mm1)*al.w
               +(d2*((pk>>4)&0xF)-mm2)*ah.x+(d2*((pk>>12)&0xF)-mm2)*ah.y+(d2*((pk>>20)&0xF)-mm2)*ah.z+(d2*((pk>>28)&0xF)-mm2)*ah.w; } }
    for(int r=0;r<nr;r++){ float a2=acc[r]; for(int o=warpSize/2;o>0;o>>=1)a2+=__shfl_down(a2,o); if(lane==0){ if(accum)y[rb+r]+=a2; else y[rb+r]=a2; } } }
}
// block-based split-K Q4_K GEMV: ALL nth threads split K for NR=4 rows (activation reuse), shared reduce.
// Matches standalone q4kc<*,4>=100-109% vs warp/row 87-99%. sh reuses shA (NR*nth=1024 floats). All threads/block uniform (same bid).
__device__ static void gemv_q4k_block(const block_q4_K_r* W,const float* act,float* y,int N,int K,int bid,int nbk,int tid,int nth,float* sh,int accum){
  const int NR=4; int nb=K/256, nunits=nb*32;
  for(int rb=bid*NR; rb<N; rb+=nbk*NR){ int nr=(N-rb<NR)?(N-rb):NR; float acc[NR];
    #pragma unroll
    for(int r=0;r<NR;r++)acc[r]=0;
    for(int u=tid;u<nunits;u+=nth){ int block=u>>5,within=u&31,group=within>>3,wb=(within&7)*4;
      const float* ag=act+block*256+group*64; float4 al=*(const float4*)(ag+wb),ah=*(const float4*)(ag+32+wb);
      for(int r=0;r<nr;r++){ const block_q4_K_r* B=W+(size_t)(rb+r)*nb+block; uint32_t pk=((const uint32_t*)B->qs)[within];
        float d=__half2float(B->d),dmin=__half2float(B->dmin);
        float d1=d*B->sc[2*group],mm1=dmin*B->m[2*group],d2=d*B->sc[2*group+1],mm2=dmin*B->m[2*group+1];
        acc[r]+=(d1*(pk&0xF)-mm1)*al.x+(d1*((pk>>8)&0xF)-mm1)*al.y+(d1*((pk>>16)&0xF)-mm1)*al.z+(d1*((pk>>24)&0xF)-mm1)*al.w
               +(d2*((pk>>4)&0xF)-mm2)*ah.x+(d2*((pk>>12)&0xF)-mm2)*ah.y+(d2*((pk>>20)&0xF)-mm2)*ah.z+(d2*((pk>>28)&0xF)-mm2)*ah.w; } }
    #pragma unroll
    for(int r=0;r<NR;r++)sh[r*nth+tid]=acc[r];
    __syncthreads();
    for(int s=nth/2;s>0;s>>=1){ if(tid<s){
      #pragma unroll
      for(int r=0;r<NR;r++)sh[r*nth+tid]+=sh[r*nth+tid+s]; } __syncthreads(); }
    if(tid<nr){ if(accum)y[rb+tid]+=sh[tid*nth]; else y[rb+tid]=sh[tid*nth]; }
    __syncthreads(); }
}
struct MegaArgsQ4K { const block_q4_K_r *Wq,*Wk,*Wv,*Wo,*Wg,*Wu,*Wd; float *Kc,*Vc; const float *an,*fn;
  float *x,*hy,*q,*kk,*vv,*ao,*g,*u,*fh; int pos; float eps,scale,base; };
__global__ void __launch_bounds__(256,4) layer_mega_q4k(MegaArgsQ4K a){
  cg::grid_group grid=cg::this_grid();
  int bid=blockIdx.x,nbk=gridDim.x,tid=threadIdx.x,nth=blockDim.x;
  int gtid=bid*nth+tid,gnth=nbk*nth;
  int wpb=nth/warpSize,wid=bid*wpb+tid/warpSize,nw=nbk*wpb,lane=tid%warpSize;
  float* x=a.x; __shared__ float red[256]; __shared__ float shA[MAXSEQ]; __shared__ float qsh[HD]; __shared__ float mx,sm;
  { float p=0; for(int j=tid;j<NE;j+=nth)p+=x[j]*x[j]; red[tid]=p; __syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    float ss=rsqrtf(red[0]/NE+a.eps); __syncthreads();
    for(int i=gtid;i<NE;i+=gnth)a.hy[i]=x[i]*ss*a.an[i]; }
  grid.sync();
  { int total=NE+2*NKV*HD;
    for(int r=wid;r<total;r+=nw){const block_q4_K_r* W;float* y;int row;
      if(r<NE){W=a.Wq;y=a.q;row=r;}else if(r<NE+NKV*HD){W=a.Wk;y=a.kk;row=r-NE;}else{W=a.Wv;y=a.vv;row=r-NE-NKV*HD;}
      float acc=q4k_warprow(W,row,a.hy,lane,NE); if(lane==0)y[row]=acc;} }
  grid.sync();
  { int totr=(NH+NKV)*(HD/2);
    for(int wk=gtid;wk<totr;wk+=gnth){int h=wk/(HD/2),i=wk%(HD/2);
      float th=a.pos*powf(a.base,-2.0f*i/(float)HD),c=cosf(th),s=sinf(th);
      if(h<NH){float* xr=a.q+(size_t)h*HD;float aa=xr[2*i],bb=xr[2*i+1];xr[2*i]=aa*c-bb*s;xr[2*i+1]=aa*s+bb*c;}
      else{int kvh=h-NH;float* xr=a.kk+(size_t)kvh*HD;float aa=xr[2*i],bb=xr[2*i+1];float r0=aa*c-bb*s,r1=aa*s+bb*c;xr[2*i]=r0;xr[2*i+1]=r1;
        size_t o=((size_t)kvh*MAXSEQ+a.pos)*HD;a.Kc[o+2*i]=r0;a.Kc[o+2*i+1]=r1;a.Vc[o+2*i]=a.vv[kvh*HD+2*i];a.Vc[o+2*i+1]=a.vv[kvh*HD+2*i+1];}} }
  grid.sync();
  if(bid<NH){int h=bid,kvh=h/GQA,seq=a.pos+1;
    for(int d=tid;d<HD;d+=nth)qsh[d]=a.q[(size_t)h*HD+d];__syncthreads();
    for(int t=tid;t<seq;t+=nth){const float* k=a.Kc+((size_t)kvh*MAXSEQ+t)*HD;float dot=0;
      #pragma unroll 4
      for(int d=0;d<HD;d++)dot+=qsh[d]*k[d];shA[t]=dot*a.scale;}__syncthreads();
    float lm=-1e30f;for(int t=tid;t<seq;t+=nth)lm=fmaxf(lm,shA[t]);red[tid]=lm;__syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]=fmaxf(red[tid],red[tid+s]);__syncthreads();}if(tid==0)mx=red[0];__syncthreads();
    float ls=0;for(int t=tid;t<seq;t+=nth){float e=expf(shA[t]-mx);shA[t]=e;ls+=e;}red[tid]=ls;__syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}if(tid==0)sm=red[0];__syncthreads();
    if(tid<HD){float acc=0,inv=1.f/sm;for(int t=0;t<seq;t++)acc+=shA[t]*inv*a.Vc[((size_t)kvh*MAXSEQ+t)*HD+tid];
      a.ao[(size_t)h*HD+tid]=acc;} }
  grid.sync();
  { for(int r=wid;r<NE;r+=nw){float acc=q4k_warprow(a.Wo,r,a.ao,lane,NE); if(lane==0)x[r]+=acc;} }
  grid.sync();
  { float p=0; for(int j=tid;j<NE;j+=nth)p+=x[j]*x[j]; red[tid]=p; __syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    float ss=rsqrtf(red[0]/NE+a.eps); __syncthreads();
    for(int i=gtid;i<NE;i+=gnth)a.hy[i]=x[i]*ss*a.fn[i]; }
  grid.sync();
  { int total=2*NFF;
    for(int r=wid;r<total;r+=nw){const block_q4_K_r* W;float* y;int row;
      if(r<NFF){W=a.Wg;y=a.g;row=r;}else{W=a.Wu;y=a.u;row=r-NFF;}
      float acc=q4k_warprow(W,row,a.hy,lane,NE); if(lane==0)y[row]=acc;} }
  grid.sync();
  { for(int i=gtid;i<NFF;i+=gnth){float xv=a.g[i];a.fh[i]=(xv/(1.f+expf(-xv)))*a.u[i];} }
  grid.sync();
  { for(int r=wid;r<NE;r+=nw){float acc=q4k_warprow(a.Wd,r,a.fh,lane,NFF); if(lane==0)x[r]+=acc;} }
}
// ===== Q6_K path (repack ql/qh -> int8 q6-32 position-order; sudot4 SIGNED dot; scales already int8, NO dmin) =====
struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; __half d; };   // on-disk 210B
struct block_q6_K_r { uint8_t ql[128]; uint8_t qh[64]; int8_t sc[16]; __half d; };        // packed 210B (ZERO inflacji), ql/qh position-order
__global__ void q6k_repack(const block_q6_K* src,block_q6_K_r* dst,long nbtot){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=nbtot)return; const block_q6_K* s=src+i; block_q6_K_r* d=dst+i; d->d=s->d;
  for(int j=0;j<16;j++)d->sc[j]=s->scales[j];
  for(int j=0;j<128;j++)d->ql[j]=0; for(int j=0;j<64;j++)d->qh[j]=0;
  for(int n=0;n<2;n++){ const uint8_t* ql=s->ql+64*n; const uint8_t* qh=s->qh+32*n; int base=128*n;
    for(int l=0;l<32;l++){
      int qv[4]={ (ql[l]&0xF)|((qh[l]&3)<<4), (ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4), (ql[l]>>4)|(((qh[l]>>4)&3)<<4), (ql[l+32]>>4)|(((qh[l]>>6)&3)<<4) };
      int pv[4]={ base+l, base+l+32, base+l+64, base+l+96 };
      for(int z=0;z<4;z++){ int qq=qv[z],pp=pv[z]; d->ql[pp>>1]|=(qq&0xF)<<(4*(pp&1)); d->qh[pp>>2]|=(qq>>4)<<(2*(pp&3)); } } } }
__device__ static float q6k_warprow_i8(const block_q6_K_r* Wr,const signed char* xq,const float* xs,int lane,int K){
  int nb=K/256, nsub=nb*16; float acc=0;
  for(int s=lane;s<nsub;s+=warpSize){ int blk=s>>4, t=s&15; const block_q6_K_r* B=Wr+blk;
    const int* xi=(const int*)(xq + (size_t)blk*256 + t*16); int idot=0;
    #pragma unroll
    for(int w=0;w<4;w++){
      uint32_t b0=B->ql[8*t+2*w], b1=B->ql[8*t+2*w+1], h=B->qh[4*t+w];
      int q0=(int)((b0&0xF)|((h&3)<<4))-32, q1=(int)((b0>>4)|(((h>>2)&3)<<4))-32;
      int q2=(int)((b1&0xF)|(((h>>4)&3)<<4))-32, q3=(int)((b1>>4)|(((h>>6)&3)<<4))-32;
      int qi=(q0&0xFF)|((q1&0xFF)<<8)|((q2&0xFF)<<16)|((q3&0xFF)<<24);
      idot=__builtin_amdgcn_sudot4(true,qi,true,xi[w],idot,false); }
    acc += xs[(size_t)blk*8+(t>>1)]*__half2float(B->d)*(float)B->sc[t]*(float)idot; }
  for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);
  return acc; }
static void loadW_q6k(const char* name,block_q6_K_r** dW,int* No,int* Ko){
  u64 e0,e1; const u8* wd=find(name,&e0,&e1); int K=e0,N=e1,nb=K/256; if(No)*No=N; if(Ko)*Ko=K;
  long nbtot=(long)N*nb; block_q6_K* draw; CK(hipMalloc(&draw,(size_t)nbtot*sizeof(block_q6_K)));
  CK(hipMemcpy(draw,wd,(size_t)nbtot*sizeof(block_q6_K),hipMemcpyHostToDevice));
  CK(hipMalloc(dW,(size_t)nbtot*sizeof(block_q6_K_r)));
  q6k_repack<<<(nbtot+255)/256,256>>>(draw,*dW,nbtot); CK(hipDeviceSynchronize()); CK(hipFree(draw)); }
// ===== Q5_K path (= Q4_K + 1 high bit z qh; dot = Q4_K nib + 16*qh_bit; scales 6-bit jak Q4_K via gsm_d) =====
struct block_q5_K_r { __half d; __half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };  // = on-disk 176B, scales dekodowane inline gsm (ZERO inflacji; Q5_K memory-bound)
__device__ static float q5k_warprow_i8(const block_q5_K_r* Wr,const signed char* xq,const float* xs,int lane,int K){
  int nb=K/256, nsub=nb*8; float acc=0;
  for(int s=lane;s<nsub;s+=warpSize){ int blk=s>>3, j=s&7; const block_q5_K_r* B=Wr+blk;
    const uint32_t* qs=(const uint32_t*)B->qs; const uint32_t* qhu=(const uint32_t*)B->qh;
    const int* xi=(const int*)(xq+(size_t)blk*256); int k=j>>1, isodd=j&1, idot=0, imsum=0;
    #pragma unroll
    for(int w=0;w<8;w++){ uint32_t wb=qs[8*k+w]; uint32_t nib=isodd?((wb>>4)&0x0F0F0F0Fu):(wb&0x0F0F0F0Fu);
      uint32_t q5=nib + (((qhu[w]>>j)&0x01010101u)<<4); int av=xi[8*j+w];
      idot=__builtin_amdgcn_sudot4(false,(int)q5,true,av,idot,false); imsum=__builtin_amdgcn_sudot4(false,0x01010101,true,av,imsum,false); }
    uint8_t sc8,m8; gsm_d(j,B->scales,&sc8,&m8);
    float d=__half2float(B->d),dmin=__half2float(B->dmin);
    acc += xs[(size_t)blk*8+j]*(d*(float)sc8*(float)idot - dmin*(float)m8*(float)imsum); }
  for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);
  return acc; }
__global__ void embed_q5k(const block_q5_K_r* We,const int* tokp,float* x,int NEv){
  int p=blockIdx.x*blockDim.x+threadIdx.x; int nb=NEv/256; if(p>=NEv)return; int tok=*tokp;
  const block_q5_K_r* B=We+(size_t)tok*nb+(p>>8); int pp=p&255; int j=pp>>5, l=pp&31, k=j>>1, isodd=j&1;
  uint8_t qb=B->qs[32*k+l]; int nib=isodd?(qb>>4):(qb&0xF); int hbit=(B->qh[l]>>j)&1; int q5=nib+(hbit<<4);
  uint8_t sc8,m8; gsm_d(j,B->scales,&sc8,&m8);
  x[p]=__half2float(B->d)*(float)sc8*(float)q5 - __half2float(B->dmin)*(float)m8; }
__global__ void gemv_q5k_i8(const block_q5_K_r* W,const signed char* xq,const float* xs,float* y,int N,int K){
  int wpb=blockDim.x/warpSize,wid=blockIdx.x*wpb+threadIdx.x/warpSize,nw=gridDim.x*wpb,lane=threadIdx.x%warpSize;
  for(int row=wid;row<N;row+=nw){ float acc=q5k_warprow_i8(W+(size_t)row*(K/256),xq,xs,lane,K); if(lane==0)y[row]=acc; } }
static void loadW_q5k(const char* name,block_q5_K_r** dW,int* No,int* Ko){
  u64 e0,e1; const u8* wd=find(name,&e0,&e1); int K=e0,N=e1,nb=K/256; if(No)*No=N; if(Ko)*Ko=K;
  long nbtot=(long)N*nb; CK(hipMalloc(dW,(size_t)nbtot*sizeof(block_q5_K_r)));   // block_q5_K_r = on-disk layout, direct upload (no repack)
  CK(hipMemcpy(*dW,wd,(size_t)nbtot*sizeof(block_q5_K_r),hipMemcpyHostToDevice)); }
// ===== Q4_K megakernel INT8-activation (sudot4 dot — attacks dequant compute: ~2-4x fewer ALU than f32 q4kc, like Q4_0 win) =====
// per-subblock(32w): idot=sum(nib*q8) via sudot4, imsum=sum(q8); acc+=xs*(d*sc*idot - dmin*m*imsum). Numerically == f32 dequant.
__device__ static float q4k_warprow_i8(const block_q4_K_r* Wr,const signed char* xq,const float* xs,int lane,int K){
  int nb=K/256, nsub=nb*8; float acc=0;
  for(int s=lane;s<nsub;s+=warpSize){ int blk=s>>3, j=s&7;
    const block_q4_K_r* B=Wr+blk; const uint32_t* qs=(const uint32_t*)B->qs;
    const int* xi=(const int*)(xq+(size_t)blk*256);
    int k=j>>1, isodd=j&1, idot=0, imsum=0;
    #pragma unroll
    for(int w=0;w<8;w++){ uint32_t wb=qs[8*k+w]; uint32_t nib=isodd?((wb>>4)&0x0F0F0F0Fu):(wb&0x0F0F0F0Fu); int av=xi[8*j+w];
      idot=__builtin_amdgcn_sudot4(false,(int)nib,true,av,idot,false); imsum=__builtin_amdgcn_sudot4(false,0x01010101,true,av,imsum,false); }
    float d=__half2float(B->d),dmin=__half2float(B->dmin);
    acc += xs[(size_t)blk*8+j]*(d*(float)B->sc[j]*(float)idot - dmin*(float)B->m[j]*(float)imsum); }
  for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);
  return acc; }
__device__ static float kq_row_i8(const void* W,int type,int row,const signed char* xq,const float* xs,int lane,int K){
  int nb=K/256;
  if(type==14) return q6k_warprow_i8((const block_q6_K_r*)W+(size_t)row*nb,xq,xs,lane,K);  // ggml Q6_K
  if(type==13) return q5k_warprow_i8((const block_q5_K_r*)W+(size_t)row*nb,xq,xs,lane,K);  // ggml Q5_K
  return q4k_warprow_i8((const block_q4_K_r*)W+(size_t)row*nb,xq,xs,lane,K); }              // Q4_K (12) default
struct MegaArgsQ4Ki8 { const void *Wq,*Wk,*Wv,*Wo,*Wg,*Wu,*Wd; int tq,tk,tv,to,tg,tu,td; float *Kc,*Vc; const float *an,*fn;
  float *x,*q,*kk,*vv,*ao,*g,*u; signed char *hq,*aoq,*fhq; float *hs,*aos,*fhs; int pos; float eps,scale,base; };
__global__ void __launch_bounds__(256,4) layer_mega_q4k_i8(MegaArgsQ4Ki8 a){
  cg::grid_group grid=cg::this_grid();
  int bid=blockIdx.x,nbk=gridDim.x,tid=threadIdx.x,nth=blockDim.x;
  int gtid=bid*nth+tid,gnth=nbk*nth;
  int wpb=nth/warpSize,wid=bid*wpb+tid/warpSize,nw=nbk*wpb,lane=tid%warpSize;
  int nbE=NE/QK,nbF=NFF/QK; float* x=a.x;
  __shared__ float red[256]; __shared__ float shA[MAXSEQ]; __shared__ float qsh[HD]; __shared__ float mx,sm;
  { float p=0; for(int j=tid;j<NE;j+=nth)p+=x[j]*x[j]; red[tid]=p; __syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    float ss=rsqrtf(red[0]/NE+a.eps); __syncthreads();
    for(int b=gtid;b<nbE;b+=gnth){float vals[QK],am=0;
      for(int e=0;e<QK;e++){int i=b*QK+e;float v=x[i]*ss*a.an[i];vals[e]=v;float aa=fabsf(v);if(aa>am)am=aa;}
      float d=am/127.f,id=d>0?1/d:0;a.hs[b]=d;
      for(int e=0;e<QK;e++){int vq=(int)lrintf(vals[e]*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.hq[b*QK+e]=vq;}} }
  grid.sync();
  { int total=NE+2*NKV*HD;
    for(int r=wid;r<total;r+=nw){const void* W;int ty;float* y;int row;
      if(r<NE){W=a.Wq;ty=a.tq;y=a.q;row=r;}else if(r<NE+NKV*HD){W=a.Wk;ty=a.tk;y=a.kk;row=r-NE;}else{W=a.Wv;ty=a.tv;y=a.vv;row=r-NE-NKV*HD;}
      float acc=kq_row_i8(W,ty,row,a.hq,a.hs,lane,NE); if(lane==0)y[row]=acc;} }
  grid.sync();
  { int totr=(NH+NKV)*(HD/2);
    for(int wk=gtid;wk<totr;wk+=gnth){int h=wk/(HD/2),i=wk%(HD/2);
      float th=a.pos*powf(a.base,-2.0f*i/(float)HD),c=cosf(th),s=sinf(th);
      if(h<NH){float* xr=a.q+(size_t)h*HD;float aa=xr[2*i],bb=xr[2*i+1];xr[2*i]=aa*c-bb*s;xr[2*i+1]=aa*s+bb*c;}
      else{int kvh=h-NH;float* xr=a.kk+(size_t)kvh*HD;float aa=xr[2*i],bb=xr[2*i+1];float r0=aa*c-bb*s,r1=aa*s+bb*c;xr[2*i]=r0;xr[2*i+1]=r1;
        size_t o=((size_t)kvh*MAXSEQ+a.pos)*HD;a.Kc[o+2*i]=r0;a.Kc[o+2*i+1]=r1;a.Vc[o+2*i]=a.vv[kvh*HD+2*i];a.Vc[o+2*i+1]=a.vv[kvh*HD+2*i+1];}} }
  grid.sync();
  if(bid<NH){int h=bid,kvh=h/GQA,seq=a.pos+1;
    for(int d=tid;d<HD;d+=nth)qsh[d]=a.q[(size_t)h*HD+d];__syncthreads();
    for(int t=tid;t<seq;t+=nth){const float* k=a.Kc+((size_t)kvh*MAXSEQ+t)*HD;float dot=0;
      #pragma unroll 4
      for(int d=0;d<HD;d++)dot+=qsh[d]*k[d];shA[t]=dot*a.scale;}__syncthreads();
    float lm=-1e30f;for(int t=tid;t<seq;t+=nth)lm=fmaxf(lm,shA[t]);red[tid]=lm;__syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]=fmaxf(red[tid],red[tid+s]);__syncthreads();}if(tid==0)mx=red[0];__syncthreads();
    float ls=0;for(int t=tid;t<seq;t+=nth){float e=expf(shA[t]-mx);shA[t]=e;ls+=e;}red[tid]=ls;__syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}if(tid==0)sm=red[0];__syncthreads();
    if(tid<HD){float acc=0,inv=1.f/sm;for(int t=0;t<seq;t++)acc+=shA[t]*inv*a.Vc[((size_t)kvh*MAXSEQ+t)*HD+tid];
      float am=fabsf(acc); for(int o=16;o>0;o>>=1)am=fmaxf(am,__shfl_down(am,o,32)); am=__shfl(am,0,32);
      float d=am/127.f,id=d>0?1/d:0; if((tid&31)==0)a.aos[h*4+tid/32]=d;
      int vq=(int)lrintf(acc*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.aoq[(size_t)h*HD+tid]=vq;} }
  grid.sync();
  { for(int r=wid;r<NE;r+=nw){float acc=kq_row_i8(a.Wo,a.to,r,a.aoq,a.aos,lane,NE); if(lane==0)x[r]+=acc;} }
  grid.sync();
  { float p=0; for(int j=tid;j<NE;j+=nth)p+=x[j]*x[j]; red[tid]=p; __syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    float ss=rsqrtf(red[0]/NE+a.eps); __syncthreads();
    for(int b=gtid;b<nbE;b+=gnth){float vals[QK],am=0;
      for(int e=0;e<QK;e++){int i=b*QK+e;float v=x[i]*ss*a.fn[i];vals[e]=v;float aa=fabsf(v);if(aa>am)am=aa;}
      float d=am/127.f,id=d>0?1/d:0;a.hs[b]=d;
      for(int e=0;e<QK;e++){int vq=(int)lrintf(vals[e]*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.hq[b*QK+e]=vq;}} }
  grid.sync();
  { int total=2*NFF;
    for(int r=wid;r<total;r+=nw){const void* W;int ty;float* y;int row;
      if(r<NFF){W=a.Wg;ty=a.tg;y=a.g;row=r;}else{W=a.Wu;ty=a.tu;y=a.u;row=r-NFF;}
      float acc=kq_row_i8(W,ty,row,a.hq,a.hs,lane,NE); if(lane==0)y[row]=acc;} }
  grid.sync();
  { for(int b=gtid;b<nbF;b+=gnth){float vals[QK],am=0;
      for(int e=0;e<QK;e++){int i=b*QK+e;float xv=a.g[i];float v=(xv/(1.f+expf(-xv)))*a.u[i];vals[e]=v;float aa=fabsf(v);if(aa>am)am=aa;}
      float d=am/127.f,id=d>0?1/d:0;a.fhs[b]=d;
      for(int e=0;e<QK;e++){int vq=(int)lrintf(vals[e]*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.fhq[b*QK+e]=vq;}} }
  grid.sync();
  { for(int r=wid;r<NE;r+=nw){float acc=kq_row_i8(a.Wd,a.td,r,a.fhq,a.fhs,lane,NFF); if(lane==0)x[r]+=acc;} }
}
__global__ void embed_q6k(const block_q6_K_r* We,const int* tokp,float* x,int NEv){
  int p=blockIdx.x*blockDim.x+threadIdx.x; if(p>=NEv)return; int tok=*tokp; int nb=NEv/256;
  const block_q6_K_r* B=We+(size_t)tok*nb+(p>>8); int pp=p&255;
  uint8_t qb=B->ql[pp>>1]; int lo=(pp&1)?(qb>>4):(qb&0xF);
  uint8_t hb=B->qh[pp>>2]; int hi=(hb>>(2*(pp&3)))&3;
  x[p]=__half2float(B->d)*(float)B->sc[pp>>4]*(float)((lo|(hi<<4))-32); }
__global__ void gemv_q6k_i8(const block_q6_K_r* W,const signed char* xq,const float* xs,float* y,int N,int K){
  int wpb=blockDim.x/warpSize,wid=blockIdx.x*wpb+threadIdx.x/warpSize,nw=gridDim.x*wpb,lane=threadIdx.x%warpSize;
  for(int row=wid;row<N;row+=nw){ float acc=q6k_warprow_i8(W+(size_t)row*(K/256),xq,xs,lane,K); if(lane==0)y[row]=acc; } }
struct MegaArgsQ6Ki8 { const block_q6_K_r *Wq,*Wk,*Wv,*Wo,*Wg,*Wu,*Wd; float *Kc,*Vc; const float *an,*fn;
  float *x,*q,*kk,*vv,*ao,*g,*u; signed char *hq,*aoq,*fhq; float *hs,*aos,*fhs; int pos; float eps,scale,base; };
__global__ void __launch_bounds__(256,4) layer_mega_q6k_i8(MegaArgsQ6Ki8 a){
  cg::grid_group grid=cg::this_grid();
  int bid=blockIdx.x,nbk=gridDim.x,tid=threadIdx.x,nth=blockDim.x;
  int gtid=bid*nth+tid,gnth=nbk*nth;
  int wpb=nth/warpSize,wid=bid*wpb+tid/warpSize,nw=nbk*wpb,lane=tid%warpSize;
  int nbE=NE/QK,nbF=NFF/QK; float* x=a.x;
  __shared__ float red[256]; __shared__ float shA[MAXSEQ]; __shared__ float qsh[HD]; __shared__ float mx,sm;
  { float p=0; for(int j=tid;j<NE;j+=nth)p+=x[j]*x[j]; red[tid]=p; __syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    float ss=rsqrtf(red[0]/NE+a.eps); __syncthreads();
    for(int b=gtid;b<nbE;b+=gnth){float vals[QK],am=0;
      for(int e=0;e<QK;e++){int i=b*QK+e;float v=x[i]*ss*a.an[i];vals[e]=v;float aa=fabsf(v);if(aa>am)am=aa;}
      float d=am/127.f,id=d>0?1/d:0;a.hs[b]=d;
      for(int e=0;e<QK;e++){int vq=(int)lrintf(vals[e]*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.hq[b*QK+e]=vq;}} }
  grid.sync();
  { int total=NE+2*NKV*HD;
    for(int r=wid;r<total;r+=nw){const block_q6_K_r* W;float* y;int row;
      if(r<NE){W=a.Wq;y=a.q;row=r;}else if(r<NE+NKV*HD){W=a.Wk;y=a.kk;row=r-NE;}else{W=a.Wv;y=a.vv;row=r-NE-NKV*HD;}
      float acc=q6k_warprow_i8(W+(size_t)row*(NE/256),a.hq,a.hs,lane,NE); if(lane==0)y[row]=acc;} }
  grid.sync();
  { int totr=(NH+NKV)*(HD/2);
    for(int wk=gtid;wk<totr;wk+=gnth){int h=wk/(HD/2),i=wk%(HD/2);
      float th=a.pos*powf(a.base,-2.0f*i/(float)HD),c=cosf(th),s=sinf(th);
      if(h<NH){float* xr=a.q+(size_t)h*HD;float aa=xr[2*i],bb=xr[2*i+1];xr[2*i]=aa*c-bb*s;xr[2*i+1]=aa*s+bb*c;}
      else{int kvh=h-NH;float* xr=a.kk+(size_t)kvh*HD;float aa=xr[2*i],bb=xr[2*i+1];float r0=aa*c-bb*s,r1=aa*s+bb*c;xr[2*i]=r0;xr[2*i+1]=r1;
        size_t o=((size_t)kvh*MAXSEQ+a.pos)*HD;a.Kc[o+2*i]=r0;a.Kc[o+2*i+1]=r1;a.Vc[o+2*i]=a.vv[kvh*HD+2*i];a.Vc[o+2*i+1]=a.vv[kvh*HD+2*i+1];}} }
  grid.sync();
  if(bid<NH){int h=bid,kvh=h/GQA,seq=a.pos+1;
    for(int d=tid;d<HD;d+=nth)qsh[d]=a.q[(size_t)h*HD+d];__syncthreads();
    for(int t=tid;t<seq;t+=nth){const float* k=a.Kc+((size_t)kvh*MAXSEQ+t)*HD;float dot=0;
      #pragma unroll 4
      for(int d=0;d<HD;d++)dot+=qsh[d]*k[d];shA[t]=dot*a.scale;}__syncthreads();
    float lm=-1e30f;for(int t=tid;t<seq;t+=nth)lm=fmaxf(lm,shA[t]);red[tid]=lm;__syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]=fmaxf(red[tid],red[tid+s]);__syncthreads();}if(tid==0)mx=red[0];__syncthreads();
    float ls=0;for(int t=tid;t<seq;t+=nth){float e=expf(shA[t]-mx);shA[t]=e;ls+=e;}red[tid]=ls;__syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}if(tid==0)sm=red[0];__syncthreads();
    if(tid<HD){float acc=0,inv=1.f/sm;for(int t=0;t<seq;t++)acc+=shA[t]*inv*a.Vc[((size_t)kvh*MAXSEQ+t)*HD+tid];
      float am=fabsf(acc); for(int o=16;o>0;o>>=1)am=fmaxf(am,__shfl_down(am,o,32)); am=__shfl(am,0,32);
      float d=am/127.f,id=d>0?1/d:0; if((tid&31)==0)a.aos[h*4+tid/32]=d;
      int vq=(int)lrintf(acc*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.aoq[(size_t)h*HD+tid]=vq;} }
  grid.sync();
  { for(int r=wid;r<NE;r+=nw){float acc=q6k_warprow_i8(a.Wo+(size_t)r*(NE/256),a.aoq,a.aos,lane,NE); if(lane==0)x[r]+=acc;} }
  grid.sync();
  { float p=0; for(int j=tid;j<NE;j+=nth)p+=x[j]*x[j]; red[tid]=p; __syncthreads();
    for(int s=nth/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    float ss=rsqrtf(red[0]/NE+a.eps); __syncthreads();
    for(int b=gtid;b<nbE;b+=gnth){float vals[QK],am=0;
      for(int e=0;e<QK;e++){int i=b*QK+e;float v=x[i]*ss*a.fn[i];vals[e]=v;float aa=fabsf(v);if(aa>am)am=aa;}
      float d=am/127.f,id=d>0?1/d:0;a.hs[b]=d;
      for(int e=0;e<QK;e++){int vq=(int)lrintf(vals[e]*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.hq[b*QK+e]=vq;}} }
  grid.sync();
  { int total=2*NFF;
    for(int r=wid;r<total;r+=nw){const block_q6_K_r* W;float* y;int row;
      if(r<NFF){W=a.Wg;y=a.g;row=r;}else{W=a.Wu;y=a.u;row=r-NFF;}
      float acc=q6k_warprow_i8(W+(size_t)row*(NE/256),a.hq,a.hs,lane,NE); if(lane==0)y[row]=acc;} }
  grid.sync();
  { for(int b=gtid;b<nbF;b+=gnth){float vals[QK],am=0;
      for(int e=0;e<QK;e++){int i=b*QK+e;float xv=a.g[i];float v=(xv/(1.f+expf(-xv)))*a.u[i];vals[e]=v;float aa=fabsf(v);if(aa>am)am=aa;}
      float d=am/127.f,id=d>0?1/d:0;a.fhs[b]=d;
      for(int e=0;e<QK;e++){int vq=(int)lrintf(vals[e]*id);if(vq>127)vq=127;if(vq<-127)vq=-127;a.fhq[b*QK+e]=vq;}} }
  grid.sync();
  { for(int r=wid;r<NE;r+=nw){float acc=q6k_warprow_i8(a.Wd+(size_t)r*(NFF/256),a.fhq,a.fhs,lane,NFF); if(lane==0)x[r]+=acc;} }
}
// ===== CATS byte-lever kernels =====
#define KSD 48
__global__ void build_active(const float* gate,float* silu,int* idx,int* cnt,float thr,int nff){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nff)return;float gg=gate[j];float sv=gg/(1.f+expf(-gg));silu[j]=sv;if(fabsf(sv)>=thr){int p=atomicAdd(cnt,1);idx[p]=j;}}
__global__ void sparse_up(const signed char* W,const __half* WS,const signed char* xq,const float* xs,const float* silu,const int* idx,const int* cntp,float* h,int K){int nact=*cntp;int wpb=blockDim.x/warpSize,wa=blockIdx.x*wpb+threadIdx.x/warpSize,nw=gridDim.x*wpb,lane=threadIdx.x%warpSize,nb=K/QK;for(int a=wa;a<nact;a+=nw){int row=idx[a];const signed char* w=W+(size_t)row*K;const __half* ws=WS+(size_t)row*nb;float acc=0;for(int b=lane;b<nb;b+=warpSize){const signed char* wp=w+b*QK;const signed char* xp=xq+b*QK;int s=0;
  #pragma unroll
  for(int i=0;i<QK;i++)s+=(int)wp[i]*(int)xp[i];acc+=(float)s*__half2float(ws[b])*xs[b];}for(int o=warpSize/2;o>0;o>>=1)acc+=__shfl_down(acc,o);if(lane==0)h[row]=silu[row]*acc;}}
// transpose ffn_down {NE rows, NFF contig}(Wd,Sd) -> neuron-major WdT {NFF rows, NE contig}(WdTq,WdTs), requant per NE-block
__global__ void transpose_dn(const signed char* Wd,const __half* Sd,signed char* WdTq,__half* WdTs,int NEo,int NFFo){int j=blockIdx.x;if(j>=NFFo)return;int nbE=NEo/QK,nbF=NFFo/QK;for(int blk=threadIdx.x;blk<nbE;blk+=blockDim.x){float vals[QK],am=0;for(int kk2=0;kk2<QK;kk2++){int i=blk*QK+kk2;float sc=__half2float(Sd[(size_t)i*nbF+j/QK]);float v=sc*(float)Wd[(size_t)i*NFFo+j];vals[kk2]=v;float a=fabsf(v);if(a>am)am=a;}float d=am/127.f,id=d>0?1/d:0;WdTs[(size_t)j*nbE+blk]=__float2half(d);for(int kk2=0;kk2<QK;kk2++){int q=(int)lrintf(vals[kk2]*id);if(q>127)q=127;if(q<-127)q=-127;WdTq[(size_t)j*NEo+blk*QK+kk2]=q;}}}
__global__ void down_cats(const signed char* WdTq,const __half* WdTs,const float* h,const int* idx,const int* cntp,float* partial,int NEo){int nact=*cntp;int t=blockIdx.x*blockDim.x+threadIdx.x;int i=t*16;if(i>=NEo)return;int ks=blockIdx.y;int nbE=NEo/QK;int blk=i>>5;int a0=(int)((long)ks*nact/KSD),a1=(int)((long)(ks+1)*nact/KSD);float c[16];for(int kk2=0;kk2<16;kk2++)c[kk2]=0;for(int a=a0;a<a1;a++){int j=idx[a];float dd=h[j]*__half2float(WdTs[(size_t)j*nbE+blk]);const int4 q4=*reinterpret_cast<const int4*>(WdTq+(size_t)j*NEo+i);const signed char* q=reinterpret_cast<const signed char*>(&q4);
  #pragma unroll
  for(int kk2=0;kk2<16;kk2++)c[kk2]+=dd*(float)q[kk2];}float* p=partial+(size_t)ks*NEo+i;
  #pragma unroll
  for(int kk2=0;kk2<16;kk2++)p[kk2]=c[kk2];}
__global__ void reducep(const float* partial,float* out,int NEo){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NEo)return;float acc=0;for(int ks=0;ks<KSD;ks++)acc+=partial[(size_t)ks*NEo+i];out[i]=acc;}
// GPU embed: dequant row *tokp of token_embd {VOCAB rows, NE contig} -> x. No host round-trip.
__global__ void embed_gpu(const signed char* We,const __half* Se,const int* tokp,float* x){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NE)return;int tok=*tokp;int b=i/QK;x[i]=__half2float(Se[(size_t)tok*(NE/QK)+b])*(float)We[(size_t)tok*NE+i];}
// NLL of target token under logits = logsumexp(logits) - logits[target]. Accumulate into *acc.
__global__ void nll_k(const float* logits,int n,const int* targp,double* acc){__shared__ float red[256];int t=threadIdx.x;float m=-1e30f;for(int i=t;i<n;i+=blockDim.x)m=fmaxf(m,logits[i]);red[t]=m;__syncthreads();for(int s=128;s>0;s>>=1){if(t<s)red[t]=fmaxf(red[t],red[t+s]);__syncthreads();}float mx=red[0];__syncthreads();float ls=0;for(int i=t;i<n;i+=blockDim.x)ls+=expf(logits[i]-mx);red[t]=ls;__syncthreads();for(int s=128;s>0;s>>=1){if(t<s)red[t]+=red[t+s];__syncthreads();}if(t==0)*acc+=(double)(mx+logf(red[0])-logits[*targp]);}

struct LW { signed char *Wq,*Wk,*Wv,*Wo,*Wg,*Wu,*Wd,*WdT; __half *Sq,*Sk,*Sv,*So,*Sg,*Su,*Sd,*WdTs; float *an,*fn; float *Kc,*Vc; int tWq,tWk,tWv,tWo,tWg,tWu,tWd; };
int main(int argc,char**argv){
  const char* path=argc>1?argv[1]:"/home/janusz/teacher_q8/Bielik-11B-v3.0-Instruct.Q8_0.gguf";
  int NGEN=argc>2?atoi(argv[2]):128;
  int fd=open(path,O_RDONLY);struct stat st;fstat(fd,&st);gg=(const u8*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);gp=0;
  loadVocab();
  printf("loading Bielik full model (%d layers)...\n",NL); fflush(stdout);
  int q6k=getenv("Q6K")?1:0; int q4k=(getenv("Q4K")&&!q6k)?1:0; int q4=(getenv("Q4")&&!q4k&&!q6k)?1:0;
  if(q4)printf("Q4_0 mode\n"); if(q4k)printf("Q4_K mode (int8-sudot4)\n"); if(q6k)printf("Q6_K mode (int8-sudot4)\n");
  std::vector<LW> L(NL); char nm[64];
  #define LD(NMV,W,S) do{ if(q4k){ u64 _e,_f; find(NMV,&_e,&_f); if(g_last_type==14) loadW_q6k(NMV,(block_q6_K_r**)&(W),0,0); else if(g_last_type==13) loadW_q5k(NMV,(block_q5_K_r**)&(W),0,0); else loadW_q4k(NMV,(block_q4_K_r**)&(W),0,0); } else if(q6k) loadW_q6k(NMV,(block_q6_K_r**)&(W),0,0); else if(q4) loadW_q4(NMV,(uint8_t**)&(W),&(S),0,0); else loadW(NMV,&(W),&(S),0,0); }while(0)
  #define EMB(P) do{ if(q6k) embed_q6k<<<(NE+255)/256,256>>>((const block_q6_K_r*)We,hist+(P),x,NE); else if(q4k){ if(tEmb==14) embed_q6k<<<(NE+255)/256,256>>>((const block_q6_K_r*)We,hist+(P),x,NE); else if(tEmb==13) embed_q5k<<<(NE+255)/256,256>>>((const block_q5_K_r*)We,hist+(P),x,NE); else embed_q4k<<<(NE/256*32+255)/256,256>>>((const block_q4_K_r*)We,hist+(P),x,NE); } else if(q4) embed_gpu_q4<<<(NE+255)/256,256>>>((const uint8_t*)We,Se,hist+(P),x); else embed_gpu<<<(NE+255)/256,256>>>(We,Se,hist+(P),x); }while(0)
  for(int l=0;l<NL;l++){
    sprintf(nm,"blk.%d.attn_q.weight",l);LD(nm,L[l].Wq,L[l].Sq);L[l].tWq=g_last_type; sprintf(nm,"blk.%d.attn_k.weight",l);LD(nm,L[l].Wk,L[l].Sk);L[l].tWk=g_last_type;
    sprintf(nm,"blk.%d.attn_v.weight",l);LD(nm,L[l].Wv,L[l].Sv);L[l].tWv=g_last_type; sprintf(nm,"blk.%d.attn_output.weight",l);LD(nm,L[l].Wo,L[l].So);L[l].tWo=g_last_type;
    sprintf(nm,"blk.%d.ffn_gate.weight",l);LD(nm,L[l].Wg,L[l].Sg);L[l].tWg=g_last_type; sprintf(nm,"blk.%d.ffn_up.weight",l);LD(nm,L[l].Wu,L[l].Su);L[l].tWu=g_last_type; sprintf(nm,"blk.%d.ffn_down.weight",l);LD(nm,L[l].Wd,L[l].Sd);L[l].tWd=g_last_type;
    sprintf(nm,"blk.%d.attn_norm.weight",l);L[l].an=loadF32(nm); sprintf(nm,"blk.%d.ffn_norm.weight",l);L[l].fn=loadF32(nm);
    CK(hipMalloc(&L[l].Kc,(size_t)NKV*MAXSEQ*HD*4));CK(hipMalloc(&L[l].Vc,(size_t)NKV*MAXSEQ*HD*4));
    if(getenv("CATS")&&!q4){ CK(hipMalloc(&L[l].WdT,(size_t)NFF*NE)); CK(hipMalloc(&L[l].WdTs,(size_t)NFF*(NE/QK)*2));
      transpose_dn<<<NFF,256>>>(L[l].Wd,L[l].Sd,L[l].WdT,L[l].WdTs,NE,NFF); CK(hipDeviceSynchronize());
      if(!getenv("PPL")){ hipFree(L[l].Wd); hipFree(L[l].Sd); L[l].Wd=0; L[l].Sd=0; } }  // free Wd (net-zero); PPL needs BOTH dense+CATS weights
  }
  u64 e0,e1; g_embd=find("token_embd.weight",&e0,&e1);
  signed char* We; __half* Se; LD("token_embd.weight",We,Se); int tEmb=g_last_type;  // embed weights to VRAM (GPU embed)
  signed char* Wout; __half* Sout; LD("output.weight",Wout,Sout); int tOut=g_last_type; float* onorm=loadF32("output_norm.weight");
  // prompt token IDs from env (space-separated, no BOS); engine prepends BOS=1
  std::vector<int> prompt; if(getenv("PROMPT")){ char* pp=strdup(getenv("PROMPT")); for(char* t=strtok(pp," ");t;t=strtok(0," "))prompt.push_back(atoi(t)); free(pp); }
  int PL=1+(int)prompt.size();                        // BOS + prompt
  int* hist; CK(hipMalloc(&hist,((size_t)PL+NGEN+8)*4));
  printf("loaded. generating %d tokens...\n",NGEN); fflush(stdout);
  // buffers
  float *x,*h,*q,*kk,*vv,*ao,*g,*u,*fh,*fo,*logits; signed char *hq,*aoq,*fhq,*xoq; float *hs,*aos,*fhs,*xos; int* dtok;
  CK(hipMalloc(&x,NE*4));CK(hipMalloc(&h,NE*4));CK(hipMalloc(&q,NE*4));CK(hipMalloc(&kk,NKV*HD*4));CK(hipMalloc(&vv,NKV*HD*4));
  CK(hipMalloc(&ao,NE*4));CK(hipMalloc(&g,NFF*4));CK(hipMalloc(&u,NFF*4));CK(hipMalloc(&fh,NFF*4));CK(hipMalloc(&fo,NE*4));CK(hipMalloc(&logits,VOCAB*4));
  CK(hipMalloc(&hq,NFF));CK(hipMalloc(&hs,(NFF/QK)*4));CK(hipMalloc(&aoq,NE));CK(hipMalloc(&aos,(NE/QK)*4));CK(hipMalloc(&fhq,NFF));CK(hipMalloc(&fhs,(NFF/QK)*4));CK(hipMalloc(&xoq,NE));CK(hipMalloc(&xos,(NE/QK)*4));CK(hipMalloc(&dtok,4));
  float* hy; CK(hipMalloc(&hy,NE*4));  // Q4_K: f32 normalized activation for q4kc
  std::vector<float> hx(NE);
  float eps=1e-5f,scale=1.f/sqrtf((float)HD),base=1e6f; int gr=2048,bl=256;
  // CATS byte-lever buffers + per-layer thresholds (deep layers sparser -> skip more at same quality)
  int cats=getenv("CATS")?1:0; float lo=getenv("CATS_LO")?atof(getenv("CATS_LO")):(getenv("CATS_THR")?atof(getenv("CATS_THR")):0.02f);
  float hi=getenv("CATS_HI")?atof(getenv("CATS_HI")):lo;
  float thr_arr[NL]; for(int l=0;l<NL;l++) thr_arr[l]=lo+(hi-lo)*(float)l/(NL-1);
  float *silu_g,*partial; int *idx,*cnt; double* nllacc;
  CK(hipMalloc(&silu_g,NFF*4));CK(hipMalloc(&idx,NFF*4));CK(hipMalloc(&cnt,4));CK(hipMalloc(&partial,(size_t)KSD*NE*4));CK(hipMalloc(&nllacc,8));
  if(cats)printf("CATS ON, thr lo=%.3f hi=%.3f (per-layer)\n",lo,hi);
  int q4k_mega=(q4k&&!getenv("Q4K_UNFUSED"))?1:0;
  int mega=(getenv("MEGA")||q4||q4k_mega||q6k)?1:0; int mega_grid=0;
  if(mega){ hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop,0));
    void* mk=q6k?(void*)layer_mega_q6k_i8:(q4k_mega?(getenv("Q4K_F32")?(void*)layer_mega_q4k:(void*)layer_mega_q4k_i8):(q4?(void*)layer_mega_q4:(void*)layer_mega));
    int maxb=0; CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&maxb,mk,256,0));
    mega_grid=maxb*prop.multiProcessorCount;
    if(getenv("MEGA_BLOCKS")){int mb=atoi(getenv("MEGA_BLOCKS"));if(mb<=mega_grid)mega_grid=mb;} // cap (grid.sync cost scales w/ blocks)
    printf("MEGA ON: maxb=%d/CU x %d CU, using %d blocks (coopLaunch=%d)\n",maxb,prop.multiProcessorCount,mega_grid,prop.cooperativeLaunch); }
  auto fwd=[&](int pos,int use_cats){
    for(int l=0;l<NL;l++){ LW&w=L[l];
      if(q6k){ MegaArgsQ6Ki8 a; a.Wq=(const block_q6_K_r*)w.Wq;a.Wk=(const block_q6_K_r*)w.Wk;a.Wv=(const block_q6_K_r*)w.Wv;a.Wo=(const block_q6_K_r*)w.Wo;a.Wg=(const block_q6_K_r*)w.Wg;a.Wu=(const block_q6_K_r*)w.Wu;a.Wd=(const block_q6_K_r*)w.Wd;
        a.Kc=w.Kc;a.Vc=w.Vc;a.an=w.an;a.fn=w.fn;a.x=x;a.q=q;a.kk=kk;a.vv=vv;a.ao=ao;a.g=g;a.u=u;
        a.hq=hq;a.hs=hs;a.aoq=aoq;a.aos=aos;a.fhq=fhq;a.fhs=fhs;a.pos=pos;a.eps=eps;a.scale=scale;a.base=base;
        void* kargs[1]={&a}; CK(hipLaunchCooperativeKernel((void*)layer_mega_q6k_i8,dim3(mega_grid),dim3(256),kargs,0,0)); continue; }
      if(q4k){
        if(q4k_mega){
          if(getenv("Q4K_F32")){ MegaArgsQ4K a; a.Wq=(const block_q4_K_r*)w.Wq;a.Wk=(const block_q4_K_r*)w.Wk;a.Wv=(const block_q4_K_r*)w.Wv;a.Wo=(const block_q4_K_r*)w.Wo;a.Wg=(const block_q4_K_r*)w.Wg;a.Wu=(const block_q4_K_r*)w.Wu;a.Wd=(const block_q4_K_r*)w.Wd;
            a.Kc=w.Kc;a.Vc=w.Vc;a.an=w.an;a.fn=w.fn;a.x=x;a.hy=hy;a.q=q;a.kk=kk;a.vv=vv;a.ao=ao;a.g=g;a.u=u;a.fh=fh;a.pos=pos;a.eps=eps;a.scale=scale;a.base=base;
            void* kargs[1]={&a}; CK(hipLaunchCooperativeKernel((void*)layer_mega_q4k,dim3(mega_grid),dim3(256),kargs,0,0)); continue; }
          MegaArgsQ4Ki8 a; a.Wq=(const void*)w.Wq;a.Wk=(const void*)w.Wk;a.Wv=(const void*)w.Wv;a.Wo=(const void*)w.Wo;a.Wg=(const void*)w.Wg;a.Wu=(const void*)w.Wu;a.Wd=(const void*)w.Wd;
          a.tq=w.tWq;a.tk=w.tWk;a.tv=w.tWv;a.to=w.tWo;a.tg=w.tWg;a.tu=w.tWu;a.td=w.tWd;
          a.Kc=w.Kc;a.Vc=w.Vc;a.an=w.an;a.fn=w.fn;a.x=x;a.q=q;a.kk=kk;a.vv=vv;a.ao=ao;a.g=g;a.u=u;
          a.hq=hq;a.hs=hs;a.aoq=aoq;a.aos=aos;a.fhq=fhq;a.fhs=fhs;a.pos=pos;a.eps=eps;a.scale=scale;a.base=base;
          void* kargs[1]={&a}; CK(hipLaunchCooperativeKernel((void*)layer_mega_q4k_i8,dim3(mega_grid),dim3(256),kargs,0,0)); continue; }
        rmsnorm<<<1,256>>>(x,w.an,hy,NE,eps);
        q4kc<64,4><<<(NH*HD)/4,64>>>((const block_q4_K_r*)w.Wq,hy,q,NH*HD,NE);
        q4kc<64,4><<<(NKV*HD)/4,64>>>((const block_q4_K_r*)w.Wk,hy,kk,NKV*HD,NE);
        q4kc<64,4><<<(NKV*HD)/4,64>>>((const block_q4_K_r*)w.Wv,hy,vv,NKV*HD,NE);
        rope2<<<NH+NKV,HD/2>>>(q,kk,pos,base); kvappend<<<(NKV*HD+255)/256,256>>>(w.Kc,w.Vc,kk,vv,pos);
        attn<<<NH,256,(pos+1)*4>>>(q,w.Kc,w.Vc,ao,pos+1,scale);
        q4kc<64,4><<<NE/4,64>>>((const block_q4_K_r*)w.Wo,ao,fo,NE,NE); residual<<<(NE+255)/256,256>>>(x,fo,NE);
        rmsnorm<<<1,256>>>(x,w.fn,hy,NE,eps);
        q4kc<64,4><<<NFF/4,64>>>((const block_q4_K_r*)w.Wg,hy,g,NFF,NE);
        q4kc<64,4><<<NFF/4,64>>>((const block_q4_K_r*)w.Wu,hy,u,NFF,NE);
        silumul_f32<<<(NFF+255)/256,256>>>(g,u,fh,NFF);
        q4kc<64,4><<<NE/4,64>>>((const block_q4_K_r*)w.Wd,fh,fo,NE,NFF); residual<<<(NE+255)/256,256>>>(x,fo,NE);
        continue; }
      if(q4){ MegaArgsQ4 a; a.Wq=(const uint8_t*)w.Wq;a.Sq=w.Sq;a.Wk=(const uint8_t*)w.Wk;a.Sk=w.Sk;a.Wv=(const uint8_t*)w.Wv;a.Sv=w.Sv;a.Wo=(const uint8_t*)w.Wo;a.So=w.So;a.Wg=(const uint8_t*)w.Wg;a.Sg=w.Sg;a.Wu=(const uint8_t*)w.Wu;a.Su=w.Su;a.Wd=(const uint8_t*)w.Wd;a.Sd=w.Sd;
        a.Kc=w.Kc;a.Vc=w.Vc;a.an=w.an;a.fn=w.fn;a.x=x;a.hq=hq;a.hs=hs;a.q=q;a.kk=kk;a.vv=vv;a.ao=ao;a.aoq=aoq;a.aos=aos;a.g=g;a.u=u;a.fhq=fhq;a.fhs=fhs;
        a.pos=pos;a.eps=eps;a.scale=scale;a.base=base;
        void* kargs[1]={&a}; CK(hipLaunchCooperativeKernel((void*)layer_mega_q4,dim3(mega_grid),dim3(256),kargs,0,0)); continue; }
      if(mega){ MegaArgs a; a.Wq=w.Wq;a.Sq=w.Sq;a.Wk=w.Wk;a.Sk=w.Sk;a.Wv=w.Wv;a.Sv=w.Sv;a.Wo=w.Wo;a.So=w.So;a.Wg=w.Wg;a.Sg=w.Sg;a.Wu=w.Wu;a.Su=w.Su;a.Wd=w.Wd;a.Sd=w.Sd;
        a.Kc=w.Kc;a.Vc=w.Vc;a.an=w.an;a.fn=w.fn;a.x=x;a.hq=hq;a.hs=hs;a.q=q;a.kk=kk;a.vv=vv;a.ao=ao;a.aoq=aoq;a.aos=aos;a.g=g;a.u=u;a.fhq=fhq;a.fhs=fhs;
        a.pos=pos;a.eps=eps;a.scale=scale;a.base=base;
        void* kargs[1]={&a}; CK(hipLaunchCooperativeKernel((void*)layer_mega,dim3(mega_grid),dim3(256),kargs,0,0)); continue; }
      rmsnorm_q<<<1,256>>>(x,w.an,hq,hs,NE,eps);                       // fused norm+quant
      gemv_qkv<<<gr,bl>>>(w.Wq,w.Sq,w.Wk,w.Sk,w.Wv,w.Sv,hq,hs,q,kk,vv,NE);  // fused q,k,v
      rope2<<<NH+NKV,HD/2>>>(q,kk,pos,base); kvappend<<<(NKV*HD+255)/256,256>>>(w.Kc,w.Vc,kk,vv,pos);  // fused rope q+k
      attn<<<NH,256,(pos+1)*4>>>(q,w.Kc,w.Vc,ao,pos+1,scale);
      quant<<<(NE/QK+255)/256,256>>>(ao,aoq,aos,NE/QK); gemv_acc<<<gr,bl>>>(w.Wo,w.So,aoq,aos,x,NE,NE);  // o-proj fused into residual
      rmsnorm_q<<<1,256>>>(x,w.fn,hq,hs,NE,eps);                       // fused norm+quant
      if(use_cats){
        gemv<<<gr,bl>>>(w.Wg,w.Sg,hq,hs,g,NFF,NE);                    // gate dense (predictor)
        hipMemsetAsync(cnt,0,4,0); hipMemsetAsync(fh,0,NFF*4,0);
        build_active<<<(NFF+255)/256,256>>>(g,silu_g,idx,cnt,thr_arr[l],NFF);
        sparse_up<<<gr,bl>>>(w.Wu,w.Su,hq,hs,silu_g,idx,cnt,fh,NE);   // h[active]=silu*up (nact read on GPU)
        dim3 dg(((NE/16)+255)/256,KSD); down_cats<<<dg,256>>>(w.WdT,w.WdTs,fh,idx,cnt,partial,NE);
        reducep<<<(NE+255)/256,256>>>(partial,fo,NE);
        residual<<<(NE+255)/256,256>>>(x,fo,NE);
      } else {
        gemv_gu<<<gr,bl>>>(w.Wg,w.Sg,w.Wu,w.Su,hq,hs,g,u,NE,NFF);     // fused gate,up
        silumul_q<<<(NFF/QK+255)/256,256>>>(g,u,fhq,fhs,NFF/QK);       // fused silu*u+quant
        gemv_acc<<<gr,bl>>>(w.Wd,w.Sd,fhq,fhs,x,NE,NFF);              // down-proj fused into residual
      }
    }
    if(q6k){ rmsnorm_q<<<1,256>>>(x,onorm,xoq,xos,NE,eps); gemv_q6k_i8<<<gr,bl>>>((const block_q6_K_r*)Wout,xoq,xos,logits,VOCAB,NE); }
    else if(q4k){ if(tOut==14){ rmsnorm_q<<<1,256>>>(x,onorm,xoq,xos,NE,eps); gemv_q6k_i8<<<gr,bl>>>((const block_q6_K_r*)Wout,xoq,xos,logits,VOCAB,NE); } else if(tOut==13){ rmsnorm_q<<<1,256>>>(x,onorm,xoq,xos,NE,eps); gemv_q5k_i8<<<gr,bl>>>((const block_q5_K_r*)Wout,xoq,xos,logits,VOCAB,NE); } else { rmsnorm<<<1,256>>>(x,onorm,hy,NE,eps); q4kc<64,4><<<VOCAB/4,64>>>((const block_q4_K_r*)Wout,hy,logits,VOCAB,NE); } }
    else { rmsnorm_q<<<1,256>>>(x,onorm,xoq,xos,NE,eps);                      // fused norm+quant
    if(q4) gemv_q4<<<gr,bl>>>((const uint8_t*)Wout,Sout,xoq,xos,logits,VOCAB,NE);
    else   gemv<<<gr,bl>>>(Wout,Sout,xoq,xos,logits,VOCAB,NE); }  // logits; argmax done in loop -> hist (fully on-GPU)
  };
  if(getenv("PPL")){ // rigorous lossless check: PPL of dense-generated ref under dense vs CATS
    int RL=getenv("RL")?atoi(getenv("RL")):64; int bos1=1; CK(hipMemcpy(hist,&bos1,4,hipMemcpyHostToDevice));
    for(int p=0;p<RL;p++){ EMB(p); fwd(p,0); argmax_k<<<1,256>>>(logits,VOCAB,hist+p+1); }
    CK(hipDeviceSynchronize()); printf("ref-gen done\n"); fflush(stdout);
    auto ppl=[&](int uc)->double{ CK(hipMemset(nllacc,0,8));
      for(int p=0;p<RL;p++){ EMB(p); fwd(p,uc); nll_k<<<1,256>>>(logits,VOCAB,hist+p+1,nllacc); }
      double acc; CK(hipMemcpy(&acc,nllacc,8,hipMemcpyDeviceToHost)); return exp(acc/RL); };
    double pd=ppl(0),pc=ppl(1);
    printf("PPL(ref=%d): dense=%.4f  CATS(lo=%.3f hi=%.3f)=%.4f  delta=%+.2f%%  %s\n",RL,pd,lo,hi,pc,100*(pc/pd-1.0),(pc/pd-1.0)<0.005?"LOSSLESS ✓":((pc/pd-1.0)<0.02?"near-lossless":"quality cost"));
    return 0; }
  // seq = BOS + prompt; load to hist[0..PL-1]
  std::vector<int> seq; seq.push_back(1); for(int id:prompt)seq.push_back(id);
  CK(hipMemcpy(hist,seq.data(),(size_t)PL*4,hipMemcpyHostToDevice));
  // prefill prompt (untimed): build KV for positions 0..PL-2, keep hist (no argmax)
  for(int p=0;p<PL-1;p++){ EMB(p); fwd(p,cats); }
  CK(hipDeviceSynchronize());
  // timed generation from last prompt token — NO per-token host sync
  hipEvent_t a,b;hipEventCreate(&a);hipEventCreate(&b);hipEventRecord(a);
  for(int p=PL-1;p<PL-1+NGEN;p++){ EMB(p); fwd(p,cats); argmax_k<<<1,256>>>(logits,VOCAB,hist+p+1); }
  hipEventRecord(b);CK(hipEventSynchronize(b)); float ms;hipEventElapsedTime(&ms,a,b);
  std::vector<int> gen(NGEN); CK(hipMemcpy(gen.data(),hist+PL,NGEN*4,hipMemcpyDeviceToHost));
  double tps=NGEN/(ms/1000.0);
  if(cats){ int nact; CK(hipMemcpy(&nact,cnt,4,hipMemcpyDeviceToHost)); printf("CATS last-layer active=%d/%d (s=%.2f skipped)\n",nact,NFF,1.0-(double)nact/NFF); }
  const char* mode=getenv("MEGA")?"MEGA":(cats?"CATS":"dense");
  printf("FULL ENGINE Bielik decode [%s]: %d tokens in %.1f ms = %.2f t/s  (vs Vulkan 48.88 same GPU)\n",mode,NGEN,ms,tps);
  // detokenized Q8 output
  std::string ptxt; for(int id:prompt)detok(id,ptxt);
  std::string gtxt; for(int g:gen){ if(g==2)break; detok(g,gtxt); }   // stop at EOS=2
  printf("\n=== Q8 OUTPUT ===\nPROMPT: %s\nGEN:    %s\n=================\n",ptxt.c_str(),gtxt.c_str());
  int uniq=0; for(int i=0;i<(int)gen.size();i++){int u=1;for(int j=0;j<i;j++)if(gen[j]==gen[i]){u=0;break;}uniq+=u;}
  printf("unique tokens: %d/%d %s\n",uniq,(int)gen.size(), uniq>NGEN/4?"(varied — not degenerate)":"(DEGENERATE — likely bug)");
  return 0;
}
