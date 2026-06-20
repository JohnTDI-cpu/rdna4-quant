// full_forward_prof.hip — PER-OP instrumented full-forward profiler, Bielik-11B prefill, gfx1201/RDNA4/HIP.
//
//   PURPOSE (forensic): decompose the realistic-COLD 50-layer full forward into EVERY op category and
//   measure each op's wall time (hipEvent, per layer, summed over 50 layers, min-of-bursts), so we know
//   EXACTLY where the time goes for the quants we LOSE to RADV:
//       Q8     @pp512  -8%   @pp8192 -3%
//       Q4_0   @pp512 -17%
//       Q4_K_M @pp512  -3%
//
//   The GEMM is int8 (the source quant Q8/Q4_K_M/Q4_0 is dequant->int8->requant32 before the GEMM), so the
//   per-op breakdown is IDENTICAL across quants EXCEPT the weight-read bandwidth into the GEMM. Therefore we
//   also isolate, per weight GEMM (qkv/gate/down), the L2-cold-weight tax: HOT (8 back-to-back, weight stays
//   L2-warm) vs COLD (64MB L2 flush before each call). The (COLD-HOT) delta = the weight-bandwidth component
//   that a native-4-bit weight read would HALVE for Q4_K_M/Q4_0; HOT ~= the WMMA-compute floor (codegen
//   ceiling, fixed regardless of source quant). This tells us how much of each losing GEMM is bandwidth
//   (recoverable by 4-bit) vs compute (fixed).
//
//   Kernels REUSED (structure copied verbatim from full_forward_v3.hip — that file is NOT modified):
//     gemm_v3p / gemm_v3p_res   — the v3p (fragment-pipeline+swizzle) INT8 WMMA GEMM (== gemm_rb_v3.hip's)
//     flash_prefill_v1          — FA2 (included from ../../hip_int4/flash_prefill.hip, #pragma once)
//     rmsnorm_quant_p8, quant_only, silumul_quant, rope_qk, split_qkv_to_fa2, fa2_to_tokenmajor
//
//   Realistic-COLD: ROT=3 distinct weight buffers per shape, layer%ROT cycles, so each layer's weight GEMM
//   reads a cold VRAM region (each weight >8MB L2 => any one evicts the others). No per-GEMM flush in the
//   main per-op pass => wall == pure realistic-cold forward, nothing to subtract.
//
//   Bielik-11B dims: NE=4096 NH=32 NKV=8 HD=128 NFF=14336 NL=50.
//
//   Build:  HIP_VISIBLE_DEVICES=1 hipcc --offload-arch=gfx1201 -O3 full_forward_prof.hip -o /tmp/ffprof
//   Run:    rocm-smi --setperflevel auto && HIP_VISIBLE_DEVICES=1 /tmp/ffprof
//
//   FRESH RADV baseline (-fa auto):  pp512=2373  pp2048=2289  pp8192=1989 t/s   (from full_forward_v3.hip)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <string>

typedef int i4  __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef __half2 h2;
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
#define CK(x) do{hipError_t e=(x);if(e){printf("HIP err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

#include "/home/janusz/AMD MXFP4/hip_int4/flash_prefill.hip"

// gfx12 split-counter wait (P1, used by v3p) — same as full_forward_v3.hip
template<int N> __device__ __forceinline__ void wait_loadN(){ asm volatile("s_wait_loadcnt %0"::"i"(N):"memory"); }

#define ROT 3   // distinct weight buffers per shape (>=3 so any one evicts the others; layer%ROT cycles)

// ===================================================================================================
//  GEMMs — gemm_v3p / gemm_v3p_res (fragment-pipeline + XOR swizzle), copied verbatim from
//  full_forward_v3.hip (which copied them from gemm_rb_v3.hip). int8 in, half out, scale-after.
// ===================================================================================================
__device__ __forceinline__ int swz16(int row,int blk16,int KC){
  int nblk=KC/16; if(nblk<=1) return blk16; return blk16 ^ ((row>>1)&(nblk-1)); }

template<int BM,int BN,int WMW,int WNW,int KC,int SWZ>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_v3p(
    const int8_t* __restrict__ act, const float* __restrict__ a_scale,
    const int8_t* __restrict__ wt,  const float* __restrict__ w_scale,
    __half* __restrict__ out, int M, int N, int K){
  constexpr int NWARP=WMW*WNW, NTHREAD=NWARP*32, WMT=BM/(WMW*16), WNT=BN/(WNW*16);
  constexpr int AV=BM*KC/16, BV=BN*KC/16, APT=(AV+NTHREAD-1)/NTHREAD, BPT=(BV+NTHREAD-1)/NTHREAD;
  constexpr int KSUB=KC/16;
  __shared__ int8_t As[2][BM][KC]; __shared__ int8_t Bs[2][BN][KC];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16, wm=warp/WNW, wn=warp%WNW;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/KC;
  float Rf[WMT][WNT][8];
  for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)for(int j=0;j<8;j++)Rf[a][b][j]=0.f;
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+blk16*16];}}
  for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[0][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+blk16*16];}}
  __syncthreads(); int buf=0;
  auto fetchA=[&](v2i Af[WMT],int b,int ks){
    #pragma unroll
    for(int mt=0;mt<WMT;mt++){int row=wm*WMT*16+mt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* ap=&As[b][row][bb*16+kg*8];Af[mt]=(v2i){*(const int*)ap,*(const int*)(ap+4)};} };
  auto fetchB=[&](v2i Bf[WNT],int b,int ks){
    #pragma unroll
    for(int nt=0;nt<WNT;nt++){int row=wn*WNT*16+nt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* bp=&Bs[b][row][bb*16+kg*8];Bf[nt]=(v2i){*(const int*)bp,*(const int*)(bp+4)};} };
  for(int k0=0;k0<K;k0+=KC){ int blk=k0/KC; i4 rA[APT],rB[BPT]; int kn=k0+KC;
    if(kn<K){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kn+c];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rB[p]=*(const i4*)&wt[(size_t)(n_base+row)*K+kn+c];}} }
    v8i cR[WMT][WNT]; for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)cR[a][b]=(v8i){};
    v2i Af0[WMT],Bf0[WNT],Af1[WMT],Bf1[WNT];
    fetchA(Af0,buf,0); fetchB(Bf0,buf,0);
    #pragma unroll
    for(int ks=0;ks<KSUB;ks++){
      v2i* Acur=(ks&1)?Af1:Af0; v2i* Bcur=(ks&1)?Bf1:Bf0;
      v2i* Anext=(ks&1)?Af0:Af1; v2i* Bnext=(ks&1)?Bf0:Bf1;
      if(ks+1<KSUB){ fetchA(Anext,buf,ks+1); fetchB(Bnext,buf,ks+1); }
      #pragma unroll
      for(int mt=0;mt<WMT;mt++)
        #pragma unroll
        for(int nt=0;nt<WNT;nt++)
          cR[mt][nt]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,Acur[mt],true,Bcur[nt],cR[mt][nt],false);
    }
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=a_scale[(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk];
    for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[mt][nt][j]*wsc[nt]*asc[mt][j];
    if(kn<K){ wait_loadN<0>();
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
}

template<int BM,int BN,int WMW,int WNW,int KC,int SWZ>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_v3p_res(
    const int8_t* __restrict__ act, const float* __restrict__ a_scale,
    const int8_t* __restrict__ wt,  const float* __restrict__ w_scale,
    __half* __restrict__ out, int M, int N, int K){
  constexpr int NWARP=WMW*WNW, NTHREAD=NWARP*32, WMT=BM/(WMW*16), WNT=BN/(WNW*16);
  constexpr int AV=BM*KC/16, BV=BN*KC/16, APT=(AV+NTHREAD-1)/NTHREAD, BPT=(BV+NTHREAD-1)/NTHREAD;
  constexpr int KSUB=KC/16;
  __shared__ int8_t As[2][BM][KC]; __shared__ int8_t Bs[2][BN][KC];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16, wm=warp/WNW, wn=warp%WNW;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/KC;
  float Rf[WMT][WNT][8];
  for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)for(int j=0;j<8;j++)Rf[a][b][j]=0.f;
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+blk16*16];}}
  for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[0][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+blk16*16];}}
  __syncthreads(); int buf=0;
  auto fetchA=[&](v2i Af[WMT],int b,int ks){
    #pragma unroll
    for(int mt=0;mt<WMT;mt++){int row=wm*WMT*16+mt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* ap=&As[b][row][bb*16+kg*8];Af[mt]=(v2i){*(const int*)ap,*(const int*)(ap+4)};} };
  auto fetchB=[&](v2i Bf[WNT],int b,int ks){
    #pragma unroll
    for(int nt=0;nt<WNT;nt++){int row=wn*WNT*16+nt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* bp=&Bs[b][row][bb*16+kg*8];Bf[nt]=(v2i){*(const int*)bp,*(const int*)(bp+4)};} };
  for(int k0=0;k0<K;k0+=KC){ int blk=k0/KC; i4 rA[APT],rB[BPT]; int kn=k0+KC;
    if(kn<K){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kn+c];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rB[p]=*(const i4*)&wt[(size_t)(n_base+row)*K+kn+c];}} }
    v8i cR[WMT][WNT]; for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)cR[a][b]=(v8i){};
    v2i Af0[WMT],Bf0[WNT],Af1[WMT],Bf1[WNT];
    fetchA(Af0,buf,0); fetchB(Bf0,buf,0);
    #pragma unroll
    for(int ks=0;ks<KSUB;ks++){
      v2i* Acur=(ks&1)?Af1:Af0; v2i* Bcur=(ks&1)?Bf1:Bf0;
      v2i* Anext=(ks&1)?Af0:Af1; v2i* Bnext=(ks&1)?Bf0:Bf1;
      if(ks+1<KSUB){ fetchA(Anext,buf,ks+1); fetchB(Bnext,buf,ks+1); }
      #pragma unroll
      for(int mt=0;mt<WMT;mt++)
        #pragma unroll
        for(int nt=0;nt<WNT;nt++)
          cR[mt][nt]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,Acur[mt],true,Bcur[nt],cR[mt][nt],false);
    }
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=a_scale[(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk];
    for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[mt][nt][j]*wsc[nt]*asc[mt][j];
    if(kn<K){ wait_loadN<0>();
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){
    int m=m_base+wm*WMT*16+mt*16+kg*8+j; size_t o=(size_t)m*N+n_base+wn*WNT*16+nt*16+nl;
    out[o]=__float2half(__half2float(out[o])+Rf[mt][nt][j]); }
}

// ===================================================================================================
//  Mem-op kernels (copied verbatim from full_forward_v3.hip)
// ===================================================================================================
__global__ void rmsnorm_quant_p8(const __half* __restrict__ x, const float* __restrict__ w,
                                 int8_t* __restrict__ q, float* __restrict__ s, int M, int K, float eps){
  int row=blockIdx.x; if(row>=M) return; int tid=threadIdx.x, nth=blockDim.x;
  const __half* xr=x+(size_t)row*K; const h2* xr2=(const h2*)xr; int K2=K/2;
  __shared__ float red[256];
  float p=0; for(int j=tid;j<K2;j+=nth){float2 v=__half22float2(xr2[j]); p+=v.x*v.x+v.y*v.y;} red[tid]=p; __syncthreads();
  for(int st=nth/2;st>0;st>>=1){if(tid<st)red[tid]+=red[tid+st];__syncthreads();}
  float inv=rsqrtf(red[0]/K+eps); __syncthreads();
  int nb=K/32;
  for(int b=tid;b<nb;b+=nth){ float vals[32],am=0;
    const h2* xb=(const h2*)(xr+b*32); const float* wb=w+b*32;
    #pragma unroll
    for(int q2=0;q2<16;q2++){ float2 f=__half22float2(xb[q2]);
      float v0=f.x*inv*wb[2*q2], v1=f.y*inv*wb[2*q2+1]; vals[2*q2]=v0; vals[2*q2+1]=v1;
      am=fmaxf(am,fmaxf(fabsf(v0),fabsf(v1))); }
    float d=am*(1.f/127.f), id=d>0?1.f/d:0.f; s[(size_t)row*nb+b]=d;
    int8_t qb[32];
    #pragma unroll
    for(int e=0;e<32;e++){int vq=(int)lrintf(vals[e]*id); vq=vq>127?127:(vq<-127?-127:vq); qb[e]=(int8_t)vq;}
    *(i4*)&q[(size_t)row*32*nb+b*32]=*(const i4*)&qb[0];
    *(i4*)&q[(size_t)row*32*nb+b*32+16]=*(const i4*)&qb[16]; }
}
__global__ void quant_only(const __half* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ s, int M, int K){
  int row=blockIdx.x; if(row>=M) return; int tid=threadIdx.x, nth=blockDim.x; int nb=K/32;
  const __half* xr=x+(size_t)row*K;
  for(int b=tid;b<nb;b+=nth){ float vals[32],am=0;
    for(int e=0;e<32;e++){int k=b*32+e; float v=__half2float(xr[k]); vals[e]=v; float a=fabsf(v); if(a>am)am=a;}
    float d=am/127.f, id=d>0?1.f/d:0.f; s[(size_t)row*nb+b]=d;
    for(int e=0;e<32;e++){int vq=(int)lrintf(vals[e]*id); if(vq>127)vq=127; if(vq<-127)vq=-127; q[(size_t)row*32*nb+b*32+e]=(int8_t)vq;} }
}
__global__ void silumul_quant(const __half* __restrict__ g, const __half* __restrict__ u,
                              int8_t* __restrict__ q, float* __restrict__ s, int M, int K){
  int row=blockIdx.x; if(row>=M)return; int tid=threadIdx.x,nth=blockDim.x; int nb=K/32;
  const __half* gr=g+(size_t)row*K; const __half* ur=u+(size_t)row*K;
  for(int b=tid;b<nb;b+=nth){ float vals[32],am=0;
    for(int e=0;e<32;e++){int k=b*32+e; float gv=__half2float(gr[k]); float v=(gv/(1.f+expf(-gv)))*__half2float(ur[k]); vals[e]=v; float a=fabsf(v); if(a>am)am=a;}
    float d=am/127.f, id=d>0?1.f/d:0.f; s[(size_t)row*nb+b]=d;
    for(int e=0;e<32;e++){int vq=(int)lrintf(vals[e]*id); if(vq>127)vq=127; if(vq<-127)vq=-127; q[(size_t)row*32*nb+b*32+e]=(int8_t)vq;} }
}
__global__ void rope_qk(__half* __restrict__ qkv, int M, int N, int HD, int n_q_cols, int n_k_cols, float theta){
  int total_pairs = M*((n_q_cols+n_k_cols)/2);
  int gid = blockIdx.x*blockDim.x + threadIdx.x; if(gid>=total_pairs) return;
  int pairs_per_tok = (n_q_cols+n_k_cols)/2;
  int tok = gid / pairs_per_tok, p = gid % pairs_per_tok; int half = HD/2;
  int col_base, head_local;
  if(p < n_q_cols/2){ col_base = 0; head_local = p; } else { col_base = n_q_cols; head_local = p - n_q_cols/2; }
  int head = head_local / half, i = head_local % half;
  float freq = powf(theta, -2.f*(float)i/(float)HD); float ang = (float)tok * freq;
  float c = cosf(ang), s = sinf(ang);
  int c0 = col_base + head*HD + i, c1 = c0 + half;
  size_t base = (size_t)tok*N;
  float x0 = __half2float(qkv[base+c0]), x1 = __half2float(qkv[base+c1]);
  qkv[base+c0] = __float2half(x0*c - x1*s); qkv[base+c1] = __float2half(x0*s + x1*c);
}
__global__ void split_qkv_to_fa2(const __half* __restrict__ qkv, __half* __restrict__ Q,
                                 __half* __restrict__ K, __half* __restrict__ V,
                                 int M, int N, int NH, int NKV, int HD){
  int gid = blockIdx.x*blockDim.x+threadIdx.x; if(gid>=M*N) return;
  int tok = gid / N, col = gid % N; int qcols = NH*HD, kcols = NKV*HD;
  __half v = qkv[(size_t)tok*N+col];
  if(col < qcols){ int h=col/HD, d=col%HD; Q[((size_t)h*M+tok)*HD+d]=v; }
  else if(col < qcols+kcols){ int c=col-qcols; int h=c/HD, d=c%HD; K[((size_t)h*M+tok)*HD+d]=v; }
  else { int c=col-qcols-kcols; int h=c/HD, d=c%HD; V[((size_t)h*M+tok)*HD+d]=v; }
}
__global__ void fa2_to_tokenmajor(const __half* __restrict__ O, __half* __restrict__ x, int M, int NH, int HD){
  int gid = blockIdx.x*blockDim.x+threadIdx.x; int N=NH*HD; if(gid>=M*N) return;
  int tok=gid/N, col=gid%N; int h=col/HD, d=col%HD;
  x[(size_t)tok*N+col]=O[((size_t)h*M+tok)*HD+d];
}

__global__ void l2_flush(int8_t* p, size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]^=1; }

// ===================================================================================================
//  GEMM dispatch wrappers — v3p only (the production winner). 256x256 8x4 KC=32 swz.
// ===================================================================================================
static void gemm_plain(const int8_t*a,const float*as,const int8_t*w,const float*ws,__half*o,int M,int N,int K,hipStream_t st){
  dim3 g(N/256,M/256),b(8*4*32);
  gemm_v3p<256,256,8,4,32,1><<<g,b,0,st>>>(a,as,w,ws,o,M,N,K);
}
static void gemm_res(const int8_t*a,const float*as,const int8_t*w,const float*ws,__half*o,int M,int N,int K,hipStream_t st){
  dim3 g(N/256,M/256),b(8*4*32);
  gemm_v3p_res<256,256,8,4,32,1><<<g,b,0,st>>>(a,as,w,ws,o,M,N,K);
}

// ===================================================================================================
//  Buffers (copied from full_forward_v3.hip).
// ===================================================================================================
struct Buffers {
  int8_t *wqkv[ROT],*wo[ROT],*wgate[ROT],*wup[ROT],*wdown[ROT];
  float  *sqkv[ROT],*so[ROT],*sgate[ROT],*sup[ROT],*sdown[ROT];
  float  *wn_attn,*wn_ffn;
  __half *x;
  int8_t *aq; float *as;
  __half *qkv;
  __half *Q,*K,*V,*Oattn;
  __half *attn_tm;
  int8_t *aq_o; float *as_o;
  int8_t *aq_ffn; float *as_ffn;
  __half *gate,*up;
  int8_t *aq_dn; float *as_dn;
};
static void alloc_buffers(Buffers&B,int M,int NE,int NH,int NKV,int HD,int NFF){
  int Nqkv=NH*HD+2*NKV*HD;
  for(int r=0;r<ROT;r++){
    CK(hipMalloc(&B.wqkv[r],(size_t)Nqkv*NE));      CK(hipMalloc(&B.sqkv[r],(size_t)Nqkv*(NE/32)*4));
    CK(hipMalloc(&B.wo[r],(size_t)NE*NE));          CK(hipMalloc(&B.so[r],(size_t)NE*(NE/32)*4));
    CK(hipMalloc(&B.wgate[r],(size_t)NFF*NE));      CK(hipMalloc(&B.sgate[r],(size_t)NFF*(NE/32)*4));
    CK(hipMalloc(&B.wup[r],(size_t)NFF*NE));        CK(hipMalloc(&B.sup[r],(size_t)NFF*(NE/32)*4));
    CK(hipMalloc(&B.wdown[r],(size_t)NE*NFF));      CK(hipMalloc(&B.sdown[r],(size_t)NE*(NFF/32)*4));
    CK(hipMemset(B.wqkv[r],1+r,(size_t)Nqkv*NE)); CK(hipMemset(B.wo[r],1+r,(size_t)NE*NE));
    CK(hipMemset(B.wgate[r],1+r,(size_t)NFF*NE)); CK(hipMemset(B.wup[r],1+r,(size_t)NFF*NE)); CK(hipMemset(B.wdown[r],1+r,(size_t)NE*NFF));
    CK(hipMemset(B.sqkv[r],0,(size_t)Nqkv*(NE/32)*4)); CK(hipMemset(B.so[r],0,(size_t)NE*(NE/32)*4));
    CK(hipMemset(B.sgate[r],0,(size_t)NFF*(NE/32)*4)); CK(hipMemset(B.sup[r],0,(size_t)NFF*(NE/32)*4)); CK(hipMemset(B.sdown[r],0,(size_t)NE*(NFF/32)*4));
  }
  CK(hipMalloc(&B.wn_attn,(size_t)NE*4));      CK(hipMalloc(&B.wn_ffn,(size_t)NE*4));
  CK(hipMalloc(&B.x,(size_t)M*NE*2));
  CK(hipMalloc(&B.aq,(size_t)M*NE));           CK(hipMalloc(&B.as,(size_t)M*(NE/32)*4));
  CK(hipMalloc(&B.qkv,(size_t)M*Nqkv*2));
  CK(hipMalloc(&B.Q,(size_t)NH*M*HD*2));       CK(hipMalloc(&B.K,(size_t)NKV*M*HD*2));
  CK(hipMalloc(&B.V,(size_t)NKV*M*HD*2));      CK(hipMalloc(&B.Oattn,(size_t)NH*M*HD*2));
  CK(hipMalloc(&B.attn_tm,(size_t)M*NE*2));
  CK(hipMalloc(&B.aq_o,(size_t)M*NE));         CK(hipMalloc(&B.as_o,(size_t)M*(NE/32)*4));
  CK(hipMalloc(&B.aq_ffn,(size_t)M*NE));       CK(hipMalloc(&B.as_ffn,(size_t)M*(NE/32)*4));
  CK(hipMalloc(&B.gate,(size_t)M*NFF*2));      CK(hipMalloc(&B.up,(size_t)M*NFF*2));
  CK(hipMalloc(&B.aq_dn,(size_t)M*NFF));       CK(hipMalloc(&B.as_dn,(size_t)M*(NFF/32)*4));
  CK(hipMemset(B.wn_attn,0x3c,(size_t)NE*4)); CK(hipMemset(B.wn_ffn,0x3c,(size_t)NE*4));
  CK(hipMemset(B.x,1,(size_t)M*NE*2));
  CK(hipMemset(B.aq,1,(size_t)M*NE)); CK(hipMemset(B.as,0,(size_t)M*(NE/32)*4));
  CK(hipMemset(B.aq_ffn,1,(size_t)M*NE)); CK(hipMemset(B.as_ffn,0,(size_t)M*(NE/32)*4));
  CK(hipMemset(B.aq_dn,1,(size_t)M*NFF)); CK(hipMemset(B.as_dn,0,(size_t)M*(NFF/32)*4));
  CK(hipMemset(B.aq_o,1,(size_t)M*NE)); CK(hipMemset(B.as_o,0,(size_t)M*(NE/32)*4));
}
static void free_buffers(Buffers&B){
  for(int r=0;r<ROT;r++){
    hipFree(B.wqkv[r]);hipFree(B.sqkv[r]);hipFree(B.wo[r]);hipFree(B.so[r]);hipFree(B.wgate[r]);hipFree(B.sgate[r]);
    hipFree(B.wup[r]);hipFree(B.sup[r]);hipFree(B.wdown[r]);hipFree(B.sdown[r]);
  }
  hipFree(B.wn_attn);hipFree(B.wn_ffn);
  hipFree(B.x);hipFree(B.aq);hipFree(B.as);hipFree(B.qkv);hipFree(B.Q);hipFree(B.K);hipFree(B.V);hipFree(B.Oattn);
  hipFree(B.attn_tm);hipFree(B.aq_o);hipFree(B.as_o);hipFree(B.aq_ffn);hipFree(B.as_ffn);
  hipFree(B.gate);hipFree(B.up);hipFree(B.aq_dn);hipFree(B.as_dn);
}

// ===================================================================================================
//  PER-OP INSTRUMENTATION
//  14 op categories. We time each with a pair of hipEvents bracketing the op's launches, ACCUMULATING
//  the elapsed ms across all 50 layers into acc[cat]. The events serialize the stream at each bracket
//  (hipEventRecord+the next op waits), which adds a tiny fixed sync cost per bracket — that cost is the
//  SAME across all categories and is exactly the "measurement perturbation"; we ALSO time the whole
//  forward with NO per-op events (clean wall) so the per-op sum can be compared to the clean wall and the
//  gap (= dispatch/scheduling overhead + event perturbation) is reported honestly.
// ===================================================================================================
enum {
  OP_INRMS=0, OP_QUANT_QKV, OP_QKV, OP_ROPE, OP_REPACK, OP_FA2, OP_QUANT_ATTN,
  OP_OPROJ, OP_POSTRMS, OP_QUANT_FFN, OP_GATE, OP_UP, OP_SILU, OP_DOWN, NOP
};
static const char* OPNAME[NOP] = {
  "1  input rmsnorm (fused quant)",
  "2  act quant ->int8 [in P8, n/a]",
  "3  qkv GEMM (N=6144)",
  "4  RoPE (Q,K)",
  "5  repack qkv->FA2 [H,M,D]",
  "6  FA2 attention",
  "7  quant attn-out ->int8",
  "8  o-proj GEMM (N=4096) + resid",
  "9  post-attn rmsnorm (fused quant)",
  "10 act quant ->int8 [in P8, n/a]",
  "11 gate GEMM (N=14336)",
  "12 up GEMM (N=14336)",
  "13 silu(gate)*up + quant",
  "14 down GEMM (N=4096 K=14336) + resid",
};

// Per-op event pool: 2*NOP events per layer-instance; we just reuse a single pair and accumulate.
struct OpTimer {
  hipEvent_t a, b;
  double acc[NOP];
  OpTimer(){ CK(hipEventCreate(&a)); CK(hipEventCreate(&b)); reset(); }
  void reset(){ for(int i=0;i<NOP;i++) acc[i]=0; }
  // run a closure between events, add elapsed to acc[cat]
};

// One INSTRUMENTED layer: brackets EACH op category with events, accumulates into T.acc.
// The events force the stream to flush at each boundary => per-op time is the op's own GPU time
// (cold weights for the GEMMs because we still rotate buffers). Note this is per-op wall incl. the
// kernel's own launch latency, which is what we want (it IS part of the forward).
static void run_layer_instrumented(Buffers&B,int M,int NE,int NH,int NKV,int HD,int NFF,float eps,float theta,
                                   int rb,hipStream_t st,OpTimer&T){
  int Nqkv=NH*HD+2*NKV*HD; int n_q_cols=NH*HD, n_k_cols=NKV*HD; int tpb=256; float ms;
  #define TIME(cat, ...) do{ CK(hipEventRecord(T.a,st)); { __VA_ARGS__; } CK(hipEventRecord(T.b,st)); \
      CK(hipEventSynchronize(T.b)); CK(hipEventElapsedTime(&ms,T.a,T.b)); T.acc[cat]+=ms; }while(0)

  // 1 input rmsnorm (fused quant) — categories 2 (act quant) folded into P8, reported n/a
  TIME(OP_INRMS, rmsnorm_quant_p8<<<M,256,0,st>>>(B.x,B.wn_attn,B.aq,B.as,M,NE,eps); );
  // 3 qkv GEMM
  TIME(OP_QKV, gemm_plain(B.aq,B.as,B.wqkv[rb],B.sqkv[rb],B.qkv,M,Nqkv,NE,st); );
  // 4 RoPE
  TIME(OP_ROPE, { int pairs=M*((n_q_cols+n_k_cols)/2); int g=(pairs+tpb-1)/tpb;
        rope_qk<<<g,tpb,0,st>>>(B.qkv,M,Nqkv,HD,n_q_cols,n_k_cols,theta); } );
  // 5 repack qkv -> FA2 layout
  TIME(OP_REPACK, { int g=(M*Nqkv+tpb-1)/tpb; split_qkv_to_fa2<<<g,tpb,0,st>>>(B.qkv,B.Q,B.K,B.V,M,Nqkv,NH,NKV,HD); } );
  // 6 FA2 attention
  TIME(OP_FA2, launch_flash_prefill_v1(B.Q,B.K,B.V,B.Oattn,M,NH,NKV,HD,st); );
  // 7 quant attn-out (incl. fa2_to_tokenmajor repack into x-major, then quant)
  TIME(OP_QUANT_ATTN, { int g=(M*NE+tpb-1)/tpb; fa2_to_tokenmajor<<<g,tpb,0,st>>>(B.Oattn,B.attn_tm,M,NH,HD);
        quant_only<<<M,256,0,st>>>(B.attn_tm,B.aq_o,B.as_o,M,NE); } );
  // 8 o-proj GEMM + residual into x
  TIME(OP_OPROJ, gemm_res(B.aq_o,B.as_o,B.wo[rb],B.so[rb],B.x,M,NE,NE,st); );
  // 9 post-attn rmsnorm (fused quant)
  TIME(OP_POSTRMS, rmsnorm_quant_p8<<<M,256,0,st>>>(B.x,B.wn_ffn,B.aq_ffn,B.as_ffn,M,NE,eps); );
  // 11 gate GEMM
  TIME(OP_GATE, gemm_plain(B.aq_ffn,B.as_ffn,B.wgate[rb],B.sgate[rb],B.gate,M,NFF,NE,st); );
  // 12 up GEMM
  TIME(OP_UP, gemm_plain(B.aq_ffn,B.as_ffn,B.wup[rb],B.sup[rb],B.up,M,NFF,NE,st); );
  // 13 silu*up + quant
  TIME(OP_SILU, silumul_quant<<<M,256,0,st>>>(B.gate,B.up,B.aq_dn,B.as_dn,M,NFF); );
  // 14 down GEMM + residual into x
  TIME(OP_DOWN, gemm_res(B.aq_dn,B.as_dn,B.wdown[rb],B.sdown[rb],B.x,M,NE,NFF,st); );
  #undef TIME
}

// One CLEAN layer (no per-op events) — for the clean-wall reference. Identical op sequence.
static void run_layer_clean(Buffers&B,int M,int NE,int NH,int NKV,int HD,int NFF,float eps,float theta,
                            int rb,hipStream_t st){
  int Nqkv=NH*HD+2*NKV*HD; int n_q_cols=NH*HD, n_k_cols=NKV*HD; int tpb=256;
  rmsnorm_quant_p8<<<M,256,0,st>>>(B.x,B.wn_attn,B.aq,B.as,M,NE,eps);
  gemm_plain(B.aq,B.as,B.wqkv[rb],B.sqkv[rb],B.qkv,M,Nqkv,NE,st);
  { int pairs=M*((n_q_cols+n_k_cols)/2); int g=(pairs+tpb-1)/tpb;
    rope_qk<<<g,tpb,0,st>>>(B.qkv,M,Nqkv,HD,n_q_cols,n_k_cols,theta); }
  { int g=(M*Nqkv+tpb-1)/tpb; split_qkv_to_fa2<<<g,tpb,0,st>>>(B.qkv,B.Q,B.K,B.V,M,Nqkv,NH,NKV,HD); }
  launch_flash_prefill_v1(B.Q,B.K,B.V,B.Oattn,M,NH,NKV,HD,st);
  { int g=(M*NE+tpb-1)/tpb; fa2_to_tokenmajor<<<g,tpb,0,st>>>(B.Oattn,B.attn_tm,M,NH,HD); }
  quant_only<<<M,256,0,st>>>(B.attn_tm,B.aq_o,B.as_o,M,NE);
  gemm_res(B.aq_o,B.as_o,B.wo[rb],B.so[rb],B.x,M,NE,NE,st);
  rmsnorm_quant_p8<<<M,256,0,st>>>(B.x,B.wn_ffn,B.aq_ffn,B.as_ffn,M,NE,eps);
  gemm_plain(B.aq_ffn,B.as_ffn,B.wgate[rb],B.sgate[rb],B.gate,M,NFF,NE,st);
  gemm_plain(B.aq_ffn,B.as_ffn,B.wup[rb],B.sup[rb],B.up,M,NFF,NE,st);
  silumul_quant<<<M,256,0,st>>>(B.gate,B.up,B.aq_dn,B.as_dn,M,NFF);
  gemm_res(B.aq_dn,B.as_dn,B.wdown[rb],B.sdown[rb],B.x,M,NE,NFF,st);
}

static double clean_wall(Buffers&B,int M,int NE,int NH,int NKV,int HD,int NFF,float eps,float theta,
                         hipStream_t st,int NL){
  for(int w=0;w<3;w++){ for(int l=0;l<NL;l++) run_layer_clean(B,M,NE,NH,NKV,HD,NFF,eps,theta,l%ROT,st); }
  CK(hipStreamSynchronize(st));
  int R=(M>=8192)?5:8; double best=1e30;
  for(int r=0;r<R;r++){ double t0=now();
    for(int l=0;l<NL;l++) run_layer_clean(B,M,NE,NH,NKV,HD,NFF,eps,theta,l%ROT,st);
    CK(hipStreamSynchronize(st)); double m=now()-t0; if(m<best)best=m; }
  return best;
}

// ===================================================================================================
//  PER-GEMM bandwidth-vs-compute split. For a given weight shape (N,K), measure:
//    HOT  = 8 back-to-back calls on ONE weight buffer (stays L2-warm; weight read mostly from L2)
//    COLD = each call preceded by a 64MB L2 flush (weight forced cold from VRAM), flush time subtracted
//  HOT ~= the WMMA-compute floor (codegen ceiling; the part a native-4-bit read CANNOT reduce).
//  (COLD - HOT) = the weight-bandwidth tax that a native 4-bit weight read would roughly HALVE for
//                 Q4_K_M / Q4_0 (since the int8 weight bytes/row would become 4-bit nibbles).
//  Returns ms for HOT (min-of-bursts of the back-to-back) and COLD (min-of-bursts, flush subtracted).
// ===================================================================================================
static void gemm_bw_split(int M,int N,int K,hipStream_t st,double& hot_ms,double& cold_ms){
  int8_t* wbuf; float* wsc; int8_t* abuf; float* asc; __half* obuf;
  size_t Wn=(size_t)N*K, Wsn=(size_t)N*(K/32)*4, An=(size_t)M*K, Asn=(size_t)M*(K/32)*4, On=(size_t)M*N*2;
  CK(hipMalloc(&wbuf,Wn)); CK(hipMalloc(&wsc,Wsn)); CK(hipMalloc(&abuf,An)); CK(hipMalloc(&asc,Asn)); CK(hipMalloc(&obuf,On));
  CK(hipMemset(wbuf,1,Wn)); CK(hipMemset(wsc,0,Wsn)); CK(hipMemset(abuf,1,An)); CK(hipMemset(asc,0,Asn));
  size_t FL=64u*1024*1024; int8_t* dF; CK(hipMalloc(&dF,FL)); CK(hipMemset(dF,7,FL));
  dim3 g(N/256,M/256),b(8*4*32);
  // warmup
  for(int w=0;w<8;w++) gemm_v3p<256,256,8,4,32,1><<<g,b,0,st>>>(abuf,asc,wbuf,wsc,obuf,M,N,K);
  CK(hipStreamSynchronize(st));
  const int BURST=8; int R=(M>=8192)?6:12;
  // HOT: BURST back-to-back, weight stays L2-warm after first call
  hot_ms=1e30;
  for(int r=0;r<R;r++){ double t0=now();
    for(int i=0;i<BURST;i++) gemm_v3p<256,256,8,4,32,1><<<g,b,0,st>>>(abuf,asc,wbuf,wsc,obuf,M,N,K);
    CK(hipStreamSynchronize(st)); double m=(now()-t0)/BURST; if(m<hot_ms)hot_ms=m; }
  // COLD: flush before each GEMM; measure (flush+gemm), then subtract flush-only.
  double withflush=1e30;
  for(int r=0;r<R;r++){ double t0=now();
    for(int i=0;i<BURST;i++){ l2_flush<<<(FL+255)/256,256,0,st>>>(dF,FL);
      gemm_v3p<256,256,8,4,32,1><<<g,b,0,st>>>(abuf,asc,wbuf,wsc,obuf,M,N,K); }
    CK(hipStreamSynchronize(st)); double m=(now()-t0)/BURST; if(m<withflush)withflush=m; }
  double flushonly=1e30;
  for(int r=0;r<R;r++){ double t0=now();
    for(int i=0;i<BURST;i++) l2_flush<<<(FL+255)/256,256,0,st>>>(dF,FL);
    CK(hipStreamSynchronize(st)); double m=(now()-t0)/BURST; if(m<flushonly)flushonly=m; }
  cold_ms = withflush - flushonly;
  hipFree(wbuf);hipFree(wsc);hipFree(abuf);hipFree(asc);hipFree(obuf);hipFree(dF);
}

// ===================================================================================================
int main(){
  const int NE=4096, NH=32, NKV=8, HD=128, NFF=14336, NL=50;
  const float eps=1e-5f, theta=1e6f;
  hipStream_t st; CK(hipStreamCreate(&st));
  double radv[2]={2373,1989}; int ctxs[2]={512,8192};

  printf("=========================================================================================\n");
  printf(" full_forward_prof — PER-OP instrumented Bielik-11B prefill forward (gfx1201/RDNA4/HIP)\n");
  printf(" dims NE=%d NH=%d NKV=%d HD=%d NFF=%d NL=%d  ROT=%d (realistic-cold weight rotation)\n",NE,NH,NKV,HD,NFF,NL,ROT);
  printf(" GEMM=gemm_v3p<256,256,8,4,32,swz>  FA2=flash_prefill_v1.  int8 GEMM => breakdown SAME for\n");
  printf(" Q8/Q4_K_M/Q4_0 except weight bandwidth (see per-GEMM bw/compute split at end).\n");
  printf(" FRESH RADV baseline: pp512=%.0f  pp8192=%.0f t/s\n",radv[0],radv[1]);
  printf("=========================================================================================\n");

  for(int ci=0;ci<2;ci++){
    int M=ctxs[ci];
    Buffers B; alloc_buffers(B,M,NE,NH,NKV,HD,NFF);

    // ---- clean wall (no per-op events) ----
    double wall = clean_wall(B,M,NE,NH,NKV,HD,NFF,eps,theta,st,NL);

    // ---- per-op instrumented: min-of-bursts of the summed-per-op accumulators ----
    OpTimer T; double bestsum=1e30; double bestacc[NOP];
    for(int i=0;i<NOP;i++) bestacc[i]=0;
    // warmup
    for(int w=0;w<2;w++){ for(int l=0;l<NL;l++) run_layer_instrumented(B,M,NE,NH,NKV,HD,NFF,eps,theta,l%ROT,st,T); T.reset(); }
    int R=(M>=8192)?4:6;
    for(int r=0;r<R;r++){ T.reset();
      for(int l=0;l<NL;l++) run_layer_instrumented(B,M,NE,NH,NKV,HD,NFF,eps,theta,l%ROT,st,T);
      double s=0; for(int i=0;i<NOP;i++) s+=T.acc[i];
      if(s<bestsum){ bestsum=s; for(int i=0;i<NOP;i++) bestacc[i]=T.acc[i]; }
    }

    double tc=M/(wall/1000.0);
    const char* verdict = tc>radv[ci]*1.005 ? "BEAT" : (tc<radv[ci]*0.995 ? "LOSE" : "TIE");
    printf("\n#########################################################################################\n");
    printf("### M=%d   clean-wall %.3f ms  = %.0f t/s   (RADV %.0f, %+.1f%%)  ==> %s\n",
           M,wall,tc,radv[ci],(tc/radv[ci]-1)*100,verdict);
    printf("###        per-op instrumented sum %.3f ms   (events add %.3f ms = %+.1f%% perturbation)\n",
           bestsum,bestsum-wall,(bestsum/wall-1)*100);
    printf("#########################################################################################\n");
    printf("  %-40s  %10s  %8s  %12s\n","OP CATEGORY (summed over 50 layers)","ms/fwd","%% of sum","us/layer");
    printf("  %-40s  %10s  %8s  %12s\n","-----------------------------------","------","-------","--------");
    for(int i=0;i<NOP;i++){
      if(i==OP_QUANT_QKV||i==OP_QUANT_FFN){ // folded into P8 fused rmsnorm
        printf("  %-40s  %10s  %8s  %12s\n",OPNAME[i],"(fused)","-","-"); continue; }
      double pct = bestacc[i]/bestsum*100.0;
      printf("  %-40s  %10.4f  %7.2f%%  %12.4f\n",OPNAME[i],bestacc[i],pct,bestacc[i]/NL*1000.0);
    }
    printf("  %-40s  %10.4f  %7.2f%%\n","SUM of instrumented ops",bestsum,100.0);
    printf("  %-40s  %10.4f  %7.2f%%\n","CLEAN WALL (no events)",wall,wall/bestsum*100.0);
    printf("  %-40s  %10.4f\n","dispatch/sched overhead (wall - hot-sum est.)",wall-bestsum);
    // GEMM-only aggregate
    double gemmsum=bestacc[OP_QKV]+bestacc[OP_OPROJ]+bestacc[OP_GATE]+bestacc[OP_UP]+bestacc[OP_DOWN];
    double memsum=bestsum-gemmsum-bestacc[OP_FA2];
    printf("  ----\n");
    printf("  %-40s  %10.4f  %7.2f%%   (qkv+o+gate+up+down)\n","5 weight GEMMs total",gemmsum,gemmsum/bestsum*100.0);
    printf("  %-40s  %10.4f  %7.2f%%\n","FA2 attention",bestacc[OP_FA2],bestacc[OP_FA2]/bestsum*100.0);
    printf("  %-40s  %10.4f  %7.2f%%   (rms+quant+rope+repack+silu)\n","non-GEMM mem ops total",memsum,memsum/bestsum*100.0);

    free_buffers(B);
  }

  // ===================================================================================================
  //  PER-GEMM BANDWIDTH vs COMPUTE split (HOT L2-warm vs COLD 64MB-flush), for the weight GEMMs.
  //  This isolates how much of each GEMM is weight-bandwidth (which a native 4-bit weight read halves
  //  for Q4_K_M/Q4_0) vs WMMA-compute (the fixed codegen ceiling, identical for every source quant).
  // ===================================================================================================
  printf("\n=========================================================================================\n");
  printf(" PER-GEMM BANDWIDTH vs COMPUTE  (HOT=L2-warm 8-back-to-back, COLD=64MB-flush-each, subtracted)\n");
  printf("   HOT  ~= WMMA-compute floor (codegen ceiling; native-4-bit CANNOT reduce this)\n");
  printf("   COLD-HOT = weight-bandwidth tax (native-4-bit read ~HALVES it for Q4_K_M/Q4_0)\n");
  printf("   int8 weight bytes/row = K; a 4-bit weight read would move ~K/2 bytes/row.\n");
  printf("=========================================================================================\n");
  struct GShape{const char* nm;int N,K;};
  GShape gs[3]={{"qkv  N=6144  K=4096",6144,4096},{"gate N=14336 K=4096",14336,4096},{"down N=4096  K=14336",4096,14336}};
  int Ms[2]={512,8192};
  for(int mi=0;mi<2;mi++){ int M=Ms[mi];
    printf("\n--- M=%d ---\n",M);
    printf("  %-22s  %10s  %10s  %12s  %12s  %10s\n","GEMM (weight)","HOT ms","COLD ms","bw tax ms","bw %% of COLD","WMMA TF(hot)");
    for(int s=0;s<3;s++){ double hot,cold;
      gemm_bw_split(M,gs[s].N,gs[s].K,st,hot,cold);
      double tax=cold-hot; double bwpct=tax/cold*100.0;
      double tf=2.0*M*gs[s].N*gs[s].K/1e12/(hot/1e3);
      printf("  %-22s  %10.4f  %10.4f  %12.4f  %11.2f%%  %10.1f\n",gs[s].nm,hot,cold,tax,bwpct,tf);
    }
  }
  printf("\n NOTE: per-fwd, qkv+gate+up GEMMs read K=4096 int8 weight-bytes/row; down reads K=14336.\n");
  printf("   For Q4_K_M/Q4_0 a native 4-bit weight load => ~half the weight bytes => the 'bw tax ms'\n");
  printf("   column is the upper bound on what a native-4-bit GEMM would save per GEMM (compute=HOT fixed).\n");

  hipStreamDestroy(st);
  return 0;
}
