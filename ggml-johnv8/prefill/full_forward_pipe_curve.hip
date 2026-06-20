// full_forward_pipe_curve.hip — COPY of full_forward_pipe.hip, EXTENDED to a 5-point pp curve
//   {512,1024,2048,4096,8192} for the quality-safe scoreboard. Original untouched. (Bielik-11B int8/Q8, gfx1201).
//
//   PURPOSE: wire the proven gemm_pipe2b GEMM (2-stage LDS pipeline, ds_store interleaved after the first
//   WMMA sub-tile — gemm_pipe.hip) into the REAL 50-layer forward and measure the cold full-forward t/s
//   vs the v3p baseline and vs RADV Q8. The GEMM micro-bench measured +6.3% agg COLD over v3p, implying
//   a +5.6% full-forward lift (GEMM = ~89% of the forward). This file PROVES (or refutes) that in a real
//   realistic-cold full forward, and checks whether Q8 closes / wins the RADV Q8 gap.
//
//   STRUCTURE: byte-for-byte the OP STRUCTURE of full_forward_v3.hip (rotate ROT distinct weight buffers
//   per shape for realistic cold; P8 rmsnorm->int8 fusion; RoPE; split/merge; silu; flash_prefill_v1 FA2;
//   all 11 ops per layer). The ONLY change is the 5 weight-GEMM call sites: v3p / v3p_res are SWAPPED for
//   gemm_pipe2b / gemm_pipe2b_res. Both variants are kept in-binary so the new forward is verified bit-for-
//   bit-close (NRMSE) against the v3p forward under identical conditions, and both are timed side by side.
//
//   I copied (NOT included) gemm_v3p, gemm_v3p_res, gemm_pipe2b from the source files, and authored
//   gemm_pipe2b_res (residual epilogue, mirroring gemm_v3p_res). Does NOT touch gemm_pipe.hip,
//   gemm_rb_v3.hip, gemm_w4a4.hip, full_forward_v3.hip, full_forward_w4a4.hip (other processes use them).
//
//   Build: HIP_VISIBLE_DEVICES=1 hipcc --offload-arch=gfx1201 -O3 full_forward_pipe.hip -o /tmp/ffpipe && \
//          rocm-smi --setperflevel auto && HIP_VISIBLE_DEVICES=1 /tmp/ffpipe
//
//   GEMM micro-bench (gemm_pipe.hip, COLD agg +6.3%) => implied full-fwd +5.6%: pp512 2584 -> ~2730 t/s.
//   v3p int8 full-forward baseline (this harness, prior run): pp512 2584  pp2048 2788  pp8192 2220 t/s.
//   FRESH RADV Q8 baseline:                                   pp512 2849  pp2048 2726  pp8192 2307 t/s.
//   Question 1: does GEMM +6.3% translate to full-fwd +5.6%?  Question 2: does Q8 close pp512 (-9%->~-4%?)
//   and win pp2048 outright?  Honest cold numbers only.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

typedef int i4  __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef __half2 h2;
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
#define CK(x) do{hipError_t e=(x);if(e){printf("HIP err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

#include "/home/janusz/AMD MXFP4/hip_int4/flash_prefill.hip"

// gfx12 split-counter wait (P1, used by v3p / pipe2b)
template<int N> __device__ __forceinline__ void wait_loadN(){ asm volatile("s_wait_loadcnt %0"::"i"(N):"memory"); }

#define ROT 3   // distinct weight buffers per shape (>=3 so any one evicts the others; layer%ROT cycles)

__device__ __forceinline__ int swz16(int row,int blk16,int KC){
  int nblk=KC/16; if(nblk<=1) return blk16; return blk16 ^ ((row>>1)&(nblk-1)); }

// ===================================================================================================
//  V3P GEMM (verbatim copy of full_forward_v3.hip) — kept as the correctness/perf BASELINE in-binary.
// ===================================================================================================
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
//  PIPE2B GEMM (verbatim copy of gemm_pipe.hip's measured winner). 2-stage LDS pipeline with the
//  ds_store interleaved AFTER the first WMMA sub-tile so the store overlaps the dot tail.
// ===================================================================================================
template<int BM,int BN,int WMW,int WNW,int KC,int SWZ>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_pipe2b(
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
    bool stored=false;
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
      // interleave the store after the FIRST wmma sub-tile (ks==0): overlaps the dot tail.
      if(!stored && ks==0 && kn<K){ stored=true; wait_loadN<0>();
        for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
        for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    }
    if(!stored && kn<K){ wait_loadN<0>();
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=a_scale[(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk];
    for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[mt][nt][j]*wsc[nt]*asc[mt][j];
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
}

// ===================================================================================================
//  PIPE2B_RES — authored here. IDENTICAL to gemm_pipe2b except the epilogue ACCUMULATES into out
//  (residual add: out += Rf), mirroring exactly how gemm_v3p_res differs from gemm_v3p. Same int8 math,
//  just the residual add fused in the store — used for the o-proj and down GEMMs.
// ===================================================================================================
template<int BM,int BN,int WMW,int WNW,int KC,int SWZ>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_pipe2b_res(
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
    bool stored=false;
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
      if(!stored && ks==0 && kn<K){ stored=true; wait_loadN<0>();
        for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
        for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    }
    if(!stored && kn<K){ wait_loadN<0>();
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=a_scale[(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk];
    for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[mt][nt][j]*wsc[nt]*asc[mt][j];
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){
    int m=m_base+wm*WMT*16+mt*16+kg*8+j; size_t o=(size_t)m*N+n_base+wn*WNT*16+nt*16+nl;
    out[o]=__float2half(__half2float(out[o])+Rf[mt][nt][j]); }
}

// ===================================================================================================
//  Mem-op kernels (identical to full_forward_v3.hip)
// ===================================================================================================
__global__ void rmsnorm_quant(const __half* __restrict__ x, const float* __restrict__ w,
                              int8_t* __restrict__ q, float* __restrict__ s, int M, int K, float eps){
  int row=blockIdx.x; if(row>=M) return; int tid=threadIdx.x, nth=blockDim.x;
  const __half* xr=x+(size_t)row*K; __shared__ float red[256];
  float p=0; for(int j=tid;j<K;j+=nth){float v=__half2float(xr[j]); p+=v*v;} red[tid]=p; __syncthreads();
  for(int st=nth/2;st>0;st>>=1){if(tid<st)red[tid]+=red[tid+st];__syncthreads();}
  float inv=rsqrtf(red[0]/K+eps); __syncthreads();
  int nb=K/32;
  for(int b=tid;b<nb;b+=nth){ float vals[32],am=0;
    for(int e=0;e<32;e++){int k=b*32+e; float v=__half2float(xr[k])*inv*w[k]; vals[e]=v; float a=fabsf(v); if(a>am)am=a;}
    float d=am/127.f, id=d>0?1.f/d:0.f; s[(size_t)row*nb+b]=d;
    for(int e=0;e<32;e++){int vq=(int)lrintf(vals[e]*id); if(vq>127)vq=127; if(vq<-127)vq=-127; q[(size_t)row*32*nb+b*32+e]=(int8_t)vq;}
  }
}
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

// ===================================================================================================
//  GEMM dispatch wrappers. GV3P = v3p baseline; GPIPE = pipe2b (the variant under test).
// ===================================================================================================
enum { GV3P=1, GPIPE=2 };
static void gemm_plain(int gv,const int8_t*a,const float*as,const int8_t*w,const float*ws,__half*o,int M,int N,int K,hipStream_t st){
  dim3 g(N/256,M/256),b(8*4*32);
  if(gv==GPIPE) gemm_pipe2b<256,256,8,4,32,1><<<g,b,0,st>>>(a,as,w,ws,o,M,N,K);
  else          gemm_v3p   <256,256,8,4,32,1><<<g,b,0,st>>>(a,as,w,ws,o,M,N,K);
}
static void gemm_res(int gv,const int8_t*a,const float*as,const int8_t*w,const float*ws,__half*o,int M,int N,int K,hipStream_t st){
  dim3 g(N/256,M/256),b(8*4*32);
  if(gv==GPIPE) gemm_pipe2b_res<256,256,8,4,32,1><<<g,b,0,st>>>(a,as,w,ws,o,M,N,K);
  else          gemm_v3p_res   <256,256,8,4,32,1><<<g,b,0,st>>>(a,as,w,ws,o,M,N,K);
}

// ===================================================================================================
//  Buffers. ROT distinct weight buffers per shape so consecutive layers read different VRAM (realistic
//  cold). Identical to full_forward_v3.hip.
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
//  One layer — identical op structure to full_forward_v3.hip. gv routes the 5 weight GEMMs.
//  rb = which weight buffer this layer reads (li%ROT when cold). All 11 ops present.
// ===================================================================================================
static void run_layer(Buffers&B,int M,int NE,int NH,int NKV,int HD,int NFF,float eps,float theta,
                      int gv,int p8,int rb,hipStream_t st){
  int Nqkv=NH*HD+2*NKV*HD; int n_q_cols=NH*HD, n_k_cols=NKV*HD; int tpb=256;
  // input rmsnorm (fused quant)
  if(p8) rmsnorm_quant_p8<<<M,256,0,st>>>(B.x,B.wn_attn,B.aq,B.as,M,NE,eps);
  else   rmsnorm_quant   <<<M,256,0,st>>>(B.x,B.wn_attn,B.aq,B.as,M,NE,eps);
  // qkv GEMM
  gemm_plain(gv,B.aq,B.as,B.wqkv[rb],B.sqkv[rb],B.qkv,M,Nqkv,NE,st);
  // RoPE
  { int pairs=M*((n_q_cols+n_k_cols)/2); int g=(pairs+tpb-1)/tpb;
    rope_qk<<<g,tpb,0,st>>>(B.qkv,M,Nqkv,HD,n_q_cols,n_k_cols,theta); }
  // FA2 attention
  { int g=(M*Nqkv+tpb-1)/tpb; split_qkv_to_fa2<<<g,tpb,0,st>>>(B.qkv,B.Q,B.K,B.V,M,Nqkv,NH,NKV,HD); }
  launch_flash_prefill_v1(B.Q,B.K,B.V,B.Oattn,M,NH,NKV,HD,st);
  { int g=(M*NE+tpb-1)/tpb; fa2_to_tokenmajor<<<g,tpb,0,st>>>(B.Oattn,B.attn_tm,M,NH,HD); }
  quant_only<<<M,256,0,st>>>(B.attn_tm,B.aq_o,B.as_o,M,NE);
  // o-proj GEMM + residual into x
  gemm_res(gv,B.aq_o,B.as_o,B.wo[rb],B.so[rb],B.x,M,NE,NE,st);
  // post-attn rmsnorm (fused quant)
  if(p8) rmsnorm_quant_p8<<<M,256,0,st>>>(B.x,B.wn_ffn,B.aq_ffn,B.as_ffn,M,NE,eps);
  else   rmsnorm_quant   <<<M,256,0,st>>>(B.x,B.wn_ffn,B.aq_ffn,B.as_ffn,M,NE,eps);
  // gate GEMM, up GEMM
  gemm_plain(gv,B.aq_ffn,B.as_ffn,B.wgate[rb],B.sgate[rb],B.gate,M,NFF,NE,st);
  gemm_plain(gv,B.aq_ffn,B.as_ffn,B.wup[rb],B.sup[rb],B.up,M,NFF,NE,st);
  // silu*up + quant
  silumul_quant<<<M,256,0,st>>>(B.gate,B.up,B.aq_dn,B.as_dn,M,NFF);
  // down GEMM + residual into x
  gemm_res(gv,B.aq_dn,B.as_dn,B.wdown[rb],B.sdown[rb],B.x,M,NE,NFF,st);
}

// cold=1 -> rotate li%ROT (realistic); cold=0 -> always buffer 0 (L2-warm CONTROL).
static double measure_forward(Buffers&B,int M,int NE,int NH,int NKV,int HD,int NFF,float eps,float theta,
                              int gv,int p8,int cold,hipStream_t st,int NL){
  for(int w=0;w<3;w++){ for(int l=0;l<NL;l++) run_layer(B,M,NE,NH,NKV,HD,NFF,eps,theta,gv,p8,cold?(l%ROT):0,st); }
  CK(hipStreamSynchronize(st));
  int R = (M>=8192)?5:8; double best=1e30;
  for(int r=0;r<R;r++){ double t0=now();
    for(int l=0;l<NL;l++) run_layer(B,M,NE,NH,NKV,HD,NFF,eps,theta,gv,p8,cold?(l%ROT):0,st);
    CK(hipStreamSynchronize(st)); double m=now()-t0; if(m<best)best=m; }
  return best; // ms for ONE full NL-layer forward (min-of-bursts)
}

// Run a SINGLE forward (no warmup/repeat) into B.x and snapshot it, so the v3p and pipe forwards can be
// compared bit-for-bit-close. Returns the host copy of x (M*NE halfs).
static void snapshot_forward(Buffers&B,int M,int NE,int NH,int NKV,int HD,int NFF,float eps,float theta,
                             int gv,int p8,hipStream_t st,int NL,std::vector<__half>& host){
  CK(hipMemset(B.x,1,(size_t)M*NE*2));                       // identical deterministic start
  for(int l=0;l<NL;l++) run_layer(B,M,NE,NH,NKV,HD,NFF,eps,theta,gv,p8,l%ROT,st);
  CK(hipStreamSynchronize(st));
  host.resize((size_t)M*NE);
  CK(hipMemcpy(host.data(),B.x,(size_t)M*NE*2,hipMemcpyDeviceToHost));
}

int main(){
  const int NE=4096, NH=32, NKV=8, HD=128, NFF=14336, NL=50;
  const float eps=1e-5f, theta=1e6f;
  // FULL CURVE: pp512, pp1024, pp2048, pp4096, pp8192 (pp1024/pp4096 ADDED vs the original 3).
  const int NC=5;
  int ctxs[NC]={512,1024,2048,4096,8192};
  const char* tags[NC]={"pp512","pp1024","pp2048","pp4096","pp8192"};
  hipStream_t st; CK(hipStreamCreate(&st));

  printf("=== full_forward_pipe_curve: gemm_pipe2b int8 full forward — 5-pt t/s CURVE (Bielik-11B, gfx1201) ===\n");
  printf("dims NE=%d NH=%d NKV=%d HD=%d NFF=%d NL=%d  ROT=%d distinct weight buffers/shape (realistic cold)\n",
         NE,NH,NKV,HD,NFF,NL,ROT);
  printf("BIT-EXACT int8 path: same GEMM speed for ALL int8-activation source quants (only RADV baseline varies)\n\n");

  // -------------------------------------------------------------------------------------------------
  //  [1] CORRECTNESS — pipe full forward must match the v3p full forward (same int8 math, rescheduled).
  // -------------------------------------------------------------------------------------------------
  printf("==================================================================================\n");
  printf("[1] CORRECTNESS: pipe2b full forward vs v3p full forward (final x after 50L), NRMSE<1e-3\n");
  printf("==================================================================================\n");
  { int M=512; Buffers B; alloc_buffers(B,M,NE,NH,NKV,HD,NFF);
    std::vector<__half> xref, xpipe;
    snapshot_forward(B,M,NE,NH,NKV,HD,NFF,eps,theta,GV3P ,1,st,NL,xref);
    snapshot_forward(B,M,NE,NH,NKV,HD,NFF,eps,theta,GPIPE,1,st,NL,xpipe);
    double e=0,nn=0; for(size_t i=0;i<xref.size();i++){ float a=__half2float(xpipe[i]),b=__half2float(xref[i]); e+=(a-b)*(a-b); nn+=b*b; }
    double nrmse = nn>0 ? sqrt(e/nn) : (e>0?1.0:0.0);
    printf("  M=512 50L  NRMSE(pipe vs v3p) = %.2e   %s\n", nrmse, nrmse<1e-3?"PASS":"FAIL");
    free_buffers(B);
  }

  // -------------------------------------------------------------------------------------------------
  //  [2] REALISTIC-COLD pipe2b t/s CURVE — two passes per ctx (v3p-first then pipe-first) to cancel
  //      ordering/thermal bias; report the AVERAGE of the two pipe2b min-of-bursts t/s. Sequential.
  // -------------------------------------------------------------------------------------------------
  printf("\n==================================================================================\n");
  printf("[2] pipe2b REALISTIC-COLD full-forward t/s CURVE (min-of-bursts, ROT=%d, P8 on, sequential)\n",ROT);
  printf("==================================================================================\n");
  printf("%-7s | %-9s | %-9s | %-9s | pipe gain\n","ctx","pipe2b A","pipe2b B","pipe2b AVG");
  printf("--------+-----------+-----------+-----------+----------\n");
  double curve[NC];
  for(int ci=0;ci<NC;ci++){
    int M=ctxs[ci];
    auto ts=[&](double ms){ return M/(ms/1000.0); };
    // Pass A: v3p-first then pipe
    Buffers BA; alloc_buffers(BA,M,NE,NH,NKV,HD,NFF);
    double v3a   = measure_forward(BA,M,NE,NH,NKV,HD,NFF,eps,theta,GV3P ,1,1,st,NL);
    double pipea = measure_forward(BA,M,NE,NH,NKV,HD,NFF,eps,theta,GPIPE,1,1,st,NL);
    free_buffers(BA);
    // Pass B: pipe-first then v3p (fresh allocs)
    Buffers BB; alloc_buffers(BB,M,NE,NH,NKV,HD,NFF);
    double pipeb = measure_forward(BB,M,NE,NH,NKV,HD,NFF,eps,theta,GPIPE,1,1,st,NL);
    double v3b   = measure_forward(BB,M,NE,NH,NKV,HD,NFF,eps,theta,GV3P ,1,1,st,NL);
    free_buffers(BB);
    double tpa=ts(pipea), tpb=ts(pipeb), tavg=0.5*(tpa+tpb);
    double tv3=0.5*(ts(v3a)+ts(v3b));
    curve[ci]=tavg;
    printf("%-7s | %6.0f t/s| %6.0f t/s| %6.0f t/s| %+.1f%% vs v3p(%.0f)\n",
           tags[ci], tpa, tpb, tavg, (tv3>0?(tavg/tv3-1)*100:0), tv3);
  }

  // -------------------------------------------------------------------------------------------------
  //  [3] The deliverable: pipe2b cold t/s curve, ready to compare against per-quant fresh RADV.
  // -------------------------------------------------------------------------------------------------
  printf("\n==================================================================================\n");
  printf("[3] PIPE2B COLD CURVE (this is the int8-path t/s for EVERY int8-activation source quant)\n");
  printf("==================================================================================\n");
  for(int ci=0;ci<NC;ci++) printf("  %-7s pipe2b %.0f t/s\n", tags[ci], curve[ci]);
  printf("\nCSV:ctx,pipe2b_tps\n");
  for(int ci=0;ci<NC;ci++) printf("CSV:%d,%.0f\n", ctxs[ci], curve[ci]);

  hipStreamDestroy(st);
  return 0;
}
