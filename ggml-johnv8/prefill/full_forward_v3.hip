// full_forward_v3.hip — REALISTIC-COLD full forward (Bielik-11B Q6_K, gfx1201/RDNA4/HIP, GPU0).
//
//   THE FIX vs full_forward_v2.hip:
//   v2 allocated ONE weight buffer per shape (B.wqkv/wo/wgate/wup/wdown) and REUSED it across all 50
//   layers. Every weight (16..58 MB int8) far exceeds the 8 MB R9700 L2, so it only PARTIALLY evicts —
//   the reused buffer stayed PARTIALLY L2-warm between consecutive layers. That is OPTIMISTIC, worst at
//   small M where PART-1 already measured the L2-cold tax at +25% (M=512). A real forward reads 50
//   DIFFERENT weights per shape, each fully cold. So v2's small-M full-forward t/s OVERSTATED.
//
//   THIS FILE rotates K=ROT (>=3) DISTINCT weight buffers per shape, cycling buffer = layer % ROT, so
//   consecutive layers read a DIFFERENT VRAM region — never the just-cached one. With ROT>=3 and each
//   weight > 8 MB (any one evicts the others from L2), the GEMM that reads weight[L%ROT] finds it cold,
//   exactly as in the real 50-distinct-weight forward. No per-GEMM flush kernel, so wall-clock = pure
//   realistic-cold forward (nothing to subtract).
//   VRAM budget: 5 shapes x ROT x up-to-58 MB. ROT=3 => ~0.9 GB. fine on 32 GB.
//
//   Everything else is IDENTICAL to v2: v3p GEMM (fragment-pipeline+swizzle) + P8 (rmsnorm->int8 fusion),
//   the FA2 (flash_prefill_v1), rope_qk, split/merge, silu, the full 50-layer structure with ALL ops
//   present per layer: input-rmsnorm, qkv GEMM, RoPE, FA2, o-proj GEMM, residual, post-attn-rmsnorm,
//   gate GEMM, up GEMM, silu, down GEMM, residual.
//
//   Also includes a CONTROL: it re-runs the v2-style SINGLE-buffer (L2-warm) forward so the gap between
//   "warm artifact" and "realistic cold" is printed side-by-side, and a sanity (B) flush-per-GEMM path.
//
//   Build: HIP_VISIBLE_DEVICES=0 hipcc --offload-arch=gfx1201 -O3 full_forward_v3.hip -o /tmp/ff3 && \
//          rocm-smi --setperflevel auto && HIP_VISIBLE_DEVICES=0 /tmp/ff3
//
//   FRESH RADV baseline (-fa auto), measured 2026-06-20:  pp512=2373  pp2048=2289  pp8192=1989 t/s
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

// gfx12 split-counter wait (P1, used by v3p)
template<int N> __device__ __forceinline__ void wait_loadN(){ asm volatile("s_wait_loadcnt %0"::"i"(N):"memory"); }

#define ROT 3   // distinct weight buffers per shape (>=3 so any one evicts the others; layer%ROT cycles)

// ===================================================================================================
//  GEMMs (identical to full_forward_v2.hip)
// ===================================================================================================
template<int BM,int BN,int WMW,int WNW,int KC>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_v0(
    const int8_t* __restrict__ act, const float* __restrict__ a_scale,
    const int8_t* __restrict__ wt,  const float* __restrict__ w_scale,
    __half* __restrict__ out, int M, int N, int K){
  constexpr int NWARP=WMW*WNW, NTHREAD=NWARP*32, WMT=BM/(WMW*16), WNT=BN/(WNW*16);
  constexpr int AV=BM*KC/16, BV=BN*KC/16, APT=(AV+NTHREAD-1)/NTHREAD, BPT=(BV+NTHREAD-1)/NTHREAD;
  __shared__ int8_t As[2][BM][KC]; __shared__ int8_t Bs[2][BN][KC];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16, wm=warp/WNW, wn=warp%WNW;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/KC;
  float Rf[WMT][WNT][8];
  for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)for(int j=0;j<8;j++)Rf[a][b][j]=0.f;
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+c];}}
  for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[0][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+c];}}
  __syncthreads(); int buf=0;
  for(int k0=0;k0<K;k0+=KC){ int blk=k0/KC; i4 rA[APT],rB[BPT]; int kn=k0+KC;
    if(kn<K){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kn+c];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rB[p]=*(const i4*)&wt[(size_t)(n_base+row)*K+kn+c];}} }
    v8i cR[WMT][WNT]; for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)cR[a][b]=(v8i){};
    for(int ks=0;ks<KC/16;ks++){ v2i Af[WMT],Bf[WNT];
      for(int mt=0;mt<WMT;mt++){const int8_t* ap=&As[buf][wm*WMT*16+mt*16+nl][ks*16+kg*8];Af[mt]=(v2i){*(const int*)ap,*(const int*)(ap+4)};}
      for(int nt=0;nt<WNT;nt++){const int8_t* bp=&Bs[buf][wn*WNT*16+nt*16+nl][ks*16+kg*8];Bf[nt]=(v2i){*(const int*)bp,*(const int*)(bp+4)};}
      for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)cR[mt][nt]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,Af[mt],true,Bf[nt],cR[mt][nt],false);}
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=a_scale[(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk];
    for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[mt][nt][j]*wsc[nt]*asc[mt][j];
    if(kn<K){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
}

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

template<int BM,int BN,int WMW,int WNW,int KC>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_v0_res(
    const int8_t* __restrict__ act, const float* __restrict__ a_scale,
    const int8_t* __restrict__ wt,  const float* __restrict__ w_scale,
    __half* __restrict__ out, int M, int N, int K){
  constexpr int NWARP=WMW*WNW, NTHREAD=NWARP*32, WMT=BM/(WMW*16), WNT=BN/(WNW*16);
  constexpr int AV=BM*KC/16, BV=BN*KC/16, APT=(AV+NTHREAD-1)/NTHREAD, BPT=(BV+NTHREAD-1)/NTHREAD;
  __shared__ int8_t As[2][BM][KC]; __shared__ int8_t Bs[2][BN][KC];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16, wm=warp/WNW, wn=warp%WNW;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/KC;
  float Rf[WMT][WNT][8];
  for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)for(int j=0;j<8;j++)Rf[a][b][j]=0.f;
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+c];}}
  for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[0][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+c];}}
  __syncthreads(); int buf=0;
  for(int k0=0;k0<K;k0+=KC){ int blk=k0/KC; i4 rA[APT],rB[BPT]; int kn=k0+KC;
    if(kn<K){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kn+c];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rB[p]=*(const i4*)&wt[(size_t)(n_base+row)*K+kn+c];}} }
    v8i cR[WMT][WNT]; for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)cR[a][b]=(v8i){};
    for(int ks=0;ks<KC/16;ks++){ v2i Af[WMT],Bf[WNT];
      for(int mt=0;mt<WMT;mt++){const int8_t* ap=&As[buf][wm*WMT*16+mt*16+nl][ks*16+kg*8];Af[mt]=(v2i){*(const int*)ap,*(const int*)(ap+4)};}
      for(int nt=0;nt<WNT;nt++){const int8_t* bp=&Bs[buf][wn*WNT*16+nt*16+nl][ks*16+kg*8];Bf[nt]=(v2i){*(const int*)bp,*(const int*)(bp+4)};}
      for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)cR[mt][nt]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,Af[mt],true,Bf[nt],cR[mt][nt],false);}
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=a_scale[(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk];
    for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[mt][nt][j]*wsc[nt]*asc[mt][j];
    if(kn<K){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){
    int m=m_base+wm*WMT*16+mt*16+kg*8+j; size_t o=(size_t)m*N+n_base+wn*WNT*16+nt*16+nl;
    out[o]=__float2half(__half2float(out[o])+Rf[mt][nt][j]); }
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
//  Mem-op kernels (identical to v2)
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
//  GEMM dispatch wrappers
// ===================================================================================================
enum { GV0=0, GV3P=1 };
static void gemm_plain(int gv,const int8_t*a,const float*as,const int8_t*w,const float*ws,__half*o,int M,int N,int K,hipStream_t st){
  dim3 g(N/256,M/256),b(8*4*32);
  if(gv==GV3P) gemm_v3p<256,256,8,4,32,1><<<g,b,0,st>>>(a,as,w,ws,o,M,N,K);
  else         gemm_v0 <256,256,8,4,32  ><<<g,b,0,st>>>(a,as,w,ws,o,M,N,K);
}
static void gemm_res(int gv,const int8_t*a,const float*as,const int8_t*w,const float*ws,__half*o,int M,int N,int K,hipStream_t st){
  dim3 g(N/256,M/256),b(8*4*32);
  if(gv==GV3P) gemm_v3p_res<256,256,8,4,32,1><<<g,b,0,st>>>(a,as,w,ws,o,M,N,K);
  else         gemm_v0_res <256,256,8,4,32  ><<<g,b,0,st>>>(a,as,w,ws,o,M,N,K);
}

// ===================================================================================================
//  Buffers. ROT distinct weight buffers per shape so consecutive layers read different VRAM.
//  COLD=1 => rotate (realistic); COLD=0 => always use buffer 0 (v2-style L2-warm CONTROL).
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
//  One layer. gv = GEMM variant, p8 = fused rmsnorm quant. li = layer index; rb = which weight buffer
//  this layer reads (li%ROT when COLD, else 0). flushW != nullptr => flush L2 before each weight GEMM
//  (option B). All 11 ops present: in-rmsnorm, qkv GEMM, RoPE, FA2, o-proj GEMM+residual,
//  post-attn-rmsnorm, gate GEMM, up GEMM, silu, down GEMM+residual.
// ===================================================================================================
__global__ void l2_flush(int8_t* p, size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]^=1; }

static void run_layer(Buffers&B,int M,int NE,int NH,int NKV,int HD,int NFF,float eps,float theta,
                      int gv,int p8,int rb,int8_t* flushW,size_t flushN,hipStream_t st){
  int Nqkv=NH*HD+2*NKV*HD; int n_q_cols=NH*HD, n_k_cols=NKV*HD; int tpb=256;
  auto FL=[&](){ if(flushW) l2_flush<<<(flushN+255)/256,256,0,st>>>(flushW,flushN); };
  // input rmsnorm (fused quant)
  if(p8) rmsnorm_quant_p8<<<M,256,0,st>>>(B.x,B.wn_attn,B.aq,B.as,M,NE,eps);
  else   rmsnorm_quant   <<<M,256,0,st>>>(B.x,B.wn_attn,B.aq,B.as,M,NE,eps);
  // qkv GEMM
  FL(); gemm_plain(gv,B.aq,B.as,B.wqkv[rb],B.sqkv[rb],B.qkv,M,Nqkv,NE,st);
  // RoPE
  { int pairs=M*((n_q_cols+n_k_cols)/2); int g=(pairs+tpb-1)/tpb;
    rope_qk<<<g,tpb,0,st>>>(B.qkv,M,Nqkv,HD,n_q_cols,n_k_cols,theta); }
  // FA2 attention
  { int g=(M*Nqkv+tpb-1)/tpb; split_qkv_to_fa2<<<g,tpb,0,st>>>(B.qkv,B.Q,B.K,B.V,M,Nqkv,NH,NKV,HD); }
  launch_flash_prefill_v1(B.Q,B.K,B.V,B.Oattn,M,NH,NKV,HD,st);
  { int g=(M*NE+tpb-1)/tpb; fa2_to_tokenmajor<<<g,tpb,0,st>>>(B.Oattn,B.attn_tm,M,NH,HD); }
  quant_only<<<M,256,0,st>>>(B.attn_tm,B.aq_o,B.as_o,M,NE);
  // o-proj GEMM + residual into x
  FL(); gemm_res(gv,B.aq_o,B.as_o,B.wo[rb],B.so[rb],B.x,M,NE,NE,st);
  // post-attn rmsnorm (fused quant)
  if(p8) rmsnorm_quant_p8<<<M,256,0,st>>>(B.x,B.wn_ffn,B.aq_ffn,B.as_ffn,M,NE,eps);
  else   rmsnorm_quant   <<<M,256,0,st>>>(B.x,B.wn_ffn,B.aq_ffn,B.as_ffn,M,NE,eps);
  // gate GEMM, up GEMM
  FL(); gemm_plain(gv,B.aq_ffn,B.as_ffn,B.wgate[rb],B.sgate[rb],B.gate,M,NFF,NE,st);
  FL(); gemm_plain(gv,B.aq_ffn,B.as_ffn,B.wup[rb],B.sup[rb],B.up,M,NFF,NE,st);
  // silu*up + quant
  silumul_quant<<<M,256,0,st>>>(B.gate,B.up,B.aq_dn,B.as_dn,M,NFF);
  // down GEMM + residual into x
  FL(); gemm_res(gv,B.aq_dn,B.as_dn,B.wdown[rb],B.sdown[rb],B.x,M,NE,NFF,st);
}

// cold=1 -> rotate li%ROT (realistic); cold=0 -> always buffer 0 (L2-warm CONTROL, == v2).
// flushW!=nullptr -> option-B: flush L2 before each weight GEMM. We then time a flush-only pass to subtract.
static double measure_forward(Buffers&B,int M,int NE,int NH,int NKV,int HD,int NFF,float eps,float theta,
                              int gv,int p8,int cold,int8_t* flushW,size_t flushN,hipStream_t st,int NL){
  for(int w=0;w<3;w++){ for(int l=0;l<NL;l++) run_layer(B,M,NE,NH,NKV,HD,NFF,eps,theta,gv,p8,cold?(l%ROT):0,flushW,flushN,st); }
  CK(hipStreamSynchronize(st));
  int R = (M>=8192)?5:8; double best=1e30;
  for(int r=0;r<R;r++){ double t0=now();
    for(int l=0;l<NL;l++) run_layer(B,M,NE,NH,NKV,HD,NFF,eps,theta,gv,p8,cold?(l%ROT):0,flushW,flushN,st);
    CK(hipStreamSynchronize(st)); double m=now()-t0; if(m<best)best=m; }
  return best; // ms for ONE full NL-layer forward (min-of-bursts)
}

// number of weight GEMMs per layer that get a flush (qkv, o, gate, up, down) = 5
#define FLUSH_GEMMS_PER_LAYER 5

int main(){
  const int NE=4096, NH=32, NKV=8, HD=128, NFF=14336, NL=50;
  const float eps=1e-5f, theta=1e6f;
  int ctxs[]={512,2048,8192};
  hipStream_t st; CK(hipStreamCreate(&st));

  printf("=== full_forward_v3: REALISTIC-COLD full forward (Bielik-11B Q6_K, gfx1201) ===\n");
  printf("dims NE=%d NH=%d NKV=%d HD=%d NFF=%d NL=%d  ROT=%d distinct weight buffers/shape\n", NE,NH,NKV,HD,NFF,NL,ROT);
  printf("FRESH RADV baseline (-fa auto):  pp512=2373  pp2048=2289  pp8192=1989 t/s\n\n");

  double radv[3]={2373,2289,1989};

  // -------------------------------------------------------------------------------------------------
  //  SOLO mode: FF3_SOLO=<0|1|2> measures ONLY v3p+P8 realistic-cold for that one context, as the
  //  FIRST thing the process does (cool GPU). Used to remove within-process thermal ordering bias.
  // -------------------------------------------------------------------------------------------------
  if(const char* solo=getenv("FF3_SOLO")){
    int ci=atoi(solo); if(ci<0||ci>2) ci=0; int M=ctxs[ci];
    Buffers B; alloc_buffers(B,M,NE,NH,NKV,HD,NFF);
    double v3p8 = measure_forward(B,M,NE,NH,NKV,HD,NFF,eps,theta,GV3P,1,1,nullptr,0,st,NL);
    double tc=M/(v3p8/1000.0);
    const char* verdict = tc>radv[ci]*1.005 ? "BEAT" : (tc<radv[ci]*0.995 ? "LOSE" : "TIE");
    printf("SOLO M=%d v3p+P8 realistic-cold: %.2f ms  %.0f t/s  (RADV %.0f, %+.1f%%)  ==> %s\n",
           M,v3p8,tc,radv[ci],(tc/radv[ci]-1)*100,verdict);
    free_buffers(B); hipStreamDestroy(st); return 0;
  }

  // -------------------------------------------------------------------------------------------------
  //  PART A — option (A): ROTATE ROT distinct weight buffers (layer%ROT). Realistic cold.
  //  Also prints the v2-style SINGLE-buffer L2-warm CONTROL for the same config, to show the artifact.
  // -------------------------------------------------------------------------------------------------
  printf("==================================================================================\n");
  printf("PART A: rotate %d distinct weight buffers/shape (layer%%%d) = REALISTIC COLD.  v3p+P8.\n",ROT,ROT);
  printf("  WARM-CONTROL = single buffer reused every layer (== full_forward_v2, L2 partially warm)\n");
  printf("==================================================================================\n");
  for(int ci=0;ci<3;ci++){
    int M=ctxs[ci];
    Buffers B; alloc_buffers(B,M,NE,NH,NKV,HD,NFF);
    // realistic cold = v3p + P8, rotating buffers
    double cold = measure_forward(B,M,NE,NH,NKV,HD,NFF,eps,theta,GV3P,1,/*cold*/1,nullptr,0,st,NL);
    // warm control = v3p + P8, single buffer (exactly v2's measurement)
    double warm = measure_forward(B,M,NE,NH,NKV,HD,NFF,eps,theta,GV3P,1,/*cold*/0,nullptr,0,st,NL);
    auto ts=[&](double ms){ return M/(ms/1000.0); };
    double tc=ts(cold), tw=ts(warm);
    const char* verdict = tc>radv[ci]*1.005 ? "BEAT" : (tc<radv[ci]*0.995 ? "LOSE" : "TIE");
    printf("\n--- M=%d (fresh RADV %.0f) ---\n",M,radv[ci]);
    printf("  WARM-CONTROL (v2-style single buf) %.2f ms  %.0f t/s  (%+.1f%% vs RADV)  [optimistic]\n",
           warm,tw,(tw/radv[ci]-1)*100);
    printf("  REALISTIC-COLD (rotate %d bufs)    %.2f ms  %.0f t/s  (%+.1f%% vs RADV)  ==> %s\n",
           ROT,cold,tc,(tc/radv[ci]-1)*100,verdict);
    printf("  warm artifact inflated t/s by %+.1f%% (cold is the honest number)\n",(tw/tc-1)*100);
    free_buffers(B);
  }

  // -------------------------------------------------------------------------------------------------
  //  PART B — option (B) cross-check: SINGLE buffer + L2 flush before each weight GEMM, flush time
  //  subtracted. Should agree with PART A's rotate-cold (independent estimate of the cold forward).
  // -------------------------------------------------------------------------------------------------
  printf("\n==================================================================================\n");
  printf("PART B: CROSS-CHECK via L2 flush (64MB) before each weight GEMM, flush time SUBTRACTED. v3p+P8.\n");
  printf("==================================================================================\n");
  size_t FL=64u*1024*1024; int8_t* dFlush; CK(hipMalloc(&dFlush,FL)); CK(hipMemset(dFlush,7,FL));
  for(int ci=0;ci<3;ci++){
    int M=ctxs[ci];
    Buffers B; alloc_buffers(B,M,NE,NH,NKV,HD,NFF);
    // measure forward WITH flush before each weight GEMM (single buffer, since flush guarantees cold)
    double withflush = measure_forward(B,M,NE,NH,NKV,HD,NFF,eps,theta,GV3P,1,/*cold*/0,dFlush,FL,st,NL);
    // measure the pure flush cost: NL*FLUSH_GEMMS_PER_LAYER flushes, min-of-bursts
    int R=(M>=8192)?5:8; double flush_only=1e30;
    for(int r=0;r<R;r++){ double t0=now();
      for(int l=0;l<NL;l++) for(int f=0;f<FLUSH_GEMMS_PER_LAYER;f++) l2_flush<<<(FL+255)/256,256,0,st>>>(dFlush,FL);
      CK(hipStreamSynchronize(st)); double m=now()-t0; if(m<flush_only)flush_only=m; }
    double cold = withflush - flush_only; // realistic-cold forward, flush time removed
    auto ts=[&](double ms){ return M/(ms/1000.0); };
    double tc=ts(cold);
    const char* verdict = tc>radv[ci]*1.005 ? "BEAT" : (tc<radv[ci]*0.995 ? "LOSE" : "TIE");
    printf("\n--- M=%d (fresh RADV %.0f) ---\n",M,radv[ci]);
    printf("  with-flush %.2f ms  - flush-only %.2f ms  = COLD %.2f ms  %.0f t/s  (%+.1f%% vs RADV)  ==> %s\n",
           withflush,flush_only,cold,tc,(tc/radv[ci]-1)*100,verdict);
    free_buffers(B);
  }
  hipFree(dFlush);

  // -------------------------------------------------------------------------------------------------
  //  PART C — FINAL VERDICT table: realistic-cold (rotate, option A) v0/v3p with/without P8.
  // -------------------------------------------------------------------------------------------------
  printf("\n==================================================================================\n");
  printf("PART C: FINAL realistic-cold full forward (rotate %d bufs) — v0 vs v3p, P8 off/on.\n",ROT);
  printf("==================================================================================\n");
  for(int ci=0;ci<3;ci++){
    int M=ctxs[ci];
    Buffers B; alloc_buffers(B,M,NE,NH,NKV,HD,NFF);
    double v0   = measure_forward(B,M,NE,NH,NKV,HD,NFF,eps,theta,GV0 ,0,1,nullptr,0,st,NL);
    double v3   = measure_forward(B,M,NE,NH,NKV,HD,NFF,eps,theta,GV3P,0,1,nullptr,0,st,NL);
    double v0p8 = measure_forward(B,M,NE,NH,NKV,HD,NFF,eps,theta,GV0 ,1,1,nullptr,0,st,NL);
    double v3p8 = measure_forward(B,M,NE,NH,NKV,HD,NFF,eps,theta,GV3P,1,1,nullptr,0,st,NL);
    auto ts=[&](double ms){ return M/(ms/1000.0); };
    const char* verdict = ts(v3p8)>radv[ci]*1.005 ? "BEAT RADV" : (ts(v3p8)<radv[ci]*0.995 ? "LOSE to RADV" : "TIE RADV");
    printf("\n--- M=%d (fresh RADV %.0f) ---\n",M,radv[ci]);
    printf("  v0   (cold)          %.2f ms  %.0f t/s  (%+.1f%% vs RADV)\n",v0,  ts(v0),  (ts(v0)/radv[ci]-1)*100);
    printf("  v3p  (cold)          %.2f ms  %.0f t/s  (%+.1f%% vs RADV)  [v3p vs v0 %+.1f%%]\n",v3,ts(v3),(ts(v3)/radv[ci]-1)*100,(v0/v3-1)*100);
    printf("  v0 + P8  (cold)      %.2f ms  %.0f t/s  (%+.1f%% vs RADV)\n",v0p8,ts(v0p8),(ts(v0p8)/radv[ci]-1)*100);
    printf("  v3p + P8 (FINAL cold)%.2f ms  %.0f t/s  (%+.1f%% vs RADV)  ==> %s\n",v3p8,ts(v3p8),(ts(v3p8)/radv[ci]-1)*100, verdict);
    free_buffers(B);
  }
  hipStreamDestroy(st);
  return 0;
}
