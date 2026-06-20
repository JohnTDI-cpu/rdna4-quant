// gemm_v4.hip — codegen/scheduling tuning of the INT8 WMMA prefill GEMM for gfx1201/RDNA4.
//   Goal: close the ~8% codegen gap to RADV coopmat and tip Q4_K_M (currently -1.5%) to a WIN.
//   Structure FIXED at 256x256 8x4 (the proven occ-bound tile). Only codegen/scheduling changes.
//
//   Build: HIP_VISIBLE_DEVICES=0 hipcc --offload-arch=gfx1201 -O3 gemm_v4.hip -o gemm_v4
//   Resource/occ: add -Rpass-analysis=kernel-resource-usage
//   GPU: HIP_VISIBLE_DEVICES=0, `rocm-smi --setperflevel auto` first. COLD numbers are the truth.
//
//   ROOT-CAUSE found by reading the ISA of the v3 winner (gemm_v0<256,256,8,4,32>, 127 VGPR / occ-10):
//     The inner loop is K-block-serial: 16 WMMA (KSUB=2 x 8 cR tiles) then a FAT ~192-op FP32
//     scale-apply tail (64x v_cvt_f32_i32 + 64x v_mul + 64x v_fmac) that the matrix core idles
//     through. The 8% gap is exactly this matrix-idle VALU window. Levers below attack it:
//
//   V4 LEVERS (each isolated; ALL MEASURED ON gfx1201/R9700/ROCm-7.2.3, COLD = truth, 2026-06-20):
//     L0      = exact copy of the v3 winner (gemm_v0 256x256). BEST. gate 54.2 TF cold / agg 2735 t/s.
//     SETPRIO : s_setprio(1) round the WMMA cluster, s_setprio(0) for the VALU tail (HipKittens recipe)
//               -> REGRESSED gate 54.3->49 TF. Serializes more than it overlaps on an occ-bound kernel.
//     SCHEDGRP: __builtin_amdgcn_sched_group_barrier pinning WMMA/VALU -> REGRESSED gate ->46 TF.
//               Over-constrains ACO; the default schedule is already better.
//     V4E     : EARLY-FREE — fold each cR tile to Rf right after its 2 WMMAs (cR live-range=1 tile)
//               -> REGRESSED agg 2735->2274 t/s. Shrinking the live range removed the overlap freedom.
//     V4H     : ROW-FREE — fold per mt-row (cR live-range=WNT tiles) -> REGRESSED agg ->2357 t/s.
//     KC64 (+ileave, removed): a 2x-accumulator KC=64 SPILLED (16 v8i tiles) and collapsed to 20 TF.
//     -mllvm scheduler knobs (metric-bias / relaxed-occupancy / prera-direction / amdgpu-trackers):
//               ALL neutral-to-negative on the gate; amdgpu-trackers & topdown regressed hard.
//   CONCLUSION: the L0 layout is at the ACO/LLVM CODEGEN CEILING for this 256x256 8x4 tile on gfx1201.
//     The matrix-idle VALU tail is real BUT it is hidden by wave-level parallelism (occ-10/12); every
//     attempt to manually feed the matrix unit (priority / sched barriers / shorter live ranges /
//     larger KC) reduces that parallelism and loses. No codegen win is available here. The standalone
//     GEMM already BEATS RADV (+17% agg cold); the residual full-forward gap to RADV is NOT in the
//     GEMM (it is dequant/attention/dispatch — see full_forward_v3.hip).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>
typedef int   i4  __attribute__((ext_vector_type(4)));
typedef int   i8v __attribute__((ext_vector_type(8)));
typedef int   v2i __attribute__((ext_vector_type(2)));
typedef int   v8i __attribute__((ext_vector_type(8)));
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
#define CK(x) do{hipError_t e=(x);if(e){printf("HIP err %s @ %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__);}}while(0)

template<int N> __device__ __forceinline__ void wait_loadN(){ asm volatile("s_wait_loadcnt %0"::"i"(N):"memory"); }
__device__ __forceinline__ void prio(int p){ asm volatile("s_setprio %0"::"i"(0):"memory"); }
__device__ __forceinline__ void prio_hi(){ __builtin_amdgcn_s_setprio(1); }
__device__ __forceinline__ void prio_lo(){ __builtin_amdgcn_s_setprio(0); }

#define WMMA(a,b,c) __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,(a),true,(b),(c),false)

// ============================================================================
// L0 — EXACT v3 winner copy (gemm_v0<256,256,8,4,KC=32>). 127 VGPR / occ-10 / 0 spill.
//   Reference inside this binary so every lever is measured against the real winner.
// ============================================================================
template<int BM,int BN,int WMW,int WNW,int KC>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_l0(
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
      for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)cR[mt][nt]=WMMA(Af[mt],Bf[nt],cR[mt][nt]);}
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=a_scale[(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk];
    for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[mt][nt][j]*wsc[nt]*asc[mt][j];
    if(kn<K){ wait_loadN<0>();
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
}

// ============================================================================
// V4 — KC32 with codegen levers SETPRIO + SCHEDGRP (structure identical to L0).
//   SETPRIO : raise wave priority around the WMMA cluster, drop it for the scale-apply tail.
//   SCHEDGRP: emit sched_group_barrier markers to pin WMMA/VALU interleave.
// ============================================================================
template<int BM,int BN,int WMW,int WNW,int KC,int SETPRIO,int SCHEDGRP>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_v4(
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
    if(SETPRIO) prio_hi();
    for(int ks=0;ks<KC/16;ks++){ v2i Af[WMT],Bf[WNT];
      for(int mt=0;mt<WMT;mt++){const int8_t* ap=&As[buf][wm*WMT*16+mt*16+nl][ks*16+kg*8];Af[mt]=(v2i){*(const int*)ap,*(const int*)(ap+4)};}
      for(int nt=0;nt<WNT;nt++){const int8_t* bp=&Bs[buf][wn*WNT*16+nt*16+nl][ks*16+kg*8];Bf[nt]=(v2i){*(const int*)bp,*(const int*)(bp+4)};}
      for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)cR[mt][nt]=WMMA(Af[mt],Bf[nt],cR[mt][nt]);
      if(SCHEDGRP){ __builtin_amdgcn_sched_group_barrier(0x8,8,0); /*8 MFMA/WMMA*/ }
    }
    if(SETPRIO) prio_lo();
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=a_scale[(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk];
    for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[mt][nt][j]*wsc[nt]*asc[mt][j];
    if(kn<K){ wait_loadN<0>();
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
}

// ============================================================================
// V4E — EARLY-FREE scale fold: same KC32 structure/register layout as L0, but each cR[mt][nt]
//   int32 tile is converted to FP32 and folded into Rf[mt][nt] IMMEDIATELY after its two WMMAs
//   (per output tile), instead of one big tail after all 8 tiles. Shrinks the cR live-range to a
//   single tile so the int->float VALU can interleave with the remaining tiles' WMMAs and peak
//   VGPR may drop (cR no longer 8-tiles-wide alongside Rf). Numerics identical to L0.
// ============================================================================
template<int BM,int BN,int WMW,int WNW,int KC>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_v4e(
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
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=a_scale[(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk];
    #pragma unroll
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int nt=0;nt<WNT;nt++){
        v8i c=(v8i){};
        #pragma unroll
        for(int ks=0;ks<KC/16;ks++){
          const int8_t* ap=&As[buf][wm*WMT*16+mt*16+nl][ks*16+kg*8];
          const int8_t* bp=&Bs[buf][wn*WNT*16+nt*16+nl][ks*16+kg*8];
          v2i Af=(v2i){*(const int*)ap,*(const int*)(ap+4)};
          v2i Bf=(v2i){*(const int*)bp,*(const int*)(bp+4)};
          c=WMMA(Af,Bf,c);
        }
        #pragma unroll
        for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)c[j]*wsc[nt]*asc[mt][j];
      }
    if(kn<K){ wait_loadN<0>();
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
}

// ============================================================================
// V4H — ROW early-free: WMMA a full mt-row of WNT tiles (saturate matrix core within the row),
//   then fold that row to Rf so peak cR = WNT tiles (4) not 8. Middle ground; pure scheduling.
// ============================================================================
template<int BM,int BN,int WMW,int WNW,int KC>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_v4h(
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
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=a_scale[(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk];
    #pragma unroll
    for(int mt=0;mt<WMT;mt++){
      v8i cR[WNT]; for(int nt=0;nt<WNT;nt++)cR[nt]=(v8i){};
      #pragma unroll
      for(int ks=0;ks<KC/16;ks++){
        const int8_t* ap=&As[buf][wm*WMT*16+mt*16+nl][ks*16+kg*8]; v2i Af=(v2i){*(const int*)ap,*(const int*)(ap+4)};
        v2i Bf[WNT];
        #pragma unroll
        for(int nt=0;nt<WNT;nt++){const int8_t* bp=&Bs[buf][wn*WNT*16+nt*16+nl][ks*16+kg*8];Bf[nt]=(v2i){*(const int*)bp,*(const int*)(bp+4)};}
        #pragma unroll
        for(int nt=0;nt<WNT;nt++)cR[nt]=WMMA(Af,Bf[nt],cR[nt]);
      }
      #pragma unroll
      for(int nt=0;nt<WNT;nt++)
        #pragma unroll
        for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[nt][j]*wsc[nt]*asc[mt][j];
    }
    if(kn<K){ wait_loadN<0>();
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
}

// ============================================================================
// CPU/GPU helpers (Q6_K dequant + per-KCB requant) — from v3.
// ============================================================================
struct __attribute__((packed)) block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; __half d; };
__global__ void dequant_q6k(const block_q6_K* __restrict__ W, int8_t* __restrict__ Wq, float* __restrict__ Ws, int N, int K){
  int idx=blockIdx.x*blockDim.x+threadIdx.x; int nblk=K/256; if(idx>=N*nblk)return;
  int n=idx/nblk, b=idx%nblk; const block_q6_K* x=&W[idx];
  float d=__half2float(x->d);
  int8_t* wq=&Wq[(size_t)n*K+b*256]; float* ws=&Ws[(size_t)n*(K/16)+b*16];
  for(int i=0;i<16;i++)ws[i]=d*(float)x->scales[i];
  for(int g=0;g<2;g++){ const uint8_t* ql=x->ql+g*64; const uint8_t* qh=x->qh+g*32;
    for(int l=0;l<32;l++){
      int q1=((ql[l]&0xF)|(((qh[l]>>0)&3)<<4))-32;
      int q2=((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4))-32;
      int q3=((ql[l]>>4)|(((qh[l]>>4)&3)<<4))-32;
      int q4=((ql[l+32]>>4)|(((qh[l]>>6)&3)<<4))-32;
      wq[g*128+l]=q1; wq[g*128+l+32]=q2; wq[g*128+l+64]=q3; wq[g*128+l+96]=q4; }
  }
}
__global__ void requant_32(const int8_t* __restrict__ Wq, const float* __restrict__ Ws, int8_t* __restrict__ Wq32, float* __restrict__ Ws32, int N, int K){
  int idx=blockIdx.x*blockDim.x+threadIdx.x; int nb32=K/32; if(idx>=N*nb32)return;
  int n=idx/nb32, b=idx%nb32, nb16=K/16; float f[32], mx=0;
  for(int j=0;j<32;j++){int k=b*32+j; f[j]=(float)Wq[(size_t)n*K+k]*Ws[(size_t)n*nb16+k/16]; mx=fmaxf(mx,fabsf(f[j]));}
  float s=mx/127.f; if(s<1e-12f)s=1e-12f; Ws32[(size_t)n*nb32+b]=s;
  for(int j=0;j<32;j++)Wq32[(size_t)n*K+b*32+j]=(int8_t)lrintf(f[j]/s);
}

// ============================================================================
// CORRECTNESS — per-32 scale ref (matches both KC32 and KC64 paths since KC64 keeps per-32 scales).
// ============================================================================
template<int BM,int BN,int A2,int B2, typename F>
static double run_correctness(const char* name, F kern){
  int M=64,N=256,K=512,nb=K/32;          // per-32 scales for both paths
  srand(11);
  std::vector<int8_t> Wq(N*K); for(auto&x:Wq)x=(int8_t)(rand()%255-127);
  std::vector<float>  Ws(N*nb); for(auto&x:Ws)x=(0.5f+(rand()%100)/100.f)*0.01f;
  std::vector<int8_t> Aq(M*K); for(auto&x:Aq)x=(int8_t)(rand()%255-127);
  std::vector<float>  As(M*nb); for(auto&x:As)x=(0.5f+(rand()%100)/100.f)*0.02f;
  std::vector<float> Oref(M*N,0);
  for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double s=0;
    for(int blk=0;blk<nb;blk++){ long acc=0; for(int kk=0;kk<32;kk++){int k=blk*32+kk; acc+=(long)Aq[m*K+k]*Wq[n*K+k];}
      s+=(double)acc*As[m*nb+blk]*Ws[n*nb+blk]; }
    Oref[m*N+n]=(float)s; }
  int8_t* dWq;float* dWs;int8_t* dAq;float* dAs;__half* dO;
  CK(hipMalloc(&dWq,N*K));CK(hipMalloc(&dWs,N*nb*4));CK(hipMalloc(&dAq,M*K));CK(hipMalloc(&dAs,M*nb*4));CK(hipMalloc(&dO,M*N*2));
  CK(hipMemcpy(dWq,Wq.data(),N*K,hipMemcpyHostToDevice));CK(hipMemcpy(dWs,Ws.data(),N*nb*4,hipMemcpyHostToDevice));
  CK(hipMemcpy(dAq,Aq.data(),M*K,hipMemcpyHostToDevice));CK(hipMemcpy(dAs,As.data(),M*nb*4,hipMemcpyHostToDevice));
  dim3 g(N/BN,M/BM),b(A2*B2*32);
  kern<<<g,b>>>(dAq,dAs,dWq,dWs,dO,M,N,K); CK(hipDeviceSynchronize());
  std::vector<__half> hO(M*N); CK(hipMemcpy(hO.data(),dO,M*N*2,hipMemcpyDeviceToHost));
  double e=0,nn=0; for(int i=0;i<M*N;i++){float v=__half2float(hO[i]);e+=(v-Oref[i])*(v-Oref[i]);nn+=Oref[i]*Oref[i];}
  double nrmse=sqrt(e/nn);
  printf("  %-30s NRMSE=%.2e  %s\n",name,nrmse,nrmse<2e-3?"PASS":"FAIL");
  hipFree(dWq);hipFree(dWs);hipFree(dAq);hipFree(dAs);hipFree(dO);
  return nrmse;
}

// ============================================================================
// SPEED — warm (min-of-bursts) and COLD (L2-thrash). Both use per-32 scales (requant_32).
// ============================================================================
template<int BM,int BN,int A2,int B2, typename F>
static double bench_warm(F kern,int M,int N,int K,int reps,double* out_ms){
  hipGetLastError();
  int nblk=K/256,nb16=K/16,nb32=K/32;
  block_q6_K* dW;int8_t*dWq,*dWq32,*dAq;float*dWs,*dWs32,*dAs;__half*dO;
  CK(hipMalloc(&dW,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMalloc(&dWq,(size_t)N*K));CK(hipMalloc(&dWs,(size_t)N*nb16*4));
  CK(hipMalloc(&dWq32,(size_t)N*K));CK(hipMalloc(&dWs32,(size_t)N*nb32*4));CK(hipMalloc(&dAq,(size_t)M*K));CK(hipMalloc(&dAs,(size_t)M*nb32*4));CK(hipMalloc(&dO,(size_t)M*N*2));
  CK(hipMemset(dW,1,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMemset(dAq,1,(size_t)M*K));CK(hipMemset(dAs,1,(size_t)M*nb32*4));
  dequant_q6k<<<(N*nblk+127)/128,128>>>(dW,dWq,dWs,N,K);
  for(int w=0;w<3;w++)requant_32<<<(N*nb32+127)/128,128>>>(dWq,dWs,dWq32,dWs32,N,K);
  CK(hipDeviceSynchronize());
  dim3 g(N/BN,M/BM),b(A2*B2*32);
  for(int w=0;w<10;w++)kern<<<g,b>>>(dAq,dAs,dWq32,dWs32,dO,M,N,K);
  hipError_t le=hipGetLastError(), se=hipDeviceSynchronize();
  double ms;
  if(le||se){ ms=-1; *out_ms=-1; }
  else { const int BURST=8; double best=1e30;
    for(int r=0;r<reps;r++){ double t0=now(); for(int i=0;i<BURST;i++)kern<<<g,b>>>(dAq,dAs,dWq32,dWs32,dO,M,N,K); hipDeviceSynchronize();
      double m=(now()-t0)/BURST; if(m<best)best=m; }
    ms=best; *out_ms=ms; }
  hipFree(dW);hipFree(dWq);hipFree(dWs);hipFree(dWq32);hipFree(dWs32);hipFree(dAq);hipFree(dAs);hipFree(dO);
  return ms<0 ? -1.0 : 2.0*M*N*K/1e12/(ms/1e3);
}
__global__ void thrash(int8_t* p, size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]^=1; }
template<int BM,int BN,int A2,int B2, typename F>
static double bench_cold(F kern,int M,int N,int K,int reps,double* out_ms){
  hipGetLastError();
  int nblk=K/256,nb16=K/16,nb32=K/32; const int NBUF=3;
  block_q6_K* dW;int8_t*dWq,*dWq32[NBUF],*dAq;float*dWs,*dWs32[NBUF],*dAs;__half*dO;
  CK(hipMalloc(&dW,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMalloc(&dWq,(size_t)N*K));CK(hipMalloc(&dWs,(size_t)N*nb16*4));
  for(int bb=0;bb<NBUF;bb++){CK(hipMalloc(&dWq32[bb],(size_t)N*K));CK(hipMalloc(&dWs32[bb],(size_t)N*nb32*4));}
  CK(hipMalloc(&dAq,(size_t)M*K));CK(hipMalloc(&dAs,(size_t)M*nb32*4));CK(hipMalloc(&dO,(size_t)M*N*2));
  CK(hipMemset(dW,1,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMemset(dAq,1,(size_t)M*K));CK(hipMemset(dAs,1,(size_t)M*nb32*4));
  dequant_q6k<<<(N*nblk+127)/128,128>>>(dW,dWq,dWs,N,K);
  for(int bb=0;bb<NBUF;bb++)requant_32<<<(N*nb32+127)/128,128>>>(dWq,dWs,dWq32[bb],dWs32[bb],N,K);
  CK(hipDeviceSynchronize());
  size_t TH=32u*1024*1024; int8_t* dT; CK(hipMalloc(&dT,TH));
  dim3 g(N/BN,M/BM),b(A2*B2*32);
  for(int w=0;w<4;w++)kern<<<g,b>>>(dAq,dAs,dWq32[0],dWs32[0],dO,M,N,K);
  hipError_t le=hipGetLastError(), se=hipDeviceSynchronize();
  if(le||se){ *out_ms=-1;
    hipFree(dT);hipFree(dW);hipFree(dWq);hipFree(dWs);for(int bb=0;bb<NBUF;bb++){hipFree(dWq32[bb]);hipFree(dWs32[bb]);}hipFree(dAq);hipFree(dAs);hipFree(dO);
    return -1.0; }
  double t0=now();
  for(int i=0;i<reps;i++){ int bi=i%NBUF; thrash<<<(TH+255)/256,256>>>(dT,TH);
    kern<<<g,b>>>(dAq,dAs,dWq32[bi],dWs32[bi],dO,M,N,K); }
  CK(hipDeviceSynchronize());
  double tot=now()-t0;
  double t1=now(); for(int i=0;i<reps;i++)thrash<<<(TH+255)/256,256>>>(dT,TH); CK(hipDeviceSynchronize());
  double thr=now()-t1;
  double ms=(tot-thr)/reps; *out_ms=ms;
  hipFree(dT);hipFree(dW);hipFree(dWq);hipFree(dWs);for(int bb=0;bb<NBUF;bb++){hipFree(dWq32[bb]);hipFree(dWs32[bb]);}hipFree(dAq);hipFree(dAs);hipFree(dO);
  return 2.0*M*N*K/1e12/(ms/1e3);
}

struct Cand{ const char* name; double(*warm)(int,int,int,int,double*); double(*cold)(int,int,int,int,double*); };

int main(int argc, char** argv){
  printf("=== gemm_v4: codegen/scheduling tuning of INT8 WMMA prefill GEMM (gfx1201) ===\n");
  printf("Baseline L0 = v3 winner gemm_v0<256,256,8,4,32>. Target: tip Q4_K_M to a WIN (>+1.5%% vs cur).\n");
  printf("RADV gate 51 TF / 2341 t/s agg. Cur GEMM-eff agg ~2611-2741 t/s. COLD = truth.\n\n");

  printf("[1] CORRECTNESS (synthetic int8, per-32 scale ref, NRMSE<2e-3):\n");
  run_correctness<64,64,2,2>("L0  KC32 base",          gemm_l0 <64,64,2,2,32>);
  run_correctness<256,256,8,4>("L0  KC32 256x256",     gemm_l0 <256,256,8,4,32>);
  run_correctness<256,256,8,4>("V4  KC32 setprio",     gemm_v4 <256,256,8,4,32,1,0>);
  run_correctness<256,256,8,4>("V4  KC32 schedgrp",    gemm_v4 <256,256,8,4,32,0,1>);
  run_correctness<256,256,8,4>("V4  KC32 prio+sched",  gemm_v4 <256,256,8,4,32,1,1>);
  run_correctness<256,256,8,4>("V4E KC32 early-free",   gemm_v4e<256,256,8,4,32>);
  run_correctness<256,256,8,4>("V4H KC32 row-free",     gemm_v4h<256,256,8,4,32>);
  run_correctness<256,256,8,4>("V4  KC32 prio+sched",   gemm_v4 <256,256,8,4,32,1,1>);

  // candidate table (256x256 8x4 only)
  #define WARM(V,...) [](int M,int N,int K,int R,double* ms){return bench_warm<256,256,8,4>(V<256,256,8,4,##__VA_ARGS__>,M,N,K,R,ms);}
  #define COLD(V,...) [](int M,int N,int K,int R,double* ms){return bench_cold<256,256,8,4>(V<256,256,8,4,##__VA_ARGS__>,M,N,K,R,ms);}
  std::vector<Cand> C = {
    {"L0   KC32 base (v3 win)", WARM(gemm_l0,32),            COLD(gemm_l0,32)},
    {"V4E  KC32 early-free",    WARM(gemm_v4e,32),           COLD(gemm_v4e,32)},
    {"V4H  KC32 row-free",      WARM(gemm_v4h,32),           COLD(gemm_v4h,32)},
    {"V4   KC32 setprio",       WARM(gemm_v4,32,1,0),        COLD(gemm_v4,32,1,0)},
    {"V4   KC32 prio+sched",    WARM(gemm_v4,32,1,1),        COLD(gemm_v4,32,1,1)},
  };
  #undef WARM
  #undef COLD

  printf("\n[2] GATE M=512 N=14336 K=4096 — WARM (min-burst,R40) + COLD (L2-thrash,R40):\n");
  { const int M=512,N=14336,K=4096; double bestc=0; const char* bn="";
    for(auto&c:C){ double mw,mc; double tw=c.warm(M,N,K,40,&mw); double tc=c.cold(M,N,K,40,&mc);
      if(tw<0||tc<0){ printf("  %-26s  --- INVALID\n",c.name); continue; }
      printf("  %-26s WARM %.3f ms %5.1f TF | COLD %.3f ms %5.1f TF\n",c.name,mw,tw,mc,tc);
      if(tc>bestc){bestc=tc;bn=c.name;} }
    printf("  >>> BEST gate COLD: %s = %.1f TFLOPS\n",bn,bestc);
  }

  printf("\n[3] 5-SHAPE AGGREGATE M=512 -> t/s (COLD = truth). 50 layers. RADV agg 2341, cur ~2611-2741.\n");
  { struct S{const char* nm;int N,K;}; const int NL=50;
    S shapes[5]={{"qkv",6144,4096},{"o",4096,4096},{"gate",14336,4096},{"up",14336,4096},{"down",4096,14336}};
    double bestc=0; const char* bn="";
    for(auto&c:C){ double totw=0, totc=0; bool ok=true; double per[5];
      for(int s=0;s<5;s++){ double mw,mc; double tw=c.warm(512,shapes[s].N,shapes[s].K,12,&mw);
        double tc=c.cold(512,shapes[s].N,shapes[s].K,40,&mc); if(mw<0||mc<0){ok=false;break;} totw+=mw; totc+=mc; per[s]=tc; }
      if(!ok){ printf("  %-26s  --- INVALID\n",c.name); continue; }
      double tsw=512.0/(totw*NL/1000.0), tsc=512.0/(totc*NL/1000.0);
      printf("  %-26s WARM %.0f t/s | COLD %.0f t/s  (q%.0f o%.0f g%.0f u%.0f d%.0f TF)\n",
        c.name,tsw,tsc,per[0],per[1],per[2],per[3],per[4]);
      if(tsc>bestc){bestc=tsc;bn=c.name;} }
    printf("  >>> BEST agg COLD: %s = %.0f t/s  vs RADV 2341 (%+.0f%%), vs cur-GEMM 2611 (%+.1f%%)  %s\n",
      bn,bestc,(bestc/2341-1)*100,(bestc/2611-1)*100, bestc>2611*1.015?"*** TIP (>+1.5%) ***":(bestc>2611?"(edge, <1.5%)":"(below cur)"));
  }
  return 0;
}
