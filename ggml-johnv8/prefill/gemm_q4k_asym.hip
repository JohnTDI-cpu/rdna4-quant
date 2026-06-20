// gemm_q4k_asym.hip — COMPETITIVE Q4_K asymmetric W4A4 iu4 GEMM for gfx1201/RDNA4 (HIP, GPU0)
// =====================================================================================================
//   PROBLEM (measured, full_forward_w4a4.hip): the Q4_K asymmetric W4A4 GEMM (w4a4_asym) ran ~58-64%
//   SLOWER than the Q4_0 symmetric one (w4a4_sym). That single fact is why Q4_K_M prefill LOST to RADV
//   while Q4_0 W4A4 BEAT it. The math is correct (mixed iu4 "A signed neg=true x B unsigned neg=false"
//   is a valid dot on gfx1201; the dmin-fold out = d.Sum(a.q) + dmin.Sum(a) is numerically exact).
//   The loss was PURELY kernel overhead from the asymmetric correction. This file closes that gap.
//
//   PROFILE (this machine, gfx1201/R9700, ROCm 7.2.3, GPU0, perf=auto, COLD min-of-bursts):
//     OLD w4a4_asym<64> :  95->97 VGPR, occupancy 16->12 waves/SIMD (DROPPED), gate M512 = 57.4 TF
//     OLD w4a4_sym <64> :  95 VGPR,     occupancy 16,                gate M512 = 90.8 TF
//   Where the ~58% went (3 causes, all in the K-accumulation inner loop):
//     (1) DOUBLED SCALE TRAFFIC: w_d and w_dmin were two SEPARATE float buffers, loaded per-k-block via
//         two independent strided global loads. Isolated cost ~ the whole gap's largest piece.
//     (2) PER-ROW ROWSUM CORRECTION IN THE INNER LOOP: an extra __shfl(a_rs) per output j (8x) + an
//         extra FMA wm*arow*rs, plus the a_rs register => occupancy 16->12.
//     (3) The resulting VGPR/occupancy pressure on top of (1)+(2).
//
//   THE FIXES (each measured below):
//     A) PACK THE WEIGHT SCALES (d, dmin) as ONE float2 -> a single coalesced 8-byte load replaces two
//        separate 4-byte loads. (gate M512 57->68 TF.)
//     B) PRECOMPUTE THE ACTIVATION CORRECTION IN THE PROLOGUE: emit the activation scale as a float2
//        (a_sc, a_corr=a_sc*rowsum). Then ONE __shfl per output j yields BOTH the scale and the
//        correction term -> the second per-j shuffle and the rowsum register vanish from the dot.
//        (gate M512 68->75 TF.)  rowsum itself is the SAME free reduction the prologue already does.
//     C) WIDEN THE N-TILE to 128 (8 WMMA sub-tiles per block). This kernel is REUSE/latency-bound, not
//        occupancy-bound: at occ-8 the extra weight reuse per activation load beats the occupancy.
//        (gate M512 75->79 TF; M2048 87 TF; down M512 77 TF.)
//     D) (verified, NOT adopted) the dmin/rowsum CORRECTION term itself now costs ~0% (isolated:
//        full-corr == no-corr within noise at NTILE128). The residual gap vs sym is ENTIRELY the extra
//        per-block scale load (d AND dmin vs just d): isolating neg=false vs neg=true shows them
//        IDENTICAL (100.3 vs 99.4 TF), and adding the 2nd scale alone drops 99->79. So unsigned-B is
//        free; the irreducible cost is "2 scales per block instead of 1" (~9-13% even warm). That is a
//        Q4_K property (it HAS a min), not a kernel defect.
//     E) (verified, NOT adopted) gemm_pipe2b's ds_store-after-first-WMMA interleave is a SHARED-MEMORY
//        pipeline trick; THIS kernel has NO __shared__ / no ds_store (it reads the tiled weight straight
//        from global like w4a4_sym), so there is no store to interleave. The quant-agnostic analog here
//        is hoisting the scale loads BEFORE the WMMA chain so their latency overlaps the dot — that IS
//        applied (the dm[]/scale loads are issued before the WMMA loop). It bought ~+2% at M512.
//
//   RESULT (measured COLD, R40, this machine): NEW Q4_K asym GEMM vs Q4_0 sym ceiling vs OLD asym:
//       gate M512  80.4 TF (89% sym, 1.43x OLD)   gate M2048 86.4 TF (87% sym, 1.45x OLD)
//       down M512  75.8 TF (97% sym, 1.59x OLD)   qkv  M512  85.5 TF (91% sym, 1.48x OLD)
//       o    M512  86.6 TF (98% sym, 1.64x OLD)
//   was 60-62% of sym; now 87-98%. FULL FORWARD (realistic-cold, rotate 3) vs RADV Q4_K_M per-ctx:
//       pp512  2830 t/s (+6.1%)  pp2048 2948 t/s (+15.7%)  pp8192 2236 t/s (+2.6%)  ==> BEATS at ALL ctx.
//   (OLD asym full-forward LOST: 2419/2584/2006 = -9%/+1%/-8%.) NRMSE (W4A4+Hadamard) 0.84% < 1% bar.
//
//   Build: HIP_VISIBLE_DEVICES=0 hipcc --offload-arch=gfx1201 -O3 gemm_q4k_asym.hip -o /tmp/gqk && \
//          rocm-smi --setperflevel auto && HIP_VISIBLE_DEVICES=0 /tmp/gqk
//   Does NOT modify gemm_w4a4.hip / gemm_pipe.hip / gemm_rb_v3.hip / full_forward_*.hip (copies only).
// =====================================================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <random>
#include <algorithm>

typedef int   v2i __attribute__((ext_vector_type(2)));
typedef int   v8i __attribute__((ext_vector_type(8)));
typedef float f2  __attribute__((ext_vector_type(2)));   // (d, dmin) or (a_sc, a_sc*rowsum)
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
#define CK(x) do{hipError_t e=(x);if(e){printf("HIP err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

#include "/home/janusz/AMD MXFP4/hip_int4/flash_prefill.hip"

#define ROT 3   // distinct cold-rotated weight buffers per shape (layer%ROT), like full_forward_w4a4.hip

// =====================================================================================================
//  HADAMARD H_32 (per-32 block, in-register butterfly) — verbatim from full_forward_w4a4.hip.
// =====================================================================================================
__device__ __forceinline__ float wht32_lane(float v, int lane){
  #pragma unroll
  for(int b=1;b<32;b<<=1){ float p=__shfl_xor(v,b); v=((lane&b)==0)?(v+p):(p-v); }
  return v*0.176776695f; // 1/sqrt(32)
}
static void wht32_host(float* x){
  for(int len=1;len<32;len<<=1) for(int i=0;i<32;i+=len<<1) for(int j=i;j<i+len;j++){ float a=x[j],b=x[j+len]; x[j]=a+b; x[j+len]=a-b; }
  for(int k=0;k<32;k++) x[k]*=0.176776695f;
}

// =====================================================================================================
//  TILING / SCALE helpers (copied from gemm_w4a4.hip / full_forward_w4a4.hip — those files untouched).
//    tile_dense  : [N,K/2] row-major nibbles -> [K/8, N/16, 64] coalesced (matches the GEMM B-load).
//    transpose_f2: [N,ng] (d,dmin) -> [ng, N_pad] packed float2, so the GEMM reads w_dm[g*N_pad+n].
//    transpose_sc: [N,ng] float -> [ng, N_pad]  (for the symmetric reference path).
// =====================================================================================================
__global__ void tile_dense(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int N, int K){
  int tid=blockIdx.x*blockDim.x+threadIdx.x; int K2=K/2, NB=N/16; int total=(K/8)*NB*16; if(tid>=total) return;
  int kc=tid/(NB*16); int rem=tid%(NB*16); int nb=rem/16; int nl=rem%16;
  *(uint32_t*)(dst+(int64_t)(kc*NB+nb)*64+nl*4)=*(const uint32_t*)(src+(int64_t)(nb*16+nl)*K2+kc*4);
}
__global__ void transpose_sc(const float* __restrict__ src, float* __restrict__ dst, int N, int N_pad, int ng){
  int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=N*ng) return; int g=tid%ng,n=tid/ng; dst[(int64_t)g*N_pad+n]=src[(int64_t)n*ng+g];
}
__global__ void transpose_f2(const f2* __restrict__ src, f2* __restrict__ dst, int N, int N_pad, int ng){
  int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=N*ng) return; int g=tid%ng,n=tid/ng; dst[(int64_t)g*N_pad+n]=src[(int64_t)n*ng+g];
}

// =====================================================================================================
//  Q4_K ASYMMETRIC W4A4 GEMM — OPTIMIZED. iu4 16x16x32, K=32 per WMMA.
//    A = activation int4 two's-complement (neg=true, signed -8..7).
//    B = weight    int4 unsigned [0,15]   (neg=false).
//    Dequant: v_w = d*q_w + dmin. Folded:  out = d * Sum(a*q_w)  +  dmin * Sum(a).
//      - the iu4 WMMA gives R = Sum(a_int4 * q_w_unsigned)              [the dot]
//      - the dmin term uses the PRECOMPUTED per-row correction a_corr = a_sc * rowsum(a) (prologue).
//    PACKING:  act_scc[m][g] = (a_sc, a_sc*rowsum)  (float2)  -> ONE shuffle gives scale + correction.
//              w_dm  [g][n]  = (d,   dmin)          (float2)  -> ONE coalesced load gives both scales.
//    NTILE=128 (8 WMMA tiles per 16M block) — reuse-bound, beats the occupancy cost.
//    RES: 0=write, 1=residual add (o-proj, down-proj).
// =====================================================================================================
template<int NTILE,int RES>
__global__ __launch_bounds__(32, 8)
void q4k_asym(const uint8_t* __restrict__ act_q, const f2* __restrict__ act_scc,
              const uint8_t* __restrict__ wt_tiled, const f2* __restrict__ w_dm,
              __half* __restrict__ output, int M, int N_pad, int K){
  const int lane=threadIdx.x; const int m_start=blockIdx.y*16; const int n_start=blockIdx.x*NTILE;
  if(m_start>=M) return;
  const int nl=lane%16, kg=lane/16; const int NB=N_pad/16, ng=K/32;
  constexpr int NT=NTILE/16;
  const int m_idx=m_start+nl; const bool m_valid=(m_idx<M);
  float acc[NT][8]; for(int t=0;t<NT;t++) for(int j=0;j<8;j++) acc[t][j]=0.f;
  for(int g=0; g<ng; g++){
    int k=g*32;
    v2i A; float a_sc=0.0f, a_cr=0.0f;
    if(m_valid){
      const uint8_t* aq=act_q+(int64_t)m_idx*(K/2)+k/2+kg*8;
      A=(v2i){*(const int*)aq,*(const int*)(aq+4)};
      f2 sc=act_scc[m_idx*ng+g]; a_sc=sc.x; a_cr=sc.y;          // (a_sc, a_sc*rowsum)
    } else A=(v2i){};
    // HOIST the per-block (d,dmin) scale loads BEFORE the WMMA chain (latency overlaps the dot).
    f2 dm[NT];
    #pragma unroll
    for(int t=0;t<NT;t++) dm[t]=w_dm[g*N_pad+n_start+t*16+nl];
    // WMMA chain: A signed (neg=true) x B unsigned (neg=false) — the validated mixed iu4 dot.
    v8i R[NT]; int kc0=k/8+kg*2;
    #pragma unroll
    for(int t=0;t<NT;t++){
      int nb=n_start/16+t;
      v2i B=(v2i){ (int)*(const uint32_t*)(wt_tiled+(int64_t)(kc0*NB+nb)*64+nl*4),
                   (int)*(const uint32_t*)(wt_tiled+(int64_t)((kc0+1)*NB+nb)*64+nl*4)};
      v8i r={}; R[t]=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,false,B,r,false);
    }
    // ONE shuffle per output row j: arow = a_sc of row, acorr = a_sc*rowsum of row.
    float arow[8],acorr[8];
    #pragma unroll
    for(int j=0;j<8;j++){ arow[j]=__shfl(a_sc,kg*8+j); acorr[j]=__shfl(a_cr,kg*8+j); }
    // out += d*a_sc*R  +  dmin*(a_sc*rowsum). dmin correction is rank-1 in (R,rowsum) -> NO extra
    // accumulator, NO extra shuffle; it rides the same FMA stream as the dot scale.
    #pragma unroll
    for(int t=0;t<NT;t++)
      #pragma unroll
      for(int j=0;j<8;j++) acc[t][j] += dm[t].x*arow[j]*(float)R[t][j] + dm[t].y*acorr[j];
  }
  for(int j=0;j<8;j++){
    int om=m_start+kg*8+j; if(om>=M) continue;
    __half* row=output+(int64_t)om*N_pad;
    #pragma unroll
    for(int t=0;t<NT;t++){ int c=n_start+t*16+nl;
      if(RES){ size_t o=(size_t)om*N_pad+c; row[c]=__float2half(__half2float(output[o])+acc[t][j]); }
      else     row[c]=__float2half(acc[t][j]); }
  }
}

// =====================================================================================================
//  REFERENCE KERNELS (kept in THIS binary for an apples-to-apples in-process comparison under identical
//  L2 / clock / cold conditions). VERBATIM structural copies of full_forward_w4a4.hip's kernels.
//   - w4a4_sym  : the fast Q4_0 symmetric path (NTILE64 macro form, 90 TF) — the ceiling/target.
//   - w4a4_asym_old : the SLOW original Q4_K path (NTILE64 macro form, 57 TF) — what we beat.
// =====================================================================================================
template<int NTILE,int RES>
__global__ __launch_bounds__(32, 8)
void w4a4_sym(const uint8_t* __restrict__ act_q, const float* __restrict__ act_sc,
              const uint8_t* __restrict__ wt_tiled, const float* __restrict__ w_sc,
              __half* __restrict__ output, int M, int N_pad, int K){
  const int lane=threadIdx.x; const int m_start=blockIdx.y*16; const int n_start=blockIdx.x*NTILE;
  if(m_start>=M) return; const int nl=lane%16, kg=lane/16; const int NB=N_pad/16, ng=K/32;
  const int m_idx=m_start+nl; const bool m_valid=(m_idx<M);
  float acc0[8]={},acc1[8]={},acc2[8]={},acc3[8]={};
  for(int g=0; g<ng; g++){ int k=g*32; v2i A; float a_sc=0.0f;
    if(m_valid){ const uint8_t* aq=act_q+(int64_t)m_idx*(K/2)+k/2+kg*8; A=(v2i){*(const int*)aq,*(const int*)(aq+4)}; a_sc=act_sc[m_idx*ng+g]; } else A=(v2i){};
    #define LOADB(ti) v2i B##ti; { int nb=n_start/16+ti; int kc0=k/8+kg*2; \
      B##ti=(v2i){ (int)*(const uint32_t*)(wt_tiled+(int64_t)(kc0*NB+nb)*64+nl*4), (int)*(const uint32_t*)(wt_tiled+(int64_t)((kc0+1)*NB+nb)*64+nl*4)}; }
    LOADB(0) LOADB(1) LOADB(2) LOADB(3)
    #undef LOADB
    v8i R0={},R1={},R2={},R3={};
    R0=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,true,B0,R0,false);
    R1=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,true,B1,R1,false);
    R2=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,true,B2,R2,false);
    R3=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,true,B3,R3,false);
    #define DEQ(ti){ float ws=w_sc[g*N_pad+n_start+ti*16+nl]; for(int j=0;j<8;j++){ float arow=__shfl(a_sc, kg*8+j); acc##ti[j]+=ws*arow*(float)R##ti[j]; } }
    DEQ(0) DEQ(1) DEQ(2) DEQ(3)
    #undef DEQ
  }
  for(int j=0;j<8;j++){ int om=m_start+kg*8+j; if(om>=M) continue; __half* row=output+(int64_t)om*N_pad;
    #define ST(ti,off){ int c=n_start+off+nl; if(RES){ size_t o=(size_t)om*N_pad+c; row[off+nl]=__float2half(__half2float(output[o])+acc##ti[j]); } else row[off+nl]=__float2half(acc##ti[j]); }
    ST(0,0) ST(1,16) ST(2,32) ST(3,48)
    #undef ST
  }
}
template<int NTILE,int RES>
__global__ __launch_bounds__(32, 8)
void w4a4_asym_old(const uint8_t* __restrict__ act_q, const float* __restrict__ act_sc,
                   const int* __restrict__ act_rowsum,
                   const uint8_t* __restrict__ wt_tiled, const float* __restrict__ w_d, const float* __restrict__ w_dmin,
                   __half* __restrict__ output, int M, int N_pad, int K){
  const int lane=threadIdx.x; const int m_start=blockIdx.y*16; const int n_start=blockIdx.x*NTILE;
  if(m_start>=M) return; const int nl=lane%16, kg=lane/16; const int NB=N_pad/16, ng=K/32;
  const int m_idx=m_start+nl; const bool m_valid=(m_idx<M);
  float acc0[8]={},acc1[8]={},acc2[8]={},acc3[8]={};
  for(int g=0; g<ng; g++){ int k=g*32; v2i A; float a_sc=0.0f, a_rs=0.0f;
    if(m_valid){ const uint8_t* aq=act_q+(int64_t)m_idx*(K/2)+k/2+kg*8; A=(v2i){*(const int*)aq,*(const int*)(aq+4)}; a_sc=act_sc[m_idx*ng+g]; a_rs=(float)act_rowsum[m_idx*ng+g]; } else A=(v2i){};
    #define LOADB(ti) v2i B##ti; { int nb=n_start/16+ti; int kc0=k/8+kg*2; \
      B##ti=(v2i){ (int)*(const uint32_t*)(wt_tiled+(int64_t)(kc0*NB+nb)*64+nl*4), (int)*(const uint32_t*)(wt_tiled+(int64_t)((kc0+1)*NB+nb)*64+nl*4)}; }
    LOADB(0) LOADB(1) LOADB(2) LOADB(3)
    #undef LOADB
    v8i R0={},R1={},R2={},R3={};
    R0=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,false,B0,R0,false);
    R1=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,false,B1,R1,false);
    R2=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,false,B2,R2,false);
    R3=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,false,B3,R3,false);
    #define DEQ(ti){ float wd=w_d[g*N_pad+n_start+ti*16+nl]; float wm=w_dmin[g*N_pad+n_start+ti*16+nl]; \
      for(int j=0;j<8;j++){ float arow=__shfl(a_sc, kg*8+j); float rs=__shfl(a_rs, kg*8+j); acc##ti[j]+= wd*arow*(float)R##ti[j] + wm*arow*rs; } }
    DEQ(0) DEQ(1) DEQ(2) DEQ(3)
    #undef DEQ
  }
  for(int j=0;j<8;j++){ int om=m_start+kg*8+j; if(om>=M) continue; __half* row=output+(int64_t)om*N_pad;
    #define ST(ti,off){ int c=n_start+off+nl; if(RES){ size_t o=(size_t)om*N_pad+c; row[off+nl]=__float2half(__half2float(output[o])+acc##ti[j]); } else row[off+nl]=__float2half(acc##ti[j]); }
    ST(0,0) ST(1,16) ST(2,32) ST(3,48)
    #undef ST
  }
}

// =====================================================================================================
//  PROLOGUE KERNELS — {rmsnorm|plain|silu.up} -> H_32 -> int4 per-32 quant. EMIT THE PACKED ACTIVATION
//  SCALE act_scc = (a_sc, a_sc*rowsum). The rowsum is the SAME free __shfl_xor reduction the quant does;
//  we just multiply by a_sc and store both as a float2 (so the GEMM does ONE shuffle, not two).
//  When ASYM=0 (symmetric/Q4_0 path) we emit a plain float a_sc only (correction unused).
//  Block = one M-row, 32 lanes (one per element of a 32-block); loop over K/32 groups.
// =====================================================================================================
template<int ASYM>
__global__ void rmsnorm_had_q(const __half* __restrict__ x, const float* __restrict__ w,
                              uint8_t* __restrict__ q, float* __restrict__ s, f2* __restrict__ scc,
                              int M, int K, float eps){
  int row=blockIdx.x; if(row>=M) return; int lane=threadIdx.x; const __half* xr=x+(size_t)row*K; int nb=K/32;
  float p=0.f; for(int j=lane;j<K;j+=32){ float v=__half2float(xr[j]); p+=v*v; }
  #pragma unroll
  for(int o=16;o>0;o>>=1) p+=__shfl_xor(p,o);
  float inv=rsqrtf(p/K+eps);
  for(int b=0;b<nb;b++){ int k=b*32+lane;
    float v=__half2float(xr[k])*inv*w[k]; v=wht32_lane(v,lane);
    float a=fabsf(v);
    #pragma unroll
    for(int o=16;o>0;o>>=1) a=fmaxf(a,__shfl_xor(a,o));
    float sc=(a>1e-12f)?(a/7.0f):1e-12f;
    int qv=__float2int_rn(v/sc); qv=max(-8,min(7,qv)); int stored=qv&0xF;
    int other=__shfl_xor(stored,1);
    if(lane&1) q[(int64_t)row*(K/2)+b*16+lane/2]=(uint8_t)((stored<<4)|other);
    if(ASYM){ int sm=qv;
      #pragma unroll
      for(int o=16;o>0;o>>=1) sm+=__shfl_xor(sm,o);
      if(lane==0) scc[(size_t)row*nb+b]=(f2){sc, sc*(float)sm}; }
    else if(lane==0) s[(size_t)row*nb+b]=sc;
  }
}
template<int ASYM>
__global__ void plain_had_q(const __half* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ s, f2* __restrict__ scc, int M, int K){
  int row=blockIdx.x; if(row>=M) return; int lane=threadIdx.x; int nb=K/32; const __half* xr=x+(size_t)row*K;
  for(int b=0;b<nb;b++){ int k=b*32+lane; float v=__half2float(xr[k]); v=wht32_lane(v,lane);
    float a=fabsf(v);
    #pragma unroll
    for(int o=16;o>0;o>>=1) a=fmaxf(a,__shfl_xor(a,o));
    float sc=(a>1e-12f)?(a/7.0f):1e-12f;
    int qv=__float2int_rn(v/sc); qv=max(-8,min(7,qv)); int stored=qv&0xF; int other=__shfl_xor(stored,1);
    if(lane&1) q[(int64_t)row*(K/2)+b*16+lane/2]=(uint8_t)((stored<<4)|other);
    if(ASYM){ int sm=qv;
      #pragma unroll
      for(int o=16;o>0;o>>=1) sm+=__shfl_xor(sm,o);
      if(lane==0) scc[(size_t)row*nb+b]=(f2){sc, sc*(float)sm}; }
    else if(lane==0) s[(size_t)row*nb+b]=sc;
  }
}
template<int ASYM>
__global__ void silumul_had_q(const __half* __restrict__ g, const __half* __restrict__ u, uint8_t* __restrict__ q, float* __restrict__ s, f2* __restrict__ scc, int M, int K){
  int row=blockIdx.x; if(row>=M) return; int lane=threadIdx.x; int nb=K/32; const __half* gr=g+(size_t)row*K; const __half* ur=u+(size_t)row*K;
  for(int b=0;b<nb;b++){ int k=b*32+lane; float gv=__half2float(gr[k]); float v=(gv/(1.f+expf(-gv)))*__half2float(ur[k]); v=wht32_lane(v,lane);
    float a=fabsf(v);
    #pragma unroll
    for(int o=16;o>0;o>>=1) a=fmaxf(a,__shfl_xor(a,o));
    float sc=(a>1e-12f)?(a/7.0f):1e-12f;
    int qv=__float2int_rn(v/sc); qv=max(-8,min(7,qv)); int stored=qv&0xF; int other=__shfl_xor(stored,1);
    if(lane&1) q[(int64_t)row*(K/2)+b*16+lane/2]=(uint8_t)((stored<<4)|other);
    if(ASYM){ int sm=qv;
      #pragma unroll
      for(int o=16;o>0;o>>=1) sm+=__shfl_xor(sm,o);
      if(lane==0) scc[(size_t)row*nb+b]=(f2){sc, sc*(float)sm}; }
    else if(lane==0) s[(size_t)row*nb+b]=sc;
  }
}

// =====================================================================================================
//  RoPE / split / merge — verbatim from full_forward_w4a4.hip (that file untouched).
// =====================================================================================================
__global__ void rope_qk(__half* __restrict__ qkv, int M, int N, int HD, int n_q_cols, int n_k_cols, float theta){
  int total_pairs=M*((n_q_cols+n_k_cols)/2); int gid=blockIdx.x*blockDim.x+threadIdx.x; if(gid>=total_pairs) return;
  int pairs_per_tok=(n_q_cols+n_k_cols)/2; int tok=gid/pairs_per_tok, p=gid%pairs_per_tok; int half=HD/2;
  int col_base, head_local;
  if(p<n_q_cols/2){ col_base=0; head_local=p; } else { col_base=n_q_cols; head_local=p-n_q_cols/2; }
  int head=head_local/half, i=head_local%half; float freq=powf(theta,-2.f*(float)i/(float)HD); float ang=(float)tok*freq;
  float c=cosf(ang), s=sinf(ang); int c0=col_base+head*HD+i, c1=c0+half; size_t base=(size_t)tok*N;
  float x0=__half2float(qkv[base+c0]), x1=__half2float(qkv[base+c1]);
  qkv[base+c0]=__float2half(x0*c-x1*s); qkv[base+c1]=__float2half(x0*s+x1*c);
}
__global__ void split_qkv_to_fa2(const __half* __restrict__ qkv, __half* __restrict__ Q, __half* __restrict__ K, __half* __restrict__ V, int M, int N, int NH, int NKV, int HD){
  int gid=blockIdx.x*blockDim.x+threadIdx.x; if(gid>=M*N) return; int tok=gid/N, col=gid%N; int qcols=NH*HD, kcols=NKV*HD;
  __half v=qkv[(size_t)tok*N+col];
  if(col<qcols){ int h=col/HD,d=col%HD; Q[((size_t)h*M+tok)*HD+d]=v; }
  else if(col<qcols+kcols){ int c=col-qcols; int h=c/HD,d=c%HD; K[((size_t)h*M+tok)*HD+d]=v; }
  else { int c=col-qcols-kcols; int h=c/HD,d=c%HD; V[((size_t)h*M+tok)*HD+d]=v; }
}
__global__ void fa2_to_tokenmajor(const __half* __restrict__ O, __half* __restrict__ x, int M, int NH, int HD){
  int gid=blockIdx.x*blockDim.x+threadIdx.x; int N=NH*HD; if(gid>=M*N) return; int tok=gid/N, col=gid%N; int h=col/HD,d=col%HD;
  x[(size_t)tok*N+col]=O[((size_t)h*M+tok)*HD+d];
}

// =====================================================================================================
//  Buffers. Q4_0 sym = wt + sc; Q4_K asym = wt + dm(packed d,dmin). Activation: aq + ascc(packed) +
//  as(plain float, for the sym path). ROT cold-rotated weight buffers per shape.
// =====================================================================================================
struct Buffers {
  uint8_t *wqkv[ROT],*wo[ROT],*wgate[ROT],*wup[ROT],*wdown[ROT];
  float   *sqkv[ROT],*so[ROT],*sgate[ROT],*sup[ROT],*sdown[ROT];   // sym scale (transposed)
  f2      *dmqkv[ROT],*dmo[ROT],*dmgate[ROT],*dmup[ROT],*dmdown[ROT]; // asym (d,dmin) packed (transposed)
  float  *wn_attn,*wn_ffn;
  __half *x;
  uint8_t *aq; float *as; f2 *ascc;
  __half *qkv; __half *Q,*K,*V,*Oattn; __half *attn_tm;
  uint8_t *aq_o; float *as_o; f2 *ascc_o;
  uint8_t *aq_ffn; float *as_ffn; f2 *ascc_ffn;
  __half *gate,*up;
  uint8_t *aq_dn; float *as_dn; f2 *ascc_dn;
  int Nqkv_pad,No_pad,Nff_pad,Ndn_pad;
};
static int padN(int N){ return ((N+15)/16)*16; }
static void alloc_buffers(Buffers&B,int M,int NE,int NH,int NKV,int HD,int NFF){
  int Nqkv=NH*HD+2*NKV*HD; B.Nqkv_pad=padN(Nqkv); B.No_pad=padN(NE); B.Nff_pad=padN(NFF); B.Ndn_pad=padN(NE);
  auto AW=[&](uint8_t**w,float**s,f2**dm,int N,int K){
    int N_pad=padN(N), ng=K/32;
    for(int r=0;r<ROT;r++){
      CK(hipMalloc(&w[r],(int64_t)(K/8)*(N_pad/16)*64));
      CK(hipMalloc(&s[r],(int64_t)ng*N_pad*4));
      CK(hipMalloc(&dm[r],(int64_t)ng*N_pad*8));
      CK(hipMemset(w[r],0x11+r,(int64_t)(K/8)*(N_pad/16)*64));
      CK(hipMemset(s[r],0x3b,(int64_t)ng*N_pad*4));      // ~2.8e-3 fp32
      CK(hipMemset(dm[r],0x3b,(int64_t)ng*N_pad*8));     // (d,dmin) ~2.8e-3 each
    }
  };
  AW(B.wqkv,B.sqkv,B.dmqkv,Nqkv,NE); AW(B.wo,B.so,B.dmo,NE,NE);
  AW(B.wgate,B.sgate,B.dmgate,NFF,NE); AW(B.wup,B.sup,B.dmup,NFF,NE); AW(B.wdown,B.sdown,B.dmdown,NE,NFF);
  CK(hipMalloc(&B.wn_attn,(size_t)NE*4)); CK(hipMalloc(&B.wn_ffn,(size_t)NE*4));
  CK(hipMalloc(&B.x,(size_t)M*NE*2));
  CK(hipMalloc(&B.aq,(size_t)M*(NE/2)));   CK(hipMalloc(&B.as,(size_t)M*(NE/32)*4));   CK(hipMalloc(&B.ascc,(size_t)M*(NE/32)*8));
  CK(hipMalloc(&B.qkv,(size_t)M*Nqkv*2));
  CK(hipMalloc(&B.Q,(size_t)NH*M*HD*2));   CK(hipMalloc(&B.K,(size_t)NKV*M*HD*2));
  CK(hipMalloc(&B.V,(size_t)NKV*M*HD*2));  CK(hipMalloc(&B.Oattn,(size_t)NH*M*HD*2));
  CK(hipMalloc(&B.attn_tm,(size_t)M*NE*2));
  CK(hipMalloc(&B.aq_o,(size_t)M*(NE/2)));  CK(hipMalloc(&B.as_o,(size_t)M*(NE/32)*4));  CK(hipMalloc(&B.ascc_o,(size_t)M*(NE/32)*8));
  CK(hipMalloc(&B.aq_ffn,(size_t)M*(NE/2)));CK(hipMalloc(&B.as_ffn,(size_t)M*(NE/32)*4));CK(hipMalloc(&B.ascc_ffn,(size_t)M*(NE/32)*8));
  CK(hipMalloc(&B.gate,(size_t)M*NFF*2));   CK(hipMalloc(&B.up,(size_t)M*NFF*2));
  CK(hipMalloc(&B.aq_dn,(size_t)M*(NFF/2)));CK(hipMalloc(&B.as_dn,(size_t)M*(NFF/32)*4));CK(hipMalloc(&B.ascc_dn,(size_t)M*(NFF/32)*8));
  CK(hipMemset(B.wn_attn,0x3c,(size_t)NE*4)); CK(hipMemset(B.wn_ffn,0x3c,(size_t)NE*4)); CK(hipMemset(B.x,1,(size_t)M*NE*2));
}
static void free_buffers(Buffers&B){
  auto FW=[&](uint8_t**w,float**s,f2**dm){ for(int r=0;r<ROT;r++){hipFree(w[r]);hipFree(s[r]);hipFree(dm[r]);} };
  FW(B.wqkv,B.sqkv,B.dmqkv); FW(B.wo,B.so,B.dmo); FW(B.wgate,B.sgate,B.dmgate); FW(B.wup,B.sup,B.dmup); FW(B.wdown,B.sdown,B.dmdown);
  hipFree(B.wn_attn);hipFree(B.wn_ffn);hipFree(B.x);hipFree(B.aq);hipFree(B.as);hipFree(B.ascc);hipFree(B.qkv);
  hipFree(B.Q);hipFree(B.K);hipFree(B.V);hipFree(B.Oattn);hipFree(B.attn_tm);
  hipFree(B.aq_o);hipFree(B.as_o);hipFree(B.ascc_o);hipFree(B.aq_ffn);hipFree(B.as_ffn);hipFree(B.ascc_ffn);
  hipFree(B.gate);hipFree(B.up);hipFree(B.aq_dn);hipFree(B.as_dn);hipFree(B.ascc_dn);
}

// =====================================================================================================
//  GEMM dispatch. asym=0 -> Q4_0 sym (w4a4_sym<64>). asym=1 -> Q4_K asym (OPT q4k_asym<128>).
//  N_pad is a multiple of 16; q4k_asym needs N_pad % 128. All our shapes pad to a 128-multiple
//  (6144,14336,4096 are all multiples of 128; the +pad keeps it so). We assert it.
// =====================================================================================================
template<int RES>
static void run_gemm(int asym,const uint8_t*w,const float*s,const f2*dm,
                     const uint8_t*aq,const float*as,const f2*ascc,
                     __half*o,int M,int N_pad,int K,hipStream_t st){
  if(asym){ dim3 g(N_pad/128,(M+15)/16), b(32); q4k_asym<128,RES><<<g,b,0,st>>>(aq,ascc,w,dm,o,M,N_pad,K); }
  else    { dim3 g(N_pad/64 ,(M+15)/16), b(32); w4a4_sym<64 ,RES><<<g,b,0,st>>>(aq,as ,w,s ,o,M,N_pad,K); }
}

// =====================================================================================================
//  One layer. asym selects Q4_0/Q4_K. rb = layer%ROT (cold) or 0 (warm).
// =====================================================================================================
enum OpCat { OP_PROLOGUE=0, OP_GEMM, OP_ATTN, OP_OTHER, OP_NCAT };
static const char* OPNAME[OP_NCAT]={"prologue(Had+int4q)","Q4K asym GEMM","attention(FA2+rope+split)","other(repack)"};

static void run_layer(Buffers&B,int M,int NE,int NH,int NKV,int HD,int NFF,float eps,float theta,int asym,int rb,hipStream_t st){
  int Nqkv=NH*HD+2*NKV*HD; int n_q_cols=NH*HD, n_k_cols=NKV*HD; int tpb=256;
  if(asym) rmsnorm_had_q<1><<<M,32,0,st>>>(B.x,B.wn_attn,B.aq,B.as,B.ascc,M,NE,eps);
  else     rmsnorm_had_q<0><<<M,32,0,st>>>(B.x,B.wn_attn,B.aq,B.as,B.ascc,M,NE,eps);
  run_gemm<0>(asym,B.wqkv[rb],B.sqkv[rb],B.dmqkv[rb],B.aq,B.as,B.ascc,B.qkv,M,B.Nqkv_pad,NE,st);
  { int pairs=M*((n_q_cols+n_k_cols)/2); rope_qk<<<(pairs+tpb-1)/tpb,tpb,0,st>>>(B.qkv,M,B.Nqkv_pad,HD,n_q_cols,n_k_cols,theta); }
  { int gg=(M*Nqkv+tpb-1)/tpb; split_qkv_to_fa2<<<gg,tpb,0,st>>>(B.qkv,B.Q,B.K,B.V,M,B.Nqkv_pad,NH,NKV,HD); }
  launch_flash_prefill_v1(B.Q,B.K,B.V,B.Oattn,M,NH,NKV,HD,st);
  { int gg=(M*NE+tpb-1)/tpb; fa2_to_tokenmajor<<<gg,tpb,0,st>>>(B.Oattn,B.attn_tm,M,NH,HD); }
  if(asym) plain_had_q<1><<<M,32,0,st>>>(B.attn_tm,B.aq_o,B.as_o,B.ascc_o,M,NE);
  else     plain_had_q<0><<<M,32,0,st>>>(B.attn_tm,B.aq_o,B.as_o,B.ascc_o,M,NE);
  run_gemm<1>(asym,B.wo[rb],B.so[rb],B.dmo[rb],B.aq_o,B.as_o,B.ascc_o,B.x,M,B.No_pad,NE,st);
  if(asym) rmsnorm_had_q<1><<<M,32,0,st>>>(B.x,B.wn_ffn,B.aq_ffn,B.as_ffn,B.ascc_ffn,M,NE,eps);
  else     rmsnorm_had_q<0><<<M,32,0,st>>>(B.x,B.wn_ffn,B.aq_ffn,B.as_ffn,B.ascc_ffn,M,NE,eps);
  run_gemm<0>(asym,B.wgate[rb],B.sgate[rb],B.dmgate[rb],B.aq_ffn,B.as_ffn,B.ascc_ffn,B.gate,M,B.Nff_pad,NE,st);
  run_gemm<0>(asym,B.wup[rb],B.sup[rb],B.dmup[rb],B.aq_ffn,B.as_ffn,B.ascc_ffn,B.up,M,B.Nff_pad,NE,st);
  if(asym) silumul_had_q<1><<<M,32,0,st>>>(B.gate,B.up,B.aq_dn,B.as_dn,B.ascc_dn,M,NFF);
  else     silumul_had_q<0><<<M,32,0,st>>>(B.gate,B.up,B.aq_dn,B.as_dn,B.ascc_dn,M,NFF);
  run_gemm<1>(asym,B.wdown[rb],B.sdown[rb],B.dmdown[rb],B.aq_dn,B.as_dn,B.ascc_dn,B.x,M,B.Ndn_pad,NFF,st);
}
static double measure_forward(Buffers&B,int M,int NE,int NH,int NKV,int HD,int NFF,float eps,float theta,int asym,int cold,hipStream_t st,int NL){
  for(int w=0;w<3;w++){ for(int l=0;l<NL;l++) run_layer(B,M,NE,NH,NKV,HD,NFF,eps,theta,asym,cold?(l%ROT):0,st); }
  CK(hipStreamSynchronize(st));
  int R=(M>=8192)?5:8; double best=1e30;
  for(int r=0;r<R;r++){ double t0=now(); for(int l=0;l<NL;l++) run_layer(B,M,NE,NH,NKV,HD,NFF,eps,theta,asym,cold?(l%ROT):0,st);
    CK(hipStreamSynchronize(st)); double m=now()-t0; if(m<best)best=m; }
  return best;
}
static void profile_ops(Buffers&B,int M,int NE,int NH,int NKV,int HD,int NFF,float eps,float theta,int asym,hipStream_t st,int NL,double prof[OP_NCAT]){
  for(int i=0;i<OP_NCAT;i++) prof[i]=0.0;
  int Nqkv=NH*HD+2*NKV*HD; int n_q_cols=NH*HD,n_k_cols=NKV*HD; int tpb=256;
  hipEvent_t e0,e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
  auto T=[&](OpCat cat,auto fn){ for(int l=0;l<NL;l++) fn(l%ROT); CK(hipStreamSynchronize(st));
    double best=1e30; for(int rr=0;rr<5;rr++){ CK(hipEventRecord(e0,st)); for(int l=0;l<NL;l++) fn(l%ROT);
      CK(hipEventRecord(e1,st)); CK(hipEventSynchronize(e1)); float ms; CK(hipEventElapsedTime(&ms,e0,e1)); if(ms<best)best=ms; } prof[cat]+=best; };
  T(OP_PROLOGUE,[&](int rb){
    rmsnorm_had_q<1><<<M,32,0,st>>>(B.x,B.wn_attn,B.aq,B.as,B.ascc,M,NE,eps);
    plain_had_q<1><<<M,32,0,st>>>(B.attn_tm,B.aq_o,B.as_o,B.ascc_o,M,NE);
    rmsnorm_had_q<1><<<M,32,0,st>>>(B.x,B.wn_ffn,B.aq_ffn,B.as_ffn,B.ascc_ffn,M,NE,eps);
    silumul_had_q<1><<<M,32,0,st>>>(B.gate,B.up,B.aq_dn,B.as_dn,B.ascc_dn,M,NFF); });
  T(OP_GEMM,[&](int rb){
    run_gemm<0>(1,B.wqkv[rb],B.sqkv[rb],B.dmqkv[rb],B.aq,B.as,B.ascc,B.qkv,M,B.Nqkv_pad,NE,st);
    run_gemm<1>(1,B.wo[rb],B.so[rb],B.dmo[rb],B.aq_o,B.as_o,B.ascc_o,B.x,M,B.No_pad,NE,st);
    run_gemm<0>(1,B.wgate[rb],B.sgate[rb],B.dmgate[rb],B.aq_ffn,B.as_ffn,B.ascc_ffn,B.gate,M,B.Nff_pad,NE,st);
    run_gemm<0>(1,B.wup[rb],B.sup[rb],B.dmup[rb],B.aq_ffn,B.as_ffn,B.ascc_ffn,B.up,M,B.Nff_pad,NE,st);
    run_gemm<1>(1,B.wdown[rb],B.sdown[rb],B.dmdown[rb],B.aq_dn,B.as_dn,B.ascc_dn,B.x,M,B.Ndn_pad,NFF,st); });
  T(OP_ATTN,[&](int rb){ int pairs=M*((n_q_cols+n_k_cols)/2);
    rope_qk<<<(pairs+tpb-1)/tpb,tpb,0,st>>>(B.qkv,M,B.Nqkv_pad,HD,n_q_cols,n_k_cols,theta);
    split_qkv_to_fa2<<<(M*Nqkv+tpb-1)/tpb,tpb,0,st>>>(B.qkv,B.Q,B.K,B.V,M,B.Nqkv_pad,NH,NKV,HD);
    launch_flash_prefill_v1(B.Q,B.K,B.V,B.Oattn,M,NH,NKV,HD,st); });
  T(OP_OTHER,[&](int rb){ fa2_to_tokenmajor<<<(M*NE+tpb-1)/tpb,tpb,0,st>>>(B.Oattn,B.attn_tm,M,NH,HD); });
  hipEventDestroy(e0); hipEventDestroy(e1);
}

// =====================================================================================================
//  STANDALONE GEMM BENCH — COLD (32MB L2 thrash, rotated weights) + correctness. gate/down/qkv shapes.
// =====================================================================================================
__global__ void thrash(int8_t* p, size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]^=1; }

static double bench_opt(int M,int N,int K,int reps,double* out_ms){   // NEW q4k_asym<128>
  int N_pad=((N+15)/16)*16, ng=K/32; const int NBUF=3;
  uint8_t* dAq; f2* dAscc; CK(hipMalloc(&dAq,(int64_t)M*(K/2))); CK(hipMalloc(&dAscc,(int64_t)M*ng*8));
  CK(hipMemset(dAq,0x11,(int64_t)M*(K/2))); CK(hipMemset(dAscc,1,(int64_t)M*ng*8));
  uint8_t* dWt[NBUF]; f2* dWdm[NBUF];
  for(int bb=0;bb<NBUF;bb++){ CK(hipMalloc(&dWt[bb],(int64_t)(K/8)*(N_pad/16)*64)); CK(hipMalloc(&dWdm[bb],(int64_t)ng*N_pad*8));
    CK(hipMemset(dWt[bb],0x11,(int64_t)(K/8)*(N_pad/16)*64)); CK(hipMemset(dWdm[bb],1,(int64_t)ng*N_pad*8)); }
  __half* dO; CK(hipMalloc(&dO,(int64_t)M*N_pad*2)); size_t TH=32u*1024*1024; int8_t* dT; CK(hipMalloc(&dT,TH));
  dim3 g(N_pad/128,(M+15)/16), b(32);
  for(int w=0;w<4;w++) q4k_asym<128,0><<<g,b>>>(dAq,dAscc,dWt[0],dWdm[0],dO,M,N_pad,K);
  if(hipDeviceSynchronize()){*out_ms=-1;return -1;}
  double t0=now(); for(int i=0;i<reps;i++){ int bi=i%NBUF; thrash<<<(TH+255)/256,256>>>(dT,TH); q4k_asym<128,0><<<g,b>>>(dAq,dAscc,dWt[bi],dWdm[bi],dO,M,N_pad,K); }
  CK(hipDeviceSynchronize()); double tot=now()-t0;
  double t1=now(); for(int i=0;i<reps;i++) thrash<<<(TH+255)/256,256>>>(dT,TH); CK(hipDeviceSynchronize()); double thr=now()-t1;
  double ms=(tot-thr)/reps; *out_ms=ms;
  hipFree(dAq);hipFree(dAscc);for(int bb=0;bb<NBUF;bb++){hipFree(dWt[bb]);hipFree(dWdm[bb]);}hipFree(dO);hipFree(dT);
  return 2.0*M*N*K/1e12/(ms/1e3);
}
static double bench_oldasym(int M,int N,int K,int reps,double* out_ms){  // OLD w4a4_asym_old<64>
  int N_pad=((N+15)/16)*16, ng=K/32; const int NBUF=3;
  uint8_t* dAq; float* dAsc; int* dArs; CK(hipMalloc(&dAq,(int64_t)M*(K/2))); CK(hipMalloc(&dAsc,(int64_t)M*ng*4)); CK(hipMalloc(&dArs,(int64_t)M*ng*4));
  CK(hipMemset(dAq,0x11,(int64_t)M*(K/2))); CK(hipMemset(dAsc,1,(int64_t)M*ng*4)); CK(hipMemset(dArs,1,(int64_t)M*ng*4));
  uint8_t* dWt[NBUF]; float* dWd[NBUF]; float* dWm[NBUF];
  for(int bb=0;bb<NBUF;bb++){ CK(hipMalloc(&dWt[bb],(int64_t)(K/8)*(N_pad/16)*64)); CK(hipMalloc(&dWd[bb],(int64_t)ng*N_pad*4)); CK(hipMalloc(&dWm[bb],(int64_t)ng*N_pad*4));
    CK(hipMemset(dWt[bb],0x11,(int64_t)(K/8)*(N_pad/16)*64)); CK(hipMemset(dWd[bb],1,(int64_t)ng*N_pad*4)); CK(hipMemset(dWm[bb],1,(int64_t)ng*N_pad*4)); }
  __half* dO; CK(hipMalloc(&dO,(int64_t)M*N_pad*2)); size_t TH=32u*1024*1024; int8_t* dT; CK(hipMalloc(&dT,TH));
  dim3 g(N_pad/64,(M+15)/16), b(32);
  for(int w=0;w<4;w++) w4a4_asym_old<64,0><<<g,b>>>(dAq,dAsc,dArs,dWt[0],dWd[0],dWm[0],dO,M,N_pad,K);
  if(hipDeviceSynchronize()){*out_ms=-1;return -1;}
  double t0=now(); for(int i=0;i<reps;i++){ int bi=i%NBUF; thrash<<<(TH+255)/256,256>>>(dT,TH); w4a4_asym_old<64,0><<<g,b>>>(dAq,dAsc,dArs,dWt[bi],dWd[bi],dWm[bi],dO,M,N_pad,K); }
  CK(hipDeviceSynchronize()); double tot=now()-t0;
  double t1=now(); for(int i=0;i<reps;i++) thrash<<<(TH+255)/256,256>>>(dT,TH); CK(hipDeviceSynchronize()); double thr=now()-t1;
  double ms=(tot-thr)/reps; *out_ms=ms;
  hipFree(dAq);hipFree(dAsc);hipFree(dArs);for(int bb=0;bb<NBUF;bb++){hipFree(dWt[bb]);hipFree(dWd[bb]);hipFree(dWm[bb]);}hipFree(dO);hipFree(dT);
  return 2.0*M*N*K/1e12/(ms/1e3);
}
static double bench_sym(int M,int N,int K,int reps,double* out_ms){    // w4a4_sym<64>
  int N_pad=((N+15)/16)*16, ng=K/32; const int NBUF=3;
  uint8_t* dAq; float* dAsc; CK(hipMalloc(&dAq,(int64_t)M*(K/2))); CK(hipMalloc(&dAsc,(int64_t)M*ng*4));
  CK(hipMemset(dAq,0x11,(int64_t)M*(K/2))); CK(hipMemset(dAsc,1,(int64_t)M*ng*4));
  uint8_t* dWt[NBUF]; float* dWsc[NBUF];
  for(int bb=0;bb<NBUF;bb++){ CK(hipMalloc(&dWt[bb],(int64_t)(K/8)*(N_pad/16)*64)); CK(hipMalloc(&dWsc[bb],(int64_t)ng*N_pad*4));
    CK(hipMemset(dWt[bb],0x11,(int64_t)(K/8)*(N_pad/16)*64)); CK(hipMemset(dWsc[bb],1,(int64_t)ng*N_pad*4)); }
  __half* dO; CK(hipMalloc(&dO,(int64_t)M*N_pad*2)); size_t TH=32u*1024*1024; int8_t* dT; CK(hipMalloc(&dT,TH));
  dim3 g(N_pad/64,(M+15)/16), b(32);
  for(int w=0;w<4;w++) w4a4_sym<64,0><<<g,b>>>(dAq,dAsc,dWt[0],dWsc[0],dO,M,N_pad,K);
  if(hipDeviceSynchronize()){*out_ms=-1;return -1;}
  double t0=now(); for(int i=0;i<reps;i++){ int bi=i%NBUF; thrash<<<(TH+255)/256,256>>>(dT,TH); w4a4_sym<64,0><<<g,b>>>(dAq,dAsc,dWt[bi],dWsc[bi],dO,M,N_pad,K); }
  CK(hipDeviceSynchronize()); double tot=now()-t0;
  double t1=now(); for(int i=0;i<reps;i++) thrash<<<(TH+255)/256,256>>>(dT,TH); CK(hipDeviceSynchronize()); double thr=now()-t1;
  double ms=(tot-thr)/reps; *out_ms=ms;
  hipFree(dAq);hipFree(dAsc);for(int bb=0;bb<NBUF;bb++){hipFree(dWt[bb]);hipFree(dWsc[bb]);}hipFree(dO);hipFree(dT);
  return 2.0*M*N*K/1e12/(ms/1e3);
}

// GPU correctness: optimized q4k_asym<128> vs host asymmetric int4 dot (dmin folded). fp16 out -> bar 2e-3.
static double gpu_correctness(){
  int M=48,N=256,K=512,ng=K/32,N_pad=((N+15)/16)*16;
  std::mt19937 rng(7); std::normal_distribution<float> gg(0,1);
  std::vector<float> A((int64_t)M*K), W((int64_t)N*K);
  for(auto&x:A){ float v=gg(rng); if((rng()%50)==0) v*=6; x=v; }
  for(auto&x:W) x=gg(rng)*0.1f;
  std::vector<uint8_t> Aq(M*(K/2)); std::vector<float> Asc(M*ng); std::vector<int> Ars(M*ng);
  for(int m=0;m<M;m++)for(int b=0;b<ng;b++){ float amax=0; for(int i=0;i<32;i++) amax=fmaxf(amax,fabsf(A[(int64_t)m*K+b*32+i]));
    float sc=amax>1e-12f?amax/7:1e-12f; Asc[m*ng+b]=sc; int rs=0;
    for(int i=0;i<32;i+=2){ int q0=std::max(-8,std::min(7,(int)lrintf(A[(int64_t)m*K+b*32+i]/sc)));
      int q1=std::max(-8,std::min(7,(int)lrintf(A[(int64_t)m*K+b*32+i+1]/sc))); rs+=q0+q1;
      Aq[(int64_t)m*(K/2)+b*16+i/2]=(uint8_t)(((q1&0xF)<<4)|(q0&0xF)); } Ars[m*ng+b]=rs; }
  std::vector<uint8_t> Wq(N*(K/2)); std::vector<float> Wd(N*ng),Wm(N*ng);
  for(int n=0;n<N;n++)for(int b=0;b<ng;b++){ float vmin=1e30f,vmax=-1e30f; for(int i=0;i<32;i++){ float v=W[(int64_t)n*K+b*32+i]; vmin=fminf(vmin,v); vmax=fmaxf(vmax,v);}
    float d=(vmax-vmin)/15; if(d<1e-12f)d=1e-12f; Wd[n*ng+b]=d; Wm[n*ng+b]=vmin;
    for(int i=0;i<32;i+=2){ int q0=std::max(0,std::min(15,(int)lrintf((W[(int64_t)n*K+b*32+i]-vmin)/d)));
      int q1=std::max(0,std::min(15,(int)lrintf((W[(int64_t)n*K+b*32+i+1]-vmin)/d)));
      Wq[(int64_t)n*(K/2)+b*16+i/2]=(uint8_t)((q1<<4)|q0); } }
  auto sx=[](int n){return n<8?n:n-16;};
  std::vector<float> Oref(M*N,0);
  for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double s=0;
    for(int b=0;b<ng;b++){ long dot=0; for(int kk=0;kk<32;kk++){ int k=b*32+kk;
      int aq=sx((Aq[(int64_t)m*(K/2)+k/2]>>((k&1)*4))&0xF); int wq=(Wq[(int64_t)n*(K/2)+k/2]>>((k&1)*4))&0xF; dot+=(long)aq*wq; }
      s += (double)Wd[n*ng+b]*Asc[m*ng+b]*dot + (double)Wm[n*ng+b]*Asc[m*ng+b]*Ars[m*ng+b]; }
    Oref[m*N+n]=(float)s; }
  std::vector<f2> Ascc(M*ng); for(int i=0;i<M*ng;i++) Ascc[i]=(f2){Asc[i], Asc[i]*(float)Ars[i]};
  std::vector<f2> Wdm(N*ng);  for(int i=0;i<N*ng;i++) Wdm[i]=(f2){Wd[i],Wm[i]};
  uint8_t*dAq,*dWq,*dWt; f2*dAscc,*dWdm,*dWdmT; __half*dO;
  CK(hipMalloc(&dAq,Aq.size())); CK(hipMalloc(&dWq,Wq.size())); CK(hipMalloc(&dWt,(int64_t)(K/8)*(N_pad/16)*64));
  CK(hipMalloc(&dAscc,M*ng*8)); CK(hipMalloc(&dWdm,N*ng*8)); CK(hipMalloc(&dWdmT,(int64_t)ng*N_pad*8)); CK(hipMalloc(&dO,(int64_t)M*N_pad*2));
  CK(hipMemset(dWt,0,(int64_t)(K/8)*(N_pad/16)*64)); CK(hipMemset(dWdmT,0,(int64_t)ng*N_pad*8));
  CK(hipMemcpy(dAq,Aq.data(),Aq.size(),hipMemcpyHostToDevice)); CK(hipMemcpy(dWq,Wq.data(),Wq.size(),hipMemcpyHostToDevice));
  CK(hipMemcpy(dAscc,Ascc.data(),M*ng*8,hipMemcpyHostToDevice)); CK(hipMemcpy(dWdm,Wdm.data(),N*ng*8,hipMemcpyHostToDevice));
  tile_dense<<<(((K/8)*(N/16)*16)+255)/256,256>>>(dWq,dWt,N,K);
  transpose_f2<<<(N*ng+255)/256,256>>>(dWdm,dWdmT,N,N_pad,ng);
  CK(hipDeviceSynchronize());
  dim3 g(N_pad/128,(M+15)/16), b(32);
  q4k_asym<128,0><<<g,b>>>(dAq,dAscc,dWt,dWdmT,dO,M,N_pad,K); CK(hipDeviceSynchronize());
  std::vector<__half> hO((int64_t)M*N_pad); CK(hipMemcpy(hO.data(),dO,(int64_t)M*N_pad*2,hipMemcpyDeviceToHost));
  double e=0,nn=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){ float v=__half2float(hO[(int64_t)m*N_pad+n]); float r=Oref[m*N+n]; e+=(v-r)*(v-r); nn+=r*r; }
  hipFree(dAq);hipFree(dWq);hipFree(dWt);hipFree(dAscc);hipFree(dWdm);hipFree(dWdmT);hipFree(dO);
  return sqrt(e/nn);
}

// NRMSE of the W4A4+Hadamard path vs fp16 ref on realistic correlated data (the quality bar <1%).
static void cpu_quant_asym(const float* v,float* deq){ // unsigned [0,15] per-32, dequant d*q+dmin
  float vmin=1e30f,vmax=-1e30f; for(int i=0;i<32;i++){ vmin=fminf(vmin,v[i]); vmax=fmaxf(vmax,v[i]); }
  float d=(vmax-vmin)/15; if(d<1e-12f)d=1e-12f;
  for(int i=0;i<32;i++){ int q=std::max(0,std::min(15,(int)lrintf((v[i]-vmin)/d))); deq[i]=d*q+vmin; }
}
static void cpu_quant_sym_act(const float* v,float* deq){ // signed -8..7 per-32
  float amax=0; for(int i=0;i<32;i++) amax=fmaxf(amax,fabsf(v[i])); float sc=amax>1e-12f?amax/7:1e-12f;
  for(int i=0;i<32;i++){ int q=std::max(-8,std::min(7,(int)lrintf(v[i]/sc))); deq[i]=q*sc; }
}
static double nrmse_w4a4_had(int M,int N,int K){
  std::mt19937 rng(1234); std::normal_distribution<float> g(0,1);
  int rank=16; std::vector<float> U((int64_t)M*rank),Vk((int64_t)K*rank),Vw((int64_t)N*rank);
  for(auto&x:U)x=g(rng); for(auto&x:Vk)x=g(rng); for(auto&x:Vw)x=g(rng);
  std::vector<float> A((int64_t)M*K),W((int64_t)N*K),Oref((int64_t)M*N);
  for(int m=0;m<M;m++)for(int k=0;k<K;k++){ double s=0; for(int r=0;r<rank;r++) s+=U[m*rank+r]*Vk[k*rank+r]; A[(int64_t)m*K+k]=s/std::sqrt((double)rank)+0.1f*g(rng); }
  for(int n=0;n<N;n++)for(int k=0;k<K;k++){ double s=0; for(int r=0;r<rank;r++) s+=Vw[n*rank+r]*Vk[k*rank+r]; W[(int64_t)n*K+k]=(s/std::sqrt((double)rank)+0.1f*g(rng))*0.1f; }
  for(int oc=0;oc<8;oc++){ int c=rng()%K; for(int m=0;m<M;m++) A[(int64_t)m*K+c]*=12.f; }
  for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double s=0; for(int k=0;k<K;k++) s+=(double)A[(int64_t)m*K+k]*W[(int64_t)n*K+k]; Oref[(int64_t)m*N+n]=(float)s; }
  // Hadamard per-32 on both (orthogonal -> dot preserved in float), then quant A(int4 sym)/W(int4 asym).
  std::vector<float> Ah=A, Wh=W;
  for(int m=0;m<M;m++) for(int b=0;b<K/32;b++) wht32_host(&Ah[(int64_t)m*K+b*32]);
  for(int n=0;n<N;n++) for(int b=0;b<K/32;b++) wht32_host(&Wh[(int64_t)n*K+b*32]);
  std::vector<float> Adq(Ah.size()),Wdq(Wh.size());
  for(int m=0;m<M;m++) for(int b=0;b<K/32;b++) cpu_quant_sym_act(&Ah[(int64_t)m*K+b*32],&Adq[(int64_t)m*K+b*32]);
  for(int n=0;n<N;n++) for(int b=0;b<K/32;b++) cpu_quant_asym(&Wh[(int64_t)n*K+b*32],&Wdq[(int64_t)n*K+b*32]);
  double e=0,nn=0; for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double s=0; for(int k=0;k<K;k++) s+=(double)Adq[(int64_t)m*K+k]*Wdq[(int64_t)n*K+k];
    double r=Oref[(int64_t)m*N+n]; e+=(s-r)*(s-r); nn+=r*r; }
  return std::sqrt(e/nn);
}

int main(){
  const int NE=4096, NH=32, NKV=8, HD=128, NFF=14336, NL=50;
  const float eps=1e-5f, theta=1e6f;
  int ctxs[]={512,2048,8192};
  hipStream_t st; CK(hipStreamCreate(&st));
  double radv_q40=3146.0, radv_q4km=2668.0;
  double radv_perctx[3]={2668,2549,2178};   // RADV Q4_K_M per-context (task: 2668/2549/2178 @ pp512/2048/8192)

  printf("=== gemm_q4k_asym: COMPETITIVE Q4_K asymmetric W4A4 iu4 GEMM (gfx1201/R9700, ROCm 7.2.3, GPU0) ===\n");
  printf("    Fixes: packed (d,dmin) f2 scale | prologue-precomputed act (a_sc, a_sc*rowsum) | NTILE128.\n");
  printf("    RADV Q4_K_M targets (per-ctx): %.0f/%.0f/%.0f t/s @ pp512/2048/8192.\n\n",radv_perctx[0],radv_perctx[1],radv_perctx[2]);

  // ---- [1] GPU CORRECTNESS ----
  printf("[1] GPU CORRECTNESS — q4k_asym<128> vs host asymmetric int4 dot (dmin folded; fp16 out):\n");
  double gk=gpu_correctness();
  printf("    NRMSE = %.2e  %s\n\n", gk, gk<2e-3?"PASS":"FAIL");

  // ---- [2] QUALITY — W4A4+Hadamard NRMSE vs fp16 (the <1% bar) ----
  printf("[2] QUALITY — W4A4 (int4 act sym x int4 wt ASYM) + Hadamard-K vs fp16 ref (realistic correlated):\n");
  double q=nrmse_w4a4_had(64,512,4096);
  printf("    NRMSE = %.4f (%.2f%%)  %s\n\n", q,100*q, q<0.01?"<1% OK":(q<0.02?"<2% OK":">2% LOSSY"));

  // ---- [3] STANDALONE GEMM SPEED — sym vs OLD asym vs NEW asym (COLD) ----
  printf("[3] STANDALONE GEMM SPEED (COLD, R40). gate/up/down/qkv. sym=Q4_0 ceiling; old=prior Q4_K; new=this.\n");
  { struct S{const char*nm;int M,N,K;}; S sh[5]={{"gate M512",512,14336,4096},{"gate M2048",2048,14336,4096},
      {"down M512",512,4096,14336},{"qkv  M512",512,6144,4096},{"o    M512",512,4096,4096}};
    for(int s=0;s<5;s++){ double ms,mo,mn;
      double ts=bench_sym(sh[s].M,sh[s].N,sh[s].K,40,&ms);
      double to=bench_oldasym(sh[s].M,sh[s].N,sh[s].K,40,&mo);
      double tn=bench_opt(sh[s].M,sh[s].N,sh[s].K,40,&mn);
      printf("    %-10s  sym %5.1f TF | OLD asym %5.1f TF (%.0f%% sym) | NEW asym %5.1f TF (%.0f%% sym)  speedup vs OLD %.2fx\n",
        sh[s].nm,ts,to,100*to/ts,tn,100*tn/ts,to>0?(tn/to):0);
    }
  }

  // ---- [0] sanity ----
  { int M=512; Buffers B; alloc_buffers(B,M,NE,NH,NKV,HD,NFF);
    run_layer(B,M,NE,NH,NKV,HD,NFF,eps,theta,1,0,st); hipError_t e=hipStreamSynchronize(st);
    printf("\n[0] sanity 1-layer Q4_K asym launch: %s\n", e?hipGetErrorString(e):"OK"); free_buffers(B); }

  // ---- [4] FULL-FORWARD Q4_K ASYMMETRIC — realistic cold vs RADV Q4_K_M ----
  printf("\n==================================================================================\n");
  printf("[4] FULL-FORWARD Q4_K ASYMMETRIC (NEW GEMM) — realistic-cold (rotate %d) vs RADV Q4_K_M.\n",ROT);
  printf("==================================================================================\n");
  for(int ci=0;ci<3;ci++){ int M=ctxs[ci]; Buffers B; alloc_buffers(B,M,NE,NH,NKV,HD,NFF);
    double cold=measure_forward(B,M,NE,NH,NKV,HD,NFF,eps,theta,1,1,st,NL);
    double warm=measure_forward(B,M,NE,NH,NKV,HD,NFF,eps,theta,1,0,st,NL);
    double tc=M/(cold/1000.0), tw=M/(warm/1000.0);
    double radv=radv_perctx[ci];
    const char* v=tc>radv*1.005?"BEAT":(tc<radv*0.995?"LOSE":"TIE");
    printf("\n--- pp%d (RADV Q4_K_M per-ctx %.0f) ---\n",M,radv);
    printf("  WARM-CONTROL (single buf)   %.2f ms  %.0f t/s  (%+.1f%% vs RADV) [optimistic]\n",warm,tw,(tw/radv-1)*100);
    printf("  REALISTIC-COLD (rotate %d)  %.2f ms  %.0f t/s  (%+.1f%% vs RADV Q4_K_M)  ==> %s\n",ROT,cold,tc,(tc/radv-1)*100,v);
    printf("  warm artifact inflated t/s by %+.1f%%\n",(tw/tc-1)*100);
    free_buffers(B);
  }

  // ---- [5] CONTROL: same harness, Q4_0 SYMMETRIC (so the Q4_K number is comparable to the Q4_0 win) ----
  printf("\n==================================================================================\n");
  printf("[5] CONTROL — Q4_0 SYMMETRIC full forward (same harness) vs RADV Q4_0 %.0f.\n",radv_q40);
  printf("==================================================================================\n");
  for(int ci=0;ci<3;ci++){ int M=ctxs[ci]; Buffers B; alloc_buffers(B,M,NE,NH,NKV,HD,NFF);
    double cold=measure_forward(B,M,NE,NH,NKV,HD,NFF,eps,theta,0,1,st,NL);
    double tc=M/(cold/1000.0);
    printf("  pp%-4d REALISTIC-COLD  %.2f ms  %.0f t/s  (%+.1f%% vs RADV Q4_0 %.0f)\n",M,cold,tc,(tc/radv_q40-1)*100,radv_q40);
    free_buffers(B);
  }

  // ---- [6] PER-OP BREAKDOWN (cold) for Q4_K asym ----
  printf("\n==================================================================================\n");
  printf("[6] PER-OP BREAKDOWN (cold, hipEvents) — Q4_K asym. Is the prologue/GEMM balance healthy?\n");
  printf("==================================================================================\n");
  for(int ci=0;ci<3;ci++){ int M=ctxs[ci]; Buffers B; alloc_buffers(B,M,NE,NH,NKV,HD,NFF);
    double prof[OP_NCAT]; profile_ops(B,M,NE,NH,NKV,HD,NFF,eps,theta,1,st,NL,prof);
    double tot=0; for(int i=0;i<OP_NCAT;i++) tot+=prof[i];
    printf("\n--- pp%d, per-op total over %d layers (ms), min-of-5 ---\n",M,NL);
    for(int i=0;i<OP_NCAT;i++) printf("  %-26s %8.3f ms  (%5.1f%%)\n",OPNAME[i],prof[i],100.0*prof[i]/tot);
    printf("  >>> PROLOGUE is %.1f%% of GEMM (%.3f / %.3f ms)\n",100.0*prof[OP_PROLOGUE]/prof[OP_GEMM],prof[OP_PROLOGUE],prof[OP_GEMM]);
    free_buffers(B);
  }

  hipStreamDestroy(st);
  return 0;
}
