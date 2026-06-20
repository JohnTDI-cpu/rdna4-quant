// gemm_pipe.hip — P1 TRUE multi-stage LDS software pipeline for INT8 WMMA GEMM on gfx1201/RDNA4.
//
//   GOAL (honest): break the HOT (L2-warm, pure-compute) occupancy-8 codegen ceiling that the forensic
//   measured at 59 TF (qkv) / 62 (gate) / 83 (down) = ~18-24% of the 339 TF int8 WMMA peak. This ceiling
//   is identical for every source quant => it is the gfx1201 occ-8 LATENCY wall, not a bandwidth limit.
//   If we break it, EVERY quant wins — critically Q8 (no bandwidth lever) which currently loses RADV Q8
//   (2849 t/s pp512) by ~9%.
//
//   THE LEVER (P1, PREFILL_OPTIMIZATION_PLAN.md): RDNA4 has NO async-copy. Load path =
//     global_load -> VGPR -> ds_store -> LDS -> ds_load -> VGPR -> WMMA.
//   The existing winner gemm_v3p<256,256,8,4,KC32,SWZ1> (gemm_rb_v3.hip, 192 VGPR / occ 8 / ~0 spill)
//   ALREADY double-buffers global->LDS (rA/rB register staging of tile N+1 overlapped with WMMA on N)
//   AND fragment-pipelines the ks-subtiles. So the COARSE P1 is present. What it does NOT do, and what
//   this file tests, is the TRUE multi-stage decoupling:
//     - move the ds_store of the prefetched tile EARLY (before/interleaved with the WMMA chain) so the
//       store latency overlaps the matrix ops instead of sitting between the dot and the __syncthreads.
//     - explicit split-counter scheduling: keep N loads in flight (loadcnt) across the WHOLE compute,
//       gate the ds_load only on dscnt, gate ds_store only on the matching loadcnt.
//     - a 3-stage variant: 3 LDS buffers, 2 register generations in flight (load N+2 while store N+1
//       while compute N). Costs more VGPR — measured here whether occupancy survives or it spills like
//       the native-read attempt (87 VGPR spill @256x256) and the deep pipeline already rejected in v3.
//
//   Build:  hipcc --offload-arch=gfx1201 -O3 gemm_pipe.hip -o gemm_pipe
//   Res:    add -Rpass-analysis=kernel-resource-usage
//   GPU:    HIP_VISIBLE_DEVICES=1 ; rocm-smi --setperflevel auto first.
//   Does NOT touch gemm_rb_v3.hip / gemm_w4a4.hip / full_forward_v3.hip (other processes use them).
//
//   Reference numbers to beat (measured 2026-06-20, this machine, HOT min-of-bursts):
//     v3p gate M=512  N=14336 K=4096 : 62.5 TF   |  down N=4096 K=14336 : 81.9 TF
//
//   ===========================================================================================
//   MEASURED VERDICT (gfx1201/R9700, ROCm 7.2.3, HIP_VISIBLE_DEVICES=1, perf=auto, 3 repeat runs):
//   THE 59-83 TF HOT OCCUPANCY-8 CEILING IS NOT STRUCTURAL — IT BREAKS. Winner = gemm_pipe2b
//   (2-stage decoupled pipeline with the ds_store INTERLEAVED after the first WMMA sub-tile).
//   185 VGPR, occ 8, 0 spill, 0 scratch, 32KB LDS, NRMSE 2.08e-4 PASS. SAME tile (256x256), SAME
//   occupancy as v3p — the win is pure scheduling, no register/occupancy cost.
//     gate  M=512 : 65.3-65.7 TF  vs v3p 60.8-62.3  (+5 to +8%)   [broke 62.5]
//     gate  M=2048: 79.2-79.8 TF  vs v3p 72.0-72.6  (+10%)
//     down  M=512 : 83.7-85.2 TF  vs v3p 78.6-79.8  (+6 to +7%)   [broke 83]
//     qkv   M=512 : 64.8 TF       vs v3p 59.3       (+9.2%)       [broke 59]
//     COLD down   : 79.9 TF       vs v3p 75.5       (+6%)         [survives cold]
//   5-shape AGG GEMM +6.3% => IMPLIED full-forward +5.6% (pp512 2584 -> 2730 t/s; closes most of the
//   ~9% RADV-Q8 2849 gap, and at M=2048 the +10% closes it entirely). EVERY shape improved.
//   WHY pipe2 (store-BEFORE-dot) won gate (+4%) but LOST down (-3%): front-loading the whole ds_store
//   contends with the first ds_loads, which hurts the many-K-block down shape. pipe2b moves the store
//   to AFTER ks=0's WMMA so it overlaps the dot TAIL (also clears the WMMA-RAW v_nop hazard, P4) — wins
//   BOTH short-K and long-K. pipe3 (3-buffer, 2 reg-gens) did NOT spill (183 VGPR/occ8) but did NOT
//   help (62/74 TF): the extra LDS buffer adds no overlap the 2-stage doesn't already extract — RDNA4
//   has no async DMA, so a third stage just sits idle. Small tiles (128x128 occ12, 128x256) REGRESSED
//   hard (36-52 TF): they lose N-reuse faster than the occupancy buys back, exactly as v3 found.
//   ROOT CAUSE the ceiling was beatable: v3p's store sat BETWEEN the dot and __syncthreads (serialized);
//   interleaving it into the dot is the missing overlap. The occ-8 latency wall is real, but there was
//   ~6-10% of un-hidden ds_store latency the codegen left on the table.
//   ===========================================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>
typedef int   i4  __attribute__((ext_vector_type(4)));
typedef int   v2i __attribute__((ext_vector_type(2)));
typedef int   v8i __attribute__((ext_vector_type(8)));
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
#define CK(x) do{hipError_t e=(x);if(e){printf("HIP err %s @ %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__);}}while(0)

// gfx12 split-counter waits (P1). N = #outstanding ops still allowed to be in flight.
template<int N> __device__ __forceinline__ void wait_loadN(){ asm volatile("s_wait_loadcnt %0"::"i"(N):"memory"); }
template<int N> __device__ __forceinline__ void wait_dsN(){   asm volatile("s_wait_dscnt   %0"::"i"(N):"memory"); }

// XOR swizzle (P2), identical bijection to gemm_rb_v3.hip swz16.
__device__ __forceinline__ int swz16(int row,int blk16,int KC){
  int nblk=KC/16; if(nblk<=1) return blk16;
  return blk16 ^ ((row>>1)&(nblk-1));
}

// ============================================================================
// V3P REFERENCE (verbatim structural copy of gemm_rb_v3.hip's winner) — kept in THIS binary so the new
//   pipelines are measured against the real baseline under identical L2 / clock / requant conditions.
//   Global->LDS double-buffer (rA/rB register staging) + fragment ks-pipeline + swz + scales-after.
// ============================================================================
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

// ============================================================================
// PIPE2 — TRUE 2-STAGE LDS software pipeline (decoupled store).
//   Key difference vs v3p: the ds_store of the prefetched tile is issued EARLY (right after the global
//   loads complete, BEFORE the WMMA chain), so the store-latency + the __syncthreads barrier overlap the
//   matrix ops. The fetch of ks=0 fragments is delayed until after the store-to-other-buffer so the
//   dscnt the ds_loads wait on does not include the producer store.
//   Order per k-block:
//     (1) issue global_load(tile N+1) -> rA/rB        [loadcnt grows]
//     (2) wait_loadN<0> ; ds_store(rA/rB -> buf^1)    [the EARLY store; dscnt grows]
//     (3) WMMA chain on buf (fragment ds_loads + dot) — overlaps the store latency of (2)
//     (4) scales-after (P3)
//     (5) __syncthreads ; buf^=1
//   The store now sits in front of the compute, so by the barrier the LDS write is already retiring.
//   SAME 2 LDS buffers, SAME register staging — no extra LDS, minimal extra VGPR (store lifetime moved).
// ============================================================================
template<int BM,int BN,int WMW,int WNW,int KC,int SWZ>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_pipe2(
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
    // (1) issue next-tile global loads
    if(kn<K){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kn+c];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rB[p]=*(const i4*)&wt[(size_t)(n_base+row)*K+kn+c];}} }
    // (2) EARLY decoupled store: wait the global loads, store into the OTHER buffer now. dscnt will be
    //     drained by the time the next __syncthreads lands — overlaps the WMMA chain below.
    if(kn<K){ wait_loadN<0>();
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    // (3) WMMA chain on the CURRENT buffer — runs while the (2) store retires.
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
    // (4) scales OUTSIDE the dot (P3)
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=a_scale[(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk];
    for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[mt][nt][j]*wsc[nt]*asc[mt][j];
    // (5) barrier; flip
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
}

// ============================================================================
// PIPE2B — 2-stage with the ds_store INTERLEAVED into the WMMA chain (P4 hazard-clear + overlap).
//   Same as pipe2 but instead of dumping the whole store before the dot, we hold the global loads in
//   rA/rB and issue the ds_store AFTER the first WMMA sub-tile, so the store traffic overlaps the *tail*
//   of the dot rather than competing at the front. Helps the long-K (down) shape where KSUB-many blocks
//   make the front-loaded store of pipe2 contend with the first ds_loads. Gate is short-K so this should
//   be a wash there. Store is split across the first store_at ks-iterations.
// ============================================================================
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

// ============================================================================
// PIPE3 — 3-STAGE LDS software pipeline (3 buffers, 2 register generations in flight).
//   Steady state per k-block i (processing buf=i%3):
//     - global_load(tile i+2)  -> rA/rB                 [loadcnt: tile i+2 in flight]
//     - ds_store(rA_prev -> buf (i+1)%3)                [store tile i+1, prefetched last iter]
//     - WMMA on buf i%3                                 [compute current; overlaps the store + loads]
//   Two register generations: rPrev (loaded iteration i-1, stored this iteration) and rCur (loaded now).
//   This deepens the load->store->compute decoupling so the global-load latency is hidden across TWO
//   k-blocks, not one. Cost: 3rd LDS buffer (+16KB => 48KB LDS, still <64KB) and a 2nd register
//   generation (rPrev) => more VGPR. WATCH for occupancy drop / spill (the deep pipeline rejected in v3
//   spilled 108 B/lane). If it spills at 256x256, the smaller-tile variants below carry it.
//   Prologue: load tile0 (done in shared-prologue store), load tile1 into rPrev, store nothing yet.
// ============================================================================
template<int BM,int BN,int WMW,int WNW,int KC,int SWZ>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_pipe3(
    const int8_t* __restrict__ act, const float* __restrict__ a_scale,
    const int8_t* __restrict__ wt,  const float* __restrict__ w_scale,
    __half* __restrict__ out, int M, int N, int K){
  constexpr int NWARP=WMW*WNW, NTHREAD=NWARP*32, WMT=BM/(WMW*16), WNT=BN/(WNW*16);
  constexpr int AV=BM*KC/16, BV=BN*KC/16, APT=(AV+NTHREAD-1)/NTHREAD, BPT=(BV+NTHREAD-1)/NTHREAD;
  constexpr int KSUB=KC/16;
  __shared__ int8_t As[3][BM][KC]; __shared__ int8_t Bs[3][BN][KC];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16, wm=warp/WNW, wn=warp%WNW;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/KC;
  float Rf[WMT][WNT][8];
  for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)for(int j=0;j<8;j++)Rf[a][b][j]=0.f;
  auto gload=[&](i4* rA,i4* rB,int kk){
    for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kk+c];}}
    for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rB[p]=*(const i4*)&wt[(size_t)(n_base+row)*K+kk+c];}} };
  auto sstore=[&](const i4* rA,const i4* rB,int b){
    for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[b][row][c]=rA[p];}}
    for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[b][row][c]=rB[p];}} };
  auto fetchA=[&](v2i Af[WMT],int b,int ks){
    #pragma unroll
    for(int mt=0;mt<WMT;mt++){int row=wm*WMT*16+mt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* ap=&As[b][row][bb*16+kg*8];Af[mt]=(v2i){*(const int*)ap,*(const int*)(ap+4)};} };
  auto fetchB=[&](v2i Bf[WNT],int b,int ks){
    #pragma unroll
    for(int nt=0;nt<WNT;nt++){int row=wn*WNT*16+nt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* bp=&Bs[b][row][bb*16+kg*8];Bf[nt]=(v2i){*(const int*)bp,*(const int*)(bp+4)};} };
  // PROLOGUE: store tile0 into buf0; prefetch tile1 into rPrev (in flight).
  { i4 r0A[APT],r0B[BPT]; gload(r0A,r0B,0); wait_loadN<0>(); sstore(r0A,r0B,0); }
  i4 rPrevA[APT],rPrevB[BPT]; int nb_k=K/KC;
  if(nb_k>1) gload(rPrevA,rPrevB,KC);   // tile1 in flight, will be stored in iteration 0
  __syncthreads();
  for(int blk=0;blk<nb_k;blk++){
    int cur=blk%3, nxt=(blk+1)%3;
    int knext=(blk+2)*KC;
    i4 rCurA[APT],rCurB[BPT];
    // (1) issue load for tile blk+2 (deep prefetch)
    if(blk+2<nb_k) gload(rCurA,rCurB,knext);
    // (2) store tile blk+1 (held in rPrev from last iter) into buffer nxt. Wait only the loads that
    //     produced rPrev — those were issued a full iteration ago, so loadcnt is already low.
    if(blk+1<nb_k){ wait_loadN<0>(); sstore(rPrevA,rPrevB,nxt); }
    // (3) WMMA on the current buffer; overlaps the (1) loads and (2) store latency.
    v8i cR[WMT][WNT]; for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)cR[a][b]=(v8i){};
    v2i Af0[WMT],Bf0[WNT],Af1[WMT],Bf1[WNT];
    fetchA(Af0,cur,0); fetchB(Bf0,cur,0);
    #pragma unroll
    for(int ks=0;ks<KSUB;ks++){
      v2i* Acur=(ks&1)?Af1:Af0; v2i* Bcur=(ks&1)?Bf1:Bf0;
      v2i* Anext=(ks&1)?Af0:Af1; v2i* Bnext=(ks&1)?Bf0:Bf1;
      if(ks+1<KSUB){ fetchA(Anext,cur,ks+1); fetchB(Bnext,cur,ks+1); }
      #pragma unroll
      for(int mt=0;mt<WMT;mt++)
        #pragma unroll
        for(int nt=0;nt<WNT;nt++)
          cR[mt][nt]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,Acur[mt],true,Bcur[nt],cR[mt][nt],false);
    }
    // (4) scales OUTSIDE the dot (P3)
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=a_scale[(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk];
    for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[mt][nt][j]*wsc[nt]*asc[mt][j];
    // (5) advance register generation: rPrev <- rCur (tile blk+2 becomes next iter's "to store")
    #pragma unroll
    for(int p=0;p<APT;p++) rPrevA[p]=rCurA[p];
    #pragma unroll
    for(int p=0;p<BPT;p++) rPrevB[p]=rCurB[p];
    __syncthreads();
  }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
}

// ============================================================================
// CPU/GPU helpers (Q6_K dequant + per-KCB requant) — copied from gemm_rb_v3.hip.
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
template<int KCB>
__global__ void requant_kc(const int8_t* __restrict__ Wq, const float* __restrict__ Ws, int8_t* __restrict__ Wqk, float* __restrict__ Wsk, int N, int K){
  int idx=blockIdx.x*blockDim.x+threadIdx.x; int nbk=K/KCB; if(idx>=N*nbk)return;
  int n=idx/nbk, b=idx%nbk, nb16=K/16; float f[KCB], mx=0;
  for(int j=0;j<KCB;j++){int k=b*KCB+j; f[j]=(float)Wq[(size_t)n*K+k]*Ws[(size_t)n*nb16+k/16]; mx=fmaxf(mx,fabsf(f[j]));}
  float s=mx/127.f; if(s<1e-12f)s=1e-12f; Wsk[(size_t)n*nbk+b]=s;
  for(int j=0;j<KCB;j++)Wqk[(size_t)n*K+b*KCB+j]=(int8_t)lrintf(f[j]/s);
}
__global__ void requant_32(const int8_t* __restrict__ Wq, const float* __restrict__ Ws, int8_t* __restrict__ Wq32, float* __restrict__ Ws32, int N, int K){
  int idx=blockIdx.x*blockDim.x+threadIdx.x; int nb32=K/32; if(idx>=N*nb32)return;
  int n=idx/nb32, b=idx%nb32, nb16=K/16; float f[32], mx=0;
  for(int j=0;j<32;j++){int k=b*32+j; f[j]=(float)Wq[(size_t)n*K+k]*Ws[(size_t)n*nb16+k/16]; mx=fmaxf(mx,fabsf(f[j]));}
  float s=mx/127.f; if(s<1e-12f)s=1e-12f; Ws32[(size_t)n*nb32+b]=s;
  for(int j=0;j<32;j++)Wq32[(size_t)n*K+b*32+j]=(int8_t)lrintf(f[j]/s);
}

// ============================================================================
// CORRECTNESS — synthetic int8, int32-accum float ref, NRMSE<2e-3 (same harness as gemm_rb_v3).
// ============================================================================
template<int BM,int BN,int A2,int B2,int KC, typename F>
static double run_correctness(const char* name, F kern){
  // Size the problem to the tile so the grid launches at least 1x2 blocks (BM/BN can be 256).
  int M=BM,N=BN*2,K=512,nb=K/KC;
  srand(11);
  std::vector<int8_t> Wq(N*K); for(auto&x:Wq)x=(int8_t)(rand()%255-127);
  std::vector<float>  Ws(N*nb); for(auto&x:Ws)x=(0.5f+(rand()%100)/100.f)*0.01f;
  std::vector<int8_t> Aq(M*K); for(auto&x:Aq)x=(int8_t)(rand()%255-127);
  std::vector<float>  As(M*nb); for(auto&x:As)x=(0.5f+(rand()%100)/100.f)*0.02f;
  std::vector<float> Oref(M*N,0);
  for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double s=0;
    for(int blk=0;blk<nb;blk++){ long acc=0; for(int kk=0;kk<KC;kk++){int k=blk*KC+kk; acc+=(long)Aq[m*K+k]*Wq[n*K+k];}
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
// SPEED — HOT (L2-warm, min-of-bursts) AND COLD (32MB L2 thrash + rotated weight copies).
// ============================================================================
template<int BM,int BN,int A2,int B2,int KC, typename F>
static double bench_hot(F kern,int M,int N,int K,int reps,double* out_ms){
  hipGetLastError();
  int nblk=K/256,nb16=K/16,nbk=K/KC;
  block_q6_K* dW;int8_t*dWq,*dWqk,*dAq;float*dWs,*dWsk,*dAs;__half*dO;
  CK(hipMalloc(&dW,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMalloc(&dWq,(size_t)N*K));CK(hipMalloc(&dWs,(size_t)N*nb16*4));
  CK(hipMalloc(&dWqk,(size_t)N*K));CK(hipMalloc(&dWsk,(size_t)N*nbk*4));CK(hipMalloc(&dAq,(size_t)M*K));CK(hipMalloc(&dAs,(size_t)M*nbk*4));CK(hipMalloc(&dO,(size_t)M*N*2));
  CK(hipMemset(dW,1,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMemset(dAq,1,(size_t)M*K));CK(hipMemset(dAs,1,(size_t)M*nbk*4));
  dequant_q6k<<<(N*nblk+127)/128,128>>>(dW,dWq,dWs,N,K);
  for(int w=0;w<3;w++)requant_kc<KC><<<(N*nbk+127)/128,128>>>(dWq,dWs,dWqk,dWsk,N,K);
  CK(hipDeviceSynchronize());
  dim3 g(N/BN,M/BM),b(A2*B2*32);
  for(int w=0;w<10;w++)kern<<<g,b>>>(dAq,dAs,dWqk,dWsk,dO,M,N,K);
  hipError_t le=hipGetLastError(); hipError_t se=hipDeviceSynchronize();
  double ms; if(le||se){ ms=-1; *out_ms=-1; }
  else { const int BURST=8; double best=1e30;
    for(int r=0;r<reps;r++){ double t0=now(); for(int i=0;i<BURST;i++)kern<<<g,b>>>(dAq,dAs,dWqk,dWsk,dO,M,N,K); hipDeviceSynchronize();
      double m=(now()-t0)/BURST; if(m<best)best=m; }
    ms=best; *out_ms=ms; }
  hipFree(dW);hipFree(dWq);hipFree(dWs);hipFree(dWqk);hipFree(dWsk);hipFree(dAq);hipFree(dAs);hipFree(dO);
  return ms<0 ? -1.0 : 2.0*M*N*K/1e12/(ms/1e3);
}
__global__ void thrash(int8_t* p, size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]^=1; }
template<int BM,int BN,int A2,int B2,int KC, typename F>
static double bench_cold(F kern,int M,int N,int K,int reps,double* out_ms){
  hipGetLastError();
  int nblk=K/256,nb16=K/16,nbk=K/KC; const int NBUF=3;
  block_q6_K* dW;int8_t*dWq,*dWqk[NBUF],*dAq;float*dWs,*dWsk[NBUF],*dAs;__half*dO;
  CK(hipMalloc(&dW,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMalloc(&dWq,(size_t)N*K));CK(hipMalloc(&dWs,(size_t)N*nb16*4));
  for(int bb=0;bb<NBUF;bb++){CK(hipMalloc(&dWqk[bb],(size_t)N*K));CK(hipMalloc(&dWsk[bb],(size_t)N*nbk*4));}
  CK(hipMalloc(&dAq,(size_t)M*K));CK(hipMalloc(&dAs,(size_t)M*nbk*4));CK(hipMalloc(&dO,(size_t)M*N*2));
  CK(hipMemset(dW,1,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMemset(dAq,1,(size_t)M*K));CK(hipMemset(dAs,1,(size_t)M*nbk*4));
  dequant_q6k<<<(N*nblk+127)/128,128>>>(dW,dWq,dWs,N,K);
  for(int bb=0;bb<NBUF;bb++)for(int w=0;w<2;w++)requant_kc<KC><<<(N*nbk+127)/128,128>>>(dWq,dWs,dWqk[bb],dWsk[bb],N,K);
  CK(hipDeviceSynchronize());
  size_t TH=32u*1024*1024; int8_t* dT; CK(hipMalloc(&dT,TH));
  dim3 g(N/BN,M/BM),b(A2*B2*32);
  for(int w=0;w<4;w++)kern<<<g,b>>>(dAq,dAs,dWqk[0],dWsk[0],dO,M,N,K);
  hipError_t le=hipGetLastError(); hipError_t se=hipDeviceSynchronize();
  if(le||se){ *out_ms=-1;
    hipFree(dT);hipFree(dW);hipFree(dWq);hipFree(dWs);for(int bb=0;bb<NBUF;bb++){hipFree(dWqk[bb]);hipFree(dWsk[bb]);}hipFree(dAq);hipFree(dAs);hipFree(dO);
    return -1.0; }
  double t0=now();
  for(int i=0;i<reps;i++){ int bi=i%NBUF;
    thrash<<<(TH+255)/256,256>>>(dT,TH);
    kern<<<g,b>>>(dAq,dAs,dWqk[bi],dWsk[bi],dO,M,N,K); }
  CK(hipDeviceSynchronize());
  double tot=now()-t0;
  double t1=now(); for(int i=0;i<reps;i++)thrash<<<(TH+255)/256,256>>>(dT,TH); CK(hipDeviceSynchronize());
  double thr=now()-t1;
  double ms=(tot-thr)/reps; *out_ms=ms;
  hipFree(dT);hipFree(dW);hipFree(dWq);hipFree(dWs);for(int bb=0;bb<NBUF;bb++){hipFree(dWqk[bb]);hipFree(dWsk[bb]);}hipFree(dAq);hipFree(dAs);hipFree(dO);
  return 2.0*M*N*K/1e12/(ms/1e3);
}

int main(int argc, char** argv){
  printf("=== gemm_pipe: P1 TRUE multi-stage LDS software pipeline (gfx1201/RDNA4) ===\n");
  printf("Baseline to beat (HOT min-burst, this machine): gate 62.5 TF, down 81.9 TF.\n");
  printf("HOT WMMA int8 peak = 339 TF. occ-8 ceiling = ~62-83 TF. Q: does the pipeline break it?\n\n");

  printf("[1] CORRECTNESS (synthetic int8, vs int32-accum float ref, NRMSE<2e-3):\n");
  run_correctness<256,256,8,4,32>("v3p ref 256x256 swz",     gemm_v3p  <256,256,8,4,32,1>);
  run_correctness<256,256,8,4,32>("pipe2 256x256 swz",       gemm_pipe2<256,256,8,4,32,1>);
  run_correctness<256,256,8,4,32>("pipe2b 256x256 swz",      gemm_pipe2b<256,256,8,4,32,1>);
  run_correctness<256,256,8,4,32>("pipe2 256x256 noswz",     gemm_pipe2<256,256,8,4,32,0>);
  run_correctness<128,256,4,4,32>("pipe2 128x256 swz",       gemm_pipe2<128,256,4,4,32,1>);
  run_correctness<128,128,4,4,32>("pipe2 128x128 swz",       gemm_pipe2<128,128,4,4,32,1>);
  run_correctness<256,256,8,4,32>("pipe3 256x256 swz",       gemm_pipe3<256,256,8,4,32,1>);
  run_correctness<256,256,8,4,32>("pipe3 256x256 noswz",     gemm_pipe3<256,256,8,4,32,0>);
  run_correctness<128,256,4,4,32>("pipe3 128x256 swz",       gemm_pipe3<128,256,4,4,32,1>);
  run_correctness<128,128,4,4,32>("pipe3 128x128 swz",       gemm_pipe3<128,128,4,4,32,1>);

  // ------------------------------------------------------------------
  // [2] HOT GATE M=512 N=14336 K=4096 — the core ceiling test. 40 reps min-of-burst.
  // ------------------------------------------------------------------
  printf("\n[2] HOT GATE M=512 N=14336 K=4096 (min-of-8-burst, 40 reps) — does it break 62.5 TF?\n");
  { const int M=512,N=14336,K=4096,R=40; double ms;
    struct Row{const char* name;double tf;}; std::vector<Row> rows;
    #define BH(label,V,bm,bn,a2,b2,kc,...) { double tf=bench_hot<bm,bn,a2,b2,kc>(V<bm,bn,a2,b2,kc,##__VA_ARGS__>,M,N,K,R,&ms); \
       if(tf<0) printf("  %-30s --- INVALID (launch/LDS fail)\n",label); \
       else { printf("  %-30s %.3f ms  %6.1f TFLOPS\n",label,ms,tf); rows.push_back({label,tf}); } }
    BH("v3p  256x256 swz (BASE)",   gemm_v3p,   256,256,8,4,32,1)
    BH("pipe2 256x256 swz",         gemm_pipe2, 256,256,8,4,32,1)
    BH("pipe2b 256x256 swz",        gemm_pipe2b,256,256,8,4,32,1)
    BH("pipe2 256x256 noswz",       gemm_pipe2, 256,256,8,4,32,0)
    BH("pipe2 128x128 swz",         gemm_pipe2, 128,128,4,4,32,1)
    BH("pipe3 256x256 swz",         gemm_pipe3, 256,256,8,4,32,1)
    #undef BH
    double best=62.5; const char* bnm="v3p BASE"; for(auto&r:rows)if(r.tf>best){best=r.tf;bnm=r.name;}
    printf("  >>> BEST gate HOT: %s = %.1f TF  %s\n",bnm,best, best>62.6?"*** BROKE THE CEILING ***":"(ceiling holds)");
  }

  // ------------------------------------------------------------------
  // [3] HOT GATE M=2048 (compute-heavy, more K-blocks reuse) — same kernels.
  // ------------------------------------------------------------------
  printf("\n[3] HOT GATE M=2048 N=14336 K=4096 (min-of-8-burst, 20 reps):\n");
  { const int M=2048,N=14336,K=4096,R=20; double ms;
    #define BH(label,V,bm,bn,a2,b2,kc,...) { double tf=bench_hot<bm,bn,a2,b2,kc>(V<bm,bn,a2,b2,kc,##__VA_ARGS__>,M,N,K,R,&ms); \
       if(tf<0) printf("  %-30s --- INVALID\n",label); else printf("  %-30s %.3f ms  %6.1f TFLOPS\n",label,ms,tf); }
    BH("v3p  256x256 swz (BASE)",   gemm_v3p,   256,256,8,4,32,1)
    BH("pipe2 256x256 swz",         gemm_pipe2, 256,256,8,4,32,1)
    BH("pipe2b 256x256 swz",        gemm_pipe2b,256,256,8,4,32,1)
    BH("pipe3 256x256 swz",         gemm_pipe3, 256,256,8,4,32,1)
    #undef BH
  }

  // ------------------------------------------------------------------
  // [4] HOT DOWN M=512 N=4096 K=14336 (long-K, where fragment pipeline already paid off to 81.9 TF).
  // ------------------------------------------------------------------
  printf("\n[4] HOT DOWN M=512 N=4096 K=14336 (long-K, min-of-8-burst, 30 reps) — does it break 81.9 TF?\n");
  { const int M=512,N=4096,K=14336,R=30; double ms;
    struct Row{const char* name;double tf;}; std::vector<Row> rows;
    #define BH(label,V,bm,bn,a2,b2,kc,...) { double tf=bench_hot<bm,bn,a2,b2,kc>(V<bm,bn,a2,b2,kc,##__VA_ARGS__>,M,N,K,R,&ms); \
       if(tf<0) printf("  %-30s --- INVALID\n",label); else { printf("  %-30s %.3f ms  %6.1f TFLOPS\n",label,ms,tf); rows.push_back({label,tf}); } }
    BH("v3p  256x256 swz (BASE)",   gemm_v3p,   256,256,8,4,32,1)
    BH("pipe2 256x256 swz",         gemm_pipe2, 256,256,8,4,32,1)
    BH("pipe2b 256x256 swz",        gemm_pipe2b,256,256,8,4,32,1)
    BH("pipe3 256x256 swz",         gemm_pipe3, 256,256,8,4,32,1)
    #undef BH
    double best=81.9; const char* bnm="v3p BASE"; for(auto&r:rows)if(r.tf>best){best=r.tf;bnm=r.name;}
    printf("  >>> BEST down HOT: %s = %.1f TF  %s\n",bnm,best, best>82.0?"*** BROKE THE CEILING ***":"(ceiling holds)");
  }

  // ------------------------------------------------------------------
  // [5] COLD gate + down — only if a HOT winner emerged, to confirm it survives real L2 pressure.
  // ------------------------------------------------------------------
  printf("\n[5] COLD (32MB L2 thrash) gate + down — sanity that hot winners survive cold:\n");
  { double ms;
    #define BC(label,V,bm,bn,a2,b2,kc,Ns,Ks,R,...) { double tf=bench_cold<bm,bn,a2,b2,kc>(V<bm,bn,a2,b2,kc,##__VA_ARGS__>,512,Ns,Ks,R,&ms); \
       if(tf<0) printf("  %-30s --- INVALID\n",label); else printf("  %-30s %.3f ms  %6.1f TFLOPS\n",label,ms,tf); }
    BC("v3p  gate (BASE)",  gemm_v3p,   256,256,8,4,32,14336,4096,40,1)
    BC("pipe2 gate",        gemm_pipe2, 256,256,8,4,32,14336,4096,40,1)
    BC("pipe2b gate",       gemm_pipe2b,256,256,8,4,32,14336,4096,40,1)
    BC("v3p  down (BASE)",  gemm_v3p,   256,256,8,4,32, 4096,14336,40,1)
    BC("pipe2 down",        gemm_pipe2, 256,256,8,4,32, 4096,14336,40,1)
    BC("pipe2b down",       gemm_pipe2b,256,256,8,4,32, 4096,14336,40,1)
    #undef BC
  }

  // ------------------------------------------------------------------
  // [6] 5-SHAPE AGGREGATE M=512 -> implied full-forward t/s. winner gemm_pipe2b vs v3p base.
  //     GEMM = 89% of the forward; this is the per-layer GEMM ms aggregated x 50 layers.
  // ------------------------------------------------------------------
  printf("\n[6] 5-SHAPE AGG M=512 (HOT) — implied per-layer GEMM ms (v3p base vs pipe2b):\n");
  { const int NL=50; const int RW=12; double tb=0,tp=0;
    struct S{const char* nm;int N,K;};
    S shapes[5]={{"qkv",6144,4096},{"o",4096,4096},{"gate",14336,4096},{"up",14336,4096},{"down",4096,14336}};
    for(int s=0;s<5;s++){ double msb,msp; int Ns=shapes[s].N,Ks=shapes[s].K;
      double tfb=bench_hot<256,256,8,4,32>(gemm_v3p  <256,256,8,4,32,1>,512,Ns,Ks,RW,&msb);
      double tfp=bench_hot<256,256,8,4,32>(gemm_pipe2b<256,256,8,4,32,1>,512,Ns,Ks,RW,&msp);
      tb+=msb; tp+=msp;
      printf("  %-5s N=%-5d K=%-5d: v3p %.3f ms %5.1f TF | pipe2b %.3f ms %5.1f TF (%+.1f%%)\n",
        shapes[s].nm,Ns,Ks,msb,tfb,msp,tfp,(tfp/tfb-1)*100); }
    double tsb=512.0/(tb*NL/1000.0), tsp=512.0/(tp*NL/1000.0);
    double gemm_gain=tb/tp-1.0;            // per-GEMM aggregate speedup
    double fwd_gain=0.89*gemm_gain;        // GEMM is 89% of the forward
    printf("  AGG 50L: v3p %.2f ms (%.0f t/s) | pipe2b %.2f ms (%.0f t/s)  GEMM %+.1f%%\n",
      tb*NL,tsb,tp*NL,tsp,gemm_gain*100);
    printf("  IMPLIED FULL-FWD (GEMM=89%%): pp512 2584 -> %.0f t/s (%+.1f%% forward)\n",
      2584.0*(1.0+fwd_gain), fwd_gain*100);
  }
  return 0;
}
