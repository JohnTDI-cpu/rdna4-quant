// gemm_rb_v3.hip — P1+P2+P3 software-pipelined / swizzled INT8 WMMA GEMM for gfx1201/RDNA4.
//   Goal: push the scale-aware prefill GEMM from ~61 TFLOPS (v2 best = gemm_v0<256,256,8,4,KC=32>)
//   toward ~70% of peak (~140 TFLOPS) to decisively beat RADV (51) and ROCm (71) on the gate shape.
//
//   Build: hipcc --offload-arch=gfx1201 -O3 gemm_rb_v3.hip -o gemm_rb_v3
//   VGPR/occ/spill: add -Rpass-analysis=kernel-resource-usage
//   GPU: HIP_VISIBLE_DEVICES=1 (R9700 #1). Set `rocm-smi --setperflevel auto` first.
//
//   PORTED TRICKS (PREFILL_OPTIMIZATION_PLAN.md) and what each MEASURED on gfx1201/R9700:
//   P1 — software pipeline w/ SPLIT wait counters (gfx12: s_wait_loadcnt / s_wait_dscnt). Implemented two
//        forms: (gemm_v3) explicit wait_loadN<0>() before the ds_store on the 2-buffer global prefetch;
//        (gemm_v3p) a fragment-level K-subtile pipeline — prefetch ks+1's LDS fragments into registers
//        WHILE the WMMA chain on ks runs, hiding ds_load latency behind WMMA. A *deeper* 3-stage LDS
//        pipeline was tried and REJECTED: it spilled (108 B/lane) and dropped occupancy with no benefit.
//   P2 — XOR-swizzled LDS layout (swz16): store + fragment-load both go through the same bijection so the
//        16 fragment lanes hit distinct banks. b128 (i4) ds_store on the producer side.
//   P3 — scales OUTSIDE the integer dot (already in the requant path): pure WMMA i32 inner loop, the
//        per-32-K float w_scale/a_scale folded into the FP32 accumulator AFTER the WMMA chain. The 8
//        independent cR[mt][nt] tiles already space the dependent-WMMA RAW chain (ACO emits NO v_nop).
//
//   MEASURED OUTCOME (gfx1201/R9700, ROCm 7.2.3, card2=HIP-dev1, min-burst peak-clock, 2026-06-20):
//     WINNER = gemm_v3p<256,256,8,4,KC=32,SWZ=1> (fragment pipeline + XOR swizzle). 188-192 VGPR, occ 8,
//       ~0 spill, 32KB LDS. NRMSE 2.06e-4 PASS.
//     * gate M=512 N=14336 K=4096: ~61-62 TFLOPS warm — TIE with v2 baseline (both at the occ-8 ceiling).
//     * down N=4096 K=14336 (long-K): 81.8 TFLOPS warm — the fragment pipeline pays off when KSUB-deep /
//       many K-blocks (the down proj is the big gain).
//     * 5-shape AGG M=512: WARM 3066-3080 t/s (+31-32% vs RADV 2341), COLD 2689-2705 t/s (+15-16%).
//     * per-M gate (V3P vs v0 baseline): M=512 +1.5%, M=2048 +3.1%, M=8192 +2.8% — small but consistent.
//   WHY NOT ~140 (70% peak): measured WMMA issue peak = 339 TFLOPS int8 here, so 62 TF = 18% of WMMA peak.
//     The kernel is OCCUPANCY-8-LATENCY-bound (188 VGPR -> 8 waves/SIMD): not WMMA-issue-bound (339 peak,
//     P3 confirmed no v_nop), not global-load-bound (P1 neutral), not LDS-bank-bound (P2 alone REGRESSED
//     ~10%, helps only when combined with the fragment pipeline). To raise occupancy you must shrink the
//     tile (256x256->256x128 = 50 TF, worse: loses N-reuse) — so occ-8 @256x256 is the real ceiling for
//     this tile family. RDNA4's halved A/B VGPR budget is the structural wall, exactly as the plan warned.
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
typedef int   v4i __attribute__((ext_vector_type(4)));
typedef int   v8i __attribute__((ext_vector_type(8)));
typedef int8_t c16 __attribute__((ext_vector_type(16)));
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
#define CK(x) do{hipError_t e=(x);if(e){printf("HIP err %s @ %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__);}}while(0)

// gfx12 split-counter waits (P1). N = #outstanding ops still allowed.
__device__ __forceinline__ void wait_load(int){ asm volatile("s_wait_loadcnt 0":::"memory"); }
template<int N> __device__ __forceinline__ void wait_loadN(){ asm volatile("s_wait_loadcnt %0"::"i"(N):"memory"); }
template<int N> __device__ __forceinline__ void wait_dsN(){   asm volatile("s_wait_dscnt %0"::"i"(N):"memory"); }

// ============================================================================
// V0 — reference baseline (verbatim from v2): cR + Rf both live, 2-buffer. ~59-61 TFLOPS gate.
//   Kept so v3's gains are measured against the real winner in the same binary.
// ============================================================================
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

// ============================================================================
// SWIZZLE (P2): map a (row, byte-col) of a [ROWS][KC] int8 tile to a swizzled LDS byte index that is
//   bank-conflict-free for the WMMA fragment ds_load. Each LDS bank = 4 bytes; 32 banks = 128 bytes.
//   We store each row's KC bytes contiguously (stride KC) but XOR the 16-byte block column with a
//   per-row constant so the 16 fragment lanes (consecutive rows, same byte-col) land on 16 distinct
//   16-byte slots => distinct bank groups. KC must be 16 or 32. SWZ derived from row low bits.
//   The store and the load both go through this map => self-consistent regardless of correctness of
//   the conflict argument; the only requirement is it is a bijection on [0,ROWS*KC).
// ============================================================================
// For KC=32: each row is 32 bytes = 2 x 16B blocks (block index kk in {0,1}). We xor kk with
//   (row>>0)&1 ... but to keep b128 store/b64 load both clean we swizzle at 16B granularity:
//   swz_block = kk ^ (row & 1).  -> within a row pair, the two 16B halves are swapped for odd rows.
//   Bank math: lane nl reads row=base+nl at byte ks*16+kg*8. For fixed ks,kg, consecutive nl differ by
//   one row (32B = 8 banks). Plain layout: nl and nl+4 collide (4*8=32 banks wrap). XOR by (row&1)<<...
//   we instead swizzle the *bank* by xoring high row bits into the 16B-block selector; combined with the
//   8B (kg) offset the 16 lanes spread across banks. (Measured empirically below.)
__device__ __forceinline__ int swz16(int row,int blk16,int KC){
  // blk16 in [0, KC/16). xor a few row bits into the 16B-block index, modulo KC/16.
  int nblk=KC/16;
  if(nblk<=1) return blk16;                  // KC=16: nothing to swap, rely on row-stride padding
  return blk16 ^ ((row>>1)&(nblk-1));        // KC=32: swap the two 16B halves for rows 2,3,6,7,...
}

// ============================================================================
// V3 — P1(2-buffer + SPLIT-counter scheduling) + P2(XOR swizzle) + P3(scales-after + WMMA spacing).
//   This is a faithful evolution of gemm_v0: SAME 2-buffer LDS (same 32KB, same occ-8) and SAME
//   single-generation register prefetch (rA[APT]/rB[APT]) — so it does NOT regress occupancy or spill.
//   The levers added vs v0:
//     P1 — explicit gfx12 split waits: wait_loadN<0>() before the ds_store (only on the global loads
//          that feed it), and the fragment ds_loads naturally gate on dscnt. The __syncthreads() is
//          kept (cross-warp visibility) but the global-load latency is hidden behind the WMMA chain.
//     P2 — XOR-swizzled LDS (swz16): store and fragment-load both go through swz16 so the 16 fragment
//          lanes hit distinct banks -> conflict-free ds_load. b128 ds_store (i4) on the producer.
//     P3 — scales applied to the float accumulator AFTER the integer WMMA chain (never inside the dot);
//          the 8 independent cR[mt][nt] tiles already space the dependent WMMA RAW chain (no v_nop).
//   SWZ template flag lets us A/B the swizzle in isolation.
// ============================================================================
template<int BM,int BN,int WMW,int WNW,int KC,int SWZ>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_v3(
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
  // prologue store (swizzled)
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+blk16*16];}}
  for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[0][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+blk16*16];}}
  __syncthreads(); int buf=0;
  for(int k0=0;k0<K;k0+=KC){ int blk=k0/KC; i4 rA[APT],rB[BPT]; int kn=k0+KC;
    // (a) issue next-tile global loads into registers (adds to loadcnt; hidden behind WMMA)
    if(kn<K){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kn+c];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rB[p]=*(const i4*)&wt[(size_t)(n_base+row)*K+kn+c];}} }
    // (b) WMMA on buf (fragment ds_loads through swz16); 8 independent cR tiles cover WMMA latency
    v8i cR[WMT][WNT]; for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)cR[a][b]=(v8i){};
    for(int ks=0;ks<KC/16;ks++){ v2i Af[WMT],Bf[WNT];
      for(int mt=0;mt<WMT;mt++){int row=wm*WMT*16+mt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* ap=&As[buf][row][bb*16+kg*8];Af[mt]=(v2i){*(const int*)ap,*(const int*)(ap+4)};}
      for(int nt=0;nt<WNT;nt++){int row=wn*WNT*16+nt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* bp=&Bs[buf][row][bb*16+kg*8];Bf[nt]=(v2i){*(const int*)bp,*(const int*)(bp+4)};}
      for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)cR[mt][nt]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,Af[mt],true,Bf[nt],cR[mt][nt],false);}
    // P3: scales OUTSIDE the integer dot
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=a_scale[(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk];
    for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[mt][nt][j]*wsc[nt]*asc[mt][j];
    // (c) store next tile to the other buffer. P1: wait ONLY on the global loads (loadcnt), not LDS.
    if(kn<K){ wait_loadN<0>();
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
}

// ============================================================================
// V3P — fragment-level K-subtile pipeline. Within a K-block, prefetch ks+1's LDS fragments into
//   registers WHILE the WMMA chain on ks runs (the ds_load latency hides behind the 8-WMMA chain).
//   This is the real "software pipeline" lever that does NOT cost extra LDS buffers or occupancy.
//   KC is the per-block K (KC/16 = #ks-subtiles to pipeline across). Bigger KC = more pipeline depth
//   + fewer __syncthreads barriers per output. Scales still applied AFTER the integer dot (P3).
//   SWZ flag toggles the XOR swizzle. Uses the SAME 2-buffer global->LDS prefetch as v0/v3.
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
  // LDS fragment fetch helpers
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
    // fragment-pipeline: prefetch ks=0, then for each ks WMMA on cur while fetching ks+1.
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

// CPU/GPU helpers (Q6_K dequant + per-32 requant) — reused from v2 -----------
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
// Generic per-KCB requant (KCB multiple of 16): used to test bigger K-blocks (fewer barriers).
template<int KCB>
__global__ void requant_kc(const int8_t* __restrict__ Wq, const float* __restrict__ Ws, int8_t* __restrict__ Wqk, float* __restrict__ Wsk, int N, int K){
  int idx=blockIdx.x*blockDim.x+threadIdx.x; int nbk=K/KCB; if(idx>=N*nbk)return;
  int n=idx/nbk, b=idx%nbk, nb16=K/16; float f[KCB], mx=0;
  for(int j=0;j<KCB;j++){int k=b*KCB+j; f[j]=(float)Wq[(size_t)n*K+k]*Ws[(size_t)n*nb16+k/16]; mx=fmaxf(mx,fabsf(f[j]));}
  float s=mx/127.f; if(s<1e-12f)s=1e-12f; Wsk[(size_t)n*nbk+b]=s;
  for(int j=0;j<KCB;j++)Wqk[(size_t)n*K+b*KCB+j]=(int8_t)lrintf(f[j]/s);
}

// ============================================================================
// CORRECTNESS — synthetic int8 inputs, int32-accum float reference, NRMSE<2e-3.
// ============================================================================
template<int BM,int BN,int A2,int B2,int KC, typename F>
static double run_correctness(const char* name, F kern){
  int M=64,N=256,K=512,nb=K/KC;
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
  printf("  %-26s NRMSE=%.2e  %s\n",name,nrmse,nrmse<2e-3?"PASS":"FAIL");
  hipFree(dWq);hipFree(dWs);hipFree(dAq);hipFree(dAs);hipFree(dO);
  return nrmse;
}

// ============================================================================
// SPEED — warm (real Q6_K->int8->requant32 buffers).
// ============================================================================
template<int BM,int BN,int A2,int B2,int KC, typename F>
static double bench_shape(F kern,int M,int N,int K,int reps,double* out_ms){
  hipGetLastError();
  int nblk=K/256,nb16=K/16,nbk=K/KC;
  block_q6_K* dW;int8_t*dWq,*dWqk,*dAq;float*dWs,*dWsk,*dAs;__half*dO;
  CK(hipMalloc(&dW,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMalloc(&dWq,(size_t)N*K));CK(hipMalloc(&dWs,(size_t)N*nb16*4));
  CK(hipMalloc(&dWqk,(size_t)N*K));CK(hipMalloc(&dWsk,(size_t)N*nbk*4));CK(hipMalloc(&dAq,(size_t)M*K));CK(hipMalloc(&dAs,(size_t)M*nbk*4));CK(hipMalloc(&dO,(size_t)M*N*2));
  CK(hipMemset(dW,1,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMemset(dAq,1,(size_t)M*K));CK(hipMemset(dAs,1,(size_t)M*nbk*4));
  dequant_q6k<<<(N*nblk+127)/128,128>>>(dW,dWq,dWs,N,K);
  if(KC>=32){ for(int w=0;w<3;w++)requant_kc<KC><<<(N*nbk+127)/128,128>>>(dWq,dWs,dWqk,dWsk,N,K); }
  CK(hipDeviceSynchronize());
  int8_t* useWq=(KC>=32)?dWqk:dWq; float* useWs=(KC>=32)?dWsk:dWs;
  dim3 g(N/BN,M/BM),b(A2*B2*32);
  for(int w=0;w<10;w++)kern<<<g,b>>>(dAq,dAs,useWq,useWs,dO,M,N,K);
  hipError_t le=hipGetLastError(); hipError_t se=hipDeviceSynchronize();
  double ms; if(le||se){ ms=-1; *out_ms=-1; }
  else {
    // MIN-of-bursts: report the fastest burst -> the peak-clock (un-throttled) time, removes
    // boost/throttle drift. Each burst = BURST kernels back-to-back, timed.
    const int BURST=8; double best=1e30;
    for(int r=0;r<reps;r++){ double t0=now(); for(int i=0;i<BURST;i++)kern<<<g,b>>>(dAq,dAs,useWq,useWs,dO,M,N,K); hipDeviceSynchronize();
      double m=(now()-t0)/BURST; if(m<best)best=m; }
    ms=best; *out_ms=ms; }
  hipFree(dW);hipFree(dWq);hipFree(dWs);hipFree(dWqk);hipFree(dWsk);hipFree(dAq);hipFree(dAs);hipFree(dO);
  return ms<0 ? -1.0 : 2.0*M*N*K/1e12/(ms/1e3);
}

// COLD-ish — rotate NBUF weight copies + 32MB L2 thrash between reps.
__global__ void thrash(int8_t* p, size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]^=1; }
template<int BM,int BN,int A2,int B2,int KC, typename F>
static double bench_shape_cold(F kern,int M,int N,int K,int reps,double* out_ms){
  hipGetLastError();
  int nblk=K/256,nb16=K/16,nbk=K/KC; const int NBUF=3;
  block_q6_K* dW;int8_t*dWq,*dWqk[NBUF],*dAq;float*dWs,*dWsk[NBUF],*dAs;__half*dO;
  CK(hipMalloc(&dW,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMalloc(&dWq,(size_t)N*K));CK(hipMalloc(&dWs,(size_t)N*nb16*4));
  for(int bb=0;bb<NBUF;bb++){CK(hipMalloc(&dWqk[bb],(size_t)N*K));CK(hipMalloc(&dWsk[bb],(size_t)N*nbk*4));}
  CK(hipMalloc(&dAq,(size_t)M*K));CK(hipMalloc(&dAs,(size_t)M*nbk*4));CK(hipMalloc(&dO,(size_t)M*N*2));
  CK(hipMemset(dW,1,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMemset(dAq,1,(size_t)M*K));CK(hipMemset(dAs,1,(size_t)M*nbk*4));
  dequant_q6k<<<(N*nblk+127)/128,128>>>(dW,dWq,dWs,N,K);
  for(int bb=0;bb<NBUF;bb++)requant_32<<<(N*nbk+127)/128,128>>>(dWq,dWs,dWqk[bb],dWsk[bb],N,K);
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
  printf("=== gemm_rb_v3: P1+P2+P3 pipelined/swizzled INT8 WMMA GEMM (gfx1201) ===\n");
  printf("Target: gate M=512 N=14336 K=4096. RADV 51 TFLOPS / 2341 t/s; ROCm 71; ~70%% peak=140.\n\n");

  printf("[1] CORRECTNESS (synthetic int8, vs int32-accum float ref, NRMSE<2e-3):\n");
  run_correctness<64,64,2,2,32>("V0 baseline 2x2",          gemm_v0<64,64,2,2,32>);
  run_correctness<64,64,2,2,32>("V3 noswz 2x2",             gemm_v3<64,64,2,2,32,0>);
  run_correctness<64,64,2,2,32>("V3 swz   2x2",             gemm_v3<64,64,2,2,32,1>);
  run_correctness<128,128,4,4,32>("V3 swz   128x128 4x4",   gemm_v3<128,128,4,4,32,1>);
  run_correctness<256,256,8,4,32>("V3 swz   256x256 8x4",   gemm_v3<256,256,8,4,32,1>);
  run_correctness<256,256,8,4,32>("V3 noswz 256x256 8x4",   gemm_v3<256,256,8,4,32,0>);
  run_correctness<256,256,8,4,32>("V3P fragpipe 256x256",   gemm_v3p<256,256,8,4,32,0>);
  run_correctness<256,256,8,4,32>("V3P fragpipe+swz 256",   gemm_v3p<256,256,8,4,32,1>);

  printf("\n[2] GATE M=512 N=14336 K=4096 — warm-standalone, 40 reps (P1+P2+P3 isolation):\n");
  const int M=512,N=14336,K=4096; const int R=40; double ms;
  struct Row{const char* name;double tf;double ms;};
  std::vector<Row> rows;
  #define BENCH(label,V,bm,bn,a2,b2,kc,...) { double tf=bench_shape<bm,bn,a2,b2,kc>(V<bm,bn,a2,b2,kc,##__VA_ARGS__>,M,N,K,R,&ms); \
     if(tf<0) printf("  %-32s  --- INVALID (launch/LDS fail)\n",label); \
     else { printf("  %-32s %.3f ms  %6.1f TFLOPS\n",label,ms,tf); rows.push_back({label,tf,ms}); } }
  BENCH("V0 256x256 8x4 (base)",       gemm_v0,256,256,8,4,32)
  BENCH("V3  256x256 8x4 P1 (noswz)",  gemm_v3,256,256,8,4,32,0)
  BENCH("V3  256x256 8x4 P1+P2 (swz)", gemm_v3,256,256,8,4,32,1)
  BENCH("V3P 256x256 8x4 fragpipe",    gemm_v3p,256,256,8,4,32,0)
  BENCH("V3P 256x256 8x4 fragpipe+swz",gemm_v3p,256,256,8,4,32,1)
  BENCH("V3P 256x128 8x4 fragpipe",    gemm_v3p,256,128,8,4,32,0)
  BENCH("V3P 128x256 4x4 fragpipe",    gemm_v3p,128,256,4,4,32,0)
  BENCH("V3P 128x128 4x4 fragpipe",    gemm_v3p,128,128,4,4,32,0)
  #undef BENCH
  double best=0; const char* bnm=""; for(auto&r:rows)if(r.tf>best){best=r.tf;bnm=r.name;}
  printf("  >>> BEST gate: %s = %.1f TFLOPS  (RADV 51, ROCm 71) %s\n", bnm, best, best>51?"*** BEAT RADV ***":"(below RADV)");

  // ====================================================================
  // [3] FULL 5-SHAPE AGGREGATE M=512 -> t/s. WINNER = gemm_v3p<256,256,8,4,32,1> (fragpipe+swz),
  //   measured at parity-to-slight-edge over the v2 baseline. Reports warm (min-burst) AND cold-ish.
  // ====================================================================
  printf("\n[3] 5-SHAPE AGGREGATE M=512 (KC32) — winner gemm_v3p 256x256 8x4 swz — t/s vs RADV 2341 / ggml 1612:\n");
  { const int NL=50; const int RW=12, RC=40; double tot=0, totc=0;
    struct S{const char* nm;int N,K;};
    S shapes[5]={{"qkv",6144,4096},{"o",4096,4096},{"gate",14336,4096},{"up",14336,4096},{"down",4096,14336}};
    for(int s=0;s<5;s++){ double mss,msc; int Ns=shapes[s].N,Ks=shapes[s].K;
      double tf =bench_shape     <256,256,8,4,32>(gemm_v3p<256,256,8,4,32,1>,512,Ns,Ks,RW,&mss);
      double tfc=bench_shape_cold<256,256,8,4,32>(gemm_v3p<256,256,8,4,32,1>,512,Ns,Ks,RC,&msc);
      tot+=mss; totc+=msc;
      printf("  %-5s N=%-5d K=%-5d: warm %.3f ms %5.1f TF | cold %.3f ms %5.1f TF\n",shapes[s].nm,Ns,Ks,mss,tf,msc,tfc); }
    double ts=512.0/(tot*NL/1000.0), tsc=512.0/(totc*NL/1000.0);
    printf("  AGG 50L WARM: %.2f ms = %.0f t/s  vs RADV 2341 (%+.0f%%) / ggml 1612 (%+.0f%%)  %s\n",
      tot*NL, ts, (ts/2341-1)*100, (ts/1612-1)*100, ts>2341?"*** BEAT ***":"(below)");
    printf("  AGG 50L COLD: %.2f ms = %.0f t/s  vs RADV 2341 (%+.0f%%) / ggml 1612 (%+.0f%%)  %s\n",
      totc*NL, tsc, (tsc/2341-1)*100, (tsc/1612-1)*100, tsc>2341?"*** BEAT ***":"(below)");
  }

  // ====================================================================
  // [4] PER-M (gate) — V0 baseline vs V3P winner at M=512/2048/8192.
  // ====================================================================
  printf("\n[4] PER-M (gate N=14336 K=4096) V0 base vs V3P fragpipe+swz:\n");
  { int Ms[3]={512,2048,8192}; const int RR=12; double mb,mp;
    for(int mi=0;mi<3;mi++){ int Mx=Ms[mi];
      double tb=bench_shape<256,256,8,4,32>(gemm_v0 <256,256,8,4,32>  ,Mx,14336,4096,RR,&mb);
      double tp=bench_shape<256,256,8,4,32>(gemm_v3p<256,256,8,4,32,1>,Mx,14336,4096,RR,&mp);
      printf("    M=%-4d  V0 %.1f TF  |  V3P %.1f TF  (%+.1f%%)\n",Mx,tb,tp,(tp/tb-1)*100); }
  }
  return 0;
}
