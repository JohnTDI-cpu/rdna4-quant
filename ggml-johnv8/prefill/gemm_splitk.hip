// gemm_splitk.hip — FILL the under-occupied small-M int8 prefill GEMM (gfx1201/RDNA4/HIP).
//
//   THE ONLY LOSING PREFILL CELL: Q8 pp512 at -4.6% vs RADV (2849 t/s). Root cause (measured, see
//   gemm_pipe.hip): the winner gemm_pipe2b hits 72 TF at M=2048 but only 62 TF at M=512 — a 14% small-M
//   penalty. WHY: at M=512 the gate (N=14336, K=4096) with the 256x256 tile launches only
//   ceil(512/256) * ceil(14336/256) = 2*56 = 112 blocks; at ~8 warps/block ~896 waves vs the GPU's
//   ~2048 wave-slots (64 CU * 4 SIMD * 8 occ) => ~44% fill. The GPU is HALF-IDLE during each pp512 GEMM.
//   RADV's Q8 short-context prefill wins there — it fills (split-K / Stream-K). This file tries to fill it.
//
//   STRATEGIES (all reuse the pipe2b 2-stage LDS-store-interleave int8 GEMM, 185 VGPR / occ-8, scale-aware
//   per-32 fp32 accumulators, scales applied post-dot — COPIED, not modified, from gemm_pipe.hip):
//     (1) SPLIT-K: split K into KS chunks; grid gets a z-dim of KS so each (M-tile,N-tile,K-chunk) is its
//         own block => KS* more blocks => fills the GPU at M=512. Because the per-32 scales apply WITHIN
//         each K-chunk, every split-K partial is ALREADY a fully-scaled fp32 256x256 sub-tile. The KS
//         partials just SUM. Two reductions:
//           (1a) split-K WORKSPACE: each block writes its fp32 partial to gws[ks_id][M][N]; a small
//                separate reduction kernel sums KS and writes fp16 out. Bit-faithful (deterministic order).
//           (1b) split-K ATOMIC: each block atomicAdds its fp32 partial to a single fp32 gws[M][N]; a
//                final cast kernel writes fp16. (Atomic order non-deterministic but fp32-accumulate, so
//                NRMSE is tiny — verified.)
//         NOTE the prior split-K-ATOMIC refutation was at LARGE M (GPU already full => reduction cost, no
//         fill gain). At M=512 the fill gain is the hypothesis to test.
//     (2) BN=128 (keep BM=256): 2* the N-blocks => 2*56 -> per-M-tile 112 N-blocks => 224 blocks at M=512.
//         Less N-reuse, better fill. The prior 'smaller tile regresses' was the M-tile (BM=128); the
//         N-tile (BN=128) is a different trade — measured here AT M=512 specifically.
//
//   M-ADAPTIVE: the winner must NOT regress M=2048 (already full). We measure both M=512 and M=2048 for
//   every config; the full-forward dispatch picks split-K/BN128 only when M is small.
//
//   Build:  HIP_VISIBLE_DEVICES=1 hipcc --offload-arch=gfx1201 -O3 gemm_splitk.hip -o /tmp/gemm_splitk && \
//           rocm-smi --setperflevel auto && HIP_VISIBLE_DEVICES=1 /tmp/gemm_splitk
//   Does NOT touch gemm_pipe.hip / gemm_rb_v3.hip / full_forward_*.hip / gemm_w4a4.hip / gemm_q4k_asym.hip.
//
//   Baselines (gemm_pipe.hip, HOT min-of-burst, this machine): gate M=512 62-65 TF, gate M=2048 79 TF.
//
//   ===========================================================================================
//   MEASURED VERDICT (gfx1201/R9700, ROCm 7.2.3, HIP_VISIBLE_DEVICES=1, perf=auto):
//   THE DIAGNOSIS WAS WRONG ABOUT THE HARDWARE — M=512 IS NOT UNDER-OCCUPIED. NOTHING FILLS IT BECAUSE
//   THERE IS NOTHING TO FILL. RADV's small-M Q8 edge is NOT a fill edge; the gemm side is already full.
//   Every fill strategy REGRESSED, even at M=512:
//     gate M=512:  base 66.2 TF | BN=128 53.4 (-19%) | splitK-WS KS2 36.2 (-45%) KS4 35.0 (-47%)
//                                                     | splitK-ATOMIC KS2 20.0 (-70%) KS4 13.1 (-80%)
//     qkv  M=512:  base 62.8 | BN128 52.5 (-16%) | WS KS2 41.3 (-34%) | ATOMIC KS2 18.7 (-70%)
//     down M=512:  base 83.9 | BN128 54.3 (-35%) | WS KS2 46.0 (-45%) | ATOMIC KS2 32.7 (-61%)
//     M=2048 (already full): identical regression pattern, as expected.
//     COLD gate M=512: base 54.9 TF, every variant slower (35/33/19/13).
//   ROOT CAUSE the diagnosis missed: R9700 gfx1201 has 32 CUs, maxThreadsPerCU=2048 = EXACTLY 2 blocks of
//   1024 threads (the 256x256 tile is a 1024-thread block). Resident capacity = 32 CU * 2 blocks = 64
//   blocks. The "44% fill" estimate assumed 64 CU * 2048 wave-slots; the real machine is HALF that and the
//   1024-thread block saturates occupancy at 2/CU. So at M=512 the gate already launches 112 blocks vs 64
//   resident slots = 1.75x OVER-subscribed — the GPU is FULL, not half-idle. SMOKING GUN: down M=512 (only
//   32 blocks, "12% fill") hits 83.9 TF — the FASTEST cell, not the slowest. If occupancy were the limiter,
//   the lowest-block-count shape would be the slowest; it is the fastest. The real M=512<M=2048 gap (66 vs
//   83 on gate) is LATENCY-PER-BLOCK (LDS pipeline depth + WMMA-chain startup amortized over fewer K-blocks
//   at the short-K gate), which split-K makes WORSE (each chunk has even fewer K-blocks to amortize over,
//   plus a global-memory reduction round-trip). BN=128 halves N-reuse for no occupancy gain (already 2/CU).
//   CONCLUSION: RADV's Q8 pp512 -4.6% edge is NOT recoverable by GPU-fill on the GEMM — the GEMM is already
//   compute/latency-bound and full. It is a genuine structural edge (likely RADV's dispatch/scheduling, or
//   its FA2/epilogue path), not a small-M occupancy hole on our side. Full-forward wiring was NOT done
//   because no GEMM config lifted M=512 — wiring a regressing kernel would only slow pp512 further.
//
//   CONFIRMING CURVE (gate N=14336 K=4096, TF vs M, HOT): 256->58.9  512->68.0  768->70.0  1024->79.5
//   1536->79.4  2048->83.0  3072->84.1  4096->84.7. It RISES SMOOTHLY to a ~84 TF ceiling and is already
//   175% resident at M=512 (112 blocks / 64 slots). This is WAVE-QUANTIZATION (ragged tail wave: 64 full +
//   48 partial = 75% util on the last wave at M=512), NOT under-occupancy. Split-K multiplies blocks but
//   shrinks per-block work => the ragged-tail cost stays and per-block latency floor rises => regression.
//   The only fix would be MORE work per wave; split-K/BN=128 do the opposite. M=512 small-M penalty is the
//   tail wave, and it is already as good as a single tiled GEMM gets on 32 CUs.
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

template<int N> __device__ __forceinline__ void wait_loadN(){ asm volatile("s_wait_loadcnt %0"::"i"(N):"memory"); }

// XOR swizzle (P2), identical bijection to gemm_pipe.hip / gemm_rb_v3.hip swz16.
__device__ __forceinline__ int swz16(int row,int blk16,int KC){
  int nblk=KC/16; if(nblk<=1) return blk16;
  return blk16 ^ ((row>>1)&(nblk-1));
}

// ============================================================================
// PIPE2B — the measured winner, COPIED verbatim from gemm_pipe.hip. 2-stage LDS pipeline, ds_store
//   interleaved after the first WMMA sub-tile. Baseline for the M=512 fill experiments.
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
// SPLIT-K, WORKSPACE reduction (1a). IDENTICAL math/scheduling to pipe2b, but:
//   - grid.z = KS. Each block handles K-range [kid*Kc, (kid+1)*Kc), Kc = K/KS.
//   - the a_scale/w_scale per-32-block index is still the GLOBAL block index (blk = (k0)/KC within the
//     chunk maps to global (kid*Kc + k0)/KC). The per-32 scale applies WITHIN each K-chunk so the partial
//     tile this block computes is a fully-scaled fp32 256x256 sub-tile.
//   - epilogue writes the fp32 partial to gws[kid * (M*N) + m*N + n] (NO half cast — keep fp32 for the
//     bit-faithful reduction). A separate reduce kernel sums KS partials -> fp16 out.
//   KS must divide K and Kc must be a multiple of KC(=32). M=512,K=4096: KS up to 4 (Kc=1024) fine.
// ============================================================================
template<int BM,int BN,int WMW,int WNW,int KC,int SWZ>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_pipe2b_splitk(
    const int8_t* __restrict__ act, const float* __restrict__ a_scale,
    const int8_t* __restrict__ wt,  const float* __restrict__ w_scale,
    float* __restrict__ gws, int M, int N, int K, int KS){
  constexpr int NWARP=WMW*WNW, NTHREAD=NWARP*32, WMT=BM/(WMW*16), WNT=BN/(WNW*16);
  constexpr int AV=BM*KC/16, BV=BN*KC/16, APT=(AV+NTHREAD-1)/NTHREAD, BPT=(BV+NTHREAD-1)/NTHREAD;
  constexpr int KSUB=KC/16;
  int kid=blockIdx.z; int Kc=K/KS; int kbase=kid*Kc;       // this block's K-chunk
  __shared__ int8_t As[2][BM][KC]; __shared__ int8_t Bs[2][BN][KC];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16, wm=warp/WNW, wn=warp%WNW;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/KC;  // nb is GLOBAL #scale-blocks
  float Rf[WMT][WNT][8];
  for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)for(int j=0;j<8;j++)Rf[a][b][j]=0.f;
  // prologue load of the chunk's first KC tile (global K offset kbase)
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+kbase+blk16*16];}}
  for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[0][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+kbase+blk16*16];}}
  __syncthreads(); int buf=0;
  auto fetchA=[&](v2i Af[WMT],int b,int ks){
    #pragma unroll
    for(int mt=0;mt<WMT;mt++){int row=wm*WMT*16+mt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* ap=&As[b][row][bb*16+kg*8];Af[mt]=(v2i){*(const int*)ap,*(const int*)(ap+4)};} };
  auto fetchB=[&](v2i Bf[WNT],int b,int ks){
    #pragma unroll
    for(int nt=0;nt<WNT;nt++){int row=wn*WNT*16+nt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* bp=&Bs[b][row][bb*16+kg*8];Bf[nt]=(v2i){*(const int*)bp,*(const int*)(bp+4)};} };
  for(int k0=0;k0<Kc;k0+=KC){ int gk=kbase+k0; int blk=gk/KC; i4 rA[APT],rB[BPT]; int kn=gk+KC; bool last=(k0+KC>=Kc);
    if(!last){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kn+c];}}
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
      if(!stored && ks==0 && !last){ stored=true; wait_loadN<0>();
        for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
        for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    }
    if(!stored && !last){ wait_loadN<0>();
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
  // write fp32 partial to gws[kid][m][n]
  size_t plane=(size_t)M*N;
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){
    int m=m_base+wm*WMT*16+mt*16+kg*8+j; int n=n_base+wn*WNT*16+nt*16+nl;
    gws[(size_t)kid*plane+(size_t)m*N+n]=Rf[mt][nt][j]; }
}

// SPLIT-K, ATOMIC reduction (1b). Same as above but atomicAdd into a single fp32 plane gws[m][n].
//   gws MUST be zeroed before launch. A final cast kernel writes fp16. Non-deterministic add order, but
//   fp32 accumulate => NRMSE tiny.
template<int BM,int BN,int WMW,int WNW,int KC,int SWZ>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_pipe2b_splitk_atomic(
    const int8_t* __restrict__ act, const float* __restrict__ a_scale,
    const int8_t* __restrict__ wt,  const float* __restrict__ w_scale,
    float* __restrict__ gws, int M, int N, int K, int KS){
  constexpr int NWARP=WMW*WNW, NTHREAD=NWARP*32, WMT=BM/(WMW*16), WNT=BN/(WNW*16);
  constexpr int AV=BM*KC/16, BV=BN*KC/16, APT=(AV+NTHREAD-1)/NTHREAD, BPT=(BV+NTHREAD-1)/NTHREAD;
  constexpr int KSUB=KC/16;
  int kid=blockIdx.z; int Kc=K/KS; int kbase=kid*Kc;
  __shared__ int8_t As[2][BM][KC]; __shared__ int8_t Bs[2][BN][KC];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16, wm=warp/WNW, wn=warp%WNW;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/KC;
  float Rf[WMT][WNT][8];
  for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)for(int j=0;j<8;j++)Rf[a][b][j]=0.f;
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+kbase+blk16*16];}}
  for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[0][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+kbase+blk16*16];}}
  __syncthreads(); int buf=0;
  auto fetchA=[&](v2i Af[WMT],int b,int ks){
    #pragma unroll
    for(int mt=0;mt<WMT;mt++){int row=wm*WMT*16+mt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* ap=&As[b][row][bb*16+kg*8];Af[mt]=(v2i){*(const int*)ap,*(const int*)(ap+4)};} };
  auto fetchB=[&](v2i Bf[WNT],int b,int ks){
    #pragma unroll
    for(int nt=0;nt<WNT;nt++){int row=wn*WNT*16+nt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* bp=&Bs[b][row][bb*16+kg*8];Bf[nt]=(v2i){*(const int*)bp,*(const int*)(bp+4)};} };
  for(int k0=0;k0<Kc;k0+=KC){ int gk=kbase+k0; int blk=gk/KC; i4 rA[APT],rB[BPT]; int kn=gk+KC; bool last=(k0+KC>=Kc);
    if(!last){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kn+c];}}
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
      if(!stored && ks==0 && !last){ stored=true; wait_loadN<0>();
        for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
        for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    }
    if(!stored && !last){ wait_loadN<0>();
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
    int m=m_base+wm*WMT*16+mt*16+kg*8+j; int n=n_base+wn*WNT*16+nt*16+nl;
    atomicAdd(&gws[(size_t)m*N+n],Rf[mt][nt][j]); }
}

// Reduce KS fp32 planes -> fp16 out (deterministic order, for splitk workspace 1a).
__global__ void reduce_splitk(const float* __restrict__ gws, __half* __restrict__ out, int M, int N, int KS){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; size_t tot=(size_t)M*N; if(i>=tot)return;
  size_t plane=tot; float s=0.f;
  for(int k=0;k<KS;k++) s+=gws[(size_t)k*plane+i];
  out[i]=__float2half(s);
}
// Cast a single fp32 plane -> fp16 (for splitk atomic 1b).
__global__ void cast_f32_f16(const float* __restrict__ gws, __half* __restrict__ out, int M, int N){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)M*N)return; out[i]=__float2half(gws[i]);
}

// ============================================================================
// Q6_K dequant + per-KCB requant (copied from gemm_pipe.hip) — feed the int8 GEMMs realistic weights.
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

// ============================================================================
// CORRECTNESS — synthetic int8, int32-accum float ref, NRMSE<2e-3. Split-K variants must match the
//   plain pipe2b epilogue bit-faithfully (fp32 partials sum to the same value modulo fp rounding).
// ============================================================================
static double ref_nrmse(const std::vector<__half>& hO,const std::vector<float>& Oref){
  double e=0,nn=0; for(size_t i=0;i<Oref.size();i++){float v=__half2float(hO[i]);e+=(v-Oref[i])*(v-Oref[i]);nn+=Oref[i]*Oref[i];}
  return sqrt(e/nn);
}
template<int BM,int BN,int KC>
static void run_correctness_all(){
  int M=BM,N=BN*2,K=512,nb=K/KC;   // K=512 divisible by KS in {1,2,4}
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
  int8_t* dWq;float* dWs;int8_t* dAq;float* dAs;__half* dO; float* dGws;
  CK(hipMalloc(&dWq,N*K));CK(hipMalloc(&dWs,N*nb*4));CK(hipMalloc(&dAq,M*K));CK(hipMalloc(&dAs,M*nb*4));CK(hipMalloc(&dO,M*N*2));
  CK(hipMalloc(&dGws,(size_t)4*M*N*4)); // up to KS=4 planes
  CK(hipMemcpy(dWq,Wq.data(),N*K,hipMemcpyHostToDevice));CK(hipMemcpy(dWs,Ws.data(),N*nb*4,hipMemcpyHostToDevice));
  CK(hipMemcpy(dAq,Aq.data(),M*K,hipMemcpyHostToDevice));CK(hipMemcpy(dAs,As.data(),M*nb*4,hipMemcpyHostToDevice));
  std::vector<__half> hO(M*N);
  // plain pipe2b 256x256
  { dim3 g(N/BN,M/BM),b(8*4*32); gemm_pipe2b<BM,BN,8,4,KC,1><<<g,b>>>(dAq,dAs,dWq,dWs,dO,M,N,K); CK(hipDeviceSynchronize());
    CK(hipMemcpy(hO.data(),dO,M*N*2,hipMemcpyDeviceToHost)); printf("  %-28s NRMSE=%.2e %s\n","pipe2b 256x256",ref_nrmse(hO,Oref),ref_nrmse(hO,Oref)<2e-3?"PASS":"FAIL"); }
  // BN=128 pipe2b
  { dim3 g(N/128,M/256),b(8*4*32); gemm_pipe2b<256,128,8,4,KC,1><<<g,b>>>(dAq,dAs,dWq,dWs,dO,M,N,K); CK(hipDeviceSynchronize());
    CK(hipMemcpy(hO.data(),dO,M*N*2,hipMemcpyDeviceToHost)); printf("  %-28s NRMSE=%.2e %s\n","pipe2b 256x128",ref_nrmse(hO,Oref),ref_nrmse(hO,Oref)<2e-3?"PASS":"FAIL"); }
  // splitk workspace KS=2,4
  for(int KS : {2,4}){ dim3 g(N/BN,M/BM,KS),b(8*4*32);
    gemm_pipe2b_splitk<BM,BN,8,4,KC,1><<<g,b>>>(dAq,dAs,dWq,dWs,dGws,M,N,K,KS); CK(hipDeviceSynchronize());
    reduce_splitk<<<(M*N+255)/256,256>>>(dGws,dO,M,N,KS); CK(hipDeviceSynchronize());
    CK(hipMemcpy(hO.data(),dO,M*N*2,hipMemcpyDeviceToHost));
    char nm[40]; snprintf(nm,40,"splitk-ws 256x256 KS=%d",KS);
    printf("  %-28s NRMSE=%.2e %s\n",nm,ref_nrmse(hO,Oref),ref_nrmse(hO,Oref)<2e-3?"PASS":"FAIL"); }
  // splitk atomic KS=2,4
  for(int KS : {2,4}){ dim3 g(N/BN,M/BM,KS),b(8*4*32);
    CK(hipMemset(dGws,0,(size_t)M*N*4));
    gemm_pipe2b_splitk_atomic<BM,BN,8,4,KC,1><<<g,b>>>(dAq,dAs,dWq,dWs,dGws,M,N,K,KS); CK(hipDeviceSynchronize());
    cast_f32_f16<<<(M*N+255)/256,256>>>(dGws,dO,M,N); CK(hipDeviceSynchronize());
    CK(hipMemcpy(hO.data(),dO,M*N*2,hipMemcpyDeviceToHost));
    char nm[40]; snprintf(nm,40,"splitk-atomic 256x256 KS=%d",KS);
    printf("  %-28s NRMSE=%.2e %s\n",nm,ref_nrmse(hO,Oref),ref_nrmse(hO,Oref)<2e-3?"PASS":"FAIL"); }
  hipFree(dWq);hipFree(dWs);hipFree(dAq);hipFree(dAs);hipFree(dO);hipFree(dGws);
}

// ============================================================================
// SPEED — HOT (L2-warm min-of-burst) AND COLD (32MB L2 thrash + rotated weight copies). Same harness
//   shape as gemm_pipe.hip. Each variant takes a launcher lambda so the split-K reduction is timed too.
// ============================================================================
// Common weight prep: dequant q6k + requant per-32. Returns device int8 weights + per-32 scales.
struct W { int8_t* wqk; float* wsk; int8_t* aq; float* as; __half* o; float* gws; };
static W prep_w(int M,int N,int K,int NBUF){
  W w{}; int nblk=K/256,nb16=K/16,nbk=K/32;
  block_q6_K* dW;int8_t* dWq;float* dWs;
  CK(hipMalloc(&dW,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMalloc(&dWq,(size_t)N*K));CK(hipMalloc(&dWs,(size_t)N*nb16*4));
  CK(hipMalloc(&w.wqk,(size_t)NBUF*N*K));CK(hipMalloc(&w.wsk,(size_t)NBUF*N*nbk*4));
  CK(hipMalloc(&w.aq,(size_t)M*K));CK(hipMalloc(&w.as,(size_t)M*nbk*4));CK(hipMalloc(&w.o,(size_t)M*N*2));
  CK(hipMalloc(&w.gws,(size_t)4*M*N*4));
  CK(hipMemset(dW,1,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMemset(w.aq,1,(size_t)M*K));CK(hipMemset(w.as,1,(size_t)M*nbk*4));
  dequant_q6k<<<(N*nblk+127)/128,128>>>(dW,dWq,dWs,N,K);
  for(int bb=0;bb<NBUF;bb++)for(int rep=0;rep<2;rep++)
    requant_kc<32><<<(N*nbk+127)/128,128>>>(dWq,dWs,w.wqk+(size_t)bb*N*K,w.wsk+(size_t)bb*N*nbk,N,K);
  CK(hipDeviceSynchronize());
  hipFree(dW);hipFree(dWq);hipFree(dWs);
  return w;
}
static void free_w(W&w){ hipFree(w.wqk);hipFree(w.wsk);hipFree(w.aq);hipFree(w.as);hipFree(w.o);hipFree(w.gws); }

__global__ void thrash(int8_t* p, size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]^=1; }

// Generic HOT bench: launcher(wqk_ptr, wsk_ptr) does the full GEMM (incl split-K reduction) on stream 0.
template<typename L>
static double bench_hot(L launcher,int M,int N,int K,int reps,double* out_ms){
  hipGetLastError();
  for(int w=0;w<10;w++)launcher();
  hipError_t le=hipGetLastError(); hipError_t se=hipDeviceSynchronize();
  if(le||se){ *out_ms=-1; return -1.0; }
  const int BURST=8; double best=1e30;
  for(int r=0;r<reps;r++){ double t0=now(); for(int i=0;i<BURST;i++)launcher(); hipDeviceSynchronize();
    double m=(now()-t0)/BURST; if(m<best)best=m; }
  *out_ms=best; return 2.0*M*N*K/1e12/(best/1e3);
}

int main(){
  printf("=== gemm_splitk: FILL the under-occupied small-M int8 GEMM (gfx1201/RDNA4) ===\n");
  printf("Goal: lift M=512 from 62 TF toward the M=2048 72 TF ceiling (44%% GPU fill -> full).\n");
  printf("Baselines (gemm_pipe.hip HOT): gate M=512 ~62-65 TF, gate M=2048 ~79 TF.\n\n");

  printf("[1] CORRECTNESS (synthetic int8, vs int32-accum float ref, NRMSE<2e-3):\n");
  run_correctness_all<256,256,32>();

  // ----------------------------------------------------------------------------------------------
  // Shapes: gate (N=14336 K=4096), qkv (N=6144 K=4096), down (N=4096 K=14336). M in {512, 2048}.
  // For each: baseline pipe2b 256x256, BN=128, split-K WS KS=2/3/4, split-K ATOMIC KS=2/3/4.
  //   KS must divide K and Kc=K/KS multiple of 32. K=4096: KS in {2,4} (KS=3 -> Kc=1365.3 INVALID, skip).
  //   K=14336: 14336/32=448 -> KS in {2,4} ok (Kc=7168/3584), KS=7 (Kc=2048) ok too; use {2,4}.
  // ----------------------------------------------------------------------------------------------
  struct Shape{const char* nm;int N,K;};
  Shape shapes[3]={{"gate",14336,4096},{"qkv",6144,4096},{"down",4096,14336}};
  int Ms[2]={512,2048};

  for(int si=0;si<3;si++){
    int N=shapes[si].N, K=shapes[si].K;
    printf("\n==================================================================================\n");
    printf("[2] HOT %s N=%d K=%d — fill experiments at M=512 and M=2048\n",shapes[si].nm,N,K);
    printf("==================================================================================\n");
    for(int mi=0;mi<2;mi++){
      int M=Ms[mi]; int R=(M>=2048)?20:40;
      W w=prep_w(M,N,K,1);
      double ms,tf;
      int nblocks_base = ((M+255)/256)*((N+255)/256);
      printf("--- M=%d (256x256 base blocks = %d, ~%.0f%% of 2048 wave-slots @8w/blk) ---\n",
             M, nblocks_base, 100.0*nblocks_base*8/2048.0);
      // baseline pipe2b 256x256
      { dim3 g(N/256,M/256),b(8*4*32);
        tf=bench_hot([&]{gemm_pipe2b<256,256,8,4,32,1><<<g,b>>>(w.aq,w.as,w.wqk,w.wsk,w.o,M,N,K);},M,N,K,R,&ms);
        printf("  %-26s %.3f ms  %6.1f TF\n","pipe2b 256x256 (BASE)",ms,tf); }
      double base_tf=tf;
      // BN=128
      { dim3 g(N/128,M/256),b(8*4*32);
        tf=bench_hot([&]{gemm_pipe2b<256,128,8,4,32,1><<<g,b>>>(w.aq,w.as,w.wqk,w.wsk,w.o,M,N,K);},M,N,K,R,&ms);
        printf("  %-26s %.3f ms  %6.1f TF  (%+.1f%% vs base)\n","pipe2b 256x128",ms,tf,(tf/base_tf-1)*100); }
      // split-K workspace KS=2,4
      for(int KS : {2,4}){
        if(K%KS || (K/KS)%32){ continue; }
        dim3 g(N/256,M/256,KS),b(8*4*32);
        int red_g=(M*N+255)/256;
        tf=bench_hot([&]{ gemm_pipe2b_splitk<256,256,8,4,32,1><<<g,b>>>(w.aq,w.as,w.wqk,w.wsk,w.gws,M,N,K,KS);
                          reduce_splitk<<<red_g,256>>>(w.gws,w.o,M,N,KS); },M,N,K,R,&ms);
        char nm[40]; snprintf(nm,40,"splitk-WS KS=%d (blocks %d)",KS,((M+255)/256)*((N+255)/256)*KS);
        printf("  %-26s %.3f ms  %6.1f TF  (%+.1f%% vs base)\n",nm,ms,tf,(tf/base_tf-1)*100); }
      // split-K atomic KS=2,4
      for(int KS : {2,4}){
        if(K%KS || (K/KS)%32){ continue; }
        dim3 g(N/256,M/256,KS),b(8*4*32);
        int cast_g=(M*N+255)/256;
        tf=bench_hot([&]{ CK(hipMemsetAsync(w.gws,0,(size_t)M*N*4,0));
                          gemm_pipe2b_splitk_atomic<256,256,8,4,32,1><<<g,b>>>(w.aq,w.as,w.wqk,w.wsk,w.gws,M,N,K,KS);
                          cast_f32_f16<<<cast_g,256>>>(w.gws,w.o,M,N); },M,N,K,R,&ms);
        char nm[40]; snprintf(nm,40,"splitk-ATOMIC KS=%d",KS);
        printf("  %-26s %.3f ms  %6.1f TF  (%+.1f%% vs base)\n",nm,ms,tf,(tf/base_tf-1)*100); }
      free_w(w);
    }
  }

  // ----------------------------------------------------------------------------------------------
  // [3] COLD — does the M=512 winner survive L2 pressure? Only the gate shape (the loser cell), all
  //   configs, M=512, with 3 rotated weight buffers + 32MB thrash between launches.
  // ----------------------------------------------------------------------------------------------
  printf("\n==================================================================================\n");
  printf("[3] COLD gate M=512 N=14336 K=4096 (3 rotated bufs + 32MB L2 thrash between launches)\n");
  printf("==================================================================================\n");
  { int M=512,N=14336,K=4096; const int NBUF=3; W w=prep_w(M,N,K,NBUF);
    size_t TH=32u*1024*1024; int8_t* dT; CK(hipMalloc(&dT,TH)); int th_g=(TH+255)/256;
    auto cold=[&](const char* nm,auto launcher){
      for(int i=0;i<4;i++){ int bi=i%NBUF; launcher(bi); }
      CK(hipDeviceSynchronize());
      int reps=40; double t0=now();
      for(int i=0;i<reps;i++){ int bi=i%NBUF; thrash<<<th_g,256>>>(dT,TH); launcher(bi); }
      CK(hipDeviceSynchronize()); double tot=now()-t0;
      double t1=now(); for(int i=0;i<reps;i++)thrash<<<th_g,256>>>(dT,TH); CK(hipDeviceSynchronize()); double thr=now()-t1;
      double ms=(tot-thr)/reps; double tf=2.0*M*N*K/1e12/(ms/1e3);
      printf("  %-26s %.3f ms  %6.1f TF\n",nm,ms,tf); return tf; };
    auto wq=[&](int bi){return w.wqk+(size_t)bi*N*K;}; auto wsf=[&](int bi){return w.wsk+(size_t)bi*N*(K/32);};
    double bt=cold("pipe2b 256x256 (BASE)",[&](int bi){ dim3 g(N/256,M/256),b(8*4*32);
                 gemm_pipe2b<256,256,8,4,32,1><<<g,b>>>(w.aq,w.as,wq(bi),wsf(bi),w.o,M,N,K); });
    cold("pipe2b 256x128",[&](int bi){ dim3 g(N/128,M/256),b(8*4*32);
                 gemm_pipe2b<256,128,8,4,32,1><<<g,b>>>(w.aq,w.as,wq(bi),wsf(bi),w.o,M,N,K); });
    for(int KS : {2,4}){ dim3 g(N/256,M/256,KS),b(8*4*32); int red_g=(M*N+255)/256;
      char nm[40]; snprintf(nm,40,"splitk-WS KS=%d",KS);
      cold(nm,[&](int bi){ gemm_pipe2b_splitk<256,256,8,4,32,1><<<g,b>>>(w.aq,w.as,wq(bi),wsf(bi),w.gws,M,N,K,KS);
                           reduce_splitk<<<red_g,256>>>(w.gws,w.o,M,N,KS); }); }
    for(int KS : {2,4}){ dim3 g(N/256,M/256,KS),b(8*4*32); int cast_g=(M*N+255)/256;
      char nm[40]; snprintf(nm,40,"splitk-ATOMIC KS=%d",KS);
      cold(nm,[&](int bi){ CK(hipMemsetAsync(w.gws,0,(size_t)M*N*4,0));
                           gemm_pipe2b_splitk_atomic<256,256,8,4,32,1><<<g,b>>>(w.aq,w.as,wq(bi),wsf(bi),w.gws,M,N,K,KS);
                           cast_f32_f16<<<cast_g,256>>>(w.gws,w.o,M,N); }); }
    (void)bt; hipFree(dT); free_w(w);
  }
  printf("\nDONE. If a config lifts M=512 cold gate >5%% WITHOUT regressing M=2048, it is the M-adaptive\n");
  printf("small-M fill. Wire it into full_forward (M-adaptive dispatch) for the real pp512 vs RADV test.\n");
  return 0;
}
