// fused_normquant.hip — MEM-OP FUSION for ggml-johnv8 prefill (gfx1201 / RDNA4 / HIP).
//
// PROBLEM: the unfused prefill does rmsnorm -> quant -> GEMM as 3 kernels. The middle two are
//   memory-bound mem-ops: rmsnorm writes y_half[M][K], quant reads y_half + writes int8[M][K]+scale.
//   That is ~3 full activation passes over global before the GEMM even starts reading int8.
//
// FUSION (Priority 1): gemm_rb_kc_normquant. The GEMM activation B-prologue reads x_half[M][K]
//   DIRECTLY, applies per-row inv-rms * norm-weight, and quantizes each KC=32 chunk to int8+scale
//   INLINE as it lands in LDS. A tiny pass-1 (rms_pass1) computes per-row inv_rms[M] (1 cheap read
//   of the activation; M is small). Net: we delete the y_half write AND the int8[M][K] write/read.
//
//   Why a 2-phase (pass1 + fused-GEMM) and not 1 kernel: the per-row RMS reduction needs ALL K of a
//   row, but the GEMM tiles M into BM blocks each seeing only KC columns at a time. A row's RMS must
//   be known before its first KC chunk is quantized. pass1 is a full row-reduction (one block per
//   BM-row-group) and writes inv_rms[M] (M floats — negligible). The fused GEMM then reads inv_rms[row].
//
// MEASURED RESULT (M=512, gfx1201, R9700; PASS=NRMSE<0.02, fused==unfused bit-exact 1.5e-6):
//   o-proj  N=4096  K=4096 : +43%   (fusion's natural win — square, weight-light GEMM)
//   qkv     N=6144  K=4096 : +1%    (break-even)
//   gate/up N=14336 K=4096 : -21%   (LOSS) -- prologue re-quantizes activation (N/BN)x(M/BM) per N-block
//   down    N=4096  K=14336: -8%    (LOSS) -- large K amplifies the same per-block re-quant redundancy
//   ADAPTIVE (fuse where it wins): +5.5% per layer (2674 -> 2831 t/s aggregate).
//
//   KEY LESSON: a *prologue* fusion that re-reads+recomputes the activation pays a redundancy factor =
//   (gridX*gridY) = (N/BN)*(M/BM), because every output-tile block redoes the quant. It WINS only when
//   the GEMM is weight-light enough that the deleted activation global round-trip > the redundant
//   recompute (square `o`). For weight-bound (gate/up) or large-K (down) the cheap standalone quant
//   (~2% of GEMM time) is optimal -> keep it. The universally-safe wins are *epilogue* fusions
//   (residual-add, sketched below) which write each output exactly once -> ZERO redundancy.
//
// Build: hipcc --offload-arch=gfx1201 -O3 fused_normquant.hip -o /tmp/fnq && HIP_VISIBLE_DEVICES=0 /tmp/fnq
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
typedef int i4 __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef __half2 h2;
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
#define CK(x) do{hipError_t e=(x);if(e){printf("HIP err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

// =====================================================================================
//  UNFUSED reference path (the 3-kernel baseline we want to beat)
// =====================================================================================
// rmsnorm + per-32 int8 quant fused into ONE kernel (the fastest *unfused-from-GEMM* baseline:
// this is already the "rmsnorm_q" trick from the decode megakernel — it removes the y_half pass,
// but STILL writes int8[M][K]+scale to global which the GEMM then reads back). One block per row.
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

// silumul + per-32 int8 quant, standalone (unfused baseline for the down-proj path). One block per row.
__global__ void silumul_quant(const __half* __restrict__ g, const __half* __restrict__ u,
                              int8_t* __restrict__ q, float* __restrict__ s, int M, int K){
  int row=blockIdx.x; if(row>=M)return; int tid=threadIdx.x,nth=blockDim.x; int nb=K/32;
  const __half* gr=g+(size_t)row*K; const __half* ur=u+(size_t)row*K;
  for(int b=tid;b<nb;b+=nth){ float vals[32],am=0;
    for(int e=0;e<32;e++){int k=b*32+e; float gv=__half2float(gr[k]); float v=(gv/(1.f+expf(-gv)))*__half2float(ur[k]); vals[e]=v; float a=fabsf(v); if(a>am)am=a;}
    float d=am/127.f, id=d>0?1.f/d:0.f; s[(size_t)row*nb+b]=d;
    for(int e=0;e<32;e++){int vq=(int)lrintf(vals[e]*id); if(vq>127)vq=127; if(vq<-127)vq=-127; q[(size_t)row*32*nb+b*32+e]=(int8_t)vq;} }
}

// plain scale-aware Q8 GEMM (reads int8 act back from global) — KC=32. (== gemm_rb_q8 from prefill_q8_gemm.hip)
template<int BM,int BN,int WMW,int WNW>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_rb_q8(
    const int8_t* __restrict__ act, const float* __restrict__ a_scale,
    const int8_t* __restrict__ wt,  const float* __restrict__ w_scale,
    __half* __restrict__ out, int M, int N, int K){
  constexpr int KC=32, NWARP=WMW*WNW, NTHREAD=NWARP*32, WMT=BM/(WMW*16), WNT=BN/(WNW*16);
  constexpr int AV=BM*KC/16, BV=BN*KC/16, APT=(AV+NTHREAD-1)/NTHREAD, BPT=(BV+NTHREAD-1)/NTHREAD;
  __shared__ int8_t As[2][BM][KC]; __shared__ int8_t Bs[2][BN][KC];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16, wm=warp/WNW, wn=warp%WNW;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/32;
  float Rf[WMT][WNT][8];
  for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)for(int j=0;j<8;j++)Rf[a][b][j]=0.f;
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+c];}}
  for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[0][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+c];}}
  __syncthreads(); int buf=0;
  for(int k0=0;k0<K;k0+=KC){ int blk=k0>>5; i4 rA[APT],rB[BPT]; int kn=k0+KC;
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

// =====================================================================================
//  FUSED path
// =====================================================================================
// PASS 1: per-row inv-rms. One block per row, reduces sum(x^2) over K. Writes inv_rms[M].
//   (Reads x_half[M][K] once. M rows -> M floats out: negligible vs the y_half/int8 passes deleted.)
__global__ void rms_pass1(const __half* __restrict__ x, float* __restrict__ inv_rms, int M, int K, float eps){
  int row=blockIdx.x; if(row>=M) return; int tid=threadIdx.x, nth=blockDim.x;
  const __half* xr=x+(size_t)row*K; __shared__ float red[256];
  float p=0;
  // vectorized half2 read for the sumsq
  const h2* xr2=(const h2*)xr; int K2=K/2;
  for(int j=tid;j<K2;j+=nth){ float2 v=__half22float2(xr2[j]); p+=v.x*v.x+v.y*v.y; }
  red[tid]=p; __syncthreads();
  for(int st=nth/2;st>0;st>>=1){if(tid<st)red[tid]+=red[tid+st];__syncthreads();}
  if(tid==0) inv_rms[row]=rsqrtf(red[0]/K+eps);
}

// PASS 2: FUSED GEMM. Identical compute core to gemm_rb_q8, but the ACTIVATION prologue reads
//   x_half DIRECTLY and produces int8 As + per-block a_scale INLINE (no global int8 round-trip).
//   The activation load is reorganized so each thread owns ONE full (row, 32-block) = 32 halves,
//   computes max|.|, the block scale, and writes 32 int8 into LDS. The norm-weight w_norm[K] is
//   applied here too (x * inv_rms[row] * w_norm[k]). a_scale lives in LDS (per BM-row x per-block).
//   Weights (B) are loaded exactly as before (already int8 in global).
template<int BM,int BN,int WMW,int WNW>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_rb_kc_normquant(
    const __half* __restrict__ x, const float* __restrict__ inv_rms, const float* __restrict__ w_norm,
    const int8_t* __restrict__ wt, const float* __restrict__ w_scale,
    __half* __restrict__ out, int M, int N, int K){
  constexpr int KC=32, NWARP=WMW*WNW, NTHREAD=NWARP*32, WMT=BM/(WMW*16), WNT=BN/(WNW*16);
  constexpr int BV=BN*KC/16, BPT=(BV+NTHREAD-1)/NTHREAD;
  // Activation load: one thread per row owns its full KC=32 chunk (no cross-lane reduction on the
  // critical path). For BM<=NTHREAD each thread does at most 1 row; idle threads are cheap vs a
  // per-tile shuffle reduction (measured: 1-thread-per-row beats cooperative TPR>1).
  constexpr int AOPT=(BM+NTHREAD-1)/NTHREAD;
  __shared__ int8_t As[2][BM][KC]; __shared__ int8_t Bs[2][BN][KC];
  __shared__ float  Asc[2][BM];          // per-row activation scale for the current KC-tile
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16, wm=warp/WNW, wn=warp%WNW;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/32;
  float Rf[WMT][WNT][8];
  for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)for(int j=0;j<8;j++)Rf[a][b][j]=0.f;
  // --- fused activation load helper: thread owns row 'r', tile starting col K0 (32 cols).
  //     Vectorized half read (16x half2), norm by inv_rms[row]*w_norm[K0+e] (w_norm tiny -> L2-cached),
  //     quant inline, write int8 to LDS as two i4 (16B) stores. ---
  #define LOAD_ACT_TILE(BUF,K0) { \
    for(int p=0;p<AOPT;p++){int r=tid+p*NTHREAD; if(r<BM){ \
      const h2* xp=(const h2*)(x+(size_t)(m_base+r)*K+(K0)); const float* wp=w_norm+(K0); \
      float iv=inv_rms[m_base+r]; float vals[32],am=0; \
      _Pragma("unroll") for(int q=0;q<16;q++){float2 f=__half22float2(xp[q]); \
        float v0=f.x*iv*wp[2*q], v1=f.y*iv*wp[2*q+1]; vals[2*q]=v0; vals[2*q+1]=v1; \
        am=fmaxf(am,fmaxf(fabsf(v0),fabsf(v1)));} \
      float d=am*(1.f/127.f), id=d>0.f?1.f/d:0.f; Asc[BUF][r]=d; \
      int8_t qb[32]; \
      _Pragma("unroll") for(int e=0;e<32;e++){int vq=(int)lrintf(vals[e]*id); vq=vq>127?127:(vq<-127?-127:vq); qb[e]=(int8_t)vq;} \
      *(i4*)&As[BUF][r][0]=*(const i4*)&qb[0]; *(i4*)&As[BUF][r][16]=*(const i4*)&qb[16]; } } }
  #define LOAD_WT_TILE(BUF,K0) { \
    for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[BUF][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+(K0)+c];}} }
  LOAD_ACT_TILE(0,0); LOAD_WT_TILE(0,0);
  __syncthreads(); int buf=0;
  for(int k0=0;k0<K;k0+=KC){ int blk=k0>>5; int kn=k0+KC;
    // prefetch next tile into registers (act -> staged int8 in regs via vals; here we just keep simple: recompute in As next iter)
    v8i cR[WMT][WNT]; for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)cR[a][b]=(v8i){};
    for(int ks=0;ks<KC/16;ks++){ v2i Af[WMT],Bf[WNT];
      for(int mt=0;mt<WMT;mt++){const int8_t* ap=&As[buf][wm*WMT*16+mt*16+nl][ks*16+kg*8];Af[mt]=(v2i){*(const int*)ap,*(const int*)(ap+4)};}
      for(int nt=0;nt<WNT;nt++){const int8_t* bp=&Bs[buf][wn*WNT*16+nt*16+nl][ks*16+kg*8];Bf[nt]=(v2i){*(const int*)bp,*(const int*)(bp+4)};}
      for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)cR[mt][nt]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,Af[mt],true,Bf[nt],cR[mt][nt],false);}
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=Asc[buf][wm*WMT*16+mt*16+kg*8+j];
    for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[mt][nt][j]*wsc[nt]*asc[mt][j];
    if(kn<K){ LOAD_ACT_TILE(buf^1,kn); LOAD_WT_TILE(buf^1,kn); }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
  #undef LOAD_ACT_TILE
  #undef LOAD_WT_TILE
}

// =====================================================================================
//  PRIORITY-2 FUSION SKETCHES (single-consumer epilogues/prologues -> NO re-quant redundancy)
//  These fuse cleanly because the output is written exactly once (epilogue) or the input is read
//  exactly once into the only consumer (prologue) -> unlike normquant, no per-N-block re-compute.
// =====================================================================================

// (A) RESIDUAL-ADD epilogue: o-proj / down-proj write x += result instead of overwriting `out`.
//     The residual stream `out` is RMW once in the epilogue (replaces a separate `residual` kernel
//     = one extra M*N read+write of the stream). Drop-in: change the store line to a load-add-store.
//     Each output element is owned by exactly one thread/block -> race-free, no redundancy.
template<int BM,int BN,int WMW,int WNW>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_rb_q8_residual(
    const int8_t* __restrict__ act, const float* __restrict__ a_scale,
    const int8_t* __restrict__ wt,  const float* __restrict__ w_scale,
    __half* __restrict__ out, int M, int N, int K){   // out is PRE-LOADED with the residual stream
  constexpr int KC=32, NWARP=WMW*WNW, NTHREAD=NWARP*32, WMT=BM/(WMW*16), WNT=BN/(WNW*16);
  constexpr int AV=BM*KC/16, BV=BN*KC/16, APT=(AV+NTHREAD-1)/NTHREAD, BPT=(BV+NTHREAD-1)/NTHREAD;
  __shared__ int8_t As[2][BM][KC]; __shared__ int8_t Bs[2][BN][KC];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16, wm=warp/WNW, wn=warp%WNW;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/32;
  float Rf[WMT][WNT][8];
  for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)for(int j=0;j<8;j++)Rf[a][b][j]=0.f;
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+c];}}
  for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[0][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+c];}}
  __syncthreads(); int buf=0;
  for(int k0=0;k0<K;k0+=KC){ int blk=k0>>5; i4 rA[APT],rB[BPT]; int kn=k0+KC;
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
  // EPILOGUE FUSION: x += result (load residual, add, store) — one RMW; no separate residual kernel.
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){
    int m=m_base+wm*WMT*16+mt*16+kg*8+j; size_t o=(size_t)m*N+n_base+wn*WNT*16+nt*16+nl;
    out[o]=__float2half(__half2float(out[o])+Rf[mt][nt][j]); }
}

// (B) SiLU+quant PROLOGUE for the down-proj GEMM. The down GEMM's activation is gate/up output
//     (M x NFF, the LARGEST activation). Instead of a separate silumul kernel (read g,u -> write
//     f_half) + a separate quant (read f_half -> write int8) + GEMM read, the down prologue reads
//     g_half,u_half DIRECTLY, computes silu(g)*u, and quantizes inline. Single consumer (down) ->
//     the re-quant redundancy still applies per N-block, BUT down has small N (=NE=4096 -> N/BN=16),
//     so this is the favourable regime (cf. normquant `o`). Deletes f_half write + int8 write+read.
template<int BM,int BN,int WMW,int WNW>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_rb_kc_siluquant(
    const __half* __restrict__ g, const __half* __restrict__ u,
    const int8_t* __restrict__ wt, const float* __restrict__ w_scale,
    __half* __restrict__ out, int M, int N, int K){   // K == NFF
  constexpr int KC=32, NWARP=WMW*WNW, NTHREAD=NWARP*32, WMT=BM/(WMW*16), WNT=BN/(WNW*16);
  constexpr int BV=BN*KC/16, BPT=(BV+NTHREAD-1)/NTHREAD, AOPT=(BM+NTHREAD-1)/NTHREAD;
  __shared__ int8_t As[2][BM][KC]; __shared__ int8_t Bs[2][BN][KC]; __shared__ float Asc[2][BM];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16, wm=warp/WNW, wn=warp%WNW;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/32;
  float Rf[WMT][WNT][8];
  for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)for(int j=0;j<8;j++)Rf[a][b][j]=0.f;
  #define LOAD_SILU_TILE(BUF,K0) { \
    for(int p=0;p<AOPT;p++){int r=tid+p*NTHREAD; if(r<BM){ \
      const h2* gp=(const h2*)(g+(size_t)(m_base+r)*K+(K0)); const h2* up=(const h2*)(u+(size_t)(m_base+r)*K+(K0)); \
      float vals[32],am=0; \
      _Pragma("unroll") for(int q=0;q<16;q++){float2 gf=__half22float2(gp[q]),uf=__half22float2(up[q]); \
        float s0=(gf.x/(1.f+expf(-gf.x)))*uf.x, s1=(gf.y/(1.f+expf(-gf.y)))*uf.y; vals[2*q]=s0; vals[2*q+1]=s1; \
        am=fmaxf(am,fmaxf(fabsf(s0),fabsf(s1)));} \
      float d=am*(1.f/127.f), id=d>0.f?1.f/d:0.f; Asc[BUF][r]=d; \
      int8_t qb[32]; _Pragma("unroll") for(int e=0;e<32;e++){int vq=(int)lrintf(vals[e]*id); vq=vq>127?127:(vq<-127?-127:vq); qb[e]=(int8_t)vq;} \
      *(i4*)&As[BUF][r][0]=*(const i4*)&qb[0]; *(i4*)&As[BUF][r][16]=*(const i4*)&qb[16]; } } }
  #define LOAD_WT_TILE(BUF,K0) { \
    for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[BUF][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+(K0)+c];}} }
  LOAD_SILU_TILE(0,0); LOAD_WT_TILE(0,0); __syncthreads(); int buf=0;
  for(int k0=0;k0<K;k0+=KC){ int blk=k0>>5; int kn=k0+KC;
    v8i cR[WMT][WNT]; for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)cR[a][b]=(v8i){};
    for(int ks=0;ks<KC/16;ks++){ v2i Af[WMT],Bf[WNT];
      for(int mt=0;mt<WMT;mt++){const int8_t* ap=&As[buf][wm*WMT*16+mt*16+nl][ks*16+kg*8];Af[mt]=(v2i){*(const int*)ap,*(const int*)(ap+4)};}
      for(int nt=0;nt<WNT;nt++){const int8_t* bp=&Bs[buf][wn*WNT*16+nt*16+nl][ks*16+kg*8];Bf[nt]=(v2i){*(const int*)bp,*(const int*)(bp+4)};}
      for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)cR[mt][nt]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,Af[mt],true,Bf[nt],cR[mt][nt],false);}
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=Asc[buf][wm*WMT*16+mt*16+kg*8+j];
    for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[mt][nt][j]*wsc[nt]*asc[mt][j];
    if(kn<K){ LOAD_SILU_TILE(buf^1,kn); LOAD_WT_TILE(buf^1,kn); }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
  #undef LOAD_SILU_TILE
  #undef LOAD_WT_TILE
}
// (C) RoPE epilogue for the qkv GEMM: SKETCH ONLY (design note, not built here — needs the FA2 layout).
//     The qkv GEMM writes Q[M,NH*HD],K[M,NKV*HD],V[M,NKV*HD] from the int32 accumulator. RoPE rotates
//     pairs (x[2i],x[2i+1]) of each Q/K head by angle pos*base^(-2i/HD). Fuse into the epilogue: when
//     scaling cR[][][j]->half for an output column c that belongs to a Q or K head, instead of a plain
//     store, compute the partner column (c^1 within the head) via __shfl (lane nl holds column nl; the
//     pair (2i,2i+1) lives in adjacent lanes), apply the rotation, then store. This deletes the separate
//     rope pass (a M*(NH+NKV)*HD read+write). Cost: 2 trig + 1 shuffle per Q/K element in the epilogue.
//     Left as a sketch: correct fusion needs the exact per-token `pos` (M tokens => per-row angle), which
//     the prefill rope applies per (row=token, head, pair). Epilogue has (m=row) and (column=head*HD+2i)
//     in registers -> all inputs are present; implementation is mechanical but layout-sensitive.

// =====================================================================================
//  Host driver: correctness + speed
// =====================================================================================
int main(){
  srand(7);
  // ---------- CORRECTNESS ----------
  {
    int M=128, N=256, K=512, nb=K/32; float eps=1e-5f;
    std::vector<float> X(M*K), Wn(K);
    for(auto&v:X)v=(rand()%2000-1000)/256.f;     // x ~ [-4,4]
    for(auto&v:Wn)v=0.5f+(rand()%1000)/1000.f;   // rmsnorm weight ~ [0.5,1.5]
    std::vector<__half> Xh(M*K); for(int i=0;i<M*K;i++)Xh[i]=__float2half(X[i]);
    // int8 weights + per-32 scale
    std::vector<int8_t> W(N*K); std::vector<float> Ws(N*nb);
    for(auto&v:W)v=rand()%255-127; for(auto&v:Ws)v=(rand()%100)/1000.f+0.005f;

    // ---- CPU reference: rmsnorm(x_half) -> y; ref GEMM = sum_k y[m,k]*deq_w[n,k] ----
    std::vector<float> Y(M*K);
    for(int m=0;m<M;m++){ double ss=0; for(int k=0;k<K;k++){float v=__half2float(Xh[m*K+k]); ss+=(double)v*v;}
      float inv=1.f/sqrtf((float)(ss/K)+eps);
      for(int k=0;k<K;k++) Y[m*K+k]=__half2float(Xh[m*K+k])*inv*Wn[k]; }
    std::vector<float> Oref(M*N);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double s=0;
      for(int b=0;b<nb;b++)for(int e=0;e<32;e++){int k=b*32+e; s+=(double)Y[m*K+k]*((float)W[n*K+k]*Ws[n*nb+b]);}
      Oref[m*N+n]=(float)s; }

    __half *dX; float *dWn,*dInv,*dWs; int8_t *dW,*dAq; float *dAs; __half *dO;
    CK(hipMalloc(&dX,(size_t)M*K*2));CK(hipMalloc(&dWn,K*4));CK(hipMalloc(&dInv,M*4));
    CK(hipMalloc(&dW,(size_t)N*K));CK(hipMalloc(&dWs,(size_t)N*nb*4));
    CK(hipMalloc(&dAq,(size_t)M*K));CK(hipMalloc(&dAs,(size_t)M*nb*4));CK(hipMalloc(&dO,(size_t)M*N*2));
    CK(hipMemcpy(dX,Xh.data(),(size_t)M*K*2,hipMemcpyHostToDevice));
    CK(hipMemcpy(dWn,Wn.data(),K*4,hipMemcpyHostToDevice));
    CK(hipMemcpy(dW,W.data(),(size_t)N*K,hipMemcpyHostToDevice));
    CK(hipMemcpy(dWs,Ws.data(),(size_t)N*nb*4,hipMemcpyHostToDevice));

    // UNFUSED: rmsnorm_quant -> gemm_rb_q8
    rmsnorm_quant<<<M,256>>>(dX,dWn,dAq,dAs,M,K,eps); CK(hipDeviceSynchronize());
    { dim3 g(N/64,M/64),b(2*2*32); gemm_rb_q8<64,64,2,2><<<g,b>>>(dAq,dAs,dW,dWs,dO,M,N,K); }
    CK(hipDeviceSynchronize());
    std::vector<__half> Ouf(M*N); CK(hipMemcpy(Ouf.data(),dO,(size_t)M*N*2,hipMemcpyDeviceToHost));

    // FUSED: rms_pass1 -> gemm_rb_kc_normquant
    rms_pass1<<<M,256>>>(dX,dInv,M,K,eps); CK(hipDeviceSynchronize());
    { dim3 g(N/64,M/64),b(2*2*32); gemm_rb_kc_normquant<64,64,2,2><<<g,b>>>(dX,dInv,dWn,dW,dWs,dO,M,N,K); }
    CK(hipDeviceSynchronize());
    std::vector<__half> Of(M*N); CK(hipMemcpy(Of.data(),dO,(size_t)M*N*2,hipMemcpyDeviceToHost));

    // errors vs float reference. Use the output RMS magnitude as the rel-err denominator floor
    // (random int8 W x random act produces near-zero individual outputs; |err|/|tiny| is meaningless).
    double sumsq=0; for(int i=0;i<M*N;i++)sumsq+=(double)Oref[i]*Oref[i];
    float orms=sqrtf((float)(sumsq/(M*N)));
    double euf=0,ef=0,den=0,maxuf=0,maxf=0;
    for(int i=0;i<M*N;i++){ float r=Oref[i];
      float a=__half2float(Ouf[i]), b=__half2float(Of[i]);
      euf+=(a-r)*(a-r); ef+=(b-r)*(b-r); den+=r*r;
      float ru=fabsf(a-r)/(fabsf(r)+0.05f*orms), rf=fabsf(b-r)/(fabsf(r)+0.05f*orms);
      if(ru>maxuf)maxuf=ru; if(rf>maxf)maxf=rf; }
    // PASS on NRMSE (whole-output relative L2, the standard for this int8-GEMM family; cf. prefill_q6k).
    // maxRelErr is informational: random-sign int8xint8 produces ~0 outputs where |err|/|tiny| is meaningless.
    printf("CORRECTNESS  M=%d N=%d K=%d (PASS = NRMSE<0.02):\n",M,N,K);
    printf("  unfused (rmsnorm_quant+GEMM) vs float: NRMSE=%.4f  (maxRelErr=%.3f)  %s\n",sqrt(euf/den),maxuf,sqrt(euf/den)<0.02?"PASS":"FAIL");
    printf("  FUSED   (normquant GEMM)     vs float: NRMSE=%.4f  (maxRelErr=%.3f)  %s\n",sqrt(ef/den),maxf,sqrt(ef/den)<0.02?"PASS":"FAIL");
    // fused-vs-unfused agreement (should be ~bit-identical: same int8 math)
    double dfu=0; double maxfu=0;
    for(int i=0;i<M*N;i++){float a=__half2float(Ouf[i]),b=__half2float(Of[i]);dfu+=(a-b)*(a-b);float rr=fabsf(a-b)/(fabsf(a)+1e-3f);if(rr>maxfu)maxfu=rr;}
    printf("  FUSED vs UNFUSED agreement:            NRMSE=%.2e  maxRelErr=%.4f  %s\n",sqrt(dfu/den),maxfu,maxfu<0.02?"PASS":"FAIL");
    hipFree(dX);hipFree(dWn);hipFree(dInv);hipFree(dW);hipFree(dWs);hipFree(dAq);hipFree(dAs);hipFree(dO);
  }

  // ---------- CORRECTNESS @ DEPLOYED FAST CONFIG (BM=256,BN=256, gate K=4096) ----------
  {
    int M=512, N=512, K=4096, nb=K/32; float eps=1e-5f;  // N=512 -> 2 N-blocks (exercises redundancy path)
    std::vector<__half> Xh((size_t)M*K); std::vector<float> Wn(K);
    for(auto&v:Xh)v=__float2half((rand()%2000-1000)/256.f);
    for(auto&v:Wn)v=0.5f+(rand()%1000)/1000.f;
    std::vector<int8_t> W((size_t)N*K); std::vector<float> Ws((size_t)N*nb);
    for(auto&v:W)v=rand()%255-127; for(auto&v:Ws)v=(rand()%100)/1000.f+0.005f;
    std::vector<float> Oref((size_t)M*N);
    for(int m=0;m<M;m++){ double ss=0; for(int k=0;k<K;k++){float v=__half2float(Xh[(size_t)m*K+k]);ss+=(double)v*v;}
      float inv=1.f/sqrtf((float)(ss/K)+eps);
      for(int n=0;n<N;n++){ double s=0; for(int b=0;b<nb;b++)for(int e=0;e<32;e++){int k=b*32+e;
          float y=__half2float(Xh[(size_t)m*K+k])*inv*Wn[k]; s+=(double)y*((float)W[(size_t)n*K+k]*Ws[(size_t)n*nb+b]);}
        Oref[(size_t)m*N+n]=(float)s; } }
    __half*dX;float*dWn,*dInv,*dWs;int8_t*dW;__half*dO;
    CK(hipMalloc(&dX,(size_t)M*K*2));CK(hipMalloc(&dWn,K*4));CK(hipMalloc(&dInv,M*4));
    CK(hipMalloc(&dW,(size_t)N*K));CK(hipMalloc(&dWs,(size_t)N*nb*4));CK(hipMalloc(&dO,(size_t)M*N*2));
    CK(hipMemcpy(dX,Xh.data(),(size_t)M*K*2,hipMemcpyHostToDevice));CK(hipMemcpy(dWn,Wn.data(),K*4,hipMemcpyHostToDevice));
    CK(hipMemcpy(dW,W.data(),(size_t)N*K,hipMemcpyHostToDevice));CK(hipMemcpy(dWs,Ws.data(),(size_t)N*nb*4,hipMemcpyHostToDevice));
    rms_pass1<<<M,256>>>(dX,dInv,M,K,eps);
    { dim3 g(N/256,M/256),b(8*4*32); gemm_rb_kc_normquant<256,256,8,4><<<g,b>>>(dX,dInv,dWn,dW,dWs,dO,M,N,K); }
    CK(hipDeviceSynchronize());
    std::vector<__half> Of((size_t)M*N); CK(hipMemcpy(Of.data(),dO,(size_t)M*N*2,hipMemcpyDeviceToHost));
    double e=0,den=0; for(size_t i=0;i<(size_t)M*N;i++){float v=__half2float(Of[i]);e+=(v-Oref[i])*(v-Oref[i]);den+=Oref[i]*Oref[i];}
    printf("CORRECTNESS  M=%d N=%d K=%d FAST CONFIG <256,256,8,4>: NRMSE=%.4f  %s\n",M,N,K,sqrt(e/den),sqrt(e/den)<0.02?"PASS":"FAIL");
    hipFree(dX);hipFree(dWn);hipFree(dInv);hipFree(dW);hipFree(dWs);hipFree(dO);
  }

  // ---------- SPEED ----------
  {
    printf("\nSPEED (M=512, gfx1201). fused = rms_pass1 + normquant-GEMM ; unfused = rmsnorm_quant + plain-GEMM\n");
    int M=512; float eps=1e-5f; int NL=50;
    double agg_uf=0, agg_f=0, agg_best=0;
    #define BENCH(NN,KK,nm,BM,BN,WMW,WNW) { int N=NN,K=KK,nb=K/32; \
      __half*dX;float*dWn,*dInv,*dWs;int8_t*dW,*dAq;float*dAs;__half*dO; \
      CK(hipMalloc(&dX,(size_t)M*K*2));CK(hipMalloc(&dWn,K*4));CK(hipMalloc(&dInv,M*4)); \
      CK(hipMalloc(&dW,(size_t)N*K));CK(hipMalloc(&dWs,(size_t)N*nb*4)); \
      CK(hipMalloc(&dAq,(size_t)M*K));CK(hipMalloc(&dAs,(size_t)M*nb*4));CK(hipMalloc(&dO,(size_t)M*N*2)); \
      hipMemset(dX,1,(size_t)M*K*2);hipMemset(dWn,0x3c,K*4);hipMemset(dW,1,(size_t)N*K);hipMemset(dWs,0,(size_t)N*nb*4); \
      dim3 g(N/(BN),M/(BM)),b((WMW)*(WNW)*32); \
      /* warmup */ \
      for(int w=0;w<5;w++){rmsnorm_quant<<<M,256>>>(dX,dWn,dAq,dAs,M,K,eps);gemm_rb_q8<BM,BN,WMW,WNW><<<g,b>>>(dAq,dAs,dW,dWs,dO,M,N,K);} \
      for(int w=0;w<5;w++){rms_pass1<<<M,256>>>(dX,dInv,M,K,eps);gemm_rb_kc_normquant<BM,BN,WMW,WNW><<<g,b>>>(dX,dInv,dWn,dW,dWs,dO,M,N,K);} \
      CK(hipDeviceSynchronize()); int R=50; \
      /* UNFUSED total (norm+quant kernel + plain GEMM) */ \
      double t0=now();for(int i=0;i<R;i++){rmsnorm_quant<<<M,256>>>(dX,dWn,dAq,dAs,M,K,eps);gemm_rb_q8<BM,BN,WMW,WNW><<<g,b>>>(dAq,dAs,dW,dWs,dO,M,N,K);}CK(hipDeviceSynchronize()); \
      double uf=(now()-t0)/R; \
      /* just the rmsnorm_quant kernel alone (the mem-op we delete) */ \
      double t1=now();for(int i=0;i<R;i++)rmsnorm_quant<<<M,256>>>(dX,dWn,dAq,dAs,M,K,eps);CK(hipDeviceSynchronize()); \
      double qk=(now()-t1)/R; \
      /* just plain GEMM alone */ \
      double t2=now();for(int i=0;i<R;i++)gemm_rb_q8<BM,BN,WMW,WNW><<<g,b>>>(dAq,dAs,dW,dWs,dO,M,N,K);CK(hipDeviceSynchronize()); \
      double gp=(now()-t2)/R; \
      /* FUSED total (pass1 + normquant GEMM) */ \
      double t3=now();for(int i=0;i<R;i++){rms_pass1<<<M,256>>>(dX,dInv,M,K,eps);gemm_rb_kc_normquant<BM,BN,WMW,WNW><<<g,b>>>(dX,dInv,dWn,dW,dWs,dO,M,N,K);}CK(hipDeviceSynchronize()); \
      double fu=(now()-t3)/R; \
      /* fused GEMM alone (to see prologue overhead vs plain GEMM) */ \
      double t4=now();for(int i=0;i<R;i++)gemm_rb_kc_normquant<BM,BN,WMW,WNW><<<g,b>>>(dX,dInv,dWn,dW,dWs,dO,M,N,K);CK(hipDeviceSynchronize()); \
      double fg=(now()-t4)/R; \
      double t5=now();for(int i=0;i<R;i++)rms_pass1<<<M,256>>>(dX,dInv,M,K,eps);CK(hipDeviceSynchronize()); \
      double p1=(now()-t5)/R; \
      double bestshape=fu<uf?fu:uf; agg_uf+=uf; agg_f+=fu; agg_best+=bestshape; \
      printf("  %-5s N=%-5d K=%-5d | UNFUSED %.3f ms (quant %.3f + gemm %.3f) | FUSED %.3f ms (pass1 %.3f + nqgemm %.3f) | save %.1f%% -> use %s\n", \
             nm,N,K,uf,qk,gp,fu,p1,fg,(uf-fu)/uf*100, fu<uf?"FUSED":"unfused"); \
      hipFree(dX);hipFree(dWn);hipFree(dInv);hipFree(dW);hipFree(dWs);hipFree(dAq);hipFree(dAs);hipFree(dO); }
    // 5 prefill GEMM shapes (Bielik-11B). Fast GEMM config = BM=256,BN=256 (prefill_q8_gemm winner).
    // M=512 => gridY=2 (activation quantized 2x from M-tiling) x N/256 (N-tiling). Honest fast-config measure.
    BENCH(6144, 4096,"qkv", 256,256,8,4)
    BENCH(4096, 4096,"o",   256,256,8,4)
    BENCH(14336,4096,"gate",256,256,8,4)
    BENCH(14336,4096,"up",  256,256,8,4)
    BENCH(4096, 14336,"down",256,256,8,4)
    #undef BENCH
    printf("\nAGGREGATE 1-layer (5 GEMM): UNFUSED %.3f ms | all-FUSED %.3f ms (%.1f%%) | ADAPTIVE best-of %.3f ms (saved %.1f%%)\n",
           agg_uf,agg_f,(agg_uf-agg_f)/agg_uf*100, agg_best,(agg_uf-agg_best)/agg_uf*100);
    printf("AGGREGATE %dL prefill:       UNFUSED %.2f ms (%.0f t/s) | ADAPTIVE %.2f ms (%.0f t/s)\n",
           NL, agg_uf*NL, M/(agg_uf*NL/1000.0), agg_best*NL, M/(agg_best*NL/1000.0));
  }

  // ---------- PRIORITY-2: SiLU+quant fused into down-proj prologue (down has small N -> favourable) ----
  {
    printf("\nPRIORITY-2 SiLU+quant fusion (down-proj M=512 N=4096 K=NFF=14336):\n");
    int M=512,N=4096,K=14336,nb=K/32;  // K = NFF
    // correctness on a small instance
    { int m=64,n=128,k=512,nbk=k/32;
      std::vector<__half> G((size_t)m*k),U((size_t)m*k); for(auto&v:G)v=__float2half((rand()%400-200)/128.f); for(auto&v:U)v=__float2half((rand()%400-200)/128.f);
      std::vector<int8_t> W((size_t)n*k); std::vector<float> Ws((size_t)n*nbk);
      for(auto&v:W)v=rand()%255-127; for(auto&v:Ws)v=(rand()%100)/1000.f+0.005f;
      std::vector<float> Oref((size_t)m*n);
      for(int i=0;i<m;i++)for(int j=0;j<n;j++){ double s=0; for(int b=0;b<nbk;b++)for(int e=0;e<32;e++){int c=b*32+e;
          float gv=__half2float(G[(size_t)i*k+c]); float f=(gv/(1.f+expf(-gv)))*__half2float(U[(size_t)i*k+c]);
          s+=(double)f*((float)W[(size_t)j*k+c]*Ws[(size_t)j*nbk+b]); } Oref[(size_t)i*n+j]=(float)s; }
      __half*dG,*dU,*dO; int8_t*dW; float*dWs;
      CK(hipMalloc(&dG,(size_t)m*k*2));CK(hipMalloc(&dU,(size_t)m*k*2));CK(hipMalloc(&dW,(size_t)n*k));CK(hipMalloc(&dWs,(size_t)n*nbk*4));CK(hipMalloc(&dO,(size_t)m*n*2));
      CK(hipMemcpy(dG,G.data(),(size_t)m*k*2,hipMemcpyHostToDevice));CK(hipMemcpy(dU,U.data(),(size_t)m*k*2,hipMemcpyHostToDevice));
      CK(hipMemcpy(dW,W.data(),(size_t)n*k,hipMemcpyHostToDevice));CK(hipMemcpy(dWs,Ws.data(),(size_t)n*nbk*4,hipMemcpyHostToDevice));
      dim3 g(n/64,m/64),b(2*2*32); gemm_rb_kc_siluquant<64,64,2,2><<<g,b>>>(dG,dU,dW,dWs,dO,m,n,k); CK(hipDeviceSynchronize());
      std::vector<__half> O((size_t)m*n); CK(hipMemcpy(O.data(),dO,(size_t)m*n*2,hipMemcpyDeviceToHost));
      double e=0,den=0; for(size_t i=0;i<(size_t)m*n;i++){float v=__half2float(O[i]);e+=(v-Oref[i])*(v-Oref[i]);den+=Oref[i]*Oref[i];}
      printf("  silu+quant fused GEMM vs float: NRMSE=%.4f  %s\n",sqrt(e/den),sqrt(e/den)<0.02?"PASS":"FAIL");
      hipFree(dG);hipFree(dU);hipFree(dW);hipFree(dWs);hipFree(dO);
    }
    // speed: unfused (silumul + quant + GEMM) vs fused (silumul+quant in down prologue)
    __half*dG,*dU,*dFh,*dO; int8_t*dW,*dFq; float*dWs,*dFs;
    CK(hipMalloc(&dG,(size_t)M*K*2));CK(hipMalloc(&dU,(size_t)M*K*2));CK(hipMalloc(&dFh,(size_t)M*K*2));
    CK(hipMalloc(&dFq,(size_t)M*K));CK(hipMalloc(&dFs,(size_t)M*nb*4));
    CK(hipMalloc(&dW,(size_t)N*K));CK(hipMalloc(&dWs,(size_t)N*nb*4));CK(hipMalloc(&dO,(size_t)M*N*2));
    hipMemset(dG,1,(size_t)M*K*2);hipMemset(dU,1,(size_t)M*K*2);hipMemset(dW,1,(size_t)N*K);hipMemset(dWs,0,(size_t)N*nb*4);
    dim3 g(N/256,M/256),b(8*4*32);
    // warmup both
    for(int w=0;w<5;w++){silumul_quant<<<M,256>>>(dG,dU,dFq,dFs,M,K);gemm_rb_q8<256,256,8,4><<<g,b>>>(dFq,dFs,dW,dWs,dO,M,N,K);}
    for(int w=0;w<5;w++)gemm_rb_kc_siluquant<256,256,8,4><<<g,b>>>(dG,dU,dW,dWs,dO,M,N,K);
    CK(hipDeviceSynchronize()); int R=50;
    double t0=now();for(int i=0;i<R;i++){silumul_quant<<<M,256>>>(dG,dU,dFq,dFs,M,K);gemm_rb_q8<256,256,8,4><<<g,b>>>(dFq,dFs,dW,dWs,dO,M,N,K);}CK(hipDeviceSynchronize());
    double uf=(now()-t0)/R;
    double t0b=now();for(int i=0;i<R;i++)silumul_quant<<<M,256>>>(dG,dU,dFq,dFs,M,K);CK(hipDeviceSynchronize());
    double sq=(now()-t0b)/R;
    double t1=now();for(int i=0;i<R;i++)gemm_rb_kc_siluquant<256,256,8,4><<<g,b>>>(dG,dU,dW,dWs,dO,M,N,K);CK(hipDeviceSynchronize());
    double fu=(now()-t1)/R;
    printf("  UNFUSED %.3f ms (silumul_quant %.3f + plain-down %.3f) | FUSED silu-prologue %.3f ms | save %.1f%% -> use %s\n",
           uf,sq,uf-sq,fu,(uf-fu)/uf*100, fu<uf?"FUSED":"unfused");
    printf("  NOTE: down has large K=NFF=14336 -> prologue silu+quant recomputed (N/BN)x(M/BM)=32x -> redundancy-bound (same lesson as gate/up).\n");
    hipFree(dG);hipFree(dU);hipFree(dFh);hipFree(dFq);hipFree(dFs);hipFree(dW);hipFree(dWs);hipFree(dO);
  }

  // ---------- PRIORITY-2: residual-add epilogue (o-proj) correctness ----------
  {
    int M=64,N=128,K=512,nb=K/32;
    std::vector<int8_t> A((size_t)M*K),W((size_t)N*K); std::vector<float> As((size_t)M*nb),Ws((size_t)N*nb);
    for(auto&v:A)v=rand()%255-127; for(auto&v:W)v=rand()%255-127;
    for(auto&v:As)v=(rand()%100)/1000.f+0.005f; for(auto&v:Ws)v=(rand()%100)/1000.f+0.005f;
    std::vector<__half> Res((size_t)M*N); for(auto&v:Res)v=__float2half((rand()%400-200)/64.f);
    std::vector<float> Oref((size_t)M*N);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double s=0; for(int b=0;b<nb;b++){int id=0;for(int e=0;e<32;e++)id+=(int)W[(size_t)n*K+b*32+e]*(int)A[(size_t)m*K+b*32+e]; s+=id*Ws[(size_t)n*nb+b]*As[(size_t)m*nb+b];}
      Oref[(size_t)m*N+n]=(float)s+__half2float(Res[(size_t)m*N+n]); }
    int8_t*dA,*dW; float*dAs,*dWs; __half*dO;
    CK(hipMalloc(&dA,(size_t)M*K));CK(hipMalloc(&dW,(size_t)N*K));CK(hipMalloc(&dAs,(size_t)M*nb*4));CK(hipMalloc(&dWs,(size_t)N*nb*4));CK(hipMalloc(&dO,(size_t)M*N*2));
    CK(hipMemcpy(dA,A.data(),(size_t)M*K,hipMemcpyHostToDevice));CK(hipMemcpy(dW,W.data(),(size_t)N*K,hipMemcpyHostToDevice));
    CK(hipMemcpy(dAs,As.data(),(size_t)M*nb*4,hipMemcpyHostToDevice));CK(hipMemcpy(dWs,Ws.data(),(size_t)N*nb*4,hipMemcpyHostToDevice));
    CK(hipMemcpy(dO,Res.data(),(size_t)M*N*2,hipMemcpyHostToDevice));   // pre-load residual stream
    dim3 g(N/64,M/64),b(2*2*32); gemm_rb_q8_residual<64,64,2,2><<<g,b>>>(dA,dAs,dW,dWs,dO,M,N,K); CK(hipDeviceSynchronize());
    std::vector<__half> O((size_t)M*N); CK(hipMemcpy(O.data(),dO,(size_t)M*N*2,hipMemcpyDeviceToHost));
    double e=0,den=0; for(size_t i=0;i<(size_t)M*N;i++){float v=__half2float(O[i]);e+=(v-Oref[i])*(v-Oref[i]);den+=Oref[i]*Oref[i];}
    printf("\nPRIORITY-2 residual-add epilogue (o-proj) vs float: NRMSE=%.4f  %s (deletes a separate residual kernel = M*N RMW)\n",sqrt(e/den),sqrt(e/den)<0.02?"PASS":"FAIL");
    hipFree(dA);hipFree(dW);hipFree(dAs);hipFree(dWs);hipFree(dO);
  }
  return 0;
}
