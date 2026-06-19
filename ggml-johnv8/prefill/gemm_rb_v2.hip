// gemm_rb_v2.hip — REGISTER-PRESSURE-OPTIMIZED scale-aware INT8 WMMA GEMM for gfx1201/RDNA4.
//   GOAL: push the scale-aware prefill GEMM from ~62 TFLOPS past RADV coopmat (~51 TFLOPS / 2341 t/s).
//   BOTTLENECK (measured): the baseline gemm_rb_kc keeps BOTH int32 WMMA accumulators v8i cR[WMT][WNT]
//     AND float Rf[WMT][WNT][8] live at once = 2x accumulator regs -> caps the tile at WMT*WNT=8.
//   LEVER: per-(mt,nt) scale-apply. Scale each cR[mt][nt] tile into Rf[mt][nt] IMMEDIATELY after its
//     WMMA chain completes, so only ~1 cR is live -> the 16-tile fits in registers -> bigger tile = higher TP.
//   COST: must reload Af[mt]/Bf[nt] fragments per (mt,nt) chain (vs once). MEASURED here.
//   Build: hipcc --offload-arch=gfx1201 -O3 gemm_rb_v2.hip -o gemm_rb_v2
//   VGPR: add -Rpass-analysis=kernel-resource-usage to see register usage / spills.
//
//   RESULT (gfx1201 / R9700, ROCm 7.2.3, sclk 3320MHz, measured 2026-06-19):
//     WINNER = V0 (cR+Rf both live) at BM=256 BN=256 WMW=8 WNW=4 KC=32 (WMT=2,WNT=4 = 8 tiles/warp).
//       189 VGPR, occupancy 8 waves/SIMD, ZERO spill. The premise that 8t->16t "spills" was about a
//       SMALL block; at 256x256 the 8-tile config does NOT spill, so the kernel is NOT register-bound
//       there -> V1/V2's freeing trick gives nothing and the extra LDS reloads only cost time.
//     gate M=512: 58-61 TFLOPS warm (RADV 51). 5-shape AGG: WARM 2849 t/s (+22% vs RADV 2341),
//       COLD-ish (L2-thrashed) 2587 t/s (+10%). Scales to 72 TFLOPS at M=2048/8192.
//     V1 (per-(mt,nt) free) peaks at 128x128 4x4 = 54 TFLOPS — BELOW V0. V2 (per-mt-row) ~56. Both lose.
//     => Lever tried & measured: register restructure works (drops VGPR 189->81..133, occ 8->10..16) but
//        does NOT help because the winning large-block config never spilled. Bottleneck = LDS/WMMA-issue.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>
typedef int i4 __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
#define CK(x) do{hipError_t e=(x);if(e){printf("HIP err %s @ %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__);}}while(0)

// ============================================================================
// V0 — BASELINE (copied verbatim from prefill_q6k.hip): cR + Rf both live.
//   Keeps ALL cR[WMT][WNT] (int32) live across the whole KC chain, then applies scales.
//   2x accumulator regs. Caps tile at WMT*WNT=8.
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
// V1 — PER-(mt,nt) SCALE-APPLY (the lever). For each output tile (mt,nt):
//   run the full ks-chain producing ONE cR, scale into Rf[mt][nt], discard cR -> reuse it.
//   Only ~1 v8i cR live -> frees ~7 accumulators vs V0 -> a 16-tile (4x4) should now fit.
//   COST: Af[mt]/Bf[nt] LDS fragments reloaded per (mt,nt) i.e. WNT/WMT-fold redundant LDS reads.
//   Scales (wsc/asc) hoisted before the (mt,nt) loop so the per-block scale gather happens once.
// ============================================================================
template<int BM,int BN,int WMW,int WNW,int KC>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_v1(
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
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+c];}}
  for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[0][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+c];}}
  __syncthreads(); int buf=0;
  for(int k0=0;k0<K;k0+=KC){ int blk=k0/KC; i4 rA[APT],rB[BPT]; int kn=k0+KC;
    if(kn<K){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kn+c];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rB[p]=*(const i4*)&wt[(size_t)(n_base+row)*K+kn+c];}} }
    // hoist scales once per K-block (gather is cheap vs the WMMA chain)
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_scale[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=a_scale[(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk];
    // PER-(mt,nt): one cR live at a time. Reload Af/Bf from LDS inside the chain.
    #pragma unroll
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int nt=0;nt<WNT;nt++){
        v8i cR=(v8i){};
        #pragma unroll
        for(int ks=0;ks<KSUB;ks++){
          const int8_t* ap=&As[buf][wm*WMT*16+mt*16+nl][ks*16+kg*8];
          const int8_t* bp=&Bs[buf][wn*WNT*16+nt*16+nl][ks*16+kg*8];
          v2i Af=(v2i){*(const int*)ap,*(const int*)(ap+4)};
          v2i Bf=(v2i){*(const int*)bp,*(const int*)(bp+4)};
          cR=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,Af,true,Bf,cR,false);
        }
        float ws=wsc[nt];
        #pragma unroll
        for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[j]*ws*asc[mt][j];
      }
    if(kn<K){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
}

// ============================================================================
// V2 — PER-mt-ROW scale-apply. Compromise: keep one ROW of cR live (cR[WNT]), i.e. WNT accumulators.
//   For each mt: run chain over all nt loading Af[mt] ONCE per ks (shared across nt -> less redundant
//   LDS than V1 which reloads Af per (mt,nt)). Then scale that row into Rf and free. WNT live cR.
//   Frees (WMT-1)*WNT accumulators vs V0. Bf[nt] still reloaded per mt-row (WMT-fold).
// ============================================================================
template<int BM,int BN,int WMW,int WNW,int KC>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_v2(
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
      v8i cR[WNT]; for(int b=0;b<WNT;b++)cR[b]=(v8i){};
      #pragma unroll
      for(int ks=0;ks<KSUB;ks++){
        const int8_t* ap=&As[buf][wm*WMT*16+mt*16+nl][ks*16+kg*8];
        v2i Af=(v2i){*(const int*)ap,*(const int*)(ap+4)};
        #pragma unroll
        for(int nt=0;nt<WNT;nt++){
          const int8_t* bp=&Bs[buf][wn*WNT*16+nt*16+nl][ks*16+kg*8];
          v2i Bf=(v2i){*(const int*)bp,*(const int*)(bp+4)};
          cR[nt]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,Af,true,Bf,cR[nt],false);
        }
      }
      #pragma unroll
      for(int nt=0;nt<WNT;nt++){ float ws=wsc[nt];
        #pragma unroll
        for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[nt][j]*ws*asc[mt][j]; }
    }
    if(kn<K){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
}

// CPU/GPU helpers reused from prefill_q6k.hip --------------------------------
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
static void dequant_q6k_cpu(const block_q6_K* x, float* y, int nblocks){
  for(int i=0;i<nblocks;i++){ float d=__half2float(x[i].d);
    const uint8_t* ql=x[i].ql; const uint8_t* qh=x[i].qh; const int8_t* sc=x[i].scales; float* yy=y+i*256;
    for(int n=0;n<256;n+=128){ for(int l=0;l<32;l++){ int is=l/16;
        int q1=((ql[l]&0xF)|(((qh[l]>>0)&3)<<4))-32, q2=((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4))-32;
        int q3=((ql[l]>>4)|(((qh[l]>>4)&3)<<4))-32, q4=((ql[l+32]>>4)|(((qh[l]>>6)&3)<<4))-32;
        yy[l]=d*sc[is]*q1; yy[l+32]=d*sc[is+2]*q2; yy[l+64]=d*sc[is+4]*q3; yy[l+96]=d*sc[is+6]*q4; }
      ql+=64; qh+=32; sc+=8; yy+=128; }
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
// CORRECTNESS: synthetic int8 inputs, int32 reference accumulation, NRMSE check.
//   Each variant (V0/V1/V2) must match the float reference (the only error is act int8 quant).
// ============================================================================
template<int BM,int BN,int A2,int B2,int KC, typename F>
static double run_correctness(const char* name, F kern){
  int M=64,N=256,K=512,nb=K/KC;
  // synthetic int8 weights with per-KC float scales, synthetic int8 act with per-KC scales
  srand(11);
  std::vector<int8_t> Wq(N*K); for(auto&x:Wq)x=(int8_t)(rand()%255-127);
  std::vector<float>  Ws(N*nb); for(auto&x:Ws)x=(0.5f+(rand()%100)/100.f)*0.01f;
  std::vector<int8_t> Aq(M*K); for(auto&x:Aq)x=(int8_t)(rand()%255-127);
  std::vector<float>  As(M*nb); for(auto&x:As)x=(0.5f+(rand()%100)/100.f)*0.02f;
  // float reference: sum over k of (Aq*As_blk)*(Wq*Ws_blk)  (true int32-accum then scale per block)
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
  printf("  %-22s NRMSE=%.2e  %s\n",name,nrmse,nrmse<1e-3?"PASS":"FAIL");
  hipFree(dWq);hipFree(dWs);hipFree(dAq);hipFree(dAs);hipFree(dO);
  return nrmse;
}

// ============================================================================
// SPEED: time one shape with a given variant+config. Returns TFLOPS (warm).
//   Uses real Q6_K->int8->requant32 buffers so memory traffic matches the forward.
// ============================================================================
template<int BM,int BN,int A2,int B2,int KC, typename F>
static double bench_shape(F kern,int M,int N,int K,int reps,double* out_ms){
  hipGetLastError(); // clear any residual error from a prior (e.g. INVALID) config
  int nblk=K/256,nb16=K/16,nbk=K/KC;
  block_q6_K* dW;int8_t*dWq,*dWqk,*dAq;float*dWs,*dWsk,*dAs;__half*dO;
  CK(hipMalloc(&dW,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMalloc(&dWq,(size_t)N*K));CK(hipMalloc(&dWs,(size_t)N*nb16*4));
  CK(hipMalloc(&dWqk,(size_t)N*K));CK(hipMalloc(&dWsk,(size_t)N*nbk*4));CK(hipMalloc(&dAq,(size_t)M*K));CK(hipMalloc(&dAs,(size_t)M*nbk*4));CK(hipMalloc(&dO,(size_t)M*N*2));
  CK(hipMemset(dW,1,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMemset(dAq,1,(size_t)M*K));CK(hipMemset(dAs,1,(size_t)M*nbk*4));
  dequant_q6k<<<(N*nblk+127)/128,128>>>(dW,dWq,dWs,N,K);
  if(KC==32){ for(int w=0;w<3;w++)requant_32<<<(N*nbk+127)/128,128>>>(dWq,dWs,dWqk,dWsk,N,K); }
  CK(hipDeviceSynchronize());
  int8_t* useWq = (KC==32)?dWqk:dWq; float* useWs=(KC==32)?dWsk:dWs;
  dim3 g(N/BN,M/BM),b(A2*B2*32);
  for(int w=0;w<8;w++)kern<<<g,b>>>(dAq,dAs,useWq,useWs,dO,M,N,K);
  hipError_t le=hipGetLastError(); hipError_t se=hipDeviceSynchronize();
  double ms; if(le||se){ ms=-1; *out_ms=-1; }  // launch/LDS failure -> flag invalid
  else { double t0=now(); for(int i=0;i<reps;i++)kern<<<g,b>>>(dAq,dAs,useWq,useWs,dO,M,N,K); hipDeviceSynchronize();
         ms=(now()-t0)/reps; *out_ms=ms; }
  hipFree(dW);hipFree(dWq);hipFree(dWs);hipFree(dWqk);hipFree(dWsk);hipFree(dAq);hipFree(dAs);hipFree(dO);
  return ms<0 ? -1.0 : 2.0*M*N*K/1e12/(ms/1e3);
}

// ============================================================================
// COLD-ish SPEED: rotate across NBUF weight copies so each rep reads L2-cold weights.
//   Gate weights = 14336*4096 = 58.7 MB >> L2 (~8 MB on RDNA4) so weights are already
//   mostly L2-cold even warm; but rotating buffers + a between-rep cache-thrash kernel
//   models the real forward (where the prior kernel evicted L2). Reports a conservative TFLOPS.
// ============================================================================
__global__ void thrash(int8_t* p, size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]^=1; }
template<int BM,int BN,int A2,int B2,int KC, typename F>
static double bench_shape_cold(F kern,int M,int N,int K,int reps,double* out_ms){
  int nblk=K/256,nb16=K/16,nbk=K/KC; const int NBUF=3;
  block_q6_K* dW;int8_t*dWq,*dWqk[NBUF],*dAq;float*dWs,*dWsk[NBUF],*dAs;__half*dO;
  CK(hipMalloc(&dW,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMalloc(&dWq,(size_t)N*K));CK(hipMalloc(&dWs,(size_t)N*nb16*4));
  for(int bb=0;bb<NBUF;bb++){CK(hipMalloc(&dWqk[bb],(size_t)N*K));CK(hipMalloc(&dWsk[bb],(size_t)N*nbk*4));}
  CK(hipMalloc(&dAq,(size_t)M*K));CK(hipMalloc(&dAs,(size_t)M*nbk*4));CK(hipMalloc(&dO,(size_t)M*N*2));
  CK(hipMemset(dW,1,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMemset(dAq,1,(size_t)M*K));CK(hipMemset(dAs,1,(size_t)M*nbk*4));
  dequant_q6k<<<(N*nblk+127)/128,128>>>(dW,dWq,dWs,N,K);
  for(int bb=0;bb<NBUF;bb++)requant_32<<<(N*nbk+127)/128,128>>>(dWq,dWs,dWqk[bb],dWsk[bb],N,K);
  CK(hipDeviceSynchronize());
  // L2 thrash buffer (32 MB)
  size_t TH=32u*1024*1024; int8_t* dT; CK(hipMalloc(&dT,TH));
  dim3 g(N/BN,M/BM),b(A2*B2*32);
  for(int w=0;w<4;w++)kern<<<g,b>>>(dAq,dAs,dWqk[0],dWsk[0],dO,M,N,K); CK(hipDeviceSynchronize());
  double t0=now();
  for(int i=0;i<reps;i++){ int bi=i%NBUF;
    thrash<<<(TH+255)/256,256>>>(dT,TH);                              // evict L2
    kern<<<g,b>>>(dAq,dAs,dWqk[bi],dWsk[bi],dO,M,N,K); }
  CK(hipDeviceSynchronize());
  double tot=now()-t0;
  // subtract thrash cost (measure it alone)
  double t1=now(); for(int i=0;i<reps;i++)thrash<<<(TH+255)/256,256>>>(dT,TH); CK(hipDeviceSynchronize());
  double thr=now()-t1;
  double ms=(tot-thr)/reps; *out_ms=ms;
  hipFree(dT);hipFree(dW);hipFree(dWq);hipFree(dWs);for(int bb=0;bb<NBUF;bb++){hipFree(dWqk[bb]);hipFree(dWsk[bb]);}hipFree(dAq);hipFree(dAs);hipFree(dO);
  return 2.0*M*N*K/1e12/(ms/1e3);
}

int main(int argc, char** argv){
  printf("=== gemm_rb_v2: register-restructured scale-aware INT8 WMMA GEMM (gfx1201) ===\n");
  printf("Target: gate M=512 N=14336 K=4096. RADV coopmat Q6_K ~51 TFLOPS / 2341 t/s.\n\n");

  printf("[1] CORRECTNESS (synthetic int8, vs int32-accum float ref, KC=32):\n");
  run_correctness<64,64,2,2,32>("V0 8tile 2x2",   gemm_v0<64,64,2,2,32>);
  run_correctness<64,64,4,2,32>("V1 16tile 4x2",  gemm_v1<64,64,4,2,32>);
  run_correctness<128,64,4,2,32>("V1 16tile 128x64",gemm_v1<128,64,4,2,32>);
  run_correctness<64,64,4,2,32>("V2 16tile 4x2",  gemm_v2<64,64,4,2,32>);
  run_correctness<64,64,2,2,16>("V0 KC16",        gemm_v0<64,64,2,2,16>);
  run_correctness<64,64,4,2,16>("V1 KC16 16tile", gemm_v1<64,64,4,2,16>);

  // ----- GATE shape sweep -----
  printf("\n[2] GATE shape M=512 N=14336 K=4096 — variant/config sweep (warm-standalone, 30 reps):\n");
  printf("    (WMT*WNT = tiles/warp; bigger=more work/warp but more regs)\n");
  const int M=512,N=14336,K=4096; const int R=30; double ms;
  struct Row{const char* name;double tf;double ms;};
  std::vector<Row> rows;
  #define BENCH(label,V,bm,bn,a2,b2,kc) { double tf=bench_shape<bm,bn,a2,b2,kc>(V<bm,bn,a2,b2,kc>,M,N,K,R,&ms); \
     if(tf<0) printf("  %-26s  --- INVALID (launch/LDS fail)\n",label); \
     else { printf("  %-26s %.3f ms  %6.1f TFLOPS\n",label,ms,tf); rows.push_back({label,tf,ms}); } }
  // V0 baseline (8-tile, the current best)
  BENCH("V0  64x64  2x2 KC32 (8t)", gemm_v0,64,64,2,2,32)
  BENCH("V0 256x256 8x4 KC32 (8t)", gemm_v0,256,256,8,4,32)
  // V1 per-(mt,nt): try the 16-tile configs that spilled under V0
  BENCH("V1  64x64  4x2 KC32 (16t)",gemm_v1,64,64,4,2,32)
  BENCH("V1 128x64  4x2 KC32 (16t)",gemm_v1,128,64,4,2,32)
  BENCH("V1 128x128 4x4 KC32 (16t)",gemm_v1,128,128,4,4,32)
  BENCH("V1 256x128 8x4 KC32 (16t)",gemm_v1,256,128,8,4,32)
  BENCH("V1 256x256 8x4 KC32 (32t)",gemm_v1,256,256,8,4,32)
  BENCH("V1  64x64  2x2 KC32 (8t)", gemm_v1,64,64,2,2,32)
  // V2 per-mt-row
  BENCH("V2  64x64  4x2 KC32 (16t)",gemm_v2,64,64,4,2,32)
  BENCH("V2 128x64  4x2 KC32 (16t)",gemm_v2,128,64,4,2,32)
  BENCH("V2 128x128 4x4 KC32 (16t)",gemm_v2,128,128,4,4,32)
  BENCH("V2 256x128 8x4 KC32 (16t)",gemm_v2,256,128,8,4,32)
  BENCH("V2 256x256 8x4 KC32 (32t)",gemm_v2,256,256,8,4,32)
  // KC=16 variants (no requant; per-16 exact) — check if KC32 requant gives real speed
  BENCH("V1 128x128 4x4 KC16 (16t)",gemm_v1,128,128,4,4,16)
  BENCH("V2 128x128 4x4 KC16 (16t)",gemm_v2,128,128,4,4,16)
  // BIGGER blocks via V1/V2 freeing — V0 would spill here; can the freeing trick push past 8t?
  BENCH("V1 256x512 8x4 KC32",      gemm_v1,256,512,8,4,32)
  BENCH("V1 512x256 8x4 KC32",      gemm_v1,512,256,8,4,32)
  BENCH("V2 256x512 8x4 KC32",      gemm_v2,256,512,8,4,32)
  BENCH("V2 512x256 8x4 KC32",      gemm_v2,512,256,8,4,32)
  BENCH("V0 256x512 8x4 KC32",      gemm_v0,256,512,8,4,32)
  BENCH("V0 512x256 8x4 KC32",      gemm_v0,512,256,8,4,32)
  BENCH("V1 256x256 8x8 KC32",      gemm_v1,256,256,8,8,32)
  BENCH("V0 256x256 8x8 KC32",      gemm_v0,256,256,8,8,32)
  BENCH("V0 128x256 4x4 KC32",      gemm_v0,128,256,4,4,32)
  BENCH("V0 256x128 4x4 KC32",      gemm_v0,256,128,4,4,32)
  #undef BENCH
  double best=0; const char* bn=""; for(auto&r:rows)if(r.tf>best){best=r.tf;bn=r.name;}
  printf("  >>> BEST gate: %s = %.1f TFLOPS  (RADV 51) %s\n", bn, best, best>51?"*** BEAT RADV ***":"(below)");

  // ====================================================================
  // [3] FULL 5-SHAPE AGGREGATE (M=512) with the WINNING config -> t/s.
  //   qkv N=6144 K=4096, o N=4096 K=4096, gate/up N=14336 K=4096, down N=4096 K=14336.
  //   t/s = M / (sum_ms * NLAYERS / 1000). Bielik-11B = 50 layers.
  // ====================================================================
  printf("\n[3] 5-SHAPE AGGREGATE M=512 (KC32) — t/s vs RADV 2341 / ggml ~1612:\n");
  { const int NL=50; const int R=40; double tot=0;
    struct S{const char* nm;int N,K;};
    S shapes[5]={{"qkv",6144,4096},{"o",4096,4096},{"gate",14336,4096},{"up",14336,4096},{"down",4096,14336}};
    // best config from phase [2] for M=512: V0 256x256 8x4 (set after measuring; use it directly)
    double totc=0;
    for(int s=0;s<5;s++){ double mss,msc; int Ns=shapes[s].N,Ks=shapes[s].K;
      double tf =bench_shape     <256,256,8,4,32>(gemm_v0<256,256,8,4,32>,512,Ns,Ks,R,&mss);
      double tfc=bench_shape_cold<256,256,8,4,32>(gemm_v0<256,256,8,4,32>,512,Ns,Ks,R,&msc);
      tot+=mss; totc+=msc;
      printf("  %-5s N=%-5d K=%-5d: warm %.3f ms %5.1f TF | cold %.3f ms %5.1f TF\n",shapes[s].nm,Ns,Ks,mss,tf,msc,tfc); }
    double ts=512.0/(tot*NL/1000.0), tsc=512.0/(totc*NL/1000.0);
    printf("  AGG 50L WARM: %.2f ms = %.0f t/s  vs RADV 2341 (%+.0f%%) / ggml 1612 (%+.0f%%)  %s\n",
      tot*NL, ts, (ts/2341-1)*100, (ts/1612-1)*100, ts>2341?"*** BEAT ***":"(below)");
    printf("  AGG 50L COLD: %.2f ms = %.0f t/s  vs RADV 2341 (%+.0f%%) / ggml 1612 (%+.0f%%)  %s\n",
      totc*NL, tsc, (tsc/2341-1)*100, (tsc/1612-1)*100, tsc>2341?"*** BEAT ***":"(below)");
  }

  // ====================================================================
  // [4] PER-M / PER-SHAPE best-config sweep. The M=512-tuned config can regress
  //   at M=2048/8192 (more row-tiles => the BM dimension wants to grow). Find best per (M, shape).
  // ====================================================================
  printf("\n[4] PER-M BEST CONFIG (gate N=14336 K=4096) — config regression check:\n");
  { int Ms[3]={512,2048,8192}; const int R=20; double mss;
    for(int mi=0;mi<3;mi++){ int Mx=Ms[mi]; double bt=0; const char* bc="";
      #define TRY(label,V,bm,bn,a2,b2) { if(Mx%(bm)==0){ double tf=bench_shape<bm,bn,a2,b2,32>(V<bm,bn,a2,b2,32>,Mx,14336,4096,R,&mss); \
         if(tf>bt){bt=tf;bc=label;} printf("    M=%-4d %-22s %5.1f TFLOPS\n",Mx,label,tf); } }
      TRY("V0 256x256 8x4",gemm_v0,256,256,8,4)
      TRY("V0 256x128 4x4",gemm_v0,256,128,4,4)
      TRY("V0 128x256 4x4",gemm_v0,128,256,4,4)
      TRY("V0 512x256 8x4",gemm_v0,512,256,8,4)
      TRY("V1 256x256 8x4",gemm_v1,256,256,8,4)
      TRY("V1 512x256 8x4",gemm_v1,512,256,8,4)
      TRY("V1 256x512 8x4",gemm_v1,256,512,8,4)
      #undef TRY
      printf("  >> M=%-4d BEST: %s = %.1f TFLOPS\n",Mx,bc,bt);
    }
  }
  return 0;
}
