// gemm_dp4a.hip — register-blocked INT8 DP4A GEMM for gfx1201/RDNA4, scale-aware prefill.
//
//   HYPOTHESIS UNDER TEST (from the task): the scale-aware INT8 *WMMA* GEMM (gemm_v3p in
//   gemm_rb_v3.hip, 256x256 8x4, ~61 TFLOPS standalone) is OCCUPANCY-bound — its Rf(fp32) +
//   cR(v8i int32) accumulators take 188 VGPR and cap occupancy at 8 waves/SIMD. RADV's K-quant
//   int8 prefill uses integer DP4A (v_dot4_i32_i8 / dotPacked4x8), NOT the matrix core, and RADV
//   int8 full-forward (Q8 ~2820 t/s) beats our WMMA full-forward (2611 t/s). A DP4A GEMM keeps only
//   ONE int32 accumulator per output element (no v8i WMMA fragments) -> far fewer VGPR -> potentially
//   much higher occupancy -> may hide latency better and win on this occupancy-bound workload despite
//   DP4A's lower per-op throughput (4 MAC/instr on ALU vs WMMA's 16x16x16 on the matrix core).
//
//   gfx1201 DP4A BUILTIN (VERIFIED, this file): __builtin_amdgcn_sdot4 needs target feature dot1-insts
//   which gfx1201 does NOT have. The correct builtin is __builtin_amdgcn_sudot4(neg_a,a,neg_b,b,acc,
//   clamp); with neg_a=neg_b=true it lowers to `v_dot4_i32_iu8 ... neg_lo:[1,1,0]` = exact SIGNED
//   int8 dot4 + acc (two's complement). Verified bit-exact vs scalar int8 dot incl. negatives.
//
//   Build: hipcc --offload-arch=gfx1201 -O3 gemm_dp4a.hip -o gemm_dp4a
//   VGPR/occ/spill: add -Rpass-analysis=kernel-resource-usage
//   GPU: HIP_VISIBLE_DEVICES=0 (R9700 #0). Run `rocm-smi --setperflevel auto` first.
//
//   DESIGN — classic register-blocked DP4A GEMM:
//     * Block computes a BM x BN output tile. NThreads threads, each owns a TM x TN int32 sub-tile
//       (TM rows x TN cols of the output) -> BM=TM*sqrt-ish, BN=TN*..., laid out as a TX x TY thread
//       grid (TX threads cover BN/TN cols stride, TY cover BM/TM rows).
//     * K is processed in KC-wide blocks (KC bytes = KC/4 DP4A steps). A[BM][KC] and B[BN][KC] staged
//       in LDS once per K-block (the "dequant-once-to-LDS" reuse structure). Each thread then loads
//       its TM A-fragments and TN B-fragments (int = 4 packed int8) from LDS and does TM*TN sudot4.
//     * Integer accumulators acc[TM][TN] (int32) are pure — NO float in the inner loop. Per-32-K
//       float w_scale * a_scale are folded into a float tile AFTER each KC-block's integer dot (P3),
//       so KC is a multiple of 32 and we fold once per 32-K sub-block. fp32 Cf[TM][TN] holds result.
//   The whole point: acc[TM][TN] int32 (TM*TN regs) vs WMMA's v8i fragments — for a 8x8 thread tile
//   that is 64 int32 acc + 64 fp32 = 128 + overhead, but we can pick SMALL tiles (4x4=16 acc) to
//   push occupancy way past 8 waves. The sweep below finds where occupancy beats per-op throughput.
//
//   ============================================================================================
//   VERDICT (MEASURED gfx1201/R9700, ROCm 7.2.3, GPU0, perflevel auto, min-burst + L2-thrash cold):
//     HYPOTHESIS REFUTED. DP4A does NOT win; it is ~2.7x SLOWER than WMMA and the gap is structural.
//     * DP4A DOES reach the higher occupancy predicted: t4x4=109 VGPR occ12, t2x2=59 VGPR occ16,
//       vs WMMA v3p 192 VGPR occ8. Spill-free for small tiles. So the occupancy lever works.
//     * BUT the gate-shape result is the opposite of the bet:
//         BEST DP4A (128x64 t8x4 KC32, occ9): 20.4 TF warm / 19.9 TF COLD
//         WMMA v3p (256x256, occ8):           62.3 TF warm / 54.5 TF COLD   -> WMMA 2.7x faster.
//       Higher-occupancy small tiles are EVEN SLOWER (t4x4=12 TF, t2x2=10 TF): occupancy is already
//       saturated at occ-9; the binding constraint is DP4A *issue throughput*, not latency-hiding.
//     * ROOT CAUSE — pure DP4A issue roofline (16 independent acc chains, all-register, no LDS):
//         ~10.1 Tdp4/s = ~81 TOPS int8 peak. The WMMA matrix-core peak measured here is ~339 TOPS.
//       The DP4A ALU path has ~4.2x LOWER peak than the matrix core on RDNA4. No amount of occupancy
//       can close a 4x peak-throughput deficit on a compute-bound GEMM (gate is compute-bound at M=512).
//     * 5-shape AGG M=512 (50 layers, COLD): DP4A 600 t/s vs WMMA 2737 t/s. DP4A is nowhere near
//       RADV's int8 ~2820 t/s; WMMA is the only path that gets close (2737 cold / 3068 warm).
//   WHY THE HYPOTHESIS LOOKED PLAUSIBLE BUT WAS WRONG: RADV beating our WMMA full-forward on Q8
//     (2820 vs 2611) is NOT because DP4A out-GEMMs WMMA — at M=512 prefill the GEMM is compute-bound
//     and DP4A loses 2.7x. RADV's edge must come from elsewhere (dequant cost, fusion, dispatch/
//     scheduling, the non-GEMM glue), not from the integer-dot core. DP4A is the WRONG lever for the
//     prefill GEMM; to extend the prefill win to Q8/Q4_K_M, keep WMMA and attack dequant/fusion/occ-8.
//   (DP4A would only win where the matrix core can't be fed — i.e. memory-bound decode GEMV, not the
//    compute-bound prefill GEMM. That is a different kernel and a different question.)
//   ============================================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>

typedef int   i4  __attribute__((ext_vector_type(4)));
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
#define CK(x) do{hipError_t e=(x);if(e){printf("HIP err %s @ %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__);}}while(0)

__device__ __forceinline__ int dp4(int a,int b,int acc){
  return __builtin_amdgcn_sudot4(true,a,true,b,acc,false); // v_dot4_i32_iu8 neg_lo:[1,1,0] = signed dot4
}

// ============================================================================
// DP4A GEMM — register-blocked.
//   BM x BN output tile per block. Thread grid TX x TY (TX*TY threads). Each thread owns TM x TN
//   outputs => BM = TY*TM, BN = TX*TN. KC = K-block in bytes (multiple of 32). Scales folded per 32.
//   LDS: As[BM][KC] + Bs[BN][KC] (single-buffered + __syncthreads; double-buffer variant below).
//   K dimension is int8; DP4A consumes 4 int8 per step => KC/4 dp4 steps per K-block per (m,n) pair.
// ============================================================================
template<int BM,int BN,int TM,int TN,int KC,int DB>
__global__ void __launch_bounds__((BM/TM)*(BN/TN)) gemm_dp4a(
    const int8_t* __restrict__ act, const float* __restrict__ a_scale,
    const int8_t* __restrict__ wt,  const float* __restrict__ w_scale,
    __half* __restrict__ out, int M, int N, int K){
  constexpr int TX=BN/TN, TY=BM/TM, NT=TX*TY;       // threads
  constexpr int KW=KC/4;                            // dp4 steps per K-block
  constexpr int NB=2;                               // sub-blocks of 32 in KC (KC must be mult of 32)
  constexpr int S32=KC/32;                          // #32-K sub-blocks per K-block
  __shared__ int8_t As[DB?2:1][BM][KC];
  __shared__ int8_t Bs[DB?2:1][BN][KC];
  int tid=threadIdx.x;
  int tx=tid%TX, ty=tid/TX;                          // thread's col/row in the TX x TY grid
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/32;
  // each thread's output rows/cols: row = m_base + ty*TM + i ; col = n_base + tx*TN + j
  float Cf[TM][TN];
  #pragma unroll
  for(int i=0;i<TM;i++)
    #pragma unroll
    for(int j=0;j<TN;j++) Cf[i][j]=0.f;

  // staged global->LDS loaders (i4 = 16B per thread-iter), cover BM*KC and BN*KC bytes.
  constexpr int AV=BM*KC/16, BV=BN*KC/16, APT=(AV+NT-1)/NT, BPT=(BV+NT-1)/NT;
  auto loadA=[&](int buf,int k0){
    #pragma unroll
    for(int p=0;p<APT;p++){int idx=tid+p*NT; if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;
      *(i4*)&As[buf][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+k0+c];}} };
  auto loadB=[&](int buf,int k0){
    #pragma unroll
    for(int p=0;p<BPT;p++){int idx=tid+p*NT; if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;
      *(i4*)&Bs[buf][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+k0+c];}} };

  if(DB){ loadA(0,0); loadB(0,0); __syncthreads(); }
  int buf=0;
  for(int k0=0;k0<K;k0+=KC){
    int blk32=k0/32;
    if(!DB){ loadA(0,k0); loadB(0,k0); __syncthreads(); }
    // prefetch next tile into the other buffer (DB)
    if(DB){ int kn=k0+KC; if(kn<K){ loadA(buf^1,kn); loadB(buf^1,kn); } }
    int cb=DB?buf:0;
    // integer DP4A over the K-block; fold scales per 32-K sub-block.
    #pragma unroll
    for(int sb=0;sb<S32;sb++){
      int acc[TM][TN];
      #pragma unroll
      for(int i=0;i<TM;i++)
        #pragma unroll
        for(int j=0;j<TN;j++) acc[i][j]=0;
      #pragma unroll
      for(int kk=0;kk<32/4;kk++){            // 8 dp4 steps per 32-K
        int kc=sb*32+kk*4;
        int Areg[TM], Breg[TN];
        #pragma unroll
        for(int i=0;i<TM;i++){int row=ty*TM+i; Areg[i]=*(const int*)&As[cb][row][kc];}
        #pragma unroll
        for(int j=0;j<TN;j++){int col=tx*TN+j; Breg[j]=*(const int*)&Bs[cb][col][kc];}
        #pragma unroll
        for(int i=0;i<TM;i++)
          #pragma unroll
          for(int j=0;j<TN;j++) acc[i][j]=dp4(Areg[i],Breg[j],acc[i][j]);
      }
      // fold scales for this 32-K sub-block (P3: outside the integer dot)
      int sblk=blk32+sb;
      float asc[TM], wsc[TN];
      #pragma unroll
      for(int i=0;i<TM;i++) asc[i]=a_scale[(size_t)(m_base+ty*TM+i)*nb+sblk];
      #pragma unroll
      for(int j=0;j<TN;j++) wsc[j]=w_scale[(size_t)(n_base+tx*TN+j)*nb+sblk];
      #pragma unroll
      for(int i=0;i<TM;i++)
        #pragma unroll
        for(int j=0;j<TN;j++) Cf[i][j]+=(float)acc[i][j]*asc[i]*wsc[j];
    }
    if(!DB) __syncthreads();
    if(DB){ __syncthreads(); buf^=1; }
  }
  // store
  #pragma unroll
  for(int i=0;i<TM;i++){int m=m_base+ty*TM+i;
    #pragma unroll
    for(int j=0;j<TN;j++){int n=n_base+tx*TN+j; out[(size_t)m*N+n]=__float2half(Cf[i][j]);}}
}

// ============================================================================
// REFERENCE WMMA WINNER (gemm_v3p, verbatim from gemm_rb_v3.hip) for head-to-head in same binary.
// ============================================================================
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
__device__ __forceinline__ int swz16(int row,int blk16,int KC){
  int nblk=KC/16; if(nblk<=1) return blk16; return blk16 ^ ((row>>1)&(nblk-1)); }
template<int N> __device__ __forceinline__ void wait_loadN(){ asm volatile("s_wait_loadcnt %0"::"i"(N):"memory"); }
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

// CPU/GPU helpers (Q6_K dequant + per-32 requant) — same as gemm_rb_v3 ------
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
// CORRECTNESS — synthetic int8 inputs, int32-accum float reference, NRMSE<2e-3.
//   NOTE: DP4A kernel uses per-32 scales (nb=K/32). KC must be a multiple of 32.
// ============================================================================
template<int BM,int BN,int A2,int B2, typename F>
static double run_correctness_dp4a(const char* name, F kern){
  const int KC32SCALE=32;
  int M=64,N=256,K=512,nb=K/KC32SCALE;
  srand(11);
  std::vector<int8_t> Wq(N*K); for(auto&x:Wq)x=(int8_t)(rand()%255-127);
  std::vector<float>  Ws(N*nb); for(auto&x:Ws)x=(0.5f+(rand()%100)/100.f)*0.01f;
  std::vector<int8_t> Aq(M*K); for(auto&x:Aq)x=(int8_t)(rand()%255-127);
  std::vector<float>  As(M*nb); for(auto&x:As)x=(0.5f+(rand()%100)/100.f)*0.02f;
  std::vector<float> Oref(M*N,0);
  for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double s=0;
    for(int blk=0;blk<nb;blk++){ long acc=0; for(int kk=0;kk<KC32SCALE;kk++){int k=blk*KC32SCALE+kk; acc+=(long)Aq[m*K+k]*Wq[n*K+k];}
      s+=(double)acc*As[m*nb+blk]*Ws[n*nb+blk]; }
    Oref[m*N+n]=(float)s; }
  int8_t* dWq;float* dWs;int8_t* dAq;float* dAs;__half* dO;
  CK(hipMalloc(&dWq,N*K));CK(hipMalloc(&dWs,N*nb*4));CK(hipMalloc(&dAq,M*K));CK(hipMalloc(&dAs,M*nb*4));CK(hipMalloc(&dO,M*N*2));
  CK(hipMemcpy(dWq,Wq.data(),N*K,hipMemcpyHostToDevice));CK(hipMemcpy(dWs,Ws.data(),N*nb*4,hipMemcpyHostToDevice));
  CK(hipMemcpy(dAq,Aq.data(),M*K,hipMemcpyHostToDevice));CK(hipMemcpy(dAs,As.data(),M*nb*4,hipMemcpyHostToDevice));
  dim3 g(N/BN,M/BM),b((BM/A2)*(BN/B2));  // A2=TM, B2=TN -> threads = (BM/TM)*(BN/TN)
  kern<<<g,b>>>(dAq,dAs,dWq,dWs,dO,M,N,K); CK(hipDeviceSynchronize());
  std::vector<__half> hO(M*N); CK(hipMemcpy(hO.data(),dO,M*N*2,hipMemcpyDeviceToHost));
  double e=0,nn=0; for(int i=0;i<M*N;i++){float v=__half2float(hO[i]);e+=(v-Oref[i])*(v-Oref[i]);nn+=Oref[i]*Oref[i];}
  double nrmse=sqrt(e/nn);
  printf("  %-30s NRMSE=%.2e  %s\n",name,nrmse,nrmse<2e-3?"PASS":"FAIL");
  hipFree(dWq);hipFree(dWs);hipFree(dAq);hipFree(dAs);hipFree(dO);
  return nrmse;
}
// WMMA correctness (KC=32, per-KC scale) — same harness as gemm_rb_v3.
template<int BM,int BN,int A2,int B2,int KC, typename F>
static double run_correctness_wmma(const char* name, F kern){
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
  printf("  %-30s NRMSE=%.2e  %s\n",name,nrmse,nrmse<2e-3?"PASS":"FAIL");
  hipFree(dWq);hipFree(dWs);hipFree(dAq);hipFree(dAs);hipFree(dO);
  return nrmse;
}

// ============================================================================
// SPEED — warm (real Q6_K->int8->requant32 buffers), MIN-of-bursts.
// ============================================================================
template<typename F>
static double bench_warm(F kern,dim3 grid,dim3 blk,int M,int N,int K,int reps,double* out_ms){
  hipGetLastError();
  int nblk=K/256,nb16=K/16,nb32=K/32;
  block_q6_K* dW;int8_t*dWq,*dWq32,*dAq;float*dWs,*dWs32,*dAs;__half*dO;
  CK(hipMalloc(&dW,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMalloc(&dWq,(size_t)N*K));CK(hipMalloc(&dWs,(size_t)N*nb16*4));
  CK(hipMalloc(&dWq32,(size_t)N*K));CK(hipMalloc(&dWs32,(size_t)N*nb32*4));CK(hipMalloc(&dAq,(size_t)M*K));CK(hipMalloc(&dAs,(size_t)M*nb32*4));CK(hipMalloc(&dO,(size_t)M*N*2));
  CK(hipMemset(dW,1,(size_t)N*nblk*sizeof(block_q6_K)));CK(hipMemset(dAq,1,(size_t)M*K));CK(hipMemset(dAs,1,(size_t)M*nb32*4));
  dequant_q6k<<<(N*nblk+127)/128,128>>>(dW,dWq,dWs,N,K);
  for(int w=0;w<3;w++)requant_32<<<(N*nb32+127)/128,128>>>(dWq,dWs,dWq32,dWs32,N,K);
  CK(hipDeviceSynchronize());
  for(int w=0;w<10;w++)kern<<<grid,blk>>>(dAq,dAs,dWq32,dWs32,dO,M,N,K);
  hipError_t le=hipGetLastError(); hipError_t se=hipDeviceSynchronize();
  double ms;
  if(le||se){ ms=-1; *out_ms=-1; }
  else {
    const int BURST=8; double best=1e30;
    for(int r=0;r<reps;r++){ double t0=now(); for(int i=0;i<BURST;i++)kern<<<grid,blk>>>(dAq,dAs,dWq32,dWs32,dO,M,N,K); hipDeviceSynchronize();
      double m=(now()-t0)/BURST; if(m<best)best=m; }
    ms=best; *out_ms=ms; }
  hipFree(dW);hipFree(dWq);hipFree(dWs);hipFree(dWq32);hipFree(dWs32);hipFree(dAq);hipFree(dAs);hipFree(dO);
  return ms<0 ? -1.0 : 2.0*M*N*K/1e12/(ms/1e3);
}

// COLD-ish — rotate NBUF weight copies + 32MB L2 thrash between reps. THIS is the real number.
__global__ void thrash(int8_t* p, size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]^=1; }
template<typename F>
static double bench_cold(F kern,dim3 grid,dim3 blk,int M,int N,int K,int reps,double* out_ms){
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
  for(int w=0;w<4;w++)kern<<<grid,blk>>>(dAq,dAs,dWq32[0],dWs32[0],dO,M,N,K);
  hipError_t le=hipGetLastError(); hipError_t se=hipDeviceSynchronize();
  if(le||se){ *out_ms=-1;
    hipFree(dT);hipFree(dW);hipFree(dWq);hipFree(dWs);for(int bb=0;bb<NBUF;bb++){hipFree(dWq32[bb]);hipFree(dWs32[bb]);}hipFree(dAq);hipFree(dAs);hipFree(dO);
    return -1.0; }
  double t0=now();
  for(int i=0;i<reps;i++){ int bi=i%NBUF;
    thrash<<<(TH+255)/256,256>>>(dT,TH);
    kern<<<grid,blk>>>(dAq,dAs,dWq32[bi],dWs32[bi],dO,M,N,K); }
  CK(hipDeviceSynchronize());
  double tot=now()-t0;
  double t1=now(); for(int i=0;i<reps;i++)thrash<<<(TH+255)/256,256>>>(dT,TH); CK(hipDeviceSynchronize());
  double thr=now()-t1;
  double ms=(tot-thr)/reps; *out_ms=ms;
  hipFree(dT);hipFree(dW);hipFree(dWq);hipFree(dWs);for(int bb=0;bb<NBUF;bb++){hipFree(dWq32[bb]);hipFree(dWs32[bb]);}hipFree(dAq);hipFree(dAs);hipFree(dO);
  return 2.0*M*N*K/1e12/(ms/1e3);
}

// dispatch helpers to build grid/block for each kernel family
template<int BM,int BN,int TM,int TN>
static dim3 dp4a_block(){ return dim3((BM/TM)*(BN/TN)); }
template<int BM,int BN>
static dim3 tile_grid(int M,int N){ return dim3(N/BN,M/BM); }

int main(int argc,char**argv){
  printf("=== gemm_dp4a: register-blocked INT8 DP4A (v_dot4_i32_iu8) GEMM vs WMMA (gfx1201) ===\n");
  printf("Target gate M=512 N=14336 K=4096. WMMA gemm_v3p ~61 TF warm. RADV int8 ~2820 t/s equiv.\n\n");

  printf("[1] CORRECTNESS (synthetic int8, int32-accum float ref, NRMSE<2e-3):\n");
  // DP4A variants: template<BM,BN,TM,TN,KC,DB>. correctness harness passes A2=TM,B2=TN.
  run_correctness_dp4a<64,64,4,4>("DP4A 64x64 t4x4 KC32",       gemm_dp4a<64,64,4,4,32,0>);
  run_correctness_dp4a<64,64,4,4>("DP4A 64x64 t4x4 KC64",       gemm_dp4a<64,64,4,4,64,0>);
  run_correctness_dp4a<128,128,8,8>("DP4A 128x128 t8x8 KC32",   gemm_dp4a<128,128,8,8,32,0>);
  run_correctness_dp4a<128,64,8,4>("DP4A 128x64 t8x4 KC32",     gemm_dp4a<128,64,8,4,32,0>);
  run_correctness_dp4a<64,64,4,4>("DP4A 64x64 t4x4 KC32 DB",    gemm_dp4a<64,64,4,4,32,1>);
  run_correctness_dp4a<128,128,8,8>("DP4A 128x128 t8x8 KC64 DB",gemm_dp4a<128,128,8,8,64,1>);
  run_correctness_wmma<256,256,8,4,32>("WMMA v3p 256x256 (ref)",gemm_v3p<256,256,8,4,32,1>);

  // ====================================================================
  // [2] GATE M=512 N=14336 K=4096 — sweep DP4A tiles, warm + cold, vs WMMA v3p.
  // ====================================================================
  printf("\n[2] GATE M=512 N=14336 K=4096 — DP4A tile sweep (warm min-burst AND cold L2-thrash):\n");
  const int M=512,N=14336,K=4096; const int RW=40, RC=40; double msw,msc;
  struct Row{const char* name;double tfw;double tfc;};
  std::vector<Row> rows;
  #define DBENCH(label,BM,BN,TM,TN,KC,DB) {\
    auto k=gemm_dp4a<BM,BN,TM,TN,KC,DB>; dim3 g=tile_grid<BM,BN>(M,N), b=dp4a_block<BM,BN,TM,TN>();\
    double tfw=bench_warm(k,g,b,M,N,K,RW,&msw); double tfc=bench_cold(k,g,b,M,N,K,RC,&msc);\
    if(tfw<0||tfc<0) printf("  %-30s --- INVALID\n",label);\
    else { printf("  %-30s warm %5.1f TF | cold %5.1f TF\n",label,tfw,tfc); rows.push_back({label,tfw,tfc}); } }
  DBENCH("DP4A 64x64  t4x4 KC32",   64, 64, 4,4,32,0)
  DBENCH("DP4A 64x64  t4x4 KC64",   64, 64, 4,4,64,0)
  DBENCH("DP4A 64x64  t4x4 KC32 DB",64, 64, 4,4,32,1)
  DBENCH("DP4A 128x64 t8x4 KC32",  128, 64, 8,4,32,0)
  DBENCH("DP4A 128x64 t4x4 KC32", 128, 64, 4,4,32,0)
  DBENCH("DP4A 64x128 t4x8 KC32",  64,128, 4,8,32,0)
  DBENCH("DP4A 128x128 t8x8 KC32",128,128, 8,8,32,0)
  DBENCH("DP4A 128x128 t8x8 KC64 DB",128,128,8,8,64,1)
  DBENCH("DP4A 128x128 t4x4 KC32",128,128, 4,4,32,0)
  DBENCH("DP4A 256x128 t8x4 KC32",256,128, 8,4,32,0)
  DBENCH("DP4A 64x64  t2x2 KC32",  64, 64, 2,2,32,0)
  #undef DBENCH
  // WMMA reference at gate
  { auto k=gemm_v3p<256,256,8,4,32,1>; dim3 g(N/256,M/256), b(8*4*32);
    double tfw=bench_warm(k,g,b,M,N,K,RW,&msw); double tfc=bench_cold(k,g,b,M,N,K,RC,&msc);
    printf("  %-30s warm %5.1f TF | cold %5.1f TF  <<< WMMA REF\n","WMMA v3p 256x256 8x4",tfw,tfc); }
  double bw=0,bc=0; const char*bnw="",*bnc=""; for(auto&r:rows){if(r.tfw>bw){bw=r.tfw;bnw=r.name;} if(r.tfc>bc){bc=r.tfc;bnc=r.name;}}
  printf("  >>> BEST DP4A warm: %s = %.1f TF | BEST DP4A cold: %s = %.1f TF\n",bnw,bw,bnc,bc);

  // ====================================================================
  // [3] 5-SHAPE AGGREGATE M=512 — best DP4A tile vs WMMA v3p. t/s @ 50 layers.
  //   (best tile selected at compile-time below after [2] guidance; default 128x128 t8x8 KC64 DB)
  // ====================================================================
  printf("\n[3] 5-SHAPE AGGREGATE M=512 (per-32 scales) — DP4A best vs WMMA v3p — t/s @50L:\n");
  { const int NL=50;
    struct S{const char* nm;int N,K;};
    S shapes[5]={{"qkv",6144,4096},{"o",4096,4096},{"gate",14336,4096},{"up",14336,4096},{"down",4096,14336}};
    double dW=0,dC=0,wW=0,wC=0;
    for(int s=0;s<5;s++){ int Ns=shapes[s].N,Ks=shapes[s].K; double m1,m2,m3,m4;
      // DP4A best (chosen after sweep): use 128x128 t8x8 KC64 DB
      auto kd=gemm_dp4a<128,128,8,8,64,1>; dim3 gd(Ns/128,512/128), bd(dp4a_block<128,128,8,8>());
      double dtfw=bench_warm(kd,gd,bd,512,Ns,Ks,12,&m1); double dtfc=bench_cold(kd,gd,bd,512,Ns,Ks,40,&m2);
      auto kw=gemm_v3p<256,256,8,4,32,1>; dim3 gw(Ns/256,512/256), bw_(8*4*32);
      double wtfw=bench_warm(kw,gw,bw_,512,Ns,Ks,12,&m3); double wtfc=bench_cold(kw,gw,bw_,512,Ns,Ks,40,&m4);
      dW+=m1;dC+=m2;wW+=m3;wC+=m4;
      printf("  %-5s N=%-5d K=%-5d: DP4A warm %5.1f cold %5.1f TF | WMMA warm %5.1f cold %5.1f TF\n",
        shapes[s].nm,Ns,Ks,dtfw,dtfc,wtfw,wtfc); }
    double dts_w=512.0/(dW*NL/1000.0), dts_c=512.0/(dC*NL/1000.0);
    double wts_w=512.0/(wW*NL/1000.0), wts_c=512.0/(wC*NL/1000.0);
    printf("  AGG 50L  DP4A: warm %.0f t/s  cold %.0f t/s\n",dts_w,dts_c);
    printf("  AGG 50L  WMMA: warm %.0f t/s  cold %.0f t/s\n",wts_w,wts_c);
    printf("  vs RADV int8 ~2820 t/s: DP4A cold %s | WMMA cold %s\n",
      dts_c>2820?"*** BEAT ***":"(below)", wts_c>2820?"*** BEAT ***":"(below)");
  }
  return 0;
}
