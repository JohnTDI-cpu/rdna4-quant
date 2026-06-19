// full_forward_q6k.hip — SYNTHETIC FULL-FORWARD PREFILL TIMING for Bielik-11B (Q6_K weights),
//   gfx1201 / RDNA4 / HIP-ROCm. GPU0 only.
//
//   GOAL: wire ALL the ggml-johnv8 prefill kernels into a real 50-layer forward and MEASURE the
//   end-to-end prefill t/s (pp512/2048/8192), including EVERY kernel-dispatch gap (~8 GEMM/quant
//   kernels + 50 FA2 launches + RoPE per layer => ~400 launches/forward). This converts the
//   per-component "+12%/+7%/+1% vs RADV" ESTIMATE into a measured number. Random int8 weights+scales
//   (NO GGUF loader): per-component correctness is already validated in the source files; this is a
//   pure WALL-TIME measurement, so weight VALUES are irrelevant — only the shapes & kernel sequence.
//
//   Per-layer dataflow (Bielik-11B: NE=4096 NH=32 NKV=8 HD=128 NFF=14336 NL=50):
//     x[M,4096]
//       -> rmsnorm+quant (attn norm)                      [rmsnorm_quant]          standalone (qkv N=6144 break-even)
//       -> qkv GEMM  N=6144 K=4096  -> Q,K,V split        [gemm_rb_kc<...,32>]
//       -> RoPE(Q,K)  (half, cheap)                        [rope_qk]               (new, written here)
//       -> FA2 (causal, GQA H=32 Hk=8 D=128)               [launch_flash_prefill_v1]
//       -> quant(FA2 out, N=4096)                          [quant_only]            standalone
//       -> o GEMM    N=4096 K=4096  + RESIDUAL epilogue    [gemm_rb_q8_residual]   (x += o)
//       -> rmsnorm+quant (ffn norm)                        [rmsnorm_quant]
//       -> gate GEMM N=14336 K=4096                        [gemm_rb_kc<...,32>]    standalone (gate fusion LOSES -21%)
//       -> up   GEMM N=14336 K=4096                        [gemm_rb_kc<...,32>]
//       -> silu(gate)*up + quant  (N=14336)                [silumul_quant]         standalone (down fusion LOSES -8%)
//       -> down GEMM N=4096 K=14336 + RESIDUAL epilogue    [gemm_rb_q8_residual]   (x += down)
//
//   ADAPTIVE FUSION POLICY (from fused_normquant.hip MEASURED results):
//     * residual-add EPILOGUE on o-proj and down-proj   -> ALWAYS win (single-consumer, zero redundancy)
//     * rmsnorm+quant as a STANDALONE kernel for the attn/ffn norms feeding qkv & gate/up
//       (qkv N=6144 = break-even, gate/up N=14336 -> prologue fusion LOSES -21%, so standalone).
//     * o-proj here is fed by quant(FA2 out) (NOT an rmsnorm) -> the +43% normquant win does not apply
//       (no norm to fuse); we use standalone quant + the residual GEMM. We ALSO measure an
//       alternative where the attn-norm->qkv would be fused, to expose the prologue tradeoff.
//
//   Build: HIP_VISIBLE_DEVICES=0 hipcc --offload-arch=gfx1201 -O3 full_forward_q6k.hip -o /tmp/ff && \
//          HIP_VISIBLE_DEVICES=0 /tmp/ff
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>

typedef int i4  __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef __half2 h2;
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
#define CK(x) do{hipError_t e=(x);if(e){printf("HIP err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

// FA2 (flash_prefill v1): launch_flash_prefill_v1(Q,K,V,O,M,H,Hk,D,stream). Layout [H,M,D].
#include "/home/janusz/AMD MXFP4/hip_int4/flash_prefill.hip"

// =====================================================================================
//  KERNELS (copied verbatim from the validated source files — speed-equivalent, same ISA)
//   gemm_rb_kc<BM,BN,WMW,WNW,KC>   : KC-templated scale-aware int8 WMMA GEMM (prefill_q6k.hip)
//   gemm_rb_q8_residual<...>       : Q8 GEMM with x+=result residual epilogue (fused_normquant.hip)
//   rmsnorm_quant / silumul_quant  : standalone mem-op quant kernels (fused_normquant.hip)
// =====================================================================================

template<int BM,int BN,int WMW,int WNW,int KC>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_rb_kc(
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

// Q8 GEMM with x += result residual epilogue (out PRE-LOADED with residual stream).
template<int BM,int BN,int WMW,int WNW>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_rb_q8_residual(
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
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){
    int m=m_base+wm*WMT*16+mt*16+kg*8+j; size_t o=(size_t)m*N+n_base+wn*WNT*16+nt*16+nl;
    out[o]=__float2half(__half2float(out[o])+Rf[mt][nt][j]); }
}

// rmsnorm + per-32 int8 quant (one block per row).
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

// plain per-32 int8 quant of a half activation (the FA2-output -> o-proj path: no norm). One block/row.
__global__ void quant_only(const __half* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ s, int M, int K){
  int row=blockIdx.x; if(row>=M) return; int tid=threadIdx.x, nth=blockDim.x; int nb=K/32;
  const __half* xr=x+(size_t)row*K;
  for(int b=tid;b<nb;b+=nth){ float vals[32],am=0;
    for(int e=0;e<32;e++){int k=b*32+e; float v=__half2float(xr[k]); vals[e]=v; float a=fabsf(v); if(a>am)am=a;}
    float d=am/127.f, id=d>0?1.f/d:0.f; s[(size_t)row*nb+b]=d;
    for(int e=0;e<32;e++){int vq=(int)lrintf(vals[e]*id); if(vq>127)vq=127; if(vq<-127)vq=-127; q[(size_t)row*32*nb+b*32+e]=(int8_t)vq;} }
}

// silumul(gate,up) + per-32 int8 quant (down-proj activation). One block per row. K==NFF.
__global__ void silumul_quant(const __half* __restrict__ g, const __half* __restrict__ u,
                              int8_t* __restrict__ q, float* __restrict__ s, int M, int K){
  int row=blockIdx.x; if(row>=M)return; int tid=threadIdx.x,nth=blockDim.x; int nb=K/32;
  const __half* gr=g+(size_t)row*K; const __half* ur=u+(size_t)row*K;
  for(int b=tid;b<nb;b+=nth){ float vals[32],am=0;
    for(int e=0;e<32;e++){int k=b*32+e; float gv=__half2float(gr[k]); float v=(gv/(1.f+expf(-gv)))*__half2float(ur[k]); vals[e]=v; float a=fabsf(v); if(a>am)am=a;}
    float d=am/127.f, id=d>0?1.f/d:0.f; s[(size_t)row*nb+b]=d;
    for(int e=0;e<32;e++){int vq=(int)lrintf(vals[e]*id); if(vq>127)vq=127; if(vq<-127)vq=-127; q[(size_t)row*32*nb+b*32+e]=(int8_t)vq;} }
}

// =====================================================================================
//  RoPE — rotate (x[2i], x[2i+1]) pairs of each Q & K head by angle pos*base^(-2i/HD).
//  Q layout [M, NH*HD]=[M,4096], K layout [M, NKV*HD]=[M,1024] (token-major, as qkv GEMM writes).
//  Cheap half kernel: grid = (M, NH+NKV-heads-worth) flattened. One thread per pair.
//  NOTE: qkv GEMM writes out[M, N=6144] = [Q(4096) | K(1024) | V(1024)] column-concatenated.
//  We RoPE Q in-place over its 4096 cols and K in-place over its 1024 cols of the SAME buffer.
// =====================================================================================
__global__ void rope_qk(__half* __restrict__ qkv, int M, int N, int HD, int n_q_cols, int n_k_cols, float theta){
  // each thread handles ONE rotation pair in ONE token. total pairs = M*(n_q_cols+n_k_cols)/2.
  int total_pairs = M*((n_q_cols+n_k_cols)/2);
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid>=total_pairs) return;
  int pairs_per_tok = (n_q_cols+n_k_cols)/2;     // 2048(Q) + 512(K)
  int tok = gid / pairs_per_tok;
  int p   = gid % pairs_per_tok;                  // pair index within the token's Q||K region
  int half = HD/2;                                // 64
  int col_base, head_local;
  if(p < n_q_cols/2){ col_base = 0;        head_local = p;            }   // Q region [0,4096)
  else              { col_base = n_q_cols; head_local = p - n_q_cols/2;}  // K region [4096,4096+1024)
  int head = head_local / half;                   // which head
  int i    = head_local % half;                   // pair index inside head (0..63)
  // angle = pos * theta^(-2i/HD)
  float freq = powf(theta, -2.f*(float)i/(float)HD);
  float ang  = (float)tok * freq;
  float c = cosf(ang), s = sinf(ang);
  int c0 = col_base + head*HD + i;                // x[2i]   (first half of head)
  int c1 = c0 + half;                             // x[2i+1] (second half — NeoX/GPT style pairing)
  size_t base = (size_t)tok*N;
  float x0 = __half2float(qkv[base+c0]);
  float x1 = __half2float(qkv[base+c1]);
  qkv[base+c0] = __float2half(x0*c - x1*s);
  qkv[base+c1] = __float2half(x0*s + x1*c);
}

// split the column-concatenated qkv[M,6144] into the FA2-required [H,M,D] Q,K,V buffers.
// (In a real engine the qkv GEMM would write directly to these; here we add an explicit repack
//  kernel so the FA2 layout is exact. It is a real per-forward cost, so it is INCLUDED.)
__global__ void split_qkv_to_fa2(const __half* __restrict__ qkv, __half* __restrict__ Q,
                                 __half* __restrict__ K, __half* __restrict__ V,
                                 int M, int N, int NH, int NKV, int HD){
  // one thread per element of the [M,N] qkv buffer
  int gid = blockIdx.x*blockDim.x+threadIdx.x; if(gid>=M*N) return;
  int tok = gid / N, col = gid % N;
  int qcols = NH*HD, kcols = NKV*HD;             // 4096, 1024
  __half v = qkv[(size_t)tok*N+col];
  if(col < qcols){ int h=col/HD, d=col%HD; Q[((size_t)h*M+tok)*HD+d]=v; }
  else if(col < qcols+kcols){ int c=col-qcols; int h=c/HD, d=c%HD; K[((size_t)h*M+tok)*HD+d]=v; }
  else { int c=col-qcols-kcols; int h=c/HD, d=c%HD; V[((size_t)h*M+tok)*HD+d]=v; }
}
// repack FA2 output [H,M,D] -> token-major [M, NH*HD] for the o-proj quant+GEMM.
__global__ void fa2_to_tokenmajor(const __half* __restrict__ O, __half* __restrict__ x,
                                  int M, int NH, int HD){
  int gid = blockIdx.x*blockDim.x+threadIdx.x; int N=NH*HD; if(gid>=M*N) return;
  int tok=gid/N, col=gid%N; int h=col/HD, d=col%HD;
  x[(size_t)tok*N+col]=O[((size_t)h*M+tok)*HD+d];
}

// =====================================================================================
//  Adaptive GEMM config picker: best <BM,BN,WMW,WNW> per shape (from prefill_q8/q6k sweeps),
//  with the M%BM==0 / N%BN==0 divisibility guard so it is valid for M in {512,2048,8192}.
//  All BN here divide 4096/6144/14336; all chosen BM divide 512/2048/8192 (256|512 div all).
// =====================================================================================

// Wrapper launchers — UNIFORM deployed config <256,256,8,4> (KC=32). The per-shape M=512 "best"
//   configs (<128,256> qkv, <512,128> o) REGRESS at M=2048/8192 (measured: qkv 93->120ms, o 66->86ms
//   at M=2048), because they are M=512-tuned. The single uniform <256,256> wins across all 3 contexts,
//   which is the realistic single-binary deployment. All BM=256 divide M in {512,2048,8192}; all BN
//   divide N in {6144,4096,14336}.
static void gemm_qkv (const int8_t*a,const float*as,const int8_t*w,const float*ws,__half*o,int M,int N,int K,hipStream_t st){
  dim3 g(N/256,M/256),b(8*4*32); gemm_rb_kc<256,256,8,4,32><<<g,b,0,st>>>(a,as,w,ws,o,M,N,K); }
static void gemm_gate(const int8_t*a,const float*as,const int8_t*w,const float*ws,__half*o,int M,int N,int K,hipStream_t st){
  dim3 g(N/256,M/256),b(8*4*32); gemm_rb_kc<256,256,8,4,32><<<g,b,0,st>>>(a,as,w,ws,o,M,N,K); }
static void gemm_o_residual(const int8_t*a,const float*as,const int8_t*w,const float*ws,__half*o,int M,int N,int K,hipStream_t st){
  dim3 g(N/256,M/256),b(8*4*32); gemm_rb_q8_residual<256,256,8,4><<<g,b,0,st>>>(a,as,w,ws,o,M,N,K); }
static void gemm_down_residual(const int8_t*a,const float*as,const int8_t*w,const float*ws,__half*o,int M,int N,int K,hipStream_t st){
  dim3 g(N/256,M/256),b(8*4*32); gemm_rb_q8_residual<256,256,8,4><<<g,b,0,st>>>(a,as,w,ws,o,M,N,K); }

// =====================================================================================
//  Per-context full-forward harness.
// =====================================================================================
struct Buffers {
  // weights (random int8 + per-32 scale) — shared across all 50 layers (speed measurement)
  int8_t *wqkv,*wo,*wgate,*wup,*wdown;
  float  *sqkv,*so,*sgate,*sup,*sdown;
  float  *wn_attn,*wn_ffn;                 // rmsnorm weights
  // activations
  __half *x;                               // residual stream [M,4096]
  int8_t *aq; float *as;                   // quantized activation [M,4096] for qkv (K=4096)
  __half *qkv;                             // qkv GEMM out [M,6144]
  __half *Q,*K,*V,*Oattn;                  // FA2 [H,M,D]/[Hk,M,D]
  __half *attn_tm;                         // FA2 out token-major [M,4096]
  int8_t *aq_o; float *as_o;               // quant of attn_tm for o-proj
  int8_t *aq_ffn; float *as_ffn;           // ffn-norm quant [M,4096]
  __half *gate,*up;                        // [M,14336]
  int8_t *aq_dn; float *as_dn;             // silu+quant [M,14336] for down
};

static void alloc_buffers(Buffers&B,int M,int NE,int NH,int NKV,int HD,int NFF){
  int Nqkv=NH*HD+2*NKV*HD; // 6144
  CK(hipMalloc(&B.wqkv,(size_t)Nqkv*NE));      CK(hipMalloc(&B.sqkv,(size_t)Nqkv*(NE/32)*4));
  CK(hipMalloc(&B.wo,(size_t)NE*NE));          CK(hipMalloc(&B.so,(size_t)NE*(NE/32)*4));
  CK(hipMalloc(&B.wgate,(size_t)NFF*NE));      CK(hipMalloc(&B.sgate,(size_t)NFF*(NE/32)*4));
  CK(hipMalloc(&B.wup,(size_t)NFF*NE));        CK(hipMalloc(&B.sup,(size_t)NFF*(NE/32)*4));
  CK(hipMalloc(&B.wdown,(size_t)NE*NFF));      CK(hipMalloc(&B.sdown,(size_t)NE*(NFF/32)*4));
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
  // fill with benign nonzero data
  CK(hipMemset(B.wqkv,1,(size_t)Nqkv*NE)); CK(hipMemset(B.wo,1,(size_t)NE*NE));
  CK(hipMemset(B.wgate,1,(size_t)NFF*NE)); CK(hipMemset(B.wup,1,(size_t)NFF*NE)); CK(hipMemset(B.wdown,1,(size_t)NE*NFF));
  CK(hipMemset(B.sqkv,0,(size_t)Nqkv*(NE/32)*4)); CK(hipMemset(B.so,0,(size_t)NE*(NE/32)*4));
  CK(hipMemset(B.sgate,0,(size_t)NFF*(NE/32)*4)); CK(hipMemset(B.sup,0,(size_t)NFF*(NE/32)*4)); CK(hipMemset(B.sdown,0,(size_t)NE*(NFF/32)*4));
  CK(hipMemset(B.wn_attn,0x3c,(size_t)NE*4)); CK(hipMemset(B.wn_ffn,0x3c,(size_t)NE*4));
  CK(hipMemset(B.x,1,(size_t)M*NE*2));
}
static void free_buffers(Buffers&B){
  hipFree(B.wqkv);hipFree(B.sqkv);hipFree(B.wo);hipFree(B.so);hipFree(B.wgate);hipFree(B.sgate);
  hipFree(B.wup);hipFree(B.sup);hipFree(B.wdown);hipFree(B.sdown);hipFree(B.wn_attn);hipFree(B.wn_ffn);
  hipFree(B.x);hipFree(B.aq);hipFree(B.as);hipFree(B.qkv);hipFree(B.Q);hipFree(B.K);hipFree(B.V);hipFree(B.Oattn);
  hipFree(B.attn_tm);hipFree(B.aq_o);hipFree(B.as_o);hipFree(B.aq_ffn);hipFree(B.as_ffn);
  hipFree(B.gate);hipFree(B.up);hipFree(B.aq_dn);hipFree(B.as_dn);
}

// timing breakdown accumulator (ms, per WHOLE forward = 50 layers)
struct Times { double rmsq, qkv, rope, split, fa2, fa2pack, quanto, oproj, ffnq, gate, up, siluq, down; };

// run ONE full layer; if T!=nullptr, time each stage with device syncs (instrumented = SLOWER, used
// only for the breakdown). The reported t/s uses the UN-instrumented path (one sync per forward).
static void run_layer(Buffers&B,int M,int NE,int NH,int NKV,int HD,int NFF,float eps,float theta,
                      Times*T,hipStream_t st){
  int Nqkv=NH*HD+2*NKV*HD;                  // 6144
  int n_q_cols=NH*HD, n_k_cols=NKV*HD;      // 4096, 1024
  int tpb=256;
  auto sync=[&](double t0,double&acc){ if(T){CK(hipStreamSynchronize(st)); acc+=now()-t0;} };
  double t0;
  // 1) attn rmsnorm + quant  (standalone — qkv N=6144 break-even)
  t0=now(); rmsnorm_quant<<<M,256,0,st>>>(B.x,B.wn_attn,B.aq,B.as,M,NE,eps); sync(t0,T?T->rmsq:t0);
  // 2) qkv GEMM  N=6144 K=4096
  t0=now(); gemm_qkv(B.aq,B.as,B.wqkv,B.sqkv,B.qkv,M,Nqkv,NE,st); sync(t0,T?T->qkv:t0);
  // 3) RoPE(Q,K) in-place on the qkv buffer
  { int pairs=M*((n_q_cols+n_k_cols)/2); int g=(pairs+tpb-1)/tpb;
    t0=now(); rope_qk<<<g,tpb,0,st>>>(B.qkv,M,Nqkv,HD,n_q_cols,n_k_cols,theta); sync(t0,T?T->rope:t0); }
  // 4) split qkv -> FA2 [H,M,D] layout
  { int g=(M*Nqkv+tpb-1)/tpb;
    t0=now(); split_qkv_to_fa2<<<g,tpb,0,st>>>(B.qkv,B.Q,B.K,B.V,M,Nqkv,NH,NKV,HD); sync(t0,T?T->split:t0); }
  // 5) FA2
  t0=now(); launch_flash_prefill_v1(B.Q,B.K,B.V,B.Oattn,M,NH,NKV,HD,st); sync(t0,T?T->fa2:t0);
  // 6) FA2 out -> token-major [M,4096]
  { int g=(M*NE+tpb-1)/tpb;
    t0=now(); fa2_to_tokenmajor<<<g,tpb,0,st>>>(B.Oattn,B.attn_tm,M,NH,HD); sync(t0,T?T->fa2pack:t0); }
  // 7) quant(attn out)  (standalone — no norm before o-proj)
  t0=now(); quant_only<<<M,256,0,st>>>(B.attn_tm,B.aq_o,B.as_o,M,NE); sync(t0,T?T->quanto:t0);
  // 8) o GEMM N=4096 K=4096 + residual epilogue (x += o)
  t0=now(); gemm_o_residual(B.aq_o,B.as_o,B.wo,B.so,B.x,M,NE,NE,st); sync(t0,T?T->oproj:t0);
  // 9) ffn rmsnorm + quant (standalone — gate/up N=14336 prologue-fusion LOSES)
  t0=now(); rmsnorm_quant<<<M,256,0,st>>>(B.x,B.wn_ffn,B.aq_ffn,B.as_ffn,M,NE,eps); sync(t0,T?T->ffnq:t0);
  // 10) gate GEMM N=14336 K=4096
  t0=now(); gemm_gate(B.aq_ffn,B.as_ffn,B.wgate,B.sgate,B.gate,M,NFF,NE,st); sync(t0,T?T->gate:t0);
  // 11) up GEMM N=14336 K=4096
  t0=now(); gemm_gate(B.aq_ffn,B.as_ffn,B.wup,B.sup,B.up,M,NFF,NE,st); sync(t0,T?T->up:t0);
  // 12) silu(gate)*up + quant (standalone — down N=4096 but large K=14336 prologue-fusion LOSES)
  t0=now(); silumul_quant<<<M,256,0,st>>>(B.gate,B.up,B.aq_dn,B.as_dn,M,NFF); sync(t0,T?T->siluq:t0);
  // 13) down GEMM N=4096 K=14336 + residual epilogue (x += down)
  t0=now(); gemm_down_residual(B.aq_dn,B.as_dn,B.wdown,B.sdown,B.x,M,NE,NFF,st); sync(t0,T?T->down:t0);
}

int main(){
  const int NE=4096, NH=32, NKV=8, HD=128, NFF=14336, NL=50;
  const float eps=1e-5f, theta=1e6f;       // Bielik/Qwen3 rope base ~1e6
  int ctxs[]={512,2048,8192};
  hipStream_t st; CK(hipStreamCreate(&st));

  printf("=== ggml-johnv8 FULL-FORWARD PREFILL (Bielik-11B Q6_K-weights, gfx1201/RDNA4) ===\n");
  printf("dims: NE=%d NH=%d NKV=%d HD=%d NFF=%d NL=%d | ADAPTIVE fusion (residual epilogues, standalone quant)\n\n",
         NE,NH,NKV,HD,NFF,NL);
  printf("RADV -fa baseline:  pp512=2307  pp2048=2223  pp8192=1938 t/s\n");
  printf("ROCm     baseline:  pp512=1655  pp2048=1620  pp8192=1462 t/s\n\n");

  for(int ci=0;ci<3;ci++){
    int M=ctxs[ci];
    Buffers B; alloc_buffers(B,M,NE,NH,NKV,HD,NFF);

    // WARMUP — RDNA4 mclk ramps; run several full forwards warm.
    for(int w=0;w<3;w++){ for(int l=0;l<NL;l++) run_layer(B,M,NE,NH,NKV,HD,NFF,eps,theta,nullptr,st); }
    CK(hipStreamSynchronize(st));

    // MEASURE — full 50-layer forward, ONE sync per forward (counts ALL ~400 dispatch gaps).
    int R = (M>=8192)?5:10;
    double t0=now();
    for(int r=0;r<R;r++){ for(int l=0;l<NL;l++) run_layer(B,M,NE,NH,NKV,HD,NFF,eps,theta,nullptr,st); }
    CK(hipStreamSynchronize(st));
    double fwd_ms=(now()-t0)/R;
    double ts = M/(fwd_ms/1000.0);

    // BREAKDOWN — instrumented pass (per-stage device syncs; sums ~ forward but inflated by sync tax).
    Times T={0}; int RB=3;
    for(int w=0;w<2;w++) for(int l=0;l<NL;l++) run_layer(B,M,NE,NH,NKV,HD,NFF,eps,theta,nullptr,st);
    CK(hipStreamSynchronize(st));
    for(int r=0;r<RB;r++){ for(int l=0;l<NL;l++) run_layer(B,M,NE,NH,NKV,HD,NFF,eps,theta,&T,st); }
    double inv=1.0/RB;
    T.rmsq*=inv;T.qkv*=inv;T.rope*=inv;T.split*=inv;T.fa2*=inv;T.fa2pack*=inv;T.quanto*=inv;
    T.oproj*=inv;T.ffnq*=inv;T.gate*=inv;T.up*=inv;T.siluq*=inv;T.down*=inv;
    double instr_total=T.rmsq+T.qkv+T.rope+T.split+T.fa2+T.fa2pack+T.quanto+T.oproj+T.ffnq+T.gate+T.up+T.siluq+T.down;
    double gemm=T.qkv+T.oproj+T.gate+T.up+T.down;
    double quant=T.rmsq+T.quanto+T.ffnq+T.siluq;
    double repack=T.split+T.fa2pack;

    printf("---------------------------------------------------------------------------\n");
    printf("M=%d  | FULL FORWARD %d L : %.3f ms  => %.0f t/s\n", M, NL, fwd_ms, ts);
    const char* tag_rocm = ts>1655? (ts>2307?"BEATS RADV":"beats ROCm"):"below ROCm";
    printf("       vs RADV(%s): %+.1f%%   vs ROCm: %+.1f%%   [%s]\n",
           ci==0?"2307":ci==1?"2223":"1938",
           (ts/(ci==0?2307.0:ci==1?2223.0:1938.0)-1)*100,
           (ts/(ci==0?1655.0:ci==1?1620.0:1462.0)-1)*100, tag_rocm);
    printf("  BREAKDOWN (instrumented 50L, ms; note: per-stage syncs inflate vs the %.3f ms wall):\n",fwd_ms);
    printf("    GEMM total      %8.3f ms (%4.1f%%)  [qkv %.2f  o %.2f  gate %.2f  up %.2f  down %.2f]\n",
           gemm,gemm/instr_total*100,T.qkv,T.oproj,T.gate,T.up,T.down);
    printf("    quant/norm      %8.3f ms (%4.1f%%)  [attnRMSq %.2f  Qattn %.2f  ffnRMSq %.2f  siluQ %.2f]\n",
           quant,quant/instr_total*100,T.rmsq,T.quanto,T.ffnq,T.siluq);
    printf("    FA2             %8.3f ms (%4.1f%%)\n",T.fa2,T.fa2/instr_total*100);
    printf("    rope            %8.3f ms (%4.1f%%)\n",T.rope,T.rope/instr_total*100);
    printf("    repack(split+pk)%8.3f ms (%4.1f%%)\n",repack,repack/instr_total*100);
    printf("    instr sum       %8.3f ms  | dispatch/overlap delta vs wall = %.3f ms (%.1f%% of wall)\n",
           instr_total, instr_total-fwd_ms, (instr_total-fwd_ms)/fwd_ms*100);
    free_buffers(B);
  }
  hipStreamDestroy(st);
  return 0;
}
