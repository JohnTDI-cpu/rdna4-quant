// gemm_ngd.hip — NATIVE-GEMM-DEQUANT: WMMA INT8 prefill GEMM that reads NATIVE K-quant weights
//   (Q4_K = 4.5 bit, Q6_K = 6.5 bit) from global and unpacks/dequants to centered int8 INSIDE the
//   global->LDS staging stage (RADV's block_a_to_shmem pattern). This HALVES (Q4_K) / cuts 25% (Q6_K)
//   the weight bandwidth vs our current path that re-quants every weight to a full int8 byte at load.
//
//   THE LEVER: full forward is partly bandwidth-bound (measured L2-cold-weight tax ~22% at M=512; the
//   58MB weight is read once per layer from VRAM > 8MB L2). Reading 4-bit instead of 8-bit halves that
//   weight traffic -> roughly halves the 22% tax -> ~+11% at M=512. For Q4_K_M (int8 path 2611 LOSES to
//   RADV ~2696 by -2%) this should TIP it to a win.
//
//   WHAT CHANGED vs gemm_rb_v3.hip's winner gemm_v3p<256,256,8,4,KC32,SWZ1>:
//     * The B (weight) global->LDS path. Instead of reading pre-dequanted int8 from global, we read the
//       native packed quant block and unpack 4-bit nibbles (Q4_K) / 6-bit (Q6_K) into the SAME centered
//       int8 LDS tile the WMMA fragment loads expect. The A (activation) path, the WMMA inner loop, the
//       fragment pipeline, the XOR swizzle, the scales-outside-the-dot — ALL identical to v3p.
//     * KC = 32 = Q4_K's per-32 sub-block (one 6-bit sc + one 6-bit m per 32). So one K-block of the GEMM
//       == one Q4_K sub-block: the per-block weight scale d*sc and the asymmetric correction dmin*m are
//       directly the on-disk values, no re-quant, no merging.
//
//   Q4_K ASYMMETRIC FOLD (the key trick). Per element W = d*sc*nib - dmin*m  (nib in 0..15, NOT centered).
//   Per 32-block: sum_k A*W = d*sc*a_s*sum(a_q*nib) - dmin*m*a_s*sum(a_q).
//     -> stage int8 = nib (raw 0..15) in LDS; WMMA gives idot = sum(a_q*nib); scale by (d*sc)*a_s OUTSIDE.
//     -> carry the asymmetric term dmin*m * a_s * arowsum[m,blk] as a per-(m,n,blk) correction applied to
//        the FP32 accumulator AFTER the dot, exactly like prefill_q6k.hip keeps scales outside. arowsum =
//        per-(activation-row, block) int8 row-sum, precomputed once per GEMM (M*nb int32, tiny & cheap).
//   This is bit-for-bit the ggml Q4_K dequant (validated NRMSE < 2e-2 vs float ref below). Staging nib as
//   UNSIGNED int8 (0..15) is fine: iu8 WMMA on gfx12 takes the signedness flag; we pass the weight operand
//   as UNSIGNED (neg=false) and the activation as SIGNED -> the dot is sum(a_q_signed * nib_unsigned).
//
//   Q6_K NATIVE FOLD (symmetric). W = d*sc*(q6-32). q6 in 0..63 -> centered int8 = q6-32 in [-32,31], a
//   true signed int8. Stage (q6-32) directly; no asymmetric correction, no arowsum. KC=16 (Q6_K per-16
//   sub-block) OR KC=32 with the per-16 scales handled as two halves. We use KC=16 (exact) and KC=32 by
//   reusing the same per-16 scale into both halves is NOT valid, so Q6_K native uses KC=16.
//
//   Build:  HIP_VISIBLE_DEVICES=0 hipcc --offload-arch=gfx1201 -O3 gemm_ngd.hip -o /tmp/ngd
//           rocm-smi --setperflevel auto ; HIP_VISIBLE_DEVICES=0 /tmp/ngd
//   GPU: GPU0 (HIP_VISIBLE_DEVICES=0), R9700 gfx1201, ROCm 7.2.3. COLD = min-of-bursts w/ L2 thrash.
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

// ---------- Q-block formats (on-disk ggml) ----------
struct __attribute__((packed)) block_q4_K { __half d; __half dmin; uint8_t scales[12]; uint8_t qs[128]; }; // 144B/256
struct __attribute__((packed)) block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; __half d; }; // 210B/256

// ggml get_scale_min_k4: decode the 6-bit packed (sc,m) pair for sub-block j in [0,8).
__device__ __host__ __forceinline__ void gsm_d(int j,const uint8_t* q,uint8_t* sc,uint8_t* mn){
  if(j<4){*sc=q[j]&63;*mn=q[j+4]&63;}
  else   {*sc=(q[j+4]&0xF)|((q[j-4]>>6)<<4);*mn=(q[j+4]>>4)|((q[j]>>6)<<4);}
}

__device__ __forceinline__ int swz16(int row,int blk16,int KC){
  int nblk=KC/16; if(nblk<=1) return blk16; return blk16 ^ ((row>>1)&(nblk-1)); }

// ===================================================================================================
//  REFERENCE int8 GEMM (= gemm_v3p winner from gemm_rb_v3.hip). Reads pre-dequanted int8 weight.
//  Kept so the native path is benchmarked against the exact current winner in the same binary.
// ===================================================================================================
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

// ===================================================================================================
//  NATIVE Q4_K GEMM. Weight is the on-disk block_q4_K stream (144B/256). KC=32 (one Q4_K sub-block).
//  B global->LDS stage: read the 16 packed nibbles for this (n-row, 32-block), unpack to int8 nib 0..15,
//  store to LDS (swizzled). A path / WMMA / fragment-pipeline IDENTICAL to gemm_v3p. Scales d*sc and the
//  asymmetric dmin*m carried OUTSIDE the dot; the dmin*m*a_s*arowsum correction applied to FP32 accum.
//
//  Weight scale buffers (precomputed once at load, per (n-row, 32-block) = N*(K/32)):
//    wsc[n,blk] = d * sc[blk%8 within the 256-block]   (float)
//    wmn[n,blk] = dmin * m[blk%8]                       (float)
//  Activation: a_scale[m,blk] (per-32, like int8 path) + arowsum[m,blk] = sum of the 32 int8 a_q (int32).
//
//  LDS staging of nibbles: each Q4_K 256-block packs qs[128] = 256 nibbles. ggml layout: for the j-th
//  32-sub-block, byte index and low/high nibble selection follow the (group,half) scheme used in decode.
//  We replicate the decode unpack so the staged int8[k] == nib for global K index. Mapping for a 256-block
//  (8 sub-blocks of 32): sub-block j -> uses qs bytes [ (j/2)*32*... ]. Concretely (ggml reference, the
//  one in m4_full q4k):  pk = ((uint32_t*)qs)[within]; within in [0,32); group=within/8; the low nibble
//  feeds elements [group*64 .. ] and high nibble feeds [group*64+32 ..]. We unpack at 32-element-block
//  granularity to fill As/Bs[row][0..31] with the 32 nibbles of K index k0..k0+31.
// ===================================================================================================
//  Unpack helper: produce the 32 int8 nibbles for (Q4_K 256-block B, sub-block j in [0,8)) into out[32].
//  ggml Q4_K element k (0..255): is = k/32 (sub-block). Within the 256-block the nibble for global element
//  k is located so that elements 0..31 (j=0) and 32..63 (j=1) share qs bytes 0..31 (j=0 low, j=1 high of
//  the SAME bytes is NOT how ggml packs it). ggml packs: for n in {0,128}: for l in 0..31: lo=qs[n/2+l]&0xF
//  gives element (n+l) of sub-block pair... we mirror the exact decode used in embed_q4k / q4kc so the
//  per-32-block scale sc[j], m[j] line up with the 32 nibbles we emit.
__device__ __forceinline__ void q4k_unpack_sub(const block_q4_K* B, int j, int8_t out[32]){
  // ggml Q4_K: the 256 quants are stored as 8 sub-blocks of 32. The packing (matching get_scale_min_k4
  // sub-block index j and the q4kc/embed_q4k decode): qs is 128 bytes = 64 uint32. For sub-block j,
  // group=j/2 (0..3), each group covers 64 elements (low nibble = even sub-block, high nibble = odd).
  // within the group, the 32 uint32 words 'within'=0..31 hold 4 elements each in their low (or high) nib.
  // Element layout per the decode in m4_full: base=(group)*64 + wb, wb=(within&7)*4 ... but that decode
  // strides elements by 'within' across the 64-wide group. To stage 32 CONTIGUOUS K elements (one GEMM
  // sub-block = ggml sub-block j), we gather them in K order.
  const uint32_t* qs=(const uint32_t*)B->qs;   // 64 words
  int group=j>>1, isodd=j&1;                   // group 0..3, low/high nibble
  // group g covers uint32 words [g*8 .. g*8+8) for the first 32-half and the SAME words for the second
  // 32-half via low/high nibble. Element order: ggml sub-block j elements = for w in 0..7, for each of 4
  // bytes in word -> nibble. j even -> low nibble, j odd -> high nibble. K index within sub-block =
  // w*4 + byte. This matches q4k_warprow_i8: qs[8*k + w], k=j>>1, nib = isodd?hi:lo, av=xi[8*j+w].
  #pragma unroll
  for(int w=0;w<8;w++){
    uint32_t wb=qs[8*group+w];
    uint32_t nib=isodd?((wb>>4)&0x0F0F0F0Fu):(wb&0x0F0F0F0Fu);
    out[w*4+0]=(int8_t)( nib        & 0xF);
    out[w*4+1]=(int8_t)((nib >>  8) & 0xF);
    out[w*4+2]=(int8_t)((nib >> 16) & 0xF);
    out[w*4+3]=(int8_t)((nib >> 24) & 0xF);
  }
}

//  B-load = RAW qs bytes (coalesced, like int8); unpack nibbles ONLY at ds_store (RADV block_a_to_shmem).
//  For KC=32 (one Q4_K sub-block j), the 32 nibbles live in qs bytes [group*32 .. group*32+32) where
//  group=j/2; low nibble (j even) or high nibble (j odd). So per (row, 32-block) we load exactly 32 qs
//  bytes = i4*2 (32 bytes) into registers, then ds_store the 32 extracted nibbles. The PAIRED sub-block
//  re-reads the same 32 bytes (L2 hit). BV measured in 32-byte units (one row's qs-half per block).
//  Layout in Bs: row-major [BN][32], swizzled. We extract nib = (byte>>(4*odd))&0xF for all 32.
template<int BM,int BN,int WMW,int WNW,int SWZ>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_q4k_native(
    const int8_t* __restrict__ act, const float* __restrict__ a_scale, const int* __restrict__ a_rowsum,
    const block_q4_K* __restrict__ W, const float* __restrict__ w_sc, const float* __restrict__ w_mn,
    __half* __restrict__ out, int M, int N, int K){
  constexpr int KC=32;
  constexpr int NWARP=WMW*WNW, NTHREAD=NWARP*32, WMT=BM/(WMW*16), WNT=BN/(WNW*16);
  constexpr int AV=BM*KC/16, APT=(AV+NTHREAD-1)/NTHREAD;
  constexpr int KSUB=KC/16;               // 2
  // B raw-qs staging: BN rows, each needs 32 qs-bytes per 32-block. One i4(16B) per (row,half). 2 halves.
  constexpr int BVQ=BN*2, BPTQ=(BVQ+NTHREAD-1)/NTHREAD;   // # 16B qs-loads per block, per-thread count
  __shared__ int8_t As[2][BM][KC]; __shared__ int8_t Bs[2][BN][KC];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16, wm=warp/WNW, wn=warp%WNW;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/KC, nb256=K/256;
  float Rf[WMT][WNT][8];
  for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)for(int j=0;j<8;j++)Rf[a][b][j]=0.f;
  // pointer to the 32 qs bytes for (row rr, 32-block blk): block256=blk/8, j=blk&7, group=j/2 -> qs+group*32.
  auto qsptr=[&](int rr,int blk)->const uint8_t*{
    int blk256=blk>>3, j=blk&7, group=j>>1;
    return (const uint8_t*)(&W[(size_t)(n_base+rr)*nb256+blk256].qs[group*32]);
  };
  // load 32 raw qs bytes (2x i4) for a row's block into rq[0],rq[1]
  auto loadBraw=[&](i4 rq[2],int rr,int blk){ const uint8_t* p=qsptr(rr,blk); rq[0]=*(const i4*)p; rq[1]=*(const i4*)(p+16); };
  // ds_store the extracted nibbles. j&1 selects low/high nibble. swizzle on the 16B-half granularity.
  auto storeB=[&](int b,int rr,int blk,const i4 rq[2]){
    int odd=blk&1; int sh=odd?4:0;
    int8_t nb8[32];
    #pragma unroll
    for(int h=0;h<2;h++){ const uint8_t* pb=(const uint8_t*)&rq[h];
      #pragma unroll
      for(int t=0;t<16;t++) nb8[h*16+t]=(int8_t)((pb[t]>>sh)&0xF); }
    int c0=(SWZ?swz16(rr,0,KC):0)*16, c1=(SWZ?swz16(rr,1,KC):1)*16;
    *(i4*)&Bs[b][rr][c0]=*(const i4*)&nb8[0];
    *(i4*)&Bs[b][rr][c1]=*(const i4*)&nb8[16];
  };
  // prologue: A stage + B stage (block 0)
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+blk16*16];}}
  for(int p=0;p<BPTQ;p++){int idx=tid+p*NTHREAD; if(idx<BVQ){ int rr=idx>>1; i4 rq[2]; loadBraw(rq,rr,0); storeB(0,rr,0,rq);} }
  __syncthreads(); int buf=0;
  auto fetchA=[&](v2i Af[WMT],int b,int ks){
    #pragma unroll
    for(int mt=0;mt<WMT;mt++){int row=wm*WMT*16+mt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* ap=&As[b][row][bb*16+kg*8];Af[mt]=(v2i){*(const int*)ap,*(const int*)(ap+4)};} };
  auto fetchB=[&](v2i Bf[WNT],int b,int ks){
    #pragma unroll
    for(int nt=0;nt<WNT;nt++){int row=wn*WNT*16+nt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* bp=&Bs[b][row][bb*16+kg*8];Bf[nt]=(v2i){*(const int*)bp,*(const int*)(bp+4)};} };
  for(int k0=0;k0<K;k0+=KC){ int blk=k0/KC; i4 rA[APT]; i4 rBq[BPTQ][2]; int kn=k0+KC; int blkn=kn/KC;
    // prefetch next A AND next B-raw-qs into registers (hidden behind WMMA), like v3p's rA/rB.
    if(kn<K){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kn+c];}}
      for(int p=0;p<BPTQ;p++){int idx=tid+p*NTHREAD; if(idx<BVQ){ int rr=idx>>1; loadBraw(rBq[p],rr,blkn);} } }
    // WMMA on buf (B holds raw nibbles 0..15 => weight operand UNSIGNED). 8 indep cR tiles space the chain.
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
          // A signed (activation int8), B UNSIGNED (nibble 0..15).
          cR[mt][nt]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,Acur[mt],false,Bcur[nt],cR[mt][nt],false);
    }
    // scales OUTSIDE the dot + asymmetric correction. wsc=d*sc, wmn=dmin*m (per n-row, blk).
    float wsc[WNT],wmn[WNT],asc[WMT][8]; int ars[WMT][8];
    for(int nt=0;nt<WNT;nt++){ size_t nr=(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk; wsc[nt]=w_sc[nr]; wmn[nt]=w_mn[nr]; }
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++){ size_t mr=(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk; asc[mt][j]=a_scale[mr]; ars[mt][j]=a_rowsum[mr]; }
    for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++)
        // d*sc * a_s * idot   -   dmin*m * a_s * arowsum
        Rf[mt][nt][j]+=asc[mt][j]*( wsc[nt]*(float)cR[mt][nt][j] - wmn[nt]*(float)ars[mt][j] );
    // store next-block B nibbles (from prefetched raw qs) + commit prefetched A.
    if(kn<K){ wait_loadN<0>();
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPTQ;p++){int idx=tid+p*NTHREAD; if(idx<BVQ){ int rr=idx>>1; storeB(buf^1,rr,blkn,rBq[p]);} } }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
}

// ===================================================================================================
//  Q4_K via ROW-MAJOR 4-BIT REPACK (W4). The native block_q4_K on-disk layout is row-major-by-256-
//  superblock, so a 32-K slice across BN rows is maximally scattered -> the global B-load is uncoalesced
//  and ~4x slower than int8 even with ZERO unpack ALU (measured). FIX: repack ONCE at load to a row-major
//  packed-nibble buffer W4[N][K/2] (K-ordered nibbles, 2/byte) + the same per-32 scale tables. Now the
//  B-load is EXACTLY the int8 access pattern but HALF the bytes (4-bit), so the bandwidth lever is real
//  AND the access is coalesced. Unpack (2 nibbles/byte) happens at ds_store (RADV block_a_to_shmem). The
//  repack is a 4-bit repack (not int8): we still HALVE the weight DRAM traffic vs our current int8 path.
//  Nibbles staged UNSIGNED (0..15); asymmetric dmin*m correction applied outside the dot via a_rowsum.
// ===================================================================================================
//  W4 layout: W4[(size_t)n*(K/2) + k/2], low nibble = even k, high nibble = odd k (K-order).
//  Per 32-block: 16 bytes -> 32 nibbles. Load 16 bytes (i4) per (row,block), unpack to 32 int8 in store.
template<int BM,int BN,int WMW,int WNW,int SWZ>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_q4k_w4(
    const uint8_t* __restrict__ W4, const float* __restrict__ w_sc, const float* __restrict__ w_mn,
    const int8_t* __restrict__ act, const float* __restrict__ a_scale, const int* __restrict__ a_rowsum,
    __half* __restrict__ out, int M, int N, int K){
  constexpr int KC=32;
  constexpr int NWARP=WMW*WNW, NTHREAD=NWARP*32, WMT=BM/(WMW*16), WNT=BN/(WNW*16);
  constexpr int AV=BM*KC/16, APT=(AV+NTHREAD-1)/NTHREAD;
  constexpr int KSUB=KC/16;
  constexpr int BV=BN, BPT=(BV+NTHREAD-1)/NTHREAD;   // one i4(16B)=32 nibbles per (row,32-block)
  int Kh=K/2;
  __shared__ int8_t As[2][BM][KC]; __shared__ int8_t Bs[2][BN][KC];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16, wm=warp/WNW, wn=warp%WNW;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/KC;
  float Rf[WMT][WNT][8];
  for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)for(int j=0;j<8;j++)Rf[a][b][j]=0.f;
  // unpack 16 packed-nibble bytes (i4) -> 32 int8 nibbles, ds_store swizzled. Build the 8 output int32
  // words by interleaving low/high nibbles purely in registers (NO stack byte array). For each input
  // int32 word x (4 bytes b0..b3 = 8 nibbles), the 8 output int8 are [lo(b0),hi(b0),lo(b1),hi(b1),...].
  // These span TWO output int32 words: w_even = lo(b0)|hi(b0)<<8|lo(b1)<<16|hi(b1)<<24, etc.
  auto storeB=[&](int b,int rr,const i4 raw){
    uint32_t W0[4],W1[4];
    #pragma unroll
    for(int wi=0;wi<4;wi++){
      uint32_t x=(uint32_t)raw[wi];
      uint32_t lo=x&0x0F0F0F0Fu, hi=(x>>4)&0x0F0F0F0Fu;
      uint32_t e=(lo&0xFF)|((hi&0xFF)<<8)|(((lo>>8)&0xFF)<<16)|(((hi>>8)&0xFF)<<24);
      uint32_t f=((lo>>16)&0xFF)|(((hi>>16)&0xFF)<<8)|(((lo>>24)&0xFF)<<16)|(((hi>>24)&0xFF)<<24);
      if(wi<2){ W0[2*wi]=e; W0[2*wi+1]=f; } else { W1[2*(wi-2)]=e; W1[2*(wi-2)+1]=f; }
    }
    int c0=(SWZ?swz16(rr,0,KC):0)*16, c1=(SWZ?swz16(rr,1,KC):1)*16;
    *(i4*)&Bs[b][rr][c0]=*(const i4*)W0;
    *(i4*)&Bs[b][rr][c1]=*(const i4*)W1;
  };
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+blk16*16];}}
  for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD; if(idx<BV){ i4 raw=*(const i4*)&W4[(size_t)(n_base+idx)*Kh+0]; storeB(0,idx,raw);} }
  __syncthreads(); int buf=0;
  auto fetchA=[&](v2i Af[WMT],int b,int ks){
    #pragma unroll
    for(int mt=0;mt<WMT;mt++){int row=wm*WMT*16+mt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* ap=&As[b][row][bb*16+kg*8];Af[mt]=(v2i){*(const int*)ap,*(const int*)(ap+4)};} };
  auto fetchB=[&](v2i Bf[WNT],int b,int ks){
    #pragma unroll
    for(int nt=0;nt<WNT;nt++){int row=wn*WNT*16+nt*16+nl;int bb=SWZ?swz16(row,ks,KC):ks;const int8_t* bp=&Bs[b][row][bb*16+kg*8];Bf[nt]=(v2i){*(const int*)bp,*(const int*)(bp+4)};} };
  for(int k0=0;k0<K;k0+=KC){ int blk=k0/KC; i4 rA[APT]; i4 rB[BPT]; int kn=k0+KC; int blkn=kn/KC;
    if(kn<K){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kn+c];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD; if(idx<BV){ rB[p]=*(const i4*)&W4[(size_t)(n_base+idx)*Kh+blkn*16];} } }
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
          cR[mt][nt]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,Acur[mt],false,Bcur[nt],cR[mt][nt],false);
    }
    // scales outside the dot. Keep register pressure == v3p: only asc[WMT][8] persists (same as v3p).
    // The asymmetric correction reads arowsum into a SMALL per-mt transient (8 floats) reused over nt.
    float wsc[WNT],wmn[WNT];
    for(int nt=0;nt<WNT;nt++){ size_t nr=(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk; wsc[nt]=w_sc[nr]; wmn[nt]=w_mn[nr]; }
    float asc[WMT][8];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++){ size_t mr=(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk; asc[mt][j]=a_scale[mr]; }
    for(int mt=0;mt<WMT;mt++){
      float acorr[8];
      #pragma unroll
      for(int j=0;j<8;j++){ size_t mr=(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk; acorr[j]=asc[mt][j]*(float)a_rowsum[mr]; }
      for(int nt=0;nt<WNT;nt++)
        #pragma unroll
        for(int j=0;j<8;j++)
          Rf[mt][nt][j]+=asc[mt][j]*wsc[nt]*(float)cR[mt][nt][j] - wmn[nt]*acorr[j];
    }
    if(kn<K){ wait_loadN<0>();
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),blk16=idx%(KC/16),c=(SWZ?swz16(row,blk16,KC):blk16)*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD; if(idx<BV){ storeB(buf^1,idx,rB[p]);} } }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
}

// repack native block_q4_K -> W4[N][K/2] row-major packed nibbles (K-order). One thread per 256-block.
__global__ void q4k_to_w4(const block_q4_K* __restrict__ W, uint8_t* __restrict__ W4, int N, int K){
  int idx=blockIdx.x*blockDim.x+threadIdx.x; int nb256=K/256; if(idx>=N*nb256)return;
  int n=idx/nb256, b=idx%nb256; const block_q4_K* B=&W[(size_t)n*nb256+b];
  // produce 256 nibbles in K order, pack 2/byte into W4[n*(K/2) + b*128 ..]
  uint8_t* o=&W4[(size_t)n*(K/2)+b*128];
  // ggml K-order: element k (0..255), is=k/32 sub-block. The decode q4k_dequant_cpu emits, for n2 in {0,64,128,192}
  //   yy[n2+l]=lo(q[l]) [sub is], yy[n2+l+32]=hi(q[l]) [sub is+1]; q advances by 32 each 64-block.
  const uint8_t* q=B->qs;
  for(int blk64=0;blk64<4;blk64++){ const uint8_t* qq=q+blk64*32;
    for(int l=0;l<32;l++){ int klo=blk64*64+l, khi=blk64*64+l+32;
      uint8_t lo=qq[l]&0xF, hi=qq[l]>>4;
      // set nibble klo=lo, khi=hi in W4 (2/byte, low=even k)
      auto setk=[&](int k,uint8_t v){ uint8_t* bp=&o[k>>1]; if(k&1)*bp=(*bp&0x0F)|(v<<4); else *bp=(*bp&0xF0)|(v&0xF); };
      setk(klo,lo); setk(khi,hi);
    }
  }
}

// ===================================================================================================
//  NATIVE Q6_K GEMM. W = d*sc*(q6-32). Stage (q6-32) as a true signed int8 -> symmetric, NO correction.
//  KC=16 (Q6_K per-16 sub-block). B stage decodes the 16 q6 values for (n-row, 16-block) and stores int8.
// ===================================================================================================
//  Unpack 16 q6 (centered: q6-32) for (Q6_K 256-block, 16-sub-block s in [0,16)) in K order.
//  ggml Q6_K: for the 256-block, two halves of 128. Within a half (n in {0,128}), for l in 0..31:
//    q1=(ql[l]&0xF)|((qh[l]>>0&3)<<4)  -> element l
//    q2=(ql[l+32]&0xF)|((qh[l]>>2&3)<<4)-> element l+32
//    q3=(ql[l]>>4)|((qh[l]>>4&3)<<4)   -> element l+64
//    q4=(ql[l+32]>>4)|((qh[l]>>6&3)<<4)-> element l+96
//  scales: sc[is], is = (element/16). So 16-sub-block s maps to a contiguous run of 16 K elements all
//  sharing sc[s]. Element e = s*16 + r (r 0..15). Need to invert the (q1..q4,l) layout to gather them.
__device__ __forceinline__ void q6k_unpack_sub16(const block_q6_K* B, int s, int8_t out[16]){
  // s in [0,16): elements [s*16 .. s*16+16). half=s/8 (0 or 1) -> ql base 0/64, qh base 0/32.
  // within the half, local element le = (s&7)*16 + r, le in [0,128). Map le -> which of q1..q4 and l.
  int half=s>>3; const uint8_t* ql=B->ql + half*64; const uint8_t* qh=B->qh + half*32;
  int le0=(s&7)*16;   // start local element in this half
  #pragma unroll
  for(int r=0;r<16;r++){
    int le=le0+r;            // 0..127
    int sel=le>>5;           // 0..3 -> q1,q2,q3,q4
    int l=le&31;             // 0..31
    int v;
    if(sel==0)      v=((ql[l]&0xF)      | (((qh[l]>>0)&3)<<4));
    else if(sel==1) v=((ql[l+32]&0xF)   | (((qh[l]>>2)&3)<<4));
    else if(sel==2) v=((ql[l]>>4)       | (((qh[l]>>4)&3)<<4));
    else            v=((ql[l+32]>>4)    | (((qh[l]>>6)&3)<<4));
    out[r]=(int8_t)(v-32);
  }
}

template<int BM,int BN,int WMW,int WNW,int SWZ>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_q6k_native(
    const int8_t* __restrict__ act, const float* __restrict__ a_scale,
    const block_q6_K* __restrict__ W, const float* __restrict__ w_sc,
    __half* __restrict__ out, int M, int N, int K){
  constexpr int KC=16;
  constexpr int NWARP=WMW*WNW, NTHREAD=NWARP*32, WMT=BM/(WMW*16), WNT=BN/(WNW*16);
  constexpr int AV=BM*KC/16, APT=(AV+NTHREAD-1)/NTHREAD;
  constexpr int KSUB=1;
  __shared__ int8_t As[2][BM][KC]; __shared__ int8_t Bs[2][BN][KC];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16, wm=warp/WNW, wn=warp%WNW;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/KC, nb256=K/256;
  float Rf[WMT][WNT][8];
  for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)for(int j=0;j<8;j++)Rf[a][b][j]=0.f;
  auto stageB=[&](int b,int blk){
    int blk256=blk/16, ssub=blk&15;
    for(int rr=tid; rr<BN; rr+=NTHREAD){
      const block_q6_K* Bblk=&W[(size_t)(n_base+rr)*nb256+blk256];
      int8_t q[16]; q6k_unpack_sub16(Bblk,ssub,q);
      *(i4*)&Bs[b][rr][0]=*(const i4*)&q[0];
    }
  };
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx,c=0;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+0];}}
  stageB(0,0);
  __syncthreads(); int buf=0;
  auto fetchA=[&](v2i Af[WMT],int b){
    #pragma unroll
    for(int mt=0;mt<WMT;mt++){int row=wm*WMT*16+mt*16+nl;const int8_t* ap=&As[b][row][kg*8];Af[mt]=(v2i){*(const int*)ap,*(const int*)(ap+4)};} };
  auto fetchB=[&](v2i Bf[WNT],int b){
    #pragma unroll
    for(int nt=0;nt<WNT;nt++){int row=wn*WNT*16+nt*16+nl;const int8_t* bp=&Bs[b][row][kg*8];Bf[nt]=(v2i){*(const int*)bp,*(const int*)(bp+4)};} };
  for(int k0=0;k0<K;k0+=KC){ int blk=k0/KC; i4 rA[APT]; int kn=k0+KC; int blkn=kn/KC;
    if(kn<K){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kn];}} }
    v8i cR[WMT][WNT]; for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)cR[a][b]=(v8i){};
    v2i Af0[WMT],Bf0[WNT];
    fetchA(Af0,buf); fetchB(Bf0,buf);
    #pragma unroll
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int nt=0;nt<WNT;nt++)
        cR[mt][nt]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,Af0[mt],true,Bf0[nt],cR[mt][nt],false);
    float wsc[WNT],asc[WMT][8];
    for(int nt=0;nt<WNT;nt++)wsc[nt]=w_sc[(size_t)(n_base+wn*WNT*16+nt*16+nl)*nb+blk];
    for(int mt=0;mt<WMT;mt++)
      #pragma unroll
      for(int j=0;j<8;j++)asc[mt][j]=a_scale[(size_t)(m_base+wm*WMT*16+mt*16+kg*8+j)*nb+blk];
    for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++)Rf[mt][nt][j]+=(float)cR[mt][nt][j]*wsc[nt]*asc[mt][j];
    if(kn<K){ stageB(buf^1,blkn);
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx;*(i4*)&As[buf^1][row][0]=rA[p];}} }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl]=__float2half(Rf[mt][nt][j]);}
}

// ===================================================================================================
//  PREP kernels: build the per-(n,blk) weight scale tables and per-(m,blk) activation rowsum.
// ===================================================================================================
// Q4_K: w_sc[n,blk]=d*sc[blk%8], w_mn[n,blk]=dmin*m[blk%8]. nb=K/32, one 256-block has 8 sub-blocks.
__global__ void q4k_make_scales(const block_q4_K* __restrict__ W, float* __restrict__ w_sc, float* __restrict__ w_mn, int N, int K){
  int idx=blockIdx.x*blockDim.x+threadIdx.x; int nb256=K/256; if(idx>=N*nb256)return;
  int n=idx/nb256, b=idx%nb256; const block_q4_K* B=&W[(size_t)n*nb256+b];
  float d=__half2float(B->d), dmin=__half2float(B->dmin);
  float* ws=&w_sc[(size_t)n*(K/32)+b*8]; float* wm=&w_mn[(size_t)n*(K/32)+b*8];
  for(int j=0;j<8;j++){ uint8_t sc,mn; gsm_d(j,B->scales,&sc,&mn); ws[j]=d*(float)sc; wm[j]=dmin*(float)mn; }
}
// Q6_K: w_sc[n,blk]=d*sc[blk%16], nb=K/16, one 256-block has 16 sub-blocks.
__global__ void q6k_make_scales(const block_q6_K* __restrict__ W, float* __restrict__ w_sc, int N, int K){
  int idx=blockIdx.x*blockDim.x+threadIdx.x; int nb256=K/256; if(idx>=N*nb256)return;
  int n=idx/nb256, b=idx%nb256; const block_q6_K* B=&W[(size_t)n*nb256+b];
  float d=__half2float(B->d); float* ws=&w_sc[(size_t)n*(K/16)+b*16];
  for(int j=0;j<16;j++)ws[j]=d*(float)B->scales[j];
}
// activation int8 row-sum per 32-block (for Q4_K asymmetric correction). nb=K/32.
__global__ void act_rowsum32(const int8_t* __restrict__ aq, int* __restrict__ ars, int M, int K){
  int idx=blockIdx.x*blockDim.x+threadIdx.x; int nb=K/32; if(idx>=M*nb)return;
  int m=idx/nb, b=idx%nb; const int8_t* a=&aq[(size_t)m*K+b*32]; int s=0;
  #pragma unroll
  for(int j=0;j<32;j++)s+=(int)a[j];
  ars[(size_t)m*nb+b]=s;
}

// ===================================================================================================
//  CPU references (ggml dequant) for correctness.
// ===================================================================================================
static void q4k_dequant_cpu(const block_q4_K* x, float* y, int nblocks){
  for(int i=0;i<nblocks;i++){ float d=__half2float(x[i].d), dmin=__half2float(x[i].dmin);
    const uint8_t* q=x[i].qs; float* yy=y+(size_t)i*256;
    int is=0; uint8_t sc,mn;
    for(int n=0;n<256;n+=64){
      gsm_d(is,x[i].scales,&sc,&mn); float d1=d*sc, m1=dmin*mn; is++;
      gsm_d(is,x[i].scales,&sc,&mn); float d2=d*sc, m2=dmin*mn; is++;
      for(int l=0;l<32;l++){ yy[n+l]   = d1*(q[l]&0xF) - m1; yy[n+l+32]= d2*(q[l]>>4) - m2; }
      q+=32;
    }
  }
}
static void q6k_dequant_cpu(const block_q6_K* x, float* y, int nblocks){
  for(int i=0;i<nblocks;i++){ float d=__half2float(x[i].d);
    const uint8_t* ql=x[i].ql; const uint8_t* qh=x[i].qh; const int8_t* sc=x[i].scales; float* yy=y+(size_t)i*256;
    for(int n=0;n<256;n+=128){ for(int l=0;l<32;l++){ int is=l/16;
        int q1=((ql[l]&0xF)|(((qh[l]>>0)&3)<<4))-32, q2=((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4))-32;
        int q3=((ql[l]>>4)|(((qh[l]>>4)&3)<<4))-32, q4=((ql[l+32]>>4)|(((qh[l]>>6)&3)<<4))-32;
        yy[l]=d*sc[is]*q1; yy[l+32]=d*sc[is+2]*q2; yy[l+64]=d*sc[is+4]*q3; yy[l+96]=d*sc[is+6]*q4; }
      ql+=64; qh+=32; sc+=8; yy+=128; }
  }
}

// fill a block_q4_K with random valid contents
static void rand_q4k(block_q4_K& b){
  b.d=__float2half(0.01f+(rand()%50)/5000.f); b.dmin=__float2half((rand()%40)/8000.f);
  for(int i=0;i<12;i++)b.scales[i]=rand()&0xFF; for(int i=0;i<128;i++)b.qs[i]=rand()&0xFF;
}
static void rand_q6k(block_q6_K& b){
  b.d=__float2half(0.008f+(rand()%50)/10000.f);
  for(int i=0;i<128;i++)b.ql[i]=rand()&0xFF; for(int i=0;i<64;i++)b.qh[i]=rand()&0xFF;
  for(int i=0;i<16;i++)b.scales[i]=rand()%80-40;
}

// ===================================================================================================
//  CORRECTNESS
// ===================================================================================================
static void correctness_q4k(){
  int M=64,N=256,K=512,nb256=K/256,nb=K/32;
  std::vector<block_q4_K> hW(N*nb256); srand(7); for(auto&b:hW)rand_q4k(b);
  std::vector<float> Wref(N*K); q4k_dequant_cpu(hW.data(),Wref.data(),N*nb256);
  std::vector<float> A(M*K); for(auto&x:A)x=(rand()%200-100)/64.f;
  std::vector<int8_t> Aq(M*K); std::vector<float> As(M*nb);
  for(int m=0;m<M;m++)for(int kb=0;kb<nb;kb++){ float mx=0; for(int k=0;k<32;k++)mx=fmaxf(mx,fabsf(A[m*K+kb*32+k]));
    float s=mx/127.f; if(s<1e-9f)s=1e-9f; As[m*nb+kb]=s; for(int k=0;k<32;k++)Aq[m*K+kb*32+k]=(int8_t)lrintf(A[m*K+kb*32+k]/s); }
  std::vector<float> Oref(M*N,0); for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=(double)Wref[n*K+k]*A[m*K+k];Oref[m*N+n]=(float)s;}
  block_q4_K* dW; CK(hipMalloc(&dW,hW.size()*sizeof(block_q4_K))); CK(hipMemcpy(dW,hW.data(),hW.size()*sizeof(block_q4_K),hipMemcpyHostToDevice));
  float *dwsc,*dwmn; CK(hipMalloc(&dwsc,(size_t)N*nb*4)); CK(hipMalloc(&dwmn,(size_t)N*nb*4));
  q4k_make_scales<<<(N*nb256+127)/128,128>>>(dW,dwsc,dwmn,N,K); CK(hipDeviceSynchronize());
  int8_t* dAq; float* dAs; int* dArs; __half* dO;
  CK(hipMalloc(&dAq,M*K)); CK(hipMalloc(&dAs,M*nb*4)); CK(hipMalloc(&dArs,M*nb*4)); CK(hipMalloc(&dO,M*N*2));
  CK(hipMemcpy(dAq,Aq.data(),M*K,hipMemcpyHostToDevice)); CK(hipMemcpy(dAs,As.data(),M*nb*4,hipMemcpyHostToDevice));
  act_rowsum32<<<(M*nb+127)/128,128>>>(dAq,dArs,M,K); CK(hipDeviceSynchronize());
  dim3 g(N/64,M/64),b(2*2*32);
  gemm_q4k_native<64,64,2,2,1><<<g,b>>>(dAq,dAs,dArs,dW,dwsc,dwmn,dO,M,N,K); CK(hipDeviceSynchronize());
  std::vector<__half> hO(M*N); CK(hipMemcpy(hO.data(),dO,M*N*2,hipMemcpyDeviceToHost));
  double e=0,nn=0; for(int i=0;i<M*N;i++){float v=__half2float(hO[i]);e+=(v-Oref[i])*(v-Oref[i]);nn+=Oref[i]*Oref[i];}
  double nrmse=sqrt(e/nn);
  printf("  Q4_K native GEMM (KC=32) vs float ref: NRMSE=%.4f  %s\n",nrmse,nrmse<2e-2?"PASS":"FAIL");
  // --- W4 (row-major 4-bit repack) path on the SAME data ---
  uint8_t* dW4; CK(hipMalloc(&dW4,(size_t)N*(K/2)));
  q4k_to_w4<<<(N*nb256+127)/128,128>>>(dW,dW4,N,K); CK(hipDeviceSynchronize());
  gemm_q4k_w4<64,64,2,2,1><<<g,b>>>(dW4,dwsc,dwmn,dAq,dAs,dArs,dO,M,N,K); CK(hipDeviceSynchronize());
  CK(hipMemcpy(hO.data(),dO,M*N*2,hipMemcpyDeviceToHost));
  double e2=0; for(int i=0;i<M*N;i++){float v=__half2float(hO[i]);e2+=(v-Oref[i])*(v-Oref[i]);}
  printf("  Q4_K W4-repack GEMM (KC=32) vs float ref: NRMSE=%.4f  %s\n",sqrt(e2/nn),sqrt(e2/nn)<2e-2?"PASS":"FAIL");
  hipFree(dW4);
  hipFree(dW);hipFree(dwsc);hipFree(dwmn);hipFree(dAq);hipFree(dAs);hipFree(dArs);hipFree(dO);
}
static void correctness_q6k(){
  int M=64,N=256,K=512,nb256=K/256,nb=K/16;
  std::vector<block_q6_K> hW(N*nb256); srand(9); for(auto&b:hW)rand_q6k(b);
  std::vector<float> Wref(N*K); q6k_dequant_cpu(hW.data(),Wref.data(),N*nb256);
  std::vector<float> A(M*K); for(auto&x:A)x=(rand()%200-100)/64.f;
  std::vector<int8_t> Aq(M*K); std::vector<float> As(M*nb);
  for(int m=0;m<M;m++)for(int kb=0;kb<nb;kb++){ float mx=0; for(int k=0;k<16;k++)mx=fmaxf(mx,fabsf(A[m*K+kb*16+k]));
    float s=mx/127.f; if(s<1e-9f)s=1e-9f; As[m*nb+kb]=s; for(int k=0;k<16;k++)Aq[m*K+kb*16+k]=(int8_t)lrintf(A[m*K+kb*16+k]/s); }
  std::vector<float> Oref(M*N,0); for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=(double)Wref[n*K+k]*A[m*K+k];Oref[m*N+n]=(float)s;}
  block_q6_K* dW; CK(hipMalloc(&dW,hW.size()*sizeof(block_q6_K))); CK(hipMemcpy(dW,hW.data(),hW.size()*sizeof(block_q6_K),hipMemcpyHostToDevice));
  float* dwsc; CK(hipMalloc(&dwsc,(size_t)N*nb*4)); q6k_make_scales<<<(N*nb256+127)/128,128>>>(dW,dwsc,N,K); CK(hipDeviceSynchronize());
  int8_t* dAq; float* dAs; __half* dO; CK(hipMalloc(&dAq,M*K)); CK(hipMalloc(&dAs,M*nb*4)); CK(hipMalloc(&dO,M*N*2));
  CK(hipMemcpy(dAq,Aq.data(),M*K,hipMemcpyHostToDevice)); CK(hipMemcpy(dAs,As.data(),M*nb*4,hipMemcpyHostToDevice));
  dim3 g(N/64,M/64),b(2*2*32);
  gemm_q6k_native<64,64,2,2,0><<<g,b>>>(dAq,dAs,dW,dwsc,dO,M,N,K); CK(hipDeviceSynchronize());
  std::vector<__half> hO(M*N); CK(hipMemcpy(hO.data(),dO,M*N*2,hipMemcpyDeviceToHost));
  double e=0,nn=0; for(int i=0;i<M*N;i++){float v=__half2float(hO[i]);e+=(v-Oref[i])*(v-Oref[i]);nn+=Oref[i]*Oref[i];}
  double nrmse=sqrt(e/nn);
  printf("  Q6_K native GEMM (KC=16) vs float ref: NRMSE=%.4f  %s\n",nrmse,nrmse<2e-2?"PASS":"FAIL");
  hipFree(dW);hipFree(dwsc);hipFree(dAq);hipFree(dAs);hipFree(dO);
}

// ===================================================================================================
//  COLD STANDALONE GEMM bench. NBUF rotated weight copies + L2 thrash between reps. Returns ms.
//  measures the native-Q4K / native-Q6K / int8-ref path on a single gate-like shape.
// ===================================================================================================
__global__ void thrash(int8_t* p, size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]^=1; }
int g_tile=0; // 0=256x256, 1=128x256, 2=128x128

enum Mode { MODE_INT8, MODE_Q4K, MODE_W4, MODE_Q6K };
// returns cold ms (thrash-subtract avg as in gemm_rb_v3 bench_shape_cold). NBUF rotated weights so each
// rep reads a DIFFERENT VRAM region (realistic cold weight). The weight bytes per mode: int8=N*K,
// Q4K-native=N*nb256*144 (=4.5bit), W4=N*K/2 (4bit), Q6K-native=N*nb256*210 (=6.5bit).
static double bench_cold(Mode mode,int M,int N,int K,int reps){
  const int NBUF=3; int nb32=K/32, nb16=K/16, nb256=K/256;
  int8_t* dWi8[NBUF]; float* dWs8[NBUF];
  block_q4_K* dW4[NBUF]; float* dwsc4[NBUF]; float* dwmn4[NBUF];
  uint8_t* dWp4[NBUF];   // W4 row-major packed nibbles
  block_q6_K* dW6[NBUF]; float* dwsc6[NBUF];
  int8_t* dAq; float* dAs; int* dArs; __half* dO;
  CK(hipMalloc(&dAq,(size_t)M*K)); CK(hipMalloc(&dAs,(size_t)M*nb32*4)); CK(hipMalloc(&dArs,(size_t)M*nb32*4)); CK(hipMalloc(&dO,(size_t)M*N*2));
  CK(hipMemset(dAq,1,(size_t)M*K)); CK(hipMemset(dAs,0,(size_t)M*nb32*4)); CK(hipMemset(dArs,0,(size_t)M*nb32*4));
  for(int bb=0;bb<NBUF;bb++){
    if(mode==MODE_INT8){ CK(hipMalloc(&dWi8[bb],(size_t)N*K)); CK(hipMalloc(&dWs8[bb],(size_t)N*nb32*4)); CK(hipMemset(dWi8[bb],1+bb,(size_t)N*K)); CK(hipMemset(dWs8[bb],0,(size_t)N*nb32*4)); }
    if(mode==MODE_Q4K){ CK(hipMalloc(&dW4[bb],(size_t)N*nb256*sizeof(block_q4_K))); CK(hipMalloc(&dwsc4[bb],(size_t)N*nb32*4)); CK(hipMalloc(&dwmn4[bb],(size_t)N*nb32*4)); CK(hipMemset(dW4[bb],1+bb,(size_t)N*nb256*sizeof(block_q4_K))); q4k_make_scales<<<(N*nb256+127)/128,128>>>(dW4[bb],dwsc4[bb],dwmn4[bb],N,K); }
    if(mode==MODE_W4){ block_q4_K* tmp; CK(hipMalloc(&tmp,(size_t)N*nb256*sizeof(block_q4_K))); CK(hipMemset(tmp,1+bb,(size_t)N*nb256*sizeof(block_q4_K)));
      CK(hipMalloc(&dWp4[bb],(size_t)N*(K/2))); CK(hipMalloc(&dwsc4[bb],(size_t)N*nb32*4)); CK(hipMalloc(&dwmn4[bb],(size_t)N*nb32*4));
      q4k_make_scales<<<(N*nb256+127)/128,128>>>(tmp,dwsc4[bb],dwmn4[bb],N,K); q4k_to_w4<<<(N*nb256+127)/128,128>>>(tmp,dWp4[bb],N,K); CK(hipDeviceSynchronize()); CK(hipFree(tmp)); }
    if(mode==MODE_Q6K){ CK(hipMalloc(&dW6[bb],(size_t)N*nb256*sizeof(block_q6_K))); CK(hipMalloc(&dwsc6[bb],(size_t)N*nb16*4)); CK(hipMemset(dW6[bb],1+bb,(size_t)N*nb256*sizeof(block_q6_K))); q6k_make_scales<<<(N*nb256+127)/128,128>>>(dW6[bb],dwsc6[bb],N,K); }
  }
  CK(hipDeviceSynchronize());
  size_t TH=32u*1024*1024; int8_t* dT; CK(hipMalloc(&dT,TH));
  // g_tile: 0=256x256(8,4), 1=128x256(4,4), 2=128x128(4,4)
  auto launch=[&](int bi){
    if(mode==MODE_INT8){
      if(g_tile==2){ dim3 g(N/128,M/128),b(4*4*32); gemm_v3p<128,128,4,4,32,1><<<g,b>>>(dAq,dAs,dWi8[bi],dWs8[bi],dO,M,N,K); }
      else if(g_tile==1){ dim3 g(N/256,M/128),b(4*4*32); gemm_v3p<128,256,4,4,32,1><<<g,b>>>(dAq,dAs,dWi8[bi],dWs8[bi],dO,M,N,K); }
      else { dim3 g(N/256,M/256),b(8*4*32); gemm_v3p<256,256,8,4,32,1><<<g,b>>>(dAq,dAs,dWi8[bi],dWs8[bi],dO,M,N,K); } }
    if(mode==MODE_Q4K ){ dim3 g(N/256,M/256),b(8*4*32); gemm_q4k_native<256,256,8,4,1><<<g,b>>>(dAq,dAs,dArs,dW4[bi],dwsc4[bi],dwmn4[bi],dO,M,N,K); }
    if(mode==MODE_W4  ){
      if(g_tile==2){ dim3 g(N/128,M/128),b(4*4*32); gemm_q4k_w4<128,128,4,4,1><<<g,b>>>(dWp4[bi],dwsc4[bi],dwmn4[bi],dAq,dAs,dArs,dO,M,N,K); }
      else if(g_tile==1){ dim3 g(N/256,M/128),b(4*4*32); gemm_q4k_w4<128,256,4,4,1><<<g,b>>>(dWp4[bi],dwsc4[bi],dwmn4[bi],dAq,dAs,dArs,dO,M,N,K); }
      else { dim3 g(N/256,M/256),b(8*4*32); gemm_q4k_w4<256,256,8,4,1><<<g,b>>>(dWp4[bi],dwsc4[bi],dwmn4[bi],dAq,dAs,dArs,dO,M,N,K); } }
    if(mode==MODE_Q6K ){ dim3 g(N/256,M/256),b(8*4*32); gemm_q6k_native<256,256,8,4,0><<<g,b>>>(dAq,dAs,dW6[bi],dwsc6[bi],dO,M,N,K); }
  };
  for(int w=0;w<4;w++)launch(0);
  hipError_t le=hipGetLastError(), se=hipDeviceSynchronize();
  double ms=-1;
  if(!le&&!se){
    double t0=now();
    for(int i=0;i<reps;i++){ thrash<<<(TH+255)/256,256>>>(dT,TH); launch(i%NBUF); }
    CK(hipDeviceSynchronize()); double tot=now()-t0;
    double t1=now(); for(int i=0;i<reps;i++)thrash<<<(TH+255)/256,256>>>(dT,TH); CK(hipDeviceSynchronize()); double thr=now()-t1;
    ms=(tot-thr)/reps;
  } else { printf("    [launch err le=%d se=%d]\n",(int)le,(int)se); }
  hipFree(dT);hipFree(dAq);hipFree(dAs);hipFree(dArs);hipFree(dO);
  for(int bb=0;bb<NBUF;bb++){
    if(mode==MODE_INT8){hipFree(dWi8[bb]);hipFree(dWs8[bb]);}
    if(mode==MODE_Q4K){hipFree(dW4[bb]);hipFree(dwsc4[bb]);hipFree(dwmn4[bb]);}
    if(mode==MODE_W4){hipFree(dWp4[bb]);hipFree(dwsc4[bb]);hipFree(dwmn4[bb]);}
    if(mode==MODE_Q6K){hipFree(dW6[bb]);hipFree(dwsc6[bb]);}
  }
  return ms;
}

int main(){
  printf("=== gemm_ngd: NATIVE K-quant-read WMMA prefill GEMM (gfx1201/R9700, GPU0) ===\n");
  printf("Lever: read 4-bit (Q4_K) / 6-bit (Q6_K) native instead of 8-bit re-quanted -> halve/cut weight BW.\n\n");

  printf("[1] CORRECTNESS (vs float ggml dequant ref, NRMSE<2e-2):\n");
  correctness_q4k();
  correctness_q6k();

  printf("\n[2] COLD STANDALONE GEMM (gate M=512 N=14336 K=4096) — int8-ref vs Q4K-native vs W4-repack, 30 reps:\n");
  { int M=512,N=14336,K=4096,R=30;
    auto tf=[&](double ms){return 2.0*M*N*K/1e12/(ms/1e3);};
    g_tile=0; double m4=bench_cold(MODE_Q4K,M,N,K,R);
    printf("  Q4K-native scatter 256x256 %.3f ms  %5.1f TFLOPS\n",m4,tf(m4));
    const char* tn[3]={"256x256","128x256","128x128"};
    for(int t=0;t<3;t++){ g_tile=t;
      double mi=bench_cold(MODE_INT8,M,N,K,R), mw=bench_cold(MODE_W4,M,N,K,R);
      printf("  [%s] int8 %.3f ms %5.1f TF | W4 %.3f ms %5.1f TF  (W4 %+.1f%% vs int8 same-tile)  %s\n",
        tn[t],mi,tf(mi),mw,tf(mw),(mi/mw-1)*100, mw<mi?"*** W4 FASTER ***":"");
    }
    g_tile=0;
  }
  printf("\n[3] LOW-M sweep (gate N=14336 K=4096) where weight-read dominates — int8 vs W4 (128x128 tile):\n");
  printf("    (at low M the GEMM is most weight-bandwidth-bound -> best case for the 4-bit lever)\n");
  { auto tf=[&](double ms,int M,int N,int K){return 2.0*M*N*K/1e12/(ms/1e3);};
    int Ms[4]={128,256,512,2048}; int N=14336,K=4096,R=30; g_tile=2;
    for(int mi=0;mi<4;mi++){ int M=Ms[mi]; if(M<128){continue;}
      double a=bench_cold(MODE_INT8,M,N,K,R), w=bench_cold(MODE_W4,M,N,K,R);
      printf("    M=%-4d  int8 %.3f ms %5.1f TF | W4 %.3f ms %5.1f TF  (W4 %+.1f%%)  %s\n",
        M,a,tf(a,M,N,K),w,tf(w,M,N,K),(a/w-1)*100, w<a?"*** W4 FASTER ***":"");
    }
    g_tile=0;
  }

  printf("\n[4] COLD STANDALONE GEMM (down M=512 N=4096 K=14336, long-K) — Q6K native vs int8-ref:\n");
  { int M=512,N=4096,K=14336,R=30;
    double mi=bench_cold(MODE_INT8,M,N,K,R), m6=bench_cold(MODE_Q6K,M,N,K,R);
    auto tf=[&](double ms){return 2.0*M*N*K/1e12/(ms/1e3);};
    printf("  int8-ref (v3p 256x256) %.3f ms  %5.1f TFLOPS\n",mi,tf(mi));
    printf("  native-Q6K (256x256)   %.3f ms  %5.1f TFLOPS   (%+.1f%% vs int8-ref)\n",m6,tf(m6),(mi/m6-1)*100);
  }
  return 0;
}
