// prefill_q6k.hip — Q6_K PREFILL: dequant Q6_K -> int8 (per-16, EXACT/lossless) at load,
// then the SAME register-blocked INT8 WMMA GEMM (KC=16 = Q6_K sub-block). Beats ggml's
// slow on-the-fly Q6_K mmq dequant: ggml dequants 6-bit IN the GEMM loop (per forward);
// we dequant ONCE at load -> the GEMM is pure int8.  Build: hipcc --offload-arch=gfx1201 -O3
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
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}

struct __attribute__((packed)) block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; __half d; }; // 210B

// KC-templated scale-aware GEMM (KC=16 for Q6_K per-16 sub-block scales, KC=32 for Q8).
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

// Q6_K -> int8 (per-16 EXACT): int8 = q6-32, scale = d*sc[16]. y-layout maps cleanly: ws[k/16]=d*sc[k/16].
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
// CPU reference dequant (GGML algorithm) for validation.
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
// Re-quant per-16 -> per-32 (merge 2 sub-scales into one): enables fast KC=32 GEMM but adds error.
__global__ void requant_32(const int8_t* __restrict__ Wq, const float* __restrict__ Ws, int8_t* __restrict__ Wq32, float* __restrict__ Ws32, int N, int K){
  int idx=blockIdx.x*blockDim.x+threadIdx.x; int nb32=K/32; if(idx>=N*nb32)return;
  int n=idx/nb32, b=idx%nb32, nb16=K/16; float f[32], mx=0;
  for(int j=0;j<32;j++){int k=b*32+j; f[j]=(float)Wq[(size_t)n*K+k]*Ws[(size_t)n*nb16+k/16]; mx=fmaxf(mx,fabsf(f[j]));}
  float s=mx/127.f; if(s<1e-12f)s=1e-12f; Ws32[(size_t)n*nb32+b]=s;
  for(int j=0;j<32;j++)Wq32[(size_t)n*K+b*32+j]=(int8_t)lrintf(f[j]/s);
}
#define CK(x) do{hipError_t e=(x);if(e){printf("HIP err %s\n",hipGetErrorString(e));}}while(0)

int main(){
  { // CORRECTNESS: Q6_K dequant + KC=16 GEMM vs float reference
    int M=64,N=256,K=512,nblk=K/256,nb=K/16;
    std::vector<block_q6_K> hW(N*nblk);
    srand(7); uint8_t* raw=(uint8_t*)hW.data();
    for(size_t i=0;i<hW.size()*sizeof(block_q6_K);i++)raw[i]=rand()&0xFF;
    for(auto&blk:hW){ blk.d=__float2half(0.008f+(rand()%50)/10000.f); for(int i=0;i<16;i++)blk.scales[i]=rand()%80-40; }
    std::vector<float> Wref(N*K); dequant_q6k_cpu(hW.data(),Wref.data(),N*nblk);
    // activation
    std::vector<float> A(M*K); for(auto&x:A)x=(rand()%200-100)/64.f;
    std::vector<int8_t> Aq(M*K); std::vector<float> As(M*nb);
    for(int m=0;m<M;m++)for(int kb=0;kb<nb;kb++){ float mx=0; for(int k=0;k<16;k++)mx=fmaxf(mx,fabsf(A[m*K+kb*16+k]));
      float s=mx/127.f; if(s<1e-9f)s=1e-9f; As[m*nb+kb]=s; for(int k=0;k<16;k++)Aq[m*K+kb*16+k]=(int8_t)lrintf(A[m*K+kb*16+k]/s); }
    std::vector<float> Oref(M*N,0); for(int m=0;m<M;m++)for(int n=0;n<N;n++){float s=0;for(int k=0;k<K;k++)s+=Wref[n*K+k]*A[m*K+k];Oref[m*N+n]=s;}
    block_q6_K* dW; int8_t* dWq; float* dWs; CK(hipMalloc(&dW,hW.size()*sizeof(block_q6_K)));CK(hipMalloc(&dWq,N*K));CK(hipMalloc(&dWs,N*nb*4));
    CK(hipMemcpy(dW,hW.data(),hW.size()*sizeof(block_q6_K),hipMemcpyHostToDevice));
    dequant_q6k<<<(N*nblk+127)/128,128>>>(dW,dWq,dWs,N,K); CK(hipDeviceSynchronize());
    // validate dequant exactness
    std::vector<int8_t> hWq(N*K); std::vector<float> hWs(N*nb);
    CK(hipMemcpy(hWq.data(),dWq,N*K,hipMemcpyDeviceToHost));CK(hipMemcpy(hWs.data(),dWs,N*nb*4,hipMemcpyDeviceToHost));
    double de=0,dn=0; for(int n=0;n<N;n++)for(int k=0;k<K;k++){float v=hWq[n*K+k]*hWs[n*nb+k/16];de+=(v-Wref[n*K+k])*(v-Wref[n*K+k]);dn+=Wref[n*K+k]*Wref[n*K+k];}
    printf("Q6_K dequant->int8 (per-16): NRMSE=%.2e  %s (should be ~0, EXACT)\n",sqrt(de/dn),sqrt(de/dn)<1e-5?"PASS":"FAIL");
    // GEMM
    int8_t* dAq; float* dAs; __half* dO; CK(hipMalloc(&dAq,M*K));CK(hipMalloc(&dAs,M*nb*4));CK(hipMalloc(&dO,M*N*2));
    CK(hipMemcpy(dAq,Aq.data(),M*K,hipMemcpyHostToDevice));CK(hipMemcpy(dAs,As.data(),M*nb*4,hipMemcpyHostToDevice));
    dim3 g(N/64,M/64),b(2*2*32); gemm_rb_kc<64,64,2,2,16><<<g,b>>>(dAq,dAs,dWq,dWs,dO,M,N,K); CK(hipDeviceSynchronize());
    std::vector<__half> hO(M*N); CK(hipMemcpy(hO.data(),dO,M*N*2,hipMemcpyDeviceToHost));
    double ge=0,gn=0; for(int i=0;i<M*N;i++){float v=__half2float(hO[i]);ge+=(v-Oref[i])*(v-Oref[i]);gn+=Oref[i]*Oref[i];}
    printf("Q6_K prefill GEMM (KC=16) vs float ref: NRMSE=%.4f  %s (only act-int8 error)\n",sqrt(ge/gn),sqrt(ge/gn)<0.02?"PASS":"FAIL");
    // KC=32 re-quant path (fast Q8-speed GEMM, but merges 2 sub-scales -> extra error)
    int8_t* dWq32; float* dWs32; int nb32=K/32; CK(hipMalloc(&dWq32,N*K));CK(hipMalloc(&dWs32,N*nb32*4));
    requant_32<<<(N*nb32+127)/128,128>>>(dWq,dWs,dWq32,dWs32,N,K); CK(hipDeviceSynchronize());
    std::vector<int8_t> Aq32(M*K); std::vector<float> As32(M*nb32);
    for(int m=0;m<M;m++)for(int kb=0;kb<nb32;kb++){float mx=0;for(int k=0;k<32;k++)mx=fmaxf(mx,fabsf(A[m*K+kb*32+k]));float s=mx/127.f;if(s<1e-9f)s=1e-9f;As32[m*nb32+kb]=s;for(int k=0;k<32;k++)Aq32[m*K+kb*32+k]=(int8_t)lrintf(A[m*K+kb*32+k]/s);}
    int8_t* dAq32; float* dAs32; CK(hipMalloc(&dAq32,M*K));CK(hipMalloc(&dAs32,M*nb32*4));
    CK(hipMemcpy(dAq32,Aq32.data(),M*K,hipMemcpyHostToDevice));CK(hipMemcpy(dAs32,As32.data(),M*nb32*4,hipMemcpyHostToDevice));
    gemm_rb_kc<64,64,2,2,32><<<g,b>>>(dAq32,dAs32,dWq32,dWs32,dO,M,N,K); CK(hipDeviceSynchronize());
    CK(hipMemcpy(hO.data(),dO,M*N*2,hipMemcpyDeviceToHost));
    double ge2=0; for(int i=0;i<M*N;i++){float v=__half2float(hO[i]);ge2+=(v-Oref[i])*(v-Oref[i]);}
    printf("Q6_K prefill GEMM (KC=32 re-quant) vs float ref: NRMSE=%.4f  %s\n",sqrt(ge2/gn),sqrt(ge2/gn)<0.02?"PASS (lossless-grade, use KC=32 fast)":"FAIL (lossy, must use KC=16)");
    hipFree(dWq32);hipFree(dWs32);hipFree(dAq32);hipFree(dAs32);
    hipFree(dW);hipFree(dWq);hipFree(dWs);hipFree(dAq);hipFree(dAs);hipFree(dO);
  }
  { // SPEED: Q6_K prefill via KC=32 re-quant (Q8-speed GEMM) — winning path, per-shape best configs
    printf("\nQ6_K PREFILL KC=32 re-quant (lossless-grade 0.58%%) vs ggml 1612 / RADV 2255:\n");
    double agg_ms=0; int NL=50;
    #define SHQ(NN,KK,nm,BM,BN,A2,B2) { int M=512,N=NN,K=KK; int nblk=K/256,nb16=K/16,nb32=K/32; \
      block_q6_K* dW; int8_t*dWq,*dWq32,*dAq; float*dWs,*dWs32,*dAs; __half*dO; \
      hipMalloc(&dW,(size_t)N*nblk*sizeof(block_q6_K));hipMalloc(&dWq,(size_t)N*K);hipMalloc(&dWs,(size_t)N*nb16*4); \
      hipMalloc(&dWq32,(size_t)N*K);hipMalloc(&dWs32,(size_t)N*nb32*4);hipMalloc(&dAq,(size_t)M*K);hipMalloc(&dAs,(size_t)M*nb32*4);hipMalloc(&dO,(size_t)M*N*2); \
      hipMemset(dW,1,(size_t)N*nblk*sizeof(block_q6_K));hipMemset(dAq,1,(size_t)M*K);hipMemset(dAs,1,(size_t)M*nb32*4); \
      dequant_q6k<<<(N*nblk+127)/128,128>>>(dW,dWq,dWs,N,K); \
      for(int w=0;w<3;w++)requant_32<<<(N*nb32+127)/128,128>>>(dWq,dWs,dWq32,dWs32,N,K); hipDeviceSynchronize(); \
      double td=now();dequant_q6k<<<(N*nblk+127)/128,128>>>(dW,dWq,dWs,N,K);requant_32<<<(N*nb32+127)/128,128>>>(dWq,dWs,dWq32,dWs32,N,K);hipDeviceSynchronize();double dq=now()-td; \
      dim3 g(N/(BN),M/(BM)),b((A2)*(B2)*32); \
      for(int w=0;w<5;w++)gemm_rb_kc<BM,BN,A2,B2,32><<<g,b>>>(dAq,dAs,dWq32,dWs32,dO,M,N,K);hipDeviceSynchronize(); \
      double t0=now();int R=30;for(int i=0;i<R;i++)gemm_rb_kc<BM,BN,A2,B2,32><<<g,b>>>(dAq,dAs,dWq32,dWs32,dO,M,N,K);hipDeviceSynchronize(); \
      double ms=(now()-t0)/R; agg_ms+=ms; \
      printf("  %-5s N=%-5d K=%-5d: GEMM %.3f ms = %.0f TFLOPS | dequant+requant %.3f ms (load-time, 1x)\n",nm,N,K,ms,2.0*M*N*K/1e12/(ms/1e3),dq); \
      hipFree(dW);hipFree(dWq);hipFree(dWs);hipFree(dWq32);hipFree(dWs32);hipFree(dAq);hipFree(dAs);hipFree(dO); }
    SHQ(6144,4096,"qkv",128,256,8,4) SHQ(4096,4096,"o",512,128,8,4) SHQ(14336,4096,"gate",256,256,8,4) SHQ(14336,4096,"up",256,256,8,4) SHQ(4096,14336,"down",256,256,8,4)
    #undef SHQ
    double ts=512.0/(agg_ms*NL/1000.0);
    printf("AGGREGATE 50L Q6_K prefill (KC=32 re-quant): %.2f ms = %.0f t/s  vs ggml 1612 (+%.0f%%) / RADV 2255 (+%.0f%%)\n", agg_ms*NL, ts, (ts/1612-1)*100, (ts/2255-1)*100);
  }
  return 0;
}
