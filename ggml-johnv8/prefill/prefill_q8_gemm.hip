// prefill_q8_gemm.hip — scale-aware register-blocked INT8 WMMA GEMM for Q8_0 prefill.
// Adds per-32-block scales (w_scale[N,K/32], a_scale[M,K/32]) to the proven gemm_rb:
// per 32-K block (= Q8_0 block): WMMA i8 dot -> int32, x w_scale x a_scale, accumulate in float.
// First building block of the full prefill forward. Build: hipcc --offload-arch=gfx1201 -O3
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
// SPLIT-K variant: gridDim.z splits K, each block does a K-slice, atomicAdd partials to float out.
// For under-utilized shapes (few M*N blocks) -> more blocks -> fill CUs -> hide latency.
template<int BM,int BN,int WMW,int WNW>
__global__ void __launch_bounds__(WMW*WNW*32) gemm_rb_q8_sk(
    const int8_t* __restrict__ act, const float* __restrict__ a_scale,
    const int8_t* __restrict__ wt,  const float* __restrict__ w_scale,
    float* __restrict__ out, int M, int N, int K){
  constexpr int KC=32, NWARP=WMW*WNW, NTHREAD=NWARP*32, WMT=BM/(WMW*16), WNT=BN/(WNW*16);
  constexpr int AV=BM*KC/16, BV=BN*KC/16, APT=(AV+NTHREAD-1)/NTHREAD, BPT=(BV+NTHREAD-1)/NTHREAD;
  __shared__ int8_t As[2][BM][KC]; __shared__ int8_t Bs[2][BN][KC];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16, wm=warp/WNW, wn=warp%WNW;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN, nb=K/32;
  int Kslice=K/gridDim.z, k0s=blockIdx.z*Kslice, k0e=k0s+Kslice;
  float Rf[WMT][WNT][8];
  for(int a=0;a<WMT;a++)for(int b=0;b<WNT;b++)for(int j=0;j<8;j++)Rf[a][b][j]=0.f;
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+k0s+c];}}
  for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[0][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+k0s+c];}}
  __syncthreads(); int buf=0;
  for(int k0=k0s;k0<k0e;k0+=KC){ int blk=k0>>5; i4 rA[APT],rB[BPT]; int kn=k0+KC;
    if(kn<k0e){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kn+c];}}
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
    if(kn<k0e){ for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}} }
    __syncthreads(); buf^=1; }
  for(int mt=0;mt<WMT;mt++)for(int nt=0;nt<WNT;nt++)for(int j=0;j<8;j++){int m=m_base+wm*WMT*16+mt*16+kg*8+j;atomicAdd(&out[(size_t)m*N+n_base+wn*WNT*16+nt*16+nl],Rf[mt][nt][j]);}
}
#define CK(x) do{hipError_t e=(x);if(e){printf("HIP err %s\n",hipGetErrorString(e));}}while(0)
int main(){
  { // VALIDATION (small, CPU reference)
    int M=64,N=128,K=128,nb=K/32;
    std::vector<int8_t> hw(N*K),ha(M*K); std::vector<float> hws(N*nb),has(M*nb);
    srand(1); for(auto&x:hw)x=rand()%255-127; for(auto&x:ha)x=rand()%255-127;
    for(auto&x:hws)x=(rand()%100)/1000.f+0.005f; for(auto&x:has)x=(rand()%100)/1000.f+0.005f;
    std::vector<float> ref(M*N);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){float s=0;for(int b=0;b<nb;b++){int id=0;for(int k=0;k<32;k++)id+=(int)hw[n*K+b*32+k]*(int)ha[m*K+b*32+k];s+=id*hws[n*nb+b]*has[m*nb+b];}ref[m*N+n]=s;}
    int8_t*dw,*da; float*dws,*das; __half*dout;
    CK(hipMalloc(&dw,N*K));CK(hipMalloc(&da,M*K));CK(hipMalloc(&dws,N*nb*4));CK(hipMalloc(&das,M*nb*4));CK(hipMalloc(&dout,M*N*2));
    CK(hipMemcpy(dw,hw.data(),N*K,hipMemcpyHostToDevice));CK(hipMemcpy(da,ha.data(),M*K,hipMemcpyHostToDevice));
    CK(hipMemcpy(dws,hws.data(),N*nb*4,hipMemcpyHostToDevice));CK(hipMemcpy(das,has.data(),M*nb*4,hipMemcpyHostToDevice));
    dim3 g(N/64,M/64),b(2*2*32); gemm_rb_q8<64,64,2,2><<<g,b>>>(da,das,dw,dws,dout,M,N,K); CK(hipDeviceSynchronize());
    std::vector<__half> ho(M*N); CK(hipMemcpy(ho.data(),dout,M*N*2,hipMemcpyDeviceToHost));
    float maxe=0; for(int i=0;i<M*N;i++){float e=fabsf(__half2float(ho[i])-ref[i])/(fabsf(ref[i])+1e-3f);if(e>maxe)maxe=e;}
    printf("VALIDATION M=64 N=128 K=128 (per-block scales): max rel err=%.4f  %s\n",maxe,maxe<0.02f?"PASS":"FAIL");
    hipFree(dw);hipFree(da);hipFree(dws);hipFree(das);hipFree(dout);
  }
  { // PER-SHAPE SWEEP — scale-aware Q8, all 5 GEMM shapes (M=512), aggregate prefill
    double agg_ms=0; int NL=50;
    #define TC(BM,BN,A,B,id) { dim3 g(N/(BN),M/(BM)),b((A)*(B)*32); \
      for(int w=0;w<5;w++)gemm_rb_q8<BM,BN,A,B><<<g,b>>>(da,das,dw,dws,dout,M,N,K);hipDeviceSynchronize(); \
      double t0=now();int R=30;for(int i=0;i<R;i++)gemm_rb_q8<BM,BN,A,B><<<g,b>>>(da,das,dw,dws,dout,M,N,K);hipDeviceSynchronize(); \
      double ms=(now()-t0)/R; if(ms<best){best=ms;bb=id;} }
    #define SH(NN,KK,nm) { int M=512,N=NN,K=KK; size_t nb=K/32; \
      int8_t*dw,*da;float*dws,*das;__half*dout; \
      hipMalloc(&dw,(size_t)N*K);hipMalloc(&da,(size_t)M*K);hipMalloc(&dws,(size_t)N*nb*4);hipMalloc(&das,(size_t)M*nb*4);hipMalloc(&dout,(size_t)M*N*2); \
      hipMemset(dw,1,(size_t)N*K);hipMemset(da,1,(size_t)M*K); double best=1e9;int bb=0; \
      TC(256,256,8,4,1) TC(512,128,8,4,2) TC(256,128,8,4,3) TC(128,256,8,4,4) TC(256,256,16,2,5) \
      printf("  %-5s N=%-5d K=%-5d: best %.3f ms = %.0f TFLOPS (cfg%d)\n",nm,N,K,best,2.0*M*N*K/1e12/(best/1e3),bb); \
      agg_ms+=best; hipFree(dw);hipFree(da);hipFree(dws);hipFree(das);hipFree(dout); }
    printf("PER-SHAPE scale-aware Q8 GEMM (M=512, vs non-scale: qkv82 o70 gate82 down102):\n");
    SH(6144,4096,"qkv") SH(4096,4096,"o") SH(14336,4096,"gate") SH(14336,4096,"up") SH(4096,14336,"down")
    #undef SH
    #undef TC
    printf("AGGREGATE 50L scale-aware Q8 prefill GEMM: %.2f ms = %.0f t/s  (ggml/ROCm pp512 ~2811 t/s = 62 TFLOPS)\n", agg_ms*NL, 512.0/(agg_ms*NL/1000.0));
  }
  { // SPLIT-K (atomic accumulation) on under-utilized shapes — more blocks fill CUs
    printf("SPLIT-K atomic <256,256,8,4> (o/qkv few blocks; per-shape non-SK was o56 qkv52):\n");
    #define SKT(NN,KK,nm) { int M=512,N=NN,K=KK; size_t nb=K/32; \
      int8_t*dw,*da;float*dws,*das,*dout; \
      hipMalloc(&dw,(size_t)N*K);hipMalloc(&da,(size_t)M*K);hipMalloc(&dws,(size_t)N*nb*4);hipMalloc(&das,(size_t)M*nb*4);hipMalloc(&dout,(size_t)M*N*4); \
      hipMemset(dw,1,(size_t)N*K);hipMemset(da,1,(size_t)M*K); \
      for(int sk=1;sk<=16;sk*=2){ dim3 g(N/256,M/256,sk),b(8*4*32); \
        hipMemset(dout,0,(size_t)M*N*4);for(int w=0;w<5;w++)gemm_rb_q8_sk<256,256,8,4><<<g,b>>>(da,das,dw,dws,dout,M,N,K);hipDeviceSynchronize(); \
        double t0=now();int R=30;for(int i=0;i<R;i++){hipMemset(dout,0,(size_t)M*N*4);gemm_rb_q8_sk<256,256,8,4><<<g,b>>>(da,das,dw,dws,dout,M,N,K);}hipDeviceSynchronize(); \
        double ms=(now()-t0)/R; printf("  %-4s splitK=%-2d (%4d blk): %.3f ms = %.0f TFLOPS\n",nm,sk,(N/256)*(M/256)*sk,ms,2.0*M*N*K/1e12/(ms/1e3)); } \
      hipFree(dw);hipFree(da);hipFree(dws);hipFree(das);hipFree(dout); }
    SKT(4096,4096,"o") SKT(6144,4096,"qkv")
    #undef SKT
  }
  return 0;
}
