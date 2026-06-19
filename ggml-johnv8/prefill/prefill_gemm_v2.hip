// Prefill GEMM-throughput v2: per-shape tuned tile configs (templated kernel).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int i4 __attribute__((ext_vector_type(4)));
#define CK(x) do{hipError_t e=(x);if(e){printf("err %s\n",hipGetErrorString(e));exit(1);}}while(0)
template<int BM,int BN,int KC>
__global__ void __launch_bounds__(BM/16*32) gemm(const int8_t* __restrict__ act,const int8_t* __restrict__ wt,int* __restrict__ out,int M,int N,int K){
  constexpr int NTHREAD=BM/16*32, NTI=BN/16, AV=BM*KC/16, BV=BN*KC/16;
  constexpr int APT=(AV+NTHREAD-1)/NTHREAD, BPT=(BV+NTHREAD-1)/NTHREAD;
  __shared__ int8_t As[2][BM][KC]; __shared__ int8_t Bs[2][BN][KC];
  int tid=threadIdx.x,warp=tid/32,lane=tid%32,nl=lane%16,kg=lane/16;
  int m_base=blockIdx.y*BM,n_base=blockIdx.x*BN; v8i R[NTI];
  #pragma unroll
  for(int i=0;i<NTI;i++)R[i]=(v8i){};
  #pragma unroll
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+c];}}
  #pragma unroll
  for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[0][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+c];}}
  __syncthreads(); int buf=0;
  for(int k0=0;k0<K;k0+=KC){ i4 rA[APT],rB[BPT]; int kn=k0+KC;
    if(kn<K){
      #pragma unroll
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kn+c];}}
      #pragma unroll
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rB[p]=*(const i4*)&wt[(size_t)(n_base+row)*K+kn+c];}}
    }
    #pragma unroll
    for(int ks=0;ks<KC/16;ks++){const int8_t* ap=&As[buf][warp*16+nl][ks*16+kg*8];v2i A={*(const int*)ap,*(const int*)(ap+4)};
      #pragma unroll
      for(int ti=0;ti<NTI;ti++){const int8_t* bp=&Bs[buf][ti*16+nl][ks*16+kg*8];v2i B={*(const int*)bp,*(const int*)(bp+4)};
        R[ti]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,A,true,B,R[ti],false);}}
    if(kn<K){
      #pragma unroll
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      #pragma unroll
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}}
    }
    __syncthreads(); buf^=1;
  }
  #pragma unroll
  for(int ti=0;ti<NTI;ti++)
    #pragma unroll
    for(int j=0;j<8;j++){int m=m_base+warp*16+kg*8+j;out[(size_t)m*N+n_base+ti*16+nl]=R[ti][j];}
}
template<int BM,int BN,int KC> void launch(int8_t*a,int8_t*w,int*o,int M,int N,int K){ dim3 g(N/BN,M/BM),b(BM/16*32); gemm<BM,BN,KC><<<g,b>>>(a,w,o,M,N,K);}
int main(){
  int M=512,NL=50,maxK=14336,maxN=28672;
  int8_t *act,*wt; int* out;
  CK(hipMalloc(&act,(size_t)M*maxK)); CK(hipMalloc(&wt,(size_t)maxN*maxK)); CK(hipMalloc(&out,(size_t)M*maxN*4));
  CK(hipMemset(act,1,(size_t)M*maxK)); CK(hipMemset(wt,1,(size_t)maxN*maxK));
  auto layer=[&](){
    launch<256,128,32>(act,wt,out,M,6144,4096);   // qkv
    launch<256,256,32>(act,wt,out,M,4096,4096);   // o
    launch<512,256,16>(act,wt,out,M,28672,4096);  // gate+up FUSED
    launch<256,256,16>(act,wt,out,M,4096,14336);  // down (256x256 = +52% vs 512x256)
  };
  layer(); CK(hipDeviceSynchronize());
  hipEvent_t a,b;hipEventCreate(&a);hipEventCreate(&b);
  double best=1e30; for(int r=0;r<8;r++){ hipEventRecord(a); for(int l=0;l<NL;l++)layer(); hipEventRecord(b); hipEventSynchronize(b); float ms;hipEventElapsedTime(&ms,a,b); best=std::min(best,(double)ms);}
  double tps=M/(best/1000.0);
  double tflop=2.0*M*NL*((double)6144*4096+4096*4096+14336*4096+14336*4096+4096*14336)/1e12;
  printf("PREFILL GEMM-only per-shape-tuned (50L, M=512): %.1f ms = %.0f t/s (%.0f TFLOPS)  [ROCm Q8=2781, Vulkan=2451]\n",best,tps,tflop/(best/1000.0));
  return 0;
}
