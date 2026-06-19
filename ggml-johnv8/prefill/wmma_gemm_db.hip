// Double-buffered (software-pipelined) tiled WMMA INT8 GEMM.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int i4 __attribute__((ext_vector_type(4)));
#define CK(x) do{hipError_t e=(x);if(e){printf("err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef KC
#define KC 32
#endif
#define NWARP (BM/16)
#define NTHREAD (NWARP*32)
#define NTI (BN/16)
#define AV (BM*KC/16)
#define BV (BN*KC/16)
#define APT ((AV+NTHREAD-1)/NTHREAD)
#define BPT ((BV+NTHREAD-1)/NTHREAD)
__global__ void __launch_bounds__(NTHREAD) wmma_gemm_db(const int8_t* __restrict__ act,const int8_t* __restrict__ wt,int* __restrict__ out,int M,int N,int K){
  __shared__ int8_t As[2][BM][KC];
  __shared__ int8_t Bs[2][BN][KC];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, nl=lane%16, kg=lane/16;
  int m_base=blockIdx.y*BM, n_base=blockIdx.x*BN;
  v8i R[NTI];
  #pragma unroll
  for(int i=0;i<NTI;i++)R[i]=(v8i){};
  // prologue: load chunk 0
  #pragma unroll
  for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[0][row][c]=*(const i4*)&act[(size_t)(m_base+row)*K+0+c];}}
  #pragma unroll
  for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[0][row][c]=*(const i4*)&wt[(size_t)(n_base+row)*K+0+c];}}
  __syncthreads();
  int buf=0;
  for(int k0=0;k0<K;k0+=KC){
    i4 rA[APT],rB[BPT]; int kn=k0+KC;
    if(kn<K){
      #pragma unroll
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rA[p]=*(const i4*)&act[(size_t)(m_base+row)*K+kn+c];}}
      #pragma unroll
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;rB[p]=*(const i4*)&wt[(size_t)(n_base+row)*K+kn+c];}}
    }
    #pragma unroll
    for(int ks=0;ks<KC/16;ks++){
      const int8_t* ap=&As[buf][warp*16+nl][ks*16+kg*8];
      v2i A={*(const int*)ap,*(const int*)(ap+4)};
      #pragma unroll
      for(int ti=0;ti<NTI;ti++){
        const int8_t* bp=&Bs[buf][ti*16+nl][ks*16+kg*8];
        v2i B={*(const int*)bp,*(const int*)(bp+4)};
        R[ti]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,A,true,B,R[ti],false);
      }
    }
    if(kn<K){
      #pragma unroll
      for(int p=0;p<APT;p++){int idx=tid+p*NTHREAD;if(idx<AV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&As[buf^1][row][c]=rA[p];}}
      #pragma unroll
      for(int p=0;p<BPT;p++){int idx=tid+p*NTHREAD;if(idx<BV){int row=idx/(KC/16),c=(idx%(KC/16))*16;*(i4*)&Bs[buf^1][row][c]=rB[p];}}
    }
    __syncthreads();
    buf^=1;
  }
  #pragma unroll
  for(int ti=0;ti<NTI;ti++)
    #pragma unroll
    for(int j=0;j<8;j++){int m=m_base+warp*16+kg*8+j;out[(size_t)m*N+n_base+ti*16+nl]=R[ti][j];}
}
static int8_t i8(int v){return (int8_t)(((v%15)-7));}
int main(int argc,char**argv){
  int M=argc>1?atoi(argv[1]):512,K=argc>2?atoi(argv[2]):4096,N=argc>3?atoi(argv[3]):14336;
  std::vector<int8_t> ha((size_t)M*K),hw((size_t)N*K);
  for(size_t i=0;i<ha.size();i++)ha[i]=i8(i*7+1);
  for(size_t i=0;i<hw.size();i++)hw[i]=i8(i*5+3);
  int8_t *da,*dw; int* do_; CK(hipMalloc(&da,ha.size()));CK(hipMalloc(&dw,hw.size()));CK(hipMalloc(&do_,(size_t)M*N*4));
  CK(hipMemcpy(da,ha.data(),ha.size(),hipMemcpyHostToDevice));CK(hipMemcpy(dw,hw.data(),hw.size(),hipMemcpyHostToDevice));
  dim3 grid(N/BN,M/BM),blk(NTHREAD);
  auto run=[&](){wmma_gemm_db<<<grid,blk>>>(da,dw,do_,M,N,K);};
  run();CK(hipDeviceSynchronize());
  std::vector<int> ho((size_t)M*N);CK(hipMemcpy(ho.data(),do_,(size_t)M*N*4,hipMemcpyDeviceToHost));
  int bad=0;for(int t=0;t<48;t++){int m=(t*37)%M,n=(t*53)%N;long ref=0;for(int k=0;k<K;k++)ref+=(int)ha[(size_t)m*K+k]*(int)hw[(size_t)n*K+k];if(ref!=ho[(size_t)m*N+n])bad++;}
  hipEvent_t s,e;hipEventCreate(&s);hipEventCreate(&e);
  double best=1e30;for(int r=0;r<50;r++){hipEventRecord(s);run();hipEventRecord(e);hipEventSynchronize(e);float ms;hipEventElapsedTime(&ms,s,e);best=std::min(best,(double)ms);}
  double tops=2.0*M*N*K/1e12/(best/1e3);
  printf("WMMA INT8 DB [%dx%d KC%d] M=%d K=%d N=%d  %s  %.3f ms  %.1f TOPS\n",BM,BN,KC,M,K,N,bad==0?"OK":"BAD",best,tops);
  return 0;
}
