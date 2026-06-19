// WMMA INT8 GEMM microbench: out[M,N] = act[M,K] @ wt[N,K]^T (int8->int32). gfx1201.
// Layout (from validated swmmac dense): 32-lane w32, nl=lane%16 (N-col), kg=lane/16 (K-half), acc[j]=M-row.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define CK(x) do{hipError_t e=(x);if(e){printf("err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

__global__ void __launch_bounds__(32,8) wmma_i8(const int8_t* __restrict__ act,const int8_t* __restrict__ wt,int* __restrict__ out,int M,int N,int K){
  int lane=threadIdx.x; int nl=lane%16, kg=lane/16;
  int m_base=blockIdx.y*16, n_base=blockIdx.x*64;
  v8i R0={},R1={},R2={},R3={};
  for(int k=0;k<K;k+=16){
    const int8_t* ap=act+(size_t)(m_base+nl)*K+k+kg*8;
    v2i A={*(const int*)ap,*(const int*)(ap+4)};
    #define LB(ti) const int8_t* bp##ti=wt+(size_t)(n_base+ti*16+nl)*K+k+kg*8; v2i B##ti={*(const int*)bp##ti,*(const int*)(bp##ti+4)};
    LB(0) LB(1) LB(2) LB(3)
    #undef LB
    R0=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,A,true,B0,R0,false);
    R1=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,A,true,B1,R1,false);
    R2=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,A,true,B2,R2,false);
    R3=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,A,true,B3,R3,false);
  }
  #pragma unroll
  for(int j=0;j<8;j++){int m=m_base+kg*8+j;
    out[(size_t)m*N+n_base+nl]=R0[j]; out[(size_t)m*N+n_base+16+nl]=R1[j];
    out[(size_t)m*N+n_base+32+nl]=R2[j]; out[(size_t)m*N+n_base+48+nl]=R3[j];}
}
static int8_t i8(int v){return (int8_t)(((v%15)-7));}
int main(int argc,char**argv){
  int M=argc>1?atoi(argv[1]):512,K=argc>2?atoi(argv[2]):4096,N=argc>3?atoi(argv[3]):14336;
  std::vector<int8_t> ha((size_t)M*K),hw((size_t)N*K);
  for(size_t i=0;i<ha.size();i++)ha[i]=i8(i*7+1);
  for(size_t i=0;i<hw.size();i++)hw[i]=i8(i*5+3);
  int8_t *da,*dw; int* do_; CK(hipMalloc(&da,ha.size()));CK(hipMalloc(&dw,hw.size()));CK(hipMalloc(&do_,(size_t)M*N*4));
  CK(hipMemcpy(da,ha.data(),ha.size(),hipMemcpyHostToDevice));CK(hipMemcpy(dw,hw.data(),hw.size(),hipMemcpyHostToDevice));
  dim3 grid(N/64,M/16),blk(32);
  auto run=[&](){wmma_i8<<<grid,blk>>>(da,dw,do_,M,N,K);};
  run(); CK(hipDeviceSynchronize());
  std::vector<int> ho((size_t)M*N); CK(hipMemcpy(ho.data(),do_,(size_t)M*N*4,hipMemcpyDeviceToHost));
  // CPU validate a few
  int bad=0; for(int t=0;t<32;t++){int m=(t*37)%M,n=(t*53)%N;long ref=0;for(int k=0;k<K;k++)ref+=(int)ha[(size_t)m*K+k]*(int)hw[(size_t)n*K+k];if(ref!=ho[(size_t)m*N+n]){if(bad<3)printf("MISMATCH m=%d n=%d got=%d ref=%ld\n",m,n,ho[(size_t)m*N+n],ref);bad++;}}
  hipEvent_t s,e;hipEventCreate(&s);hipEventCreate(&e);
  double best=1e30;for(int r=0;r<30;r++){hipEventRecord(s);run();hipEventRecord(e);hipEventSynchronize(e);float ms;hipEventElapsedTime(&ms,s,e);best=std::min(best,(double)ms);}
  double tops=2.0*M*N*K/1e12/(best/1e3);
  printf("WMMA INT8 GEMM M=%d K=%d N=%d  CORRECT=%s (bad=%d)  best=%.3f ms  %.1f TOPS (Vulkan pp512 eff ~54)\n",M,K,N,bad==0?"YES":"NO",bad,best,tops);
  return 0;
}
