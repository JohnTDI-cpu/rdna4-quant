// fa2_timing.hip — measure FA2 (flash_prefill v1) for Bielik-11B attention across contexts,
// 50 layers. Combined with the GEMM aggregate -> full-forward prefill estimate.
// Bielik dims: H(NH)=32, Hk(NKV)=8, D(HD)=128.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <ctime>
#include "/home/janusz/AMD MXFP4/hip_int4/flash_prefill.hip"
static double fnow(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
int main(){
  int H=32,Hk=8,D=128,NL=50;
  int ctxs[]={256,512,1024,2048,4096,8192,16384,32768};
  printf("FA2 (flash_prefill v1) Bielik attention H=32 Hk=8 D=128, 50 layers:\n");
  for(int ci=0;ci<8;ci++){ int M=ctxs[ci];
    __half *Q,*K,*V,*O;
    hipMalloc(&Q,(size_t)M*H*D*2);hipMalloc(&K,(size_t)M*Hk*D*2);hipMalloc(&V,(size_t)M*Hk*D*2);hipMalloc(&O,(size_t)M*H*D*2);
    hipMemset(Q,0,(size_t)M*H*D*2);hipMemset(K,0,(size_t)M*Hk*D*2);hipMemset(V,0,(size_t)M*Hk*D*2);
    for(int w=0;w<3;w++)for(int l=0;l<NL;l++)launch_flash_prefill_v1(Q,K,V,O,M,H,Hk,D,0); hipDeviceSynchronize();
    double t0=fnow();int R=3;for(int i=0;i<R;i++)for(int l=0;l<NL;l++)launch_flash_prefill_v1(Q,K,V,O,M,H,Hk,D,0);hipDeviceSynchronize();
    double fa=(fnow()-t0)/R;
    printf("  M=%-6d FA2 50L: %8.2f ms\n",M,fa);
    hipFree(Q);hipFree(K);hipFree(V);hipFree(O);
  }
  return 0;
}
