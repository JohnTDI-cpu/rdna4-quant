// fa2_v5.hip — GQA packing: 2 Q-heads sharing one KV-head processed per block.
// Bielik H=32 Hk=8 gqa_ratio=4. Heads (2b,2b+1) share kh=b/2 ... actually kh=(2b)/4; pairs (0,1)(2,3)
// fall in the same group of 4 so they share kh. K/V loaded ONCE, reused for both heads -> 2x less K/V
// global traffic. Cost: 2x Q-state (VGPR). Test vs v1 whether the K/V reuse beats the register pressure.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float    float8_t __attribute__((ext_vector_type(8)));
#include "/home/janusz/AMD MXFP4/hip_int4/flash_prefill.hip"
static double fnow(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}

template<int LB>
__global__ __launch_bounds__(32, LB)
void fa2_gqa2(const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, __half* __restrict__ O, int M, int H, int Hk)
{
    constexpr int Br=16, Bc=16, D=128, KT=D/16, NT=D/16;
    const int qb=blockIdx.x, hp=blockIdx.y, lane=threadIdx.x;     // hp = head-pair index
    const int h0=hp*2, h1=h0+1, gqa=H/Hk, kh=h0/gqa;
    const int q_row_base=qb*Br; if(q_row_base>=M) return;
    const int M_row_base=(lane/16)*8, col_lane=lane%16, q_row0=q_row_base+M_row_base;
    const _Float16* Q0=reinterpret_cast<const _Float16*>(Q)+(long)h0*M*D;
    const _Float16* Q1=reinterpret_cast<const _Float16*>(Q)+(long)h1*M*D;
    const _Float16* Kh=reinterpret_cast<const _Float16*>(K)+(long)kh*M*D;
    const _Float16* Vh=reinterpret_cast<const _Float16*>(V)+(long)kh*M*D;
    __half* O0=O+(long)h0*M*D; __half* O1=O+(long)h1*M*D;
    half8_t Qf0[KT],Qf1[KT];
    { int qr=q_row_base+col_lane; bool rv=qr<M;
      #pragma unroll
      for(int kt=0;kt<KT;kt++){ int kcb=kt*16+M_row_base;
        #pragma unroll
        for(int j=0;j<8;j++){ Qf0[kt][j]=rv?Q0[(long)qr*D+kcb+j]:(_Float16)0; Qf1[kt][j]=rv?Q1[(long)qr*D+kcb+j]:(_Float16)0; } } }
    float m0[8],l0[8],m1[8],l1[8]; float8_t Oa0[NT],Oa1[NT];
    #pragma unroll
    for(int j=0;j<8;j++){ m0[j]=-INFINITY;l0[j]=0;m1[j]=-INFINITY;l1[j]=0; }
    #pragma unroll
    for(int nt=0;nt<NT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++){ Oa0[nt][j]=0; Oa1[nt][j]=0; }
    const float log2e=1.4426950408889634f, scale=rsqrtf((float)D)*log2e;
    int s_max=q_row_base+Br; if(s_max>M)s_max=M;
    __shared__ _Float16 Ps[2][16][16];
    for(int s_start=0;s_start<s_max;s_start+=Bc){
        half8_t Kf[KT];   // load K ONCE, reuse for both heads
        { int sp=s_start+col_lane; bool sv=sp<M; const _Float16* kr=sv?(Kh+(long)sp*D):nullptr;
          #pragma unroll
          for(int kt=0;kt<KT;kt++){ int krb=kt*16+M_row_base;
            #pragma unroll
            for(int j=0;j<8;j++) Kf[kt][j]=sv?kr[krb+j]:(_Float16)0; } }
        float8_t S0,S1; for(int j=0;j<8;j++){S0[j]=0;S1[j]=0;}
        #pragma unroll
        for(int kt=0;kt<KT;kt++){ S0=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(Qf0[kt],Kf[kt],S0);
                                  S1=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(Qf1[kt],Kf[kt],S1); }
        { int cg=s_start+col_lane;
          #pragma unroll
          for(int j=0;j<8;j++){ int rg=q_row0+j; bool v=(cg<=rg)&&(cg<M); S0[j]=v?S0[j]*scale:-INFINITY; S1[j]=v?S1[j]*scale:-INFINITY; } }
        // softmax head0
        float rc0[8];
        #pragma unroll
        for(int j=0;j<8;j++){ float mm=S0[j]; mm=fmaxf(mm,__shfl_xor(mm,1));mm=fmaxf(mm,__shfl_xor(mm,2));mm=fmaxf(mm,__shfl_xor(mm,4));mm=fmaxf(mm,__shfl_xor(mm,8));
            float nm=fmaxf(m0[j],mm); float c=(nm==-INFINITY)?0:exp2f(m0[j]-nm); float e=(nm==-INFINITY)?0:exp2f(S0[j]-nm);
            S0[j]=e; float s=e; s+=__shfl_xor(s,1);s+=__shfl_xor(s,2);s+=__shfl_xor(s,4);s+=__shfl_xor(s,8); l0[j]=l0[j]*c+s; m0[j]=nm; rc0[j]=c; }
        float rc1[8];
        #pragma unroll
        for(int j=0;j<8;j++){ float mm=S1[j]; mm=fmaxf(mm,__shfl_xor(mm,1));mm=fmaxf(mm,__shfl_xor(mm,2));mm=fmaxf(mm,__shfl_xor(mm,4));mm=fmaxf(mm,__shfl_xor(mm,8));
            float nm=fmaxf(m1[j],mm); float c=(nm==-INFINITY)?0:exp2f(m1[j]-nm); float e=(nm==-INFINITY)?0:exp2f(S1[j]-nm);
            S1[j]=e; float s=e; s+=__shfl_xor(s,1);s+=__shfl_xor(s,2);s+=__shfl_xor(s,4);s+=__shfl_xor(s,8); l1[j]=l1[j]*c+s; m1[j]=nm; rc1[j]=c; }
        #pragma unroll
        for(int j=0;j<8;j++){ Ps[0][M_row_base+j][col_lane]=(_Float16)S0[j]; Ps[1][M_row_base+j][col_lane]=(_Float16)S1[j]; }
        __syncwarp();
        half8_t Pf0,Pf1;
        #pragma unroll
        for(int j=0;j<8;j++){ Pf0[j]=Ps[0][col_lane][M_row_base+j]; Pf1[j]=Ps[1][col_lane][M_row_base+j]; }
        __syncwarp();
        half8_t Vf[NT];   // load V ONCE, reuse
        #pragma unroll
        for(int nt=0;nt<NT;nt++){ int ncv=nt*16+col_lane;
          #pragma unroll
          for(int j=0;j<8;j++){ int sp=s_start+M_row_base+j; Vf[nt][j]=(sp<M)?Vh[(long)sp*D+ncv]:(_Float16)0; } }
        #pragma unroll
        for(int nt=0;nt<NT;nt++){
          #pragma unroll
          for(int j=0;j<8;j++){ Oa0[nt][j]*=rc0[j]; Oa1[nt][j]*=rc1[j]; }
          Oa0[nt]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(Pf0,Vf[nt],Oa0[nt]);
          Oa1[nt]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(Pf1,Vf[nt],Oa1[nt]); }
    }
    #pragma unroll
    for(int nt=0;nt<NT;nt++){ int col=nt*16+col_lane;
      #pragma unroll
      for(int j=0;j<8;j++){ int row=q_row0+j; if(row<M){ float i0=(l0[j]>0)?1.f/l0[j]:0, i1=(l1[j]>0)?1.f/l1[j]:0;
          O0[(long)row*D+col]=__float2half(Oa0[nt][j]*i0); O1[(long)row*D+col]=__float2half(Oa1[nt][j]*i1); } } }
}
template<int LB> void launch_gqa2(const __half*Q,const __half*K,const __half*V,__half*O,int M,int H,int Hk){
    dim3 g((M+15)/16, H/2); dim3 b(32); fa2_gqa2<LB><<<g,b>>>(Q,K,V,O,M,H,Hk);
}
#define CK(x) do{hipError_t e=(x);if(e)printf("ERR %s\n",hipGetErrorString(e));}while(0)
int main(){
    int H=32,Hk=8,D=128,NL=50;
    { int M=512; size_t qn=(size_t)M*H*D,kn=(size_t)M*Hk*D;
      std::vector<__half> hQ(qn),hK(kn),hV(kn); srand(9);
      for(auto&x:hQ)x=__float2half((rand()%200-100)/128.f); for(auto&x:hK)x=__float2half((rand()%200-100)/128.f); for(auto&x:hV)x=__float2half((rand()%200-100)/128.f);
      __half*Q,*K,*V,*O1,*O2; CK(hipMalloc(&Q,qn*2));CK(hipMalloc(&K,kn*2));CK(hipMalloc(&V,kn*2));CK(hipMalloc(&O1,qn*2));CK(hipMalloc(&O2,qn*2));
      CK(hipMemcpy(Q,hQ.data(),qn*2,hipMemcpyHostToDevice));CK(hipMemcpy(K,hK.data(),kn*2,hipMemcpyHostToDevice));CK(hipMemcpy(V,hV.data(),kn*2,hipMemcpyHostToDevice));
      launch_flash_prefill_v1(Q,K,V,O1,M,H,Hk,D,0); CK(hipDeviceSynchronize());
      launch_gqa2<2>(Q,K,V,O2,M,H,Hk); CK(hipDeviceSynchronize());
      std::vector<__half> a(qn),b(qn); CK(hipMemcpy(a.data(),O1,qn*2,hipMemcpyDeviceToHost));CK(hipMemcpy(b.data(),O2,qn*2,hipMemcpyDeviceToHost));
      double e=0,n=0; for(size_t i=0;i<qn;i++){float x=__half2float(a[i]),y=__half2float(b[i]);e+=(x-y)*(x-y);n+=x*x;}
      printf("gqa2 vs v1 (M=512): NRMSE=%.6f  %s\n",sqrt(e/n),sqrt(e/n)<0.01?"PASS":"FAIL");
      hipFree(Q);hipFree(K);hipFree(V);hipFree(O1);hipFree(O2); }
    printf("FA2 50L (ms min-of-5): v1 | gqa2\n");
    int ctxs[]={512,2048,8192};
    for(int ci=0;ci<3;ci++){ int M=ctxs[ci]; size_t qn=(size_t)M*H*D,kn=(size_t)M*Hk*D;
      __half*Q,*K,*V,*O; hipMalloc(&Q,qn*2);hipMalloc(&K,kn*2);hipMalloc(&V,kn*2);hipMalloc(&O,qn*2);
      hipMemset(Q,0,qn*2);hipMemset(K,0,kn*2);hipMemset(V,0,kn*2);
      #define TM(call) ({ for(int w=0;w<3;w++){for(int l=0;l<NL;l++)call;} hipDeviceSynchronize(); double bb=1e9; for(int r=0;r<5;r++){double t=fnow();for(int l=0;l<NL;l++)call;hipDeviceSynchronize();double d=fnow()-t;if(d<bb)bb=d;} bb; })
      double v1=TM((launch_flash_prefill_v1(Q,K,V,O,M,H,Hk,D,0)));
      double g2=TM((launch_gqa2<2>(Q,K,V,O,M,H,Hk)));
      printf("  M=%-6d %7.2f | %7.2f (%.3fx)\n",M,v1,g2,v1/g2);
      hipFree(Q);hipFree(K);hipFree(V);hipFree(O); }
    return 0;
}
