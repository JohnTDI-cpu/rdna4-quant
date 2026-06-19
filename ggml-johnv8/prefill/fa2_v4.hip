// fa2_v4.hip — FA-4 lever: off-SFU polynomial exp2 in the FA2 softmax.
// RDNA4 exp/TLU runs 1/4-rate but CO-EXECUTES with WMMA. exp2f goes through the SFU;
// a 3rd-order polynomial 2^f (Horner on FMA) runs on the full-rate VALU and overlaps the WMMA.
// Templated fa2_exp<LB,POLY>: POLY=false == current best (fa2_v3_exp2, exp2f), POLY=true == poly.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float    float8_t __attribute__((ext_vector_type(8)));
#include "/home/janusz/AMD MXFP4/hip_int4/flash_prefill.hip"   // v1 reference + launch_flash_prefill_v1
static double fnow(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}

// off-SFU 2^x for x<=0 (input already in log2 domain). Horner cubic on [0,1) + bit-manip 2^n.
__device__ __forceinline__ float fexp2(float x){
  x = fmaxf(x, -126.0f);                         // guard -inf/underflow -> ~0
  float fl = floorf(x); int n = (int)fl; float f = x - fl;   // f in [0,1)
  float p = 0.0771f; p = 0.2276f + f*p; p = 0.6951f + f*p; p = 1.0f + f*p;  // 2^f, err ~1e-4
  return p * __int_as_float((n + 127) << 23);    // * 2^n
}

template<int LB, bool POLY>
__global__ __launch_bounds__(32, LB)
void fa2_exp(const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, __half* __restrict__ O, int M, int H, int Hk)
{
    constexpr int Br = 16, Bc = 16, D = 128, KT = D/16, NT = D/16;
    const int qb = blockIdx.x, qh = blockIdx.y, lane = threadIdx.x;
    const int gqa_ratio = H / Hk, kh = qh / gqa_ratio;
    const int q_row_base = qb * Br; if (q_row_base >= M) return;
    const int M_row_base = (lane / 16) * 8, col_lane = lane % 16;
    const int q_row0 = q_row_base + M_row_base;
    const _Float16* Q_head = reinterpret_cast<const _Float16*>(Q) + (long)qh * M * D;
    const _Float16* K_head = reinterpret_cast<const _Float16*>(K) + (long)kh * M * D;
    const _Float16* V_head = reinterpret_cast<const _Float16*>(V) + (long)kh * M * D;
    __half* O_head = O + (long)qh * M * D;
    half8_t Qf[KT];
    { int q_row_lane = q_row_base + col_lane; bool rv = q_row_lane < M;
      #pragma unroll
      for (int kt = 0; kt < KT; kt++){ int kcb = kt*16 + M_row_base;
        #pragma unroll
        for (int j = 0; j < 8; j++) Qf[kt][j] = rv ? Q_head[(long)q_row_lane*D + kcb + j] : (_Float16)0; } }
    float m_i[8], l_i[8];
    #pragma unroll
    for (int j=0;j<8;j++){ m_i[j]=-INFINITY; l_i[j]=0.f; }
    float8_t Oacc[NT];
    #pragma unroll
    for (int nt=0;nt<NT;nt++)
      #pragma unroll
      for (int j=0;j<8;j++) Oacc[nt][j]=0.f;
    const float log2e = 1.4426950408889634f;
    const float scale = rsqrtf((float)D) * log2e;
    int s_max = q_row_base + Br; if (s_max > M) s_max = M;
    __shared__ _Float16 P_smem[16][16];
    for (int s_start = 0; s_start < s_max; s_start += Bc) {
        half8_t Kf[KT];
        { int s_pos = s_start + col_lane; bool sv = s_pos < M;
          const _Float16* kr = sv ? (K_head + (long)s_pos*D) : nullptr;
          #pragma unroll
          for (int kt=0;kt<KT;kt++){ int krb = kt*16 + M_row_base;
            #pragma unroll
            for (int j=0;j<8;j++) Kf[kt][j] = sv ? kr[krb+j] : (_Float16)0; } }
        float8_t Sacc; for (int j=0;j<8;j++) Sacc[j]=0.f;
        #pragma unroll
        for (int kt=0;kt<KT;kt++) Sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(Qf[kt],Kf[kt],Sacc);
        { int cg = s_start + col_lane;
          #pragma unroll
          for (int j=0;j<8;j++){ int rg = q_row0+j; bool v = (cg<=rg)&&(cg<M); Sacc[j] = v ? Sacc[j]*scale : -INFINITY; } }
        float row_max[8];
        #pragma unroll
        for (int j=0;j<8;j++){ float m=Sacc[j];
            m=fmaxf(m,__shfl_xor(m,1)); m=fmaxf(m,__shfl_xor(m,2)); m=fmaxf(m,__shfl_xor(m,4)); m=fmaxf(m,__shfl_xor(m,8));
            row_max[j]=m; }
        float row_corr[8];
        #pragma unroll
        for (int j=0;j<8;j++){ float nm=fmaxf(m_i[j],row_max[j]);
            float corr = (nm==-INFINITY)?0.f:(POLY?fexp2(m_i[j]-nm):exp2f(m_i[j]-nm));
            float es   = (nm==-INFINITY)?0.f:(POLY?fexp2(Sacc[j]-nm):exp2f(Sacc[j]-nm));
            Sacc[j]=es; float s=es;
            s+=__shfl_xor(s,1); s+=__shfl_xor(s,2); s+=__shfl_xor(s,4); s+=__shfl_xor(s,8);
            l_i[j]=l_i[j]*corr+s; m_i[j]=nm; row_corr[j]=corr; }
        #pragma unroll
        for (int j=0;j<8;j++) P_smem[M_row_base+j][col_lane]=(_Float16)Sacc[j];
        __syncwarp();
        half8_t Pf;
        #pragma unroll
        for (int j=0;j<8;j++) Pf[j]=P_smem[col_lane][M_row_base+j];
        __syncwarp();
        half8_t Vf[NT];
        #pragma unroll
        for (int nt=0;nt<NT;nt++){ int ncv = nt*16 + col_lane;
          #pragma unroll
          for (int j=0;j<8;j++){ int sp = s_start + M_row_base + j; Vf[nt][j] = (sp<M)? V_head[(long)sp*D+ncv] : (_Float16)0; } }
        #pragma unroll
        for (int nt=0;nt<NT;nt++){
          #pragma unroll
          for (int j=0;j<8;j++) Oacc[nt][j]*=row_corr[j];
          Oacc[nt]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(Pf,Vf[nt],Oacc[nt]); }
    }
    #pragma unroll
    for (int nt=0;nt<NT;nt++){ int col = nt*16 + col_lane;
      #pragma unroll
      for (int j=0;j<8;j++){ int row = q_row0+j; if (row<M){ float il=(l_i[j]>0.f)?1.f/l_i[j]:0.f; O_head[(long)row*D+col]=__float2half(Oacc[nt][j]*il); } } }
}
template<int LB,bool POLY> void launch_fa2(const __half*Q,const __half*K,const __half*V,__half*O,int M,int H,int Hk){
    dim3 grid((M+15)/16, H); dim3 block(32); fa2_exp<LB,POLY><<<grid,block>>>(Q,K,V,O,M,H,Hk);
}
#define CK(x) do{hipError_t e=(x);if(e)printf("ERR %s\n",hipGetErrorString(e));}while(0)

int main(){
    int H=32,Hk=8,D=128,NL=50;
    // correctness: poly vs v1
    { int M=512; size_t qn=(size_t)M*H*D, kn=(size_t)M*Hk*D;
      std::vector<__half> hQ(qn),hK(kn),hV(kn); srand(5);
      for(auto&x:hQ)x=__float2half((rand()%200-100)/128.f);
      for(auto&x:hK)x=__float2half((rand()%200-100)/128.f);
      for(auto&x:hV)x=__float2half((rand()%200-100)/128.f);
      __half *Q,*K,*V,*O1,*O2; CK(hipMalloc(&Q,qn*2));CK(hipMalloc(&K,kn*2));CK(hipMalloc(&V,kn*2));CK(hipMalloc(&O1,qn*2));CK(hipMalloc(&O2,qn*2));
      CK(hipMemcpy(Q,hQ.data(),qn*2,hipMemcpyHostToDevice));CK(hipMemcpy(K,hK.data(),kn*2,hipMemcpyHostToDevice));CK(hipMemcpy(V,hV.data(),kn*2,hipMemcpyHostToDevice));
      launch_flash_prefill_v1(Q,K,V,O1,M,H,Hk,D,0); CK(hipDeviceSynchronize());
      launch_fa2<4,true>(Q,K,V,O2,M,H,Hk); CK(hipDeviceSynchronize());
      std::vector<__half> a(qn),b(qn); CK(hipMemcpy(a.data(),O1,qn*2,hipMemcpyDeviceToHost));CK(hipMemcpy(b.data(),O2,qn*2,hipMemcpyDeviceToHost));
      double e=0,n=0; for(size_t i=0;i<qn;i++){float x=__half2float(a[i]),y=__half2float(b[i]);e+=(x-y)*(x-y);n+=x*x;}
      printf("poly vs v1 (M=512): NRMSE=%.6f  %s\n",sqrt(e/n),sqrt(e/n)<0.01?"PASS":"FAIL");
      hipFree(Q);hipFree(K);hipFree(V);hipFree(O1);hipFree(O2); }
    // speed: v1 vs exp2f(POLY=0) vs poly(POLY=1), min-of-bursts
    printf("FA2 50L (ms, min-of-5): v1 | exp2f | poly\n");
    int ctxs[]={512,2048,8192};
    for(int ci=0;ci<3;ci++){ int M=ctxs[ci]; size_t qn=(size_t)M*H*D,kn=(size_t)M*Hk*D;
      __half *Q,*K,*V,*O; hipMalloc(&Q,qn*2);hipMalloc(&K,kn*2);hipMalloc(&V,kn*2);hipMalloc(&O,qn*2);
      hipMemset(Q,0,qn*2);hipMemset(K,0,kn*2);hipMemset(V,0,kn*2);
      #define TIME(call) ({ for(int w=0;w<3;w++){for(int l=0;l<NL;l++)call;} hipDeviceSynchronize(); double best=1e9; for(int r=0;r<5;r++){double t=fnow();for(int l=0;l<NL;l++)call;hipDeviceSynchronize();double d=fnow()-t;if(d<best)best=d;} best; })
      double v1=TIME((launch_flash_prefill_v1(Q,K,V,O,M,H,Hk,D,0)));
      double e2=TIME((launch_fa2<4,false>(Q,K,V,O,M,H,Hk)));
      double po=TIME((launch_fa2<4,true>(Q,K,V,O,M,H,Hk)));
      printf("  M=%-6d %7.2f | %7.2f (%.3fx) | %7.2f (%.3fx vs v1, %.3fx vs exp2f)\n",M,v1,e2,v1/e2,po,v1/po,e2/po);
      hipFree(Q);hipFree(K);hipFree(V);hipFree(O); }
    return 0;
}
