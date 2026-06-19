// fa2_v2.hip — FA2 v2: multi-warp block + LDS K/V caching.
// v1 = 1 warp/block, K/V reloaded from global per Q-block (no reuse). v2 = NW warps/block,
// K/V tile loaded into LDS cooperatively ONCE per KV step, all NW warps reuse it.
// -> NW× less K/V global traffic + better occupancy. Validated vs v1 (SDPA-parity).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float    float8_t __attribute__((ext_vector_type(8)));
#include "/home/janusz/AMD MXFP4/hip_int4/flash_prefill.hip"   // v1 reference + launch_flash_prefill_v1
static double fnow(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}

template<int NW>
__global__ __launch_bounds__(NW*32, 2)
void flash_prefill_fwd_v2(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, __half* __restrict__ O, int M, int H, int Hk)
{
    constexpr int Br=16, Bc=16, D=128, KT=D/16, NT=D/16, BR_BLK=NW*16;
    const int w = threadIdx.y, lane = threadIdx.x;
    const int qh = blockIdx.y, gqa = H/Hk, kh = qh/gqa;
    const int q_row_base = blockIdx.x*BR_BLK + w*16;
    const int M_row_base = (lane/16)*8, col_lane = lane%16;
    const int q_row0 = q_row_base + M_row_base;

    const _Float16* Qh = reinterpret_cast<const _Float16*>(Q) + (long)qh*M*D;
    const _Float16* Kh = reinterpret_cast<const _Float16*>(K) + (long)kh*M*D;
    const _Float16* Vh = reinterpret_cast<const _Float16*>(V) + (long)kh*M*D;
    __half* Oh = O + (long)qh*M*D;

    __shared__ _Float16 Ks[NW>0?Bc:1][D], Vs[Bc][D];   // shared KV tile (one buffer, re-filled per step)
    __shared__ _Float16 Psm[NW][16][16];               // per-warp P transpose

    // Q fragments (per warp, loaded once)
    half8_t Qf[KT];
    { int qr = q_row_base + col_lane; bool ok = qr < M;
      #pragma unroll
      for(int kt=0;kt<KT;kt++){ int kc=kt*16+M_row_base;
        #pragma unroll
        for(int j=0;j<8;j++) Qf[kt][j] = ok ? Qh[(long)qr*D + kc+j] : (_Float16)0; } }

    float m_i[8], l_i[8]; float8_t Oacc[NT];
    #pragma unroll
    for(int j=0;j<8;j++){ m_i[j]=-INFINITY; l_i[j]=0.f; }
    #pragma unroll
    for(int nt=0;nt<NT;nt++)
      #pragma unroll
      for(int j=0;j<8;j++) Oacc[nt][j]=0.f;
    const float scale = rsqrtf((float)D);

    int warp_smax = q_row_base + Br; if(warp_smax>M) warp_smax=M;
    int blk_smax  = blockIdx.x*BR_BLK + BR_BLK; if(blk_smax>M) blk_smax=M;
    const int tid = w*32+lane, NTHR = NW*32;

    for(int s_start=0; s_start<blk_smax; s_start+=Bc){
        // cooperative LDS load of K,V tile [Bc][D]
        for(int i=tid;i<Bc*D;i+=NTHR){ int row=i/D,col=i%D; int sp=s_start+row;
            Ks[row][col] = (sp<M)? Kh[(long)sp*D+col] : (_Float16)0;
            Vs[row][col] = (sp<M)? Vh[(long)sp*D+col] : (_Float16)0; }
        __syncthreads();

        if(s_start < warp_smax){
            // S = Q @ K^T  (K from LDS)
            half8_t Kf[KT];
            #pragma unroll
            for(int kt=0;kt<KT;kt++){ int kc=kt*16+M_row_base;
              #pragma unroll
              for(int j=0;j<8;j++) Kf[kt][j]=Ks[col_lane][kc+j]; }
            float8_t S; for(int j=0;j<8;j++)S[j]=0.f;
            #pragma unroll
            for(int kt=0;kt<KT;kt++) S=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(Qf[kt],Kf[kt],S);
            int cg=s_start+col_lane;
            #pragma unroll
            for(int j=0;j<8;j++){ int rg=q_row0+j; bool v=(cg<=rg)&&(cg<M); S[j]=v?S[j]*scale:-INFINITY; }
            // online softmax
            float rmax[8];
            #pragma unroll
            for(int j=0;j<8;j++){ float m=S[j]; m=fmaxf(m,__shfl_xor(m,1)); m=fmaxf(m,__shfl_xor(m,2)); m=fmaxf(m,__shfl_xor(m,4)); m=fmaxf(m,__shfl_xor(m,8)); rmax[j]=m; }
            float rcorr[8];
            #pragma unroll
            for(int j=0;j<8;j++){ float nm=fmaxf(m_i[j],rmax[j]);
                float corr=(nm==-INFINITY)?0.f:expf(m_i[j]-nm); float es=(nm==-INFINITY)?0.f:expf(S[j]-nm);
                S[j]=es; float s=es; s+=__shfl_xor(s,1);s+=__shfl_xor(s,2);s+=__shfl_xor(s,4);s+=__shfl_xor(s,8);
                l_i[j]=l_i[j]*corr+s; m_i[j]=nm; rcorr[j]=corr; }
            // P transpose via LDS
            #pragma unroll
            for(int j=0;j<8;j++) Psm[w][M_row_base+j][col_lane]=(_Float16)S[j];
            __syncwarp();
            half8_t Pf;
            #pragma unroll
            for(int j=0;j<8;j++) Pf[j]=Psm[w][col_lane][M_row_base+j];
            __syncwarp();
            // V fragments from LDS, P@V
            half8_t Vf[NT];
            #pragma unroll
            for(int nt=0;nt<NT;nt++){ int nc=nt*16+col_lane;
              #pragma unroll
              for(int j=0;j<8;j++) Vf[nt][j]=Vs[M_row_base+j][nc]; }
            #pragma unroll
            for(int nt=0;nt<NT;nt++){
              #pragma unroll
              for(int j=0;j<8;j++) Oacc[nt][j]*=rcorr[j];
              Oacc[nt]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(Pf,Vf[nt],Oacc[nt]); }
        }
        __syncthreads();  // before refilling Ks/Vs
    }
    // write O
    #pragma unroll
    for(int nt=0;nt<NT;nt++){ int col=nt*16+col_lane;
      #pragma unroll
      for(int j=0;j<8;j++){ int row=q_row0+j; if(row<M){ float il=(l_i[j]>0.f)?1.f/l_i[j]:0.f; Oh[(long)row*D+col]=__float2half(Oacc[nt][j]*il); } } }
}

template<int NW>
void launch_v2(const __half*Q,const __half*K,const __half*V,__half*O,int M,int H,int Hk){
    dim3 g((M+NW*16-1)/(NW*16), H); dim3 b(32,NW);
    flash_prefill_fwd_v2<NW><<<g,b>>>(Q,K,V,O,M,H,Hk);
}
#define CK(x) do{hipError_t e=(x);if(e)printf("ERR %s\n",hipGetErrorString(e));}while(0)

int main(){
    int H=32,Hk=8,D=128,NL=50;
    // correctness: v2 vs v1
    { int M=512; size_t qn=(size_t)M*H*D, kn=(size_t)M*Hk*D;
      std::vector<__half> hQ(qn),hK(kn),hV(kn); srand(3);
      for(auto&x:hQ)x=__float2half((rand()%200-100)/128.f);
      for(auto&x:hK)x=__float2half((rand()%200-100)/128.f);
      for(auto&x:hV)x=__float2half((rand()%200-100)/128.f);
      __half *Q,*K,*V,*O1,*O2; CK(hipMalloc(&Q,qn*2));CK(hipMalloc(&K,kn*2));CK(hipMalloc(&V,kn*2));CK(hipMalloc(&O1,qn*2));CK(hipMalloc(&O2,qn*2));
      CK(hipMemcpy(Q,hQ.data(),qn*2,hipMemcpyHostToDevice));CK(hipMemcpy(K,hK.data(),kn*2,hipMemcpyHostToDevice));CK(hipMemcpy(V,hV.data(),kn*2,hipMemcpyHostToDevice));
      launch_flash_prefill_v1(Q,K,V,O1,M,H,Hk,D,0); CK(hipDeviceSynchronize());
      launch_v2<4>(Q,K,V,O2,M,H,Hk); CK(hipDeviceSynchronize());
      std::vector<__half> a(qn),b(qn); CK(hipMemcpy(a.data(),O1,qn*2,hipMemcpyDeviceToHost));CK(hipMemcpy(b.data(),O2,qn*2,hipMemcpyDeviceToHost));
      double e=0,n=0; for(size_t i=0;i<qn;i++){float x=__half2float(a[i]),y=__half2float(b[i]);e+=(x-y)*(x-y);n+=x*x;}
      printf("v2 vs v1 (M=512): NRMSE=%.5f  %s\n",sqrt(e/n),sqrt(e/n)<0.01?"PASS":"FAIL");
      hipFree(Q);hipFree(K);hipFree(V);hipFree(O1);hipFree(O2); }
    // speed: v1 vs v2 across contexts
    printf("FA2 50L (ms): v1 vs v2<4>\n");
    int ctxs[]={512,2048,8192,32768};
    for(int ci=0;ci<4;ci++){ int M=ctxs[ci]; size_t qn=(size_t)M*H*D,kn=(size_t)M*Hk*D;
      __half *Q,*K,*V,*O; hipMalloc(&Q,qn*2);hipMalloc(&K,kn*2);hipMalloc(&V,kn*2);hipMalloc(&O,qn*2);
      hipMemset(Q,0,qn*2);hipMemset(K,0,kn*2);hipMemset(V,0,kn*2);
      for(int w=0;w<2;w++)for(int l=0;l<NL;l++)launch_flash_prefill_v1(Q,K,V,O,M,H,Hk,D,0); hipDeviceSynchronize();
      double t0=fnow();for(int l=0;l<NL;l++)launch_flash_prefill_v1(Q,K,V,O,M,H,Hk,D,0);hipDeviceSynchronize();double v1=fnow()-t0;
      for(int w=0;w<2;w++)for(int l=0;l<NL;l++)launch_v2<4>(Q,K,V,O,M,H,Hk); hipDeviceSynchronize();
      t0=fnow();for(int l=0;l<NL;l++)launch_v2<4>(Q,K,V,O,M,H,Hk);hipDeviceSynchronize();double v2=fnow()-t0;
      printf("  M=%-6d v1=%8.2f  v2=%8.2f  (%.2fx)\n",M,v1,v2,v1/v2);
      hipFree(Q);hipFree(K);hipFree(V);hipFree(O); }
    return 0;
}
