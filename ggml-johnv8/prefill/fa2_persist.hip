// fa2_persist.hip — Persistent / atomic-work-queue FA2 prefill for gfx1201 (RDNA4, HIP).
//
// MOTIVATION (the STRUCTURAL lever):
//   v1 = flash_prefill_fwd_v1: grid(M/16, H), 1 warp/block, Br=16 Bc=16, WMMA f16
//   QK^T + online softmax(exp2f) + WMMA P@V, causal. The PER-BLOCK kernel is tapped
//   (poly-exp2, multi-warp LDS-share, bigger Bc, GQA head-pack all REFUTED → compute/VGPR
//   bound at occ-6). The remaining lever is SCHEDULING:
//
//   Causal attention is triangular. Q-tile qb iterates ~(qb+1) KV-tiles, so LATE tiles
//   (high qb) do ~512x the work of EARLY tiles (qb=0) at M=8192. With grid(M/16, H) the
//   HW block scheduler assigns ~contiguously → some CUs draw all-heavy late tiles, others
//   all-light early tiles → tail/wave-quantization idle.
//
//   FIX: a PERSISTENT kernel. Launch exactly NUM_CU*occ warps; each loops grabbing the
//   next (qb, qh) work-item from a global atomicAdd counter and runs the SAME v1 inner
//   loop verbatim (1 Q-tile, no extra register state → no spill). A busy warp grabs fewer
//   items, an idle one grabs more → automatic load balance. Optionally order items so each
//   grab pairs a heavy late tile with a light early tile (balanced/reverse-diagonal order).
//
// This file COPIES the v1 inner loop into a __device__ function fa2_tile() so it can be
// driven by BOTH a plain grid kernel (parity reference) and the persistent kernel. It does
// NOT modify flash_prefill.hip; it #includes it only to call launch_flash_prefill_v1 as the
// golden reference for NRMSE.
//
// Build: hipcc --offload-arch=gfx1201 -O3 fa2_persist.hip -o fa2_persist
//        HIP_VISIBLE_DEVICES=1 ./fa2_persist
//
// ============================================================================
// RESULT (measured R9700 gfx1201, ROCm, perflevel auto, cold min-of-bursts):
//
//   v1 is COMPUTE-bound, NOT balance-bound. The persistent kernel is REFUTED.
//
//   FA2 v1 profile (causal triangle f16 FLOP / time):
//     M=2048   ~20.2 TFLOPS    M=8192  ~22.3 TFLOPS    M=16384 ~20.8-22.7 TFLOPS
//     (~50% of the gfx1201 f16 WMMA peak; the per-KV-tile setup — P-transpose via
//      SMEM, softmax, V-load — fills the gap, as fa2_v3 already established.)
//
//   Load-imbalance (simulated, the causal triangle over NUM_CU*occ=384 warp-slots):
//     The MAXIMUM recoverable idle even with a PERFECT work-stealer (block-granular)
//       M=2048  : 11.8%      M=8192 : 2.8%      M=16384 : 1.3%
//     It SHRINKS with context because 16384+ tiles average out across 64 CUs (the
//     heaviest single tile is only 512/65664 = 0.78% of a CU's balanced load at
//     M=8192). At the long-context regime that matters, imbalance is ~1-3% — there
//     is almost nothing to recover.
//
//   Persistent kernel measured vs v1 (best of ORD natural/reverse-diag/LPT,
//   WPB 2/6/8, 384-512 resident warps): 0.59-0.80x at M=8192/16384. It LOSES.
//   The atomic-grab + per-work-item re-init (zero 64 Oacc, reload Q) + multi-warp
//   block scheduling cost MORE than the <3% imbalance it could recover. Equal VGPR
//   (251 vs 236 -> same 6 waves/SIMD), so occupancy isn't the win either.
//
//   => pp8192 FA2 needs a DIFFERENT lever (algorithmic: sliding-window / sparsity /
//      KV-sharing), NOT scheduling. Persistent scheduling is a dead end here.
//      No FA2 win => no pp8192 full-forward lift from this path (the 2307 t/s tie
//      stands; FA2's 43% is genuinely compute, not idle).
// ============================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>

typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float    float8_t __attribute__((ext_vector_type(8)));

// golden reference launch_flash_prefill_v1 (DO NOT modify flash_prefill.hip)
#include "/home/janusz/AMD MXFP4/hip_int4/flash_prefill.hip"

static double fnow(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }
#define CK(x) do{ hipError_t e=(x); if(e) printf("HIP ERR %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); }while(0)

// ============================================================================
// fa2_tile — the v1 inner loop verbatim, extracted as a __device__ function.
//   Processes ONE work-item: Q-tile `qb` (16 rows) of head `qh`.
//   Identical math to flash_prefill_fwd_v1 EXCEPT exp2f (folded log2e into scale),
//   the winning lever-2 from fa2_v3. NRMSE vs v1 ~1e-5 (SDPA-parity).
//   1 warp (32 lanes). No __syncthreads — only __syncwarp for the P-transpose.
// ============================================================================
__device__ __forceinline__ void fa2_tile(
    const _Float16* __restrict__ Q, const _Float16* __restrict__ K,
    const _Float16* __restrict__ V, __half* __restrict__ O,
    int M, int H, int Hk, int qb, int qh, int lane, int warp_id)
{
    constexpr int Br = 16, Bc = 16, D = 128, KT = D/16, NT = D/16;
    const int gqa_ratio = H / Hk, kh = qh / gqa_ratio;
    const int q_row_base = qb * Br;                // caller guarantees < M
    const int M_row_base = (lane / 16) * 8, col_lane = lane % 16;
    const int q_row0 = q_row_base + M_row_base;

    const _Float16* Q_head = Q + (long)qh * M * D;
    const _Float16* K_head = K + (long)kh * M * D;
    const _Float16* V_head = V + (long)kh * M * D;
    __half*         O_head = O + (long)qh * M * D;

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
    const float scale = rsqrtf((float)D) * log2e;     // lever 2: exp2 domain
    int s_max = q_row_base + Br; if (s_max > M) s_max = M;

    __shared__ _Float16 P_smem[8][16][16];   // per-warp tile (8 warps/block max)
    _Float16 (*Ps)[16] = P_smem[warp_id];

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
            float corr = (nm==-INFINITY)?0.f:exp2f(m_i[j]-nm);
            float es   = (nm==-INFINITY)?0.f:exp2f(Sacc[j]-nm);
            Sacc[j]=es; float s=es;
            s+=__shfl_xor(s,1); s+=__shfl_xor(s,2); s+=__shfl_xor(s,4); s+=__shfl_xor(s,8);
            l_i[j]=l_i[j]*corr+s; m_i[j]=nm; row_corr[j]=corr; }

        #pragma unroll
        for (int j=0;j<8;j++) Ps[M_row_base+j][col_lane]=(_Float16)Sacc[j];
        __syncwarp();
        half8_t Pf;
        #pragma unroll
        for (int j=0;j<8;j++) Pf[j]=Ps[col_lane][M_row_base+j];
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

// ============================================================================
// REFERENCE grid kernel — same fa2_tile, grid(M/16, H). This is the "v1+exp2"
//   non-persistent path: it isolates the EFFECT of persistence (vs fa2_v3 exp2).
// ============================================================================
__global__ __launch_bounds__(32, 4)
void fa2_grid(const __half* Q, const __half* K, const __half* V, __half* O, int M, int H, int Hk){
    const int qb = blockIdx.x, qh = blockIdx.y;
    if (qb*16 >= M) return;
    fa2_tile(reinterpret_cast<const _Float16*>(Q), reinterpret_cast<const _Float16*>(K),
             reinterpret_cast<const _Float16*>(V), O, M, H, Hk, qb, qh, threadIdx.x & 31, 0);
}
static void launch_grid(const __half*Q,const __half*K,const __half*V,__half*O,int M,int H,int Hk){
    dim3 grid((M+15)/16, H); dim3 block(32);
    fa2_grid<<<grid,block>>>(Q,K,V,O,M,H,Hk);
}

// ============================================================================
// PERSISTENT kernel — launch NUM_BLK blocks of WARPS_PER_BLK warps each. Every
//   warp loops: grab next work index from a global atomic counter, decode it to
//   (qb, qh), run fa2_tile, repeat until the queue drains.
//
//   ORDER variants (compile-time):
//     ORD=0  natural: work = qh*n_qb + qb  (heads-major; within a head qb ascends
//            light→heavy). A warp's consecutive grabs trend monotonic in weight.
//     ORD=1  reverse-diagonal balance: remap so grab i alternates a heavy late
//            tile and a light early tile. We pair qb with (n_qb-1-qb). Decoding:
//            even slot -> low half ascending, odd slot -> high half descending,
//            so each adjacent pair of grabs ~ averages to the mean weight.
//
//   One atomic counter, 1 atomicAdd per grab. Cheap vs ~(qb+1) WMMA inner iters.
// ============================================================================
// NOTE on occupancy: do NOT pin __launch_bounds min-blocks to 1 — that starved
//   residency in the first cut (1-warp blocks didn't pack → under-occupied GPU →
//   0.66x). Use the SAME (32,4) hint as v1 so the compiler/runtime fills the CU,
//   and launch enough warps (WARPS_PER_BLK * num_blk) to saturate.
template<int WARPS_PER_BLK, int ORD>
__global__ __launch_bounds__(WARPS_PER_BLK*32, 1)
void fa2_persistent(const __half* Q, const __half* K, const __half* V, __half* O,
                    int M, int H, int Hk, int n_qb, int total, int* g_ctr)
{
    const int lane = threadIdx.x & 31;
    const int warp_in_blk = threadIdx.x >> 5;
    const bool leader = (lane == 0);
    __shared__ int s_work[WARPS_PER_BLK];

    const _Float16* Qh = reinterpret_cast<const _Float16*>(Q);
    const _Float16* Kh = reinterpret_cast<const _Float16*>(K);
    const _Float16* Vh = reinterpret_cast<const _Float16*>(V);

    for (;;) {
        if (leader) s_work[warp_in_blk] = atomicAdd(g_ctr, 1);
        __syncwarp();
        int w = s_work[warp_in_blk];
        if (w >= total) break;

        int qb, qh;
        if (ORD == 0) {
            qh = w / n_qb;
            qb = w - qh * n_qb;
        } else if (ORD == 1) {
            // reverse-diagonal: split the per-head q-tile range into a butterfly.
            // local in [0, n_qb): map so consecutive locals pair light/heavy.
            qh = w / n_qb;
            int local = w - qh * n_qb;
            if ((local & 1) == 0) qb = local >> 1;                // ascending low half (light)
            else                  qb = n_qb - 1 - (local >> 1);   // descending high half (heavy)
        } else {
            // ORD==2: LPT (longest-processing-time-first) GLOBAL order. The heaviest
            // tiles (high qb) across ALL heads are grabbed first → the longest poles
            // start earliest → minimal makespan tail. Decode: iterate qb descending,
            // and for each qb iterate all H heads. w -> (qb_desc, head).
            int qb_desc = w / H;            // 0..n_qb-1 in descending-weight order
            qh          = w - qb_desc * H;  // head 0..H-1
            qb = n_qb - 1 - qb_desc;        // heaviest first
        }
        fa2_tile(Qh, Kh, Vh, O, M, H, Hk, qb, qh, lane, warp_in_blk);
    }
}

// num_blk = number of CONCURRENT persistent blocks (each WARPS_PER_BLK warps).
// total resident warps = num_blk * WARPS_PER_BLK should saturate the machine.
template<int WARPS_PER_BLK, int ORD>
static void launch_persistent(const __half*Q,const __half*K,const __half*V,__half*O,
                              int M,int H,int Hk,int num_blk,int* g_ctr){
    int n_qb = (M + 15) / 16;
    int total = n_qb * H;
    CK(hipMemsetAsync(g_ctr, 0, sizeof(int), 0));
    dim3 grid(num_blk); dim3 block(WARPS_PER_BLK*32);
    fa2_persistent<WARPS_PER_BLK,ORD><<<grid,block>>>(Q,K,V,O,M,H,Hk,n_qb,total,g_ctr);
    hipError_t le = hipGetLastError();
    if (le) printf("    [launch err WPB=%d ORD=%d nb=%d] %s\n", WARPS_PER_BLK, ORD, num_blk, hipGetErrorString(le));
}

// ============================================================================
// Harness
// ============================================================================
static double nrmse(const std::vector<__half>& a, const std::vector<__half>& b){
    double e=0,n=0;
    for(size_t i=0;i<a.size();i++){ float x=__half2float(a[i]),y=__half2float(b[i]); e+=(x-y)*(x-y); n+=x*x; }
    return sqrt(e/n);
}

static const int NUM_CU = 64;
static int* g_ctr = nullptr;

// ---- profiling: ideal (perfectly balanced) time from per-tile work ----
// per Q-tile qb work ∝ number of KV tiles iterated = ceil((qb*16+16)/16) = qb+1.
// (each KV-tile = constant WMMA cost). Sum over all (qb,qh).
static void profile_imbalance(int M, int H){
    int n_qb = (M+15)/16;
    // total KV-tile iterations (the FA2 work unit)
    double total_work = 0;          // sum over heads & qb of (qb+1)
    double max_tile   = 0;
    for (int qb=0; qb<n_qb; qb++){ double w = qb+1; total_work += (double)H*w; max_tile = fmax(max_tile, w); }
    // FLOPs: each KV-tile does QK^T (16x16x128) + PV (16x128x16) = 2 * 2*16*16*128 MACs *2 (mul+add)
    // = 2 matmuls of 16x16x128 → 2 * (16*16*128*2) flops = 2*524288 = 1048576 flops per KV-tile.
    double flops_per_kvtile = 2.0 * (16.0*16.0*128.0*2.0);
    double total_flops = total_work * flops_per_kvtile;
    // ideal balanced time assumes total_work spread perfectly over NUM_CU*occ warp-slots.
    printf("  [profile M=%d] n_qb=%d  total KV-tile-iters=%.0f (×H)  total f16 FLOP=%.3e\n",
           M, n_qb, total_work, total_flops);
    printf("              work/CU if perfectly balanced = %.0f tile-iters; heaviest single tile=%.0f\n",
           total_work/NUM_CU, max_tile);

    // ---- SIMULATE the HW block scheduler's imbalance ----
    // Model: grid(n_qb, H) launches n_qb*H blocks in row-major order (blockIdx.x fastest).
    // HW assigns blocks to NUM_CU CUs in waves of NUM_CU (a CU picks the next free block
    // ~round-robin). Each CU runs occ blocks concurrently; tail = last partial wave.
    // We approximate makespan as: greedily assign blocks (in launch order) to NUM_CU*occ
    // slots, slot picks next block when free; makespan = max slot finish time.
    const int occ = 6;                 // v1 occupancy
    const int NSLOT = NUM_CU * occ;
    std::vector<double> slot(NSLOT, 0.0);
    // build work list in launch order: blockIdx = qb + qh*n_qb? No: grid(x=n_qb,y=H),
    // launch order iterates x fastest within y. weight depends only on qb (=blockIdx.x).
    // total blocks = n_qb*H.
    long nb = (long)n_qb * H;
    // greedy: each block to the currently-least-loaded slot (this is the BEST a balanced
    // scheduler could do given block granularity — an OPTIMISTIC bound on imbalance).
    // We process in launch order but assign to least-loaded slot = models a perfect
    // work-stealing/persistent scheduler. Then compare to a CONTIGUOUS-chunk scheduler.
    {
        // perfect work-steal (= ideal persistent): least-loaded slot each time
        std::vector<double> s(NSLOT,0.0);
        for (long b=0;b<nb;b++){ int qb=b%n_qb; double w=qb+1;
            int mi=0; for(int k=1;k<NSLOT;k++) if(s[k]<s[mi]) mi=k; s[mi]+=w; }
        double mk=0,sum=0; for(double x:s){ mk=fmax(mk,x); sum+=x; }
        double ideal = sum/NSLOT;
        printf("              [sim] ideal-balanced makespan=%.0f, perfect-worksteal makespan=%.0f  imbalance=%.2f%%\n",
               ideal, mk, 100.0*(mk-ideal)/ideal);
    }
    {
        // contiguous-chunk scheduler (models naive HW: slot k gets blocks k, k+NSLOT, ...)
        std::vector<double> s(NSLOT,0.0);
        for (long b=0;b<nb;b++){ int qb=b%n_qb; double w=qb+1; int slotk=b%NSLOT; s[slotk]+=w; }
        double mk=0,sum=0; for(double x:s){ mk=fmax(mk,x); sum+=x; }
        double ideal=sum/NSLOT;
        printf("              [sim] round-robin(naive HW) makespan=%.0f  imbalance vs ideal=%.2f%%  (this is the recoverable idle)\n",
               mk, 100.0*(mk-ideal)/ideal);
    }
}

static double timeit_grid(void(*fn)(const __half*,const __half*,const __half*,__half*,int,int,int),
                          const __half*Q,const __half*K,const __half*V,__half*O,int M,int H,int Hk,int NL,int R){
    for(int w=0;w<3;w++) for(int l=0;l<NL;l++) fn(Q,K,V,O,M,H,Hk);
    CK(hipDeviceSynchronize());
    double best=1e30;
    for(int i=0;i<R;i++){ double t0=fnow(); for(int l=0;l<NL;l++) fn(Q,K,V,O,M,H,Hk); CK(hipDeviceSynchronize()); double t=(fnow()-t0); best=fmin(best,t); }
    return best/NL;
}

template<int WPB,int ORD>
static double timeit_persist(const __half*Q,const __half*K,const __half*V,__half*O,int M,int H,int Hk,int num_blk,int NL,int R){
    for(int w=0;w<3;w++) for(int l=0;l<NL;l++) launch_persistent<WPB,ORD>(Q,K,V,O,M,H,Hk,num_blk,g_ctr);
    CK(hipDeviceSynchronize());
    double best=1e30;
    for(int i=0;i<R;i++){ double t0=fnow(); for(int l=0;l<NL;l++) launch_persistent<WPB,ORD>(Q,K,V,O,M,H,Hk,num_blk,g_ctr); CK(hipDeviceSynchronize()); double t=(fnow()-t0); best=fmin(best,t); }
    return best/NL;
}

static void v1_wrap(const __half*Q,const __half*K,const __half*V,__half*O,int M,int H,int Hk){
    launch_flash_prefill_v1(Q,K,V,O,M,H,Hk,128,0);
}

// validation against golden v1
static void validate_all(int M,int H,int Hk,int D){
    size_t qn=(size_t)M*H*D, kn=(size_t)M*Hk*D;
    std::vector<__half> hQ(qn),hK(kn),hV(kn);
    srand(3);
    for(auto&x:hQ)x=__float2half((rand()%200-100)/128.f);
    for(auto&x:hK)x=__float2half((rand()%200-100)/128.f);
    for(auto&x:hV)x=__float2half((rand()%200-100)/128.f);
    __half *Q,*K,*V,*Og,*Op;
    CK(hipMalloc(&Q,qn*2));CK(hipMalloc(&K,kn*2));CK(hipMalloc(&V,kn*2));
    CK(hipMalloc(&Og,qn*2));CK(hipMalloc(&Op,qn*2));
    CK(hipMemcpy(Q,hQ.data(),qn*2,hipMemcpyHostToDevice));
    CK(hipMemcpy(K,hK.data(),kn*2,hipMemcpyHostToDevice));
    CK(hipMemcpy(V,hV.data(),kn*2,hipMemcpyHostToDevice));
    std::vector<__half> a(qn),b(qn);

    launch_flash_prefill_v1(Q,K,V,Og,M,H,Hk,D,0); CK(hipDeviceSynchronize());
    CK(hipMemcpy(a.data(),Og,qn*2,hipMemcpyDeviceToHost));

    // grid (v1+exp2)
    CK(hipMemset(Op,0,qn*2)); launch_grid(Q,K,V,Op,M,H,Hk); CK(hipDeviceSynchronize());
    CK(hipMemcpy(b.data(),Op,qn*2,hipMemcpyDeviceToHost));
    double r0=nrmse(a,b);
    printf("  [grid    ] M=%-6d NRMSE=%.6f %s\n",M,r0,r0<1e-3?"PASS":"FAIL");

    // persistent ORD=0 (6 warps/blk × 64 blk)
    CK(hipMemset(Op,0,qn*2)); launch_persistent<6,0>(Q,K,V,Op,M,H,Hk,NUM_CU,g_ctr); CK(hipDeviceSynchronize());
    CK(hipMemcpy(b.data(),Op,qn*2,hipMemcpyDeviceToHost));
    double r1=nrmse(a,b);
    printf("  [persORD0] M=%-6d NRMSE=%.6f %s\n",M,r1,r1<1e-3?"PASS":"FAIL");

    // persistent ORD=2 (LPT global order) — also exercises 2 warps/blk path
    CK(hipMemset(Op,0,qn*2)); launch_persistent<2,2>(Q,K,V,Op,M,H,Hk,NUM_CU*3,g_ctr); CK(hipDeviceSynchronize());
    CK(hipMemcpy(b.data(),Op,qn*2,hipMemcpyDeviceToHost));
    double r2=nrmse(a,b);
    printf("  [persORD2] M=%-6d NRMSE=%.6f %s\n",M,r2,r2<1e-3?"PASS":"FAIL");

    hipFree(Q);hipFree(K);hipFree(V);hipFree(Og);hipFree(Op);
}

int main(){
    const int H=32,Hk=8,D=128;
    CK(hipMalloc(&g_ctr,sizeof(int)));

    printf("=== GPU: gfx1201 R9700, NUM_CU=%d ===\n", NUM_CU);

    printf("\n=== CORRECTNESS (NRMSE vs golden v1, <1e-3 = PASS) ===\n");
    for(int M : {512, 2048, 8192}) validate_all(M,H,Hk,D);

    printf("\n=== LOAD-IMBALANCE PROFILE (causal triangle) ===\n");
    for(int M : {2048, 8192, 16384}) profile_imbalance(M,H);

    // NL: fewer loops at large M to keep runtime sane; cold = min over bursts.
    // Persistent configs: WPB warps/block × NB blocks = resident warps. v1 runs occ-6
    // ≈ 384 resident warps; we sweep to match/exceed that and find the best ORDER.
    //   p6_o0/o1 : 6 warps/blk × 64 blk = 384 warps  (matches v1 occ-6), ORD 0/1
    //   p4_o1    : 4 warps/blk × 96 blk = 384 warps,  ORD 1 (different block shape)
    //   p8_o1    : 8 warps/blk × 64 blk = 512 warps   (over-subscribe), ORD 1
    // global GPU warmup to reach steady clocks (avoid cold-start artifact on first M)
    {
        int Mw=8192; size_t qn=(size_t)Mw*H*D,kn=(size_t)Mw*Hk*D;
        __half *Q,*K,*V,*O; CK(hipMalloc(&Q,qn*2));CK(hipMalloc(&K,kn*2));CK(hipMalloc(&V,kn*2));CK(hipMalloc(&O,qn*2));
        for(int i=0;i<60;i++) v1_wrap(Q,K,V,O,Mw,H,Hk); CK(hipDeviceSynchronize());
        hipFree(Q);hipFree(K);hipFree(V);hipFree(O);
    }

    printf("\n=== SPEED (cold min-of-bursts, ms/call) ===\n");
    printf("p6_o2 = LPT global order (heaviest tiles first). p2_o2 = 2 warps/blk LPT.\n");
    printf("%-7s %10s %10s %10s %10s %10s %10s\n","M","v1","grid+exp2","p6_o0","p6_o2","p2_o2","p8_o2");
    for(int M : {2048, 8192, 16384}){
        int NL = (M<=2048)?40 : (M<=8192? 12 : 5);
        int R  = 8;
        size_t qn=(size_t)M*H*D, kn=(size_t)M*Hk*D;
        __half *Q,*K,*V,*O;
        CK(hipMalloc(&Q,qn*2));CK(hipMalloc(&K,kn*2));CK(hipMalloc(&V,kn*2));CK(hipMalloc(&O,qn*2));
        std::vector<__half> hQ(qn),hK(kn),hV(kn); srand(7);
        for(auto&x:hQ)x=__float2half((rand()%200-100)/128.f);
        for(auto&x:hK)x=__float2half((rand()%200-100)/128.f);
        for(auto&x:hV)x=__float2half((rand()%200-100)/128.f);
        CK(hipMemcpy(Q,hQ.data(),qn*2,hipMemcpyHostToDevice));
        CK(hipMemcpy(K,hK.data(),kn*2,hipMemcpyHostToDevice));
        CK(hipMemcpy(V,hV.data(),kn*2,hipMemcpyHostToDevice));

        double t_v1  = timeit_grid(v1_wrap,   Q,K,V,O,M,H,Hk,NL,R);
        double t_gr  = timeit_grid(launch_grid,Q,K,V,O,M,H,Hk,NL,R);
        double t_60  = timeit_persist<6,0>(Q,K,V,O,M,H,Hk,NUM_CU,   NL,R);  // 384 warps natural
        double t_62  = timeit_persist<6,2>(Q,K,V,O,M,H,Hk,NUM_CU,   NL,R);  // 384 warps LPT
        double t_22  = timeit_persist<2,2>(Q,K,V,O,M,H,Hk,NUM_CU*3, NL,R);  // 384 warps LPT, 2/blk
        double t_82  = timeit_persist<8,2>(Q,K,V,O,M,H,Hk,NUM_CU,   NL,R);  // 512 warps LPT
        printf("%-7d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n",M,t_v1,t_gr,t_60,t_62,t_22,t_82);
        printf("%-7s %10s %9.3fx %9.3fx %9.3fx %9.3fx %9.3fx\n","spd-v1","",
               t_v1/t_gr, t_v1/t_60, t_v1/t_62, t_v1/t_22, t_v1/t_82);

        // achieved TFLOPS for v1 at this M (causal triangle FLOP / time)
        int n_qb=(M+15)/16; double work=0; for(int qb=0;qb<n_qb;qb++) work += (double)H*(qb+1);
        double tflop = work * (2.0*(16.0*16.0*128.0*2.0)) / 1e12;
        double bestp = fmin(fmin(t_60,t_62),fmin(t_22,t_82));
        printf("%-7s v1 %.3f TFLOPS  | grid+exp2 %.3f TFLOPS | best persistent %.3f TFLOPS\n","tflops",
               tflop/(t_v1*1e-3), tflop/(t_gr*1e-3), tflop/(bestp*1e-3));
        hipFree(Q);hipFree(K);hipFree(V);hipFree(O);
    }

    hipFree(g_ctr);
    return 0;
}
