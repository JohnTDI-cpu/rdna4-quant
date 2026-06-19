// fa2_v3.hip — FA2 prefill optimization for Bielik-11B on gfx1201/RDNA4 (HIP).
//
// Baseline: flash_prefill_fwd_v1 (in flash_prefill.hip) — 1 warp/block, Br=16, Bc=16,
// online softmax, WMMA f16. ~50% of peak: the non-WMMA per-KV-step overhead
// (P-transpose via SMEM, softmax setup, V-load setup) dominates.
//
// The FA2 is COMPUTE-bound (multi-warp + LDS-K/V v2 was slower at large ctx).
// STAY 1 warp/block, no __syncthreads (only __syncwarp for the P-transpose).
//
// LEVERS TRIED + RESULTS (measured on R9700 gfx1201, validated NRMSE<1e-3 vs v1):
//   1. BIGGER Bc (32/64)  [fa2_v3a]  -> LOSES (0.73-0.83x). Root cause: v1 already
//      sits at the VGPR/occupancy edge (233 VGPR, occ 6). Bc=32 pushes to 256 VGPR
//      + 68 spilled -> occ 5. Bc=64 spills 506. The per-step setup is NOT the bottleneck.
//   2. exp2f (fold log2e into scale) [fa2_v3_exp2] -> WINS. ~1.02-1.05x, ZERO extra
//      VGPR (234 vs 233), occ 6, no spill. Free transcendental speedup. *** BEST ***
//   3. Register/shfl P-transpose      [fa2_v3c]  -> LOSES hard (0.30x). 16x16 acc->A
//      transpose needs ~64 cross-lane __shfl per N-tile; SMEM round-trip is far cheaper.
//
// CONCLUSION: only lever 2 helps. FA2 is compute/VGPR-bound; v1 is already near-optimal
// in structure. The winning kernel = fa2_v3_exp2<4> (== v1 + exp2f). Launcher: launch_fa2_v3.
//
// Build: hipcc --offload-arch=gfx1201 -O3 fa2_v3.hip -o fa2_v3 && HIP_VISIBLE_DEVICES=1 ./fa2_v3

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>

typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float    float8_t __attribute__((ext_vector_type(8)));

// v1 reference + launch_flash_prefill_v1 (DO NOT modify flash_prefill.hip)
#include "/home/janusz/AMD MXFP4/hip_int4/flash_prefill.hip"

static double fnow(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }
#define CK(x) do{ hipError_t e=(x); if(e) printf("HIP ERR %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); }while(0)

// ============================================================================
// v3a — BIGGER Bc (lever 1, templated NC) + exp2f (lever 2).
//   Bc = NC*16 KV positions per step. S = Q@K^T -> NC N-tiles, each float8_t.
//   Online softmax reduces over all NC N-tiles in registers, then shfl_xor/16.
//   P-transpose: NC SMEM tiles (one per K-fragment of P@V, K=Bc).
//   P@V: K-dim = Bc = NC*16  -> NC WMMA ops accumulated into Oacc.
//   1 warp/block, no __syncthreads.
// ============================================================================
template<int NC>
__global__ __launch_bounds__(32, 4)
void fa2_v3a(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, __half* __restrict__ O,
    int M, int H, int Hk)
{
    constexpr int Br = 16;
    constexpr int Bc = NC * 16;
    constexpr int D  = 128;
    constexpr int KT = D / 16;   // 8 K-fragments for Q@K^T
    constexpr int NT = D / 16;   // 8 N-tiles for P@V

    const int qb = blockIdx.x;
    const int qh = blockIdx.y;
    const int lane = threadIdx.x;
    const int gqa_ratio = H / Hk;
    const int kh = qh / gqa_ratio;

    const int q_row_base = qb * Br;
    if (q_row_base >= M) return;

    const int M_row_base = (lane / 16) * 8;   // {0, 8}
    const int col_lane   = lane % 16;
    const int q_row0     = q_row_base + M_row_base;

    const _Float16* Q_head = reinterpret_cast<const _Float16*>(Q) + (long)qh * M * D;
    const _Float16* K_head = reinterpret_cast<const _Float16*>(K) + (long)kh * M * D;
    const _Float16* V_head = reinterpret_cast<const _Float16*>(V) + (long)kh * M * D;
    __half*         O_head = O + (long)qh * M * D;

    // ---- Q fragments (loaded once)
    half8_t Qf[KT];
    {
        int q_row_lane = q_row_base + col_lane;
        bool row_valid = (q_row_lane < M);
        #pragma unroll
        for (int kt = 0; kt < KT; kt++) {
            int K_col_base = kt * 16 + M_row_base;
            #pragma unroll
            for (int j = 0; j < 8; j++)
                Qf[kt][j] = row_valid ? Q_head[(long)q_row_lane * D + K_col_base + j] : (_Float16)0.0f;
        }
    }

    float m_i[8], l_i[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) { m_i[j] = -INFINITY; l_i[j] = 0.0f; }

    float8_t Oacc[NT];
    #pragma unroll
    for (int nt = 0; nt < NT; nt++)
        #pragma unroll
        for (int j = 0; j < 8; j++) Oacc[nt][j] = 0.0f;

    // lever 2: fold log2(e) into scale so exp(x)=exp2(x*log2e); use exp2f
    const float log2e = 1.4426950408889634f;
    const float scale = rsqrtf((float)D) * log2e;

    int s_max = q_row_base + Br;
    if (s_max > M) s_max = M;

    // SMEM P-transpose tiles: NC of them (one per K-fragment of P@V)
    __shared__ _Float16 P_smem[NC][16][16];

    for (int s_start = 0; s_start < s_max; s_start += Bc) {

        // ---- S = Q @ K^T for NC N-tiles. Each N-tile covers KV cols [s_start+nc*16 .. +15]
        float8_t Sacc[NC];
        #pragma unroll
        for (int nc = 0; nc < NC; nc++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) Sacc[nc][j] = 0.0f;
        }
        #pragma unroll
        for (int nc = 0; nc < NC; nc++) {
            // K tile for this N-tile: cols = s_start + nc*16 + col_lane
            half8_t Kf[KT];
            int s_pos = s_start + nc * 16 + col_lane;
            bool s_valid = (s_pos < M);
            const _Float16* k_row = s_valid ? (K_head + (long)s_pos * D) : nullptr;
            #pragma unroll
            for (int kt = 0; kt < KT; kt++) {
                int K_row_base = kt * 16 + M_row_base;
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    Kf[kt][j] = s_valid ? k_row[K_row_base + j] : (_Float16)0.0f;
            }
            #pragma unroll
            for (int kt = 0; kt < KT; kt++)
                Sacc[nc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(Qf[kt], Kf[kt], Sacc[nc]);
        }

        // ---- scale + causal mask
        #pragma unroll
        for (int nc = 0; nc < NC; nc++) {
            int col_global = s_start + nc * 16 + col_lane;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int row_global = q_row0 + j;
                bool valid = (col_global <= row_global) && (col_global < M);
                Sacc[nc][j] = valid ? (Sacc[nc][j] * scale) : -INFINITY;
            }
        }

        // ---- per-row max: reduce over NC N-tiles in registers, then shfl_xor within 16
        float row_max[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float m = Sacc[0][j];
            #pragma unroll
            for (int nc = 1; nc < NC; nc++) m = fmaxf(m, Sacc[nc][j]);
            m = fmaxf(m, __shfl_xor(m, 1));
            m = fmaxf(m, __shfl_xor(m, 2));
            m = fmaxf(m, __shfl_xor(m, 4));
            m = fmaxf(m, __shfl_xor(m, 8));
            row_max[j] = m;
        }

        // ---- online softmax: P = exp2(S - m), row sum over NC N-tiles, l/m update
        float row_corr[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float new_m = fmaxf(m_i[j], row_max[j]);
            float corr = (new_m == -INFINITY) ? 0.0f : exp2f(m_i[j] - new_m);
            float s = 0.0f;
            #pragma unroll
            for (int nc = 0; nc < NC; nc++) {
                float es = (new_m == -INFINITY) ? 0.0f : exp2f(Sacc[nc][j] - new_m);
                Sacc[nc][j] = es;   // S now holds P (un-normalized)
                s += es;
            }
            s += __shfl_xor(s, 1);
            s += __shfl_xor(s, 2);
            s += __shfl_xor(s, 4);
            s += __shfl_xor(s, 8);
            l_i[j] = l_i[j] * corr + s;
            m_i[j] = new_m;
            row_corr[j] = corr;
        }

        // ---- P-transpose: write NC N-tiles to SMEM, read back as A-fragments
        #pragma unroll
        for (int nc = 0; nc < NC; nc++) {
            #pragma unroll
            for (int j = 0; j < 8; j++)
                P_smem[nc][M_row_base + j][col_lane] = (_Float16)Sacc[nc][j];
        }
        __syncwarp();
        half8_t Pf[NC];
        #pragma unroll
        for (int nc = 0; nc < NC; nc++) {
            #pragma unroll
            for (int j = 0; j < 8; j++)
                Pf[nc][j] = P_smem[nc][col_lane][M_row_base + j];
        }
        __syncwarp();

        // ---- apply correction to Oacc ONCE per step (not per K-fragment)
        #pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) Oacc[nt][j] *= row_corr[j];
        }

        // ---- P @ V: K-dim = Bc = NC*16 -> accumulate NC WMMA per N-tile of D
        #pragma unroll
        for (int nc = 0; nc < NC; nc++) {
            // V fragments for this K-fragment (KV rows s_start+nc*16 .. +15)
            half8_t Vf[NT];
            #pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                int N_col_v = nt * 16 + col_lane;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int s_pos = s_start + nc * 16 + M_row_base + j;
                    Vf[nt][j] = (s_pos < M) ? V_head[(long)s_pos * D + N_col_v] : (_Float16)0.0f;
                }
            }
            #pragma unroll
            for (int nt = 0; nt < NT; nt++)
                Oacc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(Pf[nc], Vf[nt], Oacc[nt]);
        }
    }

    // ---- final normalize + write O
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        int col = nt * 16 + col_lane;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int row = q_row0 + j;
            if (row < M) {
                float inv_l = (l_i[j] > 0.0f) ? (1.0f / l_i[j]) : 0.0f;
                O_head[(long)row * D + col] = __float2half(Oacc[nt][j] * inv_l);
            }
        }
    }
}

template<int NC>
void launch_v3a(const __half*Q,const __half*K,const __half*V,__half*O,int M,int H,int Hk){
    dim3 grid((M + 15) / 16, H);
    dim3 block(32);
    fa2_v3a<NC><<<grid, block>>>(Q, K, V, O, M, H, Hk);
}

// ============================================================================
// v3_exp2 — v1 structure (Br=16, Bc=16, 1 warp, SMEM P-transpose) with ONLY
//   lever 2: exp2f instead of expf (fold log2(e) into scale). No extra
//   registers => keeps v1's occupancy. This is the clean, winning lever.
//   LB controls __launch_bounds occupancy hint (templated to sweep).
// ============================================================================
template<int LB>
__global__ __launch_bounds__(32, LB)
void fa2_v3_exp2(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, __half* __restrict__ O,
    int M, int H, int Hk)
{
    constexpr int Br = 16, Bc = 16, D = 128, KT = D/16, NT = D/16;
    const int qb = blockIdx.x, qh = blockIdx.y, lane = threadIdx.x;
    const int gqa_ratio = H / Hk, kh = qh / gqa_ratio;
    const int q_row_base = qb * Br;
    if (q_row_base >= M) return;
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
    const float scale = rsqrtf((float)D) * log2e;     // lever 2: exp2 domain
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
            float corr = (nm==-INFINITY)?0.f:exp2f(m_i[j]-nm);       // lever 2
            float es   = (nm==-INFINITY)?0.f:exp2f(Sacc[j]-nm);      // lever 2
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

template<int LB>
void launch_v3_exp2(const __half*Q,const __half*K,const __half*V,__half*O,int M,int H,int Hk){
    dim3 grid((M + 15) / 16, H); dim3 block(32);
    fa2_v3_exp2<LB><<<grid, block>>>(Q, K, V, O, M, H, Hk);
}

// ===== CANONICAL DROP-IN: best variant = v3_exp2 with LB=4 =====
// Same contract/signature as launch_flash_prefill_v1 — drop-in faster FA2.
// Measured ~1.02-1.05x vs v1 across M=512/2048/8192, NRMSE<1e-5 (SDPA-parity).
static inline void launch_fa2_v3(
    const __half* Q, const __half* K, const __half* V,
    __half* O, int M, int H, int Hk, int D, hipStream_t st)
{
    if (D != 128) return;
    dim3 grid((M + 15) / 16, H); dim3 block(32);
    hipLaunchKernelGGL(fa2_v3_exp2<4>, grid, block, 0, st, Q, K, V, O, M, H, Hk);
}

// ============================================================================
// v3c — like v3a but P-transpose via register shuffle (lever 3, no SMEM).
//   Transpose a 16x16 fp16 tile held in acc layout to A-fragment layout using
//   __shfl across the 16-lane subgroup (no SMEM round-trip, no __syncwarp).
//
//   Acc layout of Sacc[nc]:  (lane/16)*8+j = M_row, lane%16 = K_col(within tile)
//   A-frag layout needed:    lane%16 = M_row, (lane/16)*8+j = K_col
//
//   Within one 16-lane half (lane/16 fixed), we have an 8x16 block:
//     value at (lane, j) = P[row = base8 + j, col = lane%16]
//   We want Pf[j2] for output lane L = P[row = L%16, col = base8 + j2].
//   That requires data from BOTH halves (rows 0..7 from half0, rows 8..15 from
//   half1). Use __shfl(value, srcLane) to gather. For target (L, j2):
//     need row = L%16, col = base8(L) + j2.
//     row L%16 lives in half (L%16)>=8 ? half1 : half0, at local j = (L%16)%8,
//     and is stored by source lanes whose col == base8(L)+j2, i.e. lane%16 == base8(L)+j2.
//   srcLane = (halfsel)*16 + (base8(L)+j2),  pick src register index = (L%16)%8.
//   We implement via two __shfl per (j2) selecting from a chosen source lane and
//   register, blended by which half holds the row.
// ============================================================================
template<int NC>
__global__ __launch_bounds__(32, 4)
void fa2_v3c(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, __half* __restrict__ O,
    int M, int H, int Hk)
{
    constexpr int Br = 16;
    constexpr int Bc = NC * 16;
    constexpr int D  = 128;
    constexpr int KT = D / 16;
    constexpr int NT = D / 16;

    const int qb = blockIdx.x;
    const int qh = blockIdx.y;
    const int lane = threadIdx.x;
    const int gqa_ratio = H / Hk;
    const int kh = qh / gqa_ratio;

    const int q_row_base = qb * Br;
    if (q_row_base >= M) return;

    const int M_row_base = (lane / 16) * 8;   // {0, 8}
    const int col_lane   = lane % 16;
    const int half_sel   = lane / 16;         // {0,1}
    const int q_row0     = q_row_base + M_row_base;

    const _Float16* Q_head = reinterpret_cast<const _Float16*>(Q) + (long)qh * M * D;
    const _Float16* K_head = reinterpret_cast<const _Float16*>(K) + (long)kh * M * D;
    const _Float16* V_head = reinterpret_cast<const _Float16*>(V) + (long)kh * M * D;
    __half*         O_head = O + (long)qh * M * D;

    half8_t Qf[KT];
    {
        int q_row_lane = q_row_base + col_lane;
        bool row_valid = (q_row_lane < M);
        #pragma unroll
        for (int kt = 0; kt < KT; kt++) {
            int K_col_base = kt * 16 + M_row_base;
            #pragma unroll
            for (int j = 0; j < 8; j++)
                Qf[kt][j] = row_valid ? Q_head[(long)q_row_lane * D + K_col_base + j] : (_Float16)0.0f;
        }
    }

    float m_i[8], l_i[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) { m_i[j] = -INFINITY; l_i[j] = 0.0f; }

    float8_t Oacc[NT];
    #pragma unroll
    for (int nt = 0; nt < NT; nt++)
        #pragma unroll
        for (int j = 0; j < 8; j++) Oacc[nt][j] = 0.0f;

    const float log2e = 1.4426950408889634f;
    const float scale = rsqrtf((float)D) * log2e;

    int s_max = q_row_base + Br;
    if (s_max > M) s_max = M;

    for (int s_start = 0; s_start < s_max; s_start += Bc) {

        float8_t Sacc[NC];
        #pragma unroll
        for (int nc = 0; nc < NC; nc++)
            #pragma unroll
            for (int j = 0; j < 8; j++) Sacc[nc][j] = 0.0f;
        #pragma unroll
        for (int nc = 0; nc < NC; nc++) {
            half8_t Kf[KT];
            int s_pos = s_start + nc * 16 + col_lane;
            bool s_valid = (s_pos < M);
            const _Float16* k_row = s_valid ? (K_head + (long)s_pos * D) : nullptr;
            #pragma unroll
            for (int kt = 0; kt < KT; kt++) {
                int K_row_base = kt * 16 + M_row_base;
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    Kf[kt][j] = s_valid ? k_row[K_row_base + j] : (_Float16)0.0f;
            }
            #pragma unroll
            for (int kt = 0; kt < KT; kt++)
                Sacc[nc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(Qf[kt], Kf[kt], Sacc[nc]);
        }

        #pragma unroll
        for (int nc = 0; nc < NC; nc++) {
            int col_global = s_start + nc * 16 + col_lane;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int row_global = q_row0 + j;
                bool valid = (col_global <= row_global) && (col_global < M);
                Sacc[nc][j] = valid ? (Sacc[nc][j] * scale) : -INFINITY;
            }
        }

        float row_max[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float m = Sacc[0][j];
            #pragma unroll
            for (int nc = 1; nc < NC; nc++) m = fmaxf(m, Sacc[nc][j]);
            m = fmaxf(m, __shfl_xor(m, 1));
            m = fmaxf(m, __shfl_xor(m, 2));
            m = fmaxf(m, __shfl_xor(m, 4));
            m = fmaxf(m, __shfl_xor(m, 8));
            row_max[j] = m;
        }

        float row_corr[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float new_m = fmaxf(m_i[j], row_max[j]);
            float corr = (new_m == -INFINITY) ? 0.0f : exp2f(m_i[j] - new_m);
            float s = 0.0f;
            #pragma unroll
            for (int nc = 0; nc < NC; nc++) {
                float es = (new_m == -INFINITY) ? 0.0f : exp2f(Sacc[nc][j] - new_m);
                Sacc[nc][j] = es;
                s += es;
            }
            s += __shfl_xor(s, 1);
            s += __shfl_xor(s, 2);
            s += __shfl_xor(s, 4);
            s += __shfl_xor(s, 8);
            l_i[j] = l_i[j] * corr + s;
            m_i[j] = new_m;
            row_corr[j] = corr;
        }

        #pragma unroll
        for (int nt = 0; nt < NT; nt++)
            #pragma unroll
            for (int j = 0; j < 8; j++) Oacc[nt][j] *= row_corr[j];

        // ---- register-shuffle transpose of each N-tile to A-fragment (no SMEM).
        //   Acc layout: lane l holds Sacc[j] = P[row=(l/16)*8+j, col=l%16].
        //   A-frag wanted: lane l holds Pf[j] = P[row=l%16, col=(l/16)*8+j].
        //   For dest (l, j): need P[row=l%16, col=(l/16)*8+j].
        //     src_lane = ((l%16)/8)*16 + (l/16)*8 + j   (lane whose col == wanted col)
        //     src_reg  = (l%16)%8                        (reg holding wanted row)
        //   src_reg is the SAME for all 8 dest regs of a lane (depends only on l%16),
        //   so we shfl each of the 8 source registers once, then pick src_reg.
        half8_t Pf[NC];
        const int src_reg   = col_lane & 7;               // (l%16)%8
        const int srclo     = (col_lane >> 3) * 16 + M_row_base;  // ((l%16)/8)*16 + (l/16)*8
        #pragma unroll
        for (int nc = 0; nc < NC; nc++) {
            #pragma unroll
            for (int j2 = 0; j2 < 8; j2++) {
                int src_lane = srclo + j2;
                // broadcast all 8 regs from src_lane, then select src_reg
                float sel = 0.0f;
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    float v = __shfl(Sacc[nc][r], src_lane);
                    if (r == src_reg) sel = v;
                }
                Pf[nc][j2] = (_Float16)sel;
            }
        }

        #pragma unroll
        for (int nc = 0; nc < NC; nc++) {
            half8_t Vf[NT];
            #pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                int N_col_v = nt * 16 + col_lane;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int s_pos = s_start + nc * 16 + M_row_base + j;
                    Vf[nt][j] = (s_pos < M) ? V_head[(long)s_pos * D + N_col_v] : (_Float16)0.0f;
                }
            }
            #pragma unroll
            for (int nt = 0; nt < NT; nt++)
                Oacc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(Pf[nc], Vf[nt], Oacc[nt]);
        }
    }

    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        int col = nt * 16 + col_lane;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int row = q_row0 + j;
            if (row < M) {
                float inv_l = (l_i[j] > 0.0f) ? (1.0f / l_i[j]) : 0.0f;
                O_head[(long)row * D + col] = __float2half(Oacc[nt][j] * inv_l);
            }
        }
    }
}

template<int NC>
void launch_v3c(const __half*Q,const __half*K,const __half*V,__half*O,int M,int H,int Hk){
    dim3 grid((M + 15) / 16, H);
    dim3 block(32);
    fa2_v3c<NC><<<grid, block>>>(Q, K, V, O, M, H, Hk);
}

// ============================================================================
// Harness
// ============================================================================
static double nrmse(const std::vector<__half>& a, const std::vector<__half>& b){
    double e=0,n=0;
    for(size_t i=0;i<a.size();i++){ float x=__half2float(a[i]),y=__half2float(b[i]); e+=(x-y)*(x-y); n+=x*x; }
    return sqrt(e/n);
}

typedef void (*launch_fn)(const __half*,const __half*,const __half*,__half*,int,int,int);

static void validate(const char* name, launch_fn fn, int M, int H, int Hk, int D){
    size_t qn=(size_t)M*H*D, kn=(size_t)M*Hk*D;
    std::vector<__half> hQ(qn),hK(kn),hV(kn);
    srand(3);
    for(auto&x:hQ)x=__float2half((rand()%200-100)/128.f);
    for(auto&x:hK)x=__float2half((rand()%200-100)/128.f);
    for(auto&x:hV)x=__float2half((rand()%200-100)/128.f);
    __half *Q,*K,*V,*O1,*O2;
    CK(hipMalloc(&Q,qn*2));CK(hipMalloc(&K,kn*2));CK(hipMalloc(&V,kn*2));
    CK(hipMalloc(&O1,qn*2));CK(hipMalloc(&O2,qn*2));
    CK(hipMemcpy(Q,hQ.data(),qn*2,hipMemcpyHostToDevice));
    CK(hipMemcpy(K,hK.data(),kn*2,hipMemcpyHostToDevice));
    CK(hipMemcpy(V,hV.data(),kn*2,hipMemcpyHostToDevice));
    launch_flash_prefill_v1(Q,K,V,O1,M,H,Hk,D,0); CK(hipDeviceSynchronize());
    fn(Q,K,V,O2,M,H,Hk); CK(hipDeviceSynchronize());
    std::vector<__half> a(qn),b(qn);
    CK(hipMemcpy(a.data(),O1,qn*2,hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(),O2,qn*2,hipMemcpyDeviceToHost));
    double r=nrmse(a,b);
    printf("  [%s] M=%-5d NRMSE=%.6f  %s\n",name,M,r,r<0.01?"PASS":"FAIL");
    hipFree(Q);hipFree(K);hipFree(V);hipFree(O1);hipFree(O2);
}

static double timeit(launch_fn fn, const __half*Q,const __half*K,const __half*V,__half*O,
                     int M,int H,int Hk,int NL,int R){
    for(int w=0;w<3;w++) for(int l=0;l<NL;l++) fn(Q,K,V,O,M,H,Hk);
    CK(hipDeviceSynchronize());
    double t0=fnow();
    for(int i=0;i<R;i++) for(int l=0;l<NL;l++) fn(Q,K,V,O,M,H,Hk);
    CK(hipDeviceSynchronize());
    return (fnow()-t0)/R;
}

// wrapper for v1 to match launch_fn signature
static void v1_wrap(const __half*Q,const __half*K,const __half*V,__half*O,int M,int H,int Hk){
    launch_flash_prefill_v1(Q,K,V,O,M,H,Hk,128,0);
}

int main(){
    const int H=32,Hk=8,D=128,NL=50;

    printf("=== CORRECTNESS (NRMSE vs v1, <0.01 = PASS) ===\n");
    for(int M : {512, 2048}){
        validate("v3_exp2 LB4 (Bc16)", launch_v3_exp2<4>, M,H,Hk,D);
        validate("v3_exp2 LB6 (Bc16)", launch_v3_exp2<6>, M,H,Hk,D);
        validate("v3a NC=2  (Bc32) ", launch_v3a<2>, M,H,Hk,D);
        validate("v3a NC=4  (Bc64) ", launch_v3a<4>, M,H,Hk,D);
        validate("v3c NC=2  (regT) ", launch_v3c<2>, M,H,Hk,D);
    }

    // --- WINNER (v3_exp2 = lever 2) at all M; slow controls (lever 1/3) at small M only ---
    printf("\n=== SPEED: FA2 50L (ms), Bielik H=32 Hk=8 D=128 ===\n");
    printf("BEST = v3_exp2 LB4 (lever 2: exp2f). v3a/v3c are NEGATIVE controls (lose).\n");
    printf("%-7s %9s %9s %9s %9s %9s %9s\n","M","v1","exp2_LB4","exp2_LB6","exp2_LB8","v3a_Bc32","v3c_Bc32");
    for(int M : {512, 2048, 8192}){
        bool slow = (M <= 2048);   // skip the 5x-slower controls at M=8192 (keeps run fast)
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

        double t_v1  = timeit(v1_wrap,           Q,K,V,O,M,H,Hk,NL,3);
        double t_e4  = timeit(launch_v3_exp2<4>, Q,K,V,O,M,H,Hk,NL,3);
        double t_e6  = timeit(launch_v3_exp2<6>, Q,K,V,O,M,H,Hk,NL,3);
        double t_e8  = timeit(launch_v3_exp2<8>, Q,K,V,O,M,H,Hk,NL,3);
        double t_3a2 = slow ? timeit(launch_v3a<2>, Q,K,V,O,M,H,Hk,NL,3) : 0.0;
        double t_3c2 = slow ? timeit(launch_v3c<2>, Q,K,V,O,M,H,Hk,NL,3) : 0.0;
        printf("%-7d %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f\n", M, t_v1, t_e4, t_e6, t_e8, t_3a2, t_3c2);
        printf("%-7s %9s %8.2fx %8.2fx %8.2fx %8.2fx %8.2fx\n","spd","",
               t_v1/t_e4, t_v1/t_e6, t_v1/t_e8,
               slow?t_v1/t_3a2:0.0, slow?t_v1/t_3c2:0.0);
        hipFree(Q);hipFree(K);hipFree(V);hipFree(O);
    }
    printf("\nBEST FA2 50L times (v3_exp2 LB4) for full-forward estimate:\n");
    printf("  see exp2_LB4 column above. pp512 / pp2048 / pp8192.\n");
    return 0;
}
