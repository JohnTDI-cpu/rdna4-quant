// flash_prefill.hip — FlashAttention v2 forward dla RDNA4 (gfx1201)
// Drop-in replacement dla F.scaled_dot_product_attention(causal=True, enable_gqa=True)
//
// v0: warp-per-Q-row, online softmax, bez WMMA (validation baseline)
//   - Block: (32, BR) = 32*BR threads, BR warps, każdy warp = 1 Q row
//   - Grid: (ceil(M/BR), H) — H = num_heads (Q heads, not KV)
//   - Q layout: [H, M, D]   FP16 contiguous
//   - K layout: [Hk, M, D]  FP16 contiguous
//   - V layout: [Hk, M, D]  FP16 contiguous
//   - O layout: [H, M, D]   FP16 contiguous
//   - GQA: kh = qh / (H/Hk)
//   - causal: q_row attends only to s <= q_row
//   - D = 128 (Qwen3 head_dim), 4 elementy per lane
//   - Online softmax z log-sum-exp (numerically stable)
//
// Build: included from int4_decode_step.hip
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ============================================================
// FA2 prefill forward — warp-per-Q-row baseline (v0)
// ============================================================
template <int BR, int BC, int D>
__global__ void flash_prefill_fwd_v0(
    const __half* __restrict__ Q,    // [H, M, D]
    const __half* __restrict__ K,    // [Hk, M, D]
    const __half* __restrict__ V,    // [Hk, M, D]
    __half* __restrict__ O,          // [H, M, D]
    int M, int H, int Hk
) {
    static_assert(D == 128, "v0 wymaga D=128 (4 elementy per lane)");

    const int qb   = blockIdx.x;          // Q-row block
    const int qh   = blockIdx.y;          // Q head
    const int warp = threadIdx.y;         // 0..BR-1 (warp ID = lokalny wiersz Q)
    const int lane = threadIdx.x;         // 0..31

    const int gqa_ratio = H / Hk;
    const int kh = qh / gqa_ratio;

    const int q_row = qb * BR + warp;
    if (q_row >= M) return;

    const float scale = rsqrtf((float)D);

    // ---- Załaduj Q row do rejestrów: D=128, lane trzyma 4 elementy
    float q_reg[4];
    const __half* q_ptr = Q + (long)qh * M * D + (long)q_row * D;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        q_reg[i] = __half2float(q_ptr[lane + i * 32]);

    // ---- Online softmax accumulators (per-lane partial output)
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // ---- Causal: tylko s <= q_row
    const int s_max = q_row + 1;

    const __half* k_base = K + (long)kh * M * D;
    const __half* v_base = V + (long)kh * M * D;

    // ---- Iteruj po wszystkich pozycjach K/V (causal)
    // BC nie wpływa na correctness w v0, ale zostawiam dla future tile structure
    for (int s = 0; s < s_max; s++) {
        // Q·K dot (D=128, każdy lane robi 4 mul-add, potem warp reduce)
        const __half* k_ptr = k_base + (long)s * D;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            dot += q_reg[i] * __half2float(k_ptr[lane + i * 32]);
        dot += __shfl_xor(dot, 16);
        dot += __shfl_xor(dot, 8);
        dot += __shfl_xor(dot, 4);
        dot += __shfl_xor(dot, 2);
        dot += __shfl_xor(dot, 1);
        dot *= scale;

        // Online softmax: m_i, l_i update + output rescale
        float new_m = fmaxf(m_i, dot);
        float corr  = expf(m_i - new_m);
        float exp_s = expf(dot - new_m);
        l_i = l_i * corr + exp_s;

        const __half* v_ptr = v_base + (long)s * D;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            o_reg[i] = o_reg[i] * corr + exp_s * __half2float(v_ptr[lane + i * 32]);
        m_i = new_m;
    }

    // ---- Normalize i zapisz O (l_i = sum exp; ostateczna normalizacja)
    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    __half* o_ptr = O + (long)qh * M * D + (long)q_row * D;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        o_ptr[lane + i * 32] = __float2half(o_reg[i] * inv_l);

    (void)BC; // suppress unused warning - tile struktura w v1
}

// ============================================================
// FA2 prefill v1 — FP16 WMMA Q@K^T + P@V, online softmax tile-wise
//   Block: 1 warp (32 threads)
//   Grid: (ceil(M/16), H)
//   Br=16 Q rows per block, Bc=16 KV positions per inner iter
//   D=128: 8 WMMA ops dla Q@K^T (K=128/16), 8 N-tiles dla P@V (D=128/16)
//   Acc layout: lane%16=col, (lane/16)*8+j=row (verified gfx12 column-distributed)
//   GQA: kh = qh / (H/Hk)
//   Causal: per (row,col) check; numerically -INFINITY pre-softmax
// ============================================================
typedef __attribute__((ext_vector_type(8))) _Float16 half8_t;
typedef __attribute__((ext_vector_type(8))) float    float8_t;

__global__ __launch_bounds__(32, 4)
void flash_prefill_fwd_v1(
    const __half* __restrict__ Q,   // [H, M, D=128]
    const __half* __restrict__ K,   // [Hk, M, D=128]
    const __half* __restrict__ V,   // [Hk, M, D=128]
    __half* __restrict__ O,         // [H, M, D=128]
    int M, int H, int Hk
) {
    constexpr int Br = 16;
    constexpr int Bc = 16;
    constexpr int D  = 128;
    constexpr int KT = D / 16;   // 8 fragments po K=16 dla Q@K^T
    constexpr int NT = D / 16;   // 8 N-tiles dla P@V

    const int qb = blockIdx.x;
    const int qh = blockIdx.y;
    const int lane = threadIdx.x;
    const int gqa_ratio = H / Hk;
    const int kh = qh / gqa_ratio;

    const int q_row_base = qb * Br;
    if (q_row_base >= M) return;

    const int M_row_base = (lane / 16) * 8;     // {0, 8}
    const int col_lane   = lane % 16;
    const int q_row0     = q_row_base + M_row_base;

    const _Float16* Q_head = reinterpret_cast<const _Float16*>(Q) + (long)qh * M * D;
    const _Float16* K_head = reinterpret_cast<const _Float16*>(K) + (long)kh * M * D;
    const _Float16* V_head = reinterpret_cast<const _Float16*>(V) + (long)kh * M * D;
    __half*         O_head = O + (long)qh * M * D;

    // ---- Q fragments [KT][8 elementy per lane]
    //   A-fragment layout (gfx12 wave32): lane%16 = M_row, (lane/16)*8+j = K_col
    //   (Sprawdzone w działającym wmma_dense_gemm_kernel: m_idx = m_start + lane%16)
    //   Załaduj raz, używaj przez wszystkie iteracje KV
    half8_t Qf[KT];
    {
        int q_row_lane = q_row_base + col_lane;
        bool row_valid = (q_row_lane < M);
        #pragma unroll
        for (int kt = 0; kt < KT; kt++) {
            int K_col_base = kt * 16 + M_row_base;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int K_col = K_col_base + j;
                Qf[kt][j] = row_valid ? Q_head[(long)q_row_lane * D + K_col] : (_Float16)0.0f;
            }
        }
    }

    // ---- Online softmax state (per lane: 8 rzędów × 1 wartość)
    float m_i[8], l_i[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) { m_i[j] = -INFINITY; l_i[j] = 0.0f; }

    // ---- O accumulators (per lane: NT N-tiles × 8 row entries = 64 acc)
    float8_t Oacc[NT];
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) Oacc[nt][j] = 0.0f;
    }

    const float scale = rsqrtf((float)D);

    // Causal: max KV pozycja istotna dla tego Q tile
    int s_max = q_row_base + Br;
    if (s_max > M) s_max = M;

    // ============================================================
    // Iteruj po KV tiles (Bc=16 pozycji each)
    // ============================================================
    for (int s_start = 0; s_start < s_max; s_start += Bc) {

        // ---- Załaduj K tile dla Q@K^T
        //   Q@K^T: Q jest A (M=16, K=128), K^T jest B (K=128, N=Bc=16)
        //   B fragment: lane%16 = N_col, (lane/16)*8+j = K_row
        //   Mamy K[bc, d]; dla B^T trzeba K[N_col, K_row]
        //   N_col = pozycja w Bc (= lane%16)
        //   K_row = wymiar D
        half8_t Kf[KT];
        {
            int s_pos = s_start + col_lane;
            bool s_valid = (s_pos < M);
            const _Float16* k_row = s_valid ? (K_head + (long)s_pos * D) : nullptr;
            #pragma unroll
            for (int kt = 0; kt < KT; kt++) {
                int K_row_base = kt * 16 + M_row_base;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int K_row = K_row_base + j;
                    Kf[kt][j] = s_valid ? k_row[K_row] : (_Float16)0.0f;
                }
            }
        }

        // ---- Compute S = Q @ K^T (akumuluj 8 WMMA ops)
        float8_t Sacc;
        #pragma unroll
        for (int j = 0; j < 8; j++) Sacc[j] = 0.0f;
        #pragma unroll
        for (int kt = 0; kt < KT; kt++) {
            Sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                Qf[kt], Kf[kt], Sacc);
        }

        // ---- Apply scale + causal mask
        //   Sacc[j] reprezentuje (row=q_row0+j, col=s_start+col_lane)
        {
            int col_global = s_start + col_lane;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int row_global = q_row0 + j;
                bool valid = (col_global <= row_global) && (col_global < M);
                Sacc[j] = valid ? (Sacc[j] * scale) : -INFINITY;
            }
        }

        // ---- Per-row max via shfl_xor within 16-lane subgroup (offsets 1,2,4,8)
        float row_max[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float m = Sacc[j];
            m = fmaxf(m, __shfl_xor(m, 1));
            m = fmaxf(m, __shfl_xor(m, 2));
            m = fmaxf(m, __shfl_xor(m, 4));
            m = fmaxf(m, __shfl_xor(m, 8));
            row_max[j] = m;
        }

        // ---- Online softmax: nowy m, exp, sum, correction dla l_i i O
        float row_corr[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float new_m = fmaxf(m_i[j], row_max[j]);
            // jeśli new_m == -inf (cały row zamaskowany) → corr=0, exp_s=0, l zostaje 0
            float corr = (new_m == -INFINITY) ? 0.0f : expf(m_i[j] - new_m);
            float exp_s = (new_m == -INFINITY) ? 0.0f : expf(Sacc[j] - new_m);
            Sacc[j] = exp_s;  // Sacc trzyma teraz P (un-normalized)
            // sum exp_s over row (16 lanes)
            float s = exp_s;
            s += __shfl_xor(s, 1);
            s += __shfl_xor(s, 2);
            s += __shfl_xor(s, 4);
            s += __shfl_xor(s, 8);
            l_i[j] = l_i[j] * corr + s;
            m_i[j] = new_m;
            row_corr[j] = corr;
        }

        // ---- Konwersja P (acc layout) do A-fragment layout przez LDS transpose
        //   Sacc layout:  (lane/16)*8+j = M_row, lane%16 = N_col_in_S = K_col_in_P
        //   A-fragment layout: lane%16 = M_row, (lane/16)*8+j = K_col
        //   Transpose przez 16x16 SMEM tile.
        __shared__ _Float16 P_smem[16][16];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            P_smem[M_row_base + j][col_lane] = (_Float16)Sacc[j];
        }
        __syncwarp();
        half8_t Pf;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            // P_frag[j] = P[M_row=lane%16, K_col=(lane/16)*8+j]
            Pf[j] = P_smem[col_lane][M_row_base + j];
        }
        __syncwarp();

        // ---- Załaduj V tile dla P@V
        //   P@V: P jest A (M=16, K=Bc=16), V jest B (K=Bc=16, N=D=128)
        //   B fragment: lane%16 = N_col (= d), (lane/16)*8+j = K_row (= bc)
        //   V[bc, d] → V_frag[(lane/16)*8+j, lane%16] dla każdego N-tile
        half8_t Vf[NT];
        #pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            int N_col_v = nt * 16 + col_lane;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int K_row_v = M_row_base + j;
                int s_pos = s_start + K_row_v;
                Vf[nt][j] = (s_pos < M) ? V_head[(long)s_pos * D + N_col_v] : (_Float16)0.0f;
            }
        }

        // ---- Apply correction do Oacc + akumuluj P @ V_nt
        #pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) Oacc[nt][j] *= row_corr[j];
            Oacc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                Pf, Vf[nt], Oacc[nt]);
        }
    }

    // ---- Final normalize i write O
    //   Oacc[nt][j] @ (row=q_row0+j, col=nt*16+col_lane)
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

// ============================================================
// DEBUG: pure Q @ K^T (no softmax), sprawdzenie WMMA layout
//   Q: [M=16, D=128]  K: [Bc=16, D=128]  S: [16, 16]
// ============================================================
__global__ __launch_bounds__(32)
void debug_qkt_kernel(
    const __half* __restrict__ Q,   // [M=16, D=128]
    const __half* __restrict__ K,   // [Bc=16, D=128]
    __half* __restrict__ S          // [16, 16] output FP16
) {
    constexpr int D = 128;
    constexpr int KT = D / 16;
    const int lane = threadIdx.x;
    const int M_row_base = (lane / 16) * 8;
    const int col_lane   = lane % 16;

    const _Float16* Qh = reinterpret_cast<const _Float16*>(Q);
    const _Float16* Kh = reinterpret_cast<const _Float16*>(K);

    // Q (A frag): lane%16 = M_row, (lane/16)*8+j = K_col
    half8_t Qf[KT];
    {
        int q_row = col_lane;
        #pragma unroll
        for (int kt = 0; kt < KT; kt++) {
            int K_col_base = kt * 16 + M_row_base;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                Qf[kt][j] = Qh[(long)q_row * D + K_col_base + j];
            }
        }
    }

    // K^T (B frag): lane%16 = N_col=bc, (lane/16)*8+j = K_row=d
    half8_t Kf[KT];
    #pragma unroll
    for (int kt = 0; kt < KT; kt++) {
        int K_row_base = kt * 16 + M_row_base;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int K_row = K_row_base + j;
            int bc = col_lane;
            Kf[kt][j] = Kh[(long)bc * D + K_row];
        }
    }

    float8_t Sacc;
    #pragma unroll
    for (int j = 0; j < 8; j++) Sacc[j] = 0.0f;
    #pragma unroll
    for (int kt = 0; kt < KT; kt++) {
        Sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(Qf[kt], Kf[kt], Sacc);
    }

    // Acc layout: lane%16 = N_col=bc, (lane/16)*8+j = M_row
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int row = M_row_base + j;
        int col = col_lane;
        S[row * 16 + col] = __float2half(Sacc[j]);
    }
}

static inline void launch_debug_qkt(const __half* Q, const __half* K, __half* S, hipStream_t st) {
    hipLaunchKernelGGL(debug_qkt_kernel, dim3(1), dim3(32), 0, st, Q, K, S);
}

// ============================================================
// DEBUG: full FA single tile (no causal, no GQA), M=Bc=16
// ============================================================
__global__ __launch_bounds__(32)
void debug_fa_single_tile(
    const __half* __restrict__ Q,   // [16, 128]
    const __half* __restrict__ K,   // [16, 128]
    const __half* __restrict__ V,   // [16, 128]
    __half* __restrict__ O          // [16, 128]
) {
    constexpr int D = 128;
    constexpr int KT = D / 16;
    constexpr int NT = D / 16;

    const int lane = threadIdx.x;
    const int M_row_base = (lane / 16) * 8;
    const int col_lane   = lane % 16;

    const _Float16* Qh = reinterpret_cast<const _Float16*>(Q);
    const _Float16* Kh = reinterpret_cast<const _Float16*>(K);
    const _Float16* Vh = reinterpret_cast<const _Float16*>(V);

    // Q (A): lane%16 = M_row, (lane/16)*8+j = K_col
    half8_t Qf[KT];
    {
        int q_row = col_lane;
        #pragma unroll
        for (int kt = 0; kt < KT; kt++) {
            int K_col_base = kt * 16 + M_row_base;
            #pragma unroll
            for (int j = 0; j < 8; j++) Qf[kt][j] = Qh[(long)q_row * D + K_col_base + j];
        }
    }

    // K^T (B for Q@K^T): lane%16 = N_col=bc, (lane/16)*8+j = K_row=d
    half8_t Kf[KT];
    #pragma unroll
    for (int kt = 0; kt < KT; kt++) {
        int K_row_base = kt * 16 + M_row_base;
        #pragma unroll
        for (int j = 0; j < 8; j++) Kf[kt][j] = Kh[(long)col_lane * D + K_row_base + j];
    }

    // S = Q @ K^T
    float8_t Sacc;
    #pragma unroll
    for (int j = 0; j < 8; j++) Sacc[j] = 0.0f;
    #pragma unroll
    for (int kt = 0; kt < KT; kt++)
        Sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(Qf[kt], Kf[kt], Sacc);

    // Scale (no causal)
    const float scale = rsqrtf((float)D);
    #pragma unroll
    for (int j = 0; j < 8; j++) Sacc[j] *= scale;

    // Per-row max (reduce over lane%16 within half-warp)
    float row_max[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        float m = Sacc[j];
        m = fmaxf(m, __shfl_xor(m, 1));
        m = fmaxf(m, __shfl_xor(m, 2));
        m = fmaxf(m, __shfl_xor(m, 4));
        m = fmaxf(m, __shfl_xor(m, 8));
        row_max[j] = m;
    }

    // exp + sum
    float row_sum[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        Sacc[j] = expf(Sacc[j] - row_max[j]);
        float s = Sacc[j];
        s += __shfl_xor(s, 1);
        s += __shfl_xor(s, 2);
        s += __shfl_xor(s, 4);
        s += __shfl_xor(s, 8);
        row_sum[j] = s;
    }

    // Normalize P = exp / sum (in place in Sacc)
    #pragma unroll
    for (int j = 0; j < 8; j++) Sacc[j] /= row_sum[j];

    // Transpose P via SMEM: acc layout → A fragment layout
    __shared__ _Float16 P_smem[16][16];
    #pragma unroll
    for (int j = 0; j < 8; j++) P_smem[M_row_base + j][col_lane] = (_Float16)Sacc[j];
    __syncwarp();
    half8_t Pf;
    #pragma unroll
    for (int j = 0; j < 8; j++) Pf[j] = P_smem[col_lane][M_row_base + j];
    __syncwarp();

    // V (B for P@V): lane%16 = N_col=d, (lane/16)*8+j = K_row=bc
    half8_t Vf[NT];
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        int N_col = nt * 16 + col_lane;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int K_row = M_row_base + j;
            Vf[nt][j] = Vh[(long)K_row * D + N_col];
        }
    }

    // O = P @ V (8 N-tiles)
    float8_t Oacc[NT];
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) Oacc[nt][j] = 0.0f;
        Oacc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(Pf, Vf[nt], Oacc[nt]);
    }

    // Store O (acc layout: (l/16)*8+j = row, l%16 = col)
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        int col = nt * 16 + col_lane;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int row = M_row_base + j;
            O[row * D + col] = __float2half(Oacc[nt][j]);
        }
    }
}

static inline void launch_debug_fa_single(
    const __half* Q, const __half* K, const __half* V, __half* O, hipStream_t st)
{
    hipLaunchKernelGGL(debug_fa_single_tile, dim3(1), dim3(32), 0, st, Q, K, V, O);
}

// Host launcher dla v1
static inline void launch_flash_prefill_v1(
    const __half* Q, const __half* K, const __half* V,
    __half* O, int M, int H, int Hk, int D, hipStream_t st)
{
    if (D != 128) return;
    dim3 grid((M + 15) / 16, H);
    dim3 block(32);
    hipLaunchKernelGGL(flash_prefill_fwd_v1, grid, block, 0, st,
        Q, K, V, O, M, H, Hk);
}

// ============================================================
// Host launcher v0 — kontrakt zgodny z SDPA(causal=True, GQA=True)
// ============================================================
static inline void launch_flash_prefill_v0(
    const __half* Q, const __half* K, const __half* V,
    __half* O, int M, int H, int Hk, int D, hipStream_t st)
{
    constexpr int BR = 4;     // 4 warps per block = 4 Q rows per block
    constexpr int BC = 64;    // (nieużywane w v0, placeholder)
    if (D == 128) {
        dim3 grid((M + BR - 1) / BR, H);
        dim3 block(32, BR);
        hipLaunchKernelGGL((flash_prefill_fwd_v0<BR, BC, 128>), grid, block, 0, st,
            Q, K, V, O, M, H, Hk);
    } else {
        // Fallback: nieobsługiwane head_dim
        // (na razie projekt używa tylko D=128 dla Qwen3)
    }
}
