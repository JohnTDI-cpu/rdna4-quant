// swmmac_gemm.hip — Native RDNA4 SWMMAC/WMMA INT4 GEMM kernels
// Target: V_SWMMAC_I32_16X16X64_IU4 (477 TOPS raw on R9700)
// Signed symmetric quantization: zero=0, neg=true → zero dequant overhead
// Column-distributed output: lane%16=col, (lane/16)*8+j=row (DISCOVERY verified)
//
// Build: included by int4_decode_step.hip, compiled with --offload-arch=gfx1201
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef int swmmac_v2i __attribute__((ext_vector_type(2)));
typedef int swmmac_v4i __attribute__((ext_vector_type(4)));
typedef int swmmac_v8i __attribute__((ext_vector_type(8)));

// ============================================================
// Signed symmetric activation quantization for SWMMAC
// absmax/7 scale, stored as unsigned+8 (neg=true in SWMMAC treats as signed)
// ============================================================
__global__ void swmmac_quantize_act(
    const __half* __restrict__ input,  // [M, K] FP16
    uint8_t* __restrict__ act_q,       // [M_pad, K/2] packed INT4
    float* __restrict__ act_scales,    // [M_pad, K/GS] scales
    int M, int K, int GS)
{
    int g = blockIdx.x, m = blockIdx.y, tid = threadIdx.x;
    if (m >= M) return;
    int ng = K / GS;
    // Each thread handles one element within a group of 32
    float val = __half2float(input[(int64_t)m * K + g * 32 + tid]);
    float amax = fabsf(val);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off));
    float scale = (amax > 1e-10f) ? (amax / 7.0f) : 1e-10f;
    int q = __float2int_rn(val / scale);
    q = max(-8, min(7, q));
    int stored = q + 8;  // 0..15 unsigned
    int q_other = __shfl_xor(stored, 1);
    if (tid & 1) act_q[(int64_t)m * (K/2) + g*16 + tid/2] = (uint8_t)((stored << 4) | q_other);
    if (tid == 0) act_scales[m * ng + g] = scale;
}

// ============================================================
// SWMMAC 2:4 Sparse GEMM — peak hardware utilization
//
// Tile: 16M × 64N per block, K processed in GS=64 steps
// 4 SWMMAC per K-step (4 N-tiles × 16)
// 1 warp per block, launch_bounds(32,4) for occupancy
//
// Index encoding: 2-bit per group of 4 B elements
//   0→(0,1), 1→(0,2), 2→(0,3), 3→(1,2)
// ============================================================
template<int GS=64>
__global__ __launch_bounds__(32, 4)
void swmmac_gemm_kernel(
    const uint8_t* __restrict__ act_q,       // [M_pad, K/2] dense INT4
    const float* __restrict__ act_scales,    // [M_pad, ng]
    const uint8_t* __restrict__ wt_sparse,   // [N, K/4] compressed (2 nibbles per 4)
    const short* __restrict__ wt_idx,        // [N, ng, 2] pre-packed (8×2-bit per short)
    const float* __restrict__ w_scales,      // [ng, N_pad] transposed
    __half* __restrict__ output,             // [M, N_pad]
    int M, int N_pad, int K)
{
    const int lane = threadIdx.x;
    const int m_start = blockIdx.y * 16;
    const int n_start = blockIdx.x * 64;
    if (m_start >= M) return;

    const int nl = lane % 16, kg = lane / 16;
    const int ng = K / GS;
    const int m_idx = m_start + nl;
    const bool m_valid = (m_idx < M);

    float acc0[8]={}, acc1[8]={}, acc2[8]={}, acc3[8]={};

    for (int g = 0; g < ng; g++) {
        int k = g * GS;

        // --- Load B (dense activations): v4i per lane ---
        swmmac_v4i B;
        if (m_valid) {
            const uint8_t* aq = act_q + (int64_t)m_idx * (K/2) + k/2;
            const int* ap = (const int*)(aq + kg * 16);
            B = (swmmac_v4i){ap[0], ap[1], ap[2], ap[3]};
        } else { B = (swmmac_v4i){}; }

        // --- Load 4 A tiles (sparse weights) + pack indices ---
        // Load sparse A + pre-packed index (1 short load instead of 8-byte pack loop)
        #define LOAD_SPARSE(ti) \
            swmmac_v2i A##ti; short idx##ti; { \
                int row = n_start + ti*16 + nl; \
                const uint8_t* sp = wt_sparse + (int64_t)row * (K/4) + k/4 + kg*8; \
                A##ti = (swmmac_v2i){*(const int*)sp, *(const int*)(sp+4)}; \
                idx##ti = wt_idx[(int64_t)row * ng * 2 + g * 2 + kg]; \
            }
        LOAD_SPARSE(0) LOAD_SPARSE(1) LOAD_SPARSE(2) LOAD_SPARSE(3)
        #undef LOAD_SPARSE

        // --- 4 SWMMAC: signed sparse × signed dense → INT32 ---
        swmmac_v8i R0={},R1={},R2={},R3={};
        R0 = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(true, A0, true, B, R0, idx0, false);
        R1 = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(true, A1, true, B, R1, idx1, false);
        R2 = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(true, A2, true, B, R2, idx2, false);
        R3 = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(true, A3, true, B, R3, idx3, false);

        // --- Minimal dequant: w_scale × a_scale × raw_dot ---
        float a_sc = m_valid ? act_scales[m_idx * ng + g] : 0.0f;
        #define DEQUANT(ti) { \
            float ws = w_scales[g * N_pad + n_start + ti*16 + nl]; \
            float c = ws * a_sc; \
            for(int j=0;j<8;j++) acc##ti[j] += c * (float)R##ti[j]; }
        DEQUANT(0) DEQUANT(1) DEQUANT(2) DEQUANT(3)
        #undef DEQUANT
    }

    // --- Store: column-distributed (lane%16=col, (lane/16)*8+j=row) ---
    for (int j = 0; j < 8; j++) {
        int om = m_start + kg*8 + j;
        if (om >= M) continue;
        __half* row = output + (int64_t)om * N_pad;
        row[n_start + nl]      = __float2half(acc0[j]);
        row[n_start + 16 + nl] = __float2half(acc1[j]);
        row[n_start + 32 + nl] = __float2half(acc2[j]);
        row[n_start + 48 + nl] = __float2half(acc3[j]);
    }
}

// ============================================================
// Dense WMMA variant (fallback when weights aren't sparse)
// Same interface, uses V_WMMA_I32_16X16X32_IU4
// ============================================================
template<int GS=64>
__global__ __launch_bounds__(32, 4)
void wmma_dense_gemm_kernel(
    const uint8_t* __restrict__ act_q,
    const float* __restrict__ act_scales,
    const uint8_t* __restrict__ wt_tiled,    // pre-tiled [K/8, N/16, 64]
    const float* __restrict__ w_scales,      // [ng, N_pad]
    __half* __restrict__ output,
    int M, int N_pad, int K)
{
    const int lane = threadIdx.x;
    const int m_start = blockIdx.y * 16;
    const int n_start = blockIdx.x * 64;
    if (m_start >= M) return;

    const int nl = lane % 16, kg = lane / 16;
    const int NB = N_pad / 16, ng = K / GS;
    const int m_idx = m_start + nl;
    const bool m_valid = (m_idx < M);

    float acc0[8]={}, acc1[8]={}, acc2[8]={}, acc3[8]={};

    for (int g = 0; g < ng; g++) {
        int k = g * GS;

        // Load A (activations)
        swmmac_v2i A;
        float a_sc = 0.0f;
        if (m_valid) {
            const uint8_t* aq = act_q + (int64_t)m_idx * (K/2) + k/2 + kg*8;
            A = (swmmac_v2i){*(const int*)aq, *(const int*)(aq+4)};
            a_sc = act_scales[m_idx * ng + g];
        } else { A = (swmmac_v2i){}; }

        // Load 4 B tiles (tiled weights)
        #define LOAD_B(ti) swmmac_v2i B##ti; { \
            int nb = n_start/16 + ti; int kc0 = k/8 + kg*2; \
            B##ti = (swmmac_v2i){ \
                (int)*(const uint32_t*)(wt_tiled + (int64_t)(kc0*NB+nb)*64 + nl*4), \
                (int)*(const uint32_t*)(wt_tiled + (int64_t)((kc0+1)*NB+nb)*64 + nl*4)}; }
        LOAD_B(0) LOAD_B(1) LOAD_B(2) LOAD_B(3)
        #undef LOAD_B

        // 4 dense WMMA
        swmmac_v8i R0={},R1={},R2={},R3={};
        R0 = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, A, true, B0, R0, false);
        R1 = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, A, true, B1, R1, false);
        R2 = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, A, true, B2, R2, false);
        R3 = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, A, true, B3, R3, false);

        // Dequant
        #define DEQUANT(ti) { \
            float ws = w_scales[g * N_pad + n_start + ti*16 + nl]; \
            float c = ws * a_sc; \
            for(int j=0;j<8;j++) acc##ti[j] += c * (float)R##ti[j]; }
        DEQUANT(0) DEQUANT(1) DEQUANT(2) DEQUANT(3)
        #undef DEQUANT
    }

    // Store
    for (int j = 0; j < 8; j++) {
        int om = m_start + kg*8 + j;
        if (om >= M) continue;
        __half* row = output + (int64_t)om * N_pad;
        row[n_start + nl]      = __float2half(acc0[j]);
        row[n_start + 16 + nl] = __float2half(acc1[j]);
        row[n_start + 32 + nl] = __float2half(acc2[j]);
        row[n_start + 48 + nl] = __float2half(acc3[j]);
    }
}

// ============================================================
// Weight preparation: tile for WMMA, compress+index for SWMMAC
// ============================================================
__global__ void tile_weights_kernel(
    const uint8_t* __restrict__ src,   // [N, K/2] packed INT4
    uint8_t* __restrict__ dst,         // tiled format
    int N, int K)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int NB = N / 16, total = (K/8) * NB * 16;
    if (tid >= total) return;
    int kc = tid / (NB * 16), rem = tid % (NB * 16), nb = rem / 16, nl = rem % 16;
    *(uint32_t*)(dst + (int64_t)(kc*NB+nb)*64 + nl*4)
        = *(const uint32_t*)(src + (int64_t)(nb*16+nl)*(K/2) + kc*4);
}

__global__ void transpose_scales_kernel(
    const __half* __restrict__ si,  // [N, ng, 2] interleaved (scale, zero)
    float* __restrict__ so,         // [ng, N_pad] transposed scales only (float32)
    int N, int N_pad, int ng)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * ng) return;
    int n = tid / ng, g = tid % ng;
    so[(int64_t)g * N_pad + n] = __half2float(si[(int64_t)n * ng * 2 + g * 2]);
}
