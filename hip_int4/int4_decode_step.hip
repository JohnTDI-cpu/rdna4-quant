/*
 * INT4 Decode Step V2 — Production-only extract
 * MoE decode (graph + non-graph) + prefill, no debug/coop/hybrid code.
 */

#include <hip/hip_runtime.h>
#include <chrono>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hipblas/hipblas.h>
#include <vector>
#include <cmath>

// Global hipBLAS handle for direct GEMM calls (bypasses PyTorch dispatcher ~8μs/call)
static hipblasHandle_t g_hipblas_handle = nullptr;
static void ensure_hipblas(hipStream_t stream) {
    if (!g_hipblas_handle) {
        hipblasCreate(&g_hipblas_handle);
    }
    hipblasSetStream(g_hipblas_handle, stream);
}

// Direct hipBLAS HGEMM: Y[M,N] = X[M,K] @ W[N,K]^T, all row-major FP16
static inline void direct_hgemm(
    const __half* X, const __half* W, __half* Y,
    int M_rows, int N_cols, int K_inner, hipStream_t stream)
{
    ensure_hipblas(stream);
    // Row-major → col-major trick: Y^T = W * X^T
    // C(N,M) = A(N,K) * B(K,M) where A=W(row-major=col[K,N]), B=X(row-major=col[K,M])
    // So: transA=T (transpose W from [K,N]col to [N,K]col), transB=N (X is [K,M]col)
    hipblasHalf alpha_h, beta_h;
    *(reinterpret_cast<__half*>(&alpha_h)) = __float2half(1.0f);
    *(reinterpret_cast<__half*>(&beta_h))  = __float2half(0.0f);
    hipblasHgemm(g_hipblas_handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
        N_cols, M_rows, K_inner,
        &alpha_h,
        reinterpret_cast<const hipblasHalf*>(W), K_inner,  // W[N,K] row = [K,N] col, transposed
        reinterpret_cast<const hipblasHalf*>(X), K_inner,  // X[M,K] row = [K,M] col
        &beta_h,
        reinterpret_cast<hipblasHalf*>(Y), N_cols);        // Y[M,N] row = [N,M] col
}
// ============================================================
// Device-side decode state for HIP Graph (updated between replays)
// ============================================================
__device__ int g_graph_pos;
__device__ int g_graph_seq_len;

__global__ void set_graph_state(int pos) {
    g_graph_pos = pos;
    g_graph_seq_len = pos + 1;
}

// Auto-increment for graph replay: each replay advances pos by 1
__global__ void increment_graph_state() {
    g_graph_pos++;
    g_graph_seq_len++;
}

// Graph-compatible: store token ID to output array + embed next token into h
// Uses g_graph_pos to compute output index
__global__ void store_token_and_embed_g(
    const int64_t* __restrict__ token_id,
    int64_t* __restrict__ output_ids,
    const __half* __restrict__ embed_w,
    __half* __restrict__ h,
    int initial_pos, int D
) {
    int t = g_graph_pos - initial_pos;
    int64_t id = *token_id;
    if (threadIdx.x == 0) output_ids[t] = id;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        h[i] = embed_w[id * D + i];
}

// ============================================================
// INT4 sign-extend decode
// ============================================================

// INT4 sign-extend decode: nibble [0..15] -> signed [-8..+7]
// NOTE: Legacy symmetric decode. Active kernels now use asymmetric inline decode:
//   (float)nibble - zp, where zp is loaded from interleaved S+Z tensor.
__device__ __forceinline__ float int4_decode(int nibble) {
    return (nibble > 7) ? (float)(nibble - 16) : (float)nibble;
}

// ============================================================
// FP8 E4M3 encode/decode for KV cache
// E4M3: 1 sign + 4 exponent (bias=7) + 3 mantissa
// Range: ±448, denormals supported, 0x7F = NaN
// ============================================================
__device__ __forceinline__ uint8_t f32_to_fp8e4m3(float val) {
    val = fmaxf(-448.0f, fminf(448.0f, val));
    uint32_t f = __float_as_uint(val);
    uint8_t sign = (f >> 24) & 0x80;
    uint32_t exp32 = (f >> 23) & 0xFF;
    uint32_t mant32 = f & 0x7FFFFF;

    if (exp32 == 0) return sign;  // zero/denorm → 0

    int exp = (int)exp32 - 127 + 7;  // rebias to E4M3
    int mant = (int)(mant32 >> 20);   // top 3 mantissa bits
    // Round to nearest
    if (mant32 & 0x80000) {
        mant++;
        if (mant > 7) { mant = 0; exp++; }
    }

    if (exp <= 0) return sign;         // underflow
    if (exp >= 15) return sign | 0x7E; // overflow → max (not NaN=0x7F)

    return sign | (uint8_t)(exp << 3) | (uint8_t)mant;
}

// Branchless FP8 E4M3 → float32 (no branches in common path)
__device__ __forceinline__ float fp8e4m3_to_f32(uint8_t val) {
    // Hardware FP8→F32 conversion: v_cvt_f32_fp8 (1 cycle on gfx1201)
    // Input: uint32 with FP8 value in bits [7:0]. Output: float.
    float result;
    asm volatile("v_cvt_f32_fp8 %0, %1" : "=v"(result) : "v"((uint32_t)val));
    return result;
}

// Fast Walsh-Hadamard Transform in-place on 32 float registers
// Equivalent to normalized Hadamard matrix multiply: y = (H/sqrt(32)) @ x
// where H is the Sylvester-order Hadamard matrix {+1,-1}^{32x32}
// 5 butterfly stages + normalization, all in registers — no shared memory needed
__device__ __forceinline__ void fwht_inplace_32(float* v) {
    #pragma unroll
    for (int step = 1; step < 32; step *= 2) {
        #pragma unroll
        for (int i = 0; i < 32; i += 2 * step) {
            #pragma unroll
            for (int j = i; j < i + step; j++) {
                float a = v[j], b = v[j + step];
                v[j] = a + b;
                v[j + step] = a - b;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 32; i++) v[i] *= 0.17677669529663689f; // 1/sqrt(32)
}

// Forward declarations — kernels defined later
__global__ void hadamard32_batched(const __half* X, __half* Y, int D);
__global__ void dequant_int4_kernel(
    const uint8_t* __restrict__ packed,  // [N, K/2]
    const __half* __restrict__ scales,   // [N, K/32, 2] interleaved (scale, zero)
    __half* __restrict__ out,            // [N, K]
    int N, int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int Kh = K / 2;
    if (idx >= N * Kh) return;
    const int n = idx / Kh;
    const int kh = idx % Kh;
    const uint8_t p = packed[idx];
    int lo = (p & 0xF);
    int hi = (p >> 4);
    const int block_idx = kh / 16;  // group-32: 16 bytes = 32 elements
    __half2 sz = *reinterpret_cast<const __half2*>(&scales[(n * (K / 32) + block_idx) * 2]);
    const float s = __half2float(sz.x);
    const float z = __half2float(sz.y);
    out[(int64_t)n * K + kh * 2]     = __float2half(((float)lo - z) * s);
    out[(int64_t)n * K + kh * 2 + 1] = __float2half(((float)hi - z) * s);
}

// ============================================================
// GEMV V3 — Vectorized loads (uint4 128-bit), INT4 decode
// Each warp handles BLOCK_N output rows. K dimension iterated in chunks of 32.
// Inner loop uses 128-bit loads for weights (16 bytes = 32 INT4 values at once)
// and __half2 loads for X (coalesced).
// ============================================================

template<int BLOCK_N, int GS = 32>
__global__ void gemv_warp(
    const __half* __restrict__ X,
    const uint8_t* __restrict__ W,     // [N, K/2]
    const __half* __restrict__ S,      // [N, K/GS, 2] interleaved (scale, zero_fp16)
    __half* __restrict__ Y,
    int N, int K
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / GS, nit = K / 32;
    float acc[BLOCK_N];
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i++) acc[i] = 0.0f;

    for (int it = lane; it < nit; it += 32) {
        // Load 32 X values via half2 (coalesced, 64 bytes per warp)
        float xv[32];
        #pragma unroll
        for (int i = 0; i < 32; i += 2) {
            __half2 xh = *reinterpret_cast<const __half2*>(&X[it * 32 + i]);
            xv[i] = __half2float(xh.x); xv[i+1] = __half2float(xh.y);
        }

        // Scale index: for GS=32 → it, for GS=64 → it/2, etc.
        const int scale_idx = it / (GS / 32);

        #pragma unroll
        for (int ni = 0; ni < BLOCK_N; ni++) {
            int n = n0 + ni;
            if (n >= N) continue;
            // Load interleaved (scale, zero) pair as half2
            __half2 sz = *reinterpret_cast<const __half2*>(&S[(n * nsc + scale_idx) * 2]);
            float scale = __half2float(sz.x);
            float zp = __half2float(sz.y);
            // Vectorized weight load: 16 bytes = 32 INT4 values
            const uint8_t* wr = W + n * Kh + it * 16;
            uint4 w128 = *reinterpret_cast<const uint4*>(wr);
            const uint8_t* wp = reinterpret_cast<const uint8_t*>(&w128);
            float dot = 0.0f;
            #pragma unroll
            for (int b = 0; b < 16; b++) {
                uint8_t p = wp[b];
                float lo = (float)(p & 0xF) - zp;
                float hi = (float)((p >> 4) & 0xF) - zp;
                dot += xv[b*2] * lo + xv[b*2+1] * hi;
            }
            acc[ni] += dot * scale;
        }
    }
    #pragma unroll
    for (int ni = 0; ni < BLOCK_N; ni++) {
        acc[ni] += __shfl_xor(acc[ni], 16); acc[ni] += __shfl_xor(acc[ni], 8);
        acc[ni] += __shfl_xor(acc[ni], 4);  acc[ni] += __shfl_xor(acc[ni], 2);
        acc[ni] += __shfl_xor(acc[ni], 1);
        if (lane == 0 && n0 + ni < N) Y[n0 + ni] = __float2half(acc[ni]);
    }
}

template<int BLOCK_N, int BS>
__global__ void gemv_block(
    const __half* __restrict__ X,
    const uint8_t* __restrict__ W, const __half* __restrict__ S,
    __half* __restrict__ Y,
    int N, int K
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int tid = threadIdx.x;
    const int Kh = K / 2, nsc = K / 32, nit = K / 32;
    float acc[BLOCK_N];
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i++) acc[i] = 0.0f;

    for (int it = tid; it < nit; it += BS) {
        float xv[32];
        #pragma unroll
        for (int i = 0; i < 32; i += 2) {
            __half2 xh = *reinterpret_cast<const __half2*>(&X[it * 32 + i]);
            xv[i] = __half2float(xh.x); xv[i+1] = __half2float(xh.y);
        }

        #pragma unroll
        for (int ni = 0; ni < BLOCK_N; ni++) {
            int n = n0 + ni; if (n >= N) continue;
            __half2 sz = *reinterpret_cast<const __half2*>(&S[(n * nsc + it) * 2]);
            float scale = __half2float(sz.x);
            float zp = __half2float(sz.y);
            const uint8_t* wr = W + n * Kh + it * 16;
            uint4 w128 = *reinterpret_cast<const uint4*>(wr);
            const uint8_t* wp = reinterpret_cast<const uint8_t*>(&w128);
            float dot = 0.0f;
            #pragma unroll
            for (int b = 0; b < 16; b++) { uint8_t p = wp[b]; float lo=(float)(p&0xF)-zp; float hi=(float)((p>>4)&0xF)-zp; dot += xv[b*2]*lo + xv[b*2+1]*hi; }
            acc[ni] += dot * scale;
        }
    }
    #pragma unroll
    for (int ni = 0; ni < BLOCK_N; ni++) {
        acc[ni] += __shfl_xor(acc[ni], 16); acc[ni] += __shfl_xor(acc[ni], 8);
        acc[ni] += __shfl_xor(acc[ni], 4);  acc[ni] += __shfl_xor(acc[ni], 2);
        acc[ni] += __shfl_xor(acc[ni], 1);
    }
    const int wid = tid / 32, lid = tid % 32, nw = BS / 32;
    __shared__ float sm[BLOCK_N * 32];
    if (lid == 0) { for (int ni = 0; ni < BLOCK_N; ni++) sm[ni * nw + wid] = acc[ni]; }
    __syncthreads();
    if (wid == 0 && lid < nw) {
        for (int ni = 0; ni < BLOCK_N; ni++) {
            float v = sm[ni * nw + lid];
            v += __shfl_xor(v, 16); v += __shfl_xor(v, 8);
            v += __shfl_xor(v, 4);  v += __shfl_xor(v, 2);
            v += __shfl_xor(v, 1);
            if (lid == 0 && n0 + ni < N) Y[n0 + ni] = __float2half(v);
        }
    }
}

static inline __attribute__((always_inline)) void launch_gemv(
    const __half* X, const uint8_t* W, const __half* S,
    __half* Y, int N, int K, hipStream_t st, int gs = 32) {
    // BLOCK_N=8 for large N, BLOCK_N=4 otherwise
    if (N >= 8192) {
        if (gs == 64) hipLaunchKernelGGL((gemv_warp<8, 64>), dim3((N+7)/8), dim3(32), 0, st, X, W, S, Y, N, K);
        else          hipLaunchKernelGGL((gemv_warp<8, 32>), dim3((N+7)/8), dim3(32), 0, st, X, W, S, Y, N, K);
    } else {
        if (gs == 64) hipLaunchKernelGGL((gemv_warp<4, 64>), dim3((N+3)/4), dim3(32), 0, st, X, W, S, Y, N, K);
        else          hipLaunchKernelGGL((gemv_warp<4, 32>), dim3((N+3)/4), dim3(32), 0, st, X, W, S, Y, N, K);
    }
}
template<int ROWS_PER_WARP>
__global__ __launch_bounds__(128, 4) void gemv_multiwave_rm_g64(
    const __half* __restrict__ X,
    const uint8_t* __restrict__ W,   // [N, K/2]
    const __half*  __restrict__ S,   // [K/64, N, 2] transposed — coalesced across N
    __half* __restrict__ Y,
    int N, int K
) {
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int n0      = (blockIdx.x * 4 + warp_id) * ROWS_PER_WARP;
    const int nit     = K / 32;
    float acc[ROWS_PER_WARP];
    #pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; i++) acc[i] = 0.0f;

    for (int it = lane; it < nit; it += 32) {
        if (n0 >= N) continue;
        const __half* Xit = X + it * 32;

        // Load weight + fused dot+x_sum in single pass (X loaded once from L0)
        const uint8_t* wr = W + (int64_t)n0 * (K / 2) + it * 16;
        uint4 w128 = *reinterpret_cast<const uint4*>(wr);
        const uint8_t* wp = reinterpret_cast<const uint8_t*>(&w128);
        float raw_dot = 0.0f, x_sum = 0.0f;
        #pragma unroll
        for (int b = 0; b < 16; b++) {
            uint8_t p = wp[b];
            float2 xf = __half22float2(*reinterpret_cast<const __half2*>(&Xit[b * 2]));
            raw_dot += xf.x * (float)(p & 0xF) + xf.y * (float)((p >> 4) & 0xF);
            x_sum += xf.x + xf.y;
        }

        // Row-major scale layout: S[N, K/64, 2] — coalesced across K groups within warp
        int g = it >> 1;
        __half2 sz = *reinterpret_cast<const __half2*>(&S[((int64_t)n0 * (nit >> 1) + g) * 2]);
        float sc = __half2float(sz.x);
        float zp = __half2float(sz.y);
        acc[0] += sc * (raw_dot - zp * x_sum);
    }
    acc[0] += __shfl_xor(acc[0], 16); acc[0] += __shfl_xor(acc[0], 8);
    acc[0] += __shfl_xor(acc[0], 4);  acc[0] += __shfl_xor(acc[0], 2);
    acc[0] += __shfl_xor(acc[0], 1);
    if (lane == 0 && n0 < N) Y[n0] = __float2half(acc[0]);
}

// Launch multi-warp GEMV: 4 warps/block × 2 rows/warp → 8 rows/block
static inline __attribute__((always_inline)) void launch_gemv_mw_rm_g64_nt(
    const __half* X, const uint8_t* W, const __half* S,
    __half* Y, int N, int K, hipStream_t st) {
    int nblocks = (N + 7) / 8;
    hipLaunchKernelGGL((gemv_multiwave_rm_g64<2>), dim3(nblocks), dim3(128), 0, st, X, W, S, Y, N, K);
}

// Launch multi-warp GEMV: 4 warps/block × 1 row/warp → 4 rows/block
static inline __attribute__((always_inline)) void launch_gemv_mw_rm_g64(
    const __half* X, const uint8_t* W, const __half* S,
    __half* Y, int N, int K, hipStream_t st) {
    int nblocks = (N + 3) / 4;
    hipLaunchKernelGGL((gemv_multiwave_rm_g64<1>), dim3(nblocks), dim3(128), 0, st, X, W, S, Y, N, K);
}

// ============================================================
// Split-K GEMV: splits K dimension across blockIdx.y to increase block count.
// For O GEMV (N=2048, K=4096): original 512 blocks (8/SIMD) → split-K=2: 1024 (16/SIMD).
// More blocks = more memory requests in flight = better DRAM BW utilization.
// ============================================================
template<int ROWS_PER_WARP>
__global__ __launch_bounds__(128)
void gemv_mw_rm_g64_splitk(
    const __half* __restrict__ X,
    const uint8_t* __restrict__ W,   // [N, K/2]
    const __half*  __restrict__ S,   // [K/64, N, 2] transposed — coalesced across N
    float* __restrict__ Y_parts,     // [N] float accumulator (atomicAdd)
    int N, int K, int n_splits
) {
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int n0      = (blockIdx.x * 4 + warp_id) * ROWS_PER_WARP;
    const int split_id = blockIdx.y;

    const int nit     = K / 32;
    const int chunk   = (nit + n_splits - 1) / n_splits;
    const int it_start = split_id * chunk;
    const int it_end   = min(it_start + chunk, nit);
    const int ngrp    = K / 64;

    float acc = 0.0f;

    for (int it = it_start + lane; it < it_end; it += 32) {
        if (n0 >= N) continue;
        const __half* Xit = X + it * 32;

        const uint8_t* wr = W + (int64_t)n0 * (K / 2) + it * 16;
        uint4 w128 = *reinterpret_cast<const uint4*>(wr);
        const uint8_t* wp = reinterpret_cast<const uint8_t*>(&w128);
        float raw_dot = 0.0f, x_sum = 0.0f;
        #pragma unroll
        for (int b = 0; b < 16; b++) {
            uint8_t p = wp[b];
            float2 xf = __half22float2(*reinterpret_cast<const __half2*>(&Xit[b * 2]));
            raw_dot += xf.x * (float)(p & 0xF) + xf.y * (float)((p >> 4) & 0xF);
            x_sum += xf.x + xf.y;
        }

        int g = it >> 1;
        __half2 sz = *reinterpret_cast<const __half2*>(&S[((int64_t)n0 * (K / 64) + g) * 2]);
        float sc = __half2float(sz.x);
        float zp = __half2float(sz.y);
        acc += sc * (raw_dot - zp * x_sum);
    }
    acc += __shfl_xor(acc, 16); acc += __shfl_xor(acc, 8);
    acc += __shfl_xor(acc, 4);  acc += __shfl_xor(acc, 2);
    acc += __shfl_xor(acc, 1);
    if (lane == 0 && n0 < N) atomicAdd(&Y_parts[n0], acc);
}

// Convert float accumulator to half output
__global__ void splitk_f2h(const float* __restrict__ in, __half* __restrict__ out, int N) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < N) out[i] = __float2half(in[i]);
}

// Launch split-K GEMV: memset workspace → split-K kernel → float-to-half convert
static inline __attribute__((always_inline)) void launch_gemv_mw_rm_g64_splitk(
    const __half* X, const uint8_t* W, const __half* S,
    __half* Y, float* workspace, int N, int K, int n_splits, hipStream_t st) {
    hipMemsetAsync(workspace, 0, N * sizeof(float), st);
    int nblocks = (N + 3) / 4;
    hipLaunchKernelGGL((gemv_mw_rm_g64_splitk<1>), dim3(nblocks, n_splits), dim3(128), 0, st,
        X, W, S, workspace, N, K, n_splits);
    hipLaunchKernelGGL(splitk_f2h, dim3((N + 255) / 256), dim3(256), 0, st,
        workspace, Y, N);
}

template<int BLOCK_N, int GS=32>
__global__ void gemv_warp_silu_had(
    const __half* __restrict__ Gate,
    const __half* __restrict__ Up,
    const uint8_t* __restrict__ W,
    const __half* __restrict__ S,      // [N, K/GS, 2] interleaved (scale, zero)
    __half* __restrict__ Y,
    int N, int K
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / GS, nit = K / 32;
    const int gs_ratio = GS / 32;
    float acc[BLOCK_N];
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i++) acc[i] = 0.0f;

    for (int it = lane; it < nit; it += 32) {
        float xv[32];
        #pragma unroll
        for (int i = 0; i < 32; i += 2) {
            __half2 gh = *reinterpret_cast<const __half2*>(&Gate[it * 32 + i]);
            __half2 uh = *reinterpret_cast<const __half2*>(&Up[it * 32 + i]);
            float g0 = __half2float(gh.x), g1 = __half2float(gh.y);
            float u0 = __half2float(uh.x), u1 = __half2float(uh.y);
            xv[i]   = g0 / (1.0f + expf(-g0)) * u0;
            xv[i+1] = g1 / (1.0f + expf(-g1)) * u1;
        }
        fwht_inplace_32(xv);

        const int scale_idx = it / gs_ratio;

        #pragma unroll
        for (int ni = 0; ni < BLOCK_N; ni++) {
            int n = n0 + ni;
            if (n >= N) continue;
            __half2 sz = *reinterpret_cast<const __half2*>(&S[(n * nsc + scale_idx) * 2]);
            float scale = __half2float(sz.x);
            float zp = __half2float(sz.y);
            const uint8_t* wr = W + n * Kh + it * 16;
            uint4 w128 = *reinterpret_cast<const uint4*>(wr);
            const uint8_t* wp = reinterpret_cast<const uint8_t*>(&w128);
            float dot = 0.0f;
            #pragma unroll
            for (int b = 0; b < 16; b++) {
                uint8_t p = wp[b];
                float lo = (float)(p & 0xF) - zp;
                float hi = (float)((p >> 4) & 0xF) - zp;
                dot += xv[b*2] * lo + xv[b*2+1] * hi;
            }
            acc[ni] += dot * scale;
        }
    }
    #pragma unroll
    for (int ni = 0; ni < BLOCK_N; ni++) {
        acc[ni] += __shfl_xor(acc[ni], 16); acc[ni] += __shfl_xor(acc[ni], 8);
        acc[ni] += __shfl_xor(acc[ni], 4);  acc[ni] += __shfl_xor(acc[ni], 2);
        acc[ni] += __shfl_xor(acc[ni], 1);
        if (lane == 0 && n0 + ni < N) Y[n0 + ni] = __float2half(acc[ni]);
    }
}

// launch_gemv_had, launch_fwht_then_gemv, launch_gemv_res_norm_had removed (unused in production)

static inline __attribute__((always_inline)) void launch_gemv_silu_had(
    const __half* Gate, const __half* Up,
    const uint8_t* W, const __half* S,
    __half* Y, int N, int K, hipStream_t st, int gs = 32) {
    if (gs == 64) {
        hipLaunchKernelGGL((gemv_warp_silu_had<4, 64>), dim3((N+3)/4), dim3(32), 0, st,
            Gate, Up, W, S, Y, N, K);
    } else {
        hipLaunchKernelGGL((gemv_warp_silu_had<4, 32>), dim3((N+3)/4), dim3(32), 0, st,
            Gate, Up, W, S, Y, N, K);
    }
}
__global__ void hadamard_rotate_batched(
    const __half* __restrict__ X_in,
    __half* __restrict__ X_out,
    const __half* __restrict__ H,  // unused (API compat)
    int K
) {
    const int bid = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int base = row * K + bid * 32;

    float val = __half2float(X_in[base + tid]);
    #pragma unroll
    for (int step = 1; step < 32; step *= 2) {
        float other = __shfl_xor(val, step);
        if (tid & step) val = other - val;
        else val = val + other;
    }
    val *= 0.17677669529663689f; // 1/sqrt(32)
    X_out[base + tid] = __float2half(val);
}

static inline void launch_hadamard_batched(
    const __half* X_in, __half* X_out, const __half* H,
    int M, int K, hipStream_t st) {
    hipLaunchKernelGGL(hadamard_rotate_batched, dim3(K / 32, M), dim3(32), 0, st, X_in, X_out, H, K);
}
__global__ void fused_res_norm_fwht(
    const __half* __restrict__ A, const __half* __restrict__ B,
    const __half* __restrict__ NormW,
    __half* __restrict__ Y, __half* __restrict__ ResOut,
    int D, float eps
) {
    const int tid = threadIdx.x;
    const int nwarps = blockDim.x / 32;
    const int wid = tid / 32;
    const int lane = tid % 32;

    // Phase 1: residual + sum-of-squares
    float ss = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float r = __half2float(A[i]) + __half2float(B[i]);
        ResOut[i] = __float2half(r);
        ss += r * r;
    }
    // Warp reduce
    ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
    ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
    // Cross-warp reduce (supports up to 32 warps)
    __shared__ float ws[32];
    if (lane == 0) ws[wid] = ss;
    __syncthreads();
    float rrms;
    if (wid == 0) {
        ss = (lane < nwarps) ? ws[lane] : 0.0f;
        ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
        ss += __shfl_xor(ss, 4); ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
        if (lane == 0) ws[0] = rsqrtf(ss / (float)D + eps);
    }
    __syncthreads();
    rrms = ws[0];

    // Phase 2: norm + FWHT (each warp handles multiple 32-element blocks)
    int num_blocks = D / 32;
    for (int blk = wid; blk < num_blocks; blk += nwarps) {
        int base = blk * 32;
        float normed = __half2float(ResOut[base + lane]) * rrms * __half2float(NormW[base + lane]);
        #pragma unroll
        for (int step = 1; step < 32; step *= 2) {
            float other = __shfl_xor(normed, step);
            if (lane & step) normed = other - normed;
            else normed = normed + other;
        }
        normed *= 0.17677669529663689f;
        Y[base + lane] = __float2half(normed);
    }
}

// Fused single-block: rrms + norm + FWHT (no residual add)
__global__ void fused_rmsnorm_fwht(
    const __half* __restrict__ X, const __half* __restrict__ NormW,
    __half* __restrict__ Y, int D, float eps
) {
    const int tid = threadIdx.x;
    const int nwarps = blockDim.x / 32;
    const int wid = tid / 32;
    const int lane = tid % 32;

    float ss = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float x = __half2float(X[i]);
        ss += x * x;
    }
    ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
    ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
    __shared__ float ws[32];
    if (lane == 0) ws[wid] = ss;
    __syncthreads();
    float rrms;
    if (wid == 0) {
        ss = (lane < nwarps) ? ws[lane] : 0.0f;
        ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
        ss += __shfl_xor(ss, 4); ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
        if (lane == 0) ws[0] = rsqrtf(ss / (float)D + eps);
    }
    __syncthreads();
    rrms = ws[0];

    int num_blocks = D / 32;
    for (int blk = wid; blk < num_blocks; blk += nwarps) {
        int base = blk * 32;
        float normed = __half2float(X[base + lane]) * rrms * __half2float(NormW[base + lane]);
        #pragma unroll
        for (int step = 1; step < 32; step *= 2) {
            float other = __shfl_xor(normed, step);
            if (lane & step) normed = other - normed;
            else normed = normed + other;
        }
        normed *= 0.17677669529663689f;
        Y[base + lane] = __float2half(normed);
    }
}

// Keep old kernels as fallback (still used in some paths)
__global__ void res_norm_fwht_parallel(
    const __half* __restrict__ A, const __half* __restrict__ B,
    const __half* __restrict__ NormW,
    __half* __restrict__ Y, __half* __restrict__ ResOut,
    int D, float eps
) {
    const int hb = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = hb * 32;
    float ss = 0.0f;
    for (int i = tid; i < D; i += 32) {
        float r = __half2float(A[i]) + __half2float(B[i]);
        ss += r * r;
    }
    ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
    ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
    float rrms = rsqrtf(ss / (float)D + eps);
    float ri = __half2float(A[base + tid]) + __half2float(B[base + tid]);
    ResOut[base + tid] = __float2half(ri);
    float normed = ri * rrms * __half2float(NormW[base + tid]);
    #pragma unroll
    for (int step = 1; step < 32; step *= 2) {
        float other = __shfl_xor(normed, step);
        if (tid & step) normed = other - normed;
        else normed = normed + other;
    }
    normed *= 0.17677669529663689f;
    Y[base + tid] = __float2half(normed);
}

__global__ void rmsnorm_fwht_parallel(
    const __half* __restrict__ X, const __half* __restrict__ NormW,
    __half* __restrict__ Y,
    int D, float eps
) {
    const int hb = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = hb * 32;
    float ss = 0.0f;
    for (int i = tid; i < D; i += 32) {
        float x = __half2float(X[i]);
        ss += x * x;
    }
    ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
    ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
    float rrms = rsqrtf(ss / (float)D + eps);
    float normed = __half2float(X[base + tid]) * rrms * __half2float(NormW[base + tid]);
    #pragma unroll
    for (int step = 1; step < 32; step *= 2) {
        float other = __shfl_xor(normed, step);
        if (tid & step) normed = other - normed;
        else normed = normed + other;
    }
    normed *= 0.17677669529663689f;
    Y[base + tid] = __float2half(normed);
}

// Single-block launcher: residual + rrms + norm + FWHT
static inline void launch_res_norm_fwht(
    const __half* A, const __half* B, const __half* NormW,
    __half* Y, __half* ResOut, int D, float eps, hipStream_t st) {
    hipLaunchKernelGGL(fused_res_norm_fwht, dim3(1), dim3(1024), 0, st,
        A, B, NormW, Y, ResOut, D, eps);
}

// Single-block launcher: rrms + norm + FWHT (no residual add)
static inline void launch_rmsnorm_fwht(
    const __half* X, const __half* NormW,
    __half* Y, int D, float eps, hipStream_t st) {
    hipLaunchKernelGGL(fused_rmsnorm_fwht, dim3(1), dim3(1024), 0, st,
        X, NormW, Y, D, eps);
}
// ============================================================
// Flash Decode Attention — Split-K with GQA grouping
// ============================================================
// Phase 1: Each block processes a chunk of K/V positions for all Q heads
//   sharing one KV head. Multiple blocks per KV head for parallelism.
//   Grid: (num_kv_heads, num_splits), Block: (32, gqa_ratio)
//   Each warp (threadIdx.y) handles one Q head, all warps share K/V via L2.
// Dynamic BLOCK_S: choose larger splits for larger contexts to reduce kernel count
// BLOCK_S=32 kept as minimum for buffer allocation sizing
#define FLASH_DECODE_BLOCK_S 32

template <int BLOCK_S>
__global__ void flash_decode_partial(
    const __half* __restrict__ Q,         // [H*D] flat
    const uint8_t* __restrict__ K_cache,  // [Hk, max_seq, D] FP8 E4M3
    const uint8_t* __restrict__ V_cache,  // [Hk, max_seq, D] FP8 E4M3
    float* __restrict__ partial_max,      // [num_splits, H]
    float* __restrict__ partial_sum,      // [num_splits, H]
    float* __restrict__ partial_out,      // [num_splits, H, D]
    int H, int Hk, int max_seq, int seq_len, int D,
    int num_splits
) {
    const int kh = blockIdx.x;           // KV head index
    const int split_id = blockIdx.y;     // Split index
    const int warp_id = threadIdx.y;     // Q head within GQA group
    const int lane = threadIdx.x;        // 0..31
    const int gqa_ratio = H / Hk;

    if (warp_id >= gqa_ratio) return;
    const int qh = kh * gqa_ratio + warp_id;
    const float scale = rsqrtf((float)D);

    // Load Q into registers (D=128, 4 elements per lane)
    const __half* q_row = Q + qh * D;
    float q_reg[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        q_reg[i] = __half2float(q_row[lane + i * 32]);

    // Pack Q as FP8 E4M3 for hardware dot4 (done once, reused for all K positions)
    uint32_t q_packed_fp8 = (uint32_t)f32_to_fp8e4m3(q_reg[0])
                          | ((uint32_t)f32_to_fp8e4m3(q_reg[1]) << 8)
                          | ((uint32_t)f32_to_fp8e4m3(q_reg[2]) << 16)
                          | ((uint32_t)f32_to_fp8e4m3(q_reg[3]) << 24);

    // Range for this split
    const int s_start = split_id * BLOCK_S;
    const int s_end = min(s_start + BLOCK_S, seq_len);

    const uint8_t* k_base = K_cache + (long)kh * max_seq * D;
    const uint8_t* v_base = V_cache + (long)kh * max_seq * D;

    // Online softmax accumulators for this chunk
    float local_max = -1e30f;
    float local_sum = 0.0f;
    float out_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = s_start; s < s_end; s++) {
        // Q·K dot product using hardware v_dot4_f32_fp8_fp8
        // Packs 4 strided K bytes into uint32, then 1 instruction computes dot product
        const uint8_t* k_pos = k_base + (long)s * D + lane;
        uint32_t k_packed = (uint32_t)k_pos[0]
                          | ((uint32_t)k_pos[32] << 8)
                          | ((uint32_t)k_pos[64] << 16)
                          | ((uint32_t)k_pos[96] << 24);
        float dot = __builtin_amdgcn_dot4_f32_fp8_fp8(q_packed_fp8, k_packed, 0.0f);

        // Warp reduction
        dot += __shfl_xor(dot, 16);
        dot += __shfl_xor(dot, 8);
        dot += __shfl_xor(dot, 4);
        dot += __shfl_xor(dot, 2);
        dot += __shfl_xor(dot, 1);
        dot *= scale;

        // Online softmax update
        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float exp_score = expf(dot - new_max);
        local_sum = local_sum * correction + exp_score;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            out_reg[i] = out_reg[i] * correction +
                         exp_score * fp8e4m3_to_f32(v_base[(long)s * D + lane + i * 32]);
        local_max = new_max;
    }

    // Write partial results
    const long out_offset = (long)split_id * H * D + (long)qh * D;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        partial_out[out_offset + lane + i * 32] = out_reg[i];

    if (lane == 0) {
        partial_max[split_id * H + qh] = local_max;
        partial_sum[split_id * H + qh] = local_sum;
    }
}

// Phase 2: Reduce partial results across splits.
// Grid: (H), Block: (32)
// Parallel reduce: 4 warps per block, each warp handles num_splits/4 splits.
// Then inter-warp reduce via shared memory. 4× faster than serial.
__global__ __launch_bounds__(128)
void flash_decode_reduce(
    const float* __restrict__ partial_max,   // [num_splits, H]
    const float* __restrict__ partial_sum,   // [num_splits, H]
    const float* __restrict__ partial_out,   // [num_splits, H, D]
    __half* __restrict__ O,                  // [H*D]
    int H, int D, int num_splits
) {
    const int qh = blockIdx.x;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int n_warps = 4;

    // Each warp processes a chunk of splits
    int sp_start = (num_splits * warp_id) / n_warps;
    int sp_end = (num_splits * (warp_id + 1)) / n_warps;

    // Phase 1: per-warp local max
    float local_max = -1e30f;
    for (int sp = sp_start; sp < sp_end; sp++)
        local_max = fmaxf(local_max, partial_max[sp * H + qh]);

    // Inter-warp max reduce via shared memory
    __shared__ float smem_max[4];
    __shared__ float smem_sum[4];
    __shared__ float smem_out[4][128];  // 4 warps × D=128

    if (lane == 0) smem_max[warp_id] = local_max;
    __syncthreads();

    float global_max = fmaxf(fmaxf(smem_max[0], smem_max[1]),
                             fmaxf(smem_max[2], smem_max[3]));

    // Phase 2: per-warp accumulate with correction
    float warp_sum = 0.0f;
    float out_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int sp = sp_start; sp < sp_end; sp++) {
        float correction = expf(partial_max[sp * H + qh] - global_max);
        warp_sum += partial_sum[sp * H + qh] * correction;
        const long off = (long)sp * H * D + (long)qh * D;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            out_reg[i] += partial_out[off + lane + i * 32] * correction;
    }

    // Store per-warp results to shared memory
    if (lane == 0) smem_sum[warp_id] = warp_sum;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        smem_out[warp_id][lane + i * 32] = out_reg[i];
    __syncthreads();

    // Phase 3: warp 0 combines all 4 warps and writes output + FWHT
    if (warp_id == 0) {
        float total_sum = smem_sum[0] + smem_sum[1] + smem_sum[2] + smem_sum[3];
        float inv = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float val = (smem_out[0][lane + i*32] + smem_out[1][lane + i*32] +
                        smem_out[2][lane + i*32] + smem_out[3][lane + i*32]) * inv;
            // FWHT
            #pragma unroll
            for (int step = 1; step < 32; step *= 2) {
                float other = __shfl_xor(val, step);
                if (lane & step) val = other - val;
                else val = val + other;
            }
            O[qh * D + lane + i * 32] = __float2half(val * 0.17677669529663689f);
        }
    }
}
__global__ void rmsnorm_batched(const __half* X, const __half* W, __half* Y, int D, float eps) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x, BS = blockDim.x;
    const __half* x_row = X + (int64_t)row * D;
    __half* y_row = Y + (int64_t)row * D;
    float sq = 0.0f;
    for (int i = tid; i < D; i += BS) { float x = __half2float(x_row[i]); smem[i] = x; sq += x * x; }
    sq += __shfl_xor(sq, 16); sq += __shfl_xor(sq, 8);
    sq += __shfl_xor(sq, 4);  sq += __shfl_xor(sq, 2); sq += __shfl_xor(sq, 1);
    __shared__ float ws[32];
    int wid = tid / 32, lid = tid % 32, nw = BS / 32;
    if (lid == 0) ws[wid] = sq;
    __syncthreads();
    if (wid == 0 && lid < nw) {
        float v = ws[lid];
        v += __shfl_xor(v, 16); v += __shfl_xor(v, 8);
        v += __shfl_xor(v, 4);  v += __shfl_xor(v, 2); v += __shfl_xor(v, 1);
        if (lid == 0) ws[0] = v;
    }
    __syncthreads();
    float rrms = rsqrtf(ws[0] / (float)D + eps);
    for (int i = tid; i < D; i += BS) y_row[i] = __float2half(smem[i] * rrms * __half2float(W[i]));
}

// Fused RMSNorm + Walsh-Hadamard(32) transform.
// Each block handles one row (length D). D must be divisible by 32.
// Uses __shfl_xor butterfly for the Hadamard (5 stages, scale 1/sqrt(32)).
// Grid: (M), Block: (BS) where BS must be multiple of 32.
// Shared mem: D * sizeof(float) — same as rmsnorm_batched.
__global__ void rmsnorm_hadamard_batched(const __half* X, const __half* W, __half* Y,
                                          __half* Y_pre, // if non-null, also stores pre-hadamard normed output
                                          int D, float eps) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x, BS = blockDim.x;
    const __half* x_row = X + (int64_t)row * D;
    __half* y_row = Y + (int64_t)row * D;

    // Pass 1: load + compute sum of squares
    float sq = 0.0f;
    for (int i = tid; i < D; i += BS) {
        float x = __half2float(x_row[i]);
        smem[i] = x;
        sq += x * x;
    }
    // Warp reduce
    sq += __shfl_xor(sq, 16); sq += __shfl_xor(sq, 8);
    sq += __shfl_xor(sq, 4);  sq += __shfl_xor(sq, 2); sq += __shfl_xor(sq, 1);
    __shared__ float ws[32];
    int wid = tid / 32, lid = tid % 32, nw = BS / 32;
    if (lid == 0) ws[wid] = sq;
    __syncthreads();
    if (wid == 0 && lid < nw) {
        float v = ws[lid];
        v += __shfl_xor(v, 16); v += __shfl_xor(v, 8);
        v += __shfl_xor(v, 4);  v += __shfl_xor(v, 2); v += __shfl_xor(v, 1);
        if (lid == 0) ws[0] = v;
    }
    __syncthreads();
    float rrms = rsqrtf(ws[0] / (float)D + eps);

    // Pass 2: apply weight + Hadamard butterfly
    // Each thread processes elements at positions tid, tid+BS, tid+2*BS, ...
    // The Hadamard operates on groups of 32 consecutive elements.
    // We need threads within same warp to handle consecutive elements in same group.
    // With BS threads and D elements, each thread handles D/BS elements.
    // Since BS is a multiple of 32, consecutive threads in a warp handle consecutive indices.
    const float had_scale = 0.17677669529663687f; // 1/sqrt(32)
    for (int base = wid * 32; base < D; base += BS) {
        // Thread 'lid' handles element base+lid in this group
        int idx = base + lid;
        float val = smem[idx] * rrms * (1.0f + __half2float(W[idx]));
        if (Y_pre) Y_pre[(int64_t)row * D + idx] = __float2half(val);
        // 5-stage butterfly Hadamard
        #pragma unroll
        for (int s = 0; s < 5; s++) {
            float other = __shfl_xor(val, 1 << s);
            val = (lid & (1 << s)) ? (other - val) : (val + other);
        }
        y_row[idx] = __float2half(val * had_scale);
    }
}

// Standalone Walsh-Hadamard(32) butterfly transform.
// No shared memory needed — each warp reads 32 consecutive elements,
// does 5-stage butterfly via __shfl_xor, writes result.
// Grid: (M, ceil(D/BS)), Block: (BS, multiple of 32).
__global__ void hadamard32_batched(const __half* X, __half* Y, int D) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x, BS = blockDim.x;
    const int base_col = blockIdx.y * BS;
    const __half* x_row = X + (int64_t)row * D;
    __half* y_row = Y + (int64_t)row * D;
    int lid = tid % 32;

    int col = base_col + tid;
    if (col >= D) return;

    float val = __half2float(x_row[col]);
    const float had_scale = 0.17677669529663687f;
    #pragma unroll
    for (int s = 0; s < 5; s++) {
        float other = __shfl_xor(val, 1 << s);
        val = (lid & (1 << s)) ? (other - val) : (val + other);
    }
    y_row[col] = __float2half(val * had_scale);
}

// Batched residual + RMSNorm
__global__ void fused_res_norm_batched(const __half* X, const __half* Res, const __half* W,
                                        __half* Y, __half* ResOut, int D, float eps) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x, BS = blockDim.x;
    const __half* x_row = X + (int64_t)row * D;
    const __half* r_row = Res + (int64_t)row * D;
    __half* y_row = Y + (int64_t)row * D;
    __half* ro_row = ResOut + (int64_t)row * D;
    float sq = 0.0f;
    for (int i = tid; i < D; i += BS) {
        float xi = __half2float(x_row[i]) + __half2float(r_row[i]);
        smem[i] = xi; sq += xi * xi;
    }
    sq += __shfl_xor(sq, 16); sq += __shfl_xor(sq, 8);
    sq += __shfl_xor(sq, 4);  sq += __shfl_xor(sq, 2); sq += __shfl_xor(sq, 1);
    __shared__ float ws[32];
    int wid = tid / 32, lid = tid % 32, nw = BS / 32;
    if (lid == 0) ws[wid] = sq;
    __syncthreads();
    if (wid == 0 && lid < nw) {
        float v = ws[lid];
        v += __shfl_xor(v, 16); v += __shfl_xor(v, 8);
        v += __shfl_xor(v, 4);  v += __shfl_xor(v, 2); v += __shfl_xor(v, 1);
        if (lid == 0) ws[0] = v;
    }
    __syncthreads();
    float rrms = rsqrtf(ws[0] / (float)D + eps);
    for (int i = tid; i < D; i += BS) {
        float xi = smem[i];
        y_row[i] = __float2half(xi * rrms * __half2float(W[i]));
        ro_row[i] = __float2half(xi);
    }
}

// Batched head RMSNorm + RoPE for prefill (M positions)
// Grid: (H + Hk, M)
// QKV buffer is [M, qkv_dim] where qkv_dim = H*D + 2*Hk*D
__global__ void head_rmsnorm_rope_prefill(
    __half* QKV,                           // [M, qkv_dim] in-place
    const __half* QW, const __half* KW,
    const __half* COS, const __half* SIN,  // [max_seq, D/2]
    int M, int H, int Hk, int D, float eps, int cos_stride, int qkv_stride
) {
    int pid = blockIdx.x;  // head index
    int m = blockIdx.y;    // position index
    int tid = threadIdx.x;
    int half_D = D / 2;
    int q_dim = H * D;
    int kv_dim_total = Hk * D;

    const __half* cos_m = COS + m * cos_stride;
    const __half* sin_m = SIN + m * cos_stride;

    if (pid < H) {
        // Q head: in-place RMSNorm + RoPE
        __half* src = QKV + (int64_t)m * qkv_stride + pid * D;
        float sq = 0.0f;
        for (int i = tid; i < D; i += 32) {
            float v = __half2float(src[i]); sq += v * v;
        }
        sq += __shfl_xor(sq, 16); sq += __shfl_xor(sq, 8);
        sq += __shfl_xor(sq, 4);  sq += __shfl_xor(sq, 2); sq += __shfl_xor(sq, 1);
        float rrms = rsqrtf(sq / (float)D + eps);
        for (int i = tid; i < half_D; i += 32) {
            float ne = __half2float(src[i]) * rrms * __half2float(QW[i]);
            float no = __half2float(src[i + half_D]) * rrms * __half2float(QW[i + half_D]);
            float c = __half2float(cos_m[i]), s = __half2float(sin_m[i]);
            src[i] = __float2half(ne * c - no * s);
            src[i + half_D] = __float2half(no * c + ne * s);
        }
    }
    int kid = pid - H;
    if (kid >= 0 && kid < Hk) {
        // K head: in-place RMSNorm + RoPE
        __half* src = QKV + (int64_t)m * qkv_stride + q_dim + kid * D;
        float sq = 0.0f;
        for (int i = tid; i < D; i += 32) {
            float v = __half2float(src[i]); sq += v * v;
        }
        sq += __shfl_xor(sq, 16); sq += __shfl_xor(sq, 8);
        sq += __shfl_xor(sq, 4);  sq += __shfl_xor(sq, 2); sq += __shfl_xor(sq, 1);
        float rrms = rsqrtf(sq / (float)D + eps);
        for (int i = tid; i < half_D; i += 32) {
            float ne = __half2float(src[i]) * rrms * __half2float(KW[i]);
            float no = __half2float(src[i + half_D]) * rrms * __half2float(KW[i + half_D]);
            float c = __half2float(cos_m[i]), s = __half2float(sin_m[i]);
            src[i] = __float2half(ne * c - no * s);
            src[i + half_D] = __float2half(no * c + ne * s);
        }
    }
}

// Batched SiLU*up: Y_row[i] = SiLU(G_row[i]) * U_row[i]
// G, U have in_stride per row; Y has out_stride per row
__global__ void silu_mul_batched(const __half* G, const __half* U, __half* Y,
                                  int N, int in_stride, int out_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (idx < N) {
        int in_off = (int64_t)row * in_stride + idx;
        int out_off = (int64_t)row * out_stride + idx;
        float g = __half2float(G[in_off]);
        Y[out_off] = __float2half(g / (1.0f + expf(-g)) * __half2float(U[in_off]));
    }
}
template<int BN, int BM>
__global__ __launch_bounds__(32)
void gemv_w4a4_udot8_v3(
    const uint8_t* __restrict__ X_q,       // [total, K/2] packed uint4
    const __half*  __restrict__ X_sc,      // [total, K/32, 2]
    const uint8_t* __restrict__ W_base,    // [E, N, K/2] packed INT4
    const __half*  __restrict__ S_base,    // [E, K/64, N, 2]
    __half*        __restrict__ Y_all,     // [total, N]
    const int*     __restrict__ expert_ids,
    const int*     __restrict__ offsets,
    const int*     __restrict__ counts,
    int64_t packed_stride, int64_t scales_stride,
    int N, int K
) {
    const int eidx = blockIdx.y;
    const int M_j  = counts[eidx];
    if (M_j == 0) return;

    const int ej    = expert_ids[eidx];
    const int x_off = offsets[eidx];
    const uint8_t* W  = W_base + (int64_t)ej * packed_stride;
    const __half*  S  = S_base + (int64_t)ej * scales_stride;
    const uint8_t* X  = X_q  + (int64_t)x_off * (K / 2);
    const __half*  XS = X_sc + (int64_t)x_off * (K / 32) * 2;
    __half*        Y  = Y_all + (int64_t)x_off * N;

    const int n0   = blockIdx.x * BN;
    const int lane = threadIdx.x;
    const int nit  = K / 32;

    // Process tokens in tiles of BM
    for (int m0 = 0; m0 < M_j; m0 += BM) {
        const int m_end = (m0 + BM <= M_j) ? BM : (M_j - m0);

        float acc[BM][BN];
        #pragma unroll
        for (int dm = 0; dm < BM; dm++)
            #pragma unroll
            for (int ni = 0; ni < BN; ni++)
                acc[dm][ni] = 0.0f;

        // K-outer loop: load W once, process all tokens in M-tile
        for (int it = lane; it < nit; it += 32) {
            // Load weight scales (same for all tokens) — 64-element groups
            constexpr int SZ = (BN + 3) / 4;
            __half sz[SZ * 8];
            #pragma unroll
            for (int sl = 0; sl < SZ; sl++)
                *reinterpret_cast<uint4*>(&sz[sl * 8]) =
                    *reinterpret_cast<const uint4*>(&S[((it >> 1) * N + n0 + sl * 4) * 2]);

            // Load W for BN rows at this K-chunk — reused across all tokens
            uint4 w128[BN];
            uint32_t w_sum_u[BN];
            const uint32_t ones = 0x11111111u;

            #pragma unroll
            for (int ni = 0; ni < BN; ni++) {
                if (n0 + ni < N) {
                    w128[ni] = *reinterpret_cast<const uint4*>(
                        W + (int64_t)(n0 + ni) * (K / 2) + it * 16);
                    const uint32_t* wp = reinterpret_cast<const uint32_t*>(&w128[ni]);
                    w_sum_u[ni] = 0;
                    #pragma unroll
                    for (int i = 0; i < 4; i++)
                        w_sum_u[ni] = __builtin_amdgcn_udot8(wp[i], ones, w_sum_u[ni], false);
                }
            }

            // M-inner loop: reuse W from registers
            #pragma unroll
            for (int dm = 0; dm < BM; dm++) {
                if (dm >= m_end) break;
                int m = m0 + dm;

                uint4 x128 = *reinterpret_cast<const uint4*>(X + m * (K / 2) + it * 16);
                const uint32_t* xp32 = reinterpret_cast<const uint32_t*>(&x128);

                uint32_t x_sum_u = 0;
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    x_sum_u = __builtin_amdgcn_udot8(xp32[i], ones, x_sum_u, false);
                float x_sum = (float)x_sum_u;

                float a_sc = __half2float(XS[(m * (K / 32) + it) * 2]);
                float a_mn = __half2float(XS[(m * (K / 32) + it) * 2 + 1]);
                float xz_term = a_sc * x_sum + a_mn * 32.0f;

                #pragma unroll
                for (int ni = 0; ni < BN; ni++) {
                    if (n0 + ni >= N) continue;
                    float w_sc = __half2float(sz[ni * 2]);
                    float w_zp = __half2float(sz[ni * 2 + 1]);

                    const uint32_t* wp32 = reinterpret_cast<const uint32_t*>(&w128[ni]);
                    uint32_t raw_u = 0;
                    #pragma unroll
                    for (int i = 0; i < 4; i++)
                        raw_u = __builtin_amdgcn_udot8(wp32[i], xp32[i], raw_u, false);

                    acc[dm][ni] += w_sc * (a_sc * (float)raw_u + a_mn * (float)w_sum_u[ni]
                                   - w_zp * xz_term);
                }
            }
        }

        // Warp reduction and store for each token in tile
        #pragma unroll
        for (int dm = 0; dm < BM; dm++) {
            if (dm >= m_end) break;
            int m = m0 + dm;
            #pragma unroll
            for (int ni = 0; ni < BN; ni++) {
                if (n0 + ni >= N) continue;
                acc[dm][ni] += __shfl_xor(acc[dm][ni], 16);
                acc[dm][ni] += __shfl_xor(acc[dm][ni], 8);
                acc[dm][ni] += __shfl_xor(acc[dm][ni], 4);
                acc[dm][ni] += __shfl_xor(acc[dm][ni], 2);
                acc[dm][ni] += __shfl_xor(acc[dm][ni], 1);
                if (lane == 0) Y[m * N + n0 + ni] = __float2half(acc[dm][ni]);
            }
        }
    }
}
// ============================================================
// Pre-compute w_sum: sum of uint4 values per (row, K-chunk-of-32).
// Eliminates 4 udot8 instructions per row per iteration in gemv_w4a4_udot8.
// Output: W_sum[E, N, K/32] as uint16 (max value = 32*15 = 480, fits uint16)
// Grid: (ceil(N*nit/256), E), Block: 256
// ============================================================
__global__ void precompute_wsum_kernel(
    const uint8_t* __restrict__ W_base,   // [E, N, K/2] packed uint4
    uint16_t*      __restrict__ W_sum,    // [E, N, K/32] output
    int64_t packed_stride,                // N * K/2
    int N, int K
) {
    const int ej = blockIdx.y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nit = K / 32;
    const int total = N * nit;
    if (tid >= total) return;

    const int n = tid / nit;
    const int it = tid % nit;

    const uint8_t* W = W_base + (int64_t)ej * packed_stride + (int64_t)n * (K / 2) + it * 16;
    const uint32_t* wp32 = reinterpret_cast<const uint32_t*>(W);
    const uint32_t ones = 0x11111111u;

    uint32_t wsum = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        wsum = __builtin_amdgcn_udot8(wp32[i], ones, wsum, false);

    W_sum[(int64_t)ej * N * nit + (int64_t)n * nit + it] = (uint16_t)wsum;
}
__global__ void quantize_activations_u4_v2(
    const __half* __restrict__ X,       // [M, K] FP16 input
    uint8_t*      __restrict__ X_q,     // [M, K/2] packed uint4 output
    __half*       __restrict__ X_sc,    // [M, K/32, 2] (scale, min_val)
    int M, int K
) {
    const int groups_per_row = K / 32;
    const int gid = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);  // group index
    if (gid >= M * groups_per_row) return;

    const int lane = threadIdx.x & 31;
    const int m = gid / groups_per_row;
    const int g = gid % groups_per_row;
    const __half* xp = X + m * K + g * 32;

    // Each lane loads 1 value
    float val = __half2float(xp[lane]);

    // Warp-level min/max reduction
    float mn = val, mx = val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        mn = fminf(mn, __shfl_xor(mn, offset));
        mx = fmaxf(mx, __shfl_xor(mx, offset));
    }
    // mn and mx are now broadcast to all lanes

    float range = mx - mn;
    float inv_s = range > 0.0f ? 15.0f / range : 0.0f;
    float sc    = range / 15.0f;

    // Quantize
    int q = (int)roundf((val - mn) * inv_s);
    q = q < 0 ? 0 : q > 15 ? 15 : q;

    // Pack pairs: even lanes hold low nibble, odd lanes hold high nibble
    // IMPORTANT: shuffle must be unconditional so all lanes participate on RDNA4
    int q_neighbor = __shfl(q, lane ^ 1);  // even↔odd swap

    // Only odd lanes write (each writes a packed byte for lane-1 and lane)
    uint8_t* qp = X_q + m * (K / 2) + g * 16;
    if (lane & 1) {
        // lane is odd: q = high nibble, q_neighbor = low nibble (from even lane)
        qp[lane / 2] = (uint8_t)((q << 4) | q_neighbor);
    }

    // Lane 0 writes scale and min
    if (lane == 0) {
        X_sc[(m * groups_per_row + g) * 2]     = __float2half(sc);
        X_sc[(m * groups_per_row + g) * 2 + 1] = __float2half(mn);
    }
}
// ============================================================
// dequant_int4_g64 with TRANSPOSED scales [nit, N, 2] — for expert dispatch
// ============================================================
__global__ void dequant_int4_g64_ts_kernel(
    const uint8_t* __restrict__ packed,   // [N, K/2]
    const __half* __restrict__ scales,    // [K/64, N, 2] transposed interleaved
    __half* __restrict__ out,             // [N, K]
    int N, int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int Kh = K / 2;
    if (idx >= N * Kh) return;
    const int n = idx / Kh;
    const int kh = idx % Kh;
    const uint8_t p = packed[idx];
    int lo = (p & 0xF);
    int hi = (p >> 4);
    const int block_idx = kh / 32;  // group-64
    // Transposed: scales[(block_idx * N + n) * 2] instead of [(n * nit + block_idx) * 2]
    __half2 sz = *reinterpret_cast<const __half2*>(&scales[(block_idx * N + n) * 2]);
    const float s = __half2float(sz.x);
    const float z = __half2float(sz.y);
    out[(int64_t)n * K + kh * 2]     = __float2half(((float)lo - z) * s);
    out[(int64_t)n * K + kh * 2 + 1] = __float2half(((float)hi - z) * s);
}

// ============================================================
// Batched dequant: dequant multiple experts in one kernel launch
// Grid: (ceil(N*Kh/256), num_act), Block: 256
// ============================================================
__global__ void dequant_int4_g64_ts_batched_kernel(
    const uint8_t* __restrict__ packed_base,  // [E, N, K/2] all experts stacked
    const __half*  __restrict__ scales_base,  // [E, K/64, N, 2] all experts stacked
    __half*        __restrict__ out,           // [num_act, N, K] output buffer
    const int*     __restrict__ expert_ids,   // [num_act] which experts to dequant
    int N, int K,
    int64_t packed_stride, int64_t scales_stride
) {
    const int eidx = blockIdx.y;
    const int ej = expert_ids[eidx];
    const uint8_t* P = packed_base + (int64_t)ej * packed_stride;
    const __half*  S = scales_base + (int64_t)ej * scales_stride;
    __half*        O = out + (int64_t)eidx * N * K;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int Kh = K / 2;
    if (idx >= N * Kh) return;
    const int n = idx / Kh;
    const int kh = idx % Kh;
    const uint8_t p = P[idx];
    int lo = (p & 0xF);
    int hi = (p >> 4);
    const int block_idx = kh / 32;
    __half2 sz = *reinterpret_cast<const __half2*>(&S[(block_idx * N + n) * 2]);
    const float s = __half2float(sz.x);
    const float z = __half2float(sz.y);
    O[(int64_t)n * K + kh * 2]     = __float2half(((float)lo - z) * s);
    O[(int64_t)n * K + kh * 2 + 1] = __float2half(((float)hi - z) * s);
}

// ============================================================
// Gather tokens into padded [num_act, max_mj, K] layout
// Grid: (ceil(K/256), max_mj, num_act), Block: 256
// ============================================================
__global__ void gather_to_padded(
    const __half* __restrict__ src,  // [total_tok, K]
    __half*       __restrict__ dst,  // [num_act, max_mj, K]
    const int*    __restrict__ offsets,
    const int*    __restrict__ counts,
    int max_mj, int K
) {
    const int eidx = blockIdx.z;
    const int m = blockIdx.y;
    if (m >= counts[eidx]) return;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const int off = offsets[eidx];
    dst[((int64_t)eidx * max_mj + m) * K + k] = src[((int64_t)(off + m)) * K + k];
}

// ============================================================
// Scatter results from padded [num_act, max_mj, N] back to flat [total_tok, N]
// Grid: (ceil(N/256), max_mj, num_act), Block: 256
// ============================================================
__global__ void scatter_from_padded(
    const __half* __restrict__ src,  // [num_act, max_mj, N]
    __half*       __restrict__ dst,  // [total_tok, N]
    const int*    __restrict__ offsets,
    const int*    __restrict__ counts,
    int max_mj, int N
) {
    const int eidx = blockIdx.z;
    const int m = blockIdx.y;
    if (m >= counts[eidx]) return;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int off = offsets[eidx];
    dst[((int64_t)(off + m)) * N + n] = src[((int64_t)eidx * max_mj + m) * N + n];
}

// Dequant g64 (transposed scales) + rocBLAS GEMM → pre-allocated output
static inline void dequant_matmul_out_g64_ts(
    torch::Tensor X_t,    // [M_j, K] FP16 input
    torch::Tensor out_t,  // [M_j, N] FP16 output (pre-allocated or sliced)
    const uint8_t* W, const __half* S_ts,
    int N, int K,
    torch::Tensor w_tmp,  // pre-allocated dequant buffer (may be larger than [N,K])
    hipStream_t stream
) {
    // Kernel writes contiguously with stride [K,1], so create a matching view
    // (w_tmp.narrow would keep original stride which misaligns when K < w_tmp.size(1))
    auto w_fp16 = torch::from_blob(w_tmp.data_ptr(), {(long)N, (long)K}, w_tmp.options());
    int total = N * (K / 2);
    int threads = 256, blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(dequant_int4_g64_ts_kernel, dim3(blocks), dim3(threads), 0, stream,
        W, S_ts, reinterpret_cast<__half*>(w_tmp.data_ptr()), N, K);
    at::mm_out(out_t, X_t, w_fp16.t());
}

// ============================================================
// MoE Prefill — FP16 attention + INT4 expert dispatch, all in C++
// Returns logits [M, vocab] for all positions.
// ============================================================
// Fused weighted scatter: moe_out_f32[token[i], d] += weight[i] * dn[i, d]
// Uses float32 accumulator to avoid FP16 atomicAdd race conditions.
// 1D grid to avoid grid.y > 65536 limit on RDNA4.
// Grid: ceil(total_tok * D / 256), Block: 256
__global__ void weighted_scatter_add_f32(
    float* __restrict__ moe_out_f32,  // [M, D] float32 accumulator
    const __half* __restrict__ dn,    // [total_tok, D]
    const __half* __restrict__ weight, // [total_tok]
    const int* __restrict__ token_ids, // [total_tok] int32
    int D, int total_tok
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / D;
    int d = tid % D;
    if (i >= total_tok) return;
    float w = __half2float(weight[i]);
    float v = __half2float(dn[(int64_t)i * D + d]) * w;
    int out_idx = token_ids[i] * D + d;
    atomicAdd(&moe_out_f32[out_idx], v);
}

// Convert float32 moe buffer to FP16
__global__ void f32_to_fp16(const float* __restrict__ in, __half* __restrict__ out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = __float2half(in[i]);
}

// ============================================================
// WMMA INT4 GEMM — native gfx12 v_wmma_i32_16x16x32_iu4
// Tile: 16M × 64N per block, 1 warp, 4 WMMA per K-step
// Column-distributed store: lane%16=col, (lane/16)*8+j=row
// From DISCOVERY_GFX12_WMMA_OUTPUT.md — verified correct on R9700
// ============================================================
typedef int wmma_int2 __attribute__((ext_vector_type(2)));
typedef int wmma_int8 __attribute__((ext_vector_type(8)));

// Tile weights: [N, K/2] packed → [K/8, N/16, 64] for coalesced WMMA loads
__global__ void wmma_tile_weights(const uint8_t* src, uint8_t* dst, int N, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int K2=K/2, NB=N/16, total=(K/8)*NB*16;
    if (tid >= total) return;
    int kc=tid/(NB*16), rem=tid%(NB*16), nb=rem/16, nl=rem%16;
    *(uint32_t*)(dst+(int64_t)(kc*NB+nb)*64+nl*4)
        = *(const uint32_t*)(src+(int64_t)(nb*16+nl)*K2+kc*4);
}

// Transpose scales: [N, ng, 2] → [ng, N, 2] (group-major for coalesced access)
__global__ void wmma_transpose_scales(const __half* si, __half* so, int N, int ng) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N*ng) return;
    int n=tid/ng, g=tid%ng;
    so[(int64_t)g*N*2+n*2]   = si[(int64_t)n*ng*2+g*2];
    so[(int64_t)g*N*2+n*2+1] = si[(int64_t)n*ng*2+g*2+1];
}

// Quantize activations to unsigned INT4 with per-group scale+zero+sum
__global__ void wmma_quantize_act(
    const __half* input, uint8_t* act_q, float* act_meta, int M, int K, int M_pad)
{
    int g=blockIdx.x, m=blockIdx.y, tid=threadIdx.x;
    if (m >= M) return;
    float val = __half2float(input[(int64_t)m*K + g*32 + tid]);
    float vmin=val, vmax=val;
    for (int off=16;off>0;off>>=1) { vmin=fminf(vmin,__shfl_xor(vmin,off)); vmax=fmaxf(vmax,__shfl_xor(vmax,off)); }
    float a_range=vmax-vmin, a_scale=(a_range>1e-10f)?(a_range/15.0f):1e-10f;
    int q=__float2int_rn((val-vmin)/a_scale); q=max(0,min(15,q));
    float a_sum=(float)q;
    for (int off=16;off>0;off>>=1) a_sum+=__shfl_xor(a_sum,off);
    int q_other=__shfl_xor(q,1);
    if (tid&1) act_q[(int64_t)m*(K/2)+g*16+tid/2]=(uint8_t)((q<<4)|q_other);
    if (tid==0) { int ng=K/32; act_meta[((int64_t)m*ng+g)*3]=a_scale; act_meta[((int64_t)m*ng+g)*3+1]=vmin; act_meta[((int64_t)m*ng+g)*3+2]=a_sum; }
}

// Symmetric activation quantization: signed INT4 stored as unsigned+8
// Only stores scale (no zero, no sum needed)
__global__ void wmma_quantize_act_sym(
    const __half* input, uint8_t* act_q, float* act_scales, int M, int K, int M_pad)
{
    int g=blockIdx.x, m=blockIdx.y, tid=threadIdx.x;
    if (m >= M) { return; }
    float val = __half2float(input[(int64_t)m*K + g*32 + tid]);
    // Find absmax across group (32 elements)
    float amax = fabsf(val);
    for (int off=16;off>0;off>>=1) amax=fmaxf(amax,__shfl_xor(amax,off));
    float scale = (amax > 1e-10f) ? (amax / 7.0f) : 1e-10f;
    // Quantize to signed [-8,+7], store as unsigned [0,15]
    int q = __float2int_rn(val / scale);
    q = max(-8, min(7, q));
    int stored = q + 8;  // 0..15
    // Pack pairs
    int q_other = __shfl_xor(stored, 1);
    if (tid & 1) act_q[(int64_t)m*(K/2) + g*16 + tid/2] = (uint8_t)((stored << 4) | q_other);
    if (tid == 0) act_scales[m * (K/32) + g] = scale;
}

// WMMA INT4 GEMM — SIGNED SYMMETRIC, 1 warp per block, high occupancy
// No LDS, no barriers. Each warp independently loads A + B from VRAM.
// B loads benefit from L0/L1 cache when multiple blocks share same N-tile.
template<int GS=32>
__global__ __launch_bounds__(32, 8)  // 1 warp, 8 blocks/CU — max occupancy for latency hiding
void wmma_int4_gemm_sym(
    const uint8_t* act_q, const float* act_scales,
    const uint8_t* wt, const float* w_scales,
    __half* output, int M, int M_pad, int N_pad, int K)
{
    const int lane = threadIdx.x;
    const int m_start = blockIdx.y * 16;
    const int n_start = blockIdx.x * 64;
    if (m_start >= M) return;
    const int nl = lane % 16, kg = lane / 16, NB = N_pad / 16, ng = K / GS;
    const int m_idx = m_start + nl;
    const bool m_valid = (m_idx < M);

    float a0[8]={},a1[8]={},a2[8]={},a3[8]={};
    for (int g = 0; g < ng; g++) {
        int k = g * GS;

        // Load A
        wmma_int2 A; float a_sc = 0.0f;
        if (m_valid) {
            const uint8_t* aq = act_q + (int64_t)m_idx * (K/2) + k/2 + kg * 8;
            A = (wmma_int2){*(const int*)(aq), *(const int*)(aq + 4)};
            a_sc = act_scales[m_idx * ng + g];
        } else { A = {}; }

        // Load B directly (L0/L1 cached when multiple blocks access same N-tile)
        #define LOAD_B(idx) wmma_int2 B##idx; \
            {int nb=n_start/16+idx;int kc0=k/8+kg*2,kc1=kc0+1; \
             B##idx=(wmma_int2){ \
                (int)*(const uint32_t*)(wt+(int64_t)(kc0*NB+nb)*64+nl*4), \
                (int)*(const uint32_t*)(wt+(int64_t)(kc1*NB+nb)*64+nl*4)};}
        LOAD_B(0) LOAD_B(1) LOAD_B(2) LOAD_B(3)
        #undef LOAD_B

        // 4 WMMA — SIGNED, zero correction
        wmma_int8 R0={},R1={},R2={},R3={};
        R0=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,true,B0,R0,false);
        R1=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,true,B1,R1,false);
        R2=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,true,B2,R2,false);
        R3=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,true,B3,R3,false);

        // Minimal dequant: scale × signed_dot
        #define DEQUANT_SYM(idx) { \
            float ws = w_scales[g * N_pad + n_start + idx*16 + nl]; \
            float combined = ws * a_sc; \
            for(int j=0;j<8;j++) a##idx[j] += combined * (float)R##idx[j];}
        DEQUANT_SYM(0) DEQUANT_SYM(1) DEQUANT_SYM(2) DEQUANT_SYM(3)
        #undef DEQUANT_SYM
    }
    // Store: column-distributed
    for(int j=0;j<8;j++){int om=m_start+kg*8+j;if(om>=M)continue;
        __half*row=output+(int64_t)om*N_pad;
        row[n_start+nl]=__float2half(a0[j]);row[n_start+16+nl]=__float2half(a1[j]);
        row[n_start+32+nl]=__float2half(a2[j]);row[n_start+48+nl]=__float2half(a3[j]);}
}

// Original asymmetric WMMA (kept for backward compatibility)
template<int GS=32>
__global__ __launch_bounds__(32, 8)
void wmma_int4_gemm(
    const uint8_t* act_q, const float* act_meta,
    const uint8_t* wt, const __half* sz_gn,
    __half* output, int M, int M_pad, int N_pad, int K)
{
    const int lane=threadIdx.x, m_start=blockIdx.y*16, n_start=blockIdx.x*64;
    if(m_start>=M) return;
    const int nl=lane%16, kg=lane/16, NB=N_pad/16, ng=K/GS;
    const int m_idx=m_start+nl;
    const bool m_valid=(m_idx<M);
    const uint32_t ones4=0x11111111u;
    float a0[8]={},a1[8]={},a2[8]={},a3[8]={};
    for(int g=0;g<ng;g++){
        int k=g*GS; int64_t sb=(int64_t)g*N_pad*2;
        wmma_int2 A; float as,az,xz;
        if(m_valid){
            const uint8_t*aq=act_q+(int64_t)m_idx*(K/2)+k/2+kg*8;
            A=(wmma_int2){*(const int*)(aq),*(const int*)(aq+4)};
            const float*meta=act_meta+((int64_t)m_idx*ng+g)*3;
            as=meta[0];az=meta[1];xz=as*meta[2]+az*32.0f;
        }else{A={};as=0;az=0;xz=0;}
        // Load B tiles + w_sum (interleaved with WMMA for latency hiding)
        #define LOAD_B(idx) wmma_int2 B##idx; float bs##idx; \
            {int nb=n_start/16+idx;int kc0=k/8+kg*2,kc1=kc0+1; \
             uint32_t p0=*(const uint32_t*)(wt+(int64_t)(kc0*NB+nb)*64+nl*4); \
             uint32_t p1=*(const uint32_t*)(wt+(int64_t)(kc1*NB+nb)*64+nl*4); \
             B##idx=(wmma_int2){(int)p0,(int)p1}; \
             uint32_t bs=__builtin_amdgcn_udot8(p0,ones4,0u,false); \
             bs=__builtin_amdgcn_udot8(p1,ones4,bs,false); \
             bs##idx=(float)bs+__shfl_xor((float)bs,16);}
        LOAD_B(0) LOAD_B(1) LOAD_B(2) LOAD_B(3)
        #undef LOAD_B
        // 4 WMMA + dequant (interleaved: WMMA hides memory latency of next load)
        wmma_int8 R0={},R1={},R2={},R3={};
        R0=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,A,false,B0,R0,false);
        R1=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,A,false,B1,R1,false);
        R2=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,A,false,B2,R2,false);
        R3=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false,A,false,B3,R3,false);
        #define DEQUANT(idx) {int ni=n_start+idx*16+nl; \
            float ws=__half2float(sz_gn[sb+ni*2]),wz=__half2float(sz_gn[sb+ni*2+1]); \
            float ws_as=ws*as,corr=ws*az*bs##idx-ws*wz*xz; \
            for(int j=0;j<8;j++) a##idx[j]+=ws_as*(float)R##idx[j]+corr;}
        DEQUANT(0) DEQUANT(1) DEQUANT(2) DEQUANT(3)
        #undef DEQUANT
    }
    // Store: column-distributed (lane%16=col, (lane/16)*8+j=row)
    for(int j=0;j<8;j++){int om=m_start+kg*8+j;if(om>=M)continue;
        __half*row=output+(int64_t)om*N_pad;
        row[n_start+nl]=__float2half(a0[j]);row[n_start+16+nl]=__float2half(a1[j]);
        row[n_start+32+nl]=__float2half(a2[j]);row[n_start+48+nl]=__float2half(a3[j]);}
}

// ============================================================
// Pure-HIP MoE prefill kernels — zero PyTorch ATen calls in hot loop
// ============================================================

// Fused softmax + top-k + normalize: one kernel replaces 5 ATen ops
// Grid: (M), Block: (128 or 256). Each block processes one token's E=128 expert logits.
__global__ void softmax_topk_batched(
    const __half* __restrict__ router_logits,  // [M, E] FP16
    int* __restrict__ flat_expert,             // [M*k] int32 output
    __half* __restrict__ flat_weight,          // [M*k] FP16 output
    int* __restrict__ flat_token,              // [M*k] int32 output (token_id = blockIdx.x)
    int E, int k
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Load logits into shared memory
    __shared__ float svals[256];  // max E=256
    float val = (tid < E) ? __half2float(router_logits[row * E + tid]) : -1e30f;

    // Softmax: max reduction
    float lmax = val;
    for (int off = 16; off >= 1; off >>= 1) lmax = fmaxf(lmax, __shfl_xor(lmax, off));
    __shared__ float scratch[8];
    if (tid % 32 == 0) scratch[tid / 32] = lmax;
    __syncthreads();
    if (tid < blockDim.x / 32) {
        float v = scratch[tid];
        for (int off = 4; off >= 1; off >>= 1) v = fmaxf(v, __shfl_xor(v, off));
        if (tid == 0) scratch[0] = v;
    }
    __syncthreads();
    float gmax = scratch[0];

    // Softmax: exp + sum
    float e = (tid < E) ? expf(val - gmax) : 0.0f;
    float lsum = e;
    for (int off = 16; off >= 1; off >>= 1) lsum += __shfl_xor(lsum, off);
    if (tid % 32 == 0) scratch[tid / 32] = lsum;
    __syncthreads();
    if (tid < blockDim.x / 32) {
        float v = scratch[tid];
        for (int off = 4; off >= 1; off >>= 1) v += __shfl_xor(v, off);
        if (tid == 0) scratch[0] = v;
    }
    __syncthreads();
    svals[tid] = (tid < E && scratch[0] > 0.0f) ? (e / scratch[0]) : 0.0f;
    __syncthreads();

    // Serial top-K by thread 0 (k=8, E=128 → 8×128 = 1024 comparisons, trivial)
    if (tid == 0) {
        float sum_k = 0.0f;
        int base = row * k;
        for (int r = 0; r < k; r++) {
            int best = 0; float bv = svals[0];
            for (int i = 1; i < E; i++) {
                if (svals[i] > bv) { bv = svals[i]; best = i; }
            }
            flat_expert[base + r] = best;
            flat_weight[base + r] = __float2half(bv);
            flat_token[base + r] = row;
            sum_k += bv;
            svals[best] = -1.0f;
        }
        float inv = (sum_k > 0.0f) ? (1.0f / sum_k) : 1.0f;
        for (int r = 0; r < k; r++)
            flat_weight[base + r] = __float2half(__half2float(flat_weight[base + r]) * inv);
    }
}

// GPU-side exclusive prefix sum for E=128 elements (single warp-group)
__global__ void prefix_sum_small(
    const int* __restrict__ counts,  // [E] input
    int* __restrict__ offsets,       // [E] exclusive prefix sum output
    int* __restrict__ write_pos,     // [E] copy of offsets (for atomic scatter)
    int E
) {
    int tid = threadIdx.x;
    __shared__ int smem[256];
    smem[tid] = (tid < E) ? counts[tid] : 0;
    __syncthreads();
    // Blelloch scan (up-sweep + down-sweep) for E<=256
    for (int d = 1; d < E; d <<= 1) {
        int ai = (tid + 1) * (d << 1) - 1;
        if (ai < E) smem[ai] += smem[ai - d];
        __syncthreads();
    }
    if (tid == 0) smem[E - 1] = 0;
    __syncthreads();
    for (int d = E >> 1; d >= 1; d >>= 1) {
        int ai = (tid + 1) * (d << 1) - 1;
        if (ai < E) {
            int t = smem[ai - d];
            smem[ai - d] = smem[ai];
            smem[ai] += t;
        }
        __syncthreads();
    }
    if (tid < E) {
        offsets[tid] = smem[tid];
        write_pos[tid] = smem[tid];
    }
}

// Element-wise add: C[i] = A[i] + B[i] for FP16
__global__ void vadd_fp16(const __half* A, const __half* B, __half* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = __float2half(__half2float(A[i]) + __half2float(B[i]));
}

// Simple row gather: dst[i] = src[indices[i]] for each row
__global__ void gather_rows(
    const __half* __restrict__ src,  // [M, D]
    __half* __restrict__ dst,        // [total_tok, D]
    const int* __restrict__ indices, // [total_tok] int32
    int D, int total_tok
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < total_tok && col < D) {
        dst[(int64_t)row * D + col] = src[(int64_t)indices[row] * D + col];
    }
}

// int32 variants of bucket sort (eliminate int64 overhead)
__global__ void bucket_count_i32(
    const int* __restrict__ expert_ids, int* __restrict__ counts, int total_tok
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total_tok) atomicAdd(&counts[expert_ids[tid]], 1);
}

__global__ void bucket_scatter_i32(
    const int* __restrict__ expert_ids, const int* __restrict__ token_ids,
    const __half* __restrict__ weights, int* __restrict__ write_pos,
    int* __restrict__ sorted_tokens, __half* __restrict__ sorted_weights,
    int total_tok
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total_tok) {
        int pos = atomicAdd(&write_pos[expert_ids[tid]], 1);
        sorted_tokens[pos] = token_ids[tid];
        sorted_weights[pos] = weights[tid];
    }
}

// Bucket sort by expert ID: O(N) instead of O(N log N)
// Phase 1: count tokens per expert (histogram)
__global__ void bucket_count(
    const int64_t* __restrict__ expert_ids,  // [total_tok] expert IDs (0..E-1)
    int* __restrict__ counts,                // [E] output counts
    int total_tok
) {
    // Simple atomicAdd histogram
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total_tok) {
        int eid = (int)expert_ids[tid];
        atomicAdd(&counts[eid], 1);
    }
}

// Phase 2: scatter tokens into sorted order using atomic offsets
__global__ void bucket_scatter(
    const int64_t* __restrict__ expert_ids,  // [total_tok]
    const int64_t* __restrict__ token_ids,   // [total_tok] original token indices
    const __half* __restrict__ weights,      // [total_tok]
    int* __restrict__ write_pos,             // [E] atomic write positions (initialized to offsets)
    int64_t* __restrict__ sorted_tokens,     // [total_tok] output
    __half* __restrict__ sorted_weights,     // [total_tok] output
    int total_tok
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total_tok) {
        int eid = (int)expert_ids[tid];
        int pos = atomicAdd(&write_pos[eid], 1);
        sorted_tokens[pos] = token_ids[tid];
        sorted_weights[pos] = weights[tid];
    }
}

// Row-major g64 dequant kernel: scales layout [N, K/64, 2] flattened to [N, K/64*2]
__global__ void dequant_int4_g64_rm_kernel(
    const uint8_t* __restrict__ packed,  // [N, K/2]
    const __half* __restrict__ scales,   // [N, K/64*2] row-major flat (scale, zero pairs)
    __half* __restrict__ out,            // [N, K]
    int N, int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int Kh = K / 2;
    if (idx >= N * Kh) return;
    const int n = idx / Kh;
    const int kh = idx % Kh;
    const uint8_t p = packed[idx];
    int lo = (p & 0xF);
    int hi = (p >> 4);
    const int g = kh / 32;  // group-64: 32 packed bytes = 64 elements
    const int ng = K / 64;
    // Row-major: scales[n * ng * 2 + g * 2]
    __half2 sz = *reinterpret_cast<const __half2*>(&scales[n * ng * 2 + g * 2]);
    const float s = __half2float(sz.x);
    const float z = __half2float(sz.y);
    out[(int64_t)n * K + kh * 2]     = __float2half(((float)lo - z) * s);
    out[(int64_t)n * K + kh * 2 + 1] = __float2half(((float)hi - z) * s);
}
// Forward declaration
void dequant_gemm_tiled_g64(torch::Tensor input, torch::Tensor packed, torch::Tensor scales,
                            int64_t N, int64_t K,
                            torch::Tensor w_buf, torch::Tensor out,
                            int64_t N_tile);

torch::Tensor prefill_moe_logits(
    torch::Tensor input_ids,              // [M] long
    torch::Tensor embed_weight,           // [vocab, D] FP16
    torch::Tensor final_norm_w,           // [D] FP16
    torch::Tensor lm_w, torch::Tensor lm_s,
    int64_t lm_N, int64_t lm_K,
    // Attention weights: FP16 OR INT4 (pass empty list for INT4 fallback)
    std::vector<torch::Tensor> fp16_qkv_w,  // [q_dim+2*kv_dim, D] FP16 (or empty)
    std::vector<torch::Tensor> fp16_o_w,    // [D, q_dim] FP16 (or empty)
    // INT4 attention weights (used when FP16 is empty)
    std::vector<torch::Tensor> int4_qkv_w,  // [N, K/2] uint8
    std::vector<torch::Tensor> int4_qkv_s,  // [N, K/64*2] FP16 flat
    std::vector<int64_t> int4_qkv_N,
    std::vector<torch::Tensor> int4_o_w,    // [N, K/2] uint8
    std::vector<torch::Tensor> int4_o_s,    // [N, K/64*2] FP16 flat
    std::vector<int64_t> int4_o_N,
    std::vector<int64_t> int4_o_K,
    // MoE weights
    std::vector<torch::Tensor> gate_w,       // [E, D] FP16
    std::vector<torch::Tensor> exp_gu_w,     // [E, N_gu, K_gu/2] uint8
    std::vector<torch::Tensor> exp_gu_s,     // [E, nit_gu, N_gu, 2] FP16 transposed
    std::vector<torch::Tensor> exp_dn_w,     // [E, N_dn, K_dn/2] uint8
    std::vector<torch::Tensor> exp_dn_s,     // [E, nit_dn, N_dn, 2] FP16 transposed
    // Norms
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w,  std::vector<torch::Tensor> k_norm_w,
    // RoPE
    torch::Tensor rope_cos, torch::Tensor rope_sin,
    // Config
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t moe_inter, int64_t num_active,
    int64_t num_experts,
    float rms_eps,
    // KV caches (FP16, filled during prefill)
    std::vector<torch::Tensor> k_caches,  // [kv_heads, max_seq, head_dim]
    std::vector<torch::Tensor> v_caches,
    // Hadamard
    torch::Tensor hadamard_mat  // [32, 32] FP16
) {
    c10::InferenceMode guard;

    bool use_fp16_attn = !fp16_qkv_w.empty() && fp16_qkv_w[0].numel() > 0;
    int num_layers = use_fp16_attn ? fp16_qkv_w.size() : gate_w.size();
    int M = input_ids.size(0);
    int D = hidden_size;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int qkv_dim = q_dim + 2 * kv_dim;
    int inter = moe_inter;
    int E = num_experts;
    int k = num_active;
    auto device = input_ids.device();
    auto stream = at::hip::getCurrentHIPStream();
    auto opts = torch::dtype(torch::kHalf).device(device);
    auto opts_f = torch::dtype(torch::kFloat32).device(device);
    auto opts_l = torch::dtype(torch::kLong).device(device);

    int norm_bs = 256;
    size_t smem_norm = D * sizeof(float);
    int half_D = head_dim / 2;
    auto* had_p = reinterpret_cast<const __half*>(hadamard_mat.data_ptr());

    // Pre-allocate buffers
    auto x = torch::embedding(embed_weight, input_ids);  // [M, D]
    auto normed = torch::empty({M, D}, opts);
    auto residual = torch::empty({M, D}, opts);

    int N_gu = 2 * inter;
    int K_gu = D;
    int N_dn = D;
    int K_dn = inter;

    // Batched expert buffers: share memory to reduce VRAM at large M
    int total_dispatch = M * k;
    if (M >= 16384) {
        size_t fb, tb; hipMemGetInfo(&fb, &tb);
        fprintf(stderr, "prefill alloc: M=%d total_dispatch=%d free=%.1fGB\n", M, total_dispatch, fb/1073741824.0);
    }
    auto gu_buf = torch::empty({(long)total_dispatch, (long)N_gu}, opts);
    auto silu_buf = torch::empty({(long)total_dispatch, (long)inter}, opts);
    auto rot_silu_buf = torch::empty({(long)total_dispatch, (long)inter}, opts);
    // x_gather_buf doubles as dn_buf after GU has consumed gathered input (non-overlapping lifetimes)
    auto x_gather_buf = torch::empty({(long)total_dispatch, (long)D}, opts);

    // LM head dequant buffer
    auto lm_w_tmp = torch::empty({(long)lm_N * lm_K}, opts);

    // W4A4 buffers: quantized activations for all-experts dispatch
    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);
    auto xq_gu_buf = torch::empty({(long)total_dispatch, (long)(K_gu / 2)}, opts_u8);
    auto xq_gu_sc  = torch::empty({(long)total_dispatch, (long)(K_gu / 32), 2L}, opts);
    auto xq_dn_buf = torch::empty({(long)total_dispatch, (long)(K_dn / 2)}, opts_u8);
    auto xq_dn_sc  = torch::empty({(long)total_dispatch, (long)(K_dn / 32), 2L}, opts);

    // INT4 attention dequant buffer (used when FP16 not available)
    int max_attn_N = use_fp16_attn ? 0 : qkv_dim;
    int max_attn_K = D;
    torch::Tensor attn_w_buf;
    if (!use_fp16_attn) {
        max_attn_N = 0;
        for (int li = 0; li < num_layers; li++) {
            int n = int4_qkv_N[li];
            if (n > max_attn_N) max_attn_N = n;
            int no = int4_o_N[li];
            if (no > max_attn_N) max_attn_N = no;
        }
        int max_attn_K2 = D;
        for (int li = 0; li < num_layers; li++) {
            int ko = int4_o_K[li];
            if (ko > max_attn_K2) max_attn_K2 = ko;
        }
        // w_buf must be 1D flat of size N_tile * max_K for dequant_gemm_tiled_g64
        int attn_tile = 4096;
        long attn_buf_size = (long)std::min(attn_tile, max_attn_N) * max_attn_K2;
        attn_w_buf = torch::empty({attn_buf_size}, opts);
    }

    auto* x_p = reinterpret_cast<__half*>(x.data_ptr());
    auto* n_p = reinterpret_cast<__half*>(normed.data_ptr());
    auto* r_p = reinterpret_cast<__half*>(residual.data_ptr());

    const __half* cos_p = reinterpret_cast<const __half*>(rope_cos.data_ptr());
    const __half* sin_p = reinterpret_cast<const __half*>(rope_sin.data_ptr());

    // Pre-extract norm pointers
    std::vector<const __half*> in_np(num_layers), post_np(num_layers);
    std::vector<const __half*> q_np(num_layers), k_np(num_layers);
    for (int li = 0; li < num_layers; li++) {
        in_np[li] = reinterpret_cast<const __half*>(in_norm_w[li].data_ptr());
        post_np[li] = reinterpret_cast<const __half*>(post_norm_w[li].data_ptr());
        q_np[li] = reinterpret_cast<const __half*>(q_norm_w[li].data_ptr());
        k_np[li] = reinterpret_cast<const __half*>(k_norm_w[li].data_ptr());
    }

    // Pre-allocate ALL reusable buffers (avoid per-layer allocation)
    int total_tok = M * k;
    auto rot_buf = torch::empty({(long)M, (long)D}, opts);
    auto moe_out = torch::empty({(long)M, (long)D}, opts);
    // Allocate qkv_out large enough for both FP16 (qkv_dim) and INT4 (int4_qkv_N) paths
    int qkv_buf_cols = qkv_dim;
    if (!use_fp16_attn) {
        for (int li = 0; li < num_layers; li++) {
            int n = int4_qkv_N[li];
            if (n > qkv_buf_cols) qkv_buf_cols = n;
        }
    }
    auto qkv_out = torch::empty({(long)M, (long)qkv_buf_cols}, opts);
    auto router_out = torch::empty({(long)M, (long)E}, opts);
    auto* router_out_p = reinterpret_cast<__half*>(router_out.data_ptr());
    auto d_expert_ids_all = at::arange(E, torch::dtype(torch::kInt32).device(device));
    auto* eid_all_p = d_expert_ids_all.data_ptr<int>();

    // Pre-allocated MoE sort buffers (int32, reused every layer — zero ATen in hot loop)
    auto opts_i32 = torch::dtype(torch::kInt32).device(device);
    auto moe_flat_expert = torch::empty({(long)total_tok}, opts_i32);
    auto moe_flat_weight = torch::empty({(long)total_tok}, opts);
    auto moe_flat_token = torch::empty({(long)total_tok}, opts_i32);
    auto moe_counts = torch::empty({(long)E}, opts_i32);
    auto moe_offsets = torch::empty({(long)E}, opts_i32);
    auto moe_write_pos = torch::empty({(long)E}, opts_i32);
    auto moe_sorted_token = torch::empty({(long)total_tok}, opts_i32);
    auto moe_sorted_weight = torch::empty({(long)total_tok}, opts);
    auto* mfe_p = moe_flat_expert.data_ptr<int>();
    auto* mfw_p = reinterpret_cast<__half*>(moe_flat_weight.data_ptr());
    auto* mft_p = moe_flat_token.data_ptr<int>();
    auto* mcnt_p = moe_counts.data_ptr<int>();
    auto* moff_p = moe_offsets.data_ptr<int>();
    auto* mwp_p = moe_write_pos.data_ptr<int>();
    auto* mst_p = moe_sorted_token.data_ptr<int>();
    auto* msw_p = reinterpret_cast<__half*>(moe_sorted_weight.data_ptr());

    // Pre-extract gate weight pointers
    std::vector<const __half*> gate_wp(num_layers);
    for (int li = 0; li < num_layers; li++)
        gate_wp[li] = reinterpret_cast<const __half*>(gate_w[li].data_ptr());

    // Strides for expert weight indexing (constant across layers)
    int64_t gu_packed_stride = (int64_t)N_gu * (K_gu / 2);
    int64_t gu_scales_stride = (int64_t)(K_gu / 64) * N_gu * 2;
    int64_t dn_packed_stride = (int64_t)N_dn * (K_dn / 2);
    int64_t dn_scales_stride = (int64_t)(K_dn / 64) * N_dn * 2;

    // Layer loop
    for (int li = 0; li < num_layers; li++) {
        // Debug (disabled for production)
        // if (li == 0) fprintf(stderr, "prefill: L%d M=%d fp16=%d\n", li, M, (int)use_fp16_attn);
        // 1. RMSNorm
        hipLaunchKernelGGL(rmsnorm_batched, dim3(M), dim3(norm_bs), smem_norm, stream,
            x_p, in_np[li], n_p, D, rms_eps);
        { auto e = hipGetLastError(); if (e != hipSuccess) { fprintf(stderr, "  rmsnorm launch FAIL: %s\n", hipGetErrorString(e)); } }
        if (li == 0 || M >= 8192) { hipDeviceSynchronize(); auto e = hipGetLastError(); fprintf(stderr, "  rmsnorm %s\n", e == hipSuccess ? "OK" : hipGetErrorString(e)); }

        // 2. QKV projection
        auto normed_t = torch::from_blob(n_p, {(long)M, (long)D}, opts);
        if (use_fp16_attn) {
            at::mm_out(qkv_out, normed_t, fp16_qkv_w[li].t());
        } else {
            // INT4: Hadamard rotate → dequant tiled GEMM
            auto rot_attn = torch::empty({(long)M, (long)D}, opts);
            launch_hadamard_batched(n_p, reinterpret_cast<__half*>(rot_attn.data_ptr()),
                                    had_p, M, D, stream);
            if (li == 0) { hipStreamSynchronize(stream); fprintf(stderr, "  hadamard OK\n"); }
            int Nq = int4_qkv_N[li];
            auto qkv_slice = qkv_out.narrow(1, 0, Nq);
            if (li == 0) fprintf(stderr, "  qkv: Nq=%d, K=%d, w_buf=%ld, qkv_slice=[%ld,%ld]\n",
                Nq, D, attn_w_buf.numel(), qkv_slice.size(0), qkv_slice.size(1));
            dequant_gemm_tiled_g64(rot_attn, int4_qkv_w[li], int4_qkv_s[li],
                                   Nq, D, attn_w_buf, qkv_slice, 4096);
            if (li == 0) { hipStreamSynchronize(stream); fprintf(stderr, "  qkv GEMM OK\n"); }
        }

        // 3. Head RMSNorm + RoPE
        auto* qkv_p = reinterpret_cast<__half*>(qkv_out.data_ptr());
        // Split into chunks if M is large to avoid resource limits
        {
            const int MAX_M_BATCH = 8192;
            for (int m_start = 0; m_start < M; m_start += MAX_M_BATCH) {
                int m_batch = std::min(MAX_M_BATCH, M - m_start);
                __half* qkv_batch = qkv_p + (int64_t)m_start * qkv_buf_cols;
                const __half* cos_batch = cos_p + (int64_t)m_start * half_D;
                const __half* sin_batch = sin_p + (int64_t)m_start * half_D;
                hipLaunchKernelGGL(head_rmsnorm_rope_prefill,
                    dim3(num_heads + num_kv_heads, m_batch), dim3(32), 0, stream,
                    qkv_batch, q_np[li], k_np[li],
                    cos_batch, sin_batch,
                    m_batch, (int)num_heads, (int)num_kv_heads, (int)head_dim, rms_eps, half_D, qkv_buf_cols);
                if ((li == 0 || M >= 8192) && M > 8192) {
                    hipDeviceSynchronize(); auto eb = hipGetLastError();
                    fprintf(stderr, "  head_norm batch m_start=%d m_batch=%d %s\n", m_start, m_batch, eb==hipSuccess?"OK":hipGetErrorString(eb));
                }
            }
        }
        if (li == 0 || M >= 8192) { hipDeviceSynchronize(); auto e2 = hipGetLastError(); fprintf(stderr, "  head_norm_rope %s\n", e2==hipSuccess?"OK":hipGetErrorString(e2)); }

        // 4. Write KV cache + SDPA
        {
            auto q_t = qkv_out.narrow(1, 0, q_dim).view({M, (long)num_heads, (long)head_dim})
                            .permute({1, 0, 2}).unsqueeze(0);
            auto k_t = qkv_out.narrow(1, q_dim, kv_dim).view({M, (long)num_kv_heads, (long)head_dim})
                            .permute({1, 0, 2}).unsqueeze(0);
            auto v_t = qkv_out.narrow(1, q_dim + kv_dim, kv_dim).view({M, (long)num_kv_heads, (long)head_dim})
                            .permute({1, 0, 2}).unsqueeze(0);

            // Fill KV cache
            k_caches[li].narrow(1, 0, M).copy_(k_t.squeeze(0));
            v_caches[li].narrow(1, 0, M).copy_(v_t.squeeze(0));

            auto attn = at::scaled_dot_product_attention(q_t, k_t, v_t,
                /*attn_mask=*/c10::nullopt, /*dropout_p=*/0.0,
                /*is_causal=*/true, /*scale=*/c10::nullopt, /*enable_gqa=*/true);
            auto attn_out = attn.squeeze(0).permute({1, 0, 2}).reshape({M, q_dim});

            // 5. O projection
            torch::Tensor o_out;
            if (use_fp16_attn) {
                o_out = at::mm(attn_out, fp16_o_w[li].t());
            } else {
                // INT4: Hadamard rotate attn_out → dequant tiled GEMM
                int No = int4_o_N[li];
                int Ko = int4_o_K[li];
                auto rot_attn_out = torch::empty({(long)M, (long)Ko}, opts);
                launch_hadamard_batched(
                    reinterpret_cast<const __half*>(attn_out.data_ptr()),
                    reinterpret_cast<__half*>(rot_attn_out.data_ptr()),
                    had_p, M, Ko, stream);
                o_out = torch::empty({(long)M, (long)No}, opts);
                dequant_gemm_tiled_g64(rot_attn_out, int4_o_w[li], int4_o_s[li],
                                       No, Ko, attn_w_buf, o_out, 4096);
                if (li == 0) { hipStreamSynchronize(stream); fprintf(stderr, "  O GEMM OK\n"); }
            }

            // 6. Fused residual + post_norm
            auto* o_p = reinterpret_cast<__half*>(o_out.data_ptr());
            hipLaunchKernelGGL(fused_res_norm_batched, dim3(M), dim3(norm_bs), smem_norm, stream,
                o_p, x_p, post_np[li], n_p, r_p, D, rms_eps);
        }

        // 7. MoE FFN — with optional per-phase timing on layer 0
        hipEvent_t moe_ev[8];
        bool do_moe_profile = false;  // Set true for profiling
        if (do_moe_profile) for (auto& e : moe_ev) hipEventCreate(&e);

        #define MOE_RECORD(idx) if (do_moe_profile) hipEventRecord(moe_ev[idx], stream)

        MOE_RECORD(0);  // start

        // === PURE HIP MoE dispatch — zero ATen calls ===

        // 1. Router GEMM: normed @ gate.T → [M, E]
        auto* rot_p = reinterpret_cast<__half*>(rot_buf.data_ptr());
        {
            auto normed_norot = torch::from_blob(n_p, {(long)M, (long)D}, opts);
            at::mm_out(router_out, normed_norot, gate_w[li].t());
        }
        launch_hadamard_batched(n_p, rot_p, had_p, M, D, stream);

        MOE_RECORD(1);

        // 2. Fused softmax + top-k + normalize (1 kernel, no ATen)
        hipLaunchKernelGGL(softmax_topk_batched,
            dim3(M), dim3(128), 0, stream,
            router_out_p, mfe_p, mfw_p, mft_p, E, k);

        MOE_RECORD(2);

        // 3. Bucket sort: histogram + prefix sum + scatter (all HIP, no ATen)
        hipMemsetAsync(mcnt_p, 0, E * sizeof(int), stream);
        {
            int bs = 256;
            hipLaunchKernelGGL(bucket_count_i32, dim3((total_tok+bs-1)/bs), dim3(bs), 0, stream,
                mfe_p, mcnt_p, total_tok);
        }
        hipLaunchKernelGGL(prefix_sum_small, dim3(1), dim3(128), 0, stream,
            mcnt_p, moff_p, mwp_p, E);
        {
            int bs = 256;
            hipLaunchKernelGGL(bucket_scatter_i32, dim3((total_tok+bs-1)/bs), dim3(bs), 0, stream,
                mfe_p, mft_p, mfw_p, mwp_p, mst_p, msw_p, total_tok);
        }

        MOE_RECORD(3);

        // 4. Gather: rot_buf[sorted_token] → x_gather_buf (HIP kernel, no ATen)
        auto* xg_p = reinterpret_cast<__half*>(x_gather_buf.data_ptr());
        auto* dn_out_p = xg_p;
        hipLaunchKernelGGL(gather_rows,
            dim3((D+255)/256, total_tok), dim3(256), 0, stream,
            rot_p, xg_p, mst_p, D, total_tok);

        MOE_RECORD(4);

        // 5. Zero moe output (hipMemset, no ATen)
        hipMemsetAsync(reinterpret_cast<__half*>(moe_out.data_ptr()), 0, M * D * sizeof(__half), stream);


        // --- Expert GEMMs: hybrid W4A4 (small M) / dequant+rocBLAS (large M) ---
        auto* gu_out_p = reinterpret_cast<__half*>(gu_buf.data_ptr());
        auto* silu_p = reinterpret_cast<__half*>(silu_buf.data_ptr());
        auto* rot_silu_p = reinterpret_cast<__half*>(rot_silu_buf.data_ptr());

        if (M <= 1024) {  // W4A4: no CPU-GPU sync, 2 kernels per GU/DN
            // W4A4 all-experts dispatch — ZERO CPU-GPU sync, 2 kernel launches total
            int gu_groups = total_tok * (K_gu / 32);
            hipLaunchKernelGGL(quantize_activations_u4_v2,
                dim3((gu_groups + 7) / 8), dim3(256), 0, stream,
                xg_p,
                reinterpret_cast<uint8_t*>(xq_gu_buf.data_ptr()),
                reinterpret_cast<__half*>(xq_gu_sc.data_ptr()),
                total_tok, K_gu);

            constexpr int BN = 4, BM = 8;
            hipLaunchKernelGGL((gemv_w4a4_udot8_v3<BN, BM>),
                dim3((N_gu + BN - 1) / BN, E), dim3(32), 0, stream,
                reinterpret_cast<const uint8_t*>(xq_gu_buf.data_ptr()),
                reinterpret_cast<const __half*>(xq_gu_sc.data_ptr()),
                exp_gu_w[li].data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(exp_gu_s[li].data_ptr()),
                gu_out_p, eid_all_p, moff_p, mcnt_p,
                gu_packed_stride, gu_scales_stride, N_gu, K_gu);

            const int MAX_BATCH = 32768;
            for (int t_start = 0; t_start < total_tok; t_start += MAX_BATCH) {
                int t_batch = std::min(MAX_BATCH, total_tok - t_start);
                __half* gu_batch = gu_out_p + (int64_t)t_start * N_gu;
                __half* silu_batch = silu_p + (int64_t)t_start * inter;
                hipLaunchKernelGGL(silu_mul_batched,
                    dim3((inter + 255) / 256, t_batch), dim3(256), 0, stream,
                    gu_batch, gu_batch + inter, silu_batch, inter, N_gu, inter);
                __half* rot_batch = rot_silu_p + (int64_t)t_start * K_dn;
                launch_hadamard_batched(silu_batch, rot_batch, had_p, t_batch, K_dn, stream);
            }

            MOE_RECORD(5);

            int dn_groups = total_tok * (K_dn / 32);
            hipLaunchKernelGGL(quantize_activations_u4_v2,
                dim3((dn_groups + 7) / 8), dim3(256), 0, stream,
                rot_silu_p,
                reinterpret_cast<uint8_t*>(xq_dn_buf.data_ptr()),
                reinterpret_cast<__half*>(xq_dn_sc.data_ptr()),
                total_tok, K_dn);

            hipLaunchKernelGGL((gemv_w4a4_udot8_v3<BN, BM>),
                dim3((N_dn + BN - 1) / BN, E), dim3(32), 0, stream,
                reinterpret_cast<const uint8_t*>(xq_dn_buf.data_ptr()),
                reinterpret_cast<const __half*>(xq_dn_sc.data_ptr()),
                exp_dn_w[li].data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(exp_dn_s[li].data_ptr()),
                dn_out_p, eid_all_p, moff_p, mcnt_p,
                dn_packed_stride, dn_scales_stride, N_dn, K_dn);

        } else {
            // Large M: dequant+rocBLAS per expert — NO CPU-GPU SYNC
            // Launch for ALL 128 experts unconditionally (count=0 experts → M=0 GEMM = no-op)
            // Read counts/offsets on CPU ONCE outside layer loop (pre-sync before first layer)
            // For now: use hipMemcpy sync (TODO: eliminate by launching all experts unconditionally)
            int h_counts[128], h_offsets[128];
            hipMemcpyAsync(h_counts, mcnt_p, E * sizeof(int), hipMemcpyDeviceToHost, stream);
            hipMemcpyAsync(h_offsets, moff_p, E * sizeof(int), hipMemcpyDeviceToHost, stream);
            hipStreamSynchronize(stream);

            auto* gu_w_p = reinterpret_cast<__half*>(lm_w_tmp.data_ptr());
            int gu_total_elem = N_gu * (K_gu / 2);
            for (int e = 0; e < E; e++) {
                int cnt = h_counts[e];
                if (cnt == 0) continue;
                int off = h_offsets[e];
                auto* gu_packed_e = exp_gu_w[li].data_ptr<uint8_t>() + (int64_t)e * gu_packed_stride;
                auto* gu_scales_e = reinterpret_cast<const __half*>(exp_gu_s[li].data_ptr()) + (int64_t)e * gu_scales_stride;
                hipLaunchKernelGGL(dequant_int4_g64_rm_kernel,
                    dim3((gu_total_elem + 255) / 256), dim3(256), 0, stream,
                    gu_packed_e, gu_scales_e, gu_w_p, N_gu, K_gu);
                // direct hipBLAS — no from_blob, no ATen dispatch
                direct_hgemm(xg_p + (int64_t)off*K_gu, gu_w_p,
                             gu_out_p + (int64_t)off*N_gu, cnt, N_gu, K_gu, stream);
            }

            const int MAX_BATCH = 32768;
            for (int t_start = 0; t_start < total_tok; t_start += MAX_BATCH) {
                int t_batch = std::min(MAX_BATCH, total_tok - t_start);
                __half* gu_batch = gu_out_p + (int64_t)t_start * N_gu;
                __half* silu_batch = silu_p + (int64_t)t_start * inter;
                hipLaunchKernelGGL(silu_mul_batched,
                    dim3((inter + 255) / 256, t_batch), dim3(256), 0, stream,
                    gu_batch, gu_batch + inter, silu_batch, inter, N_gu, inter);
                __half* rot_batch = rot_silu_p + (int64_t)t_start * K_dn;
                launch_hadamard_batched(silu_batch, rot_batch, had_p, t_batch, K_dn, stream);
            }

            MOE_RECORD(5);

            auto* dn_w_p = gu_w_p;
            int dn_total_elem = N_dn * (K_dn / 2);
            for (int e = 0; e < E; e++) {
                int cnt = h_counts[e];
                if (cnt == 0) continue;
                int off = h_offsets[e];
                auto* dn_packed_e = exp_dn_w[li].data_ptr<uint8_t>() + (int64_t)e * dn_packed_stride;
                auto* dn_scales_e = reinterpret_cast<const __half*>(exp_dn_s[li].data_ptr()) + (int64_t)e * dn_scales_stride;
                hipLaunchKernelGGL(dequant_int4_g64_rm_kernel,
                    dim3((dn_total_elem + 255) / 256), dim3(256), 0, stream,
                    dn_packed_e, dn_scales_e, dn_w_p, N_dn, K_dn);
                direct_hgemm(rot_silu_p + (int64_t)off*K_dn, dn_w_p,
                             dn_out_p + (int64_t)off*D, cnt, N_dn, K_dn, stream);
            }
        }


        // --- Weighted scatter: single call ---
        // Fused weighted scatter using float32 atomicAdd (avoids PyTorch broadcast bug at large M)
        {
            auto moe_f32 = torch::from_blob(
                reinterpret_cast<float*>(moe_out.data_ptr()),
                {(long)M * D}, torch::dtype(torch::kFloat32).device(device));
            // moe_f32 is reinterpret of moe_out memory — need separate buffer
            // Actually, moe_out is FP16 [M,D], we need float32 [M,D] = 2x memory
            // Use a static buffer to avoid per-layer allocation
            static torch::Tensor moe_f32_buf;
            if (!moe_f32_buf.defined() || moe_f32_buf.numel() < (long)M * D) {
                moe_f32_buf = torch::zeros({(long)M, (long)D}, torch::dtype(torch::kFloat32).device(device));
            } else {
                hipMemsetAsync(moe_f32_buf.data_ptr<float>(), 0, M * D * sizeof(float), stream);
            }
            auto* moe_f32_p = moe_f32_buf.data_ptr<float>();

            int64_t scatter_total = (int64_t)total_tok * D;
            int scatter_blocks = (scatter_total + 255) / 256;
            hipLaunchKernelGGL(weighted_scatter_add_f32, dim3(scatter_blocks), dim3(256), 0, stream,
                moe_f32_p, dn_out_p, msw_p, mst_p, D, total_tok);

            // Convert float32 accumulator to FP16 moe_out
            auto* moe_hp = reinterpret_cast<__half*>(moe_out.data_ptr());
            int total_elem = M * D;
            hipLaunchKernelGGL(f32_to_fp16, dim3((total_elem + 255) / 256), dim3(256), 0, stream,
                moe_f32_p, moe_hp, total_elem);
        }
        MOE_RECORD(6);  // after DN GEMM
        MOE_RECORD(7);  // after scatter (same event — scatter is fused below)

        if (do_moe_profile) {
            hipStreamSynchronize(stream);
            float ms[7];
            for (int i = 0; i < 7; i++) hipEventElapsedTime(&ms[i], moe_ev[i], moe_ev[i+1]);
            fprintf(stderr, "  MoE L0 profile (M=%d, total_tok=%d):\n", M, total_tok);
            fprintf(stderr, "    Router+Had:    %.2f ms\n", ms[0]);
            fprintf(stderr, "    TopK+softmax:  %.2f ms\n", ms[1]);
            fprintf(stderr, "    Sort+counts:   %.2f ms\n", ms[2]);
            fprintf(stderr, "    Gather:        %.2f ms\n", ms[3]);
            fprintf(stderr, "    GU+SiLU+Had:   %.2f ms\n", ms[4]);
            fprintf(stderr, "    DN GEMM:       %.2f ms\n", ms[5]);
            fprintf(stderr, "    Scatter:       %.2f ms\n", ms[6]);
            fprintf(stderr, "    MoE total:     %.2f ms (×48 = %.0f ms)\n",
                    ms[0]+ms[1]+ms[2]+ms[3]+ms[4]+ms[5]+ms[6],
                    (ms[0]+ms[1]+ms[2]+ms[3]+ms[4]+ms[5]+ms[6])*48);
            for (auto& e : moe_ev) hipEventDestroy(e);
        }
        if (li == 0 || M >= 8192) { auto e = hipGetLastError(); fprintf(stderr, "  L%d scatter %s\n", li, e==hipSuccess?"OK":hipGetErrorString(e)); }

        // 8. Final residual: x = residual + moe_out
        // DISABLED: record_start();
        auto* moe_p = reinterpret_cast<const __half*>(moe_out.data_ptr());
        if (li < num_layers - 1) {
            hipLaunchKernelGGL(fused_res_norm_batched, dim3(M), dim3(norm_bs), smem_norm, stream,
                moe_p, r_p, in_np[li + 1], n_p, x_p, D, rms_eps);
            if (li <= 1 || M >= 8192) { hipDeviceSynchronize(); auto e = hipGetLastError(); fprintf(stderr, "  L%d final_res_norm %s\n", li, e==hipSuccess?"OK":hipGetErrorString(e)); }
        } else {
            // Last layer: x = residual + moe_out (pure HIP)
            int total = M * D;
            hipLaunchKernelGGL(vadd_fp16, dim3((total+255)/256), dim3(256), 0, stream,
                r_p, moe_p, x_p, total);
        }
    }

    // Final RMSNorm + Hadamard + LM head (LAST TOKEN ONLY to save VRAM)
    auto* fn_p = reinterpret_cast<const __half*>(final_norm_w.data_ptr());
    // Only process last token for LM head (saves M × vocab × 2 bytes)
    __half* last_tok_x = x_p + (int64_t)(M - 1) * D;
    auto last_normed = torch::empty({1, (long)D}, opts);
    auto* last_n_p = reinterpret_cast<__half*>(last_normed.data_ptr());
    hipLaunchKernelGGL(rmsnorm_batched, dim3(1), dim3(norm_bs), smem_norm, stream,
        last_tok_x, fn_p, last_n_p, D, rms_eps);

    auto rot_final = torch::empty({1, (long)D}, opts);
    auto* rot_f_p = reinterpret_cast<__half*>(rot_final.data_ptr());
    launch_hadamard_batched(last_n_p, rot_f_p, had_p, 1, D, stream);

    // LM head: [1, vocab] instead of [M, vocab] — saves ~9 GB at 32k context
    auto logits = torch::empty({1, (long)lm_N}, opts);
    dequant_matmul_out_g64_ts(rot_final, logits,
        lm_w.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(lm_s.data_ptr()),
        (int)lm_N, (int)lm_K, lm_w_tmp, stream);

    return logits;  // [1, vocab] (last token only)
}
// ============================================================
// dequant_int4_g64 — group-64 variant (scale index = kh / 32)
// S is interleaved (scale, zero_fp16) pairs, [N, K/64, 2].
// ============================================================
__global__ void dequant_int4_g64_kernel(
    const uint8_t* __restrict__ packed,   // [N, K/2]
    const __half* __restrict__ scales,    // [N, K/64, 2] interleaved (scale, zero)
    __half* __restrict__ out,             // [N, K]
    int N, int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int Kh = K / 2;
    if (idx >= N * Kh) return;
    const int n = idx / Kh;
    const int kh = idx % Kh;
    const uint8_t p = packed[idx];
    int lo = (p & 0xF);
    int hi = (p >> 4);
    const int block_idx = kh / 32;  // group-64: 32 bytes = 64 elements
    __half2 sz = *reinterpret_cast<const __half2*>(&scales[(n * (K / 64) + block_idx) * 2]);
    const float s = __half2float(sz.x);
    const float z = __half2float(sz.y);
    out[(int64_t)n * K + kh * 2]     = __float2half(((float)lo - z) * s);
    out[(int64_t)n * K + kh * 2 + 1] = __float2half(((float)hi - z) * s);
}

torch::Tensor dequant_int4_g64(torch::Tensor packed, torch::Tensor scales,
                               int64_t N, int64_t K) {
    auto out = torch::empty({N, K}, torch::dtype(torch::kHalf).device(packed.device()));
    auto stream = at::hip::getCurrentHIPStream();
    int total = N * (K / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(dequant_int4_g64_kernel, dim3(blocks), dim3(threads), 0, stream,
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        N, K);
    return out;
}

void dequant_gemm_out_g64(torch::Tensor input, torch::Tensor packed, torch::Tensor scales,
                          int64_t N, int64_t K,
                          torch::Tensor w_buf, torch::Tensor out) {
    auto stream = at::hip::getCurrentHIPStream();
    int total = N * (K / 2);
    int threads = 256, blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(dequant_int4_g64_kernel, dim3(blocks), dim3(threads), 0, stream,
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr()),
        reinterpret_cast<__half*>(w_buf.data_ptr()),
        N, K);
    auto w_fp16 = w_buf.narrow(0, 0, N * K).view({N, K});
    at::mm_out(out, input, w_fp16.t());
}

// Dequant INT4 g64 into pre-allocated buffer on a specific stream (for overlapped prefill)
void dequant_into_g64(torch::Tensor packed, torch::Tensor scales,
                      int64_t N, int64_t K, torch::Tensor w_buf) {
    auto stream = at::hip::getCurrentHIPStream();
    int total = N * (K / 2);
    int threads = 256, blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(dequant_int4_g64_kernel, dim3(blocks), dim3(threads), 0, stream,
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr()),
        reinterpret_cast<__half*>(w_buf.data_ptr()),
        N, K);
}

// GEMM with pre-dequanted FP16 weights (no dequant step)
void gemm_dequanted_g64(torch::Tensor input, torch::Tensor w_buf,
                        int64_t N, int64_t K, torch::Tensor out) {
    auto w_fp16 = w_buf.narrow(0, 0, N * K).view({N, K});
    at::mm_out(out, input, w_fp16.t());
}
// Tiled dequant+GEMM: dequant in N-tiles to keep FP16 intermediate in L2 cache.
// ~37% faster than full-matrix dequant+GEMM for large N (gate_up N=34816).
void dequant_gemm_tiled_g64(torch::Tensor input, torch::Tensor packed, torch::Tensor scales,
                            int64_t N, int64_t K,
                            torch::Tensor w_buf, torch::Tensor out,
                            int64_t N_tile = 4096) {
    auto stream = at::hip::getCurrentHIPStream();
    int M = input.size(0);
    const int Kh = K / 2;
    const int ng = K / 64;

    for (int64_t n_start = 0; n_start < N; n_start += N_tile) {
        int64_t ns = std::min(N_tile, N - n_start);
        int total = ns * Kh;
        int threads = 256, blocks = (total + threads - 1) / threads;

        // Dequant this strip into w_buf (row-major scale layout)
        hipLaunchKernelGGL(dequant_int4_g64_rm_kernel, dim3(blocks), dim3(threads), 0, stream,
            packed.data_ptr<uint8_t>() + n_start * Kh,
            reinterpret_cast<const __half*>(scales.data_ptr()) + n_start * ng * 2,
            reinterpret_cast<__half*>(w_buf.data_ptr()),
            ns, K);

        // GEMM: out[:, n_start:n_start+ns] = input @ w_strip.T
        auto w_strip = w_buf.narrow(0, 0, ns * K).view({ns, K});
        auto out_strip = out.narrow(1, n_start, ns);
        at::mm_out(out_strip, input, w_strip.t());
    }
}

// Tiled dequant+GEMM for group-32: same L2-friendly tiling as g64 variant.
void dequant_gemm_tiled_g32(torch::Tensor input, torch::Tensor packed, torch::Tensor scales,
                            int64_t N, int64_t K,
                            torch::Tensor w_buf, torch::Tensor out,
                            int64_t N_tile = 4096) {
    auto stream = at::hip::getCurrentHIPStream();
    int M = input.size(0);
    const int Kh = K / 2;
    const int ng = K / 32;

    for (int64_t n_start = 0; n_start < N; n_start += N_tile) {
        int64_t ns = std::min(N_tile, N - n_start);
        int total = ns * Kh;
        int threads = 256, blocks = (total + threads - 1) / threads;

        // Dequant this strip into w_buf
        hipLaunchKernelGGL(dequant_int4_kernel, dim3(blocks), dim3(threads), 0, stream,
            packed.data_ptr<uint8_t>() + n_start * Kh,
            reinterpret_cast<const __half*>(scales.data_ptr()) + n_start * ng * 2,
            reinterpret_cast<__half*>(w_buf.data_ptr()),
            ns, K);

        // GEMM: out[:, n_start:n_start+ns] = input @ w_strip.T
        auto w_strip = w_buf.narrow(0, 0, ns * K).view({ns, K});
        auto out_strip = out.narrow(1, n_start, ns);
        at::mm_out(out_strip, input, w_strip.t());
    }
}
// Batch convert FP16 → FP8 E4M3: src [N] FP16 → dst [N] uint8
__global__ void fp16_to_fp8_kernel(const __half* src, uint8_t* dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = f32_to_fp8e4m3(__half2float(src[idx]));
}

torch::Tensor fp16_to_fp8(torch::Tensor src) {
    auto out = torch::empty(src.sizes(), torch::dtype(torch::kUInt8).device(src.device()));
    int N = src.numel();
    auto stream = at::hip::getCurrentHIPStream();
    hipLaunchKernelGGL(fp16_to_fp8_kernel, dim3((N + 255) / 256), dim3(256), 0, stream,
        reinterpret_cast<const __half*>(src.data_ptr()), out.data_ptr<uint8_t>(), N);
    return out;
}
// ============================================================
// MoE GEMV — top-k expert dispatch for batch=1 decode
//
// For each of the top-k active experts:
//   1. GEMV gate+up (K=hidden, N=2*moe_inter)
//   2. Fused SiLU+FWHT+GEMV down (K=moe_inter, N=hidden)
//   3. Weighted accumulate: moe_out += weight * down_out
//
// Inputs:
//   normed       [D=hidden] FP16 — already RMSNormed hidden state
//   exp_gu_packed [E, N_gu=2*moe_inter, K_gu/2] uint8  — stacked gate+up
//   exp_gu_scales [E, N_gu, K_gu/16] FP16               — interleaved (scale, zero)
//   exp_dn_packed [E, N_dn=hidden, K_dn/2] uint8        — stacked down
//   exp_dn_scales [E, N_dn, K_dn/16] FP16
//   top_k_ids    [k] int64 — pre-computed routing indices
//   top_k_weights [k] FP16 — routing probabilities (sum=1)
//   moe_inter    scalar int
//   hidden       scalar int
//
// Returns: moe_out [hidden] FP16
// ============================================================

// Tiny kernel: Y[i] += alpha * X[i] (weighted accumulate)
__global__ void scale_accumulate_kernel(
    __half* __restrict__ Y, const __half* __restrict__ X, float alpha, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        Y[i] = __float2half(__half2float(Y[i]) + alpha * __half2float(X[i]));
}

static inline void launch_scale_accumulate(
    __half* Y, const __half* X, float alpha, int N, hipStream_t st) {
    hipLaunchKernelGGL(scale_accumulate_kernel,
        dim3((N + 255) / 256), dim3(256), 0, st, Y, X, alpha, N);
}

// Reads alpha from GPU float pointer — no CPU-GPU sync needed.
__global__ void scale_accumulate_ptr_kernel(
    __half* __restrict__ Y, const __half* __restrict__ X,
    const float* __restrict__ alpha_ptr, int N)
{
    float alpha = *alpha_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        Y[i] = __float2half(__half2float(Y[i]) + alpha * __half2float(X[i]));
}

static inline void launch_scale_accumulate_ptr(
    __half* Y, const __half* X, const float* alpha_ptr, int N, hipStream_t st) {
    hipLaunchKernelGGL(scale_accumulate_ptr_kernel,
        dim3((N + 255) / 256), dim3(256), 0, st, Y, X, alpha_ptr, N);
}

// ============================================================
// Indirect GEMV kernels: read expert index j from GPU memory.
// W_base/S_base point to the start of the stacked [E, N, K/2] tensor.
// expert_id_ptr points to a single int64 in GPU memory (top_k_ids[rank]).
// Eliminates CPU-GPU sync vs. the non-indirect versions above.
// ============================================================
template<int BLOCK_N>
__global__ void gemv_warp_indirect(
    const __half* __restrict__ X,
    const uint8_t* __restrict__ W_base,
    const __half* __restrict__ S_base,
    __half* __restrict__ Y,
    const int64_t* __restrict__ expert_id_ptr,
    int64_t packed_stride,   // uint8 elements per expert = N * K/2
    int64_t scales_stride,   // FP16 elements per expert = N * K/16 * 2
    int N, int K
) {
    int64_t j = *expert_id_ptr;
    const uint8_t* W = W_base + j * packed_stride;
    const __half*  S = S_base + j * scales_stride;

    const int n0 = blockIdx.x * BLOCK_N;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / 32, nit = K / 32;
    float acc[BLOCK_N];
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i++) acc[i] = 0.0f;

    for (int it = lane; it < nit; it += 32) {
        float xv[32];
        #pragma unroll
        for (int i = 0; i < 32; i += 2) {
            __half2 xh = *reinterpret_cast<const __half2*>(&X[it * 32 + i]);
            xv[i] = __half2float(xh.x); xv[i+1] = __half2float(xh.y);
        }
        #pragma unroll
        for (int ni = 0; ni < BLOCK_N; ni++) {
            int n = n0 + ni;
            if (n >= N) continue;
            __half2 sz = *reinterpret_cast<const __half2*>(&S[(n * nsc + it) * 2]);
            float scale = __half2float(sz.x);
            float zp = __half2float(sz.y);
            const uint8_t* wr = W + n * Kh + it * 16;
            uint4 w128 = *reinterpret_cast<const uint4*>(wr);
            const uint8_t* wp = reinterpret_cast<const uint8_t*>(&w128);
            float dot = 0.0f;
            #pragma unroll
            for (int b = 0; b < 16; b++) {
                uint8_t p = wp[b];
                float lo = (float)(p & 0xF) - zp;
                float hi = (float)((p >> 4) & 0xF) - zp;
                dot += xv[b*2] * lo + xv[b*2+1] * hi;
            }
            acc[ni] += dot * scale;
        }
    }
    #pragma unroll
    for (int ni = 0; ni < BLOCK_N; ni++) {
        acc[ni] += __shfl_xor(acc[ni], 16); acc[ni] += __shfl_xor(acc[ni], 8);
        acc[ni] += __shfl_xor(acc[ni], 4);  acc[ni] += __shfl_xor(acc[ni], 2);
        acc[ni] += __shfl_xor(acc[ni], 1);
        if (lane == 0 && n0 + ni < N) Y[n0 + ni] = __float2half(acc[ni]);
    }
}

// Indirect fused SiLU*Up + FWHT + GEMV — reads expert j from GPU pointer.
template<int BLOCK_N>
__global__ void gemv_warp_silu_had_indirect(
    const __half* __restrict__ Gate,
    const __half* __restrict__ Up,
    const uint8_t* __restrict__ W_base,
    const __half* __restrict__ S_base,
    __half* __restrict__ Y,
    const int64_t* __restrict__ expert_id_ptr,
    int64_t packed_stride,
    int64_t scales_stride,
    int N, int K
) {
    int64_t j = *expert_id_ptr;
    const uint8_t* W = W_base + j * packed_stride;
    const __half*  S = S_base + j * scales_stride;

    const int n0 = blockIdx.x * BLOCK_N;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / 32, nit = K / 32;
    float acc[BLOCK_N];
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i++) acc[i] = 0.0f;

    for (int it = lane; it < nit; it += 32) {
        float xv[32];
        #pragma unroll
        for (int i = 0; i < 32; i += 2) {
            __half2 gh = *reinterpret_cast<const __half2*>(&Gate[it * 32 + i]);
            __half2 uh = *reinterpret_cast<const __half2*>(&Up[it * 32 + i]);
            float g0 = __half2float(gh.x), g1 = __half2float(gh.y);
            float u0 = __half2float(uh.x), u1 = __half2float(uh.y);
            xv[i]   = g0 / (1.0f + expf(-g0)) * u0;
            xv[i+1] = g1 / (1.0f + expf(-g1)) * u1;
        }
        fwht_inplace_32(xv);

        #pragma unroll
        for (int ni = 0; ni < BLOCK_N; ni++) {
            int n = n0 + ni;
            if (n >= N) continue;
            __half2 sz = *reinterpret_cast<const __half2*>(&S[(n * nsc + it) * 2]);
            float scale = __half2float(sz.x);
            float zp = __half2float(sz.y);
            const uint8_t* wr = W + n * Kh + it * 16;
            uint4 w128 = *reinterpret_cast<const uint4*>(wr);
            const uint8_t* wp = reinterpret_cast<const uint8_t*>(&w128);
            float dot = 0.0f;
            #pragma unroll
            for (int b = 0; b < 16; b++) {
                uint8_t p = wp[b];
                float lo = (float)(p & 0xF) - zp;
                float hi = (float)((p >> 4) & 0xF) - zp;
                dot += xv[b*2] * lo + xv[b*2+1] * hi;
            }
            acc[ni] += dot * scale;
        }
    }
    #pragma unroll
    for (int ni = 0; ni < BLOCK_N; ni++) {
        acc[ni] += __shfl_xor(acc[ni], 16); acc[ni] += __shfl_xor(acc[ni], 8);
        acc[ni] += __shfl_xor(acc[ni], 4);  acc[ni] += __shfl_xor(acc[ni], 2);
        acc[ni] += __shfl_xor(acc[ni], 1);
        if (lane == 0 && n0 + ni < N) Y[n0 + ni] = __float2half(acc[ni]);
    }
}
template<int ROWS_PER_WARP>
__global__ void gemv_mw_batch_g64(
    const __half*   __restrict__ X,
    const uint8_t*  __restrict__ W_base,
    const __half*   __restrict__ S_base,   // [E, K/64, N, 2] FP16 interleaved
    __half* __restrict__ Y_base,
    const int64_t*  __restrict__ tids,
    int64_t packed_stride, int64_t scales_stride, int64_t out_stride,
    int N, int K
) {
    int rank = blockIdx.y;
    int64_t j = tids[rank];
    const uint8_t* W = W_base + j * packed_stride;
    const __half*  S = S_base + j * scales_stride;
    __half* Y = Y_base + (int64_t)rank * out_stride;

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int n0      = (blockIdx.x * 4 + warp_id) * ROWS_PER_WARP;
    const int nit     = K / 32;
    float acc[ROWS_PER_WARP];
    #pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; i++) acc[i] = 0.0f;

    for (int it = lane; it < nit; it += 32) {
        if (n0 >= N) continue;
        const __half* Xit = X + it * 32;

        // Fused dot+x_sum: single pass over X from L0 cache
        const uint8_t* wr = W + (int64_t)n0 * (K / 2) + it * 16;
        uint4 w128 = *reinterpret_cast<const uint4*>(wr);
        const uint8_t* wp = reinterpret_cast<const uint8_t*>(&w128);
        float raw_dot = 0.0f, x_sum = 0.0f;
        #pragma unroll
        for (int b = 0; b < 16; b++) {
            uint8_t p = wp[b];
            float2 xf = __half22float2(*reinterpret_cast<const __half2*>(&Xit[b * 2]));
            raw_dot += xf.x * (float)(p & 0xF) + xf.y * (float)((p >> 4) & 0xF);
            x_sum += xf.x + xf.y;
        }

        // Scale correction (row-major scale format)
        __half2 sz = *reinterpret_cast<const __half2*>(&S[((int64_t)n0 * (nit >> 1) + (it >> 1)) * 2]);
        float sc = __half2float(sz.x);
        float zp = __half2float(sz.y);
        acc[0] += sc * (raw_dot - zp * x_sum);
    }
    acc[0] += __shfl_xor(acc[0], 16); acc[0] += __shfl_xor(acc[0], 8);
    acc[0] += __shfl_xor(acc[0], 4);  acc[0] += __shfl_xor(acc[0], 2);
    acc[0] += __shfl_xor(acc[0], 1);
    if (lane == 0 && n0 < N) Y[n0] = __float2half(acc[0]);
}
// ============================================================
// Batched SiLU + FWHT kernel (separated from GEMV for lower VGPR pressure).
// Each lane holds one element; FWHT butterfly via __shfl_xor.
// Grid: (K_dn / 32, num_experts), Block: (32)
// ============================================================
__global__ void silu_fwht_batch(
    const __half* __restrict__ GU_base,  // [k, 2*K_dn] — gate in [:K], up in [K:]
    __half* __restrict__ OUT_base,       // [k, K_dn]
    int64_t gu_stride,                   // = 2*K_dn (half elements between experts)
    int K                                // K_dn
) {
    int group = blockIdx.x;   // which group of 32
    int rank  = blockIdx.y;   // which expert
    int lane  = threadIdx.x;  // 0..31

    const __half* Gate = GU_base + (int64_t)rank * gu_stride + group * 32;
    const __half* Up   = Gate + K;
    __half* Out = OUT_base + (int64_t)rank * K + group * 32;

    // Each lane loads its own gate+up element, computes SiLU(gate)*up
    float g = __half2float(Gate[lane]);
    float u = __half2float(Up[lane]);
    float val = g / (1.0f + expf(-g)) * u;

    // FWHT butterfly via warp shuffles — all 32 lanes participate
    #pragma unroll
    for (int step = 1; step < 32; step *= 2) {
        float partner = __shfl_xor(val, step);
        float orig = val;
        val = (lane & step) ? (partner - orig) : (orig + partner);
    }
    val *= 0.17677669529663689f;  // 1/sqrt(32)

    Out[lane] = __float2half(val);
}
// ============================================================
// Group-64 batched GEMV with per-expert input (X_base indexed by rank).
// Used for down-projection after silu_fwht_batch splits the FWHT out.
// ============================================================
template<int BLOCK_N>
__global__ __launch_bounds__(32, 2) void gemv_warp_batch_xbatch_g64(
    const __half*   __restrict__ X_base,   // [k, K] — per-expert input
    const uint8_t*  __restrict__ W_base,
    const __half*   __restrict__ S_base,   // [E, K/64, N, 2] FP16 interleaved
    __half* __restrict__ Y_base,
    const int64_t*  __restrict__ tids,
    int64_t x_stride,                      // K (half elements between expert inputs)
    int64_t packed_stride, int64_t scales_stride, int64_t out_stride,
    int N, int K
) {
    int rank = blockIdx.y;
    int64_t j = tids[rank];
    const __half*  X = X_base + (int64_t)rank * x_stride;
    const uint8_t* W = W_base + j * packed_stride;
    const __half*  S = S_base + j * scales_stride;
    __half* Y = Y_base + (int64_t)rank * out_stride;

    const int n0   = blockIdx.x * BLOCK_N;
    const int lane = threadIdx.x;
    const int nit  = K / 32;
    float acc[BLOCK_N];
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i++) acc[i] = 0.0f;

    for (int it = lane; it < nit; it += 32) {
        __half2 xh[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            xh[i] = *reinterpret_cast<const __half2*>(&X[it * 32 + i * 2]);
        }
        // Load group-64 scale at it/2 (row-major: S[N, K/64, 2])
        const int ngrp = nit >> 1;
        __half sz_data[BLOCK_N * 2];
        #pragma unroll
        for (int ni = 0; ni < BLOCK_N; ni++) {
            *reinterpret_cast<__half2*>(&sz_data[ni * 2]) =
                *reinterpret_cast<const __half2*>(&S[((int64_t)(n0 + ni) * ngrp + (it >> 1)) * 2]);
        }
        float x_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float2 xf = __half22float2(xh[i]);
            x_sum += xf.x + xf.y;
        }
        float sc[BLOCK_N], zp_xsum[BLOCK_N];
        #pragma unroll
        for (int ni = 0; ni < BLOCK_N; ni++) {
            sc[ni]      = __half2float(sz_data[ni * 2]);
            float zp    = __half2float(sz_data[ni * 2 + 1]);
            zp_xsum[ni] = zp * x_sum;
        }
        #pragma unroll
        for (int ni = 0; ni < BLOCK_N; ni++) {
            const uint8_t* wr = W + (n0 + ni) * (K / 2) + it * 16;
            uint4 w128 = *reinterpret_cast<const uint4*>(wr);
            const uint8_t* wp = reinterpret_cast<const uint8_t*>(&w128);
            float raw_dot = 0.0f;
            #pragma unroll
            for (int b = 0; b < 16; b++) {
                uint8_t p = wp[b];
                float2 xf = __half22float2(xh[b]);
                raw_dot += xf.x * (float)(p & 0xF) + xf.y * (float)((p >> 4) & 0xF);
            }
            acc[ni] += sc[ni] * (raw_dot - zp_xsum[ni]);
        }
    }
    #pragma unroll
    for (int ni = 0; ni < BLOCK_N; ni++) {
        acc[ni] += __shfl_xor(acc[ni], 16); acc[ni] += __shfl_xor(acc[ni], 8);
        acc[ni] += __shfl_xor(acc[ni], 4);  acc[ni] += __shfl_xor(acc[ni], 2);
        acc[ni] += __shfl_xor(acc[ni], 1);
        if (lane == 0 && n0 + ni < N) Y[n0 + ni] = __float2half(acc[ni]);
    }
}
__global__ void fused_moe_reduce_norm_fwht(
    const __half* __restrict__ batch_dn,  // [k, D]
    const float*  __restrict__ weights,   // [k]
    const __half* __restrict__ R,         // [D] residual
    const __half* __restrict__ NormW,     // [D] norm weights
    __half* __restrict__ Y_rot,           // [D] FWHT output
    __half* __restrict__ H_out,           // [D] hidden state
    int k, int D, float eps
) {
    const int tid = threadIdx.x;
    const int nwarps = blockDim.x / 32;
    const int wid = tid / 32;
    const int lane = tid % 32;

    // Load expert weights into shared memory
    __shared__ float sw[16];
    if (tid < k) sw[tid] = weights[tid];
    __syncthreads();

    // Phase 1: weighted sum + residual + sum-of-squares
    float ss = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float sum = 0.0f;
        for (int r = 0; r < k; r++)
            sum += sw[r] * __half2float(batch_dn[(int64_t)r * D + i]);
        float ri = __half2float(R[i]) + sum;
        H_out[i] = __float2half(ri);
        ss += ri * ri;
    }
    // Warp reduce
    ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
    ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
    __shared__ float ws_mnf[32];
    if (lane == 0) ws_mnf[wid] = ss;
    __syncthreads();
    float rrms;
    if (wid == 0) {
        ss = (lane < nwarps) ? ws_mnf[lane] : 0.0f;
        ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
        ss += __shfl_xor(ss, 4); ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
        if (lane == 0) ws_mnf[0] = rsqrtf(ss / (float)D + eps);
    }
    __syncthreads();
    rrms = ws_mnf[0];

    // Phase 2: norm + FWHT
    int num_blocks = D / 32;
    for (int blk = wid; blk < num_blocks; blk += nwarps) {
        int base = blk * 32;
        float normed = __half2float(H_out[base + lane]) * rrms * __half2float(NormW[base + lane]);
        #pragma unroll
        for (int step = 1; step < 32; step *= 2) {
            float other = __shfl_xor(normed, step);
            if (lane & step) normed = other - normed;
            else normed = normed + other;
        }
        normed *= 0.17677669529663689f;
        Y_rot[base + lane] = __float2half(normed);
    }
}

static inline void launch_moe_reduce_norm_fwht(
    const __half* batch_dn, const float* weights, const __half* R,
    const __half* NormW, __half* Y_rot, __half* H_out,
    int k, int D, float eps, hipStream_t st) {
    hipLaunchKernelGGL(fused_moe_reduce_norm_fwht,
        dim3(1), dim3(1024), 0, st,
        batch_dn, weights, R, NormW, Y_rot, H_out, k, D, eps);
}

// Simple MOE reduce + residual: H[i] = R[i] + sum(weights[r] * DN[r*D + i])
// No norm or FWHT — those are deferred to the next layer's fused QKV GEMV.
__global__ void moe_reduce_res(
    const __half* __restrict__ batch_dn,  // [k, D]
    const float*  __restrict__ weights,   // [k]
    const __half* __restrict__ R,         // [D] residual
    __half* __restrict__ H_out,           // [D] hidden state output
    int k, int D
) {
    const int tid = threadIdx.x;
    __shared__ float sw[16];
    if (tid < k) sw[tid] = weights[tid];
    __syncthreads();
    for (int i = tid; i < D; i += blockDim.x) {
        float sum = 0.0f;
        for (int r = 0; r < k; r++)
            sum += sw[r] * __half2float(batch_dn[(int64_t)r * D + i]);
        H_out[i] = __float2half(__half2float(R[i]) + sum);
    }
}

static inline void launch_moe_reduce_res(
    const __half* batch_dn, const float* weights, const __half* R,
    __half* H_out, int k, int D, hipStream_t st) {
    hipLaunchKernelGGL(moe_reduce_res,
        dim3(1), dim3(1024), 0, st,
        batch_dn, weights, R, H_out, k, D);
}
// ============================================================
// Graph-compatible FP16-KV variants (read pos/seq_len from device globals)
// ============================================================
__global__ void head_rmsnorm_rope_kv_fp16_g(
    const __half* Q, const __half* K, const __half* V,
    const __half* QW, const __half* KW,
    __half* Q_out, __half* K_out,
    __half* K_cache, __half* V_cache,
    const __half* COS_base, const __half* SIN_base,
    int H, int Hk, int D, float eps, int max_seq
) {
    int pos = g_graph_pos;
    int half_D = D / 2;
    const __half* COS = COS_base + pos * half_D;
    const __half* SIN = SIN_base + pos * half_D;

    int pid = blockIdx.x;
    int tid = threadIdx.x;

    if (pid < H) {
        const __half* src = Q + pid * D;
        __half* dst = Q_out + pid * D;
        float sq = 0.0f;
        for (int i = tid; i < D; i += 32) {
            float v = __half2float(src[i]); sq += v * v;
        }
        sq += __shfl_xor(sq, 16); sq += __shfl_xor(sq, 8);
        sq += __shfl_xor(sq, 4);  sq += __shfl_xor(sq, 2); sq += __shfl_xor(sq, 1);
        float rrms = rsqrtf(sq / (float)D + eps);
        for (int i = tid; i < half_D; i += 32) {
            float ne = __half2float(src[i]) * rrms * __half2float(QW[i]);
            float no = __half2float(src[i + half_D]) * rrms * __half2float(QW[i + half_D]);
            float c = __half2float(COS[i]), s = __half2float(SIN[i]);
            dst[i]          = __float2half(ne * c - no * s);
            dst[i + half_D] = __float2half(no * c + ne * s);
        }
    }
    int kid = pid - H;
    if (kid >= 0 && kid < Hk) {
        const __half* src = K + kid * D;
        __half* dst = K_out + kid * D;
        int cache_off = kid * max_seq * D + pos * D;
        float sq = 0.0f;
        for (int i = tid; i < D; i += 32) {
            float v = __half2float(src[i]); sq += v * v;
        }
        sq += __shfl_xor(sq, 16); sq += __shfl_xor(sq, 8);
        sq += __shfl_xor(sq, 4);  sq += __shfl_xor(sq, 2); sq += __shfl_xor(sq, 1);
        float rrms = rsqrtf(sq / (float)D + eps);
        for (int i = tid; i < half_D; i += 32) {
            float ne = __half2float(src[i]) * rrms * __half2float(KW[i]);
            float no = __half2float(src[i + half_D]) * rrms * __half2float(KW[i + half_D]);
            float c = __half2float(COS[i]), s = __half2float(SIN[i]);
            float k_lo = ne * c - no * s;
            float k_hi = no * c + ne * s;
            dst[i]          = __float2half(k_lo);
            dst[i + half_D] = __float2half(k_hi);
            K_cache[cache_off + i]          = __float2half(k_lo);
            K_cache[cache_off + i + half_D] = __float2half(k_hi);
        }
        for (int i = tid; i < D; i += 32)
            V_cache[cache_off + i] = V[kid * D + i];
    }
}

// FP8 E4M3 variant: identical logic but writes uint8 KV cache (50% less bandwidth)
__global__ void head_rmsnorm_rope_kv_fp8_g(
    const __half* Q, const __half* K, const __half* V,
    const __half* QW, const __half* KW,
    __half* Q_out, __half* K_out,
    uint8_t* K_cache_fp8, uint8_t* V_cache_fp8,
    const __half* COS_base, const __half* SIN_base,
    int H, int Hk, int D, float eps, int max_seq
) {
    int pos = g_graph_pos;
    int half_D = D / 2;
    const __half* COS = COS_base + pos * half_D;
    const __half* SIN = SIN_base + pos * half_D;
    int pid = blockIdx.x, tid = threadIdx.x;

    if (pid < H) {  // Q path — identical to FP16
        const __half* src = Q + pid * D;
        __half* dst = Q_out + pid * D;
        float sq = 0.0f;
        for (int i = tid; i < D; i += 32) { float v = __half2float(src[i]); sq += v * v; }
        sq += __shfl_xor(sq, 16); sq += __shfl_xor(sq, 8);
        sq += __shfl_xor(sq, 4);  sq += __shfl_xor(sq, 2); sq += __shfl_xor(sq, 1);
        float rrms = rsqrtf(sq / (float)D + eps);
        for (int i = tid; i < half_D; i += 32) {
            float ne = __half2float(src[i]) * rrms * __half2float(QW[i]);
            float no = __half2float(src[i + half_D]) * rrms * __half2float(QW[i + half_D]);
            float c = __half2float(COS[i]), s = __half2float(SIN[i]);
            dst[i]          = __float2half(ne * c - no * s);
            dst[i + half_D] = __float2half(no * c + ne * s);
        }
    }
    int kid = pid - H;
    if (kid >= 0 && kid < Hk) {
        const __half* src = K + kid * D;
        __half* dst = K_out + kid * D;
        int cache_off = kid * max_seq * D + pos * D;
        float sq = 0.0f;
        for (int i = tid; i < D; i += 32) { float v = __half2float(src[i]); sq += v * v; }
        sq += __shfl_xor(sq, 16); sq += __shfl_xor(sq, 8);
        sq += __shfl_xor(sq, 4);  sq += __shfl_xor(sq, 2); sq += __shfl_xor(sq, 1);
        float rrms = rsqrtf(sq / (float)D + eps);
        for (int i = tid; i < half_D; i += 32) {
            float ne = __half2float(src[i]) * rrms * __half2float(KW[i]);
            float no = __half2float(src[i + half_D]) * rrms * __half2float(KW[i + half_D]);
            float c = __half2float(COS[i]), s = __half2float(SIN[i]);
            float k_lo = ne * c - no * s, k_hi = no * c + ne * s;
            dst[i]          = __float2half(k_lo);
            dst[i + half_D] = __float2half(k_hi);
            // FP8 E4M3 cache write — 1 byte per element instead of 2
            K_cache_fp8[cache_off + i]          = f32_to_fp8e4m3(k_lo);
            K_cache_fp8[cache_off + i + half_D] = f32_to_fp8e4m3(k_hi);
        }
        for (int i = tid; i < D; i += 32)
            V_cache_fp8[cache_off + i] = f32_to_fp8e4m3(__half2float(V[kid * D + i]));
    }
}

// LDS-tiled flash decode with horizontal K/V sharing across GQA warps.
// LDS layout: [BLOCK_S * 128] __half for K, then [BLOCK_S * 128] __half for V
// Total LDS: BLOCK_S * 128 * 2 * 2 bytes = BLOCK_S * 512 bytes
// BLOCK_S=64 → 32 KB, BLOCK_S=128 → 64 KB (max per workgroup)
//
// Cooperative load: ALL warps load K/V into LDS once (1× global traffic).
// Inner loop: each warp reads K/V from LDS (5-cycle latency vs 30-cycle L0).
// Uses packed __half2 loads for efficiency and zero bank conflicts.
// High-occupancy variant: no limiter, best for short ctx (few splits)
template<int BLOCK_S>
__global__ void flash_decode_partial_fp16_g(
    const __half* __restrict__ Q,
    const __half* __restrict__ K_cache,
    const __half* __restrict__ V_cache,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_out,
    int H, int Hk, int max_seq, int D,
    int max_n_splits
) {
    // LDS: K tile [BLOCK_S][D] + V tile [BLOCK_S][D], both FP16
    // D=128 always for this model. Using constexpr for bit ops.
    constexpr int D_CONST = 128;
    constexpr int D_HALF2 = D_CONST / 2;  // 64 half2 per row
    constexpr int D_SHIFT = 6;             // log2(D_HALF2) = log2(64) = 6
    extern __shared__ __half lds_tile[];   // [2 * BLOCK_S * D_CONST]
    __half* lds_k = lds_tile;                         // [BLOCK_S * D_CONST]
    __half* lds_v = lds_tile + BLOCK_S * D_CONST;     // [BLOCK_S * D_CONST]

    int seq_len = g_graph_seq_len;
    const int kh = blockIdx.x;
    const int split_id = blockIdx.y;
    const int warp_id = threadIdx.y;
    const int lane = threadIdx.x;
    const int gqa_ratio = H / Hk;

    if (warp_id >= gqa_ratio) return;
    const int qh = kh * gqa_ratio + warp_id;

    int n_splits_actual = (seq_len + BLOCK_S - 1) / BLOCK_S;

    if (split_id >= n_splits_actual) {
        if (lane == 0) {
            partial_max[split_id * H + qh] = -1e30f;
            partial_sum[split_id * H + qh] = 0.0f;
        }
        for (int i = 0; i < 4; i++)
            partial_out[(long)split_id * H * D + (long)qh * D + lane + i * 32] = 0.0f;
        return;
    }

    const float scale = rsqrtf((float)D);
    const __half* q_row = Q + qh * D;
    float q_reg[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) q_reg[i] = __half2float(q_row[lane + i * 32]);

    const int s_start = split_id * BLOCK_S;
    const int s_end = min(s_start + BLOCK_S, seq_len);
    const int tile_len = s_end - s_start;

    const __half* k_base = K_cache + (long)kh * max_seq * D_CONST;
    const __half* v_base = V_cache + (long)kh * max_seq * D_CONST;

    const int tid = warp_id * 32 + lane;
    const int n_threads = gqa_ratio * 32;

    // Online softmax state — persists across ALL tiles
    float local_max = -1e30f, local_sum = 0.0f;
    float out_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // === SINGLE TILE: cooperative load + compute ===
    int total_h2 = tile_len * D_HALF2;
    for (int idx = tid; idx < total_h2; idx += n_threads) {
        int s_local = idx >> D_SHIFT;
        int d2 = idx & (D_HALF2 - 1);
        long g_off = (long)(s_start + s_local) * D_CONST + d2 * 2;
        int l_off = s_local * D_CONST + d2 * 2;
        *reinterpret_cast<__half2*>(&lds_k[l_off]) =
            *reinterpret_cast<const __half2*>(&k_base[g_off]);
        *reinterpret_cast<__half2*>(&lds_v[l_off]) =
            *reinterpret_cast<const __half2*>(&v_base[g_off]);
    }
    __syncthreads();

    for (int s = 0; s < tile_len; s++) {
        const __half* k_lds = lds_k + s * D_CONST;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            dot += q_reg[i] * __half2float(k_lds[lane + i * 32]);

        dot += __shfl_xor(dot, 16); dot += __shfl_xor(dot, 8);
        dot += __shfl_xor(dot, 4);  dot += __shfl_xor(dot, 2);
        dot += __shfl_xor(dot, 1);
        dot *= scale;

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float exp_s = expf(dot - new_max);
        local_sum = local_sum * correction + exp_s;

        const __half* v_lds = lds_v + s * D_CONST;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            out_reg[i] = out_reg[i] * correction +
                         exp_s * __half2float(v_lds[lane + i * 32]);
        local_max = new_max;
    }

    const long out_off = (long)split_id * H * D + (long)qh * D;
    #pragma unroll
    for (int i = 0; i < 4; i++) partial_out[out_off + lane + i * 32] = out_reg[i];
    if (lane == 0) {
        partial_max[split_id * H + qh] = local_max;
        partial_sum[split_id * H + qh] = local_sum;
    }
}

// Low-occupancy variant: 2 blocks/CU max. Better L1 reuse for large ctx.
// Identical code but __launch_bounds__ forces compiler to use fewer VGPRs.
template<int BLOCK_S>
__global__ __launch_bounds__(160, 2)
void flash_decode_partial_fp16_g_lo(
    const __half* __restrict__ Q,
    const __half* __restrict__ K_cache,
    const __half* __restrict__ V_cache,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_out,
    int H, int Hk, int max_seq, int D,
    int max_n_splits
) {
    constexpr int D_CONST = 128, D_HALF2 = 64, D_SHIFT = 6;
    extern __shared__ __half lds_tile[];
    __half* lds_k = lds_tile;
    __half* lds_v = lds_tile + BLOCK_S * D_CONST;
    int seq_len = g_graph_seq_len;
    const int kh = blockIdx.x, split_id = blockIdx.y;
    const int warp_id = threadIdx.y, lane = threadIdx.x;
    const int gqa_ratio = H / Hk;
    if (warp_id >= gqa_ratio) return;
    const int qh = kh * gqa_ratio + warp_id;
    int n_splits_actual = (seq_len + BLOCK_S - 1) / BLOCK_S;
    if (split_id >= n_splits_actual) {
        if (lane == 0) { partial_max[split_id*H+qh]=-1e30f; partial_sum[split_id*H+qh]=0.0f; }
        for (int i=0;i<4;i++) partial_out[(long)split_id*H*D+(long)qh*D+lane+i*32]=0.0f;
        return;
    }
    const float scale = rsqrtf((float)D);
    float q_reg[4];
    for (int i=0;i<4;i++) q_reg[i]=__half2float(Q[qh*D+lane+i*32]);
    const int s_start=split_id*BLOCK_S, s_end=min(s_start+BLOCK_S,seq_len), tile_len=s_end-s_start;
    const __half* k_base=K_cache+(long)kh*max_seq*D_CONST;
    const __half* v_base=V_cache+(long)kh*max_seq*D_CONST;
    const int tid=warp_id*32+lane, n_threads=gqa_ratio*32;
    int total_h2=tile_len*D_HALF2;
    for (int idx=tid;idx<total_h2;idx+=n_threads) {
        int s_local=idx>>D_SHIFT, d2=idx&(D_HALF2-1);
        long g_off=(long)(s_start+s_local)*D_CONST+d2*2;
        int l_off=s_local*D_CONST+d2*2;
        *reinterpret_cast<__half2*>(&lds_k[l_off])=*reinterpret_cast<const __half2*>(&k_base[g_off]);
        *reinterpret_cast<__half2*>(&lds_v[l_off])=*reinterpret_cast<const __half2*>(&v_base[g_off]);
    }
    __syncthreads();
    float local_max=-1e30f, local_sum=0.0f, out_reg[4]={0,0,0,0};
    for (int s=0;s<tile_len;s++) {
        const __half* k_lds=lds_k+s*D_CONST;
        float dot=0.0f;
        for (int i=0;i<4;i++) dot+=q_reg[i]*__half2float(k_lds[lane+i*32]);
        dot+=__shfl_xor(dot,16);dot+=__shfl_xor(dot,8);
        dot+=__shfl_xor(dot,4);dot+=__shfl_xor(dot,2);dot+=__shfl_xor(dot,1);
        dot*=scale;
        float new_max=fmaxf(local_max,dot), correction=expf(local_max-new_max), exp_s=expf(dot-new_max);
        local_sum=local_sum*correction+exp_s;
        const __half* v_lds=lds_v+s*D_CONST;
        for (int i=0;i<4;i++) out_reg[i]=out_reg[i]*correction+exp_s*__half2float(v_lds[lane+i*32]);
        local_max=new_max;
    }
    const long out_off=(long)split_id*H*D+(long)qh*D;
    for (int i=0;i<4;i++) partial_out[out_off+lane+i*32]=out_reg[i];
    if (lane==0) { partial_max[split_id*H+qh]=local_max; partial_sum[split_id*H+qh]=local_sum; }
}

// ============================================================
// FP8 KV cache flash decode — 50% less bandwidth, native V_CVT_F32_FP8
// KV stored as uint8 (FP8 E4M3FN). LDS tile = BLOCK_S * 128 bytes (vs 256 for FP16).
// Q remains FP16 (full precision for query).
// ============================================================
template<int BLOCK_S>
__global__ void flash_decode_partial_fp8_g(
    const __half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache_fp8,  // [Hk, max_seq, D] uint8 FP8
    const uint8_t* __restrict__ V_cache_fp8,  // [Hk, max_seq, D] uint8 FP8
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_out,
    int H, int Hk, int max_seq, int D,
    int max_n_splits
) {
    constexpr int D_CONST = 128;
    // LDS: K[BLOCK_S][128] + V[BLOCK_S][128] in FP8 = BLOCK_S * 256 bytes
    extern __shared__ uint8_t lds_fp8[];
    uint8_t* lds_k = lds_fp8;
    uint8_t* lds_v = lds_fp8 + BLOCK_S * D_CONST;

    int seq_len = g_graph_seq_len;
    const int kh = blockIdx.x;
    const int split_id = blockIdx.y;
    const int warp_id = threadIdx.y;
    const int lane = threadIdx.x;
    const int gqa_ratio = H / Hk;

    if (warp_id >= gqa_ratio) return;
    const int qh = kh * gqa_ratio + warp_id;

    int n_splits_actual = (seq_len + BLOCK_S - 1) / BLOCK_S;
    if (split_id >= n_splits_actual) {
        if (lane == 0) {
            partial_max[split_id * H + qh] = -1e30f;
            partial_sum[split_id * H + qh] = 0.0f;
        }
        for (int i = 0; i < 4; i++)
            partial_out[(long)split_id * H * D + (long)qh * D + lane + i * 32] = 0.0f;
        return;
    }

    const float scale = rsqrtf((float)D);
    const __half* q_row = Q + qh * D;
    float q_reg[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) q_reg[i] = __half2float(q_row[lane + i * 32]);

    const int s_start = split_id * BLOCK_S;
    const int s_end = min(s_start + BLOCK_S, seq_len);
    const int tile_len = s_end - s_start;

    const uint8_t* k_base = K_cache_fp8 + (long)kh * max_seq * D_CONST;
    const uint8_t* v_base = V_cache_fp8 + (long)kh * max_seq * D_CONST;

    // Cooperative load FP8 KV into LDS (1 byte per element, 4× coalesced)
    const int tid = warp_id * 32 + lane;
    const int n_threads = gqa_ratio * 32;
    const int total_bytes = tile_len * D_CONST;  // FP8: 1 byte per element
    for (int idx = tid * 4; idx < total_bytes; idx += n_threads * 4) {
        if (idx + 3 < total_bytes) {
            *reinterpret_cast<uint32_t*>(&lds_k[idx]) =
                *reinterpret_cast<const uint32_t*>(&k_base[s_start * D_CONST + idx]);
            *reinterpret_cast<uint32_t*>(&lds_v[idx]) =
                *reinterpret_cast<const uint32_t*>(&v_base[s_start * D_CONST + idx]);
        }
    }
    __syncthreads();

    float local_max = -1e30f, local_sum = 0.0f;
    float out_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = 0; s < tile_len; s++) {
        const uint8_t* k_fp8 = lds_k + s * D_CONST;

        // Q·K dot: convert FP8 K to float on-the-fly
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t kb = k_fp8[lane + i * 32];
            // FP8 E4M3FN → float via hardware CVT
            float kf;
            kf = fp8e4m3_to_f32(kb);
            dot += q_reg[i] * kf;
        }

        dot += __shfl_xor(dot, 16); dot += __shfl_xor(dot, 8);
        dot += __shfl_xor(dot, 4);  dot += __shfl_xor(dot, 2);
        dot += __shfl_xor(dot, 1);
        dot *= scale;

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float exp_s = expf(dot - new_max);
        local_sum = local_sum * correction + exp_s;

        const uint8_t* v_fp8 = lds_v + s * D_CONST;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t vb = v_fp8[lane + i * 32];
            float vf;
            vf = fp8e4m3_to_f32(vb);
            out_reg[i] = out_reg[i] * correction + exp_s * vf;
        }
        local_max = new_max;
    }

    const long out_off = (long)split_id * H * D + (long)qh * D;
    #pragma unroll
    for (int i = 0; i < 4; i++) partial_out[out_off + lane + i * 32] = out_reg[i];
    if (lane == 0) {
        partial_max[split_id * H + qh] = local_max;
        partial_sum[split_id * H + qh] = local_sum;
    }
}

// Launcher: choose FP16 or FP8 based on KV cache type
static inline void launch_flash_decode_fp16_g(
    const __half* Q, const __half* K_cache, const __half* V_cache,
    __half* O,
    float* p_max, float* p_sum, float* p_out,
    int num_heads, int num_kv_heads, int max_seq,
    int max_n_splits, int block_s, int head_dim,
    hipStream_t stream)
{
    int gqa_ratio = num_heads / num_kv_heads;

    // Use low-occupancy variant when many splits (large ctx) for better L1 reuse
    bool use_lo = (max_n_splits >= 16);  // ~1024+ ctx with bs=64

    #define LAUNCH_FP16_G(BS, KERN) \
        hipLaunchKernelGGL((KERN<BS>), \
            dim3(num_kv_heads, max_n_splits), dim3(32, gqa_ratio), \
            2 * (BS) * 128 * sizeof(__half), stream, \
            Q, K_cache, V_cache, p_max, p_sum, p_out, \
            num_heads, num_kv_heads, max_seq, head_dim, max_n_splits)

    if (use_lo) {
        switch (block_s) {
            case 128: LAUNCH_FP16_G(128, flash_decode_partial_fp16_g_lo); break;
            default:  LAUNCH_FP16_G(64,  flash_decode_partial_fp16_g_lo); break;
        }
    } else {
        switch (block_s) {
            case 128: LAUNCH_FP16_G(128, flash_decode_partial_fp16_g); break;
            default:  LAUNCH_FP16_G(64,  flash_decode_partial_fp16_g); break;
        }
    }
    #undef LAUNCH_FP16_G

    hipLaunchKernelGGL(flash_decode_reduce,
        dim3(num_heads), dim3(128), 0, stream,
        p_max, p_sum, p_out, O, num_heads, head_dim, max_n_splits);
}

// FP8 KV cache launcher — same interface, 50% less LDS
static inline void launch_flash_decode_fp8_g(
    const __half* Q, const uint8_t* K_fp8, const uint8_t* V_fp8,
    __half* O,
    float* p_max, float* p_sum, float* p_out,
    int num_heads, int num_kv_heads, int max_seq,
    int max_n_splits, int block_s, int head_dim,
    hipStream_t stream)
{
    int gqa_ratio = num_heads / num_kv_heads;

    // LDS: 2 * BLOCK_S * 128 bytes (FP8, not FP16)
    #define LAUNCH_FP8_G(BS) \
        hipLaunchKernelGGL((flash_decode_partial_fp8_g<BS>), \
            dim3(num_kv_heads, max_n_splits), dim3(32, gqa_ratio), \
            2 * (BS) * 128, stream, \
            Q, K_fp8, V_fp8, p_max, p_sum, p_out, \
            num_heads, num_kv_heads, max_seq, head_dim, max_n_splits)

    switch (block_s) {
        case 128: LAUNCH_FP8_G(128); break;
        default:  LAUNCH_FP8_G(64);  break;
    }
    #undef LAUNCH_FP8_G

    hipLaunchKernelGGL(flash_decode_reduce,
        dim3(num_heads), dim3(128), 0, stream,
        p_max, p_sum, p_out, O, num_heads, head_dim, max_n_splits);
}
torch::Tensor moe_gemv_top_k(
    torch::Tensor normed,         // [D=hidden] FP16
    torch::Tensor exp_gu_packed,  // [E, N_gu, K_gu/2] uint8
    torch::Tensor exp_gu_scales,  // [E, N_gu, K_gu/16] FP16  (interleaved sz)
    torch::Tensor exp_dn_packed,  // [E, N_dn, K_dn/2] uint8
    torch::Tensor exp_dn_scales,  // [E, N_dn, K_dn/16] FP16
    torch::Tensor top_k_ids,      // [k] int64
    torch::Tensor top_k_weights,  // [k] FP16
    int64_t moe_inter,
    int64_t hidden)
{
    TORCH_CHECK(normed.dim() == 1 && normed.size(0) == hidden,
        "normed must be [hidden]");
    TORCH_CHECK(exp_gu_packed.is_contiguous(), "exp_gu_packed must be contiguous");
    TORCH_CHECK(exp_dn_packed.is_contiguous(), "exp_dn_packed must be contiguous");

    auto stream = at::hip::getCurrentHIPStream();

    const int64_t N_gu = 2 * moe_inter;   // gate+up stacked output dim
    const int64_t K_gu = hidden;           // input dim for gate+up
    const int64_t N_dn = hidden;           // down output dim
    const int64_t K_dn = moe_inter;        // down input dim

    // Bytes per expert in stacked packed tensors
    const int64_t gu_expert_stride_bytes = N_gu * (K_gu / 2);   // uint8 bytes
    const int64_t dn_expert_stride_bytes = N_dn * (K_dn / 2);
    // FP16 elements per expert in stacked scales ([N, K/16] per expert)
    const int64_t gu_scales_stride = N_gu * (K_gu / 16);
    const int64_t dn_scales_stride = N_dn * (K_dn / 16);

    auto moe_out = torch::zeros({hidden},
        torch::dtype(torch::kHalf).device(normed.device()));
    auto gu_buf  = torch::empty({N_gu},
        torch::dtype(torch::kHalf).device(normed.device()));
    auto dn_buf  = torch::empty({N_dn},
        torch::dtype(torch::kHalf).device(normed.device()));

    const __half* normed_ptr = reinterpret_cast<const __half*>(normed.data_ptr());
    __half* moe_out_ptr = reinterpret_cast<__half*>(moe_out.data_ptr());
    __half* gu_buf_ptr  = reinterpret_cast<__half*>(gu_buf.data_ptr());
    __half* dn_buf_ptr  = reinterpret_cast<__half*>(dn_buf.data_ptr());

    const uint8_t* gu_packed_base = exp_gu_packed.data_ptr<uint8_t>();
    const __half*  gu_scales_base = reinterpret_cast<const __half*>(exp_gu_scales.data_ptr());
    const uint8_t* dn_packed_base = exp_dn_packed.data_ptr<uint8_t>();
    const __half*  dn_scales_base = reinterpret_cast<const __half*>(exp_dn_scales.data_ptr());

    int64_t k = top_k_ids.size(0);
    auto ids_cpu = top_k_ids.cpu();
    auto wts_cpu = top_k_weights.cpu().to(torch::kFloat32);

    for (int64_t rank = 0; rank < k; rank++) {
        int64_t j = ids_cpu[rank].item<int64_t>();
        float   w = wts_cpu[rank].item<float>();

        // Pointers for expert j
        const uint8_t* gu_w = gu_packed_base + j * gu_expert_stride_bytes;
        const __half*  gu_s = gu_scales_base  + j * gu_scales_stride;
        const uint8_t* dn_w = dn_packed_base + j * dn_expert_stride_bytes;
        const __half*  dn_s = dn_scales_base  + j * dn_scales_stride;

        // Step 1: GEMV gate+up  (gu_buf = W_gu @ normed)
        launch_gemv(normed_ptr, gu_w, gu_s, gu_buf_ptr, (int)N_gu, (int)K_gu, stream);

        // Step 2: Fused SiLU(gate)*Up + FWHT + GEMV down
        //   Gate = gu_buf[0 : moe_inter], Up = gu_buf[moe_inter : 2*moe_inter]
        launch_gemv_silu_had(
            gu_buf_ptr,               // gate
            gu_buf_ptr + moe_inter,   // up
            dn_w, dn_s,
            dn_buf_ptr, (int)N_dn, (int)K_dn, stream);

        // Step 3: moe_out += w * dn_buf
        launch_scale_accumulate(moe_out_ptr, dn_buf_ptr, w, (int)N_dn, stream);
    }

    return moe_out;
}
// ============================================================
// moe_gemv_top_k_v2: same as moe_gemv_top_k but with NO CPU-GPU sync.
//   top_k_ids    [k] int64  — ON GPU
//   top_k_weights [k] float32 — ON GPU
// All kernel launches are async; no .cpu()/.item() calls.
// ============================================================
torch::Tensor moe_gemv_top_k_v2(
    torch::Tensor normed,         // [D=hidden] FP16
    torch::Tensor exp_gu_packed,  // [E, N_gu, K_gu/2] uint8
    torch::Tensor exp_gu_scales,  // [E, N_gu, K_gu/16] FP16  (interleaved sz)
    torch::Tensor exp_dn_packed,  // [E, N_dn, K_dn/2] uint8
    torch::Tensor exp_dn_scales,  // [E, N_dn, K_dn/16] FP16
    torch::Tensor top_k_ids,      // [k] int64  — on GPU
    torch::Tensor top_k_weights,  // [k] float32 — on GPU
    int64_t moe_inter,
    int64_t hidden)
{
    TORCH_CHECK(normed.dim() == 1 && normed.size(0) == hidden, "normed must be [hidden]");
    TORCH_CHECK(exp_gu_packed.is_contiguous(), "exp_gu_packed must be contiguous");
    TORCH_CHECK(exp_dn_packed.is_contiguous(), "exp_dn_packed must be contiguous");
    TORCH_CHECK(top_k_ids.is_cuda(), "top_k_ids must be on GPU");
    TORCH_CHECK(top_k_weights.is_cuda(), "top_k_weights must be on GPU");
    TORCH_CHECK(top_k_weights.scalar_type() == torch::kFloat32, "top_k_weights must be float32");

    auto stream = at::hip::getCurrentHIPStream();

    const int64_t N_gu = 2 * moe_inter;
    const int64_t K_gu = hidden;
    const int64_t N_dn = hidden;
    const int64_t K_dn = moe_inter;

    // Strides in elements
    const int64_t gu_packed_stride = N_gu * (K_gu / 2);    // uint8 per expert
    const int64_t dn_packed_stride = N_dn * (K_dn / 2);
    const int64_t gu_scales_stride = N_gu * (K_gu / 16);   // FP16 per expert (scales tensor is [E,N,K/16])
    const int64_t dn_scales_stride = N_dn * (K_dn / 16);

    auto moe_out = torch::zeros({hidden},
        torch::dtype(torch::kHalf).device(normed.device()));
    auto gu_buf  = torch::empty({N_gu},
        torch::dtype(torch::kHalf).device(normed.device()));
    auto dn_buf  = torch::empty({N_dn},
        torch::dtype(torch::kHalf).device(normed.device()));

    const __half* normed_ptr = reinterpret_cast<const __half*>(normed.data_ptr());
    __half* moe_out_ptr = reinterpret_cast<__half*>(moe_out.data_ptr());
    __half* gu_buf_ptr  = reinterpret_cast<__half*>(gu_buf.data_ptr());
    __half* dn_buf_ptr  = reinterpret_cast<__half*>(dn_buf.data_ptr());

    const uint8_t* gu_packed_base = exp_gu_packed.data_ptr<uint8_t>();
    const __half*  gu_scales_base = reinterpret_cast<const __half*>(exp_gu_scales.data_ptr());
    const uint8_t* dn_packed_base = exp_dn_packed.data_ptr<uint8_t>();
    const __half*  dn_scales_base = reinterpret_cast<const __half*>(exp_dn_scales.data_ptr());

    const int64_t* ids_ptr = top_k_ids.data_ptr<int64_t>();
    const float*   wts_ptr = top_k_weights.data_ptr<float>();

    int64_t k = top_k_ids.size(0);

    for (int64_t rank = 0; rank < k; rank++) {
        // Step 1: GEMV gate+up — reads j from GPU, no sync
        hipLaunchKernelGGL((gemv_warp_indirect<4>),
            dim3((N_gu + 3) / 4), dim3(32), 0, stream,
            normed_ptr,
            gu_packed_base, gu_scales_base,
            gu_buf_ptr,
            ids_ptr + rank,
            gu_packed_stride, gu_scales_stride,
            (int)N_gu, (int)K_gu);

        // Step 2: Fused SiLU(gate)*Up + FWHT + GEMV down — reads j from GPU, no sync
        hipLaunchKernelGGL((gemv_warp_silu_had_indirect<4>),
            dim3((N_dn + 3) / 4), dim3(32), 0, stream,
            gu_buf_ptr,               // gate
            gu_buf_ptr + moe_inter,   // up
            dn_packed_base, dn_scales_base,
            dn_buf_ptr,
            ids_ptr + rank,
            dn_packed_stride, dn_scales_stride,
            (int)N_dn, (int)K_dn);

        // Step 3: moe_out += w * dn_buf — reads w from GPU float pointer, no sync
        launch_scale_accumulate_ptr(moe_out_ptr, dn_buf_ptr, wts_ptr + rank, (int)N_dn, stream);
    }

    return moe_out;
}
// ============================================================
// FP16 Flash Decode — for MoE model with FP16 KV cache
// ============================================================
template <int BLOCK_S>
__global__ void flash_decode_partial_fp16(
    const __half* __restrict__ Q,
    const __half* __restrict__ K_cache,   // [Hk, max_seq, D] FP16
    const __half* __restrict__ V_cache,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_out,
    int H, int Hk, int max_seq, int seq_len, int D, int num_splits
) {
    const int kh = blockIdx.x;
    const int split_id = blockIdx.y;
    const int warp_id = threadIdx.y;
    const int lane = threadIdx.x;
    const int gqa_ratio = H / Hk;
    if (warp_id >= gqa_ratio) return;
    const int qh = kh * gqa_ratio + warp_id;
    const float scale = rsqrtf((float)D);

    const __half* q_row = Q + qh * D;
    float q_reg[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        q_reg[i] = __half2float(q_row[lane + i * 32]);

    const int s_start = split_id * BLOCK_S;
    const int s_end = min(s_start + BLOCK_S, seq_len);

    const __half* k_base = K_cache + (long)kh * max_seq * D;
    const __half* v_base = V_cache + (long)kh * max_seq * D;

    float local_max = -1e30f, local_sum = 0.0f;
    float out_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = s_start; s < s_end; s++) {
        const __half* k_pos = k_base + (long)s * D;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            dot += q_reg[i] * __half2float(k_pos[lane + i * 32]);
        dot += __shfl_xor(dot, 16); dot += __shfl_xor(dot, 8);
        dot += __shfl_xor(dot, 4);  dot += __shfl_xor(dot, 2); dot += __shfl_xor(dot, 1);
        dot *= scale;

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float exp_score = expf(dot - new_max);
        local_sum = local_sum * correction + exp_score;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            out_reg[i] = out_reg[i] * correction +
                         exp_score * __half2float(v_base[(long)s * D + lane + i * 32]);
        local_max = new_max;
    }

    const long out_offset = (long)split_id * H * D + (long)qh * D;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        partial_out[out_offset + lane + i * 32] = out_reg[i];
    if (lane == 0) {
        partial_max[split_id * H + qh] = local_max;
        partial_sum[split_id * H + qh] = local_sum;
    }
}

// Self-reducing flash decode: partial + reduce in one dispatch.
// Grid: (num_kv_heads, n_splits), Block: (32, gqa_ratio).
// Last block per KV head does the reduction + FWHT inline.
// Eliminates the separate flash_decode_reduce dispatch.
template<int BLOCK_S>
__global__ void flash_decode_selfreduce_fp16(
    const __half* __restrict__ Q,
    const __half* __restrict__ K_cache,   // [Hk, max_seq, D] FP16
    const __half* __restrict__ V_cache,
    __half* __restrict__ O,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_out,
    int* __restrict__ reduce_ctr,         // [Hk] atomics
    int H, int Hk, int max_seq, int seq_len, int D, int num_splits
) {
    const int kh = blockIdx.x;
    const int split_id = blockIdx.y;
    const int warp_id = threadIdx.y;
    const int lane = threadIdx.x;
    const int gqa_ratio = H / Hk;
    if (warp_id >= gqa_ratio) return;
    const int qh = kh * gqa_ratio + warp_id;
    const float scale = rsqrtf((float)D);

    const __half* q_row = Q + qh * D;
    float q_reg[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        q_reg[i] = __half2float(q_row[lane + i * 32]);

    const int s_start = split_id * BLOCK_S;
    const int s_end = min(s_start + BLOCK_S, seq_len);

    const __half* k_base = K_cache + (long)kh * max_seq * D;
    const __half* v_base = V_cache + (long)kh * max_seq * D;

    float local_max = -1e30f, local_sum = 0.0f;
    float out_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = s_start; s < s_end; s++) {
        const __half* k_pos = k_base + (long)s * D;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            dot += q_reg[i] * __half2float(k_pos[lane + i * 32]);
        dot += __shfl_xor(dot, 16); dot += __shfl_xor(dot, 8);
        dot += __shfl_xor(dot, 4);  dot += __shfl_xor(dot, 2); dot += __shfl_xor(dot, 1);
        dot *= scale;

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float exp_score = expf(dot - new_max);
        local_sum = local_sum * correction + exp_score;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            out_reg[i] = out_reg[i] * correction +
                         exp_score * __half2float(v_base[(long)s * D + lane + i * 32]);
        local_max = new_max;
    }

    // Write partial results
    const long p_out_offset = (long)split_id * H * D + (long)qh * D;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        partial_out[p_out_offset + lane + i * 32] = out_reg[i];
    if (lane == 0) {
        partial_max[split_id * H + qh] = local_max;
        partial_sum[split_id * H + qh] = local_sum;
    }

    // Ensure partial writes are globally visible
    __threadfence();
    __syncthreads();

    // One thread per block increments counter
    __shared__ int s_is_last;
    if (warp_id == 0 && lane == 0) {
        int old = atomicAdd(&reduce_ctr[kh], 1);
        s_is_last = (old == num_splits - 1) ? 1 : 0;
    }
    __syncthreads();

    if (!s_is_last) return;

    // === Last block: reduce all splits for this KV head's GQA group ===
    // Each warp handles one query head (same as flash_decode_reduce)
    float global_max = -1e30f;
    for (int sp = 0; sp < num_splits; sp++) {
        float m = partial_max[sp * H + qh];
        global_max = fmaxf(global_max, m);
    }

    float global_sum = 0.0f;
    float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int sp = 0; sp < num_splits; sp++) {
        float correction = expf(partial_max[sp * H + qh] - global_max);
        global_sum += partial_sum[sp * H + qh] * correction;
        const long off = (long)sp * H * D + (long)qh * D;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            final_out[i] += partial_out[off + lane + i * 32] * correction;
    }

    float inv = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    __half* o_row = O + qh * D;
    // FWHT on output (same as flash_decode_reduce)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float val = final_out[i] * inv;
        #pragma unroll
        for (int step = 1; step < 32; step *= 2) {
            float other = __shfl_xor(val, step);
            if (lane & step) val = other - val;
            else val = val + other;
        }
        o_row[lane + i * 32] = __float2half(val * 0.17677669529663689f);
    }

    // Reset counter for next layer
    if (warp_id == 0 && lane == 0) {
        reduce_ctr[kh] = 0;
        __threadfence();
    }
}

// Graph-compatible self-reducing flash decode: reads seq_len from g_graph_seq_len.
// Grid: (num_kv_heads, max_n_splits) — fixed at capture time.
// Inactive splits (beyond actual n_splits) early-exit without touching atomics.
template<int BLOCK_S>
__global__ void flash_decode_selfreduce_fp16_g(
    const __half* __restrict__ Q,
    const __half* __restrict__ K_cache,
    const __half* __restrict__ V_cache,
    __half* __restrict__ O,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_out,
    int* __restrict__ reduce_ctr,
    int H, int Hk, int max_seq, int D, int max_n_splits
) {
    int seq_len = g_graph_seq_len;
    const int kh = blockIdx.x;
    const int split_id = blockIdx.y;
    const int warp_id = threadIdx.y;
    const int lane = threadIdx.x;
    const int gqa_ratio = H / Hk;
    if (warp_id >= gqa_ratio) return;
    const int qh = kh * gqa_ratio + warp_id;

    int n_splits_actual = (seq_len + BLOCK_S - 1) / BLOCK_S;
    if (n_splits_actual < 1) n_splits_actual = 1;

    // Inactive splits: write dummy partials and return (don't touch atomics)
    if (split_id >= n_splits_actual) {
        if (lane == 0) {
            partial_max[split_id * H + qh] = -1e30f;
            partial_sum[split_id * H + qh] = 0.0f;
        }
        for (int i = 0; i < 4; i++)
            partial_out[(long)split_id * H * D + (long)qh * D + lane + i * 32] = 0.0f;
        return;
    }

    const float scale = rsqrtf((float)D);
    const __half* q_row = Q + qh * D;
    float q_reg[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        q_reg[i] = __half2float(q_row[lane + i * 32]);

    const int s_start = split_id * BLOCK_S;
    const int s_end = min(s_start + BLOCK_S, seq_len);

    const __half* k_base = K_cache + (long)kh * max_seq * D;
    const __half* v_base = V_cache + (long)kh * max_seq * D;

    float local_max = -1e30f, local_sum = 0.0f;
    float out_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = s_start; s < s_end; s++) {
        const __half* k_pos = k_base + (long)s * D;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            dot += q_reg[i] * __half2float(k_pos[lane + i * 32]);
        dot += __shfl_xor(dot, 16); dot += __shfl_xor(dot, 8);
        dot += __shfl_xor(dot, 4);  dot += __shfl_xor(dot, 2); dot += __shfl_xor(dot, 1);
        dot *= scale;

        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        float exp_score = expf(dot - new_max);
        local_sum = local_sum * correction + exp_score;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            out_reg[i] = out_reg[i] * correction +
                         exp_score * __half2float(v_base[(long)s * D + lane + i * 32]);
        local_max = new_max;
    }

    // Write partial results
    const long p_out_offset = (long)split_id * H * D + (long)qh * D;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        partial_out[p_out_offset + lane + i * 32] = out_reg[i];
    if (lane == 0) {
        partial_max[split_id * H + qh] = local_max;
        partial_sum[split_id * H + qh] = local_sum;
    }

    __threadfence();
    __syncthreads();

    __shared__ int s_is_last;
    if (warp_id == 0 && lane == 0) {
        int old = atomicAdd(&reduce_ctr[kh], 1);
        s_is_last = (old == n_splits_actual - 1) ? 1 : 0;
    }
    __syncthreads();

    if (!s_is_last) return;

    // === Last block: reduce all splits ===
    float global_max = -1e30f;
    for (int sp = 0; sp < n_splits_actual; sp++) {
        float m = partial_max[sp * H + qh];
        global_max = fmaxf(global_max, m);
    }

    float global_sum = 0.0f;
    float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int sp = 0; sp < n_splits_actual; sp++) {
        float correction = expf(partial_max[sp * H + qh] - global_max);
        global_sum += partial_sum[sp * H + qh] * correction;
        const long off = (long)sp * H * D + (long)qh * D;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            final_out[i] += partial_out[off + lane + i * 32] * correction;
    }

    float inv = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    __half* o_row = O + qh * D;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float val = final_out[i] * inv;
        #pragma unroll
        for (int step = 1; step < 32; step *= 2) {
            float other = __shfl_xor(val, step);
            if (lane & step) val = other - val;
            else val = val + other;
        }
        o_row[lane + i * 32] = __float2half(val * 0.17677669529663689f);
    }

    if (warp_id == 0 && lane == 0) {
        reduce_ctr[kh] = 0;
        __threadfence();
    }
}

// Graph-compatible self-reducing flash decode launcher
static inline void launch_flash_decode_fp16_sr_g(
    const __half* Q, const __half* K_cache, const __half* V_cache,
    __half* O,
    float* p_max, float* p_sum, float* p_out,
    int* reduce_ctr,
    int num_heads, int num_kv_heads, int max_seq,
    int max_n_splits, int block_s, int head_dim,
    hipStream_t stream)
{
    int gqa_ratio = num_heads / num_kv_heads;
    #define LAUNCH_SR_G_FP16(BS) \
        hipLaunchKernelGGL((flash_decode_selfreduce_fp16_g<BS>), \
            dim3(num_kv_heads, max_n_splits), dim3(32, gqa_ratio), 0, stream, \
            Q, K_cache, V_cache, O, p_max, p_sum, p_out, reduce_ctr, \
            num_heads, num_kv_heads, max_seq, head_dim, max_n_splits)
    switch (block_s) {
        case 32: LAUNCH_SR_G_FP16(32); break;
        case 64: LAUNCH_SR_G_FP16(64); break;
    }
    #undef LAUNCH_SR_G_FP16
}

// GQA-aware fused attention: groups Q heads sharing a KV head into the same block
// for L1 cache sharing. Grid: (num_kv_heads), Block: (32, gqa_ratio).
// Each warp processes one Q head; all warps in a block share K/V reads via L1.
__global__ void flash_decode_fused_fp16(
    const __half* __restrict__ Q,
    const __half* __restrict__ K_cache,
    const __half* __restrict__ V_cache,
    __half* __restrict__ O,
    int H, int Hk, int max_seq, int seq_len, int D
) {
    const int kh = blockIdx.x;                  // KV head index
    const int gqa_idx = threadIdx.y;            // which Q head within GQA group
    const int gqa_ratio = H / Hk;
    const int qh = kh * gqa_ratio + gqa_idx;   // global Q head index
    const int lane = threadIdx.x;
    const float scale = rsqrtf((float)D);

    const __half* q_row = Q + qh * D;
    float q_reg[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        q_reg[i] = __half2float(q_row[lane + i * 32]);

    const __half* k_base = K_cache + (long)kh * max_seq * D;
    const __half* v_base = V_cache + (long)kh * max_seq * D;

    float max_score = -1e30f, sum_exp = 0.0f;
    float out_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = 0; s < seq_len; s++) {
        const __half* k_pos = k_base + (long)s * D;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            dot += q_reg[i] * __half2float(k_pos[lane + i * 32]);
        dot += __shfl_xor(dot, 16); dot += __shfl_xor(dot, 8);
        dot += __shfl_xor(dot, 4);  dot += __shfl_xor(dot, 2); dot += __shfl_xor(dot, 1);
        dot *= scale;

        float new_max = fmaxf(max_score, dot);
        float correction = expf(max_score - new_max);
        float exp_score = expf(dot - new_max);
        sum_exp = sum_exp * correction + exp_score;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            out_reg[i] = out_reg[i] * correction +
                         exp_score * __half2float(v_base[(long)s * D + lane + i * 32]);
        max_score = new_max;
    }

    float inv = 1.0f / sum_exp;
    __half* o_row = O + qh * D;
    // Apply FWHT on output — fused
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float val = out_reg[i] * inv;
        #pragma unroll
        for (int step = 1; step < 32; step *= 2) {
            float other = __shfl_xor(val, step);
            if (lane & step) val = other - val;
            else val = val + other;
        }
        o_row[lane + i * 32] = __float2half(val * 0.17677669529663689f);
    }
}

static inline void launch_flash_decode_fp16(
    const __half* Q, const __half* K_cache, const __half* V_cache,
    __half* O,
    float* p_max, float* p_sum, float* p_out,
    int num_heads, int num_kv_heads, int max_seq, int seq_len, int head_dim,
    hipStream_t stream)
{
    int gqa_ratio = num_heads / num_kv_heads;
    if (seq_len <= 32) {
        hipLaunchKernelGGL(flash_decode_fused_fp16,
            dim3(num_kv_heads), dim3(32, gqa_ratio), 0, stream,
            Q, K_cache, V_cache, O,
            num_heads, num_kv_heads, max_seq, seq_len, head_dim);
        return;
    }
    int n_splits = (seq_len + 31) / 32;
    hipLaunchKernelGGL((flash_decode_partial_fp16<32>),
        dim3(num_kv_heads, n_splits), dim3(32, gqa_ratio), 0, stream,
        Q, K_cache, V_cache, p_max, p_sum, p_out,
        num_heads, num_kv_heads, max_seq, seq_len, head_dim, n_splits);
    hipLaunchKernelGGL(flash_decode_reduce,
        dim3(num_heads), dim3(32), 0, stream,
        p_max, p_sum, p_out, O, num_heads, head_dim, n_splits);
}

// Self-reducing flash decode launcher: 1 dispatch instead of 2
static inline void launch_flash_decode_fp16_sr(
    const __half* Q, const __half* K_cache, const __half* V_cache,
    __half* O,
    float* p_max, float* p_sum, float* p_out,
    int* reduce_ctr,
    int num_heads, int num_kv_heads, int max_seq, int seq_len, int head_dim,
    hipStream_t stream)
{
    int gqa_ratio = num_heads / num_kv_heads;
    if (seq_len <= 32) {
        hipLaunchKernelGGL(flash_decode_fused_fp16,
            dim3(num_kv_heads), dim3(32, gqa_ratio), 0, stream,
            Q, K_cache, V_cache, O,
            num_heads, num_kv_heads, max_seq, seq_len, head_dim);
        return;
    }
    // Always use block_s=32: more blocks but each does less work per block,
    // giving smoother context scaling (no discontinuity at the 256 boundary)
    int n_splits = (seq_len + 31) / 32;
    hipLaunchKernelGGL((flash_decode_selfreduce_fp16<32>),
        dim3(num_kv_heads, n_splits), dim3(32, gqa_ratio), 0, stream,
        Q, K_cache, V_cache, O, p_max, p_sum, p_out, reduce_ctr,
        num_heads, num_kv_heads, max_seq, seq_len, head_dim, n_splits);
}
// ============================================================
// FP16 KV cache write — same as head_rmsnorm_rope_kv but stores FP16.
// ============================================================
__global__ void head_rmsnorm_rope_kv_fp16(
    const __half* Q, const __half* K, const __half* V,
    const __half* QW, const __half* KW,
    __half* Q_out, __half* K_out,
    __half* K_cache, __half* V_cache,
    const __half* COS, const __half* SIN,
    int H, int Hk, int D, float eps, int max_seq, int pos
) {
    int pid = blockIdx.x;
    int tid = threadIdx.x;
    int half_D = D / 2;

    if (pid < H) {
        const __half* src = Q + pid * D;
        __half* dst = Q_out + pid * D;
        float sq = 0.0f;
        for (int i = tid; i < D; i += 32) { float v = __half2float(src[i]); sq += v * v; }
        sq += __shfl_xor(sq, 16); sq += __shfl_xor(sq, 8);
        sq += __shfl_xor(sq, 4);  sq += __shfl_xor(sq, 2); sq += __shfl_xor(sq, 1);
        float rrms = rsqrtf(sq / (float)D + eps);
        for (int i = tid; i < half_D; i += 32) {
            float ne = __half2float(src[i]) * rrms * __half2float(QW[i]);
            float no = __half2float(src[i + half_D]) * rrms * __half2float(QW[i + half_D]);
            float c = __half2float(COS[i]), s = __half2float(SIN[i]);
            dst[i]        = __float2half(ne * c - no * s);
            dst[i + half_D] = __float2half(no * c + ne * s);
        }
    }
    int kid = pid - H;
    if (kid >= 0 && kid < Hk) {
        const __half* src = K + kid * D;
        __half* dst = K_out + kid * D;
        int cache_off = kid * max_seq * D + pos * D;
        float sq = 0.0f;
        for (int i = tid; i < D; i += 32) { float v = __half2float(src[i]); sq += v * v; }
        sq += __shfl_xor(sq, 16); sq += __shfl_xor(sq, 8);
        sq += __shfl_xor(sq, 4);  sq += __shfl_xor(sq, 2); sq += __shfl_xor(sq, 1);
        float rrms = rsqrtf(sq / (float)D + eps);
        for (int i = tid; i < half_D; i += 32) {
            float ne = __half2float(src[i]) * rrms * __half2float(KW[i]);
            float no = __half2float(src[i + half_D]) * rrms * __half2float(KW[i + half_D]);
            float c = __half2float(COS[i]), s = __half2float(SIN[i]);
            float k_lo = ne * c - no * s;
            float k_hi = no * c + ne * s;
            dst[i]          = __float2half(k_lo);
            dst[i + half_D] = __float2half(k_hi);
            K_cache[cache_off + i]          = __float2half(k_lo);
            K_cache[cache_off + i + half_D] = __float2half(k_hi);
        }
        const __half* v_src = V + kid * D;
        for (int i = tid; i < D; i += 32)
            V_cache[cache_off + i] = v_src[i];
    }
}
template<int BLOCK_N>
__global__ __launch_bounds__(32)
void fused_resnorm_router_gemv(
    const __half* __restrict__ A,         // obuf (O projection output)
    const __half* __restrict__ B,         // h (residual stream)
    const __half* __restrict__ NormW,     // post-attention norm weights
    __half* __restrict__ Y_rot,           // FWHT output for expert GEMVs
    __half* __restrict__ ResOut,          // residual A+B for next-layer reduce
    const __half* __restrict__ gate_W,    // [E, D] router weights (FP16)
    float* __restrict__ logits,           // [E] intermediate logits
    int64_t* __restrict__ topk_ids,       // [k] output expert indices
    float* __restrict__ topk_probs,       // [k] output expert probs
    int* __restrict__ block_counter,      // [1] atomic counter (self-resetting)
    int D, int E, int k, float eps
) {
    const int bid = blockIdx.x;
    const int lane = threadIdx.x;
    const int n0 = bid * BLOCK_N;
    const int nit = D / 32;  // 64 for D=2048

    // ======= Single pass: ss accumulation + GEMV(r*NormW, W) =======
    // Factor out rrms: GEMV(normed, W) = rrms * GEMV(r*NormW, W)
    float ss = 0.0f;
    float acc[BLOCK_N];
    #pragma unroll
    for (int ni = 0; ni < BLOCK_N; ni++) acc[ni] = 0.0f;

    for (int it = lane; it < nit; it += 32) {
        // Load 32 elements of A, B, NormW, compute r*NormW (without rrms)
        float rn_v[32];
        int base = it * 32;
        #pragma unroll
        for (int i = 0; i < 32; i += 2) {
            __half2 ah = *reinterpret_cast<const __half2*>(&A[base + i]);
            __half2 bh = *reinterpret_cast<const __half2*>(&B[base + i]);
            __half2 nh = *reinterpret_cast<const __half2*>(&NormW[base + i]);
            float r0 = __half2float(ah.x) + __half2float(bh.x);
            float r1 = __half2float(ah.y) + __half2float(bh.y);
            ss += r0 * r0 + r1 * r1;
            rn_v[i]   = r0 * __half2float(nh.x);
            rn_v[i+1] = r1 * __half2float(nh.y);
        }

        // GEMV inner loop: batch-32 dot product (high ILP, matches original)
        #pragma unroll
        for (int ni = 0; ni < BLOCK_N; ni++) {
            int n = n0 + ni;
            if (n >= E) continue;
            const __half* w_row = gate_W + (int64_t)n * D + base;
            float dot = 0.0f;
            #pragma unroll
            for (int b = 0; b < 32; b++)
                dot += rn_v[b] * __half2float(w_row[b]);
            acc[ni] += dot;
        }
    }

    // Warp reduce ss → rrms
    ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
    ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
    float rrms = rsqrtf(ss / (float)D + eps);

    // Apply rrms, warp reduce GEMV accumulators, write logits
    #pragma unroll
    for (int ni = 0; ni < BLOCK_N; ni++) {
        acc[ni] *= rrms;
        acc[ni] += __shfl_xor(acc[ni], 16); acc[ni] += __shfl_xor(acc[ni], 8);
        acc[ni] += __shfl_xor(acc[ni], 4);  acc[ni] += __shfl_xor(acc[ni], 2);
        acc[ni] += __shfl_xor(acc[ni], 1);
        if (lane == 0 && n0 + ni < E) logits[n0 + ni] = acc[ni];
    }

    // ======= All blocks: distribute ResOut + FWHT → Y_rot evenly =======
    // Each block writes D/num_blocks elements. A, B, NormW hot in L2.
    {
        int nblocks = gridDim.x;
        int groups_per_block = (nit + nblocks - 1) / nblocks;
        int g_start = bid * groups_per_block;
        int g_end = g_start + groups_per_block;
        if (g_end > nit) g_end = nit;
        for (int it = g_start; it < g_end; it++) {
            int base = it * 32;
            float r = __half2float(A[base + lane]) + __half2float(B[base + lane]);
            ResOut[base + lane] = __float2half(r);
            float normed = r * rrms * __half2float(NormW[base + lane]);
            float fwht = normed;
            #pragma unroll
            for (int step = 1; step < 32; step *= 2) {
                float other = __shfl_xor(fwht, step);
                if (lane & step) fwht = other - fwht;
                else fwht = fwht + other;
            }
            Y_rot[base + lane] = __float2half(fwht * 0.17677669529663689f);
        }
    }

    // ======= Last-block: softmax + top-K (self-resetting counter) =======
    // Same pattern as flash_decode_selfreduce: last block does reduce, resets counter.
    __threadfence();
    __shared__ int is_last;
    if (lane == 0) {
        int done = atomicAdd(block_counter, 1) + 1;
        is_last = (done == (int)gridDim.x) ? 1 : 0;
    }
    is_last = __shfl(is_last, 0);
    if (!is_last) return;

    // 32 threads process E=128 logits (4 per thread)
    float vals[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        vals[i] = (lane + i * 32 < E) ? logits[lane + i * 32] : -1e30f;

    // Global max
    float lmax = fmaxf(fmaxf(vals[0], vals[1]), fmaxf(vals[2], vals[3]));
    lmax = fmaxf(lmax, __shfl_xor(lmax, 16)); lmax = fmaxf(lmax, __shfl_xor(lmax, 8));
    lmax = fmaxf(lmax, __shfl_xor(lmax, 4));  lmax = fmaxf(lmax, __shfl_xor(lmax, 2));
    lmax = fmaxf(lmax, __shfl_xor(lmax, 1));

    // Softmax exp + sum
    float evals[4];
    float lsum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        evals[i] = (lane + i * 32 < E) ? expf(vals[i] - lmax) : 0.0f;
        lsum += evals[i];
    }
    lsum += __shfl_xor(lsum, 16); lsum += __shfl_xor(lsum, 8);
    lsum += __shfl_xor(lsum, 4);  lsum += __shfl_xor(lsum, 2);
    lsum += __shfl_xor(lsum, 1);
    float inv_sum = (lsum > 0.0f) ? (1.0f / lsum) : 1.0f;

    // Store softmax probs to shared memory for serial top-K
    __shared__ float sprobs[256];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        sprobs[lane + i * 32] = evals[i] * inv_sum;

    // Serial top-K by thread 0
    if (lane == 0) {
        float sum_k = 0.0f;
        for (int r = 0; r < k; r++) {
            int best = 0;
            float bv = sprobs[0];
            for (int i = 1; i < E; i++) {
                if (sprobs[i] > bv) { bv = sprobs[i]; best = i; }
            }
            topk_ids[r] = best;
            topk_probs[r] = bv;
            sum_k += bv;
            sprobs[best] = -1.0f;
        }
        float inv_k = (sum_k > 0.0f) ? (1.0f / sum_k) : 1.0f;
        for (int r = 0; r < k; r++) topk_probs[r] *= inv_k;

        // Reset counter for next layer / next graph replay
        block_counter[0] = 0;
        __threadfence();
    }
}

static inline __attribute__((always_inline)) void launch_fused_resnorm_router(
    const __half* A, const __half* B, const __half* NormW,
    __half* Y_rot, __half* ResOut,
    const __half* gate_W, float* logits,
    int64_t* topk_ids, float* topk_probs, int* block_counter,
    int D, int E, int k, float eps, hipStream_t st) {
    hipLaunchKernelGGL((fused_resnorm_router_gemv<4>),
        dim3((E + 3) / 4), dim3(32), 0, st,
        A, B, NormW, Y_rot, ResOut, gate_W, logits,
        topk_ids, topk_probs, block_counter, D, E, k, eps);
}
// ============================================================
// Softmax + Top-K with int64 output (for GEMV batched kernels that use int64_t tids)
// Same as softmax_topk_gpu but outputs int64_t* topk_ids.
// ============================================================
__global__ void softmax_topk_i64(
    const float* __restrict__ logits,
    int64_t* __restrict__ topk_ids,
    float* __restrict__ topk_probs,
    int E, int k
) {
    const int tid = threadIdx.x;
    __shared__ float svals[256];
    __shared__ float scratch_f[8];
    __shared__ int   scratch_i[8];

    float val = (tid < E) ? logits[tid] : -1e30f;
    float lmax = val;
    lmax = fmaxf(lmax, __shfl_xor(lmax, 16)); lmax = fmaxf(lmax, __shfl_xor(lmax, 8));
    lmax = fmaxf(lmax, __shfl_xor(lmax, 4));  lmax = fmaxf(lmax, __shfl_xor(lmax, 2));
    lmax = fmaxf(lmax, __shfl_xor(lmax, 1));
    int nwarps = blockDim.x / 32;
    if (tid % 32 == 0) scratch_f[tid / 32] = lmax;
    __syncthreads();
    if (tid < nwarps) {
        float v = scratch_f[tid];
        v = fmaxf(v, __shfl_xor(v, 4)); v = fmaxf(v, __shfl_xor(v, 2)); v = fmaxf(v, __shfl_xor(v, 1));
        if (tid == 0) scratch_f[0] = v;
    }
    __syncthreads();
    float gmax = scratch_f[0];

    float e = (tid < E) ? expf(val - gmax) : 0.0f;
    svals[tid] = (tid < E) ? e : 0.0f;
    float lsum = e;
    lsum += __shfl_xor(lsum, 16); lsum += __shfl_xor(lsum, 8);
    lsum += __shfl_xor(lsum, 4);  lsum += __shfl_xor(lsum, 2); lsum += __shfl_xor(lsum, 1);
    if (tid % 32 == 0) scratch_f[tid / 32] = lsum;
    __syncthreads();
    if (tid < nwarps) {
        float v = scratch_f[tid];
        v += __shfl_xor(v, 4); v += __shfl_xor(v, 2); v += __shfl_xor(v, 1);
        if (tid == 0) scratch_f[0] = v;
    }
    __syncthreads();
    float gsum = scratch_f[0];
    svals[tid] = (tid < E && gsum > 0.0f) ? (e / gsum) : 0.0f;
    __syncthreads();

    for (int r = 0; r < k; r++) {
        float my_val = (tid < E) ? svals[tid] : -1e30f;
        int my_idx = tid;
        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            float other_val = __shfl_xor(my_val, offset);
            int other_idx = __shfl_xor(my_idx, offset);
            if (other_val > my_val) { my_val = other_val; my_idx = other_idx; }
        }
        if (tid % 32 == 0) { scratch_f[tid / 32] = my_val; scratch_i[tid / 32] = my_idx; }
        __syncthreads();
        if (tid < nwarps) {
            float v = scratch_f[tid]; int idx = scratch_i[tid];
            for (int offset = nwarps / 2; offset >= 1; offset >>= 1) {
                float ov = __shfl_xor(v, offset);
                int oi = __shfl_xor(idx, offset);
                if (ov > v) { v = ov; idx = oi; }
            }
            if (tid == 0) { scratch_f[0] = v; scratch_i[0] = idx; }
        }
        __syncthreads();
        if (tid == 0) {
            topk_ids[r]   = scratch_i[0];
            topk_probs[r] = scratch_f[0];
        }
        if (tid == scratch_i[0]) svals[tid] = -1e30f;
        __syncthreads();
    }
    if (tid == 0) {
        float sum_k = 0.0f;
        for (int r = 0; r < k; r++) sum_k += topk_probs[r];
        float inv = (sum_k > 0.0f) ? (1.0f / sum_k) : 1.0f;
        for (int r = 0; r < k; r++) topk_probs[r] *= inv;
    }
}
// ============================================================
// Fused kernel 1: res_norm_norot_fwht + router_gemv + softmax_topk
// Single block of 1024 threads. Eliminates 2 kernel launches per layer.
// Phase 1: Residual + RMSNorm → norot (shared memory) + FWHT → rot (global)
// Phase 2: Router GEMV reading norot from shared → rlogits (shared)
// Phase 3: Softmax + Top-K on rlogits in shared → topk_ids, topk_probs (global)
// ============================================================
__global__ void fused_res_norm_router_topk(
    const __half* __restrict__ A,      // attention output (obuf)
    const __half* __restrict__ B,      // current hidden state (h)
    const __half* __restrict__ NormW,  // post_norm weight
    __half* __restrict__ Y_rot,        // FWHT output → expert dispatch input
    __half* __restrict__ ResOut,       // residual A+B → r
    const __half* __restrict__ GateW,  // [E, D] router weights FP16
    int64_t* __restrict__ topk_ids,    // [k] output
    float* __restrict__ topk_probs,    // [k] output
    int D, int E, int k, float eps
) {
    const int tid = threadIdx.x;
    const int nwarps = blockDim.x / 32;  // 32
    const int wid = tid / 32;
    const int lane = tid % 32;

    // Shared memory layout: norot[D] half (= D/2 floats), rlogits[E] float, ws[32] float
    extern __shared__ float smem[];
    __half* norot_smem = reinterpret_cast<__half*>(smem);
    float* rlogits = smem + D / 2;  // after D halves = D/2 floats
    float* ws = rlogits + E;  // [32]

    // === Phase 1: Residual + RMSNorm ===
    float ss = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float r = __half2float(A[i]) + __half2float(B[i]);
        ResOut[i] = __float2half(r);
        ss += r * r;
    }
    ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
    ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
    if (lane == 0) ws[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        ss = (lane < nwarps) ? ws[lane] : 0.0f;
        ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
        ss += __shfl_xor(ss, 4); ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
        if (lane == 0) ws[0] = rsqrtf(ss / (float)D + eps);
    }
    __syncthreads();
    float rrms = ws[0];

    // === Phase 2: Norm + FWHT → Y_rot, store norot → shared ===
    int num_blocks = D / 32;
    for (int blk = wid; blk < num_blocks; blk += nwarps) {
        int base = blk * 32;
        float normed = __half2float(ResOut[base + lane]) * rrms * __half2float(NormW[base + lane]);
        norot_smem[base + lane] = __float2half(normed);
        #pragma unroll
        for (int step = 1; step < 32; step *= 2) {
            float other = __shfl_xor(normed, step);
            if (lane & step) normed = other - normed;
            else normed = normed + other;
        }
        normed *= 0.17677669529663689f;
        Y_rot[base + lane] = __float2half(normed);
    }
    __syncthreads();

    // === Phase 3: Router GEMV — GateW[E,D] @ norot_smem[D] → rlogits[E] ===
    // 32 warps, each handles 4 rows (E=128, 128/32=4)
    {
        const int rows_per_warp = 4;
        const int n0 = wid * rows_per_warp;
        const int nit = D / 32;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int it = lane; it < nit; it += 32) {
            float xv[32];
            #pragma unroll
            for (int i = 0; i < 32; i += 2) {
                __half2 xh = *reinterpret_cast<const __half2*>(&norot_smem[it * 32 + i]);
                xv[i] = __half2float(xh.x); xv[i+1] = __half2float(xh.y);
            }
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int n = n0 + ni;
                if (n >= E) continue;
                const __half* w_row = GateW + (int64_t)n * D + it * 32;
                float dot = 0.0f;
                #pragma unroll
                for (int b = 0; b < 32; b++)
                    dot += xv[b] * __half2float(w_row[b]);
                acc[ni] += dot;
            }
        }
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            acc[ni] += __shfl_xor(acc[ni], 16); acc[ni] += __shfl_xor(acc[ni], 8);
            acc[ni] += __shfl_xor(acc[ni], 4);  acc[ni] += __shfl_xor(acc[ni], 2);
            acc[ni] += __shfl_xor(acc[ni], 1);
            if (lane == 0 && n0 + ni < E) rlogits[n0 + ni] = acc[ni];
        }
    }
    __syncthreads();

    // === Phase 4: Softmax + Top-K on rlogits[E] in shared memory ===
    {
        float val = (tid < E) ? rlogits[tid] : -1e30f;

        // Max reduce
        float lmax = val;
        lmax = fmaxf(lmax, __shfl_xor(lmax, 16)); lmax = fmaxf(lmax, __shfl_xor(lmax, 8));
        lmax = fmaxf(lmax, __shfl_xor(lmax, 4));  lmax = fmaxf(lmax, __shfl_xor(lmax, 2));
        lmax = fmaxf(lmax, __shfl_xor(lmax, 1));
        int nw_sm = (E + 31) / 32;
        if (lane == 0 && wid < nw_sm) ws[wid] = lmax;
        __syncthreads();
        if (tid == 0) {
            float m = ws[0];
            for (int i = 1; i < nw_sm; i++) m = fmaxf(m, ws[i]);
            ws[0] = m;
        }
        __syncthreads();
        float gmax = ws[0];

        // Exp + sum
        float e = (tid < E) ? expf(val - gmax) : 0.0f;
        float lsum = e;
        lsum += __shfl_xor(lsum, 16); lsum += __shfl_xor(lsum, 8);
        lsum += __shfl_xor(lsum, 4);  lsum += __shfl_xor(lsum, 2); lsum += __shfl_xor(lsum, 1);
        if (lane == 0 && wid < nw_sm) ws[wid] = lsum;
        __syncthreads();
        if (tid == 0) {
            float s = ws[0];
            for (int i = 1; i < nw_sm; i++) s += ws[i];
            ws[0] = s;
        }
        __syncthreads();
        float gsum = ws[0];

        // Store softmax probs to rlogits (reuse shared buffer)
        if (tid < E) rlogits[tid] = (gsum > 0.0f) ? (e / gsum) : 0.0f;
        __syncthreads();

        // Serial top-k by thread 0
        if (tid == 0) {
            float sum_k = 0.0f;
            for (int r = 0; r < k; r++) {
                int best = 0;
                float bv = rlogits[0];
                for (int i = 1; i < E; i++) {
                    if (rlogits[i] > bv) { bv = rlogits[i]; best = i; }
                }
                topk_ids[r] = best;
                topk_probs[r] = bv;
                sum_k += bv;
                rlogits[best] = -1.0f;
            }
            float inv_k = (sum_k > 0.0f) ? (1.0f / sum_k) : 1.0f;
            for (int r = 0; r < k; r++) topk_probs[r] *= inv_k;
        }
    }
}
// ============================================================
// Fused kernel 2: moe_weighted_sum + res_norm_fwht
// Single block of 1024 threads. Eliminates 1 kernel launch per layer.
// Phase 1: Weighted sum of expert outputs + residual add
// Phase 2: RMSNorm + FWHT
// ============================================================
__global__ void fused_moe_res_norm_fwht(
    const __half* __restrict__ dn_base,  // [k, D] expert down-proj outputs
    const float*  __restrict__ weights,  // [k] expert routing weights
    const __half* __restrict__ B,        // residual (previous hidden state)
    const __half* __restrict__ NormW,    // next layer's input norm weight
    __half* __restrict__ Y,              // FWHT output (next layer's GEMV input)
    __half* __restrict__ ResOut,         // new hidden state (h)
    int k_val, int D, float eps
) {
    const int tid = threadIdx.x;
    const int nwarps = blockDim.x / 32;
    const int wid = tid / 32;
    const int lane = tid % 32;

    // Phase 1: Weighted sum + residual + sum-of-squares
    float ss = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float moe = 0.0f;
        for (int r = 0; r < k_val; r++)
            moe += weights[r] * __half2float(dn_base[(int64_t)r * D + i]);
        float res = moe + __half2float(B[i]);
        ResOut[i] = __float2half(res);
        ss += res * res;
    }

    // Phase 2: RMSNorm reduction
    ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
    ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
    __shared__ float ws[32];
    if (lane == 0) ws[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        ss = (lane < nwarps) ? ws[lane] : 0.0f;
        ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
        ss += __shfl_xor(ss, 4); ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
        if (lane == 0) ws[0] = rsqrtf(ss / (float)D + eps);
    }
    __syncthreads();
    float rrms = ws[0];

    // Phase 3: Norm + FWHT
    int num_blocks = D / 32;
    for (int blk = wid; blk < num_blocks; blk += nwarps) {
        int base = blk * 32;
        float normed = __half2float(ResOut[base + lane]) * rrms * __half2float(NormW[base + lane]);
        #pragma unroll
        for (int step = 1; step < 32; step *= 2) {
            float other = __shfl_xor(normed, step);
            if (lane & step) normed = other - normed;
            else normed = normed + other;
        }
        normed *= 0.17677669529663689f;
        Y[base + lane] = __float2half(normed);
    }
}
// ============================================================
// MoE Decode Context — scratch buffers
// ============================================================
struct MoeDSContext {
    torch::Tensor h, residual, rot, norot;
    torch::Tensor qkv_buf, o_buf;
    torch::Tensor gu_buf, dn_buf, moe_out_buf;
    torch::Tensor batch_gu_buf;    // [k * N_gu] — all experts gate+up in parallel
    torch::Tensor batch_fwht_buf;  // [k * K_dn] — FWHT'd activation for down GEMV
    torch::Tensor batch_dn_buf;    // [k * N_dn] — all experts down in parallel
    torch::Tensor router_logits;   // [E] float32
    torch::Tensor topk_ids;        // [k] int64
    torch::Tensor topk_probs;      // [k] float32
    torch::Tensor fa_partial_max, fa_partial_sum, fa_partial_out;
    torch::Tensor splitk_buf;      // [SPLIT * max(Nqkv, No)] float32 for split-K GEMV
    torch::Tensor done_counters;   // [k] int32 for fused GU+SiLU atomic last-block detection
    torch::Tensor router_counter;  // [1] int32 for fused router GEMV+topk last-block detection
    torch::Tensor flash_reduce_ctr; // [MAX_KV_HEADS] int32 for self-reducing flash decode
    // W4A4 scratch buffers (quantized activations for udot8 GEMV)
    torch::Tensor w4a4_xq_gu;     // [K_gu/2] uint8 — shared quantized GU input
    torch::Tensor w4a4_xsc_gu;    // [K_gu/32*2] half — GU activation scales
    torch::Tensor w4a4_xq_dn;     // [k, K_dn/2] uint8 — per-expert quantized DN input
    torch::Tensor w4a4_xsc_dn;    // [k, K_dn/32*2] half — DN activation scales
    int fa_max_splits = 0;
    int batch_k = 0;               // cached k for buffer sizing
    bool init = false;
};
static MoeDSContext g_moe_ds;
// ============================================================
// MoE decode step: all layers in C++, returns logits [1,1,vocab].
// KV cache: FP16 [num_kv_heads, max_seq, head_dim].
// ============================================================
torch::Tensor int4_decode_step_moe_logits(
    torch::Tensor hidden_in,
    // Per-layer attention weights (INT4 g64)
    std::vector<torch::Tensor> qkv_w, std::vector<torch::Tensor> qkv_s,
    std::vector<int64_t> qkv_N,
    std::vector<torch::Tensor> o_w, std::vector<torch::Tensor> o_s,
    std::vector<int64_t> o_N,
    // Per-layer MoE weights
    std::vector<torch::Tensor> gate_w,    // [E, D] FP16 (router)
    std::vector<torch::Tensor> exp_gu_w,  // [E, N_gu, K_gu/2] uint8
    std::vector<torch::Tensor> exp_gu_s,  // [E, K_gu/64, N_gu, 2] FP16 interleaved
    std::vector<torch::Tensor> exp_dn_w,  // [E, N_dn, K_dn/2] uint8
    std::vector<torch::Tensor> exp_dn_s,  // [E, K_dn/64, N_dn, 2] FP16 interleaved
    std::vector<torch::Tensor> exp_gu_wsum, // [E, N_gu, K_gu/32] int16 (unused, kept for API compat)
    std::vector<torch::Tensor> exp_dn_wsum, // [E, N_dn, K_dn/32] int16 (unused, kept for API compat)
    // Norm weights
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w,  std::vector<torch::Tensor> k_norm_w,
    // RoPE
    torch::Tensor rope_cos, torch::Tensor rope_sin,
    int64_t pos,
    // Model config
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t moe_inter, int64_t num_active,
    float rms_eps,
    // FP16 KV caches: [num_kv_heads, max_seq, head_dim]
    std::vector<torch::Tensor> k_caches, std::vector<torch::Tensor> v_caches,
    // Final norm + LM head
    torch::Tensor final_norm_w,
    torch::Tensor lm_w, torch::Tensor lm_s, int64_t lm_N, int64_t lm_K
) {
    c10::InferenceMode guard;

    int num_layers = qkv_w.size();
    int D = hidden_size;
    int E = gate_w[0].size(0);
    int k = num_active;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int max_seq = k_caches[0].size(1);  // [Hk, max_seq, D]
    int seq_len = pos + 1;
    auto device = hidden_in.device();
    auto stream = at::hip::getCurrentHIPStream();

    int fa_max_splits = (max_seq + FLASH_DECODE_BLOCK_S - 1) / FLASH_DECODE_BLOCK_S;

    if (!g_moe_ds.init) {
        auto opts    = torch::dtype(torch::kHalf).device(device);
        auto opts_f  = torch::dtype(torch::kFloat32).device(device);
        auto opts_i  = torch::dtype(torch::kInt64).device(device);
        g_moe_ds.h          = torch::empty({D}, opts);
        g_moe_ds.residual   = torch::empty({D}, opts);
        g_moe_ds.rot        = torch::empty({D}, opts);
        g_moe_ds.norot      = torch::empty({D}, opts);
        g_moe_ds.qkv_buf    = torch::empty({q_dim + 2 * kv_dim}, opts);
        g_moe_ds.o_buf      = torch::empty({D}, opts);
        g_moe_ds.gu_buf     = torch::empty({2 * (int)moe_inter}, opts);
        g_moe_ds.dn_buf     = torch::empty({D}, opts);
        g_moe_ds.moe_out_buf= torch::empty({D}, opts);
        // Batched expert buffers — all k experts in parallel
        g_moe_ds.batch_gu_buf = torch::empty({k * 2 * (int)moe_inter}, opts);
        g_moe_ds.batch_fwht_buf = torch::empty({k * (int)moe_inter}, opts);
        g_moe_ds.batch_dn_buf = torch::empty({k * D}, opts);
        g_moe_ds.done_counters = torch::zeros({k}, torch::dtype(torch::kInt32).device(device));
        g_moe_ds.router_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(device));
        g_moe_ds.batch_k    = k;
        g_moe_ds.router_logits = torch::empty({E}, opts_f);
        g_moe_ds.topk_ids   = torch::empty({k}, opts_i);
        g_moe_ds.topk_probs = torch::empty({k}, opts_f);
        g_moe_ds.fa_partial_max = torch::empty({fa_max_splits, (long)num_heads}, opts_f);
        g_moe_ds.fa_partial_sum = torch::empty({fa_max_splits, (long)num_heads}, opts_f);
        g_moe_ds.fa_partial_out = torch::empty({fa_max_splits, (long)num_heads, (long)head_dim}, opts_f);
        g_moe_ds.fa_max_splits = fa_max_splits;
        // Split-K partial buffer: 2 splits × max(Nqkv, q_dim) float32
        long sk_max_n = std::max((long)(q_dim + 2 * kv_dim), (long)(q_dim));
        g_moe_ds.splitk_buf = torch::empty({2 * sk_max_n}, opts_f);
        // W4A4 scratch buffers for udot8 GEMV
        { auto ou8 = torch::dtype(torch::kUInt8).device(device);
          int Kg = D, Kd = (int)moe_inter;
          g_moe_ds.w4a4_xq_gu  = torch::empty({Kg / 2}, ou8);
          g_moe_ds.w4a4_xsc_gu = torch::empty({Kg / 32 * 2}, opts);
          g_moe_ds.w4a4_xq_dn  = torch::empty({k * Kd / 2}, ou8);
          g_moe_ds.w4a4_xsc_dn = torch::empty({k * Kd / 32 * 2}, opts); }
        g_moe_ds.init = true;
    }
    if (fa_max_splits > g_moe_ds.fa_max_splits) {
        auto opts_f = torch::dtype(torch::kFloat32).device(device);
        g_moe_ds.fa_partial_max = torch::empty({fa_max_splits, (long)num_heads}, opts_f);
        g_moe_ds.fa_partial_sum = torch::empty({fa_max_splits, (long)num_heads}, opts_f);
        g_moe_ds.fa_partial_out = torch::empty({fa_max_splits, (long)num_heads, (long)head_dim}, opts_f);
        g_moe_ds.fa_max_splits = fa_max_splits;
    }

    auto h         = reinterpret_cast<__half*>(g_moe_ds.h.data_ptr());
    auto r         = reinterpret_cast<__half*>(g_moe_ds.residual.data_ptr());
    auto rot       = reinterpret_cast<__half*>(g_moe_ds.rot.data_ptr());
    auto norot     = reinterpret_cast<__half*>(g_moe_ds.norot.data_ptr());
    auto qkv       = reinterpret_cast<__half*>(g_moe_ds.qkv_buf.data_ptr());
    auto obuf      = reinterpret_cast<__half*>(g_moe_ds.o_buf.data_ptr());
    auto gubuf     = reinterpret_cast<__half*>(g_moe_ds.gu_buf.data_ptr());
    auto dnbuf     = reinterpret_cast<__half*>(g_moe_ds.dn_buf.data_ptr());
    auto moebuf    = reinterpret_cast<__half*>(g_moe_ds.moe_out_buf.data_ptr());
    auto batch_gu   = reinterpret_cast<__half*>(g_moe_ds.batch_gu_buf.data_ptr());
    auto batch_fwht = reinterpret_cast<__half*>(g_moe_ds.batch_fwht_buf.data_ptr());
    auto batch_dn   = reinterpret_cast<__half*>(g_moe_ds.batch_dn_buf.data_ptr());
    auto rlogits    = reinterpret_cast<float*>(g_moe_ds.router_logits.data_ptr());
    auto tids       = g_moe_ds.topk_ids.data_ptr<int64_t>();
    auto tprobs    = g_moe_ds.topk_probs.data_ptr<float>();
    auto rctr      = g_moe_ds.router_counter.data_ptr<int>();
    auto p_max  = reinterpret_cast<float*>(g_moe_ds.fa_partial_max.data_ptr());
    auto p_sum  = reinterpret_cast<float*>(g_moe_ds.fa_partial_sum.data_ptr());
    auto p_out  = reinterpret_cast<float*>(g_moe_ds.fa_partial_out.data_ptr());
    auto sk_buf = reinterpret_cast<float*>(g_moe_ds.splitk_buf.data_ptr());

    // Pre-extract raw pointers (avoids PyTorch overhead in loop)
    std::vector<const uint8_t*> qkv_wp(num_layers), o_wp(num_layers);
    std::vector<const __half*>  qkv_sp(num_layers), o_sp(num_layers);
    std::vector<const __half*>  gate_wp(num_layers);
    std::vector<const uint8_t*> egu_wp(num_layers), edn_wp(num_layers);
    std::vector<const __half*>  egu_sp(num_layers), edn_sp(num_layers);
    std::vector<const __half*>  in_np(num_layers), post_np(num_layers);
    std::vector<const __half*>  q_np(num_layers), k_np(num_layers);
    std::vector<__half*>        kc_p(num_layers), vc_p(num_layers);
    for (int li = 0; li < num_layers; li++) {
        qkv_wp[li] = qkv_w[li].data_ptr<uint8_t>();
        qkv_sp[li] = reinterpret_cast<const __half*>(qkv_s[li].data_ptr());
        o_wp[li]   = o_w[li].data_ptr<uint8_t>();
        o_sp[li]   = reinterpret_cast<const __half*>(o_s[li].data_ptr());
        gate_wp[li]= reinterpret_cast<const __half*>(gate_w[li].data_ptr());
        egu_wp[li] = exp_gu_w[li].data_ptr<uint8_t>();
        egu_sp[li] = reinterpret_cast<const __half*>(exp_gu_s[li].data_ptr());
        edn_wp[li] = exp_dn_w[li].data_ptr<uint8_t>();
        edn_sp[li] = reinterpret_cast<const __half*>(exp_dn_s[li].data_ptr());
        in_np[li]  = reinterpret_cast<const __half*>(in_norm_w[li].data_ptr());
        post_np[li]= reinterpret_cast<const __half*>(post_norm_w[li].data_ptr());
        q_np[li]   = reinterpret_cast<const __half*>(q_norm_w[li].data_ptr());
        k_np[li]   = reinterpret_cast<const __half*>(k_norm_w[li].data_ptr());
        kc_p[li]   = reinterpret_cast<__half*>(k_caches[li].data_ptr());
        vc_p[li]   = reinterpret_cast<__half*>(v_caches[li].data_ptr());
    }

    const __half* cos_pos = reinterpret_cast<const __half*>(rope_cos.data_ptr()) + pos * (head_dim / 2);
    const __half* sin_pos = reinterpret_cast<const __half*>(rope_sin.data_ptr()) + pos * (head_dim / 2);

    // Expert tensor strides — g64 interleaved [E, K/64, N, 2] FP16
    const int64_t N_gu = 2 * moe_inter;
    const int64_t K_gu = hidden_size;
    const int64_t N_dn = hidden_size;
    const int64_t K_dn = moe_inter;
    const int64_t gu_packed_stride = N_gu * (K_gu / 2);
    const int64_t dn_packed_stride = N_dn * (K_dn / 2);
    const int64_t gu_scales_stride = (K_gu / 64) * N_gu * 2;  // half elements per expert
    const int64_t dn_scales_stride = (K_dn / 64) * N_dn * 2;

    // Copy input hidden state → h
    hipMemcpyAsync(h, hidden_in.data_ptr(), D * sizeof(__half), hipMemcpyDeviceToDevice, stream);

    // First layer: RMSNorm + FWHT → rot
    launch_rmsnorm_fwht(h, in_np[0], rot, D, rms_eps, stream);

    for (int li = 0; li < num_layers; li++) {
        int Nqkv = qkv_N[li];
        int No   = o_N[li];

        // 1. QKV GEMV (multi-warp for better bandwidth with small K=2048)
        launch_gemv_mw_rm_g64(rot, qkv_wp[li], qkv_sp[li], qkv, Nqkv, D, stream);

        // 2. Head RMSNorm + RoPE + FP16 KV write
        __half* q_ptr = qkv;
        __half* k_ptr = qkv + q_dim;
        __half* v_ptr = qkv + q_dim + kv_dim;
        hipLaunchKernelGGL(head_rmsnorm_rope_kv_fp16,
            dim3(num_heads + num_kv_heads), dim3(32), 0, stream,
            q_ptr, k_ptr, v_ptr, q_np[li], k_np[li],
            q_ptr, k_ptr,
            kc_p[li], vc_p[li],
            cos_pos, sin_pos,
            (int)num_heads, (int)num_kv_heads, (int)head_dim, rms_eps, max_seq, (int)pos);

        // 3. FP16 flash decode
        launch_flash_decode_fp16(qkv, kc_p[li], vc_p[li], qkv,
            p_max, p_sum, p_out,
            (int)num_heads, (int)num_kv_heads, max_seq, seq_len, (int)head_dim, stream);

        // 4. O projection
        launch_gemv_mw_rm_g64(qkv, o_wp[li], o_sp[li], obuf, No, q_dim, stream);

        // 5+6+7. Fused ResNorm+FWHT+Router GEMV+Softmax+TopK (last-block atomic)
        launch_fused_resnorm_router(obuf, h, post_np[li], rot, r,
            gate_wp[li], rlogits, tids, tprobs, rctr,
            D, E, k, rms_eps, stream);

        // 8. Expert GU GEMV (FP16 scalar dequant — bandwidth-bound, W4A4 adds overhead)
        hipLaunchKernelGGL((gemv_mw_batch_g64<1>),
            dim3((N_gu + 3) / 4, k), dim3(128), 0, stream,
            rot, egu_wp[li], egu_sp[li], batch_gu, tids,
            gu_packed_stride, gu_scales_stride, N_gu,
            (int)N_gu, (int)K_gu);

        // Phase 2: SiLU + FWHT (separate, lower VGPR pressure)
        hipLaunchKernelGGL(silu_fwht_batch,
            dim3(K_dn / 32, k), dim3(32), 0, stream,
            batch_gu, batch_fwht, (int64_t)N_gu, (int)K_dn);
        // Phase 3: Down-projection GEMV (FP16 scalar dequant)
        hipLaunchKernelGGL((gemv_warp_batch_xbatch_g64<2>),
            dim3((N_dn + 1) / 2, k), dim3(32), 0, stream,
            batch_fwht, edn_wp[li], edn_sp[li], batch_dn, tids,
            (int64_t)K_dn,
            dn_packed_stride, dn_scales_stride, N_dn,
            (int)N_dn, (int)K_dn);

        // Phase 4: Fused weighted sum + residual + norm + FWHT
        const __half* next_norm = (li < num_layers - 1) ? in_np[li + 1] : in_np[0];
        launch_moe_reduce_norm_fwht(batch_dn, tprobs, r, next_norm, rot, h, k, D, rms_eps, stream);
    }

    // Final norm + FWHT → rot, then LM head GEMV → logits
    const __half* fn_p = reinterpret_cast<const __half*>(final_norm_w.data_ptr());
    launch_rmsnorm_fwht(h, fn_p, rot, D, rms_eps, stream);

    auto logits = torch::empty({lm_N}, torch::dtype(torch::kHalf).device(device));
    launch_gemv_mw_rm_g64(rot, lm_w.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(lm_s.data_ptr()),
        reinterpret_cast<__half*>(logits.data_ptr()),
        (int)lm_N, (int)lm_K, stream);

    return logits.view({1, 1, lm_N});
}
static hipGraph_t g_persistent_graph = nullptr;
static hipGraphExec_t g_persistent_graph_exec = nullptr;
static hipStream_t g_persistent_stream = nullptr;
static torch::Tensor g_persistent_logits;
static const void* g_persistent_kv_ptr = nullptr;
static bool g_persistent_graph_valid = false;
// ============================================================
// Graph-based decode step: captures full decode as HIP graph on first call,
// subsequent calls use graph_replay_fast for zero-overhead replay.
// Returns logits [lm_N] FP16.
// ============================================================
torch::Tensor decode_step_moe_graph_logits(
    torch::Tensor hidden_in,
    std::vector<torch::Tensor> qkv_w, std::vector<torch::Tensor> qkv_s,
    std::vector<int64_t> qkv_N,
    std::vector<torch::Tensor> o_w,   std::vector<torch::Tensor> o_s,
    std::vector<int64_t> o_N,
    std::vector<torch::Tensor> gate_w,
    std::vector<torch::Tensor> exp_gu_w, std::vector<torch::Tensor> exp_gu_s,
    std::vector<torch::Tensor> exp_dn_w, std::vector<torch::Tensor> exp_dn_s,
    std::vector<torch::Tensor> exp_gu_wsum, std::vector<torch::Tensor> exp_dn_wsum,
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w,  std::vector<torch::Tensor> k_norm_w,
    torch::Tensor rope_cos, torch::Tensor rope_sin,
    int64_t pos,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t moe_inter, int64_t num_active,
    float rms_eps,
    std::vector<torch::Tensor> k_caches, std::vector<torch::Tensor> v_caches,
    torch::Tensor final_norm_w,
    torch::Tensor lm_w, torch::Tensor lm_s, int64_t lm_N, int64_t lm_K
) {
    c10::InferenceMode guard;

    // Destroy old graph if exists
    if (g_persistent_graph_valid) {
        hipGraphExecDestroy(g_persistent_graph_exec);
        hipGraphDestroy(g_persistent_graph);
        g_persistent_graph_valid = false;
    }

    auto device = hidden_in.device();

    // Create dedicated stream for graph capture
    if (!g_persistent_stream) {
        hipStreamCreate(&g_persistent_stream);
    }
    hipStream_t stream = g_persistent_stream;

    int num_layers = qkv_w.size();
    int D = hidden_size;
    int E = gate_w[0].size(0);
    int k = num_active;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int max_seq = k_caches[0].size(1);

    // Single-tile, adaptive block_s: 64 for ctx<=4k, 128 for larger
    int graph_max_ctx = std::min(max_seq, (int)(pos + 512));
    // Adaptive: bs=64 short ctx (occupancy), bs=128 long (fewer reduce iterations)
    int block_s = (graph_max_ctx <= 4608) ? 64 : 128;
    int max_n_splits = (graph_max_ctx + block_s - 1) / block_s;

    // Worst-case splits for buffer allocation
    int fa_max_splits = (max_seq + 64 - 1) / 64;
    if (!g_moe_ds.init) {
        auto opts   = torch::dtype(torch::kHalf).device(device);
        auto opts_f = torch::dtype(torch::kFloat32).device(device);
        auto opts_i = torch::dtype(torch::kInt64).device(device);
        g_moe_ds.h           = torch::empty({D}, opts);
        g_moe_ds.residual    = torch::empty({D}, opts);
        g_moe_ds.rot         = torch::empty({D}, opts);
        g_moe_ds.norot       = torch::empty({D}, opts);
        g_moe_ds.qkv_buf     = torch::empty({q_dim + 2 * kv_dim}, opts);
        g_moe_ds.o_buf       = torch::empty({D}, opts);
        g_moe_ds.gu_buf      = torch::empty({2 * (int)moe_inter}, opts);
        g_moe_ds.dn_buf      = torch::empty({D}, opts);
        g_moe_ds.moe_out_buf = torch::empty({D}, opts);
        g_moe_ds.batch_gu_buf = torch::empty({k * 2 * (int)moe_inter}, opts);
        g_moe_ds.batch_fwht_buf = torch::empty({k * (int)moe_inter}, opts);
        g_moe_ds.batch_dn_buf = torch::empty({k * D}, opts);
        g_moe_ds.batch_k     = k;
        g_moe_ds.router_logits = torch::empty({E}, opts_f);
        g_moe_ds.topk_ids    = torch::empty({k}, opts_i);
        g_moe_ds.topk_probs  = torch::empty({k}, opts_f);
        g_moe_ds.fa_partial_max = torch::empty({fa_max_splits, (long)num_heads}, opts_f);
        g_moe_ds.fa_partial_sum = torch::empty({fa_max_splits, (long)num_heads}, opts_f);
        g_moe_ds.fa_partial_out = torch::empty({fa_max_splits, (long)num_heads, (long)head_dim}, opts_f);
        g_moe_ds.fa_max_splits = fa_max_splits;
        long sk_max_n = std::max((long)(q_dim + 2 * kv_dim), (long)(q_dim));
        g_moe_ds.splitk_buf = torch::empty({2 * sk_max_n}, opts_f);
        g_moe_ds.router_counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(device));
        // W4A4 scratch buffers for udot8 GEMV
        { auto ou8 = torch::dtype(torch::kUInt8).device(device);
          int Kg = D, Kd = (int)moe_inter;
          g_moe_ds.w4a4_xq_gu  = torch::empty({Kg / 2}, ou8);
          g_moe_ds.w4a4_xsc_gu = torch::empty({Kg / 32 * 2}, opts);
          g_moe_ds.w4a4_xq_dn  = torch::empty({k * Kd / 2}, ou8);
          g_moe_ds.w4a4_xsc_dn = torch::empty({k * Kd / 32 * 2}, opts); }
        g_moe_ds.init = true;
    }

    // Persistent logits buffer
    g_persistent_logits = torch::empty({lm_N}, torch::dtype(torch::kHalf).device(device));

    auto h      = reinterpret_cast<__half*>(g_moe_ds.h.data_ptr());
    auto r      = reinterpret_cast<__half*>(g_moe_ds.residual.data_ptr());
    auto rot    = reinterpret_cast<__half*>(g_moe_ds.rot.data_ptr());
    auto norot  = reinterpret_cast<__half*>(g_moe_ds.norot.data_ptr());
    auto qkv    = reinterpret_cast<__half*>(g_moe_ds.qkv_buf.data_ptr());
    auto obuf   = reinterpret_cast<__half*>(g_moe_ds.o_buf.data_ptr());
    auto batch_gu   = reinterpret_cast<__half*>(g_moe_ds.batch_gu_buf.data_ptr());
    auto batch_fwht = reinterpret_cast<__half*>(g_moe_ds.batch_fwht_buf.data_ptr());
    auto batch_dn   = reinterpret_cast<__half*>(g_moe_ds.batch_dn_buf.data_ptr());
    auto moebuf  = reinterpret_cast<__half*>(g_moe_ds.moe_out_buf.data_ptr());
    auto rlogits = reinterpret_cast<float*>(g_moe_ds.router_logits.data_ptr());
    auto tids    = g_moe_ds.topk_ids.data_ptr<int64_t>();
    auto tprobs  = g_moe_ds.topk_probs.data_ptr<float>();
    auto p_max  = reinterpret_cast<float*>(g_moe_ds.fa_partial_max.data_ptr());
    auto p_sum  = reinterpret_cast<float*>(g_moe_ds.fa_partial_sum.data_ptr());
    auto p_out  = reinterpret_cast<float*>(g_moe_ds.fa_partial_out.data_ptr());
    auto sk_buf = reinterpret_cast<float*>(g_moe_ds.splitk_buf.data_ptr());
    auto rctr   = g_moe_ds.router_counter.data_ptr<int>();
    auto lmb_p  = reinterpret_cast<__half*>(g_persistent_logits.data_ptr());

    // Extract weight pointers
    std::vector<const uint8_t*> qkv_wp(num_layers), o_wp(num_layers);
    std::vector<const __half*>  qkv_sp(num_layers), o_sp(num_layers);
    std::vector<const __half*>  gate_wp(num_layers);
    std::vector<const uint8_t*> egu_wp(num_layers), edn_wp(num_layers);
    std::vector<const __half*>  egu_sp(num_layers), edn_sp(num_layers);
    std::vector<const __half*>  in_np(num_layers), post_np(num_layers);
    std::vector<const __half*>  q_np(num_layers), k_np(num_layers);
    std::vector<__half*>        kc_p(num_layers), vc_p(num_layers);
    for (int li = 0; li < num_layers; li++) {
        qkv_wp[li] = qkv_w[li].data_ptr<uint8_t>();
        qkv_sp[li] = reinterpret_cast<const __half*>(qkv_s[li].data_ptr());
        o_wp[li]   = o_w[li].data_ptr<uint8_t>();
        o_sp[li]   = reinterpret_cast<const __half*>(o_s[li].data_ptr());
        gate_wp[li]= reinterpret_cast<const __half*>(gate_w[li].data_ptr());
        egu_wp[li] = exp_gu_w[li].data_ptr<uint8_t>();
        egu_sp[li] = reinterpret_cast<const __half*>(exp_gu_s[li].data_ptr());
        edn_wp[li] = exp_dn_w[li].data_ptr<uint8_t>();
        edn_sp[li] = reinterpret_cast<const __half*>(exp_dn_s[li].data_ptr());
        in_np[li]  = reinterpret_cast<const __half*>(in_norm_w[li].data_ptr());
        post_np[li]= reinterpret_cast<const __half*>(post_norm_w[li].data_ptr());
        q_np[li]   = reinterpret_cast<const __half*>(q_norm_w[li].data_ptr());
        k_np[li]   = reinterpret_cast<const __half*>(k_norm_w[li].data_ptr());
        kc_p[li]   = reinterpret_cast<__half*>(k_caches[li].data_ptr());
        vc_p[li]   = reinterpret_cast<__half*>(v_caches[li].data_ptr());
    }

    const int64_t N_gu = 2 * moe_inter;
    const int64_t K_gu = hidden_size;
    const int64_t N_dn = hidden_size;
    const int64_t K_dn = moe_inter;
    const int64_t gu_packed_stride = N_gu * (K_gu / 2);
    const int64_t dn_packed_stride = N_dn * (K_dn / 2);
    const int64_t gu_scales_stride = (K_gu / 64) * N_gu * 2;
    const int64_t dn_scales_stride = (K_dn / 64) * N_dn * 2;

    const __half* cos_base = reinterpret_cast<const __half*>(rope_cos.data_ptr());
    const __half* sin_base = reinterpret_cast<const __half*>(rope_sin.data_ptr());
    const __half* fn_p = reinterpret_cast<const __half*>(final_norm_w.data_ptr());
    // Copy initial hidden state and set graph state
    hipMemcpyAsync(h, hidden_in.data_ptr(), D * sizeof(__half), hipMemcpyDeviceToDevice, stream);
    hipLaunchKernelGGL(set_graph_state, dim3(1), dim3(1), 0, stream, (int)pos);
    hipStreamSynchronize(stream);

    // ---- Capture the decode graph ----
    hipGraph_t graph;
    hipError_t err = hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    if (err != hipSuccess) {
        fprintf(stderr, "graph_logits: hipStreamBeginCapture failed: %s\n", hipGetErrorString(err));
        return g_persistent_logits;
    }

    launch_rmsnorm_fwht(h, in_np[0], rot, D, rms_eps, stream);

    for (int li = 0; li < num_layers; li++) {
        int Nqkv = qkv_N[li], No = o_N[li];

        launch_gemv_mw_rm_g64(rot, qkv_wp[li], qkv_sp[li], qkv, Nqkv, D, stream);

        __half* q_ptr = qkv;
        __half* k_ptr = qkv + q_dim;
        __half* v_ptr = qkv + q_dim + kv_dim;

        hipLaunchKernelGGL(head_rmsnorm_rope_kv_fp16_g,
            dim3(num_heads + num_kv_heads), dim3(32), 0, stream,
            q_ptr, k_ptr, v_ptr, q_np[li], k_np[li],
            q_ptr, k_ptr, kc_p[li], vc_p[li],
            cos_base, sin_base,
            (int)num_heads, (int)num_kv_heads, (int)head_dim, rms_eps, max_seq);

        // LDS-tiled flash decode with parallel reduce
        launch_flash_decode_fp16_g(qkv, kc_p[li], vc_p[li], qkv,
            p_max, p_sum, p_out,
            (int)num_heads, (int)num_kv_heads, max_seq,
            max_n_splits, block_s, (int)head_dim, stream);

        launch_gemv_mw_rm_g64(qkv, o_wp[li], o_sp[li], obuf, No, q_dim, stream);

        // Fused ResNorm+FWHT+Router GEMV+Softmax+TopK (last-block atomic)
        launch_fused_resnorm_router(obuf, h, post_np[li], rot, r,
            gate_wp[li], rlogits, tids, tprobs, rctr,
            D, E, k, rms_eps, stream);

        hipLaunchKernelGGL((gemv_mw_batch_g64<1>),
            dim3((N_gu + 3) / 4, k), dim3(128), 0, stream,
            rot, egu_wp[li], egu_sp[li], batch_gu, tids,
            gu_packed_stride, gu_scales_stride, N_gu,
            (int)N_gu, (int)K_gu);

        // SiLU + FWHT (separate)
        hipLaunchKernelGGL(silu_fwht_batch,
            dim3(K_dn / 32, k), dim3(32), 0, stream,
            batch_gu, batch_fwht, (int64_t)N_gu, (int)K_dn);
        // Down-projection GEMV
        hipLaunchKernelGGL((gemv_warp_batch_xbatch_g64<2>),
            dim3((N_dn + 1) / 2, k), dim3(32), 0, stream,
            batch_fwht, edn_wp[li], edn_sp[li], batch_dn, tids,
            (int64_t)K_dn,
            dn_packed_stride, dn_scales_stride, N_dn,
            (int)N_dn, (int)K_dn);

        const __half* next_norm = (li < num_layers - 1) ? in_np[li + 1] : in_np[0];
        launch_moe_reduce_norm_fwht(batch_dn, tprobs, r, next_norm, rot, h, k, D, rms_eps, stream);
    }

    launch_rmsnorm_fwht(h, fn_p, rot, D, rms_eps, stream);
    launch_gemv_mw_rm_g64(rot, lm_w.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(lm_s.data_ptr()),
        lmb_p, (int)lm_N, (int)lm_K, stream);

    err = hipStreamEndCapture(stream, &graph);
    if (err != hipSuccess) {
        fprintf(stderr, "graph_logits: hipStreamEndCapture failed: %s\n", hipGetErrorString(err));
        return g_persistent_logits;
    }

    hipGraphExec_t graph_exec;
    err = hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
    if (err != hipSuccess) {
        fprintf(stderr, "graph_logits: hipGraphInstantiate failed: %s\n", hipGetErrorString(err));
        hipGraphDestroy(graph);
        return g_persistent_logits;
    }

    // Store persistent state
    g_persistent_graph = graph;
    g_persistent_graph_exec = graph_exec;
    g_persistent_kv_ptr = k_caches[0].data_ptr();
    g_persistent_graph_valid = true;

    // Execute graph once for correct first-call result
    // Use default stream for proper ordering with PyTorch ops
    hipStream_t default_stream = at::hip::getCurrentHIPStream();
    hipGraphLaunch(graph_exec, default_stream);
    hipStreamSynchronize(default_stream);

    return g_persistent_logits;
}
// ============================================================
// Fast graph replay: minimal args, replays captured graph.
// Returns logits tensor if graph valid, empty tensor otherwise.
// ============================================================
torch::Tensor graph_replay_fast(
    torch::Tensor hidden_in,
    int64_t pos,
    torch::Tensor kv_check
) {
    // Return empty tensor if graph not captured or KV cache reallocated
    if (!g_persistent_graph_valid ||
        kv_check.data_ptr() != g_persistent_kv_ptr) {
        return torch::Tensor();  // None in Python
    }

    // Use PyTorch's default stream for proper ordering with downstream ops
    hipStream_t stream = at::hip::getCurrentHIPStream();
    int D = hidden_in.numel();
    auto h = reinterpret_cast<__half*>(g_moe_ds.h.data_ptr());

    // Copy hidden state and set position
    hipMemcpyAsync(h, hidden_in.data_ptr(), D * sizeof(__half), hipMemcpyDeviceToDevice, stream);
    hipLaunchKernelGGL(set_graph_state, dim3(1), dim3(1), 0, stream, (int)pos);

    // Replay graph (no sync — caller handles synchronization)
    hipGraphLaunch(g_persistent_graph_exec, stream);

    return g_persistent_logits;
}
// ============================================================
// Benchmark graph replay: captures graph at context_len, replays num_iters times.
// Uses existing persistent graph if available, otherwise requires prior
// decode_step_moe_graph_logits call.
// ============================================================
void bench_graph_replay(int64_t num_iters, int64_t context_len) {
    if (!g_persistent_graph_valid) {
        fprintf(stderr, "bench_graph_replay: no graph captured. Call decode_step_moe_graph_logits first.\n");
        return;
    }

    hipStream_t stream = at::hip::getCurrentHIPStream();

    // Warmup
    for (int i = 0; i < 3; i++) {
        hipLaunchKernelGGL(set_graph_state, dim3(1), dim3(1), 0, stream, (int)context_len);
        hipGraphLaunch(g_persistent_graph_exec, stream);
    }
    hipStreamSynchronize(stream);

    // Timed benchmark (HIP events — max ~100 iters to avoid HIP graph queue overflow)
    hipEvent_t t_start, t_end;
    hipEventCreate(&t_start);
    hipEventCreate(&t_end);
    hipEventRecord(t_start, stream);
    for (int64_t t = 0; t < num_iters; t++) {
        int pos = context_len + t;
        hipLaunchKernelGGL(set_graph_state, dim3(1), dim3(1), 0, stream, pos);
        hipGraphLaunch(g_persistent_graph_exec, stream);
    }
    hipEventRecord(t_end, stream);
    hipStreamSynchronize(stream);

    float total_ms;
    hipEventElapsedTime(&total_ms, t_start, t_end);
    float ms_per_tok = total_ms / num_iters;
    fprintf(stderr, "Graph replay bench: %.3f ms/tok, %.1f t/s (%ld iters, ctx=%ld)\n",
            ms_per_tok, 1000.0f / ms_per_tok, num_iters, context_len);

    hipEventDestroy(t_start);
    hipEventDestroy(t_end);
}
// Pre-compute w_sum for all experts: returns [E, N, K/32] uint16 tensor
torch::Tensor precompute_wsum(
    torch::Tensor W_packed,   // [E, N, K/2] uint8
    int64_t N, int64_t K
) {
    int E = W_packed.size(0);
    int nit = K / 32;
    auto device = W_packed.device();
    auto stream = at::hip::getCurrentHIPStream();
    auto W_sum = torch::empty({E, N, (long)nit},
        torch::dtype(torch::kInt16).device(device));  // uint16 stored as int16

    int total_per_expert = N * nit;
    dim3 grid((total_per_expert + 255) / 256, E);
    dim3 block(256);
    int64_t packed_stride = (int64_t)N * (K / 2);

    hipLaunchKernelGGL(precompute_wsum_kernel, grid, block, 0, stream,
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<uint16_t*>(W_sum.data_ptr<int16_t>()),
        packed_stride, N, K);

    return W_sum;
}

// ============================================================
// INT4 WMMA v7: 2-pass W4A4 GEMM — zero extra VRAM, 916 t/s pp128
// ============================================================

std::vector<torch::Tensor> v7_prepare_weights(
    torch::Tensor packed_w, torch::Tensor scales, int64_t N, int64_t K, int64_t group_size)
{
    auto device = packed_w.device();
    auto stream = at::hip::getCurrentHIPStream();
    int ng = K / group_size;
    int N_pad = ((N + 63) / 64) * 64;
    auto wt = torch::empty({(long)((K/8)*(N_pad/16)*64)}, torch::dtype(torch::kUInt8).device(device));
    torch::Tensor w_padded = (N_pad != N) ?
        torch::zeros({N_pad, K/2}, torch::dtype(torch::kUInt8).device(device)) : packed_w.contiguous();
    if (N_pad != N) w_padded.narrow(0, 0, N).copy_(packed_w);
    int w_total = (K/8) * (N_pad/16) * 16;
    hipLaunchKernelGGL(wmma_tile_weights, dim3((w_total+255)/256), dim3(256), 0, stream,
        w_padded.data_ptr<uint8_t>(), wt.data_ptr<uint8_t>(), N_pad, K);
    torch::Tensor s_flat = scales.reshape({(long)N, (long)(ng*2)}).contiguous();
    torch::Tensor s_padded = (N_pad != N) ?
        torch::zeros({N_pad, (long)(ng*2)}, torch::dtype(torch::kHalf).device(device)) : s_flat;
    if (N_pad != N) s_padded.narrow(0, 0, N).copy_(s_flat);
    auto sz_gn = torch::empty({(long)ng, N_pad, 2}, torch::dtype(torch::kHalf).device(device));
    hipLaunchKernelGGL(wmma_transpose_scales, dim3((N_pad*ng+255)/256), dim3(256), 0, stream,
        reinterpret_cast<const __half*>(s_padded.data_ptr()),
        reinterpret_cast<__half*>(sz_gn.data_ptr()), N_pad, ng);
    return {wt, sz_gn.reshape({(long)(ng*N_pad*2)})};
}

void v7_gemm(
    torch::Tensor input, torch::Tensor output,
    torch::Tensor wt, torch::Tensor sz_gn,
    int64_t N_pad, int64_t K, int64_t group_size,
    torch::Tensor act_q_buf, torch::Tensor act_meta_buf)
{
    int M = input.size(0);
    int M_pad = ((M + 15) / 16) * 16;
    int ng = K / group_size;
    auto stream = at::hip::getCurrentHIPStream();
   hipLaunchKernelGGL(( wmma_quantize_act), dim3(dim3(ng, M_pad)), dim3(32), 0, stream, 
        reinterpret_cast<const __half*>(input.data_ptr()),
        act_q_buf.data_ptr<uint8_t>(), act_meta_buf.data_ptr<float>(), M, K, M_pad);
    dim3 grid((N_pad + 63) / 64, (M + 15) / 16);
    if (group_size == 64) {
       hipLaunchKernelGGL(( wmma_int4_gemm<64>), dim3(grid), dim3(32), 0, stream, 
            act_q_buf.data_ptr<uint8_t>(), act_meta_buf.data_ptr<float>(),
            wt.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(sz_gn.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()), M, M_pad, (int)N_pad, (int)K);
    } else {
       hipLaunchKernelGGL(( wmma_int4_gemm<32>), dim3(grid), dim3(32), 0, stream, 
            act_q_buf.data_ptr<uint8_t>(), act_meta_buf.data_ptr<float>(),
            wt.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(sz_gn.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()), M, M_pad, (int)N_pad, (int)K);
    }
}

std::vector<torch::Tensor> v7_alloc_scratch(int64_t M_max, int64_t K_max, torch::Device device) {
    int M_pad = ((M_max + 15) / 16) * 16;
    auto aq = torch::empty({M_pad, K_max / 2}, torch::dtype(torch::kUInt8).device(device));
    auto am = torch::empty({M_pad, K_max / 32, 3}, torch::dtype(torch::kFloat32).device(device));
    return {aq, am};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prefill_moe_logits", &prefill_moe_logits,
        "MoE prefill: FP16 attention + INT4 expert dispatch, all in C++. Returns [M, vocab].");
    m.def("dequant_int4_g64", &dequant_int4_g64, "Dequant INT4 g64 packed+scales to FP16 [N, K]");
    m.def("dequant_gemm_tiled_g64", &dequant_gemm_tiled_g64,
        "Tiled dequant+GEMM g64: dequant in N-tiles for L2 caching.");
    m.def("dequant_gemm_tiled_g32", &dequant_gemm_tiled_g32,
        "Tiled dequant+GEMM g32: dequant in N-tiles for L2 caching.");
    m.def("fp16_to_fp8", &fp16_to_fp8, "Convert FP16 tensor to FP8 E4M3 (uint8)");
    m.def("moe_gemv_top_k", &moe_gemv_top_k,
        "MoE GEMV: top-k expert dispatch for batch=1 decode.");
    m.def("moe_gemv_top_k_v2", &moe_gemv_top_k_v2,
        "MoE GEMV top-k, no CPU-GPU sync: ids/weights stay on GPU.");
    m.def("decode_step_moe_logits", &int4_decode_step_moe_logits,
        "MoE decode step: all layers in C++, FP16 KV cache, returns logits [1,1,vocab].");
    m.def("decode_step_moe_graph_logits", &decode_step_moe_graph_logits,
        "MoE HIP Graph decode step: captures full decode as graph. Returns logits [vocab] FP16.");
    m.def("graph_replay_fast", &graph_replay_fast,
        "Fast graph replay: minimal args (hidden, pos, kv_check).");
    m.def("bench_graph_replay", &bench_graph_replay,
        "Benchmark persistent graph replay.");
    m.def("precompute_wsum", &precompute_wsum,
        "Pre-compute w_sum [E, N, K/32] uint16 for W4A4 udot8 kernel.");
    m.def("hadamard32", [](torch::Tensor X) -> torch::Tensor {
        c10::InferenceMode guard;
        auto stream = at::hip::getCurrentHIPStream();
        auto X_c = X.contiguous();
        auto sizes = X_c.sizes().vec();
        int D = sizes.back();
        int M = X_c.numel() / D;
        auto Y = torch::empty_like(X_c);
        int BS = 1024;
        int grid_y = (D + BS - 1) / BS;
        hipLaunchKernelGGL(hadamard32_batched, dim3(M, grid_y), dim3(BS), 0, stream,
            reinterpret_cast<const __half*>(X_c.data_ptr()),
            reinterpret_cast<__half*>(Y.data_ptr()), D);
        return Y;
    }, "Walsh-Hadamard(32) butterfly transform on last dim.");
    // v7 INT4 WMMA kernels
    m.def("v7_prepare_weights", &v7_prepare_weights,
        "Prepare tiled weights + transposed scales for v7 WMMA GEMM. Returns [wt, sz_gn].");
    m.def("v7_gemm", &v7_gemm,
        "v7 INT4 WMMA GEMM: 2-pass (quantize activations + WMMA). Zero extra VRAM.");
    m.def("v7_alloc_scratch", &v7_alloc_scratch,
        "Allocate v7 scratch buffers. Returns [act_q, act_meta].");

    // Symmetric WMMA INT4 GEMM — zero correction in inner loop
    m.def("wmma_sym_gemm", [](
        torch::Tensor input,      // [M, K] FP16
        torch::Tensor wt,         // tiled weights (from v7_tile_weights)
        torch::Tensor w_scales,   // [ng, N_pad] float32 (transposed weight scales)
        torch::Tensor output,     // [M, N_pad] FP16
        int64_t N_pad, int64_t K, int64_t group_size
    ) {
        int M = input.size(0);
        int M_pad = ((M + 15) / 16) * 16;
        int ng = K / group_size;
        auto device = input.device();
        auto stream = at::hip::getCurrentHIPStream();

        // Quantize activations to signed symmetric INT4
        auto act_q = torch::empty({M_pad, K / 2}, torch::dtype(torch::kUInt8).device(device));
        auto act_scales = torch::empty({M_pad, (long)ng}, torch::dtype(torch::kFloat32).device(device));

        // Signed symmetric quantize: absmax / 7, store unsigned+8
        hipLaunchKernelGGL(wmma_quantize_act_sym, dim3(ng, M_pad), dim3(32), 0, stream,
            reinterpret_cast<const __half*>(input.data_ptr()),
            act_q.data_ptr<uint8_t>(), act_scales.data_ptr<float>(), M, K, M_pad);

        // WMMA GEMM — 1-warp symmetric, 16M×64N per block, max occupancy
        dim3 grid((N_pad + 63) / 64, (M + 15) / 16);
        if (group_size == 64) {
            hipLaunchKernelGGL((wmma_int4_gemm_sym<64>), grid, dim3(32), 0, stream,
                act_q.data_ptr<uint8_t>(), act_scales.data_ptr<float>(),
                wt.data_ptr<uint8_t>(), w_scales.data_ptr<float>(),
                reinterpret_cast<__half*>(output.data_ptr()), M, M_pad, (int)N_pad, (int)K);
        } else {
            hipLaunchKernelGGL((wmma_int4_gemm_sym<32>), grid, dim3(32), 0, stream,
                act_q.data_ptr<uint8_t>(), act_scales.data_ptr<float>(),
                wt.data_ptr<uint8_t>(), w_scales.data_ptr<float>(),
                reinterpret_cast<__half*>(output.data_ptr()), M, M_pad, (int)N_pad, (int)K);
        }
    }, "Symmetric WMMA INT4 GEMM — zero scale correction in inner loop. ~5x faster than asymmetric.");
}
