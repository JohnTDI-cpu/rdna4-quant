/*
 * INT4 Single Decode Step — C++ loop over all layers
 * Eliminates Python per-layer overhead (~3ms → ~0ms).
 *
 * Architecture:
 * - HIP GEMV V2 (warp-based, row-major weights) for all projections
 * - HIP fused residual+RMSNorm
 * - HIP SiLU*mul
 * - Triton QKNorm+RoPE called via PyTorch C++ (or inlined HIP RoPE)
 * - PyTorch SDPA called from C++
 * - Pre-allocated scratch buffers
 */

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <vector>
#include <cmath>

// ============================================================
// Device-side decode state for HIP Graph (updated between replays)
// ============================================================
__device__ int g_graph_pos;
__device__ int g_graph_seq_len;

__global__ void set_graph_state(int pos) {
    g_graph_pos = pos;
    g_graph_seq_len = pos + 1;
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
    // Normal path (vast majority of values): bit-compose FP32
    // sign | ((exp + 120) << 23) | (mant << 20)
    uint32_t sign = ((uint32_t)val & 0x80) << 24;
    uint32_t exp = ((uint32_t)val >> 3) & 0xF;
    uint32_t mant = (uint32_t)val & 0x7;
    uint32_t f32 = sign | ((exp + 120u) << 23) | (mant << 20);
    // Fix zero: if val&0x7F == 0, force result to ±0
    // Fix denorm: small error acceptable for inference (denorms are rare in KV cache)
    return (val & 0x7F) ? __uint_as_float(f32) : 0.0f;
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

// Forward declaration — kernel defined later, needed by dequant_matmul helper
__global__ void dequant_int4_kernel(
    const uint8_t* __restrict__ packed, const __half* __restrict__ scales,
    __half* __restrict__ out, int N, int K);

// ============================================================
// GEMV V3 — Vectorized loads (uint4 128-bit), INT4 decode
// Each warp handles BLOCK_N output rows. K dimension iterated in chunks of 32.
// Inner loop uses 128-bit loads for weights (16 bytes = 32 INT4 values at once)
// and __half2 loads for X (coalesced).
// ============================================================

template<int BLOCK_N>
__global__ void gemv_warp(
    const __half* __restrict__ X,
    const uint8_t* __restrict__ W,     // [N, K/2]
    const __half* __restrict__ S,      // [N, K/32, 2] interleaved (scale, zero_fp16)
    __half* __restrict__ Y,
    int N, int K
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / 32, nit = K / 32;
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

        #pragma unroll
        for (int ni = 0; ni < BLOCK_N; ni++) {
            int n = n0 + ni;
            if (n >= N) continue;
            // Load interleaved (scale, zero) pair as half2
            __half2 sz = *reinterpret_cast<const __half2*>(&S[(n * nsc + it) * 2]);
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
    __half* Y, int N, int K, hipStream_t st) {
    // BLOCK_N=4: optimal for this K dimension (nit=K/32 divides evenly by 32 lanes)
    hipLaunchKernelGGL((gemv_warp<4>), dim3((N+3)/4), dim3(32), 0, st, X, W, S, Y, N, K);
}

// ============================================================
// Fused SiLU-input GEMV: Y = W @ (SiLU(Gate) * Up)
// Eliminates separate silu_mul_ds kernel launch.
// ============================================================
template<int BLOCK_N>
__global__ void gemv_warp_silu_in(
    const __half* __restrict__ Gate,
    const __half* __restrict__ Up,
    const uint8_t* __restrict__ W,
    const __half* __restrict__ S,
    __half* __restrict__ Y,
    int N, int K
) {
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

static inline __attribute__((always_inline)) void launch_gemv_silu_in(
    const __half* Gate, const __half* Up,
    const uint8_t* W, const __half* S,
    __half* Y, int N, int K, hipStream_t st) {
    hipLaunchKernelGGL((gemv_warp_silu_in<4>), dim3((N+3)/4), dim3(32), 0, st, Gate, Up, W, S, Y, N, K);
}

// ============================================================
// Fused Norm+GEMV kernels: eliminate norm kernel + inter-kernel gap
// Each block independently computes RMS norm of X (10KB, L2 cached)
// then does GEMV with the normed values — no global store/load of normed X.
// ============================================================

// Fused RMSNorm + GEMV: Y = W_proj @ RMSNorm(X, NormW)
template<int BLOCK_N>
__global__ void gemv_warp_norm(
    const __half* __restrict__ X,        // [K] raw input
    const __half* __restrict__ NormW,    // [K] norm weights
    const uint8_t* __restrict__ W,       // [N, K/2] INT4 weights
    const __half* __restrict__ S,        // [N, K/32, 2] interleaved (scale, zero)
    __half* __restrict__ Y,              // [N] output
    int N, int K, float rms_eps
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / 32, nit = K / 32;

    // Pass 1: compute sum-of-squares for RMS norm
    float ss = 0.0f;
    for (int it = lane; it < nit; it += 32) {
        #pragma unroll
        for (int i = 0; i < 32; i += 2) {
            __half2 xh = *reinterpret_cast<const __half2*>(&X[it * 32 + i]);
            float x0 = __half2float(xh.x), x1 = __half2float(xh.y);
            ss += x0 * x0 + x1 * x1;
        }
    }
    ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
    ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
    float rrms = rsqrtf(ss / (float)K + rms_eps);

    // Pass 2: GEMV with inline norm (X re-read from L2 cache)
    float acc[BLOCK_N];
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i++) acc[i] = 0.0f;

    for (int it = lane; it < nit; it += 32) {
        float xv[32];
        #pragma unroll
        for (int i = 0; i < 32; i += 2) {
            __half2 xh = *reinterpret_cast<const __half2*>(&X[it * 32 + i]);
            __half2 wh = *reinterpret_cast<const __half2*>(&NormW[it * 32 + i]);
            xv[i]   = __half2float(xh.x) * __half2float(wh.x) * rrms;
            xv[i+1] = __half2float(xh.y) * __half2float(wh.y) * rrms;
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

// Fused Residual+RMSNorm + GEMV: ResOut = A+B, Y = W_proj @ RMSNorm(A+B, NormW)
// Block 0 writes ResOut; all blocks independently compute norm.
template<int BLOCK_N>
__global__ void gemv_warp_res_norm(
    const __half* __restrict__ A,        // [K] first addend
    const __half* __restrict__ B,        // [K] second addend (residual)
    __half* __restrict__ ResOut,         // [K] output residual = A+B
    const __half* __restrict__ NormW,    // [K] norm weights
    const uint8_t* __restrict__ W,       // [N, K/2] INT4 weights
    const __half* __restrict__ S,        // [N, K/32, 2] interleaved (scale, zero)
    __half* __restrict__ Y,              // [N] output
    int N, int K, float rms_eps
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / 32, nit = K / 32;

    // Pass 1: compute residual + sum-of-squares
    float ss = 0.0f;
    for (int it = lane; it < nit; it += 32) {
        #pragma unroll
        for (int i = 0; i < 32; i += 2) {
            __half2 ah = *reinterpret_cast<const __half2*>(&A[it * 32 + i]);
            __half2 bh = *reinterpret_cast<const __half2*>(&B[it * 32 + i]);
            float r0 = __half2float(ah.x) + __half2float(bh.x);
            float r1 = __half2float(ah.y) + __half2float(bh.y);
            ss += r0 * r0 + r1 * r1;
            // Block 0 writes residual output (safe: no other kernel reads ResOut until this finishes)
            if (blockIdx.x == 0) {
                ResOut[it * 32 + i]     = __float2half(r0);
                ResOut[it * 32 + i + 1] = __float2half(r1);
            }
        }
    }
    ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
    ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
    float rrms = rsqrtf(ss / (float)K + rms_eps);

    // Pass 2: GEMV with inline residual+norm (A,B re-read from L2)
    float acc[BLOCK_N];
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i++) acc[i] = 0.0f;

    for (int it = lane; it < nit; it += 32) {
        float xv[32];
        #pragma unroll
        for (int i = 0; i < 32; i += 2) {
            __half2 ah = *reinterpret_cast<const __half2*>(&A[it * 32 + i]);
            __half2 bh = *reinterpret_cast<const __half2*>(&B[it * 32 + i]);
            __half2 wh = *reinterpret_cast<const __half2*>(&NormW[it * 32 + i]);
            float r0 = __half2float(ah.x) + __half2float(bh.x);
            float r1 = __half2float(ah.y) + __half2float(bh.y);
            xv[i]   = r0 * __half2float(wh.x) * rrms;
            xv[i+1] = r1 * __half2float(wh.y) * rrms;
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

static inline __attribute__((always_inline)) void launch_gemv_norm(
    const __half* X, const __half* NormW,
    const uint8_t* W, const __half* S,
    __half* Y, int N, int K, float rms_eps, hipStream_t st) {
    if (N <= 2048) {
        hipLaunchKernelGGL((gemv_warp_norm<4>), dim3((N+3)/4), dim3(32), 0, st, X, NormW, W, S, Y, N, K, rms_eps);
    } else if (N <= 16384) {
        hipLaunchKernelGGL((gemv_warp_norm<2>), dim3((N+1)/2), dim3(32), 0, st, X, NormW, W, S, Y, N, K, rms_eps);
    } else {
        hipLaunchKernelGGL((gemv_warp_norm<4>), dim3((N+3)/4), dim3(32), 0, st, X, NormW, W, S, Y, N, K, rms_eps);
    }
}

static inline __attribute__((always_inline)) void launch_gemv_res_norm(
    const __half* A, const __half* B, __half* ResOut, const __half* NormW,
    const uint8_t* W, const __half* S,
    __half* Y, int N, int K, float rms_eps, hipStream_t st) {
    if (N <= 2048) {
        hipLaunchKernelGGL((gemv_warp_res_norm<4>), dim3((N+3)/4), dim3(32), 0, st, A, B, ResOut, NormW, W, S, Y, N, K, rms_eps);
    } else if (N <= 16384) {
        hipLaunchKernelGGL((gemv_warp_res_norm<2>), dim3((N+1)/2), dim3(32), 0, st, A, B, ResOut, NormW, W, S, Y, N, K, rms_eps);
    } else {
        hipLaunchKernelGGL((gemv_warp_res_norm<4>), dim3((N+3)/4), dim3(32), 0, st, A, B, ResOut, NormW, W, S, Y, N, K, rms_eps);
    }
}

// ============================================================
// FWHT-fused GEMV kernels — eliminate serial norm+Hadamard bottleneck
// Each kernel embeds the Fast Walsh-Hadamard Transform inline,
// replacing the explicit H[32,32] matrix multiply with butterfly ops.
// ============================================================

// Fused Hadamard + GEMV: Y = W @ FWHT(X)
// Used for O projection after attention output.
template<int BLOCK_N>
__global__ void gemv_warp_had(
    const __half* __restrict__ X,
    const uint8_t* __restrict__ W,
    const __half* __restrict__ S,
    __half* __restrict__ Y,
    int N, int K
) {
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

// Fused Residual+RMSNorm+FWHT + GEMV: ResOut=A+B, Y = W @ FWHT(RMSNorm(A+B, NormW))
// Replaces serial fused_res_norm_hadamard + launch_gemv.
// Each block redundantly computes norm from L2-cached input (~10KB), then does GEMV.
template<int BLOCK_N>
__global__ void gemv_warp_res_norm_had(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ ResOut,
    const __half* __restrict__ NormW,
    const uint8_t* __restrict__ W,
    const __half* __restrict__ S,      // [N, K/32, 2] interleaved (scale, zero)
    __half* __restrict__ Y,
    int N, int K, float rms_eps
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / 32, nit = K / 32;

    // Pass 1: compute residual + sum-of-squares
    float ss = 0.0f;
    for (int it = lane; it < nit; it += 32) {
        #pragma unroll
        for (int i = 0; i < 32; i += 2) {
            __half2 ah = *reinterpret_cast<const __half2*>(&A[it * 32 + i]);
            __half2 bh = *reinterpret_cast<const __half2*>(&B[it * 32 + i]);
            float r0 = __half2float(ah.x) + __half2float(bh.x);
            float r1 = __half2float(ah.y) + __half2float(bh.y);
            ss += r0 * r0 + r1 * r1;
            if (blockIdx.x == 0) {
                ResOut[it * 32 + i]     = __float2half(r0);
                ResOut[it * 32 + i + 1] = __float2half(r1);
            }
        }
    }
    ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
    ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
    float rrms = rsqrtf(ss / (float)K + rms_eps);

    // Pass 2: GEMV with inline residual+norm+FWHT
    float acc[BLOCK_N];
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i++) acc[i] = 0.0f;

    for (int it = lane; it < nit; it += 32) {
        float xv[32];
        #pragma unroll
        for (int i = 0; i < 32; i += 2) {
            __half2 ah = *reinterpret_cast<const __half2*>(&A[it * 32 + i]);
            __half2 bh = *reinterpret_cast<const __half2*>(&B[it * 32 + i]);
            __half2 wh = *reinterpret_cast<const __half2*>(&NormW[it * 32 + i]);
            float r0 = __half2float(ah.x) + __half2float(bh.x);
            float r1 = __half2float(ah.y) + __half2float(bh.y);
            xv[i]   = r0 * __half2float(wh.x) * rrms;
            xv[i+1] = r1 * __half2float(wh.y) * rrms;
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

// Fused SiLU*Up + FWHT + GEMV: Y = W @ FWHT(SiLU(Gate) * Up)
// Replaces silu_hadamard_ds + launch_gemv.
template<int BLOCK_N>
__global__ void gemv_warp_silu_had(
    const __half* __restrict__ Gate,
    const __half* __restrict__ Up,
    const uint8_t* __restrict__ W,
    const __half* __restrict__ S,      // [N, K/32, 2] interleaved (scale, zero)
    __half* __restrict__ Y,
    int N, int K
) {
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

static inline __attribute__((always_inline)) void launch_gemv_had(
    const __half* X, const uint8_t* W, const __half* S,
    __half* Y, int N, int K, hipStream_t st) {
    hipLaunchKernelGGL((gemv_warp_had<4>), dim3((N+3)/4), dim3(32), 0, st, X, W, S, Y, N, K);
}

static inline __attribute__((always_inline)) void launch_gemv_res_norm_had(
    const __half* A, const __half* B, __half* ResOut, const __half* NormW,
    const uint8_t* W, const __half* S,
    __half* Y, int N, int K, float rms_eps, hipStream_t st) {
    hipLaunchKernelGGL((gemv_warp_res_norm_had<4>), dim3((N+3)/4), dim3(32), 0, st,
        A, B, ResOut, NormW, W, S, Y, N, K, rms_eps);
}

static inline __attribute__((always_inline)) void launch_gemv_silu_had(
    const __half* Gate, const __half* Up,
    const uint8_t* W, const __half* S,
    __half* Y, int N, int K, hipStream_t st) {
    hipLaunchKernelGGL((gemv_warp_silu_had<4>), dim3((N+3)/4), dim3(32), 0, st,
        Gate, Up, W, S, Y, N, K);
}

// ============================================================
// Legacy fused kernels (kept for reference/fallback)
// ============================================================

// Block-diagonal FWHT rotation: X[K] -> X_rotated[K]
// Warp-shuffle butterfly — no shared memory, no H matrix, O(n log n) instead of O(n²).
// Grid: (K/32), Block: (32)
__global__ void hadamard_rotate(
    const __half* __restrict__ X_in,
    __half* __restrict__ X_out,
    const __half* __restrict__ H,  // unused (kept for API compat)
    int K
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = bid * 32;

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

static inline void launch_hadamard(
    const __half* X_in, __half* X_out, const __half* H,
    int K, hipStream_t st) {
    hipLaunchKernelGGL(hadamard_rotate, dim3(K / 32), dim3(32), 0, st, X_in, X_out, H, K);
}

// Batched FWHT rotation: X[M, K] -> X_out[M, K]
// Warp-shuffle butterfly — no shared memory.
// Grid: (K/32, M), Block: (32)
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

// RMSNorm
__global__ void rmsnorm_ds(const __half* X, const __half* W, __half* Y, int D, float eps) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x, BS = blockDim.x;
    float sq = 0.0f;
    for (int i = tid; i < D; i += BS) { float x = __half2float(X[i]); smem[i] = x; sq += x * x; }
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
    for (int i = tid; i < D; i += BS) Y[i] = __float2half(smem[i] * rrms * __half2float(W[i]));
}

// Fused residual + RMSNorm: Y = RMSNorm(X+Res, W), ResOut = X+Res
__global__ void fused_res_norm(const __half* X, const __half* Res, const __half* W,
                                __half* Y, __half* ResOut, int D, float eps) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x, BS = blockDim.x;
    float sq = 0.0f;
    for (int i = tid; i < D; i += BS) {
        float xi = __half2float(X[i]) + __half2float(Res[i]);
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
        Y[i] = __float2half(xi * rrms * __half2float(W[i]));
        ResOut[i] = __float2half(xi);
    }
}

// Per-head RMSNorm for Q and K
__global__ void head_rmsnorm(const __half* Q, const __half* K,
                              const __half* QW, const __half* KW,
                              __half* Q_out, __half* K_out,
                              int H, int Hk, int D, float eps) {
    int pid = blockIdx.x;
    int tid = threadIdx.x;

    // Process Q heads
    if (pid < H) {
        const __half* src = Q + pid * D;
        __half* dst = Q_out + pid * D;
        float sq = 0.0f;
        for (int i = tid; i < D; i += blockDim.x) {
            float v = __half2float(src[i]); sq += v * v;
        }
        sq += __shfl_xor(sq, 16); sq += __shfl_xor(sq, 8);
        sq += __shfl_xor(sq, 4);  sq += __shfl_xor(sq, 2); sq += __shfl_xor(sq, 1);
        float rrms = rsqrtf(sq / (float)D + eps);
        for (int i = tid; i < D; i += blockDim.x)
            dst[i] = __float2half(__half2float(src[i]) * rrms * __half2float(QW[i]));
    }
    // Process K heads (reuse same grid)
    int kid = pid - H;
    if (kid >= 0 && kid < Hk) {
        const __half* src = K + kid * D;
        __half* dst = K_out + kid * D;
        float sq = 0.0f;
        for (int i = tid; i < D; i += blockDim.x) {
            float v = __half2float(src[i]); sq += v * v;
        }
        sq += __shfl_xor(sq, 16); sq += __shfl_xor(sq, 8);
        sq += __shfl_xor(sq, 4);  sq += __shfl_xor(sq, 2); sq += __shfl_xor(sq, 1);
        float rrms = rsqrtf(sq / (float)D + eps);
        for (int i = tid; i < D; i += blockDim.x)
            dst[i] = __float2half(__half2float(src[i]) * rrms * __half2float(KW[i]));
    }
}

// Half-interleaved RoPE (Qwen3-style)
__global__ void rope_ds(
    __half* Q, __half* K, const __half* COS, const __half* SIN,
    int H, int Hk, int D
) {
    int pid = blockIdx.x;
    int tid = threadIdx.x;
    int half_D = D / 2;
    if (tid >= half_D) return;
    float c = __half2float(COS[tid]), s = __half2float(SIN[tid]);
    if (pid < H) {
        int base = pid * D;
        float e = __half2float(Q[base + tid]);
        float o = __half2float(Q[base + tid + half_D]);
        Q[base + tid] = __float2half(e * c - o * s);
        Q[base + tid + half_D] = __float2half(o * c + e * s);
    }
    if (pid < Hk) {
        int base = pid * D;
        float e = __half2float(K[base + tid]);
        float o = __half2float(K[base + tid + half_D]);
        K[base + tid] = __float2half(e * c - o * s);
        K[base + tid + half_D] = __float2half(o * c + e * s);
    }
}

// SiLU * up
__global__ void silu_mul_ds(const __half* G, const __half* U, __half* Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float g = __half2float(G[i]);
        Y[i] = __float2half(g / (1.0f + expf(-g)) * __half2float(U[i]));
    }
}

// ============================================================
// Fused Norm+Hadamard kernels — eliminate separate hadamard launches
// ============================================================

// RMSNorm + Hadamard rotation: Y = Hadamard(RMSNorm(X, W))
// Single kernel replaces rmsnorm_ds + hadamard_rotate
// Grid: (1), Block: (256)
__global__ void rmsnorm_hadamard(
    const __half* __restrict__ X, const __half* __restrict__ W,
    __half* __restrict__ Y, const __half* __restrict__ H,
    int D, float eps
) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x, BS = blockDim.x;

    // Phase 1: load X, compute sum-of-squares
    float sq = 0.0f;
    for (int i = tid; i < D; i += BS) {
        float x = __half2float(X[i]); smem[i] = x; sq += x * x;
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

    // Phase 2: apply norm weights, store normed values in smem
    for (int i = tid; i < D; i += BS)
        smem[i] = smem[i] * rrms * __half2float(W[i]);
    __syncthreads();

    // Phase 3: Hadamard rotation from smem → Y
    for (int hb = tid / 32; hb < D / 32; hb += BS / 32) {
        int local = tid % 32;
        int base = hb * 32;
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++)
            sum += smem[base + j] * __half2float(H[j * 32 + local]);
        Y[base + local] = __float2half(sum);
    }
}

// Fused Residual + RMSNorm + Hadamard: Y = Hadamard(RMSNorm(X+Res, W)), ResOut = X+Res
// Replaces fused_res_norm + hadamard_rotate
// Grid: (1), Block: (256)
__global__ void fused_res_norm_hadamard(
    const __half* __restrict__ X, const __half* __restrict__ Res,
    const __half* __restrict__ W,
    __half* __restrict__ Y, __half* __restrict__ ResOut,
    const __half* __restrict__ H,
    int D, float eps
) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x, BS = blockDim.x;

    // Phase 1: residual + sum-of-squares
    float sq = 0.0f;
    for (int i = tid; i < D; i += BS) {
        float xi = __half2float(X[i]) + __half2float(Res[i]);
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

    // Phase 2: apply norm weights + write residual, store normed in smem
    for (int i = tid; i < D; i += BS) {
        float xi = smem[i];
        ResOut[i] = __float2half(xi);
        smem[i] = xi * rrms * __half2float(W[i]);
    }
    __syncthreads();

    // Phase 3: Hadamard rotation from smem → Y
    for (int hb = tid / 32; hb < D / 32; hb += BS / 32) {
        int local = tid % 32;
        int base = hb * 32;
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++)
            sum += smem[base + j] * __half2float(H[j * 32 + local]);
        Y[base + local] = __float2half(sum);
    }
}

// SiLU*Up + FWHT: Y = FWHT(SiLU(G) * U)
// Warp-shuffle butterfly — no shared memory, no H matrix.
// Grid: (N/32), Block: (32) — one warp per 32-element block
__global__ void silu_hadamard_ds(
    const __half* __restrict__ G, const __half* __restrict__ U,
    __half* __restrict__ Y, const __half* __restrict__ H,  // H unused (API compat)
    int N
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = bid * 32;
    if (base + tid >= N) return;

    float g = __half2float(G[base + tid]);
    float val = g / (1.0f + expf(-g)) * __half2float(U[base + tid]);
    #pragma unroll
    for (int step = 1; step < 32; step *= 2) {
        float other = __shfl_xor(val, step);
        if (tid & step) val = other - val;
        else val = val + other;
    }
    val *= 0.17677669529663689f; // 1/sqrt(32)
    Y[base + tid] = __float2half(val);
}

// ============================================================
// ============================================================
// 2-phase norm kernels — eliminate redundant sum-of-squares
// Phase 1: single block computes residual + rrms (1 scalar output)
// Phase 2: D/32 blocks each apply norm + FWHT (read rrms, no redundant work)
// ============================================================

// Phase 1: Residual add + compute rrms. Single block of 256 threads.
// Writes ResOut = A + B, and rrms_out = rsqrt(mean((A+B)^2) + eps)
__global__ void residual_add_rrms(
    const __half* __restrict__ A, const __half* __restrict__ B,
    __half* __restrict__ ResOut, float* __restrict__ rrms_out,
    int D, float eps
) {
    const int tid = threadIdx.x;
    const int BS = blockDim.x;
    float ss = 0.0f;
    for (int i = tid; i < D; i += BS) {
        float r = __half2float(A[i]) + __half2float(B[i]);
        ResOut[i] = __float2half(r);
        ss += r * r;
    }
    // Warp reduce within each warp
    ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
    ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
    // Cross-warp reduce via shared memory
    __shared__ float ws[8]; // max 8 warps for BS=256
    int wid = tid / 32, lid = tid % 32;
    if (lid == 0) ws[wid] = ss;
    __syncthreads();
    if (wid == 0 && lid < (BS / 32)) {
        ss = ws[lid];
        ss += __shfl_xor(ss, 4); ss += __shfl_xor(ss, 2);
        ss += __shfl_xor(ss, 1);
        if (lid == 0) *rrms_out = rsqrtf(ss / (float)D + eps);
    }
}

// Phase 1b: Compute rrms from X (no residual add). Single block.
__global__ void compute_rrms(
    const __half* __restrict__ X, float* __restrict__ rrms_out,
    int D, float eps
) {
    const int tid = threadIdx.x;
    const int BS = blockDim.x;
    float ss = 0.0f;
    for (int i = tid; i < D; i += BS) {
        float x = __half2float(X[i]);
        ss += x * x;
    }
    ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
    ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
    __shared__ float ws[8];
    int wid = tid / 32, lid = tid % 32;
    if (lid == 0) ws[wid] = ss;
    __syncthreads();
    if (wid == 0 && lid < (BS / 32)) {
        ss = ws[lid];
        ss += __shfl_xor(ss, 4); ss += __shfl_xor(ss, 2);
        ss += __shfl_xor(ss, 1);
        if (lid == 0) *rrms_out = rsqrtf(ss / (float)D + eps);
    }
}

// Phase 2: Apply norm + FWHT using precomputed rrms.
// Grid: (D/32), Block: (32) — one warp per 32-element block.
// Each warp reads only its 32 elements (no redundant D-wide reads).
__global__ void norm_fwht_apply(
    const __half* __restrict__ X, const __half* __restrict__ NormW,
    const float* __restrict__ rrms_ptr, __half* __restrict__ Y,
    int D
) {
    const int hb = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = hb * 32;
    float rrms = *rrms_ptr;

    float normed = __half2float(X[base + tid]) * rrms * __half2float(NormW[base + tid]);
    #pragma unroll
    for (int step = 1; step < 32; step *= 2) {
        float other = __shfl_xor(normed, step);
        if (tid & step) normed = other - normed;
        else normed = normed + other;
    }
    normed *= 0.17677669529663689f; // 1/sqrt(32)
    Y[base + tid] = __float2half(normed);
}

// Fused single-block: residual + rrms + norm + FWHT
// 1 block, up to 1024 threads. __syncthreads() barrier between phases.
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

// Fused single-block launcher: residual + rrms + norm + FWHT
// 1 launch instead of 2 — saves ~5μs launch overhead per norm op.
// No redundant reads (unlike parallel approach), no inter-kernel gap.
static inline void launch_res_norm_fwht(
    const __half* A, const __half* B, const __half* NormW,
    __half* Y, __half* ResOut, int D, float eps, hipStream_t st) {
    hipLaunchKernelGGL(fused_res_norm_fwht, dim3(1), dim3(1024), 0, st,
        A, B, NormW, Y, ResOut, D, eps);
}

// Fused single-block launcher: rrms + norm + FWHT (no residual add)
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
__global__ void flash_decode_reduce(
    const float* __restrict__ partial_max,   // [num_splits, H]
    const float* __restrict__ partial_sum,   // [num_splits, H]
    const float* __restrict__ partial_out,   // [num_splits, H, D]
    __half* __restrict__ O,                  // [H*D]
    int H, int D, int num_splits
) {
    const int qh = blockIdx.x;
    const int lane = threadIdx.x;

    // Find global max across splits
    float global_max = -1e30f;
    for (int sp = 0; sp < num_splits; sp++) {
        float m = partial_max[sp * H + qh];
        global_max = fmaxf(global_max, m);
    }
    // lane 0 has the correct value (all lanes read same address so all have it)

    // Combine with online softmax correction
    float global_sum = 0.0f;
    float out_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sp = 0; sp < num_splits; sp++) {
        float correction = expf(partial_max[sp * H + qh] - global_max);
        float sp_sum = partial_sum[sp * H + qh] * correction;
        global_sum += sp_sum;

        const long out_offset = (long)sp * H * D + (long)qh * D;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            out_reg[i] += partial_out[out_offset + lane + i * 32] * correction;
    }

    float inv = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    __half* o_row = O + qh * D;
    // Apply FWHT on output — fused, eliminates separate hadamard_rotate launch
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

// Fused single-split attention (for small seq_len <= FLASH_DECODE_BLOCK_S)
// Same as legacy but with GQA grouping for better occupancy
// Grid: (num_kv_heads), Block: (32, gqa_ratio)
// flash_decode_fused: 1 block per Q head (not per KV head)
// Grid: (num_heads), Block: (32)
// 40 blocks on 48 CUs = 83% CU utilization (was 8 blocks = 17%)
__global__ void flash_decode_fused(
    const __half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache,  // FP8 E4M3
    const uint8_t* __restrict__ V_cache,  // FP8 E4M3
    __half* __restrict__ O,
    int H, int Hk, int max_seq, int seq_len, int D
) {
    const int qh = blockIdx.x;
    const int kh = qh / (H / Hk);
    const int lane = threadIdx.x;
    const float scale = rsqrtf((float)D);

    const __half* q_row = Q + qh * D;
    float q_reg[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        q_reg[i] = __half2float(q_row[lane + i * 32]);

    // Pack Q as FP8 E4M3 for hardware dot4
    uint32_t q_packed_fp8 = (uint32_t)f32_to_fp8e4m3(q_reg[0])
                          | ((uint32_t)f32_to_fp8e4m3(q_reg[1]) << 8)
                          | ((uint32_t)f32_to_fp8e4m3(q_reg[2]) << 16)
                          | ((uint32_t)f32_to_fp8e4m3(q_reg[3]) << 24);

    const uint8_t* k_base = K_cache + (long)kh * max_seq * D;
    const uint8_t* v_base = V_cache + (long)kh * max_seq * D;

    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float out_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = 0; s < seq_len; s++) {
        const uint8_t* k_pos = k_base + (long)s * D + lane;
        uint32_t k_packed = (uint32_t)k_pos[0]
                          | ((uint32_t)k_pos[32] << 8)
                          | ((uint32_t)k_pos[64] << 16)
                          | ((uint32_t)k_pos[96] << 24);
        float dot = __builtin_amdgcn_dot4_f32_fp8_fp8(q_packed_fp8, k_packed, 0.0f);

        dot += __shfl_xor(dot, 16);
        dot += __shfl_xor(dot, 8);
        dot += __shfl_xor(dot, 4);
        dot += __shfl_xor(dot, 2);
        dot += __shfl_xor(dot, 1);
        dot *= scale;

        float new_max = fmaxf(max_score, dot);
        float correction = expf(max_score - new_max);
        float exp_score = expf(dot - new_max);
        sum_exp = sum_exp * correction + exp_score;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            out_reg[i] = out_reg[i] * correction +
                         exp_score * fp8e4m3_to_f32(v_base[(long)s * D + lane + i * 32]);
        max_score = new_max;
    }

    float inv = 1.0f / sum_exp;
    __half* o_row = O + qh * D;
    // Apply FWHT on output — fused, eliminates separate hadamard_rotate launch
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

// Dispatch flash_decode_partial with optimal BLOCK_S based on context length.
// Larger BLOCK_S = fewer splits = fewer blocks = less scheduler overhead.
// Caps splits at ~16-32 max regardless of context length.
static inline void launch_flash_decode(
    const __half* Q, const uint8_t* K_cache, const uint8_t* V_cache,
    __half* O,
    float* p_max, float* p_sum, float* p_out,
    int num_heads, int num_kv_heads, int max_seq, int seq_len, int head_dim,
    hipStream_t stream)
{
    int gqa_ratio = num_heads / num_kv_heads;

    // Fused single-pass for very small contexts (saves 1 kernel launch)
    if (seq_len <= 32) {
        hipLaunchKernelGGL(flash_decode_fused,
            dim3(num_heads), dim3(32), 0, stream,
            Q, K_cache, V_cache, O,
            num_heads, num_kv_heads, max_seq, seq_len, head_dim);
        return;
    }

    // Split-K with adaptive BLOCK_S:
    // - Small ctx (≤256): BLOCK_S=32 → more splits → better CU utilization
    // - Medium/large ctx (>256): BLOCK_S=64 → fewer splits → less reduce overhead
    int block_s, n_splits;
    if (seq_len <= 256) {
        block_s = 32;
        n_splits = (seq_len + 31) / 32;
    } else {
        block_s = 64;
        n_splits = (seq_len + 63) / 64;
    }

    #define LAUNCH_PARTIAL(BS) \
        hipLaunchKernelGGL((flash_decode_partial<BS>), \
            dim3(num_kv_heads, n_splits), dim3(32, gqa_ratio), 0, stream, \
            Q, K_cache, V_cache, p_max, p_sum, p_out, \
            num_heads, num_kv_heads, max_seq, seq_len, head_dim, n_splits)

    switch (block_s) {
        case 32:  LAUNCH_PARTIAL(32);  break;
        case 64:  LAUNCH_PARTIAL(64);  break;
    }
    #undef LAUNCH_PARTIAL

    hipLaunchKernelGGL(flash_decode_reduce,
        dim3(num_heads), dim3(32), 0, stream,
        p_max, p_sum, p_out, O,
        num_heads, head_dim, n_splits);
}

// KV cache write (FP16 → FP8 E4M3)
__global__ void kv_write(const __half* src, uint8_t* cache, int Hk, int max_seq, int D, int pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Hk * D) {
        int h = idx / D, d = idx % D;
        cache[h * max_seq * D + pos * D + d] = f32_to_fp8e4m3(__half2float(src[idx]));
    }
}

// Fused KV cache write — K and V in one kernel launch (FP16 → FP8 E4M3)
__global__ void kv_write_fused(const __half* K, const __half* V,
                                uint8_t* k_cache, uint8_t* v_cache,
                                int Hk, int max_seq, int D, int pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Hk * D) {
        int h = idx / D, d = idx % D;
        int off = h * max_seq * D + pos * D + d;
        k_cache[off] = f32_to_fp8e4m3(__half2float(K[idx]));
        v_cache[off] = f32_to_fp8e4m3(__half2float(V[idx]));
    }
}

// Fused head RMSNorm + RoPE + KV cache write (saves 2 kernel launches per layer)
// Blocks 0..H-1: Q heads (RMSNorm + RoPE)
// Blocks H..H+Hk-1: K heads (RMSNorm + RoPE + write to K cache + copy V to V cache)
__global__ void head_rmsnorm_rope_kv(
    const __half* Q, const __half* K, const __half* V,
    const __half* QW, const __half* KW,
    __half* Q_out, __half* K_out,
    uint8_t* K_cache, uint8_t* V_cache,  // FP8 E4M3 KV cache
    const __half* COS, const __half* SIN,
    int H, int Hk, int D, float eps, int max_seq, int pos
) {
    int pid = blockIdx.x;
    int tid = threadIdx.x;
    int half_D = D / 2;

    // Process Q heads
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
            dst[i] = __float2half(ne * c - no * s);
            dst[i + half_D] = __float2half(no * c + ne * s);
        }
    }
    // Process K heads + KV cache write (FP8 E4M3)
    int kid = pid - H;
    if (kid >= 0 && kid < Hk) {
        const __half* src = K + kid * D;
        __half* dst = K_out + kid * D;
        int cache_off = kid * max_seq * D + pos * D;

        // K: RMSNorm + RoPE + FP8 cache write
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
            dst[i] = __float2half(k_lo);
            dst[i + half_D] = __float2half(k_hi);
            K_cache[cache_off + i] = f32_to_fp8e4m3(k_lo);
            K_cache[cache_off + i + half_D] = f32_to_fp8e4m3(k_hi);
        }

        // V: convert FP16 → FP8 and write to cache
        for (int i = tid; i < D; i += 32) {
            V_cache[cache_off + i] = f32_to_fp8e4m3(__half2float(V[kid * D + i]));
        }
    }
}

// ============================================================
// Graph-compatible head_rmsnorm_rope_kv: reads pos from g_graph_pos
// COS_base/SIN_base are full RoPE tables; kernel computes offset.
// ============================================================
__global__ void head_rmsnorm_rope_kv_g(
    const __half* Q, const __half* K, const __half* V,
    const __half* QW, const __half* KW,
    __half* Q_out, __half* K_out,
    uint8_t* K_cache, uint8_t* V_cache,
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
            dst[i] = __float2half(ne * c - no * s);
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
            dst[i] = __float2half(k_lo);
            dst[i + half_D] = __float2half(k_hi);
            K_cache[cache_off + i] = f32_to_fp8e4m3(k_lo);
            K_cache[cache_off + i + half_D] = f32_to_fp8e4m3(k_hi);
        }

        for (int i = tid; i < D; i += 32) {
            V_cache[cache_off + i] = f32_to_fp8e4m3(__half2float(V[kid * D + i]));
        }
    }
}

// ============================================================
// Graph-compatible flash_decode_partial: reads seq_len from g_graph_seq_len.
// Launched with max_n_splits; excess blocks write neutral values.
// ============================================================
template <int BLOCK_S>
__global__ void flash_decode_partial_g(
    const __half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache,
    const uint8_t* __restrict__ V_cache,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_out,
    int H, int Hk, int max_seq, int D,
    int max_n_splits
) {
    int seq_len = g_graph_seq_len;
    const int kh = blockIdx.x;
    const int split_id = blockIdx.y;
    const int warp_id = threadIdx.y;
    const int lane = threadIdx.x;
    const int gqa_ratio = H / Hk;

    if (warp_id >= gqa_ratio) return;
    const int qh = kh * gqa_ratio + warp_id;

    // Excess split — write neutral values for reduce
    int n_splits_actual = (seq_len + BLOCK_S - 1) / BLOCK_S;
    if (split_id >= n_splits_actual) {
        if (lane == 0) {
            partial_max[split_id * H + qh] = -1e30f;
            partial_sum[split_id * H + qh] = 0.0f;
        }
        const long out_offset = (long)split_id * H * D + (long)qh * D;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            partial_out[out_offset + lane + i * 32] = 0.0f;
        return;
    }

    const float scale = rsqrtf((float)D);
    const __half* q_row = Q + qh * D;
    float q_reg[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        q_reg[i] = __half2float(q_row[lane + i * 32]);

    uint32_t q_packed_fp8 = (uint32_t)f32_to_fp8e4m3(q_reg[0])
                          | ((uint32_t)f32_to_fp8e4m3(q_reg[1]) << 8)
                          | ((uint32_t)f32_to_fp8e4m3(q_reg[2]) << 16)
                          | ((uint32_t)f32_to_fp8e4m3(q_reg[3]) << 24);

    const int s_start = split_id * BLOCK_S;
    const int s_end = min(s_start + BLOCK_S, seq_len);

    const uint8_t* k_base = K_cache + (long)kh * max_seq * D;
    const uint8_t* v_base = V_cache + (long)kh * max_seq * D;

    float local_max = -1e30f;
    float local_sum = 0.0f;
    float out_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = s_start; s < s_end; s++) {
        const uint8_t* k_pos = k_base + (long)s * D + lane;
        uint32_t k_packed = (uint32_t)k_pos[0]
                          | ((uint32_t)k_pos[32] << 8)
                          | ((uint32_t)k_pos[64] << 16)
                          | ((uint32_t)k_pos[96] << 24);
        float dot = __builtin_amdgcn_dot4_f32_fp8_fp8(q_packed_fp8, k_packed, 0.0f);

        dot += __shfl_xor(dot, 16);
        dot += __shfl_xor(dot, 8);
        dot += __shfl_xor(dot, 4);
        dot += __shfl_xor(dot, 2);
        dot += __shfl_xor(dot, 1);
        dot *= scale;

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

    const long out_offset = (long)split_id * H * D + (long)qh * D;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        partial_out[out_offset + lane + i * 32] = out_reg[i];

    if (lane == 0) {
        partial_max[split_id * H + qh] = local_max;
        partial_sum[split_id * H + qh] = local_sum;
    }
}

// Graph-compatible flash decode launcher (fixed grid, device-side seq_len)
static inline void launch_flash_decode_g(
    const __half* Q, const uint8_t* K_cache, const uint8_t* V_cache,
    __half* O,
    float* p_max, float* p_sum, float* p_out,
    int num_heads, int num_kv_heads, int max_seq,
    int max_n_splits, int block_s, int head_dim,
    hipStream_t stream)
{
    int gqa_ratio = num_heads / num_kv_heads;

    if (block_s == 64) {
        hipLaunchKernelGGL((flash_decode_partial_g<64>),
            dim3(num_kv_heads, max_n_splits), dim3(32, gqa_ratio), 0, stream,
            Q, K_cache, V_cache, p_max, p_sum, p_out,
            num_heads, num_kv_heads, max_seq, head_dim, max_n_splits);
    } else {
        hipLaunchKernelGGL((flash_decode_partial_g<32>),
            dim3(num_kv_heads, max_n_splits), dim3(32, gqa_ratio), 0, stream,
            Q, K_cache, V_cache, p_max, p_sum, p_out,
            num_heads, num_kv_heads, max_seq, head_dim, max_n_splits);
    }

    hipLaunchKernelGGL(flash_decode_reduce,
        dim3(num_heads), dim3(32), 0, stream,
        p_max, p_sum, p_out, O,
        num_heads, head_dim, max_n_splits);
}

// ============================================================
// GEMM kernel — batched GEMV for M>1 (prefill)
// Grid: (ceil(N/BLOCK_N), M), Block: (32,) = 1 warp per (m, n_block)
// Each warp computes BLOCK_N output elements for one row of X.
// ============================================================
template<int BLOCK_N>
__global__ void gemm_warp(
    const __half* __restrict__ X,      // [M, K]
    const uint8_t* __restrict__ W,     // [N, K/2]
    const __half* __restrict__ S,      // [N, K/32, 2] interleaved (scale, zero_fp16)
    __half* __restrict__ Y,            // [M, N]
    int M, int N, int K
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int m = blockIdx.y;
    if (m >= M) return;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / 32, nit = K / 32;
    float acc[BLOCK_N];
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i++) acc[i] = 0.0f;

    const __half* X_row = X + (int64_t)m * K;

    for (int it = lane; it < nit; it += 32) {
        float xv[32];
        #pragma unroll
        for (int i = 0; i < 32; i += 2) {
            __half2 xh = *reinterpret_cast<const __half2*>(&X_row[it * 32 + i]);
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
            float dot = 0.0f;
            #pragma unroll
            for (int b = 0; b < 16; b++) {
                uint8_t p = wr[b];
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
        if (lane == 0 && n0 + ni < N) Y[(int64_t)m * N + n0 + ni] = __float2half(acc[ni]);
    }
}

// GEMM launch helper (custom kernel, for small M or fallback)
static inline void launch_gemm(
    const __half* X, const uint8_t* W, const __half* S,
    __half* Y, int M, int N, int K, hipStream_t st) {
    if (N <= 2048) {
        hipLaunchKernelGGL((gemm_warp<4>), dim3((N+3)/4, M), dim3(32), 0, st, X, W, S, Y, M, N, K);
    } else if (N <= 16384) {
        hipLaunchKernelGGL((gemm_warp<2>), dim3((N+1)/2, M), dim3(32), 0, st, X, W, S, Y, M, N, K);
    } else {
        hipLaunchKernelGGL((gemm_warp<2>), dim3((N+1)/2, M), dim3(32), 0, st, X, W, S, Y, M, N, K);
    }
}

// Dequant INT4 weights + rocBLAS GEMM via at::matmul (much faster for prefill)
// Returns result tensor [M, N] = X @ dequant(W).T
static inline torch::Tensor dequant_matmul(
    torch::Tensor X_t,  // [M, K] FP16
    const uint8_t* W, const __half* S,
    int N, int K, torch::Device device, hipStream_t stream
) {
    auto opts = torch::dtype(torch::kHalf).device(device);
    auto w_fp16 = torch::empty({(long)N, (long)K}, opts);
    int total = N * (K / 2);
    int threads = 256, blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(dequant_int4_kernel, dim3(blocks), dim3(threads), 0, stream,
        W, S, reinterpret_cast<__half*>(w_fp16.data_ptr()), N, K);
    // rocBLAS GEMM: [M, K] @ [N, K].T = [M, N]
    auto result = at::matmul(X_t, w_fp16.t());
    return result;
}

// Dequant + rocBLAS GEMM, writing result directly to output tensor (avoids copy)
// w_tmp: pre-allocated buffer [max_N, max_K] — reused across calls to avoid allocs
static inline void dequant_matmul_out(
    torch::Tensor X_t,    // [M, K] FP16 input
    torch::Tensor out_t,  // [M, N] FP16 output (pre-allocated)
    const uint8_t* W, const __half* S,
    int N, int K,
    torch::Tensor w_tmp,  // pre-allocated dequant buffer (sliced to [N, K])
    hipStream_t stream
) {
    auto w_fp16 = w_tmp.narrow(0, 0, N).narrow(1, 0, K);  // view into pre-alloc buffer
    int total = N * (K / 2);
    int threads = 256, blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(dequant_int4_kernel, dim3(blocks), dim3(threads), 0, stream,
        W, S, reinterpret_cast<__half*>(w_tmp.data_ptr()), N, K);
    // rocBLAS GEMM: out = X @ W.T
    at::mm_out(out_t, X_t, w_fp16.t());
}

// Batched RMSNorm: one block per row of X
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

// ============================================================
// Prefill step — all layers in C++, returns logits [M, vocab]
// ============================================================
torch::Tensor int4_prefill_logits(
    torch::Tensor input_ids,              // [M] long — token IDs
    torch::Tensor embed_weight,           // [vocab, D] FP16
    torch::Tensor final_norm_w,           // [D] FP16
    torch::Tensor lm_w, torch::Tensor lm_s,
    int64_t lm_N, int64_t lm_K,
    // Per-layer weights
    std::vector<torch::Tensor> qkv_w, std::vector<torch::Tensor> qkv_s,
    std::vector<int64_t> qkv_N,
    std::vector<torch::Tensor> o_w, std::vector<torch::Tensor> o_s,
    std::vector<int64_t> o_N,
    std::vector<torch::Tensor> gu_w, std::vector<torch::Tensor> gu_s,
    std::vector<int64_t> gu_N,
    std::vector<torch::Tensor> down_w, std::vector<torch::Tensor> down_s,
    std::vector<int64_t> down_N,
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w, std::vector<torch::Tensor> k_norm_w,
    torch::Tensor rope_cos, torch::Tensor rope_sin,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    float rms_eps,
    torch::Tensor hadamard_mat  // [32, 32] FP16 normalized Hadamard matrix
) {
    c10::InferenceMode guard;

    int num_layers = qkv_w.size();
    int M = input_ids.size(0);
    int D = hidden_size;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int qkv_dim = q_dim + 2 * kv_dim;
    int inter = intermediate_size;
    auto device = input_ids.device();
    auto stream = at::hip::getCurrentHIPStream();
    auto opts = torch::dtype(torch::kHalf).device(device);

    int norm_bs = 256;
    size_t smem_norm = D * sizeof(float);
    int half_D = head_dim / 2;

    auto* had_p = reinterpret_cast<const __half*>(hadamard_mat.data_ptr());

    // Allocate buffers [M, ...]
    auto x = torch::embedding(embed_weight, input_ids);  // [M, D]
    auto normed = torch::empty({M, D}, opts);
    auto residual = torch::empty({M, D}, opts);
    auto rot_buf = torch::empty({M, D}, opts);         // Hadamard rotation output (D-width)
    auto rot_inter_buf = torch::empty({M, (long)inter}, opts);  // Hadamard rotation output (inter-width)
    auto qkv_buf = torch::empty({M, qkv_dim}, opts);
    auto o_buf = torch::empty({M, D}, opts);
    auto gu_buf = torch::empty({M, 2 * inter}, opts);
    auto silu_buf = torch::empty({M, inter}, opts);  // contiguous SiLU output for down_proj input
    auto down_buf = torch::empty({M, D}, opts);

    // Pre-allocate dequant weight buffer (largest = gate_up [Ngu, D] or down [Nd, inter])
    int max_N = std::max({(int)qkv_N[0], (int)o_N[0], (int)gu_N[0], (int)down_N[0], (int)lm_N});
    int max_K = std::max(D, inter);
    auto w_tmp = torch::empty({(long)max_N, (long)max_K}, opts);

    auto* x_p = reinterpret_cast<__half*>(x.data_ptr());
    auto* n_p = reinterpret_cast<__half*>(normed.data_ptr());
    auto* r_p = reinterpret_cast<__half*>(residual.data_ptr());
    auto* rot_p = reinterpret_cast<__half*>(rot_buf.data_ptr());
    auto* rot_inter_p = reinterpret_cast<__half*>(rot_inter_buf.data_ptr());
    auto* qkv_p = reinterpret_cast<__half*>(qkv_buf.data_ptr());
    auto* o_p = reinterpret_cast<__half*>(o_buf.data_ptr());
    auto* gu_p = reinterpret_cast<__half*>(gu_buf.data_ptr());
    auto* silu_p = reinterpret_cast<__half*>(silu_buf.data_ptr());
    auto* d_p = reinterpret_cast<__half*>(down_buf.data_ptr());

    // Pre-extract pointers
    std::vector<const uint8_t*> qkv_wp(num_layers);
    std::vector<const __half*> qkv_sp(num_layers);
    std::vector<const uint8_t*> o_wp(num_layers);
    std::vector<const __half*> o_sp(num_layers);
    std::vector<const uint8_t*> gu_wp(num_layers);
    std::vector<const __half*> gu_sp(num_layers);
    std::vector<const uint8_t*> down_wp(num_layers);
    std::vector<const __half*> down_sp(num_layers);
    std::vector<const __half*> in_np(num_layers), post_np(num_layers);
    std::vector<const __half*> q_np(num_layers), k_np(num_layers);
    for (int li = 0; li < num_layers; li++) {
        qkv_wp[li] = qkv_w[li].data_ptr<uint8_t>();
        qkv_sp[li] = reinterpret_cast<const __half*>(qkv_s[li].data_ptr());
        o_wp[li] = o_w[li].data_ptr<uint8_t>();
        o_sp[li] = reinterpret_cast<const __half*>(o_s[li].data_ptr());
        gu_wp[li] = gu_w[li].data_ptr<uint8_t>();
        gu_sp[li] = reinterpret_cast<const __half*>(gu_s[li].data_ptr());
        down_wp[li] = down_w[li].data_ptr<uint8_t>();
        down_sp[li] = reinterpret_cast<const __half*>(down_s[li].data_ptr());
        in_np[li] = reinterpret_cast<const __half*>(in_norm_w[li].data_ptr());
        post_np[li] = reinterpret_cast<const __half*>(post_norm_w[li].data_ptr());
        q_np[li] = reinterpret_cast<const __half*>(q_norm_w[li].data_ptr());
        k_np[li] = reinterpret_cast<const __half*>(k_norm_w[li].data_ptr());
    }

    const __half* cos_p = reinterpret_cast<const __half*>(rope_cos.data_ptr());
    const __half* sin_p = reinterpret_cast<const __half*>(rope_sin.data_ptr());

    // Layer loop — uses dequant + rocBLAS GEMM for maximum prefill throughput
    for (int li = 0; li < num_layers; li++) {
        int Nqkv = qkv_N[li], Kqkv = D;
        int No = o_N[li], Ko = D;
        int Ngu = gu_N[li], Kgu = D;
        int Nd = down_N[li], Kd = inter;

        // 1. RMSNorm (batched)
        if (li == 0) {
            hipLaunchKernelGGL(rmsnorm_batched, dim3(M), dim3(norm_bs), smem_norm, stream,
                x_p, in_np[0], n_p, D, rms_eps);
        }

        // 2. Hadamard rotate normed → rot_buf, then QKV via dequant + rocBLAS
        launch_hadamard_batched(n_p, rot_p, had_p, M, Kqkv, stream);
        {
            auto rot_t = torch::from_blob(rot_p, {(long)M, (long)Kqkv}, opts);
            dequant_matmul_out(rot_t, qkv_buf, qkv_wp[li], qkv_sp[li], Nqkv, Kqkv, w_tmp, stream);
        }

        // 3. Head RMSNorm + RoPE (batched, in-place on qkv_buf)
        hipLaunchKernelGGL(head_rmsnorm_rope_prefill,
            dim3(num_heads + num_kv_heads, M), dim3(32), 0, stream,
            qkv_p, q_np[li], k_np[li],
            cos_p, sin_p,
            M, num_heads, num_kv_heads, head_dim, rms_eps, half_D, qkv_dim);

        // 4. Causal self-attention via PyTorch SDPA
        {
            auto q_t = qkv_buf.narrow(1, 0, q_dim).view({M, (long)num_heads, (long)head_dim})
                            .permute({1, 0, 2}).unsqueeze(0);
            auto k_t = qkv_buf.narrow(1, q_dim, kv_dim).view({M, (long)num_kv_heads, (long)head_dim})
                            .permute({1, 0, 2}).unsqueeze(0);
            auto v_t = qkv_buf.narrow(1, q_dim + kv_dim, kv_dim).view({M, (long)num_kv_heads, (long)head_dim})
                            .permute({1, 0, 2}).unsqueeze(0);

            auto attn = at::scaled_dot_product_attention(q_t, k_t, v_t,
                /*attn_mask=*/c10::nullopt, /*dropout_p=*/0.0,
                /*is_causal=*/true, /*scale=*/c10::nullopt, /*enable_gqa=*/true);
            auto attn_out = attn.squeeze(0).permute({1, 0, 2}).reshape({M, q_dim});

            // 5. Hadamard rotate attn_out → rot_buf, then O projection via dequant + rocBLAS
            launch_hadamard_batched(reinterpret_cast<const __half*>(attn_out.data_ptr()),
                rot_p, had_p, M, Ko, stream);
            auto rot_t = torch::from_blob(rot_p, {(long)M, (long)Ko}, opts);
            dequant_matmul_out(rot_t, o_buf, o_wp[li], o_sp[li], No, Ko, w_tmp, stream);
        }

        // 6. Fused residual + post_norm (batched)
        hipLaunchKernelGGL(fused_res_norm_batched, dim3(M), dim3(norm_bs), smem_norm, stream,
            o_p, x_p, post_np[li], n_p, r_p, D, rms_eps);

        // 7. Hadamard rotate normed → rot_buf, then Gate+Up via dequant + rocBLAS
        launch_hadamard_batched(n_p, rot_p, had_p, M, Kgu, stream);
        {
            auto rot_t = torch::from_blob(rot_p, {(long)M, (long)Kgu}, opts);
            dequant_matmul_out(rot_t, gu_buf, gu_wp[li], gu_sp[li], Ngu, Kgu, w_tmp, stream);
        }

        // 8. SiLU(gate) * up → contiguous silu_buf [M, inter]
        hipLaunchKernelGGL(silu_mul_batched, dim3((inter+255)/256, M), dim3(256), 0, stream,
            gu_p, gu_p + inter, silu_p, inter, 2 * inter, inter);

        // 9. Hadamard rotate silu_buf → rot_inter_buf, then Down via dequant + rocBLAS
        launch_hadamard_batched(silu_p, rot_inter_p, had_p, M, Kd, stream);
        {
            auto rot_t = torch::from_blob(rot_inter_p, {(long)M, (long)Kd}, opts);
            dequant_matmul_out(rot_t, down_buf, down_wp[li], down_sp[li], Nd, Kd, w_tmp, stream);
        }

        // 10. Fused residual + next input_norm (batched)
        if (li < num_layers - 1) {
            hipLaunchKernelGGL(fused_res_norm_batched, dim3(M), dim3(norm_bs), smem_norm, stream,
                d_p, r_p, in_np[li + 1], n_p, x_p, D, rms_eps);
        } else {
            hipLaunchKernelGGL(fused_res_norm_batched, dim3(M), dim3(norm_bs), smem_norm, stream,
                d_p, r_p, in_np[0], n_p, x_p, D, rms_eps);
        }
    }

    // Final RMSNorm
    auto* fn_p = reinterpret_cast<const __half*>(final_norm_w.data_ptr());
    hipLaunchKernelGGL(rmsnorm_batched, dim3(M), dim3(norm_bs), smem_norm, stream,
        x_p, fn_p, n_p, D, rms_eps);

    // Hadamard rotate normed → rot_buf, then LM head via dequant + rocBLAS
    launch_hadamard_batched(n_p, rot_p, had_p, M, (int)lm_K, stream);
    auto rot_t = torch::from_blob(rot_p, {(long)M, (long)lm_K}, opts);
    auto logits = torch::empty({(long)M, (long)lm_N}, opts);
    dequant_matmul_out(rot_t, logits,
        lm_w.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(lm_s.data_ptr()),
        lm_N, lm_K, w_tmp, stream);

    return logits;  // [M, vocab]
}

// ============================================================
// Decode step context
// ============================================================
struct DSContext {
    torch::Tensor normed, hidden, residual;
    torch::Tensor rotated;          // [D] for Hadamard rotation output
    torch::Tensor qkv_buf, o_buf, attn_buf;
    torch::Tensor gu_buf, silu_buf, down_buf;
    torch::Tensor rrms_buf;         // [1] float32 — scratch for 2-phase norm
    // Flash decode attention partial buffers
    torch::Tensor fa_partial_max;   // [max_splits, H] float32
    torch::Tensor fa_partial_sum;   // [max_splits, H] float32
    torch::Tensor fa_partial_out;   // [max_splits, H, D] float32
    int fa_max_splits = 0;
    bool init = false;
};
static DSContext g_ds;

// ============================================================
// Single decode step — all 40 layers in C++
// ============================================================
torch::Tensor int4_decode_step(
    torch::Tensor hidden_in,          // [1, 1, D] current hidden state
    // Per-layer weights (vectors of tensors, pre-transposed [N, K/2] etc.)
    std::vector<torch::Tensor> qkv_w, std::vector<torch::Tensor> qkv_s,
    std::vector<int64_t> qkv_N,
    std::vector<torch::Tensor> o_w, std::vector<torch::Tensor> o_s,
    std::vector<int64_t> o_N,
    std::vector<torch::Tensor> gu_w, std::vector<torch::Tensor> gu_s,
    std::vector<int64_t> gu_N,
    std::vector<torch::Tensor> down_w, std::vector<torch::Tensor> down_s,
    std::vector<int64_t> down_N,
    // Norm weights
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w, std::vector<torch::Tensor> k_norm_w,
    // RoPE
    torch::Tensor rope_cos, torch::Tensor rope_sin,  // [max_seq, D/2]
    int64_t pos,
    // Model config
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    float rms_eps,
    // KV caches: [num_kv_heads, max_seq, head_dim]
    std::vector<torch::Tensor> k_caches, std::vector<torch::Tensor> v_caches,
    torch::Tensor hadamard_mat  // [32, 32] FP16 normalized Hadamard matrix
) {
    c10::InferenceMode guard;

    int num_layers = qkv_w.size();
    int D = hidden_size;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int max_seq = k_caches[0].size(2);  // [1, Hk, max_seq, D]
    int seq_len = pos + 1;
    auto device = hidden_in.device();
    auto stream = at::hip::getCurrentHIPStream();
    auto* had_p = reinterpret_cast<const __half*>(hadamard_mat.data_ptr());

    // Initialize scratch buffers
    int fa_max_splits = (max_seq + FLASH_DECODE_BLOCK_S - 1) / FLASH_DECODE_BLOCK_S;
    if (!g_ds.init) {
        auto opts = torch::dtype(torch::kHalf).device(device);
        g_ds.normed = torch::empty({D}, opts);
        g_ds.hidden = torch::empty({D}, opts);
        g_ds.residual = torch::empty({D}, opts);
        int max_qkv = q_dim + 2 * kv_dim;
        g_ds.qkv_buf = torch::empty({max_qkv}, opts);
        g_ds.o_buf = torch::empty({D}, opts);
        g_ds.attn_buf = torch::empty({D}, opts);
        int max_gu = 2 * intermediate_size;
        g_ds.gu_buf = torch::empty({max_gu}, opts);
        g_ds.silu_buf = torch::empty({intermediate_size}, opts);
        g_ds.down_buf = torch::empty({D}, opts);
        g_ds.rotated = torch::empty({D}, opts);  // for Hadamard rotation output
        // Flash decode attention partial buffers
        auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
        g_ds.fa_partial_max = torch::empty({fa_max_splits, (long)num_heads}, opts_f32);
        g_ds.fa_partial_sum = torch::empty({fa_max_splits, (long)num_heads}, opts_f32);
        g_ds.fa_partial_out = torch::empty({fa_max_splits, (long)num_heads, (long)head_dim}, opts_f32);
        g_ds.fa_max_splits = fa_max_splits;
        g_ds.rrms_buf = torch::empty({1}, opts_f32);
        g_ds.init = true;
    }
    // Grow flash attention buffers if needed (larger KV cache than initial)
    if (fa_max_splits > g_ds.fa_max_splits) {
        auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
        g_ds.fa_partial_max = torch::empty({fa_max_splits, (long)num_heads}, opts_f32);
        g_ds.fa_partial_sum = torch::empty({fa_max_splits, (long)num_heads}, opts_f32);
        g_ds.fa_partial_out = torch::empty({fa_max_splits, (long)num_heads, (long)head_dim}, opts_f32);
        g_ds.fa_max_splits = fa_max_splits;
    }

    auto h = reinterpret_cast<__half*>(g_ds.hidden.data_ptr());
    auto n = reinterpret_cast<__half*>(g_ds.normed.data_ptr());
    auto r = reinterpret_cast<__half*>(g_ds.residual.data_ptr());
    auto rot = reinterpret_cast<__half*>(g_ds.rotated.data_ptr());
    auto qkv = reinterpret_cast<__half*>(g_ds.qkv_buf.data_ptr());
    auto obuf = reinterpret_cast<__half*>(g_ds.o_buf.data_ptr());
    auto abuf = reinterpret_cast<__half*>(g_ds.attn_buf.data_ptr());
    auto gubuf = reinterpret_cast<__half*>(g_ds.gu_buf.data_ptr());
    auto silubuf = reinterpret_cast<__half*>(g_ds.silu_buf.data_ptr());
    auto dbuf = reinterpret_cast<__half*>(g_ds.down_buf.data_ptr());

    int norm_bs = 256;
    size_t smem_norm = D * sizeof(float);
    float sdpa_scale = 1.0f / sqrtf((float)head_dim);
    int max_heads = std::max((int)num_heads, (int)num_kv_heads);
    int half_D = head_dim / 2;

    const __half* cos_pos = reinterpret_cast<const __half*>(rope_cos.data_ptr()) + pos * (head_dim / 2);
    const __half* sin_pos = reinterpret_cast<const __half*>(rope_sin.data_ptr()) + pos * (head_dim / 2);

    // Copy input hidden state
    hipMemcpyAsync(h, hidden_in.data_ptr(), D * sizeof(__half),
                   hipMemcpyDeviceToDevice, stream);

    // First layer: RMSNorm + FWHT → rot (multi-block parallel, no H matrix needed)
    launch_rmsnorm_fwht(h, reinterpret_cast<const __half*>(in_norm_w[0].data_ptr()),
        rot, D, rms_eps, stream);

    // Pre-extract all raw pointers to avoid per-layer PyTorch API overhead
    std::vector<const uint8_t*> qkv_wp(num_layers);
    std::vector<const __half*> qkv_sp(num_layers);
    std::vector<const uint8_t*> o_wp(num_layers);
    std::vector<const __half*> o_sp(num_layers);
    std::vector<const uint8_t*> gu_wp(num_layers);
    std::vector<const __half*> gu_sp(num_layers);
    std::vector<const uint8_t*> down_wp(num_layers);
    std::vector<const __half*> down_sp(num_layers);
    std::vector<const __half*> in_np(num_layers), post_np(num_layers);
    std::vector<const __half*> q_np(num_layers), k_np(num_layers);
    std::vector<uint8_t*> kc_p(num_layers), vc_p(num_layers);
    for (int li = 0; li < num_layers; li++) {
        qkv_wp[li] = qkv_w[li].data_ptr<uint8_t>();
        qkv_sp[li] = reinterpret_cast<const __half*>(qkv_s[li].data_ptr());
        o_wp[li] = o_w[li].data_ptr<uint8_t>();
        o_sp[li] = reinterpret_cast<const __half*>(o_s[li].data_ptr());
        gu_wp[li] = gu_w[li].data_ptr<uint8_t>();
        gu_sp[li] = reinterpret_cast<const __half*>(gu_s[li].data_ptr());
        down_wp[li] = down_w[li].data_ptr<uint8_t>();
        down_sp[li] = reinterpret_cast<const __half*>(down_s[li].data_ptr());
        in_np[li] = reinterpret_cast<const __half*>(in_norm_w[li].data_ptr());
        post_np[li] = reinterpret_cast<const __half*>(post_norm_w[li].data_ptr());
        q_np[li] = reinterpret_cast<const __half*>(q_norm_w[li].data_ptr());
        k_np[li] = reinterpret_cast<const __half*>(k_norm_w[li].data_ptr());
        kc_p[li] = k_caches[li].data_ptr<uint8_t>();
        vc_p[li] = v_caches[li].data_ptr<uint8_t>();
    }

    int kv_total = num_kv_heads * head_dim;
    int inter = intermediate_size;

    for (int li = 0; li < num_layers; li++) {
        int Nqkv = qkv_N[li], Kqkv = D;
        int No = o_N[li], Ko = D;
        int Ngu = gu_N[li], Kgu = D;
        int Nd = down_N[li], Kd = intermediate_size;

        // 1. QKV GEMV (rot already has FWHT-rotated normed input)
        launch_gemv(rot, qkv_wp[li], qkv_sp[li], qkv, Nqkv, Kqkv, stream);

        // 2-4. Fused head RMSNorm + RoPE + KV cache write
        __half* q_ptr = qkv;
        __half* k_ptr = qkv + q_dim;
        __half* v_ptr = qkv + q_dim + kv_dim;
        hipLaunchKernelGGL(head_rmsnorm_rope_kv, dim3(num_heads + num_kv_heads), dim3(32), 0, stream,
            q_ptr, k_ptr, v_ptr, q_np[li], k_np[li],
            q_ptr, k_ptr, kc_p[li], vc_p[li],
            cos_pos, sin_pos,
            num_heads, num_kv_heads, head_dim, rms_eps, max_seq, pos);

        // 5. Flash decode attention + Hadamard + O projection
        {
            int cur_seq_len = pos + 1;
            auto* p_max = reinterpret_cast<float*>(g_ds.fa_partial_max.data_ptr());
            auto* p_sum = reinterpret_cast<float*>(g_ds.fa_partial_sum.data_ptr());
            auto* p_out = reinterpret_cast<float*>(g_ds.fa_partial_out.data_ptr());
            launch_flash_decode(qkv, kc_p[li], vc_p[li], rot,
                p_max, p_sum, p_out,
                (int)num_heads, (int)num_kv_heads, max_seq, cur_seq_len, (int)head_dim, stream);
            launch_gemv(rot, o_wp[li], o_sp[li], obuf, No, Ko, stream);
        }

        // 6. Residual + Norm + FWHT → rot
        launch_res_norm_fwht(obuf, h, post_np[li], rot, r, D, rms_eps, stream);

        // 7. Gate+Up GEMV
        launch_gemv(rot, gu_wp[li], gu_sp[li], gubuf, Ngu, Kgu, stream);

        // 8. SiLU*Up + Hadamard
        hipLaunchKernelGGL(silu_hadamard_ds, dim3((Kd + 31) / 32), dim3(32), 0, stream,
            gubuf, gubuf + inter, gubuf, had_p, Kd);

        // 9. Down GEMV
        launch_gemv(gubuf, down_wp[li], down_sp[li], dbuf, Nd, Kd, stream);

        // 10. Residual + Norm + FWHT → rot
        if (li < num_layers - 1) {
            launch_res_norm_fwht(dbuf, r, in_np[li + 1], rot, h, D, rms_eps, stream);
        } else {
            launch_res_norm_fwht(dbuf, r, in_np[0], rot, h, D, rms_eps, stream);
        }
    }

    // Return final hidden state [1, 1, D]
    return g_ds.hidden.view({1, 1, D}).clone();
}

// ============================================================
// Single decode step returning logits (folds final_norm + LM head)
// Eliminates 2 Python→C++ round trips per token.
// ============================================================
torch::Tensor int4_decode_step_logits(
    torch::Tensor hidden_in,
    std::vector<torch::Tensor> qkv_w, std::vector<torch::Tensor> qkv_s,
    std::vector<int64_t> qkv_N,
    std::vector<torch::Tensor> o_w, std::vector<torch::Tensor> o_s,
    std::vector<int64_t> o_N,
    std::vector<torch::Tensor> gu_w, std::vector<torch::Tensor> gu_s,
    std::vector<int64_t> gu_N,
    std::vector<torch::Tensor> down_w, std::vector<torch::Tensor> down_s,
    std::vector<int64_t> down_N,
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w, std::vector<torch::Tensor> k_norm_w,
    torch::Tensor rope_cos, torch::Tensor rope_sin,
    int64_t pos,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    float rms_eps,
    std::vector<torch::Tensor> k_caches, std::vector<torch::Tensor> v_caches,
    // Final norm + LM head
    torch::Tensor final_norm_w,
    torch::Tensor lm_w, torch::Tensor lm_s,
    int64_t lm_N, int64_t lm_K,
    torch::Tensor hadamard_mat
) {
    // Run the transformer layers
    auto h_out = int4_decode_step(
        hidden_in,
        qkv_w, qkv_s, qkv_N,
        o_w, o_s, o_N,
        gu_w, gu_s, gu_N,
        down_w, down_s, down_N,
        in_norm_w, post_norm_w,
        q_norm_w, k_norm_w,
        rope_cos, rope_sin,
        pos, num_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size, rms_eps,
        k_caches, v_caches,
        hadamard_mat
    );

    // Final norm + LM head in C++ (no Python round trip)
    int D = hidden_size;
    auto stream = at::hip::getCurrentHIPStream();
    auto device = h_out.device();
    auto* had_p = reinterpret_cast<const __half*>(hadamard_mat.data_ptr());

    // Final norm + FWHT → rot, then LM head GEMV (multi-block parallel)
    auto rot_lm = torch::empty({D}, torch::dtype(torch::kHalf).device(device));
    auto* h_ptr = reinterpret_cast<const __half*>(h_out.data_ptr());
    auto* fn_ptr = reinterpret_cast<const __half*>(final_norm_w.data_ptr());
    auto* rot_p = reinterpret_cast<__half*>(rot_lm.data_ptr());

    launch_rmsnorm_fwht(h_ptr, fn_ptr, rot_p, D, rms_eps, stream);

    // LM head GEMV (rot already has rotated normed input)
    auto logits = torch::empty({lm_N}, torch::dtype(torch::kHalf).device(device));
    launch_gemv(rot_p, lm_w.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(lm_s.data_ptr()),
        reinterpret_cast<__half*>(logits.data_ptr()),
        lm_N, lm_K, stream);

    return logits.view({1, 1, lm_N});
}

// ============================================================
// Decode benchmark — N tokens in tight C++ loop, zero Python overhead
// Uses a fixed dummy embedding to isolate pure kernel throughput
// ============================================================
void int4_decode_bench(
    torch::Tensor hidden_in,           // [1, 1, D] first token hidden
    torch::Tensor embed_weight,        // [vocab, D] for embedding lookup
    int64_t num_tokens,
    // Same layer args
    std::vector<torch::Tensor> qkv_w, std::vector<torch::Tensor> qkv_s,
    std::vector<int64_t> qkv_N,
    std::vector<torch::Tensor> o_w, std::vector<torch::Tensor> o_s,
    std::vector<int64_t> o_N,
    std::vector<torch::Tensor> gu_w, std::vector<torch::Tensor> gu_s,
    std::vector<int64_t> gu_N,
    std::vector<torch::Tensor> down_w, std::vector<torch::Tensor> down_s,
    std::vector<int64_t> down_N,
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w, std::vector<torch::Tensor> k_norm_w,
    torch::Tensor rope_cos, torch::Tensor rope_sin,
    int64_t start_pos,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    float rms_eps,
    std::vector<torch::Tensor> k_caches, std::vector<torch::Tensor> v_caches,
    torch::Tensor final_norm_w,
    torch::Tensor lm_w, torch::Tensor lm_s,
    int64_t lm_N, int64_t lm_K,
    torch::Tensor hadamard_mat
) {
    c10::InferenceMode guard;

    int D = hidden_size;
    auto device = hidden_in.device();
    auto stream = at::hip::getCurrentHIPStream();
    auto* had_p = reinterpret_cast<const __half*>(hadamard_mat.data_ptr());
    int norm_bs = 256;
    size_t smem_norm = D * sizeof(float);

    int num_layers = qkv_w.size();
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int max_seq = k_caches[0].size(2);
    int half_D = head_dim / 2;
    int inter = intermediate_size;

    // Ensure g_ds buffers are initialized
    int fa_max_splits_needed = (max_seq + FLASH_DECODE_BLOCK_S - 1) / FLASH_DECODE_BLOCK_S;
    if (!g_ds.init) {
        auto opts = torch::dtype(torch::kHalf).device(device);
        g_ds.normed = torch::empty({D}, opts);
        g_ds.hidden = torch::empty({D}, opts);
        g_ds.residual = torch::empty({D}, opts);
        g_ds.rotated = torch::empty({D}, opts);
        int max_qkv = q_dim + 2 * kv_dim;
        g_ds.qkv_buf = torch::empty({max_qkv}, opts);
        g_ds.o_buf = torch::empty({D}, opts);
        g_ds.attn_buf = torch::empty({D}, opts);
        int max_gu = 2 * (int)intermediate_size;
        g_ds.gu_buf = torch::empty({max_gu}, opts);
        g_ds.silu_buf = torch::empty({(long)intermediate_size}, opts);
        g_ds.down_buf = torch::empty({D}, opts);
        auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
        g_ds.fa_partial_max = torch::empty({fa_max_splits_needed, (long)num_heads}, opts_f32);
        g_ds.fa_partial_sum = torch::empty({fa_max_splits_needed, (long)num_heads}, opts_f32);
        g_ds.fa_partial_out = torch::empty({fa_max_splits_needed, (long)num_heads, (long)head_dim}, opts_f32);
        g_ds.fa_max_splits = fa_max_splits_needed;
        g_ds.rrms_buf = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
        g_ds.init = true;
    }
    if (fa_max_splits_needed > g_ds.fa_max_splits) {
        auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
        g_ds.fa_partial_max = torch::empty({fa_max_splits_needed, (long)num_heads}, opts_f32);
        g_ds.fa_partial_sum = torch::empty({fa_max_splits_needed, (long)num_heads}, opts_f32);
        g_ds.fa_partial_out = torch::empty({fa_max_splits_needed, (long)num_heads, (long)head_dim}, opts_f32);
        g_ds.fa_max_splits = fa_max_splits_needed;
    }

    // Pre-extract pointers once
    auto* h = reinterpret_cast<__half*>(g_ds.hidden.data_ptr());
    auto* n = reinterpret_cast<__half*>(g_ds.normed.data_ptr());
    auto* r = reinterpret_cast<__half*>(g_ds.residual.data_ptr());
    auto* rot = reinterpret_cast<__half*>(g_ds.rotated.data_ptr());
    auto* qkv_buf = reinterpret_cast<__half*>(g_ds.qkv_buf.data_ptr());
    auto* obuf = reinterpret_cast<__half*>(g_ds.o_buf.data_ptr());
    auto* abuf = reinterpret_cast<__half*>(g_ds.attn_buf.data_ptr());
    auto* gubuf = reinterpret_cast<__half*>(g_ds.gu_buf.data_ptr());
    auto* silubuf = reinterpret_cast<__half*>(g_ds.silu_buf.data_ptr());
    auto* dbuf = reinterpret_cast<__half*>(g_ds.down_buf.data_ptr());

    std::vector<const uint8_t*> qkv_wp(num_layers);
    std::vector<const __half*> qkv_sp(num_layers);
    std::vector<const uint8_t*> o_wp(num_layers);
    std::vector<const __half*> o_sp(num_layers);
    std::vector<const uint8_t*> gu_wp(num_layers);
    std::vector<const __half*> gu_sp(num_layers);
    std::vector<const uint8_t*> down_wp(num_layers);
    std::vector<const __half*> down_sp(num_layers);
    std::vector<const __half*> in_np(num_layers), post_np(num_layers);
    std::vector<const __half*> q_np(num_layers), k_np(num_layers);
    std::vector<uint8_t*> kc_p(num_layers), vc_p(num_layers);
    for (int li = 0; li < num_layers; li++) {
        qkv_wp[li] = qkv_w[li].data_ptr<uint8_t>();
        qkv_sp[li] = reinterpret_cast<const __half*>(qkv_s[li].data_ptr());
        o_wp[li] = o_w[li].data_ptr<uint8_t>();
        o_sp[li] = reinterpret_cast<const __half*>(o_s[li].data_ptr());
        gu_wp[li] = gu_w[li].data_ptr<uint8_t>();
        gu_sp[li] = reinterpret_cast<const __half*>(gu_s[li].data_ptr());
        down_wp[li] = down_w[li].data_ptr<uint8_t>();
        down_sp[li] = reinterpret_cast<const __half*>(down_s[li].data_ptr());
        in_np[li] = reinterpret_cast<const __half*>(in_norm_w[li].data_ptr());
        post_np[li] = reinterpret_cast<const __half*>(post_norm_w[li].data_ptr());
        q_np[li] = reinterpret_cast<const __half*>(q_norm_w[li].data_ptr());
        k_np[li] = reinterpret_cast<const __half*>(k_norm_w[li].data_ptr());
        kc_p[li] = k_caches[li].data_ptr<uint8_t>();
        vc_p[li] = v_caches[li].data_ptr<uint8_t>();
    }

    auto* fn_p = reinterpret_cast<const __half*>(final_norm_w.data_ptr());

    // Allocate normed buffer + logits buffer
    auto normed_buf = torch::empty({D}, torch::dtype(torch::kHalf).device(device));
    auto logits_buf = torch::empty({lm_N}, torch::dtype(torch::kHalf).device(device));
    auto* nb_p = reinterpret_cast<__half*>(normed_buf.data_ptr());
    auto* lmw_p = lm_w.data_ptr<uint8_t>();
    auto* lms_p = reinterpret_cast<const __half*>(lm_s.data_ptr());
    auto* lmb_p = reinterpret_cast<__half*>(logits_buf.data_ptr());

    auto* embed_ptr = reinterpret_cast<const __half*>(embed_weight.data_ptr());

    // Copy first token hidden state
    hipMemcpyAsync(h, hidden_in.data_ptr(), D * sizeof(__half), hipMemcpyDeviceToDevice, stream);

    for (int64_t t = 0; t < num_tokens; t++) {
        int64_t pos = start_pos + t;
        int seq_len = pos + 1;
        const __half* cos_pos = reinterpret_cast<const __half*>(rope_cos.data_ptr()) + pos * half_D;
        const __half* sin_pos = reinterpret_cast<const __half*>(rope_sin.data_ptr()) + pos * half_D;

        // First layer: RMSNorm + FWHT → rot (multi-block parallel)
        launch_rmsnorm_fwht(h, in_np[0], rot, D, rms_eps, stream);

        for (int li = 0; li < num_layers; li++) {
            int Nqkv = qkv_N[li], Kqkv = D;
            int No = o_N[li];
            int Ngu = gu_N[li], Kgu = D;
            int Nd = down_N[li], Kd = intermediate_size;

            launch_gemv(rot, qkv_wp[li], qkv_sp[li], qkv_buf, Nqkv, Kqkv, stream);

            __half* q_ptr = qkv_buf;
            __half* k_ptr = qkv_buf + q_dim;
            __half* v_ptr = qkv_buf + q_dim + kv_dim;

            hipLaunchKernelGGL(head_rmsnorm_rope_kv, dim3(num_heads + num_kv_heads), dim3(32), 0, stream,
                q_ptr, k_ptr, v_ptr, q_np[li], k_np[li],
                q_ptr, k_ptr, kc_p[li], vc_p[li],
                cos_pos, sin_pos,
                num_heads, num_kv_heads, head_dim, rms_eps, max_seq, pos);

            {
                auto* p_max = reinterpret_cast<float*>(g_ds.fa_partial_max.data_ptr());
                auto* p_sum = reinterpret_cast<float*>(g_ds.fa_partial_sum.data_ptr());
                auto* p_out = reinterpret_cast<float*>(g_ds.fa_partial_out.data_ptr());
                launch_flash_decode(qkv_buf, kc_p[li], vc_p[li], rot,
                    p_max, p_sum, p_out,
                    (int)num_heads, (int)num_kv_heads, max_seq, seq_len, (int)head_dim, stream);
                launch_gemv(rot, o_wp[li], o_sp[li], obuf, No, D, stream);
            }

            // Residual + Norm + FWHT → rot (MULTI-BLOCK PARALLEL)
            launch_res_norm_fwht(obuf, h, post_np[li], rot, r, D, rms_eps, stream);

            launch_gemv(rot, gu_wp[li], gu_sp[li], gubuf, Ngu, Kgu, stream);

            hipLaunchKernelGGL(silu_hadamard_ds, dim3((Kd + 31) / 32), dim3(32), 0, stream,
                gubuf, gubuf + inter, gubuf, had_p, Kd);
            launch_gemv(gubuf, down_wp[li], down_sp[li], dbuf, Nd, Kd, stream);

            // Residual + Norm + FWHT → rot (MULTI-BLOCK PARALLEL)
            if (li < num_layers - 1) {
                launch_res_norm_fwht(dbuf, r, in_np[li + 1], rot, h, D, rms_eps, stream);
            } else {
                launch_res_norm_fwht(dbuf, r, in_np[0], rot, h, D, rms_eps, stream);
            }
        }

        // Final norm + FWHT + LM head (multi-block parallel)
        launch_rmsnorm_fwht(h, fn_p, rot, D, rms_eps, stream);
        launch_gemv(rot, lmw_p, lms_p, lmb_p, lm_N, lm_K, stream);

        // Argmax -> embedding lookup for next token (all GPU, no sync)
        if (t < num_tokens - 1) {
            auto next_id = logits_buf.argmax(0);
            auto next_embed = embed_weight.index({next_id.unsqueeze(0)});
            hipMemcpyAsync(h, next_embed.data_ptr(), D * sizeof(__half), hipMemcpyDeviceToDevice, stream);
        }
    }
}

// ============================================================
// HIP Graph decode benchmark — captures entire decode as graph,
// replays per token. Eliminates ~5ms kernel launch overhead.
// ============================================================
void int4_decode_bench_graph(
    torch::Tensor hidden_in,
    torch::Tensor embed_weight,
    int64_t num_tokens,
    std::vector<torch::Tensor> qkv_w, std::vector<torch::Tensor> qkv_s,
    std::vector<int64_t> qkv_N,
    std::vector<torch::Tensor> o_w, std::vector<torch::Tensor> o_s,
    std::vector<int64_t> o_N,
    std::vector<torch::Tensor> gu_w, std::vector<torch::Tensor> gu_s,
    std::vector<int64_t> gu_N,
    std::vector<torch::Tensor> down_w, std::vector<torch::Tensor> down_s,
    std::vector<int64_t> down_N,
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w, std::vector<torch::Tensor> k_norm_w,
    torch::Tensor rope_cos, torch::Tensor rope_sin,
    int64_t start_pos,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    float rms_eps,
    std::vector<torch::Tensor> k_caches, std::vector<torch::Tensor> v_caches,
    torch::Tensor final_norm_w,
    torch::Tensor lm_w, torch::Tensor lm_s,
    int64_t lm_N, int64_t lm_K,
    torch::Tensor hadamard_mat
) {
    c10::InferenceMode guard;

    int D = hidden_size;
    auto device = hidden_in.device();

    // Create a NON-DEFAULT stream for graph capture (required by HIP/CUDA)
    hipStream_t graph_stream;
    hipStreamCreate(&graph_stream);
    // Use the default stream for setup operations
    auto default_stream = at::hip::getCurrentHIPStream();
    // All operations will use graph_stream
    hipStream_t stream = graph_stream;

    int num_layers = qkv_w.size();
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int max_seq = k_caches[0].size(2);
    int half_D = head_dim / 2;
    int inter = intermediate_size;

    // Ensure g_ds buffers (use default stream for allocations)
    int fa_max_splits_needed = (max_seq + FLASH_DECODE_BLOCK_S - 1) / FLASH_DECODE_BLOCK_S;
    if (!g_ds.init) {
        auto opts = torch::dtype(torch::kHalf).device(device);
        g_ds.normed = torch::empty({D}, opts);
        g_ds.hidden = torch::empty({D}, opts);
        g_ds.residual = torch::empty({D}, opts);
        g_ds.rotated = torch::empty({D}, opts);
        g_ds.qkv_buf = torch::empty({q_dim + 2 * kv_dim}, opts);
        g_ds.o_buf = torch::empty({D}, opts);
        g_ds.attn_buf = torch::empty({D}, opts);
        g_ds.gu_buf = torch::empty({2 * inter}, opts);
        g_ds.silu_buf = torch::empty({(long)inter}, opts);
        g_ds.down_buf = torch::empty({D}, opts);
        auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
        g_ds.fa_partial_max = torch::empty({fa_max_splits_needed, (long)num_heads}, opts_f32);
        g_ds.fa_partial_sum = torch::empty({fa_max_splits_needed, (long)num_heads}, opts_f32);
        g_ds.fa_partial_out = torch::empty({fa_max_splits_needed, (long)num_heads, (long)head_dim}, opts_f32);
        g_ds.fa_max_splits = fa_max_splits_needed;
        g_ds.rrms_buf = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
        g_ds.init = true;
    }

    // Determine flash decode grid dims for entire run
    int max_seq_len_run = start_pos + num_tokens;
    int block_s, max_n_splits;
    if (max_seq_len_run <= 256) {
        block_s = 32;
        max_n_splits = (max_seq_len_run + 31) / 32;
    } else {
        block_s = 64;
        max_n_splits = (max_seq_len_run + 63) / 64;
    }

    // Ensure partial buffers large enough for max_n_splits
    if (max_n_splits > g_ds.fa_max_splits) {
        auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
        g_ds.fa_partial_max = torch::empty({max_n_splits, (long)num_heads}, opts_f32);
        g_ds.fa_partial_sum = torch::empty({max_n_splits, (long)num_heads}, opts_f32);
        g_ds.fa_partial_out = torch::empty({max_n_splits, (long)num_heads, (long)head_dim}, opts_f32);
        g_ds.fa_max_splits = max_n_splits;
    }

    // Extract pointers
    auto* h = reinterpret_cast<__half*>(g_ds.hidden.data_ptr());
    auto* r = reinterpret_cast<__half*>(g_ds.residual.data_ptr());
    auto* rot = reinterpret_cast<__half*>(g_ds.rotated.data_ptr());
    auto* qkv_buf = reinterpret_cast<__half*>(g_ds.qkv_buf.data_ptr());
    auto* obuf = reinterpret_cast<__half*>(g_ds.o_buf.data_ptr());
    auto* gubuf = reinterpret_cast<__half*>(g_ds.gu_buf.data_ptr());
    auto* dbuf = reinterpret_cast<__half*>(g_ds.down_buf.data_ptr());
    auto* p_max = reinterpret_cast<float*>(g_ds.fa_partial_max.data_ptr());
    auto* p_sum = reinterpret_cast<float*>(g_ds.fa_partial_sum.data_ptr());
    auto* p_out = reinterpret_cast<float*>(g_ds.fa_partial_out.data_ptr());

    std::vector<const uint8_t*> qkv_wp(num_layers), o_wp(num_layers),
                                 gu_wp(num_layers), down_wp(num_layers);
    std::vector<const __half*> qkv_sp(num_layers), o_sp(num_layers),
                                gu_sp(num_layers), down_sp(num_layers);
    std::vector<const __half*> in_np(num_layers), post_np(num_layers),
                                q_np(num_layers), k_np(num_layers);
    std::vector<uint8_t*> kc_p(num_layers), vc_p(num_layers);
    for (int li = 0; li < num_layers; li++) {
        qkv_wp[li] = qkv_w[li].data_ptr<uint8_t>();
        qkv_sp[li] = reinterpret_cast<const __half*>(qkv_s[li].data_ptr());
        o_wp[li] = o_w[li].data_ptr<uint8_t>();
        o_sp[li] = reinterpret_cast<const __half*>(o_s[li].data_ptr());
        gu_wp[li] = gu_w[li].data_ptr<uint8_t>();
        gu_sp[li] = reinterpret_cast<const __half*>(gu_s[li].data_ptr());
        down_wp[li] = down_w[li].data_ptr<uint8_t>();
        down_sp[li] = reinterpret_cast<const __half*>(down_s[li].data_ptr());
        in_np[li] = reinterpret_cast<const __half*>(in_norm_w[li].data_ptr());
        post_np[li] = reinterpret_cast<const __half*>(post_norm_w[li].data_ptr());
        q_np[li] = reinterpret_cast<const __half*>(q_norm_w[li].data_ptr());
        k_np[li] = reinterpret_cast<const __half*>(k_norm_w[li].data_ptr());
        kc_p[li] = k_caches[li].data_ptr<uint8_t>();
        vc_p[li] = v_caches[li].data_ptr<uint8_t>();
    }

    auto* fn_p = reinterpret_cast<const __half*>(final_norm_w.data_ptr());
    auto* cos_base = reinterpret_cast<const __half*>(rope_cos.data_ptr());
    auto* sin_base = reinterpret_cast<const __half*>(rope_sin.data_ptr());
    auto logits_buf = torch::empty({lm_N}, torch::dtype(torch::kHalf).device(device));
    auto* lmw_p = lm_w.data_ptr<uint8_t>();
    auto* lms_p = reinterpret_cast<const __half*>(lm_s.data_ptr());
    auto* lmb_p = reinterpret_cast<__half*>(logits_buf.data_ptr());
    auto* embed_ptr = reinterpret_cast<const __half*>(embed_weight.data_ptr());

    // Copy initial hidden state
    hipMemcpyAsync(h, hidden_in.data_ptr(), D * sizeof(__half), hipMemcpyDeviceToDevice, stream);

    // Set initial graph state
    hipLaunchKernelGGL(set_graph_state, dim3(1), dim3(1), 0, stream, (int)start_pos);
    hipStreamSynchronize(stream);

    // ---- Capture graph: one full decode token ----
    hipGraph_t graph;
    hipGraphExec_t graph_exec;
    hipError_t err;

    err = hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    if (err != hipSuccess) {
        fprintf(stderr, "hipStreamBeginCapture failed: %s\n", hipGetErrorString(err));
        hipStreamDestroy(graph_stream);
        return;
    }

    // First layer norm
    launch_rmsnorm_fwht(h, in_np[0], rot, D, rms_eps, stream);

    for (int li = 0; li < num_layers; li++) {
        int Nqkv = qkv_N[li], Kqkv = D;
        int No = o_N[li];
        int Ngu = gu_N[li], Kgu = D;
        int Nd = down_N[li], Kd = inter;

        launch_gemv(rot, qkv_wp[li], qkv_sp[li], qkv_buf, Nqkv, Kqkv, stream);

        __half* q_ptr = qkv_buf;
        __half* k_ptr = qkv_buf + q_dim;
        __half* v_ptr = qkv_buf + q_dim + kv_dim;

        // Graph-compatible RoPE (reads pos from g_graph_pos)
        hipLaunchKernelGGL(head_rmsnorm_rope_kv_g,
            dim3(num_heads + num_kv_heads), dim3(32), 0, stream,
            q_ptr, k_ptr, v_ptr, q_np[li], k_np[li],
            q_ptr, k_ptr, kc_p[li], vc_p[li],
            cos_base, sin_base,
            (int)num_heads, (int)num_kv_heads, (int)head_dim, rms_eps, max_seq);

        // Graph-compatible flash decode (reads seq_len from g_graph_seq_len)
        launch_flash_decode_g(qkv_buf, kc_p[li], vc_p[li], rot,
            p_max, p_sum, p_out,
            (int)num_heads, (int)num_kv_heads, max_seq,
            max_n_splits, block_s, (int)head_dim, stream);

        launch_gemv(rot, o_wp[li], o_sp[li], obuf, No, D, stream);

        launch_res_norm_fwht(obuf, h, post_np[li], rot, r, D, rms_eps, stream);

        launch_gemv(rot, gu_wp[li], gu_sp[li], gubuf, Ngu, Kgu, stream);

        hipLaunchKernelGGL(silu_hadamard_ds, dim3((Kd + 31) / 32), dim3(32), 0, stream,
            gubuf, gubuf + inter, gubuf, reinterpret_cast<const __half*>(hadamard_mat.data_ptr()), Kd);

        launch_gemv(gubuf, down_wp[li], down_sp[li], dbuf, Nd, Kd, stream);

        if (li < num_layers - 1)
            launch_res_norm_fwht(dbuf, r, in_np[li + 1], rot, h, D, rms_eps, stream);
        else
            launch_res_norm_fwht(dbuf, r, in_np[0], rot, h, D, rms_eps, stream);
    }

    // Final norm + LM head
    launch_rmsnorm_fwht(h, fn_p, rot, D, rms_eps, stream);
    launch_gemv(rot, lmw_p, lms_p, lmb_p, lm_N, lm_K, stream);

    err = hipStreamEndCapture(stream, &graph);
    if (err != hipSuccess) {
        fprintf(stderr, "hipStreamEndCapture failed: %s\n", hipGetErrorString(err));
        return;
    }
    fprintf(stderr, "Graph captured. Instantiating...\n");

    err = hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
    if (err != hipSuccess) {
        fprintf(stderr, "hipGraphInstantiate failed: %s\n", hipGetErrorString(err));
        hipGraphDestroy(graph);
        return;
    }
    fprintf(stderr, "Graph instantiated. Running %ld tokens...\n", num_tokens);

    // ---- Replay loop ----
    // Note: skip argmax/embed between tokens for pure graph overhead measurement.
    // The decode kernel throughput is identical regardless of input embedding.
    for (int64_t t = 0; t < num_tokens; t++) {
        int pos = start_pos + t;

        // Update graph state (tiny 1-thread kernel, ~2μs)
        hipLaunchKernelGGL(set_graph_state, dim3(1), dim3(1), 0, stream, pos);

        // Replay captured graph (1 launch = all ~440 kernels)
        err = hipGraphLaunch(graph_exec, stream);
        if (err != hipSuccess) {
            fprintf(stderr, "hipGraphLaunch failed at t=%ld: %s\n", t, hipGetErrorString(err));
            break;
        }
    }

    hipStreamSynchronize(stream);

    // Cleanup
    hipGraphExecDestroy(graph_exec);
    hipGraphDestroy(graph);
    hipStreamDestroy(graph_stream);
}

// ============================================================
// Profiled decode — times per-category kernel groups for 1 token
// Returns vector of floats: [total_ms, gemv_ms, attn_ms, norm_ms, lmhead_ms, other_ms]
// ============================================================
std::vector<float> int4_decode_profile(
    torch::Tensor hidden_in,
    torch::Tensor embed_weight,
    std::vector<torch::Tensor> qkv_w, std::vector<torch::Tensor> qkv_s,
    std::vector<int64_t> qkv_N,
    std::vector<torch::Tensor> o_w, std::vector<torch::Tensor> o_s,
    std::vector<int64_t> o_N,
    std::vector<torch::Tensor> gu_w, std::vector<torch::Tensor> gu_s,
    std::vector<int64_t> gu_N,
    std::vector<torch::Tensor> down_w, std::vector<torch::Tensor> down_s,
    std::vector<int64_t> down_N,
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w, std::vector<torch::Tensor> k_norm_w,
    torch::Tensor rope_cos, torch::Tensor rope_sin,
    int64_t start_pos,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    float rms_eps,
    std::vector<torch::Tensor> k_caches, std::vector<torch::Tensor> v_caches,
    torch::Tensor final_norm_w,
    torch::Tensor lm_w, torch::Tensor lm_s,
    int64_t lm_N, int64_t lm_K,
    torch::Tensor hadamard_mat
) {
    c10::InferenceMode guard;

    int D = hidden_size;
    auto device = hidden_in.device();
    auto stream = at::hip::getCurrentHIPStream();
    auto* had_p = reinterpret_cast<const __half*>(hadamard_mat.data_ptr());
    int norm_bs = 256;
    size_t smem_norm = D * sizeof(float);

    int num_layers = qkv_w.size();
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int max_seq = k_caches[0].size(2);
    int half_D = head_dim / 2;
    int inter = intermediate_size;

    // Ensure g_ds buffers initialized
    int fa_max_splits_needed = (max_seq + FLASH_DECODE_BLOCK_S - 1) / FLASH_DECODE_BLOCK_S;
    if (!g_ds.init) {
        auto opts = torch::dtype(torch::kHalf).device(device);
        g_ds.normed = torch::empty({D}, opts);
        g_ds.hidden = torch::empty({D}, opts);
        g_ds.residual = torch::empty({D}, opts);
        g_ds.rotated = torch::empty({D}, opts);
        g_ds.qkv_buf = torch::empty({q_dim + 2 * kv_dim}, opts);
        g_ds.o_buf = torch::empty({D}, opts);
        g_ds.attn_buf = torch::empty({D}, opts);
        g_ds.gu_buf = torch::empty({2 * (int)intermediate_size}, opts);
        g_ds.silu_buf = torch::empty({(long)intermediate_size}, opts);
        g_ds.down_buf = torch::empty({D}, opts);
        auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
        g_ds.fa_partial_max = torch::empty({fa_max_splits_needed, (long)num_heads}, opts_f32);
        g_ds.fa_partial_sum = torch::empty({fa_max_splits_needed, (long)num_heads}, opts_f32);
        g_ds.fa_partial_out = torch::empty({fa_max_splits_needed, (long)num_heads, (long)head_dim}, opts_f32);
        g_ds.fa_max_splits = fa_max_splits_needed;
        g_ds.rrms_buf = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
        g_ds.init = true;
    }
    // Grow flash attention buffers if needed (larger KV cache than initial)
    if (fa_max_splits_needed > g_ds.fa_max_splits) {
        auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
        g_ds.fa_partial_max = torch::empty({fa_max_splits_needed, (long)num_heads}, opts_f32);
        g_ds.fa_partial_sum = torch::empty({fa_max_splits_needed, (long)num_heads}, opts_f32);
        g_ds.fa_partial_out = torch::empty({fa_max_splits_needed, (long)num_heads, (long)head_dim}, opts_f32);
        g_ds.fa_max_splits = fa_max_splits_needed;
    }

    auto* h = reinterpret_cast<__half*>(g_ds.hidden.data_ptr());
    auto* n = reinterpret_cast<__half*>(g_ds.normed.data_ptr());
    auto* r = reinterpret_cast<__half*>(g_ds.residual.data_ptr());
    auto* rot = reinterpret_cast<__half*>(g_ds.rotated.data_ptr());
    auto* qkv_buf = reinterpret_cast<__half*>(g_ds.qkv_buf.data_ptr());
    auto* obuf = reinterpret_cast<__half*>(g_ds.o_buf.data_ptr());
    auto* abuf = reinterpret_cast<__half*>(g_ds.attn_buf.data_ptr());
    auto* gubuf = reinterpret_cast<__half*>(g_ds.gu_buf.data_ptr());
    auto* silubuf = reinterpret_cast<__half*>(g_ds.silu_buf.data_ptr());
    auto* dbuf = reinterpret_cast<__half*>(g_ds.down_buf.data_ptr());

    // Pre-extract pointers
    std::vector<const uint8_t*> qkv_wp(num_layers);
    std::vector<const __half*> qkv_sp(num_layers);
    std::vector<const uint8_t*> o_wp(num_layers);
    std::vector<const __half*> o_sp(num_layers);
    std::vector<const uint8_t*> gu_wp(num_layers);
    std::vector<const __half*> gu_sp(num_layers);
    std::vector<const uint8_t*> down_wp(num_layers);
    std::vector<const __half*> down_sp(num_layers);
    std::vector<const __half*> in_np(num_layers), post_np(num_layers);
    std::vector<const __half*> q_np(num_layers), k_np(num_layers);
    std::vector<uint8_t*> kc_p(num_layers), vc_p(num_layers);
    for (int li = 0; li < num_layers; li++) {
        qkv_wp[li] = qkv_w[li].data_ptr<uint8_t>();
        qkv_sp[li] = reinterpret_cast<const __half*>(qkv_s[li].data_ptr());
        o_wp[li] = o_w[li].data_ptr<uint8_t>();
        o_sp[li] = reinterpret_cast<const __half*>(o_s[li].data_ptr());
        gu_wp[li] = gu_w[li].data_ptr<uint8_t>();
        gu_sp[li] = reinterpret_cast<const __half*>(gu_s[li].data_ptr());
        down_wp[li] = down_w[li].data_ptr<uint8_t>();
        down_sp[li] = reinterpret_cast<const __half*>(down_s[li].data_ptr());
        in_np[li] = reinterpret_cast<const __half*>(in_norm_w[li].data_ptr());
        post_np[li] = reinterpret_cast<const __half*>(post_norm_w[li].data_ptr());
        q_np[li] = reinterpret_cast<const __half*>(q_norm_w[li].data_ptr());
        k_np[li] = reinterpret_cast<const __half*>(k_norm_w[li].data_ptr());
        kc_p[li] = k_caches[li].data_ptr<uint8_t>();
        vc_p[li] = v_caches[li].data_ptr<uint8_t>();
    }

    auto* fn_p = reinterpret_cast<const __half*>(final_norm_w.data_ptr());
    auto normed_buf = torch::empty({D}, torch::dtype(torch::kHalf).device(device));
    auto logits_buf = torch::empty({lm_N}, torch::dtype(torch::kHalf).device(device));
    auto* nb_p = reinterpret_cast<__half*>(normed_buf.data_ptr());
    auto* lmw_p = lm_w.data_ptr<uint8_t>();
    auto* lms_p = reinterpret_cast<const __half*>(lm_s.data_ptr());
    auto* lmb_p = reinterpret_cast<__half*>(logits_buf.data_ptr());

    // Create per-layer event arrays (no per-layer sync!)
    // 9 events per layer: start, after_qkv, after_hn, after_attn, after_o, after_rn1, after_gu, after_down, after_rn2
    const int EVTS_PER_LAYER = 9;
    std::vector<hipEvent_t> events(num_layers * EVTS_PER_LAYER);
    hipEvent_t e_total_start, e_total_end;
    hipEventCreate(&e_total_start);
    hipEventCreate(&e_total_end);
    for (auto& ev : events) hipEventCreate(&ev);

    hipMemcpyAsync(h, hidden_in.data_ptr(), D * sizeof(__half), hipMemcpyDeviceToDevice, stream);

    int64_t pos = start_pos;
    int seq_len = pos + 1;
    const __half* cos_pos = reinterpret_cast<const __half*>(rope_cos.data_ptr()) + pos * half_D;
    const __half* sin_pos = reinterpret_cast<const __half*>(rope_sin.data_ptr()) + pos * half_D;

    hipEventRecord(e_total_start, stream);

    // First layer: RMSNorm + FWHT → rot (multi-block parallel)
    launch_rmsnorm_fwht(h, in_np[0], rot, D, rms_eps, stream);

    for (int li = 0; li < num_layers; li++) {
        hipEvent_t* ev = &events[li * EVTS_PER_LAYER];
        int Nqkv = qkv_N[li], No = o_N[li];
        int Ngu = gu_N[li], Kgu = D;
        int Nd = down_N[li], Kd = intermediate_size;

        hipEventRecord(ev[0], stream);  // start
        launch_gemv(rot, qkv_wp[li], qkv_sp[li], qkv_buf, Nqkv, D, stream);
        hipEventRecord(ev[1], stream);  // after_qkv

        __half* q_ptr = qkv_buf;
        __half* k_ptr = qkv_buf + q_dim;
        __half* v_ptr = qkv_buf + q_dim + kv_dim;

        hipLaunchKernelGGL(head_rmsnorm_rope_kv, dim3(num_heads + num_kv_heads), dim3(32), 0, stream,
            q_ptr, k_ptr, v_ptr, q_np[li], k_np[li],
            q_ptr, k_ptr, kc_p[li], vc_p[li],
            cos_pos, sin_pos,
            num_heads, num_kv_heads, head_dim, rms_eps, max_seq, pos);
        hipEventRecord(ev[2], stream);  // after_headnorm

        {
            auto* p_max = reinterpret_cast<float*>(g_ds.fa_partial_max.data_ptr());
            auto* p_sum = reinterpret_cast<float*>(g_ds.fa_partial_sum.data_ptr());
            auto* p_out = reinterpret_cast<float*>(g_ds.fa_partial_out.data_ptr());
            launch_flash_decode(qkv_buf, kc_p[li], vc_p[li], rot,
                p_max, p_sum, p_out,
                (int)num_heads, (int)num_kv_heads, max_seq, seq_len, (int)head_dim, stream);
        }
        hipEventRecord(ev[3], stream);  // after_attn

        launch_gemv(rot, o_wp[li], o_sp[li], obuf, No, D, stream);
        hipEventRecord(ev[4], stream);  // after_o

        // Residual + Norm + FWHT → rot (MULTI-BLOCK PARALLEL)
        launch_res_norm_fwht(obuf, h, post_np[li], rot, r, D, rms_eps, stream);
        hipEventRecord(ev[5], stream);  // after_rn1

        launch_gemv(rot, gu_wp[li], gu_sp[li], gubuf, Ngu, Kgu, stream);
        hipEventRecord(ev[6], stream);  // after_gu

        // SiLU*Up + Hadamard + Down GEMV
        hipLaunchKernelGGL(silu_hadamard_ds, dim3((Kd + 31) / 32), dim3(32), 0, stream,
            gubuf, gubuf + inter, gubuf, had_p, Kd);
        launch_gemv(gubuf, down_wp[li], down_sp[li], dbuf, Nd, Kd, stream);
        hipEventRecord(ev[7], stream);  // after_down

        // Residual + Norm + FWHT → rot (MULTI-BLOCK PARALLEL)
        if (li < num_layers - 1)
            launch_res_norm_fwht(dbuf, r, in_np[li + 1], rot, h, D, rms_eps, stream);
        else
            launch_res_norm_fwht(dbuf, r, in_np[0], rot, h, D, rms_eps, stream);
        hipEventRecord(ev[8], stream);  // after_rn2
    }

    // Final norm + FWHT + LM head (multi-block parallel)
    launch_rmsnorm_fwht(h, fn_p, rot, D, rms_eps, stream);
    launch_gemv(rot, lmw_p, lms_p, lmb_p, lm_N, lm_K, stream);

    hipEventRecord(e_total_end, stream);
    hipEventSynchronize(e_total_end);  // Single sync at the end

    // Now compute all elapsed times
    float total_gemv_ms = 0, total_attn_ms = 0, total_norm_ms = 0;
    for (int li = 0; li < num_layers; li++) {
        hipEvent_t* ev = &events[li * EVTS_PER_LAYER];
        float ms;
        hipEventElapsedTime(&ms, ev[0], ev[1]); total_gemv_ms += ms;  // QKV
        hipEventElapsedTime(&ms, ev[1], ev[2]); total_norm_ms += ms;  // headnorm+rope
        hipEventElapsedTime(&ms, ev[2], ev[3]); total_attn_ms += ms;  // attention
        hipEventElapsedTime(&ms, ev[3], ev[4]); total_gemv_ms += ms;  // O
        hipEventElapsedTime(&ms, ev[4], ev[5]); total_norm_ms += ms;  // resnorm1
        hipEventElapsedTime(&ms, ev[5], ev[6]); total_gemv_ms += ms;  // GU
        hipEventElapsedTime(&ms, ev[6], ev[7]); total_gemv_ms += ms;  // Down
        hipEventElapsedTime(&ms, ev[7], ev[8]); total_norm_ms += ms;  // resnorm2
    }

    float total_ms;
    hipEventElapsedTime(&total_ms, e_total_start, e_total_end);
    float rest_ms = total_ms - total_gemv_ms - total_attn_ms - total_norm_ms;

    // Cleanup
    hipEventDestroy(e_total_start); hipEventDestroy(e_total_end);
    for (auto& ev : events) hipEventDestroy(ev);

    return {total_ms, total_gemv_ms, total_attn_ms, total_norm_ms, rest_ms};
}

// ============================================================
// Full decode loop — N tokens entirely in C++
// ============================================================
struct FullDSContext {
    torch::Tensor lm_buf;  // [lm_N]
    bool init = false;
};
static FullDSContext g_full;

torch::Tensor int4_decode_loop(
    torch::Tensor embed_weight,        // [vocab, D] FP16
    torch::Tensor final_norm_w,        // [D] FP16
    torch::Tensor lm_w, torch::Tensor lm_s,
    int64_t lm_N, int64_t lm_K,
    torch::Tensor first_token_hidden,  // [1, 1, D] FP16
    int64_t num_tokens,
    // Same args as decode_step
    std::vector<torch::Tensor> qkv_w, std::vector<torch::Tensor> qkv_s,
    std::vector<int64_t> qkv_N,
    std::vector<torch::Tensor> o_w, std::vector<torch::Tensor> o_s,
    std::vector<int64_t> o_N,
    std::vector<torch::Tensor> gu_w, std::vector<torch::Tensor> gu_s,
    std::vector<int64_t> gu_N,
    std::vector<torch::Tensor> down_w, std::vector<torch::Tensor> down_s,
    std::vector<int64_t> down_N,
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w, std::vector<torch::Tensor> k_norm_w,
    torch::Tensor rope_cos, torch::Tensor rope_sin,
    int64_t start_pos,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    float rms_eps,
    std::vector<torch::Tensor> k_caches, std::vector<torch::Tensor> v_caches,
    torch::Tensor hadamard_mat
) {
    c10::InferenceMode guard;

    int D = hidden_size;
    auto device = first_token_hidden.device();
    auto stream = at::hip::getCurrentHIPStream();
    auto* had_p = reinterpret_cast<const __half*>(hadamard_mat.data_ptr());
    int norm_bs = 256;
    size_t smem_norm = D * sizeof(float);

    // Init lm_head buffer
    if (!g_full.init) {
        g_full.lm_buf = torch::empty({lm_N}, torch::dtype(torch::kHalf).device(device));
        g_full.init = true;
    }

    // All token IDs on GPU -- no per-token GPU->CPU sync
    auto token_ids_gpu = torch::empty({(long)num_tokens}, torch::dtype(torch::kLong).device(device));

    // Pre-allocate normed buffer
    auto normed_buf = torch::empty({D}, torch::dtype(torch::kHalf).device(device));

    int num_layers = qkv_w.size();
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int max_seq = k_caches[0].size(2);
    int half_D = head_dim / 2;
    int kv_total = num_kv_heads * head_dim;

    // Pre-extract all raw pointers (once, outside token loop)
    auto* h = reinterpret_cast<__half*>(g_ds.hidden.data_ptr());
    auto* n = reinterpret_cast<__half*>(g_ds.normed.data_ptr());
    auto* r = reinterpret_cast<__half*>(g_ds.residual.data_ptr());
    auto* rot = reinterpret_cast<__half*>(g_ds.rotated.data_ptr());
    auto* qkv_buf = reinterpret_cast<__half*>(g_ds.qkv_buf.data_ptr());
    auto* obuf = reinterpret_cast<__half*>(g_ds.o_buf.data_ptr());
    auto* abuf = reinterpret_cast<__half*>(g_ds.attn_buf.data_ptr());
    auto* gubuf = reinterpret_cast<__half*>(g_ds.gu_buf.data_ptr());
    auto* silubuf = reinterpret_cast<__half*>(g_ds.silu_buf.data_ptr());
    auto* dbuf = reinterpret_cast<__half*>(g_ds.down_buf.data_ptr());

    std::vector<const uint8_t*> qkv_wp(num_layers);
    std::vector<const __half*> qkv_sp(num_layers);
    std::vector<const uint8_t*> o_wp(num_layers);
    std::vector<const __half*> o_sp(num_layers);
    std::vector<const uint8_t*> gu_wp(num_layers);
    std::vector<const __half*> gu_sp(num_layers);
    std::vector<const uint8_t*> down_wp(num_layers);
    std::vector<const __half*> down_sp(num_layers);
    std::vector<const __half*> in_np(num_layers), post_np(num_layers);
    std::vector<const __half*> q_np(num_layers), k_np(num_layers);
    std::vector<uint8_t*> kc_p(num_layers), vc_p(num_layers);
    for (int li = 0; li < num_layers; li++) {
        qkv_wp[li] = qkv_w[li].data_ptr<uint8_t>();
        qkv_sp[li] = reinterpret_cast<const __half*>(qkv_s[li].data_ptr());
        o_wp[li] = o_w[li].data_ptr<uint8_t>();
        o_sp[li] = reinterpret_cast<const __half*>(o_s[li].data_ptr());
        gu_wp[li] = gu_w[li].data_ptr<uint8_t>();
        gu_sp[li] = reinterpret_cast<const __half*>(gu_s[li].data_ptr());
        down_wp[li] = down_w[li].data_ptr<uint8_t>();
        down_sp[li] = reinterpret_cast<const __half*>(down_s[li].data_ptr());
        in_np[li] = reinterpret_cast<const __half*>(in_norm_w[li].data_ptr());
        post_np[li] = reinterpret_cast<const __half*>(post_norm_w[li].data_ptr());
        q_np[li] = reinterpret_cast<const __half*>(q_norm_w[li].data_ptr());
        k_np[li] = reinterpret_cast<const __half*>(k_norm_w[li].data_ptr());
        kc_p[li] = k_caches[li].data_ptr<uint8_t>();
        vc_p[li] = v_caches[li].data_ptr<uint8_t>();
    }

    // Pre-extract final norm / lm_head pointers
    auto* fn_p = reinterpret_cast<const __half*>(final_norm_w.data_ptr());
    auto* nb_p = reinterpret_cast<__half*>(normed_buf.data_ptr());
    auto* lmw_p = lm_w.data_ptr<uint8_t>();
    auto* lms_p = reinterpret_cast<const __half*>(lm_s.data_ptr());
    auto* lmb_p = reinterpret_cast<__half*>(g_full.lm_buf.data_ptr());

    // Process first token (hidden state already computed from embedding in Python)
    auto h_tensor = first_token_hidden;

    for (int64_t t = 0; t < num_tokens; t++) {
        // Copy input
        hipMemcpyAsync(h, h_tensor.data_ptr(),
            D * sizeof(__half), hipMemcpyDeviceToDevice, stream);

        int64_t pos = start_pos + t;
        int seq_len = pos + 1;

        const __half* cos_pos = reinterpret_cast<const __half*>(rope_cos.data_ptr()) + pos * half_D;
        const __half* sin_pos = reinterpret_cast<const __half*>(rope_sin.data_ptr()) + pos * half_D;

        // First layer: RMSNorm + FWHT → rot (multi-block parallel)
        launch_rmsnorm_fwht(h, in_np[0], rot, D, rms_eps, stream);

        int inter = intermediate_size;

        for (int li = 0; li < num_layers; li++) {
            int Nqkv = qkv_N[li], Kqkv = D;
            int No = o_N[li], Ko = D;
            int Ngu = gu_N[li], Kgu = D;
            int Nd = down_N[li], Kd = intermediate_size;

            launch_gemv(rot, qkv_wp[li], qkv_sp[li],
                qkv_buf, Nqkv, Kqkv, stream);

            __half* q_ptr = qkv_buf;
            __half* k_ptr = qkv_buf + q_dim;
            __half* v_ptr = qkv_buf + q_dim + kv_dim;

            hipLaunchKernelGGL(head_rmsnorm_rope_kv, dim3(num_heads + num_kv_heads), dim3(32), 0, stream,
                q_ptr, k_ptr, v_ptr, q_np[li], k_np[li],
                q_ptr, k_ptr, kc_p[li], vc_p[li],
                cos_pos, sin_pos,
                num_heads, num_kv_heads, head_dim, rms_eps, max_seq, pos);

            {
                auto* p_max = reinterpret_cast<float*>(g_ds.fa_partial_max.data_ptr());
                auto* p_sum = reinterpret_cast<float*>(g_ds.fa_partial_sum.data_ptr());
                auto* p_out = reinterpret_cast<float*>(g_ds.fa_partial_out.data_ptr());
                launch_flash_decode(qkv_buf, kc_p[li], vc_p[li], rot,
                    p_max, p_sum, p_out,
                    (int)num_heads, (int)num_kv_heads, max_seq, seq_len, (int)head_dim, stream);
                launch_gemv(rot, o_wp[li], o_sp[li], obuf, No, Ko, stream);
            }

            // Residual + Norm + FWHT → rot (MULTI-BLOCK PARALLEL)
            launch_res_norm_fwht(obuf, h, post_np[li], rot, r, D, rms_eps, stream);

            launch_gemv(rot, gu_wp[li], gu_sp[li], gubuf, Ngu, Kgu, stream);

            hipLaunchKernelGGL(silu_hadamard_ds, dim3((Kd + 31) / 32), dim3(32), 0, stream,
                gubuf, gubuf + inter, gubuf, had_p, Kd);
            launch_gemv(gubuf, down_wp[li], down_sp[li], dbuf, Nd, Kd, stream);

            // Residual + Norm + FWHT → rot (MULTI-BLOCK PARALLEL)
            if (li < num_layers - 1) {
                launch_res_norm_fwht(dbuf, r, in_np[li + 1], rot, h, D, rms_eps, stream);
            } else {
                launch_res_norm_fwht(dbuf, r, in_np[0], rot, h, D, rms_eps, stream);
            }
        }

        // Final norm + FWHT + LM head (multi-block parallel)
        launch_rmsnorm_fwht(h, fn_p, rot, D, rms_eps, stream);
        launch_gemv(rot, lmw_p, lms_p, lmb_p, lm_N, lm_K, stream);

        // 4. Argmax (stays on GPU)
        auto next_id = g_full.lm_buf.argmax(0);  // scalar tensor on GPU
        token_ids_gpu[t] = next_id;

        // 5. Embedding lookup for next token (GPU->GPU, no sync)
        if (t < num_tokens - 1) {
            h_tensor = embed_weight.index({next_id.unsqueeze(0)}).view({1, 1, D});
        }
    }

    // Single GPU->CPU copy at the end
    return token_ids_gpu.to(torch::kCPU);
}


// ============================================================
// Debug wrappers -- expose individual kernels to Python
// ============================================================

torch::Tensor dbg_rmsnorm(torch::Tensor X, torch::Tensor W, float eps) {
    int D = X.numel();
    auto Y = torch::empty_like(X);
    auto stream = at::hip::getCurrentHIPStream();
    int bs = 256;
    hipLaunchKernelGGL(rmsnorm_ds, dim3(1), dim3(bs), D * sizeof(float), stream,
        reinterpret_cast<const __half*>(X.data_ptr()),
        reinterpret_cast<const __half*>(W.data_ptr()),
        reinterpret_cast<__half*>(Y.data_ptr()),
        D, eps);
    return Y;
}

std::vector<torch::Tensor> dbg_head_rmsnorm(
    torch::Tensor Q, torch::Tensor K,
    torch::Tensor QW, torch::Tensor KW,
    int64_t H, int64_t Hk, int64_t D, float eps
) {
    // Q: [H*D], K: [Hk*D] flat
    auto Q_out = Q.clone();
    auto K_out = K.clone();
    auto stream = at::hip::getCurrentHIPStream();
    hipLaunchKernelGGL(head_rmsnorm, dim3(H + Hk), dim3(32), 0, stream,
        reinterpret_cast<const __half*>(Q.data_ptr()),
        reinterpret_cast<const __half*>(K.data_ptr()),
        reinterpret_cast<const __half*>(QW.data_ptr()),
        reinterpret_cast<const __half*>(KW.data_ptr()),
        reinterpret_cast<__half*>(Q_out.data_ptr()),
        reinterpret_cast<__half*>(K_out.data_ptr()),
        H, Hk, D, eps);
    return {Q_out, K_out};
}

std::vector<torch::Tensor> dbg_rope(
    torch::Tensor Q, torch::Tensor K,
    torch::Tensor COS, torch::Tensor SIN,
    int64_t H, int64_t Hk, int64_t D
) {
    auto Q_out = Q.clone();
    auto K_out = K.clone();
    auto stream = at::hip::getCurrentHIPStream();
    int max_heads = std::max((int)H, (int)Hk);
    hipLaunchKernelGGL(rope_ds, dim3(max_heads), dim3(D/2), 0, stream,
        reinterpret_cast<__half*>(Q_out.data_ptr()),
        reinterpret_cast<__half*>(K_out.data_ptr()),
        reinterpret_cast<const __half*>(COS.data_ptr()),
        reinterpret_cast<const __half*>(SIN.data_ptr()),
        H, Hk, D);
    return {Q_out, K_out};
}

std::vector<torch::Tensor> dbg_fused_res_norm(
    torch::Tensor X, torch::Tensor Res, torch::Tensor W, float eps
) {
    int D = X.numel();
    auto Y = torch::empty_like(X);
    auto ResOut = torch::empty_like(X);
    auto stream = at::hip::getCurrentHIPStream();
    hipLaunchKernelGGL(fused_res_norm, dim3(1), dim3(256), D * sizeof(float), stream,
        reinterpret_cast<const __half*>(X.data_ptr()),
        reinterpret_cast<const __half*>(Res.data_ptr()),
        reinterpret_cast<const __half*>(W.data_ptr()),
        reinterpret_cast<__half*>(Y.data_ptr()),
        reinterpret_cast<__half*>(ResOut.data_ptr()),
        D, eps);
    return {Y, ResOut};
}

torch::Tensor dbg_gemv(torch::Tensor X, torch::Tensor W, torch::Tensor S,
                        int64_t N, int64_t K) {
    auto Y = torch::empty({N}, torch::dtype(torch::kHalf).device(X.device()));
    auto stream = at::hip::getCurrentHIPStream();
    launch_gemv(reinterpret_cast<const __half*>(X.data_ptr()),
                W.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(S.data_ptr()),
                reinterpret_cast<__half*>(Y.data_ptr()),
                N, K, stream);
    return Y;
}

torch::Tensor dbg_gemm(torch::Tensor X, torch::Tensor W, torch::Tensor S,
                        int64_t M, int64_t N, int64_t K) {
    auto Y = torch::empty({M, N}, torch::dtype(torch::kHalf).device(X.device()));
    auto stream = at::hip::getCurrentHIPStream();
    launch_gemm(reinterpret_cast<const __half*>(X.data_ptr()),
                W.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(S.data_ptr()),
                reinterpret_cast<__half*>(Y.data_ptr()),
                M, N, K, stream);
    return Y;
}

// ============================================================
// Fast INT4 dequant kernel: [N, K/2] uint8 + [N, K/32, 2] FP16 → [N, K] FP16
// Asymmetric: val = (unsigned_nibble - zero_point) * scale
// S is interleaved (scale, zero_fp16) pairs.
// ============================================================
__global__ void dequant_int4_kernel(
    const uint8_t* __restrict__ packed,   // [N, K/2]
    const __half* __restrict__ scales,    // [N, K/32, 2] interleaved (scale, zero)
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
    const int block_idx = kh / 16;  // which block of 32 elements
    __half2 sz = *reinterpret_cast<const __half2*>(&scales[(n * (K / 32) + block_idx) * 2]);
    const float s = __half2float(sz.x);
    const float z = __half2float(sz.y);
    out[(int64_t)n * K + kh * 2]     = __float2half(((float)lo - z) * s);
    out[(int64_t)n * K + kh * 2 + 1] = __float2half(((float)hi - z) * s);
}

torch::Tensor dequant_int4(torch::Tensor packed, torch::Tensor scales,
                           int64_t N, int64_t K) {
    auto out = torch::empty({N, K}, torch::dtype(torch::kHalf).device(packed.device()));
    auto stream = at::hip::getCurrentHIPStream();
    int total = N * (K / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(dequant_int4_kernel, dim3(blocks), dim3(threads), 0, stream,
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        N, K);
    return out;
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_step", &int4_decode_step, "INT4 decode step (all layers in C++)");
    m.def("decode_step_logits", &int4_decode_step_logits, "INT4 decode step -> logits (fused final_norm + LM head)");
    m.def("decode_loop", &int4_decode_loop, "INT4 full decode loop (N tokens in C++)");
    m.def("decode_bench", &int4_decode_bench, "INT4 decode benchmark (N tokens in C++, zero Python overhead)");
    m.def("decode_bench_graph", &int4_decode_bench_graph, "INT4 decode benchmark with HIP Graph (eliminates launch overhead)");
    m.def("decode_profile", &int4_decode_profile, "INT4 profiled decode (1 token, returns per-category timings)");
    m.def("prefill_logits", &int4_prefill_logits, "INT4 prefill -> logits [M, vocab] (all layers in C++)");
    m.def("dbg_rmsnorm", &dbg_rmsnorm, "Debug: RMSNorm");
    m.def("dbg_head_rmsnorm", &dbg_head_rmsnorm, "Debug: Head RMSNorm");
    m.def("dbg_rope", &dbg_rope, "Debug: RoPE");
    m.def("dbg_fused_res_norm", &dbg_fused_res_norm, "Debug: Fused residual+RMSNorm");
    m.def("dbg_gemv", &dbg_gemv, "Debug: GEMV");
    m.def("dbg_gemm", &dbg_gemm, "Debug: GEMM");
    m.def("dequant_int4", &dequant_int4, "Dequant INT4 packed+scales to FP16 [N, K]");
    m.def("fp16_to_fp8", &fp16_to_fp8, "Convert FP16 tensor to FP8 E4M3 (uint8)");
}
