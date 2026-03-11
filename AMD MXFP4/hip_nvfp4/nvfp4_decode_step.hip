/*
 * NVFP4 Single Decode Step — C++ loop over all layers
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
// FP4 E2M1 and E4M3 LUTs
// ============================================================
__constant__ float FP4_LUT_DS[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// E4M3 LUT in constant memory — faster cached access than global memory pointer
__constant__ float E4M3_LUT_C[256];
static bool ds_lut_init = false;

static void ensure_lut() {
    if (ds_lut_init) return;
    float host[256];
    for (int i = 0; i < 256; i++) {
        int s = (i >> 7) & 1, e = (i >> 3) & 0xF, m = i & 0x7;
        float v;
        if (e == 0) v = ldexpf((float)m / 8.0f, -6);
        else if (e == 15 && m == 7) v = 0.0f;
        else v = ldexpf(1.0f + (float)m / 8.0f, e - 7);
        if (s) v = -v;
        host[i] = v;
    }
    hipMemcpyToSymbol(E4M3_LUT_C, host, 256 * sizeof(float));
    ds_lut_init = true;
}

// ============================================================
// GEMV V3 — Vectorized loads (uint4 128-bit), register-cached FP4 LUT
// Each warp handles BLOCK_N output rows. K dimension iterated in chunks of 32.
// Inner loop uses 128-bit loads for weights (16 bytes = 32 FP4 values at once)
// and __half2 loads for X (coalesced).
// ============================================================

// Register-cached FP4 decode: avoids __constant__ memory bank conflicts
__device__ __forceinline__ float fp4_decode(int nibble) {
    // FP4 E2M1: 0,0.5,1,1.5,2,3,4,6,0,-0.5,-1,-1.5,-2,-3,-4,-6
    constexpr float LUT[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return LUT[nibble];
}

template<int BLOCK_N>
__global__ void gemv_warp(
    const __half* __restrict__ X,
    const uint8_t* __restrict__ W,     // [N, K/2]
    const uint8_t* __restrict__ S,     // [N, K/16]
    __half* __restrict__ Y,
    int N, int K, float ts
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / 16, nit = K / 32;
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
            float s0 = E4M3_LUT_C[S[n * nsc + it * 2]];
            float s1 = E4M3_LUT_C[S[n * nsc + it * 2 + 1]];
            // Vectorized weight load: 16 bytes = 32 FP4 values
            const uint8_t* wr = W + n * Kh + it * 16;
            uint4 w128 = *reinterpret_cast<const uint4*>(wr);
            const uint8_t* wp = reinterpret_cast<const uint8_t*>(&w128);
            float dot0 = 0.0f, dot1 = 0.0f;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                uint8_t p = wp[b];
                dot0 += xv[b*2] * fp4_decode(p & 0xF) + xv[b*2+1] * fp4_decode((p>>4) & 0xF);
            }
            #pragma unroll
            for (int b = 8; b < 16; b++) {
                uint8_t p = wp[b];
                dot1 += xv[b*2] * fp4_decode(p & 0xF) + xv[b*2+1] * fp4_decode((p>>4) & 0xF);
            }
            acc[ni] += dot0 * s0 + dot1 * s1;
        }
    }
    #pragma unroll
    for (int ni = 0; ni < BLOCK_N; ni++) {
        acc[ni] *= ts;
        acc[ni] += __shfl_xor(acc[ni], 16); acc[ni] += __shfl_xor(acc[ni], 8);
        acc[ni] += __shfl_xor(acc[ni], 4);  acc[ni] += __shfl_xor(acc[ni], 2);
        acc[ni] += __shfl_xor(acc[ni], 1);
        if (lane == 0 && n0 + ni < N) Y[n0 + ni] = __float2half(acc[ni]);
    }
}

template<int BLOCK_N, int BS>
__global__ void gemv_block(
    const __half* __restrict__ X,
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
    __half* __restrict__ Y,
    int N, int K, float ts
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int tid = threadIdx.x;
    const int Kh = K / 2, nsc = K / 16, nit = K / 32;
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
            float s0 = E4M3_LUT_C[S[n * nsc + it * 2]];
            float s1 = E4M3_LUT_C[S[n * nsc + it * 2 + 1]];
            const uint8_t* wr = W + n * Kh + it * 16;
            uint4 w128 = *reinterpret_cast<const uint4*>(wr);
            const uint8_t* wp = reinterpret_cast<const uint8_t*>(&w128);
            float dot0 = 0.0f, dot1 = 0.0f;
            #pragma unroll
            for (int b = 0; b < 8; b++) { uint8_t p = wp[b]; dot0 += xv[b*2] * fp4_decode(p&0xF) + xv[b*2+1] * fp4_decode((p>>4)&0xF); }
            #pragma unroll
            for (int b = 8; b < 16; b++) { uint8_t p = wp[b]; dot1 += xv[b*2] * fp4_decode(p&0xF) + xv[b*2+1] * fp4_decode((p>>4)&0xF); }
            acc[ni] += dot0 * s0 + dot1 * s1;
        }
    }
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i++) acc[i] *= ts;
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
    const __half* X, const uint8_t* W, const uint8_t* S,
    __half* Y, int N, int K, float ts, hipStream_t st) {
    // BLOCK_N=4 for all sizes: fewer blocks = less scheduler pressure + better X reuse
    hipLaunchKernelGGL((gemv_warp<4>), dim3((N+3)/4), dim3(32), 0, st, X, W, S, Y, N, K, ts);
}

// ============================================================
// GEMV with fused AWQ scale: Y = W @ (X * AWQ_scale)
// Eliminates separate vecmul_inplace kernel launch.
// AWQ scale is applied during X loading (outside inner BLOCK_N loop).
// ============================================================
template<int BLOCK_N>
__global__ void gemv_warp_awq(
    const __half* __restrict__ X,
    const __half* __restrict__ AWQ,   // [K] per-channel AWQ inv_scale
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ S,
    __half* __restrict__ Y,
    int N, int K, float ts
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / 16, nit = K / 32;
    float acc[BLOCK_N];
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i++) acc[i] = 0.0f;

    for (int it = lane; it < nit; it += 32) {
        float xv[32];
        #pragma unroll
        for (int i = 0; i < 32; i += 2) {
            __half2 xh = *reinterpret_cast<const __half2*>(&X[it * 32 + i]);
            __half2 ah = *reinterpret_cast<const __half2*>(&AWQ[it * 32 + i]);
            xv[i]   = __half2float(xh.x) * __half2float(ah.x);
            xv[i+1] = __half2float(xh.y) * __half2float(ah.y);
        }
        #pragma unroll
        for (int ni = 0; ni < BLOCK_N; ni++) {
            int n = n0 + ni;
            if (n >= N) continue;
            float s0 = E4M3_LUT_C[S[n * nsc + it * 2]];
            float s1 = E4M3_LUT_C[S[n * nsc + it * 2 + 1]];
            const uint8_t* wr = W + n * Kh + it * 16;
            uint4 w128 = *reinterpret_cast<const uint4*>(wr);
            const uint8_t* wp = reinterpret_cast<const uint8_t*>(&w128);
            float dot0 = 0.0f, dot1 = 0.0f;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                uint8_t p = wp[b];
                dot0 += xv[b*2] * fp4_decode(p & 0xF) + xv[b*2+1] * fp4_decode((p>>4) & 0xF);
            }
            #pragma unroll
            for (int b = 8; b < 16; b++) {
                uint8_t p = wp[b];
                dot1 += xv[b*2] * fp4_decode(p & 0xF) + xv[b*2+1] * fp4_decode((p>>4) & 0xF);
            }
            acc[ni] += dot0 * s0 + dot1 * s1;
        }
    }
    #pragma unroll
    for (int ni = 0; ni < BLOCK_N; ni++) {
        acc[ni] *= ts;
        acc[ni] += __shfl_xor(acc[ni], 16); acc[ni] += __shfl_xor(acc[ni], 8);
        acc[ni] += __shfl_xor(acc[ni], 4);  acc[ni] += __shfl_xor(acc[ni], 2);
        acc[ni] += __shfl_xor(acc[ni], 1);
        if (lane == 0 && n0 + ni < N) Y[n0 + ni] = __float2half(acc[ni]);
    }
}

static inline __attribute__((always_inline)) void launch_gemv_awq(
    const __half* X, const __half* AWQ,
    const uint8_t* W, const uint8_t* S,
    __half* Y, int N, int K, float ts, hipStream_t st) {
    hipLaunchKernelGGL((gemv_warp_awq<4>), dim3((N+3)/4), dim3(32), 0, st, X, AWQ, W, S, Y, N, K, ts);
}

// ============================================================
// Fused SiLU-input GEMV with AWQ: Y = W @ (SiLU(Gate) * Up * AWQ_scale)
// Eliminates separate silu_mul_ds + vecmul_inplace kernel launches.
// ============================================================
template<int BLOCK_N>
__global__ void gemv_warp_silu_in_awq(
    const __half* __restrict__ Gate,
    const __half* __restrict__ Up,
    const __half* __restrict__ AWQ,   // [K] per-channel AWQ inv_scale (nullptr if no AWQ)
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ S,
    __half* __restrict__ Y,
    int N, int K, float ts
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / 16, nit = K / 32;
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
            float x0 = g0 / (1.0f + expf(-g0)) * u0;
            float x1 = g1 / (1.0f + expf(-g1)) * u1;
            if (AWQ) {
                __half2 ah = *reinterpret_cast<const __half2*>(&AWQ[it * 32 + i]);
                x0 *= __half2float(ah.x);
                x1 *= __half2float(ah.y);
            }
            xv[i]   = x0;
            xv[i+1] = x1;
        }
        #pragma unroll
        for (int ni = 0; ni < BLOCK_N; ni++) {
            int n = n0 + ni;
            if (n >= N) continue;
            float s0 = E4M3_LUT_C[S[n * nsc + it * 2]];
            float s1 = E4M3_LUT_C[S[n * nsc + it * 2 + 1]];
            const uint8_t* wr = W + n * Kh + it * 16;
            uint4 w128 = *reinterpret_cast<const uint4*>(wr);
            const uint8_t* wp = reinterpret_cast<const uint8_t*>(&w128);
            float dot0 = 0.0f, dot1 = 0.0f;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                uint8_t p = wp[b];
                dot0 += xv[b*2] * fp4_decode(p & 0xF) + xv[b*2+1] * fp4_decode((p>>4) & 0xF);
            }
            #pragma unroll
            for (int b = 8; b < 16; b++) {
                uint8_t p = wp[b];
                dot1 += xv[b*2] * fp4_decode(p & 0xF) + xv[b*2+1] * fp4_decode((p>>4) & 0xF);
            }
            acc[ni] += dot0 * s0 + dot1 * s1;
        }
    }
    #pragma unroll
    for (int ni = 0; ni < BLOCK_N; ni++) {
        acc[ni] *= ts;
        acc[ni] += __shfl_xor(acc[ni], 16); acc[ni] += __shfl_xor(acc[ni], 8);
        acc[ni] += __shfl_xor(acc[ni], 4);  acc[ni] += __shfl_xor(acc[ni], 2);
        acc[ni] += __shfl_xor(acc[ni], 1);
        if (lane == 0 && n0 + ni < N) Y[n0 + ni] = __float2half(acc[ni]);
    }
}

static inline __attribute__((always_inline)) void launch_gemv_silu_in_awq(
    const __half* Gate, const __half* Up, const __half* AWQ,
    const uint8_t* W, const uint8_t* S,
    __half* Y, int N, int K, float ts, hipStream_t st) {
    hipLaunchKernelGGL((gemv_warp_silu_in_awq<4>), dim3((N+3)/4), dim3(32), 0, st, Gate, Up, AWQ, W, S, Y, N, K, ts);
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
    const uint8_t* __restrict__ W,       // [N, K/2] FP4 weights
    const uint8_t* __restrict__ S,       // [N, K/16] E4M3 scales
    __half* __restrict__ Y,              // [N] output
    int N, int K, float ts, float rms_eps
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / 16, nit = K / 32;

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
            float s0 = E4M3_LUT_C[S[n * nsc + it * 2]];
            float s1 = E4M3_LUT_C[S[n * nsc + it * 2 + 1]];
            const uint8_t* wr = W + n * Kh + it * 16;
            uint4 w128 = *reinterpret_cast<const uint4*>(wr);
            const uint8_t* wp = reinterpret_cast<const uint8_t*>(&w128);
            float dot0 = 0.0f, dot1 = 0.0f;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                uint8_t p = wp[b];
                dot0 += xv[b*2] * fp4_decode(p & 0xF) + xv[b*2+1] * fp4_decode((p>>4) & 0xF);
            }
            #pragma unroll
            for (int b = 8; b < 16; b++) {
                uint8_t p = wp[b];
                dot1 += xv[b*2] * fp4_decode(p & 0xF) + xv[b*2+1] * fp4_decode((p>>4) & 0xF);
            }
            acc[ni] += dot0 * s0 + dot1 * s1;
        }
    }
    #pragma unroll
    for (int ni = 0; ni < BLOCK_N; ni++) {
        acc[ni] *= ts;
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
    const uint8_t* __restrict__ W,       // [N, K/2] FP4 weights
    const uint8_t* __restrict__ S,       // [N, K/16] E4M3 scales
    __half* __restrict__ Y,              // [N] output
    int N, int K, float ts, float rms_eps
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / 16, nit = K / 32;

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
            float s0 = E4M3_LUT_C[S[n * nsc + it * 2]];
            float s1 = E4M3_LUT_C[S[n * nsc + it * 2 + 1]];
            const uint8_t* wr = W + n * Kh + it * 16;
            uint4 w128 = *reinterpret_cast<const uint4*>(wr);
            const uint8_t* wp = reinterpret_cast<const uint8_t*>(&w128);
            float dot0 = 0.0f, dot1 = 0.0f;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                uint8_t p = wp[b];
                dot0 += xv[b*2] * fp4_decode(p & 0xF) + xv[b*2+1] * fp4_decode((p>>4) & 0xF);
            }
            #pragma unroll
            for (int b = 8; b < 16; b++) {
                uint8_t p = wp[b];
                dot1 += xv[b*2] * fp4_decode(p & 0xF) + xv[b*2+1] * fp4_decode((p>>4) & 0xF);
            }
            acc[ni] += dot0 * s0 + dot1 * s1;
        }
    }
    #pragma unroll
    for (int ni = 0; ni < BLOCK_N; ni++) {
        acc[ni] *= ts;
        acc[ni] += __shfl_xor(acc[ni], 16); acc[ni] += __shfl_xor(acc[ni], 8);
        acc[ni] += __shfl_xor(acc[ni], 4);  acc[ni] += __shfl_xor(acc[ni], 2);
        acc[ni] += __shfl_xor(acc[ni], 1);
        if (lane == 0 && n0 + ni < N) Y[n0 + ni] = __float2half(acc[ni]);
    }
}

static inline __attribute__((always_inline)) void launch_gemv_norm(
    const __half* X, const __half* NormW,
    const uint8_t* W, const uint8_t* S,
    __half* Y, int N, int K, float ts, float rms_eps, hipStream_t st) {
    if (N <= 2048) {
        hipLaunchKernelGGL((gemv_warp_norm<4>), dim3((N+3)/4), dim3(32), 0, st, X, NormW, W, S, Y, N, K, ts, rms_eps);
    } else if (N <= 16384) {
        hipLaunchKernelGGL((gemv_warp_norm<2>), dim3((N+1)/2), dim3(32), 0, st, X, NormW, W, S, Y, N, K, ts, rms_eps);
    } else {
        hipLaunchKernelGGL((gemv_warp_norm<4>), dim3((N+3)/4), dim3(32), 0, st, X, NormW, W, S, Y, N, K, ts, rms_eps);
    }
}

static inline __attribute__((always_inline)) void launch_gemv_res_norm(
    const __half* A, const __half* B, __half* ResOut, const __half* NormW,
    const uint8_t* W, const uint8_t* S,
    __half* Y, int N, int K, float ts, float rms_eps, hipStream_t st) {
    if (N <= 2048) {
        hipLaunchKernelGGL((gemv_warp_res_norm<4>), dim3((N+3)/4), dim3(32), 0, st, A, B, ResOut, NormW, W, S, Y, N, K, ts, rms_eps);
    } else if (N <= 16384) {
        hipLaunchKernelGGL((gemv_warp_res_norm<2>), dim3((N+1)/2), dim3(32), 0, st, A, B, ResOut, NormW, W, S, Y, N, K, ts, rms_eps);
    } else {
        hipLaunchKernelGGL((gemv_warp_res_norm<4>), dim3((N+3)/4), dim3(32), 0, st, A, B, ResOut, NormW, W, S, Y, N, K, ts, rms_eps);
    }
}

// ============================================================
// Fused kernels
// ============================================================

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

// AWQ: in-place elementwise multiply X[i] *= S[i]
__global__ void vecmul_inplace(__half* X, const __half* S, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        X[i] = __float2half(__half2float(X[i]) * __half2float(S[i]));
    }
}

// Legacy decode attention (kept for reference, replaced by flash_decode below)
__global__ void decode_attn_legacy(
    const __half* __restrict__ Q,        // [H*D] flat (after RMSNorm+RoPE)
    const __half* __restrict__ K_cache,  // [Hk, max_seq, D]
    const __half* __restrict__ V_cache,  // [Hk, max_seq, D]
    __half* __restrict__ O,              // [H*D] flat
    int H, int Hk, int max_seq, int seq_len, int D
) {
    const int qh = blockIdx.x;
    if (qh >= H) return;
    const int kh = qh / (H / Hk);
    const int lane = threadIdx.x;
    const float scale = rsqrtf((float)D);

    const __half* q_row = Q + qh * D;
    float q_reg[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        q_reg[i] = __half2float(q_row[lane + i * 32]);

    const __half* k_base = K_cache + kh * max_seq * D;
    const __half* v_base = V_cache + kh * max_seq * D;

    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float out_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = 0; s < seq_len; s++) {
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            dot += q_reg[i] * __half2float(k_base[s * D + lane + i * 32]);
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
            out_reg[i] = out_reg[i] * correction + exp_score * __half2float(v_base[s * D + lane + i * 32]);
        max_score = new_max;
    }

    float inv = 1.0f / sum_exp;
    __half* o_row = O + qh * D;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        o_row[lane + i * 32] = __float2half(out_reg[i] * inv);
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
    const __half* __restrict__ K_cache,   // [Hk, max_seq, D]
    const __half* __restrict__ V_cache,   // [Hk, max_seq, D]
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

    // Range for this split
    const int s_start = split_id * BLOCK_S;
    const int s_end = min(s_start + BLOCK_S, seq_len);

    const __half* k_base = K_cache + (long)kh * max_seq * D;
    const __half* v_base = V_cache + (long)kh * max_seq * D;

    // Online softmax accumulators for this chunk
    float local_max = -1e30f;
    float local_sum = 0.0f;
    float out_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = s_start; s < s_end; s++) {
        // Q·K dot product
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            dot += q_reg[i] * __half2float(k_base[(long)s * D + lane + i * 32]);

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
                         exp_score * __half2float(v_base[(long)s * D + lane + i * 32]);
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
    #pragma unroll
    for (int i = 0; i < 4; i++)
        o_row[lane + i * 32] = __float2half(out_reg[i] * inv);
}

// Fused single-split attention (for small seq_len <= FLASH_DECODE_BLOCK_S)
// Same as legacy but with GQA grouping for better occupancy
// Grid: (num_kv_heads), Block: (32, gqa_ratio)
// flash_decode_fused: 1 block per Q head (not per KV head)
// Grid: (num_heads), Block: (32)
// 40 blocks on 48 CUs = 83% CU utilization (was 8 blocks = 17%)
__global__ void flash_decode_fused(
    const __half* __restrict__ Q,
    const __half* __restrict__ K_cache,
    const __half* __restrict__ V_cache,
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

    const __half* k_base = K_cache + (long)kh * max_seq * D;
    const __half* v_base = V_cache + (long)kh * max_seq * D;

    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float out_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = 0; s < seq_len; s++) {
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            dot += q_reg[i] * __half2float(k_base[(long)s * D + lane + i * 32]);

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
                         exp_score * __half2float(v_base[(long)s * D + lane + i * 32]);
        max_score = new_max;
    }

    float inv = 1.0f / sum_exp;
    __half* o_row = O + qh * D;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        o_row[lane + i * 32] = __float2half(out_reg[i] * inv);
}

// Dispatch flash_decode_partial with optimal BLOCK_S based on context length.
// Larger BLOCK_S = fewer splits = fewer blocks = less scheduler overhead.
// Caps splits at ~16-32 max regardless of context length.
static inline void launch_flash_decode(
    const __half* Q, const __half* K_cache, const __half* V_cache,
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

// KV cache write
__global__ void kv_write(const __half* src, __half* cache, int Hk, int max_seq, int D, int pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Hk * D) {
        int h = idx / D, d = idx % D;
        cache[h * max_seq * D + pos * D + d] = src[idx];
    }
}

// Fused KV cache write — K and V in one kernel launch
__global__ void kv_write_fused(const __half* K, const __half* V,
                                __half* k_cache, __half* v_cache,
                                int Hk, int max_seq, int D, int pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Hk * D) {
        int h = idx / D, d = idx % D;
        int off = h * max_seq * D + pos * D + d;
        k_cache[off] = K[idx];
        v_cache[off] = V[idx];
    }
}

// Fused head RMSNorm + RoPE + KV cache write (saves 2 kernel launches per layer)
// Blocks 0..H-1: Q heads (RMSNorm + RoPE)
// Blocks H..H+Hk-1: K heads (RMSNorm + RoPE + write to K cache + copy V to V cache)
__global__ void head_rmsnorm_rope_kv(
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
    // Process K heads + KV cache write
    int kid = pid - H;
    if (kid >= 0 && kid < Hk) {
        const __half* src = K + kid * D;
        __half* dst = K_out + kid * D;
        int cache_off = kid * max_seq * D + pos * D;

        // K: RMSNorm + RoPE + cache write
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
            __half k_lo = __float2half(ne * c - no * s);
            __half k_hi = __float2half(no * c + ne * s);
            dst[i] = k_lo;
            dst[i + half_D] = k_hi;
            K_cache[cache_off + i] = k_lo;
            K_cache[cache_off + i + half_D] = k_hi;
        }

        // V: raw copy to cache
        for (int i = tid; i < D; i += 32) {
            V_cache[cache_off + i] = V[kid * D + i];
        }
    }
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
    const uint8_t* __restrict__ S,     // [N, K/16]
    __half* __restrict__ Y,            // [M, N]
    int M, int N, int K, float ts
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int m = blockIdx.y;
    if (m >= M) return;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / 16, nit = K / 32;
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
            float s0 = E4M3_LUT_C[S[n * nsc + it * 2]];
            float s1 = E4M3_LUT_C[S[n * nsc + it * 2 + 1]];
            const uint8_t* wr = W + n * Kh + it * 16;
            float dot0 = 0.0f, dot1 = 0.0f;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                uint8_t p = wr[b];
                dot0 += xv[b*2] * FP4_LUT_DS[p & 0xF] + xv[b*2+1] * FP4_LUT_DS[(p>>4) & 0xF];
            }
            #pragma unroll
            for (int b = 8; b < 16; b++) {
                uint8_t p = wr[b];
                dot1 += xv[b*2] * FP4_LUT_DS[p & 0xF] + xv[b*2+1] * FP4_LUT_DS[(p>>4) & 0xF];
            }
            acc[ni] += dot0 * s0 + dot1 * s1;
        }
    }
    #pragma unroll
    for (int ni = 0; ni < BLOCK_N; ni++) {
        acc[ni] *= ts;
        acc[ni] += __shfl_xor(acc[ni], 16); acc[ni] += __shfl_xor(acc[ni], 8);
        acc[ni] += __shfl_xor(acc[ni], 4);  acc[ni] += __shfl_xor(acc[ni], 2);
        acc[ni] += __shfl_xor(acc[ni], 1);
        if (lane == 0 && n0 + ni < N) Y[(int64_t)m * N + n0 + ni] = __float2half(acc[ni]);
    }
}

// GEMM with fused AWQ: Y[m,:] = W @ (X[m,:] * AWQ)
template<int BLOCK_N>
__global__ void gemm_warp_awq(
    const __half* __restrict__ X,      // [M, K]
    const __half* __restrict__ AWQ,    // [K]
    const uint8_t* __restrict__ W,     // [N, K/2]
    const uint8_t* __restrict__ S,     // [N, K/16]
    __half* __restrict__ Y,            // [M, N]
    int M, int N, int K, float ts
) {
    const int n0 = blockIdx.x * BLOCK_N;
    const int m = blockIdx.y;
    if (m >= M) return;
    const int lane = threadIdx.x;
    const int Kh = K / 2, nsc = K / 16, nit = K / 32;
    float acc[BLOCK_N];
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i++) acc[i] = 0.0f;

    const __half* X_row = X + (int64_t)m * K;

    for (int it = lane; it < nit; it += 32) {
        float xv[32];
        #pragma unroll
        for (int i = 0; i < 32; i += 2) {
            __half2 xh = *reinterpret_cast<const __half2*>(&X_row[it * 32 + i]);
            __half2 ah = *reinterpret_cast<const __half2*>(&AWQ[it * 32 + i]);
            xv[i]   = __half2float(xh.x) * __half2float(ah.x);
            xv[i+1] = __half2float(xh.y) * __half2float(ah.y);
        }
        #pragma unroll
        for (int ni = 0; ni < BLOCK_N; ni++) {
            int n = n0 + ni;
            if (n >= N) continue;
            float s0 = E4M3_LUT_C[S[n * nsc + it * 2]];
            float s1 = E4M3_LUT_C[S[n * nsc + it * 2 + 1]];
            const uint8_t* wr = W + n * Kh + it * 16;
            float dot0 = 0.0f, dot1 = 0.0f;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                uint8_t p = wr[b];
                dot0 += xv[b*2] * FP4_LUT_DS[p & 0xF] + xv[b*2+1] * FP4_LUT_DS[(p>>4) & 0xF];
            }
            #pragma unroll
            for (int b = 8; b < 16; b++) {
                uint8_t p = wr[b];
                dot1 += xv[b*2] * FP4_LUT_DS[p & 0xF] + xv[b*2+1] * FP4_LUT_DS[(p>>4) & 0xF];
            }
            acc[ni] += dot0 * s0 + dot1 * s1;
        }
    }
    #pragma unroll
    for (int ni = 0; ni < BLOCK_N; ni++) {
        acc[ni] *= ts;
        acc[ni] += __shfl_xor(acc[ni], 16); acc[ni] += __shfl_xor(acc[ni], 8);
        acc[ni] += __shfl_xor(acc[ni], 4);  acc[ni] += __shfl_xor(acc[ni], 2);
        acc[ni] += __shfl_xor(acc[ni], 1);
        if (lane == 0 && n0 + ni < N) Y[(int64_t)m * N + n0 + ni] = __float2half(acc[ni]);
    }
}

// GEMM launch helper
static inline void launch_gemm(
    const __half* X, const uint8_t* W, const uint8_t* S,
    __half* Y, int M, int N, int K, float ts, hipStream_t st) {
    if (N <= 2048) {
        hipLaunchKernelGGL((gemm_warp<4>), dim3((N+3)/4, M), dim3(32), 0, st, X, W, S, Y, M, N, K, ts);
    } else if (N <= 16384) {
        hipLaunchKernelGGL((gemm_warp<2>), dim3((N+1)/2, M), dim3(32), 0, st, X, W, S, Y, M, N, K, ts);
    } else {
        hipLaunchKernelGGL((gemm_warp<2>), dim3((N+1)/2, M), dim3(32), 0, st, X, W, S, Y, M, N, K, ts);
    }
}

static inline void launch_gemm_awq(
    const __half* X, const __half* AWQ,
    const uint8_t* W, const uint8_t* S,
    __half* Y, int M, int N, int K, float ts, hipStream_t st) {
    if (N <= 2048) {
        hipLaunchKernelGGL((gemm_warp_awq<4>), dim3((N+3)/4, M), dim3(32), 0, st, X, AWQ, W, S, Y, M, N, K, ts);
    } else if (N <= 16384) {
        hipLaunchKernelGGL((gemm_warp_awq<2>), dim3((N+1)/2, M), dim3(32), 0, st, X, AWQ, W, S, Y, M, N, K, ts);
    } else {
        hipLaunchKernelGGL((gemm_warp_awq<2>), dim3((N+1)/2, M), dim3(32), 0, st, X, AWQ, W, S, Y, M, N, K, ts);
    }
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

// Batched AWQ vecmul: X[i] *= S[i]
__global__ void vecmul_batched(__half* X, const __half* S, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (idx < N) {
        int off = (int64_t)row * stride + idx;
        X[off] = __float2half(__half2float(X[off]) * __half2float(S[idx]));
    }
}

// ============================================================
// Prefill step — all layers in C++, returns logits [M, vocab]
// ============================================================
torch::Tensor nvfp4_prefill_logits(
    torch::Tensor input_ids,              // [M] long — token IDs
    torch::Tensor embed_weight,           // [vocab, D] FP16
    torch::Tensor final_norm_w,           // [D] FP16
    torch::Tensor lm_w, torch::Tensor lm_s, float lm_ts,
    int64_t lm_N, int64_t lm_K,
    // Per-layer weights
    std::vector<torch::Tensor> qkv_w, std::vector<torch::Tensor> qkv_s, std::vector<float> qkv_ts,
    std::vector<int64_t> qkv_N,
    std::vector<torch::Tensor> o_w, std::vector<torch::Tensor> o_s, std::vector<float> o_ts,
    std::vector<int64_t> o_N,
    std::vector<torch::Tensor> gu_w, std::vector<torch::Tensor> gu_s, std::vector<float> gu_ts,
    std::vector<int64_t> gu_N,
    std::vector<torch::Tensor> down_w, std::vector<torch::Tensor> down_s, std::vector<float> down_ts,
    std::vector<int64_t> down_N,
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w, std::vector<torch::Tensor> k_norm_w,
    std::vector<torch::Tensor> o_awq_inv, std::vector<torch::Tensor> down_awq_inv,
    torch::Tensor rope_cos, torch::Tensor rope_sin,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    float rms_eps
) {
    c10::InferenceMode guard;
    ensure_lut();
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
    bool has_awq = !o_awq_inv.empty();

    // Allocate buffers [M, ...]
    auto x = torch::embedding(embed_weight, input_ids);  // [M, D]
    auto normed = torch::empty({M, D}, opts);
    auto residual = torch::empty({M, D}, opts);
    auto qkv_buf = torch::empty({M, qkv_dim}, opts);
    auto o_buf = torch::empty({M, D}, opts);
    auto gu_buf = torch::empty({M, 2 * inter}, opts);
    auto silu_buf = torch::empty({M, inter}, opts);  // contiguous SiLU output for down_proj input
    auto down_buf = torch::empty({M, D}, opts);

    auto* x_p = reinterpret_cast<__half*>(x.data_ptr());
    auto* n_p = reinterpret_cast<__half*>(normed.data_ptr());
    auto* r_p = reinterpret_cast<__half*>(residual.data_ptr());
    auto* qkv_p = reinterpret_cast<__half*>(qkv_buf.data_ptr());
    auto* o_p = reinterpret_cast<__half*>(o_buf.data_ptr());
    auto* gu_p = reinterpret_cast<__half*>(gu_buf.data_ptr());
    auto* silu_p = reinterpret_cast<__half*>(silu_buf.data_ptr());
    auto* d_p = reinterpret_cast<__half*>(down_buf.data_ptr());

    // Pre-extract pointers
    std::vector<const uint8_t*> qkv_wp(num_layers), qkv_sp(num_layers);
    std::vector<const uint8_t*> o_wp(num_layers), o_sp(num_layers);
    std::vector<const uint8_t*> gu_wp(num_layers), gu_sp(num_layers);
    std::vector<const uint8_t*> down_wp(num_layers), down_sp(num_layers);
    std::vector<const __half*> in_np(num_layers), post_np(num_layers);
    std::vector<const __half*> q_np(num_layers), k_np(num_layers);
    std::vector<const __half*> o_awq_p(num_layers, nullptr), down_awq_p(num_layers, nullptr);
    for (int li = 0; li < num_layers; li++) {
        qkv_wp[li] = qkv_w[li].data_ptr<uint8_t>();
        qkv_sp[li] = qkv_s[li].data_ptr<uint8_t>();
        o_wp[li] = o_w[li].data_ptr<uint8_t>();
        o_sp[li] = o_s[li].data_ptr<uint8_t>();
        gu_wp[li] = gu_w[li].data_ptr<uint8_t>();
        gu_sp[li] = gu_s[li].data_ptr<uint8_t>();
        down_wp[li] = down_w[li].data_ptr<uint8_t>();
        down_sp[li] = down_s[li].data_ptr<uint8_t>();
        in_np[li] = reinterpret_cast<const __half*>(in_norm_w[li].data_ptr());
        post_np[li] = reinterpret_cast<const __half*>(post_norm_w[li].data_ptr());
        q_np[li] = reinterpret_cast<const __half*>(q_norm_w[li].data_ptr());
        k_np[li] = reinterpret_cast<const __half*>(k_norm_w[li].data_ptr());
        if (has_awq) {
            o_awq_p[li] = reinterpret_cast<const __half*>(o_awq_inv[li].data_ptr());
            down_awq_p[li] = reinterpret_cast<const __half*>(down_awq_inv[li].data_ptr());
        }
    }

    const __half* cos_p = reinterpret_cast<const __half*>(rope_cos.data_ptr());
    const __half* sin_p = reinterpret_cast<const __half*>(rope_sin.data_ptr());

    // Layer loop
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

        // 2. QKV GEMM
        launch_gemm(n_p, qkv_wp[li], qkv_sp[li], qkv_p, M, Nqkv, Kqkv, qkv_ts[li], stream);

        // 3. Head RMSNorm + RoPE (batched, in-place on qkv_buf)
        hipLaunchKernelGGL(head_rmsnorm_rope_prefill,
            dim3(num_heads + num_kv_heads, M), dim3(32), 0, stream,
            qkv_p, q_np[li], k_np[li],
            cos_p, sin_p,
            M, num_heads, num_kv_heads, head_dim, rms_eps, half_D, qkv_dim);

        // 4. Causal self-attention via PyTorch SDPA
        // Reshape Q [M, H, D] -> [1, H, M, D], K [M, Hk, D] -> [1, Hk, M, D]
        {
            // qkv_buf layout: [M, q_dim + kv_dim + kv_dim]
            // We need to extract Q, K, V and reshape for SDPA
            auto q_t = qkv_buf.narrow(1, 0, q_dim).view({M, (long)num_heads, (long)head_dim})
                            .permute({1, 0, 2}).unsqueeze(0);       // [1, H, M, D]
            auto k_t = qkv_buf.narrow(1, q_dim, kv_dim).view({M, (long)num_kv_heads, (long)head_dim})
                            .permute({1, 0, 2}).unsqueeze(0);       // [1, Hk, M, D]
            auto v_t = qkv_buf.narrow(1, q_dim + kv_dim, kv_dim).view({M, (long)num_kv_heads, (long)head_dim})
                            .permute({1, 0, 2}).unsqueeze(0);       // [1, Hk, M, D]

            auto attn = at::scaled_dot_product_attention(q_t, k_t, v_t,
                /*attn_mask=*/c10::nullopt, /*dropout_p=*/0.0,
                /*is_causal=*/true, /*scale=*/c10::nullopt, /*enable_gqa=*/true);
            // attn: [1, H, M, D] -> [M, H*D]
            auto attn_out = attn.squeeze(0).permute({1, 0, 2}).reshape({M, q_dim});

            // 5. AWQ + O projection
            if (has_awq) {
                // Batched vecmul
                hipLaunchKernelGGL(vecmul_batched, dim3((q_dim+255)/256, M), dim3(256), 0, stream,
                    reinterpret_cast<__half*>(attn_out.data_ptr()),
                    o_awq_p[li], q_dim, q_dim);
                launch_gemm(reinterpret_cast<const __half*>(attn_out.data_ptr()),
                    o_wp[li], o_sp[li], o_p, M, No, Ko, o_ts[li], stream);
            } else {
                launch_gemm(reinterpret_cast<const __half*>(attn_out.data_ptr()),
                    o_wp[li], o_sp[li], o_p, M, No, Ko, o_ts[li], stream);
            }
        }

        // 6. Fused residual + post_norm (batched)
        hipLaunchKernelGGL(fused_res_norm_batched, dim3(M), dim3(norm_bs), smem_norm, stream,
            o_p, x_p, post_np[li], n_p, r_p, D, rms_eps);

        // 7. Gate+Up GEMM
        launch_gemm(n_p, gu_wp[li], gu_sp[li], gu_p, M, Ngu, Kgu, gu_ts[li], stream);

        // 8. SiLU(gate) * up → contiguous silu_buf [M, inter]
        hipLaunchKernelGGL(silu_mul_batched, dim3((inter+255)/256, M), dim3(256), 0, stream,
            gu_p, gu_p + inter, silu_p, inter, 2 * inter, inter);

        // 9. AWQ + Down GEMM (silu_buf is contiguous [M, inter])
        if (has_awq) {
            hipLaunchKernelGGL(vecmul_batched, dim3((inter+255)/256, M), dim3(256), 0, stream,
                silu_p, down_awq_p[li], inter, inter);
        }
        launch_gemm(silu_p, down_wp[li], down_sp[li], d_p, M, Nd, Kd, down_ts[li], stream);

        // 10. Fused residual + next input_norm (batched)
        if (li < num_layers - 1) {
            hipLaunchKernelGGL(fused_res_norm_batched, dim3(M), dim3(norm_bs), smem_norm, stream,
                d_p, r_p, in_np[li + 1], n_p, x_p, D, rms_eps);
        } else {
            // Last layer: just residual add (store in x_p)
            hipLaunchKernelGGL(fused_res_norm_batched, dim3(M), dim3(norm_bs), smem_norm, stream,
                d_p, r_p, in_np[0], n_p, x_p, D, rms_eps);
            // x_p now = residual, n_p = normed with wrong weights. Redo final norm:
        }
    }

    // Final RMSNorm
    auto* fn_p = reinterpret_cast<const __half*>(final_norm_w.data_ptr());
    hipLaunchKernelGGL(rmsnorm_batched, dim3(M), dim3(norm_bs), smem_norm, stream,
        x_p, fn_p, n_p, D, rms_eps);

    // LM head GEMM: [M, D] -> [M, vocab]
    auto logits = torch::empty({M, lm_N}, opts);
    launch_gemm(n_p, lm_w.data_ptr<uint8_t>(), lm_s.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(logits.data_ptr()), M, lm_N, lm_K, lm_ts, stream);

    return logits;  // [M, vocab]
}

// ============================================================
// Decode step context
// ============================================================
struct DSContext {
    torch::Tensor normed, hidden, residual;
    torch::Tensor qkv_buf, o_buf, attn_buf;
    torch::Tensor gu_buf, silu_buf, down_buf;
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
torch::Tensor nvfp4_decode_step(
    torch::Tensor hidden_in,          // [1, 1, D] current hidden state
    // Per-layer weights (vectors of tensors, pre-transposed [N, K/2] etc.)
    std::vector<torch::Tensor> qkv_w, std::vector<torch::Tensor> qkv_s, std::vector<float> qkv_ts,
    std::vector<int64_t> qkv_N,
    std::vector<torch::Tensor> o_w, std::vector<torch::Tensor> o_s, std::vector<float> o_ts,
    std::vector<int64_t> o_N,
    std::vector<torch::Tensor> gu_w, std::vector<torch::Tensor> gu_s, std::vector<float> gu_ts,
    std::vector<int64_t> gu_N,
    std::vector<torch::Tensor> down_w, std::vector<torch::Tensor> down_s, std::vector<float> down_ts,
    std::vector<int64_t> down_N,
    // Norm weights (AWQ already absorbed for qkv/gate_up)
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w, std::vector<torch::Tensor> k_norm_w,
    // AWQ inv_scale for o_proj and down_proj (1/awq_scale, per-channel)
    std::vector<torch::Tensor> o_awq_inv, std::vector<torch::Tensor> down_awq_inv,
    // RoPE
    torch::Tensor rope_cos, torch::Tensor rope_sin,  // [max_seq, D/2]
    int64_t pos,
    // Model config
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    float rms_eps,
    // KV caches: [num_kv_heads, max_seq, head_dim]
    std::vector<torch::Tensor> k_caches, std::vector<torch::Tensor> v_caches
) {
    c10::InferenceMode guard;
    ensure_lut();
    int num_layers = qkv_w.size();
    int D = hidden_size;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int max_seq = k_caches[0].size(2);  // [1, Hk, max_seq, D]
    int seq_len = pos + 1;
    auto device = hidden_in.device();
    auto stream = at::hip::getCurrentHIPStream();

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
        // Flash decode attention partial buffers
        auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
        g_ds.fa_partial_max = torch::empty({fa_max_splits, (long)num_heads}, opts_f32);
        g_ds.fa_partial_sum = torch::empty({fa_max_splits, (long)num_heads}, opts_f32);
        g_ds.fa_partial_out = torch::empty({fa_max_splits, (long)num_heads, (long)head_dim}, opts_f32);
        g_ds.fa_max_splits = fa_max_splits;
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

    // First layer: standalone RMSNorm
    hipLaunchKernelGGL(rmsnorm_ds, dim3(1), dim3(norm_bs), smem_norm, stream,
        h, reinterpret_cast<const __half*>(in_norm_w[0].data_ptr()), n, D, rms_eps);

    // Pre-extract all raw pointers to avoid per-layer PyTorch API overhead
    std::vector<const uint8_t*> qkv_wp(num_layers), qkv_sp(num_layers);
    std::vector<const uint8_t*> o_wp(num_layers), o_sp(num_layers);
    std::vector<const uint8_t*> gu_wp(num_layers), gu_sp(num_layers);
    std::vector<const uint8_t*> down_wp(num_layers), down_sp(num_layers);
    std::vector<const __half*> in_np(num_layers), post_np(num_layers);
    std::vector<const __half*> q_np(num_layers), k_np(num_layers);
    std::vector<__half*> kc_p(num_layers), vc_p(num_layers);
    std::vector<const __half*> o_awq_p(num_layers, nullptr), down_awq_p(num_layers, nullptr);
    bool has_awq = !o_awq_inv.empty();
    for (int li = 0; li < num_layers; li++) {
        qkv_wp[li] = qkv_w[li].data_ptr<uint8_t>();
        qkv_sp[li] = qkv_s[li].data_ptr<uint8_t>();
        o_wp[li] = o_w[li].data_ptr<uint8_t>();
        o_sp[li] = o_s[li].data_ptr<uint8_t>();
        gu_wp[li] = gu_w[li].data_ptr<uint8_t>();
        gu_sp[li] = gu_s[li].data_ptr<uint8_t>();
        down_wp[li] = down_w[li].data_ptr<uint8_t>();
        down_sp[li] = down_s[li].data_ptr<uint8_t>();
        in_np[li] = reinterpret_cast<const __half*>(in_norm_w[li].data_ptr());
        post_np[li] = reinterpret_cast<const __half*>(post_norm_w[li].data_ptr());
        q_np[li] = reinterpret_cast<const __half*>(q_norm_w[li].data_ptr());
        k_np[li] = reinterpret_cast<const __half*>(k_norm_w[li].data_ptr());
        kc_p[li] = reinterpret_cast<__half*>(k_caches[li].data_ptr<at::Half>());
        vc_p[li] = reinterpret_cast<__half*>(v_caches[li].data_ptr<at::Half>());
        if (has_awq) {
            o_awq_p[li] = reinterpret_cast<const __half*>(o_awq_inv[li].data_ptr());
            down_awq_p[li] = reinterpret_cast<const __half*>(down_awq_inv[li].data_ptr());
        }
    }

    int kv_total = num_kv_heads * head_dim;
    int inter = intermediate_size;

    for (int li = 0; li < num_layers; li++) {
        int Nqkv = qkv_N[li], Kqkv = D;
        int No = o_N[li], Ko = D;
        int Ngu = gu_N[li], Kgu = D;
        int Nd = down_N[li], Kd = intermediate_size;

        // 1. QKV GEMV
        launch_gemv(n, qkv_wp[li], qkv_sp[li], qkv, Nqkv, Kqkv, qkv_ts[li], stream);

        // 2-4. Fused head RMSNorm + RoPE + KV cache write
        __half* q_ptr = qkv;
        __half* k_ptr = qkv + q_dim;
        __half* v_ptr = qkv + q_dim + kv_dim;
        hipLaunchKernelGGL(head_rmsnorm_rope_kv, dim3(num_heads + num_kv_heads), dim3(32), 0, stream,
            q_ptr, k_ptr, v_ptr, q_np[li], k_np[li],
            q_ptr, k_ptr, kc_p[li], vc_p[li],
            cos_pos, sin_pos,
            num_heads, num_kv_heads, head_dim, rms_eps, max_seq, pos);

        // 5. Flash decode attention + AWQ+O projection
        {
            int cur_seq_len = pos + 1;
            auto* p_max = reinterpret_cast<float*>(g_ds.fa_partial_max.data_ptr());
            auto* p_sum = reinterpret_cast<float*>(g_ds.fa_partial_sum.data_ptr());
            auto* p_out = reinterpret_cast<float*>(g_ds.fa_partial_out.data_ptr());
            launch_flash_decode(qkv, kc_p[li], vc_p[li], abuf,
                p_max, p_sum, p_out,
                (int)num_heads, (int)num_kv_heads, max_seq, cur_seq_len, (int)head_dim, stream);
            if (has_awq) {
                launch_gemv_awq(abuf, o_awq_p[li], o_wp[li], o_sp[li], obuf, No, Ko, o_ts[li], stream);
            } else {
                launch_gemv(abuf, o_wp[li], o_sp[li], obuf, No, Ko, o_ts[li], stream);
            }
        }

        // 7. Fused residual + post_norm
        hipLaunchKernelGGL(fused_res_norm, dim3(1), dim3(norm_bs), smem_norm, stream,
            obuf, h, post_np[li], n, r, D, rms_eps);

        // 8. Gate+Up GEMV
        launch_gemv(n, gu_wp[li], gu_sp[li], gubuf, Ngu, Kgu, gu_ts[li], stream);

        // 9-10. Fused SiLU + AWQ + Down projection (saves silu_mul + vecmul_inplace launches)
        launch_gemv_silu_in_awq(gubuf, gubuf + inter,
            has_awq ? down_awq_p[li] : nullptr,
            down_wp[li], down_sp[li], dbuf, Nd, Kd, down_ts[li], stream);

        // 11. Fused residual + next input_norm (or just residual add for last layer)
        if (li < num_layers - 1) {
            hipLaunchKernelGGL(fused_res_norm, dim3(1), dim3(norm_bs), smem_norm, stream,
                dbuf, r, in_np[li + 1], n, h, D, rms_eps);
        } else {
            hipLaunchKernelGGL(fused_res_norm, dim3(1), dim3(norm_bs), smem_norm, stream,
                dbuf, r, in_np[0], n, h, D, rms_eps);
        }
    }

    // Return final hidden state [1, 1, D]
    return g_ds.hidden.view({1, 1, D}).clone();
}

// ============================================================
// Single decode step returning logits (folds final_norm + LM head)
// Eliminates 2 Python→C++ round trips per token.
// ============================================================
torch::Tensor nvfp4_decode_step_logits(
    torch::Tensor hidden_in,
    std::vector<torch::Tensor> qkv_w, std::vector<torch::Tensor> qkv_s, std::vector<float> qkv_ts,
    std::vector<int64_t> qkv_N,
    std::vector<torch::Tensor> o_w, std::vector<torch::Tensor> o_s, std::vector<float> o_ts,
    std::vector<int64_t> o_N,
    std::vector<torch::Tensor> gu_w, std::vector<torch::Tensor> gu_s, std::vector<float> gu_ts,
    std::vector<int64_t> gu_N,
    std::vector<torch::Tensor> down_w, std::vector<torch::Tensor> down_s, std::vector<float> down_ts,
    std::vector<int64_t> down_N,
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w, std::vector<torch::Tensor> k_norm_w,
    std::vector<torch::Tensor> o_awq_inv, std::vector<torch::Tensor> down_awq_inv,
    torch::Tensor rope_cos, torch::Tensor rope_sin,
    int64_t pos,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    float rms_eps,
    std::vector<torch::Tensor> k_caches, std::vector<torch::Tensor> v_caches,
    // Final norm + LM head
    torch::Tensor final_norm_w,
    torch::Tensor lm_w, torch::Tensor lm_s, float lm_ts,
    int64_t lm_N, int64_t lm_K
) {
    // Run the transformer layers
    auto h_out = nvfp4_decode_step(
        hidden_in,
        qkv_w, qkv_s, qkv_ts, qkv_N,
        o_w, o_s, o_ts, o_N,
        gu_w, gu_s, gu_ts, gu_N,
        down_w, down_s, down_ts, down_N,
        in_norm_w, post_norm_w,
        q_norm_w, k_norm_w,
        o_awq_inv, down_awq_inv,
        rope_cos, rope_sin,
        pos, num_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size, rms_eps,
        k_caches, v_caches
    );

    // Final norm + LM head in C++ (no Python round trip)
    int D = hidden_size;
    auto stream = at::hip::getCurrentHIPStream();
    auto device = h_out.device();

    // Reuse scratch buffer for normed output
    auto normed = torch::empty({D}, torch::dtype(torch::kHalf).device(device));
    auto* h_ptr = reinterpret_cast<const __half*>(h_out.data_ptr());
    auto* fn_ptr = reinterpret_cast<const __half*>(final_norm_w.data_ptr());
    auto* n_ptr = reinterpret_cast<__half*>(normed.data_ptr());

    int norm_bs = 256;
    size_t smem_norm = D * sizeof(float);
    hipLaunchKernelGGL(rmsnorm_ds, dim3(1), dim3(norm_bs), smem_norm, stream,
        h_ptr, fn_ptr, n_ptr, D, rms_eps);

    // LM head GEMV
    auto logits = torch::empty({lm_N}, torch::dtype(torch::kHalf).device(device));
    launch_gemv(n_ptr, lm_w.data_ptr<uint8_t>(), lm_s.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(logits.data_ptr()),
        lm_N, lm_K, lm_ts, stream);

    return logits.view({1, 1, lm_N});
}

// ============================================================
// Decode benchmark — N tokens in tight C++ loop, zero Python overhead
// Uses a fixed dummy embedding to isolate pure kernel throughput
// ============================================================
void nvfp4_decode_bench(
    torch::Tensor hidden_in,           // [1, 1, D] first token hidden
    torch::Tensor embed_weight,        // [vocab, D] for embedding lookup
    int64_t num_tokens,
    // Same layer args
    std::vector<torch::Tensor> qkv_w, std::vector<torch::Tensor> qkv_s, std::vector<float> qkv_ts,
    std::vector<int64_t> qkv_N,
    std::vector<torch::Tensor> o_w, std::vector<torch::Tensor> o_s, std::vector<float> o_ts,
    std::vector<int64_t> o_N,
    std::vector<torch::Tensor> gu_w, std::vector<torch::Tensor> gu_s, std::vector<float> gu_ts,
    std::vector<int64_t> gu_N,
    std::vector<torch::Tensor> down_w, std::vector<torch::Tensor> down_s, std::vector<float> down_ts,
    std::vector<int64_t> down_N,
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w, std::vector<torch::Tensor> k_norm_w,
    std::vector<torch::Tensor> o_awq_inv, std::vector<torch::Tensor> down_awq_inv,
    torch::Tensor rope_cos, torch::Tensor rope_sin,
    int64_t start_pos,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    float rms_eps,
    std::vector<torch::Tensor> k_caches, std::vector<torch::Tensor> v_caches,
    torch::Tensor final_norm_w,
    torch::Tensor lm_w, torch::Tensor lm_s, float lm_ts,
    int64_t lm_N, int64_t lm_K
) {
    c10::InferenceMode guard;
    ensure_lut();
    int D = hidden_size;
    auto device = hidden_in.device();
    auto stream = at::hip::getCurrentHIPStream();
    int norm_bs = 256;
    size_t smem_norm = D * sizeof(float);

    int num_layers = qkv_w.size();
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int max_seq = k_caches[0].size(2);
    int half_D = head_dim / 2;
    int inter = intermediate_size;
    bool has_awq = !o_awq_inv.empty();

    // Ensure g_ds buffers are initialized
    int fa_max_splits_needed = (max_seq + FLASH_DECODE_BLOCK_S - 1) / FLASH_DECODE_BLOCK_S;
    if (!g_ds.init) {
        auto opts = torch::dtype(torch::kHalf).device(device);
        g_ds.normed = torch::empty({D}, opts);
        g_ds.hidden = torch::empty({D}, opts);
        g_ds.residual = torch::empty({D}, opts);
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
    auto* qkv_buf = reinterpret_cast<__half*>(g_ds.qkv_buf.data_ptr());
    auto* obuf = reinterpret_cast<__half*>(g_ds.o_buf.data_ptr());
    auto* abuf = reinterpret_cast<__half*>(g_ds.attn_buf.data_ptr());
    auto* gubuf = reinterpret_cast<__half*>(g_ds.gu_buf.data_ptr());
    auto* silubuf = reinterpret_cast<__half*>(g_ds.silu_buf.data_ptr());
    auto* dbuf = reinterpret_cast<__half*>(g_ds.down_buf.data_ptr());

    std::vector<const uint8_t*> qkv_wp(num_layers), qkv_sp(num_layers);
    std::vector<const uint8_t*> o_wp(num_layers), o_sp(num_layers);
    std::vector<const uint8_t*> gu_wp(num_layers), gu_sp(num_layers);
    std::vector<const uint8_t*> down_wp(num_layers), down_sp(num_layers);
    std::vector<const __half*> in_np(num_layers), post_np(num_layers);
    std::vector<const __half*> q_np(num_layers), k_np(num_layers);
    std::vector<__half*> kc_p(num_layers), vc_p(num_layers);
    for (int li = 0; li < num_layers; li++) {
        qkv_wp[li] = qkv_w[li].data_ptr<uint8_t>();
        qkv_sp[li] = qkv_s[li].data_ptr<uint8_t>();
        o_wp[li] = o_w[li].data_ptr<uint8_t>();
        o_sp[li] = o_s[li].data_ptr<uint8_t>();
        gu_wp[li] = gu_w[li].data_ptr<uint8_t>();
        gu_sp[li] = gu_s[li].data_ptr<uint8_t>();
        down_wp[li] = down_w[li].data_ptr<uint8_t>();
        down_sp[li] = down_s[li].data_ptr<uint8_t>();
        in_np[li] = reinterpret_cast<const __half*>(in_norm_w[li].data_ptr());
        post_np[li] = reinterpret_cast<const __half*>(post_norm_w[li].data_ptr());
        q_np[li] = reinterpret_cast<const __half*>(q_norm_w[li].data_ptr());
        k_np[li] = reinterpret_cast<const __half*>(k_norm_w[li].data_ptr());
        kc_p[li] = reinterpret_cast<__half*>(k_caches[li].data_ptr<at::Half>());
        vc_p[li] = reinterpret_cast<__half*>(v_caches[li].data_ptr<at::Half>());
    }

    std::vector<const __half*> o_awq_p(num_layers, nullptr), down_awq_p(num_layers, nullptr);
    if (has_awq) {
        for (int li = 0; li < num_layers; li++) {
            o_awq_p[li] = reinterpret_cast<const __half*>(o_awq_inv[li].data_ptr());
            down_awq_p[li] = reinterpret_cast<const __half*>(down_awq_inv[li].data_ptr());
        }
    }

    auto* fn_p = reinterpret_cast<const __half*>(final_norm_w.data_ptr());

    // Allocate normed buffer + logits buffer
    auto normed_buf = torch::empty({D}, torch::dtype(torch::kHalf).device(device));
    auto logits_buf = torch::empty({lm_N}, torch::dtype(torch::kHalf).device(device));
    auto* nb_p = reinterpret_cast<__half*>(normed_buf.data_ptr());
    auto* lmw_p = lm_w.data_ptr<uint8_t>();
    auto* lms_p = lm_s.data_ptr<uint8_t>();
    auto* lmb_p = reinterpret_cast<__half*>(logits_buf.data_ptr());

    auto* embed_ptr = reinterpret_cast<const __half*>(embed_weight.data_ptr());

    // Copy first token hidden state
    hipMemcpyAsync(h, hidden_in.data_ptr(), D * sizeof(__half), hipMemcpyDeviceToDevice, stream);

    for (int64_t t = 0; t < num_tokens; t++) {
        int64_t pos = start_pos + t;
        int seq_len = pos + 1;
        const __half* cos_pos = reinterpret_cast<const __half*>(rope_cos.data_ptr()) + pos * half_D;
        const __half* sin_pos = reinterpret_cast<const __half*>(rope_sin.data_ptr()) + pos * half_D;

        // First layer norm
        hipLaunchKernelGGL(rmsnorm_ds, dim3(1), dim3(norm_bs), smem_norm, stream,
            h, in_np[0], n, D, rms_eps);

        for (int li = 0; li < num_layers; li++) {
            int Nqkv = qkv_N[li], Kqkv = D;
            int No = o_N[li];
            int Ngu = gu_N[li], Kgu = D;
            int Nd = down_N[li], Kd = intermediate_size;

            launch_gemv(n, qkv_wp[li], qkv_sp[li], qkv_buf, Nqkv, Kqkv, qkv_ts[li], stream);

            __half* q_ptr = qkv_buf;
            __half* k_ptr = qkv_buf + q_dim;
            __half* v_ptr = qkv_buf + q_dim + kv_dim;

            hipLaunchKernelGGL(head_rmsnorm_rope_kv, dim3(num_heads + num_kv_heads), dim3(32), 0, stream,
                q_ptr, k_ptr, v_ptr, q_np[li], k_np[li],
                q_ptr, k_ptr, kc_p[li], vc_p[li],
                cos_pos, sin_pos,
                num_heads, num_kv_heads, head_dim, rms_eps, max_seq, pos);

            // Flash decode attention
            {
                auto* p_max = reinterpret_cast<float*>(g_ds.fa_partial_max.data_ptr());
                auto* p_sum = reinterpret_cast<float*>(g_ds.fa_partial_sum.data_ptr());
                auto* p_out = reinterpret_cast<float*>(g_ds.fa_partial_out.data_ptr());
                launch_flash_decode(qkv_buf, kc_p[li], vc_p[li], abuf,
                    p_max, p_sum, p_out,
                    (int)num_heads, (int)num_kv_heads, max_seq, seq_len, (int)head_dim, stream);
                if (has_awq) {
                    launch_gemv_awq(abuf, o_awq_p[li], o_wp[li], o_sp[li], obuf, No, D, o_ts[li], stream);
                } else {
                    launch_gemv(abuf, o_wp[li], o_sp[li], obuf, No, D, o_ts[li], stream);
                }
            }

            hipLaunchKernelGGL(fused_res_norm, dim3(1), dim3(norm_bs), smem_norm, stream,
                obuf, h, post_np[li], n, r, D, rms_eps);

            launch_gemv(n, gu_wp[li], gu_sp[li], gubuf, Ngu, Kgu, gu_ts[li], stream);

            launch_gemv_silu_in_awq(gubuf, gubuf + inter,
                has_awq ? down_awq_p[li] : nullptr,
                down_wp[li], down_sp[li], dbuf, Nd, Kd, down_ts[li], stream);

            if (li < num_layers - 1) {
                hipLaunchKernelGGL(fused_res_norm, dim3(1), dim3(norm_bs), smem_norm, stream,
                    dbuf, r, in_np[li + 1], n, h, D, rms_eps);
            } else {
                hipLaunchKernelGGL(fused_res_norm, dim3(1), dim3(norm_bs), smem_norm, stream,
                    dbuf, r, in_np[0], n, h, D, rms_eps);
            }
        }

        // Final norm
        hipLaunchKernelGGL(rmsnorm_ds, dim3(1), dim3(norm_bs), smem_norm, stream,
            h, fn_p, nb_p, D, rms_eps);

        // LM head GEMV
        launch_gemv(nb_p, lmw_p, lms_p, lmb_p, lm_N, lm_K, lm_ts, stream);

        // Argmax → embedding lookup for next token (all GPU, no sync)
        if (t < num_tokens - 1) {
            auto next_id = logits_buf.argmax(0);
            auto next_embed = embed_weight.index({next_id.unsqueeze(0)});
            hipMemcpyAsync(h, next_embed.data_ptr(), D * sizeof(__half), hipMemcpyDeviceToDevice, stream);
        }
    }
}

// ============================================================
// Profiled decode — times per-category kernel groups for 1 token
// Returns vector of floats: [total_ms, gemv_ms, attn_ms, norm_ms, lmhead_ms, other_ms]
// ============================================================
std::vector<float> nvfp4_decode_profile(
    torch::Tensor hidden_in,
    torch::Tensor embed_weight,
    std::vector<torch::Tensor> qkv_w, std::vector<torch::Tensor> qkv_s, std::vector<float> qkv_ts,
    std::vector<int64_t> qkv_N,
    std::vector<torch::Tensor> o_w, std::vector<torch::Tensor> o_s, std::vector<float> o_ts,
    std::vector<int64_t> o_N,
    std::vector<torch::Tensor> gu_w, std::vector<torch::Tensor> gu_s, std::vector<float> gu_ts,
    std::vector<int64_t> gu_N,
    std::vector<torch::Tensor> down_w, std::vector<torch::Tensor> down_s, std::vector<float> down_ts,
    std::vector<int64_t> down_N,
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w, std::vector<torch::Tensor> k_norm_w,
    std::vector<torch::Tensor> o_awq_inv, std::vector<torch::Tensor> down_awq_inv,
    torch::Tensor rope_cos, torch::Tensor rope_sin,
    int64_t start_pos,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    float rms_eps,
    std::vector<torch::Tensor> k_caches, std::vector<torch::Tensor> v_caches,
    torch::Tensor final_norm_w,
    torch::Tensor lm_w, torch::Tensor lm_s, float lm_ts,
    int64_t lm_N, int64_t lm_K
) {
    c10::InferenceMode guard;
    ensure_lut();
    int D = hidden_size;
    auto device = hidden_in.device();
    auto stream = at::hip::getCurrentHIPStream();
    int norm_bs = 256;
    size_t smem_norm = D * sizeof(float);

    int num_layers = qkv_w.size();
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int max_seq = k_caches[0].size(2);
    int half_D = head_dim / 2;
    int inter = intermediate_size;
    bool has_awq = !o_awq_inv.empty();

    // Ensure g_ds buffers initialized
    int fa_max_splits_needed = (max_seq + FLASH_DECODE_BLOCK_S - 1) / FLASH_DECODE_BLOCK_S;
    if (!g_ds.init) {
        auto opts = torch::dtype(torch::kHalf).device(device);
        g_ds.normed = torch::empty({D}, opts);
        g_ds.hidden = torch::empty({D}, opts);
        g_ds.residual = torch::empty({D}, opts);
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
    auto* qkv_buf = reinterpret_cast<__half*>(g_ds.qkv_buf.data_ptr());
    auto* obuf = reinterpret_cast<__half*>(g_ds.o_buf.data_ptr());
    auto* abuf = reinterpret_cast<__half*>(g_ds.attn_buf.data_ptr());
    auto* gubuf = reinterpret_cast<__half*>(g_ds.gu_buf.data_ptr());
    auto* dbuf = reinterpret_cast<__half*>(g_ds.down_buf.data_ptr());

    // Pre-extract pointers
    std::vector<const uint8_t*> qkv_wp(num_layers), qkv_sp(num_layers);
    std::vector<const uint8_t*> o_wp(num_layers), o_sp(num_layers);
    std::vector<const uint8_t*> gu_wp(num_layers), gu_sp(num_layers);
    std::vector<const uint8_t*> down_wp(num_layers), down_sp(num_layers);
    std::vector<const __half*> in_np(num_layers), post_np(num_layers);
    std::vector<const __half*> q_np(num_layers), k_np(num_layers);
    std::vector<__half*> kc_p(num_layers), vc_p(num_layers);
    std::vector<const __half*> o_awq_p(num_layers, nullptr), down_awq_p(num_layers, nullptr);
    for (int li = 0; li < num_layers; li++) {
        qkv_wp[li] = qkv_w[li].data_ptr<uint8_t>();
        qkv_sp[li] = qkv_s[li].data_ptr<uint8_t>();
        o_wp[li] = o_w[li].data_ptr<uint8_t>();
        o_sp[li] = o_s[li].data_ptr<uint8_t>();
        gu_wp[li] = gu_w[li].data_ptr<uint8_t>();
        gu_sp[li] = gu_s[li].data_ptr<uint8_t>();
        down_wp[li] = down_w[li].data_ptr<uint8_t>();
        down_sp[li] = down_s[li].data_ptr<uint8_t>();
        in_np[li] = reinterpret_cast<const __half*>(in_norm_w[li].data_ptr());
        post_np[li] = reinterpret_cast<const __half*>(post_norm_w[li].data_ptr());
        q_np[li] = reinterpret_cast<const __half*>(q_norm_w[li].data_ptr());
        k_np[li] = reinterpret_cast<const __half*>(k_norm_w[li].data_ptr());
        kc_p[li] = reinterpret_cast<__half*>(k_caches[li].data_ptr<at::Half>());
        vc_p[li] = reinterpret_cast<__half*>(v_caches[li].data_ptr<at::Half>());
        if (has_awq) {
            o_awq_p[li] = reinterpret_cast<const __half*>(o_awq_inv[li].data_ptr());
            down_awq_p[li] = reinterpret_cast<const __half*>(down_awq_inv[li].data_ptr());
        }
    }

    auto* fn_p = reinterpret_cast<const __half*>(final_norm_w.data_ptr());
    auto normed_buf = torch::empty({D}, torch::dtype(torch::kHalf).device(device));
    auto logits_buf = torch::empty({lm_N}, torch::dtype(torch::kHalf).device(device));
    auto* nb_p = reinterpret_cast<__half*>(normed_buf.data_ptr());
    auto* lmw_p = lm_w.data_ptr<uint8_t>();
    auto* lms_p = lm_s.data_ptr<uint8_t>();
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

    hipLaunchKernelGGL(rmsnorm_ds, dim3(1), dim3(norm_bs), smem_norm, stream,
        h, in_np[0], n, D, rms_eps);

    for (int li = 0; li < num_layers; li++) {
        hipEvent_t* ev = &events[li * EVTS_PER_LAYER];
        int Nqkv = qkv_N[li], No = o_N[li];
        int Ngu = gu_N[li], Kgu = D;
        int Nd = down_N[li], Kd = intermediate_size;

        hipEventRecord(ev[0], stream);  // start
        launch_gemv(n, qkv_wp[li], qkv_sp[li], qkv_buf, Nqkv, D, qkv_ts[li], stream);
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
            launch_flash_decode(qkv_buf, kc_p[li], vc_p[li], abuf,
                p_max, p_sum, p_out,
                (int)num_heads, (int)num_kv_heads, max_seq, seq_len, (int)head_dim, stream);
        }
        hipEventRecord(ev[3], stream);  // after_attn

        if (has_awq)
            launch_gemv_awq(abuf, o_awq_p[li], o_wp[li], o_sp[li], obuf, No, D, o_ts[li], stream);
        else
            launch_gemv(abuf, o_wp[li], o_sp[li], obuf, No, D, o_ts[li], stream);
        hipEventRecord(ev[4], stream);  // after_o

        hipLaunchKernelGGL(fused_res_norm, dim3(1), dim3(norm_bs), smem_norm, stream,
            obuf, h, post_np[li], n, r, D, rms_eps);
        hipEventRecord(ev[5], stream);  // after_rn1

        launch_gemv(n, gu_wp[li], gu_sp[li], gubuf, Ngu, Kgu, gu_ts[li], stream);
        hipEventRecord(ev[6], stream);  // after_gu

        launch_gemv_silu_in_awq(gubuf, gubuf + inter,
            has_awq ? down_awq_p[li] : nullptr,
            down_wp[li], down_sp[li], dbuf, Nd, Kd, down_ts[li], stream);
        hipEventRecord(ev[7], stream);  // after_down

        if (li < num_layers - 1)
            hipLaunchKernelGGL(fused_res_norm, dim3(1), dim3(norm_bs), smem_norm, stream,
                dbuf, r, in_np[li + 1], n, h, D, rms_eps);
        else
            hipLaunchKernelGGL(fused_res_norm, dim3(1), dim3(norm_bs), smem_norm, stream,
                dbuf, r, in_np[0], n, h, D, rms_eps);
        hipEventRecord(ev[8], stream);  // after_rn2
    }

    // Final norm + LM head
    hipLaunchKernelGGL(rmsnorm_ds, dim3(1), dim3(norm_bs), smem_norm, stream,
        h, fn_p, nb_p, D, rms_eps);
    launch_gemv(nb_p, lmw_p, lms_p, lmb_p, lm_N, lm_K, lm_ts, stream);

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

torch::Tensor nvfp4_decode_loop(
    torch::Tensor embed_weight,        // [vocab, D] FP16
    torch::Tensor final_norm_w,        // [D] FP16
    torch::Tensor lm_w, torch::Tensor lm_s, float lm_ts,
    int64_t lm_N, int64_t lm_K,
    torch::Tensor first_token_hidden,  // [1, 1, D] FP16
    int64_t num_tokens,
    // Same args as decode_step
    std::vector<torch::Tensor> qkv_w, std::vector<torch::Tensor> qkv_s, std::vector<float> qkv_ts,
    std::vector<int64_t> qkv_N,
    std::vector<torch::Tensor> o_w, std::vector<torch::Tensor> o_s, std::vector<float> o_ts,
    std::vector<int64_t> o_N,
    std::vector<torch::Tensor> gu_w, std::vector<torch::Tensor> gu_s, std::vector<float> gu_ts,
    std::vector<int64_t> gu_N,
    std::vector<torch::Tensor> down_w, std::vector<torch::Tensor> down_s, std::vector<float> down_ts,
    std::vector<int64_t> down_N,
    std::vector<torch::Tensor> in_norm_w, std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> q_norm_w, std::vector<torch::Tensor> k_norm_w,
    torch::Tensor rope_cos, torch::Tensor rope_sin,
    int64_t start_pos,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    float rms_eps,
    std::vector<torch::Tensor> k_caches, std::vector<torch::Tensor> v_caches
) {
    c10::InferenceMode guard;
    ensure_lut();
    int D = hidden_size;
    auto device = first_token_hidden.device();
    auto stream = at::hip::getCurrentHIPStream();
    int norm_bs = 256;
    size_t smem_norm = D * sizeof(float);

    // Init lm_head buffer
    if (!g_full.init) {
        g_full.lm_buf = torch::empty({lm_N}, torch::dtype(torch::kHalf).device(device));
        g_full.init = true;
    }

    // All token IDs on GPU — no per-token GPU→CPU sync
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
    auto* qkv_buf = reinterpret_cast<__half*>(g_ds.qkv_buf.data_ptr());
    auto* obuf = reinterpret_cast<__half*>(g_ds.o_buf.data_ptr());
    auto* abuf = reinterpret_cast<__half*>(g_ds.attn_buf.data_ptr());
    auto* gubuf = reinterpret_cast<__half*>(g_ds.gu_buf.data_ptr());
    auto* silubuf = reinterpret_cast<__half*>(g_ds.silu_buf.data_ptr());
    auto* dbuf = reinterpret_cast<__half*>(g_ds.down_buf.data_ptr());

    std::vector<const uint8_t*> qkv_wp(num_layers), qkv_sp(num_layers);
    std::vector<const uint8_t*> o_wp(num_layers), o_sp(num_layers);
    std::vector<const uint8_t*> gu_wp(num_layers), gu_sp(num_layers);
    std::vector<const uint8_t*> down_wp(num_layers), down_sp(num_layers);
    std::vector<const __half*> in_np(num_layers), post_np(num_layers);
    std::vector<const __half*> q_np(num_layers), k_np(num_layers);
    std::vector<__half*> kc_p(num_layers), vc_p(num_layers);
    for (int li = 0; li < num_layers; li++) {
        qkv_wp[li] = qkv_w[li].data_ptr<uint8_t>();
        qkv_sp[li] = qkv_s[li].data_ptr<uint8_t>();
        o_wp[li] = o_w[li].data_ptr<uint8_t>();
        o_sp[li] = o_s[li].data_ptr<uint8_t>();
        gu_wp[li] = gu_w[li].data_ptr<uint8_t>();
        gu_sp[li] = gu_s[li].data_ptr<uint8_t>();
        down_wp[li] = down_w[li].data_ptr<uint8_t>();
        down_sp[li] = down_s[li].data_ptr<uint8_t>();
        in_np[li] = reinterpret_cast<const __half*>(in_norm_w[li].data_ptr());
        post_np[li] = reinterpret_cast<const __half*>(post_norm_w[li].data_ptr());
        q_np[li] = reinterpret_cast<const __half*>(q_norm_w[li].data_ptr());
        k_np[li] = reinterpret_cast<const __half*>(k_norm_w[li].data_ptr());
        kc_p[li] = reinterpret_cast<__half*>(k_caches[li].data_ptr<at::Half>());
        vc_p[li] = reinterpret_cast<__half*>(v_caches[li].data_ptr<at::Half>());
    }

    // Pre-extract final norm / lm_head pointers
    auto* fn_p = reinterpret_cast<const __half*>(final_norm_w.data_ptr());
    auto* nb_p = reinterpret_cast<__half*>(normed_buf.data_ptr());
    auto* lmw_p = lm_w.data_ptr<uint8_t>();
    auto* lms_p = lm_s.data_ptr<uint8_t>();
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

        // First layer norm
        hipLaunchKernelGGL(rmsnorm_ds, dim3(1), dim3(norm_bs), smem_norm, stream,
            h, in_np[0], n, D, rms_eps);

        int inter = intermediate_size;

        for (int li = 0; li < num_layers; li++) {
            int Nqkv = qkv_N[li], Kqkv = D;
            int No = o_N[li], Ko = D;
            int Ngu = gu_N[li], Kgu = D;
            int Nd = down_N[li], Kd = intermediate_size;

            launch_gemv(n, qkv_wp[li], qkv_sp[li],
                qkv_buf, Nqkv, Kqkv, qkv_ts[li], stream);

            __half* q_ptr = qkv_buf;
            __half* k_ptr = qkv_buf + q_dim;
            __half* v_ptr = qkv_buf + q_dim + kv_dim;

            hipLaunchKernelGGL(head_rmsnorm_rope_kv, dim3(num_heads + num_kv_heads), dim3(32), 0, stream,
                q_ptr, k_ptr, v_ptr, q_np[li], k_np[li],
                q_ptr, k_ptr, kc_p[li], vc_p[li],
                cos_pos, sin_pos,
                num_heads, num_kv_heads, head_dim, rms_eps, max_seq, pos);

            // Flash decode attention + O projection
            {
                auto* p_max = reinterpret_cast<float*>(g_ds.fa_partial_max.data_ptr());
                auto* p_sum = reinterpret_cast<float*>(g_ds.fa_partial_sum.data_ptr());
                auto* p_out = reinterpret_cast<float*>(g_ds.fa_partial_out.data_ptr());
                launch_flash_decode(qkv_buf, kc_p[li], vc_p[li], abuf,
                    p_max, p_sum, p_out,
                    (int)num_heads, (int)num_kv_heads, max_seq, seq_len, (int)head_dim, stream);
                launch_gemv(abuf, o_wp[li], o_sp[li], obuf, No, Ko, o_ts[li], stream);
            }

            hipLaunchKernelGGL(fused_res_norm, dim3(1), dim3(norm_bs), smem_norm, stream,
                obuf, h, post_np[li], n, r, D, rms_eps);

            launch_gemv(n, gu_wp[li], gu_sp[li], gubuf, Ngu, Kgu, gu_ts[li], stream);

            // Fused SiLU + Down projection
            launch_gemv_silu_in_awq(gubuf, gubuf + inter, nullptr,
                down_wp[li], down_sp[li], dbuf, Nd, Kd, down_ts[li], stream);

            if (li < num_layers - 1) {
                hipLaunchKernelGGL(fused_res_norm, dim3(1), dim3(norm_bs), smem_norm, stream,
                    dbuf, r, in_np[li + 1], n, h, D, rms_eps);
            } else {
                hipLaunchKernelGGL(fused_res_norm, dim3(1), dim3(norm_bs), smem_norm, stream,
                    dbuf, r, in_np[0], n, h, D, rms_eps);
            }
        }

        // 2. Final norm (write to normed_buf)
        hipLaunchKernelGGL(rmsnorm_ds, dim3(1), dim3(norm_bs), smem_norm, stream,
            h, fn_p, nb_p, D, rms_eps);

        // 3. LM head GEMV
        launch_gemv(nb_p, lmw_p, lms_p, lmb_p, lm_N, lm_K, lm_ts, stream);

        // 4. Argmax (stays on GPU)
        auto next_id = g_full.lm_buf.argmax(0);  // scalar tensor on GPU
        token_ids_gpu[t] = next_id;

        // 5. Embedding lookup for next token (GPU→GPU, no sync)
        if (t < num_tokens - 1) {
            h_tensor = embed_weight.index({next_id.unsqueeze(0)}).view({1, 1, D});
        }
    }

    // Single GPU→CPU copy at the end
    return token_ids_gpu.to(torch::kCPU);
}


// ============================================================
// Debug wrappers — expose individual kernels to Python
// ============================================================

torch::Tensor dbg_rmsnorm(torch::Tensor X, torch::Tensor W, float eps) {
    ensure_lut();
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
                        int64_t N, int64_t K, float ts) {
    ensure_lut();
    auto Y = torch::empty({N}, torch::dtype(torch::kHalf).device(X.device()));
    auto stream = at::hip::getCurrentHIPStream();
    launch_gemv(reinterpret_cast<const __half*>(X.data_ptr()),
                W.data_ptr<uint8_t>(), S.data_ptr<uint8_t>(),
                reinterpret_cast<__half*>(Y.data_ptr()),
                N, K, ts, stream);
    return Y;
}

torch::Tensor dbg_gemm(torch::Tensor X, torch::Tensor W, torch::Tensor S,
                        int64_t M, int64_t N, int64_t K, float ts) {
    ensure_lut();
    auto Y = torch::empty({M, N}, torch::dtype(torch::kHalf).device(X.device()));
    auto stream = at::hip::getCurrentHIPStream();
    launch_gemm(reinterpret_cast<const __half*>(X.data_ptr()),
                W.data_ptr<uint8_t>(), S.data_ptr<uint8_t>(),
                reinterpret_cast<__half*>(Y.data_ptr()),
                M, N, K, ts, stream);
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_step", &nvfp4_decode_step, "NVFP4 decode step (all layers in C++)");
    m.def("decode_step_logits", &nvfp4_decode_step_logits, "NVFP4 decode step → logits (fused final_norm + LM head)");
    m.def("decode_loop", &nvfp4_decode_loop, "NVFP4 full decode loop (N tokens in C++)");
    m.def("decode_bench", &nvfp4_decode_bench, "NVFP4 decode benchmark (N tokens in C++, zero Python overhead)");
    m.def("decode_profile", &nvfp4_decode_profile, "NVFP4 profiled decode (1 token, returns per-category timings)");
    m.def("prefill_logits", &nvfp4_prefill_logits, "NVFP4 prefill → logits [M, vocab] (all layers in C++)");
    m.def("dbg_rmsnorm", &dbg_rmsnorm, "Debug: RMSNorm");
    m.def("dbg_head_rmsnorm", &dbg_head_rmsnorm, "Debug: Head RMSNorm");
    m.def("dbg_rope", &dbg_rope, "Debug: RoPE");
    m.def("dbg_fused_res_norm", &dbg_fused_res_norm, "Debug: Fused residual+RMSNorm");
    m.def("dbg_gemv", &dbg_gemv, "Debug: GEMV");
    m.def("dbg_gemm", &dbg_gemm, "Debug: GEMM");
}
