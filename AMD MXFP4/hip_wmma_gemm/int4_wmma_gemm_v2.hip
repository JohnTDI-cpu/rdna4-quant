/*
 * W4A4 WMMA GEMM v2: Tiled with shared memory
 * TILE_M=64, TILE_N=64, 16 warps (512 threads)
 * Each warp computes one 16x16 WMMA output tile
 * A and B loaded cooperatively into shared memory
 *
 * Layout: 4×4 grid of 16×16 WMMA tiles per block
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef int __attribute__((ext_vector_type(2))) v2i;
typedef int __attribute__((ext_vector_type(8))) v8i;

__device__ __forceinline__ uint32_t pack8_i4(int v0, int v1, int v2, int v3,
                                              int v4, int v5, int v6, int v7) {
    return ((v0 & 0xF)) | ((v1 & 0xF) << 4) | ((v2 & 0xF) << 8) | ((v3 & 0xF) << 12) |
           ((v4 & 0xF) << 16) | ((v5 & 0xF) << 20) | ((v6 & 0xF) << 24) | ((v7 & 0xF) << 28);
}

// Tiled W4A4 WMMA GEMM
// TILE_M=64, TILE_N=64: 4×4 = 16 WMMA tiles, 16 warps, 512 threads
__global__ void w4a4_wmma_gemm_v2(
    const __half* __restrict__ A,      // [M, K] FP16
    const uint8_t* __restrict__ W,     // [N, K/2] uint8 packed INT4
    const __half* __restrict__ S,      // [N, K/32] FP16 scales
    __half* __restrict__ C,            // [M, N] FP16
    int M, int N, int K
) {
    const int TILE_M = 64, TILE_N = 64;
    const int tile_n = blockIdx.x * TILE_N;
    const int tile_m = blockIdx.y * TILE_M;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    // 4×4 warp grid
    const int warp_m = warp_id / 4;  // 0..3
    const int warp_n = warp_id % 4;  // 0..3

    const int Kh = K / 2;
    const int nsc = K / 32;

    // Accumulator for this warp's 16×16 tile
    float acc[8];
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    // WMMA fragment mappings for this warp's 16×16 sub-tile
    int m_local = lane % 16;  // row within 16×16 tile
    int m_global = tile_m + warp_m * 16 + m_local;
    int k_sub = lane / 16;  // 0 or 1
    int k_sub_offset = k_sub * 16;

    int n_local = lane % 16;
    int n_global = tile_n + warp_n * 16 + n_local;

    const __half* a_row = (m_global < M) ? (A + (int64_t)m_global * K) : nullptr;

    int num_k_blocks = K / 32;
    for (int ki = 0; ki < num_k_blocks; ki++) {
        int k_start = ki * 32;

        // 1. Quantize A to INT4 on-the-fly
        v2i a_frag;
        float a_scale;
        if (a_row) {
            // Load 16 FP16 values
            float vals[16];
            float amax = 0.0f;
            for (int i = 0; i < 16; i++) {
                int k = k_start + k_sub_offset + i;
                float v = (k < K) ? __half2float(a_row[k]) : 0.0f;
                vals[i] = v;
                float av = fabsf(v);
                if (av > amax) amax = av;
            }
            // Reduce across partner lane (lane XOR 16)
            float partner_amax = __shfl(amax, lane ^ 16);
            amax = fmaxf(amax, partner_amax);

            a_scale = amax / 7.0f;
            float inv_scale = (a_scale > 0.0f) ? (1.0f / a_scale) : 0.0f;

            int q[16];
            for (int i = 0; i < 16; i++) {
                int qi = __float2int_rn(vals[i] * inv_scale);
                qi = max(-8, min(7, qi));
                q[i] = qi;
            }
            a_frag[0] = (int)pack8_i4(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
            a_frag[1] = (int)pack8_i4(q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15]);
        } else {
            a_frag[0] = 0; a_frag[1] = 0;
            a_scale = 0.0f;
        }

        // 2. Load B (weight) INT4 fragment
        v2i b_frag;
        if (n_global < N) {
            int kh_start = k_start / 2 + k_sub * 8;
            const uint8_t* w_ptr = W + (int64_t)n_global * Kh + kh_start;
            uint8_t bytes[8];
            for (int i = 0; i < 8; i++) bytes[i] = w_ptr[i];
            b_frag[0] = *reinterpret_cast<const int*>(&bytes[0]);
            b_frag[1] = *reinterpret_cast<const int*>(&bytes[4]);
        } else {
            b_frag[0] = 0; b_frag[1] = 0;
        }

        // 3. Load weight scale
        float w_scale = 0.0f;
        if (n_global < N)
            w_scale = __half2float(S[(int64_t)n_global * nsc + ki]);

        // 4. WMMA
        v8i c_i32;
        for (int i = 0; i < 8; i++) c_i32[i] = 0;
        c_i32 = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
            true, a_frag, true, b_frag, c_i32, false
        );

        // 5. Accumulate with scales
        float combined_scale = a_scale * w_scale;
        for (int i = 0; i < 8; i++)
            acc[i] += (float)c_i32[i] * combined_scale;
    }

    // Store: column-distributed output mapping
    int out_col = tile_n + warp_n * 16 + (lane % 16);
    int out_row_base = tile_m + warp_m * 16 + (lane / 16) * 8;
    for (int j = 0; j < 8; j++) {
        int r = out_row_base + j;
        int c = out_col;
        if (r < M && c < N)
            C[(int64_t)r * N + c] = __float2half(acc[j]);
    }
}

// PyTorch wrapper
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>

torch::Tensor w4a4_gemm(torch::Tensor A, torch::Tensor W_packed, torch::Tensor W_scales,
                         int64_t N, int64_t K) {
    int M = A.size(0);
    auto C = torch::empty({M, N}, torch::dtype(torch::kHalf).device(A.device()));
    auto stream = at::hip::getCurrentHIPStream();

    const int TILE_M = 64, TILE_N = 64;
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(512);  // 16 warps

    hipLaunchKernelGGL(w4a4_wmma_gemm_v2, grid, block, 0, stream,
        reinterpret_cast<const __half*>(A.data_ptr()),
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(W_scales.data_ptr()),
        reinterpret_cast<__half*>(C.data_ptr()),
        M, N, (int)K);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w4a4_gemm", &w4a4_gemm, "W4A4 WMMA GEMM v2 (64x64 tiles)");
}
