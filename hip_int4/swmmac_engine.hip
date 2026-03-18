// swmmac_engine.hip — RDNA4 native SWMMAC/WMMA INT4 engine
// Target: V_SWMMAC_I32_16X16X64_IU4 → 1531 TOPS peak (2:4 sparse)
// Target: V_WMMA_I32_16X16X32_IU4  → 766 TOPS peak (dense)
//
// All kernels use signed symmetric INT4 (stored unsigned+8, neg=true)
// Column-distributed output: lane%16=col, (lane/16)*8+j=row
// Verified on R9700 (gfx1201) via DISCOVERY_GFX12_WMMA_OUTPUT.md
//
// Tiled weight format for coalesced reads:
//   Dense:  [K/8, N/16, 64] — 64 bytes per tile block
//   Sparse: [K/64, N/16, 32] packed + [K/64, N/16, 2] short indices
//
// Build: #include from int4_decode_step.hip, --offload-arch=gfx1201
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ---- Vector types for WMMA/SWMMAC intrinsics ----
typedef int _v2i __attribute__((ext_vector_type(2)));
typedef int _v4i __attribute__((ext_vector_type(4)));
typedef int _v8i __attribute__((ext_vector_type(8)));

// ================================================================
// 1. ACTIVATION QUANTIZATION — signed symmetric INT4
//    scale = absmax / 7, stored unsigned+8 (WMMA neg=true → signed)
//    32 threads per group (matches WMMA K=32 and SWMMAC K=64 half)
// ================================================================
__global__ void swmmac_quantize_act(
    const __half* __restrict__ input,   // [M, K] FP16
    uint8_t* __restrict__ out_q,        // [M_pad, K/2] packed uint4
    float* __restrict__ out_scales,     // [M_pad, ng] float32
    int M, int K, int group_size)
{
    const int g = blockIdx.x;           // which group
    const int m = blockIdx.y;           // which row
    const int tid = threadIdx.x;        // 0..31
    if (m >= M) return;

    const int ng = K / group_size;
    float val = __half2float(input[(int64_t)m * K + g * 32 + tid]);

    // Warp-wide absmax
    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, off));

    float scale = (amax > 1e-10f) ? (amax / 7.0f) : 1e-10f;

    // Quantize: signed [-8,+7] → stored unsigned [0,15]
    int q = __float2int_rn(val / scale);
    q = max(-8, min(7, q));
    int stored = q + 8;

    // Pack pairs (even thread stores both)
    int other = __shfl_xor(stored, 1);
    if (tid & 1)
        out_q[(int64_t)m * (K/2) + g * 16 + tid / 2] = (uint8_t)((stored << 4) | other);

    if (tid == 0)
        out_scales[m * ng + g] = scale;
}

// ================================================================
// 2. TILE WEIGHTS — dense [N, K/2] → tiled [K/8, N/16, 64]
//    Coalesced: lane nl reads byte at offset nl*4 within 64-byte block
// ================================================================
__global__ void swmmac_tile_dense(
    const uint8_t* __restrict__ src,    // [N, K/2] row-major packed
    uint8_t* __restrict__ dst,          // [K/8, N/16, 64] tiled
    int N, int K)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int K2 = K / 2, NB = N / 16;
    int total = (K / 8) * NB * 16;
    if (tid >= total) return;

    int kc = tid / (NB * 16);
    int rem = tid % (NB * 16);
    int nb = rem / 16;
    int nl = rem % 16;

    // src row = nb*16 + nl, column offset = kc*4
    *(uint32_t*)(dst + (int64_t)(kc * NB + nb) * 64 + nl * 4)
        = *(const uint32_t*)(src + (int64_t)(nb * 16 + nl) * K2 + kc * 4);
}

// ================================================================
// 3. TILE SPARSE WEIGHTS — [N, K/4] + [N, K/4] idx → tiled format
//    Output: wt_tiled[K/64, N/16, 32] + idx_tiled[K/64, N/16, 2]
//
//    Per 32-byte sparse block (per K-group per N-tile per K-half):
//      16 lanes × 2 bytes = 32 bytes of compressed nibbles
//      Coalesced: lane nl reads 2 bytes at offset nl*2
//
//    Index: 1 short per K-half per N-tile (broadcast to all lanes)
// ================================================================
__global__ void swmmac_tile_sparse(
    const uint8_t* __restrict__ src_packed,  // [N, K/4] compressed
    const uint8_t* __restrict__ src_idx,     // [N, K/4] raw 2-bit per group
    uint8_t* __restrict__ dst_packed,        // [K/64, N/16, 32] tiled
    short* __restrict__ dst_idx,             // [K/64, N/16, 2] packed shorts
    int N, int K)
{
    // Each thread handles one (kgroup, ntile, nl, khalf) position
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ng = K / 64;
    int NB = N / 16;
    int total = ng * NB * 16 * 2;  // ×2 for K-halves
    if (tid >= total) return;

    int kh = tid % 2;               // K-half (0 or 1)
    int nl = (tid / 2) % 16;        // lane within N-tile
    int nb = (tid / 32) % NB;       // N-tile index
    int kg = tid / (32 * NB);       // K-group index

    int n_row = nb * 16 + nl;       // source N row
    int k_base = kg * 64;           // K offset for this group
    int k_half_base = k_base + kh * 32;  // K offset for this half

    // Copy 8 compressed bytes (8 groups × 1 byte each)
    // Source: src_packed[n_row, k_half_base/4 .. +8]
    int src_off = n_row * (K / 4) + k_half_base / 4;
    int dst_off = (kg * NB + nb) * 32 + kh * 16 + nl * 1;

    // Each lane copies 1 byte? No — need 8 bytes per lane.
    // Actually: per K-half, each N-row has 32/4 = 8 groups = 8 compressed bytes.
    // But SWMMAC A fragment is v2i = 8 bytes per lane.
    // So lane nl needs 8 consecutive bytes from row nl.

    // Tiled layout: [kg][nb][kh*16 + nl] → but each lane needs 8 bytes not 1.
    // Rethink: tile should be [kg][nb][64] where first 32 = kh=0, next 32 = kh=1
    // Within each 32-byte half: lane nl occupies bytes [nl*2 .. nl*2+1]? No, v2i = 8 bytes.

    // Actually v2i = 2 ints = 8 bytes = 16 nibbles.
    // 16 lanes × 8 bytes = 128 bytes per K-half per N-tile.
    // But we have 32 bytes per K-half in the plan... that's wrong.

    // Correct: per K-half per N-tile: 16 rows × 8 bytes = 128 bytes
    // Total per K-group per N-tile: 256 bytes (2 K-halves)
    // Plus index: 2 shorts = 4 bytes

    // Simpler approach: just copy 8 bytes per lane (same as dense tile but K/4 stride)
    const uint8_t* sp = src_packed + (int64_t)n_row * (K/4) + k_half_base / 4;
    uint8_t* dp = dst_packed + ((int64_t)kg * NB + nb) * 256 + kh * 128 + nl * 8;
    // Copy 8 bytes (2 ints)
    *(uint32_t*)(dp) = *(const uint32_t*)(sp);
    *(uint32_t*)(dp + 4) = *(const uint32_t*)(sp + 4);

    // Pack index for this K-half: 8 groups × 2-bit → short
    if (nl == 0) {
        const uint8_t* si = src_idx + (int64_t)n_row * (K/4) + k_half_base / 4;
        uint16_t packed = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            packed |= ((uint16_t)(si[i] & 3)) << (i * 2);
        // But index should be SAME for all lanes in same N-tile + K-group + K-half
        // Because sparsity pattern is per-row, not per-tile...
        // Actually each row has different sparsity! idx is per-lane.
        // So we need [kg][nb][kh][nl] = 16 shorts per K-half per N-tile.
        // Not 1 short as in plan. Plan was wrong about broadcast!
    }
}

// The tiling is more complex than planned. Let me simplify:
// Just do row-to-column transpose (same as dense tile_weights but for K/4 stride)

__global__ void swmmac_tile_sparse_v2(
    const uint8_t* __restrict__ src,    // [N, K/4] compressed packed
    uint8_t* __restrict__ dst,          // [K/16, N/16, 128] tiled (8 bytes per lane per K-half)
    int N, int K)
{
    // Same pattern as dense tiling but reading K/4 stride
    // Each lane loads 8 bytes (v2i = 2 ints) from its row
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int K4 = K / 4;
    int NB = N / 16;
    // K/4 is divided into chunks of 8 bytes (matching v2i per lane)
    int num_kchunks = K4 / 8;  // K=2048 → K/4=512 → 64 chunks of 8 bytes
    int total = num_kchunks * NB * 16;
    if (tid >= total) return;

    int kc = tid / (NB * 16);
    int rem = tid % (NB * 16);
    int nb = rem / 16;
    int nl = rem % 16;

    int src_row = nb * 16 + nl;
    int src_col = kc * 8;

    // Copy 8 bytes from row-major to tiled
    // dst layout: [kc][nb][nl * 8 .. nl*8+7]  → 16 lanes × 8 bytes = 128 per block
    int64_t dst_off = (int64_t)(kc * NB + nb) * 128 + nl * 8;
    int64_t src_off = (int64_t)src_row * K4 + src_col;

    *(uint32_t*)(dst + dst_off) = *(const uint32_t*)(src + src_off);
    *(uint32_t*)(dst + dst_off + 4) = *(const uint32_t*)(src + src_off + 4);
}

// Tile raw indices: [N, K/4] → [K/16, N/16, 16] (1 byte per lane per K-chunk)
// Then pack on GPU into [K/32, N/16, 16] shorts (2 K-chunks → 1 short per lane)
__global__ void swmmac_tile_idx(
    const uint8_t* __restrict__ src_idx,  // [N, K/4] raw 2-bit indices
    short* __restrict__ dst_idx,          // [K/32, N/16, 16] packed shorts
    int N, int K)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int K4 = K / 4;
    int NB = N / 16;
    int num_idx_groups = K4 / 16;  // 16 raw bytes → 2 shorts (8 groups × 2 bits each)
    int total = num_idx_groups * NB * 16;
    if (tid >= total) return;

    int ig = tid / (NB * 16);
    int rem = tid % (NB * 16);
    int nb = rem / 16;
    int nl = rem % 16;

    int src_row = nb * 16 + nl;
    const uint8_t* si = src_idx + (int64_t)src_row * K4 + ig * 16;

    // Pack first 8 raw bytes → short 0
    uint16_t p0 = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) p0 |= ((uint16_t)(si[i] & 3)) << (i * 2);

    // Pack next 8 raw bytes → short 1
    uint16_t p1 = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) p1 |= ((uint16_t)(si[8 + i] & 3)) << (i * 2);

    int64_t dst_off = (int64_t)(ig * 2) * NB * 16 + nb * 16 + nl;
    dst_idx[dst_off] = (short)p0;
    dst_idx[dst_off + NB * 16] = (short)p1;
}

// ================================================================
// 4. TRANSPOSE SCALES — [N, ng] FP16 → [ng, N_pad] float32
// ================================================================
__global__ void swmmac_transpose_scales(
    const __half* __restrict__ scales,   // [N, ng] FP16 (scale only, no zero)
    float* __restrict__ out,             // [ng, N_pad] float32
    int N, int N_pad, int ng)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * ng) return;
    int n = tid / ng, g = tid % ng;
    out[(int64_t)g * N_pad + n] = __half2float(scales[(int64_t)n * ng + g]);
}

// ================================================================
// 5. SWMMAC 2:4 SPARSE GEMM — V_SWMMAC_I32_16X16X64_IU4
//    16M × 64N per block, 4 SWMMAC per K-step
//    Tiled weights: coalesced loads → target 477 TOPS
// ================================================================
template<int GS = 64>
__global__ __launch_bounds__(32, 8)
void swmmac_gemm_tiled(
    const uint8_t* __restrict__ act_q,      // [M_pad, K/2] dense INT4
    const float* __restrict__ act_scales,    // [M_pad, ng]
    const uint8_t* __restrict__ wt_tiled,   // [K/16, N/16, 128] tiled sparse
    const short* __restrict__ idx_tiled,     // [K/32, N/16, 16] tiled indices
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
    const int NB = N_pad / 16;
    const int m_idx = m_start + nl;
    const bool m_valid = (m_idx < M);

    float acc0[8]={}, acc1[8]={}, acc2[8]={}, acc3[8]={};

    for (int g = 0; g < ng; g++) {
        int k = g * GS;

        // Load B (dense activations) — v4i per lane
        _v4i B;
        if (m_valid) {
            const int* ap = (const int*)(act_q + (int64_t)m_idx * (K/2) + k/2 + kg * 16);
            B = (_v4i){ap[0], ap[1], ap[2], ap[3]};
        } else { B = (_v4i){}; }

        // Load 4 A tiles (COALESCED from tiled sparse) + indices
        // wt_tiled layout: [K/16, N/16, 128]
        // For K-group g with GS=64: 2 K-chunks of 8 bytes each (kc = g*4 + kg*2 + {0,1})
        // Each lane reads 8 bytes = v2i from tiled[kc][nb][nl*8]
        #define LOAD_SPARSE_TILED(ti) \
            _v2i A##ti; short sidx##ti; { \
                int nb = n_start/16 + ti; \
                int kc_base = (k/4) / 8;  /* which 8-byte chunk in K/4 space */ \
                int kc0 = kc_base + kg * 2; \
                const uint8_t* tp = wt_tiled + (int64_t)(kc0 * NB + nb) * 128 + nl * 8; \
                A##ti = (_v2i){*(const int*)tp, *(const int*)(tp + 4)}; \
                /* Index: [K/32, N/16, 16]. ig = g (K-group), K-half = kg */ \
                sidx##ti = idx_tiled[((int64_t)g * 2 + kg) * NB * 16 + nb * 16 + nl]; \
            }
        LOAD_SPARSE_TILED(0) LOAD_SPARSE_TILED(1)
        LOAD_SPARSE_TILED(2) LOAD_SPARSE_TILED(3)
        #undef LOAD_SPARSE_TILED

        // 4 SWMMAC — signed sparse × signed dense
        _v8i R0={}, R1={}, R2={}, R3={};
        R0 = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(true, A0, true, B, R0, sidx0, false);
        R1 = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(true, A1, true, B, R1, sidx1, false);
        R2 = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(true, A2, true, B, R2, sidx2, false);
        R3 = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(true, A3, true, B, R3, sidx3, false);

        // Minimal dequant: w_scale × a_scale × raw_dot
        float a_sc = m_valid ? act_scales[m_idx * ng + g] : 0.0f;
        #define DEQUANT(ti) { \
            float ws = w_scales[g * N_pad + n_start + ti*16 + nl]; \
            float c = ws * a_sc; \
            for(int j=0;j<8;j++) acc##ti[j] += c * (float)R##ti[j]; }
        DEQUANT(0) DEQUANT(1) DEQUANT(2) DEQUANT(3)
        #undef DEQUANT
    }

    // Store: column-distributed (lane%16=col, (lane/16)*8+j=row)
    for (int j = 0; j < 8; j++) {
        int om = m_start + kg * 8 + j;
        if (om >= M) continue;
        __half* row = output + (int64_t)om * N_pad;
        row[n_start + nl]      = __float2half(acc0[j]);
        row[n_start + 16 + nl] = __float2half(acc1[j]);
        row[n_start + 32 + nl] = __float2half(acc2[j]);
        row[n_start + 48 + nl] = __float2half(acc3[j]);
    }
}

// ================================================================
// 6. DENSE WMMA GEMM — V_WMMA_I32_16X16X32_IU4
//    Same structure as SWMMAC but dense weights, K=32 per WMMA
// ================================================================
template<int GS = 64>
__global__ __launch_bounds__(32, 8)
void wmma_gemm_tiled(
    const uint8_t* __restrict__ act_q,
    const float* __restrict__ act_scales,
    const uint8_t* __restrict__ wt_tiled,   // [K/8, N/16, 64] dense tiled
    const float* __restrict__ w_scales,
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
        _v2i A;
        float a_sc = 0.0f;
        if (m_valid) {
            const uint8_t* aq = act_q + (int64_t)m_idx * (K/2) + k/2 + kg * 8;
            A = (_v2i){*(const int*)aq, *(const int*)(aq + 4)};
            a_sc = act_scales[m_idx * ng + g];
        } else { A = (_v2i){}; }

        // Load 4 B tiles (coalesced from tiled dense)
        #define LOAD_B(ti) _v2i B##ti; { \
            int nb = n_start/16 + ti; \
            int kc0 = k/8 + kg*2; \
            B##ti = (_v2i){ \
                (int)*(const uint32_t*)(wt_tiled + (int64_t)(kc0*NB+nb)*64 + nl*4), \
                (int)*(const uint32_t*)(wt_tiled + (int64_t)((kc0+1)*NB+nb)*64 + nl*4)}; }
        LOAD_B(0) LOAD_B(1) LOAD_B(2) LOAD_B(3)
        #undef LOAD_B

        // 4 dense WMMA
        _v8i R0={}, R1={}, R2={}, R3={};
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
        int om = m_start + kg * 8 + j;
        if (om >= M) continue;
        __half* row = output + (int64_t)om * N_pad;
        row[n_start + nl]      = __float2half(acc0[j]);
        row[n_start + 16 + nl] = __float2half(acc1[j]);
        row[n_start + 32 + nl] = __float2half(acc2[j]);
        row[n_start + 48 + nl] = __float2half(acc3[j]);
    }
}

// ================================================================
// 7. BATCHED EXPERT WMMA — all-experts dispatch, zero CPU sync
//    Like gemv_w4a4_udot8_v3 but uses WMMA instead of DOT8.
//    Grid: (N_tiles, E). Each block handles one (N-tile, expert) pair.
//    Reads expert counts/offsets from GPU memory (no hipMemcpy needed).
//    Quantizes activations to signed symmetric INT4 on-the-fly.
// ================================================================
template<int GS = 64>
__global__ __launch_bounds__(32, 4)  // 64N tile, 4 blocks/CU
void wmma_expert_batched(
    const uint8_t* __restrict__ act_q,      // [total_tok, K/2] pre-quantized INT4
    const float* __restrict__ act_scales,    // [total_tok, ng]
    const uint8_t* __restrict__ wt_base,    // [E, K/8, N/16, 64] TILED per expert
    const __half* __restrict__ ws_base,     // [E, N, K/GS, 2] interleaved (scale, zero)
    __half* __restrict__ output,             // [total_tok, N]
    const int* __restrict__ expert_ids,      // [E] identity map (0,1,...,E-1)
    const int* __restrict__ offsets,          // [E] token offsets per expert
    const int* __restrict__ counts,           // [E] token counts per expert
    int64_t packed_stride,                   // N * K/2 (bytes between experts in wt_base)
    int64_t scales_stride,                   // ng * N_pad (floats between experts in ws_base)
    int N, int K)
{
    const int eidx = blockIdx.y;
    const int M_j = counts[eidx];
    if (M_j == 0) return;

    const int ej = expert_ids[eidx];
    const int x_off = offsets[eidx];
    const int N_pad = ((N + 15) / 16) * 16;
    const int ng = K / GS;

    const uint8_t* W = wt_base + (int64_t)ej * packed_stride;
    const __half* WS = ws_base + (int64_t)ej * scales_stride;

    const int lane = threadIdx.x;
    const int nl = lane % 16, kg = lane / 16;
    const int n_start = blockIdx.x * 64;  // 64N per block (4 WMMA)
    if (n_start >= N) return;

    for (int m0 = 0; m0 < M_j; m0 += 16) {
        int m_end = min(m0 + 16, M_j);
        int m_idx = x_off + m0 + nl;
        bool m_valid = (m0 + nl < m_end);

        float acc0[8]={}, acc1[8]={}, acc2[8]={}, acc3[8]={};

        for (int g = 0; g < ng; g++) {
            int k = g * GS;

            _v2i A; float a_sc = 0.0f;
            if (m_valid) {
                const uint8_t* aq = act_q + (int64_t)m_idx * (K/2) + k/2 + kg * 8;
                A = (_v2i){*(const int*)aq, *(const int*)(aq + 4)};
                a_sc = act_scales[m_idx * ng + g];
            } else { A = (_v2i){}; }

            // TILED weight load: W is [K/8, N/16, 64] per expert
            // COALESCED: 16 lanes × 4 bytes = 64 consecutive bytes = 1 cache line
            int NB_w = ((N + 15) / 16);
            #define LOAD_W(ti) _v2i B##ti; { \
                int nb = n_start/16 + ti; \
                int kc0 = k/8 + kg*2; \
                if (nb*16+nl < N) { \
                    B##ti = (_v2i){ \
                        (int)*(const uint32_t*)(W + (int64_t)(kc0*NB_w+nb)*64 + nl*4), \
                        (int)*(const uint32_t*)(W + (int64_t)((kc0+1)*NB_w+nb)*64 + nl*4)}; \
                } else { B##ti = (_v2i){}; } }
            LOAD_W(0) LOAD_W(1) LOAD_W(2) LOAD_W(3)
            #undef LOAD_W

            _v8i R0={}, R1={}, R2={}, R3={};
            R0=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,true,B0,R0,false);
            R1=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,true,B1,R1,false);
            R2=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,true,B2,R2,false);
            R3=__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true,A,true,B3,R3,false);

            #define DEQUANT(ti) { \
                int n_row = n_start + ti*16 + nl; \
                float ws = (n_row < N) ? __half2float(WS[((int64_t)n_row * ng + g) * 2]) : 0.0f; \
                float c = ws * a_sc; \
                for(int j=0;j<8;j++) acc##ti[j] += c * (float)R##ti[j]; }
            DEQUANT(0) DEQUANT(1) DEQUANT(2) DEQUANT(3)
            #undef DEQUANT
        }

        for (int j = 0; j < 8; j++) {
            int om = m0 + kg * 8 + j;
            if (om >= m_end) continue;
            int out_row = x_off + om;
            __half* row = output + (int64_t)out_row * N;
            if (n_start+nl<N)    row[n_start+nl]=__float2half(acc0[j]);
            if (n_start+16+nl<N) row[n_start+16+nl]=__float2half(acc1[j]);
            if (n_start+32+nl<N) row[n_start+32+nl]=__float2half(acc2[j]);
            if (n_start+48+nl<N) row[n_start+48+nl]=__float2half(acc3[j]);
        }
    }
}
