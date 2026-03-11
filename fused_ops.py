"""
Fused Triton kernels for RMSNorm and RoPE — AMD RDNA4
Replaces slow PyTorch element-wise operations on tiny tensors.
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def rmsnorm_kernel(X_ptr, W_ptr, Y_ptr, D: tl.constexpr, eps: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x = tl.load(X_ptr + row * D + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0).to(tl.float32)

    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / D
    rrms = tl.math.rsqrt(mean_sq + eps)
    y = x * rrms * w
    tl.store(Y_ptr + row * D + cols, y.to(tl.float16), mask=mask)


def fused_rmsnorm(x, weight, eps=1e-6):
    """
    x: [*, D] float16
    weight: [D] float16
    Returns: [*, D] float16
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D).contiguous()
    N = x_2d.shape[0]
    y = torch.empty_like(x_2d)
    BLOCK_D = triton.next_power_of_2(D)
    rmsnorm_kernel[(N,)](x_2d, weight, y, D, eps=eps, BLOCK_D=BLOCK_D)
    return y.reshape(orig_shape)


@triton.jit
def rope_kernel(Q_ptr, K_ptr, COS_ptr, SIN_ptr, Q_out_ptr, K_out_ptr,
                pos_ptr, H, Hk, D, stride_cos, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    half_D = D // 2
    cols = tl.arange(0, BLOCK_D)
    mask = cols < half_D

    pos = tl.load(pos_ptr)
    cos_vals = tl.load(COS_ptr + pos * stride_cos + cols, mask=mask, other=1).to(tl.float32)
    sin_vals = tl.load(SIN_ptr + pos * stride_cos + cols, mask=mask, other=0).to(tl.float32)

    if pid < H:
        q_base = pid * D
        first_idx = cols
        second_idx = cols + half_D
        first_mask = first_idx < half_D
        second_mask = second_idx < D

        q_first = tl.load(Q_ptr + q_base + first_idx, mask=mask & first_mask, other=0).to(tl.float32)
        q_second = tl.load(Q_ptr + q_base + second_idx, mask=mask & second_mask, other=0).to(tl.float32)

        q_first_out = q_first * cos_vals - q_second * sin_vals
        q_second_out = q_second * cos_vals + q_first * sin_vals

        tl.store(Q_out_ptr + q_base + first_idx, q_first_out.to(tl.float16), mask=mask & first_mask)
        tl.store(Q_out_ptr + q_base + second_idx, q_second_out.to(tl.float16), mask=mask & second_mask)

    if pid < Hk:
        k_base = pid * D
        first_idx = cols
        second_idx = cols + half_D
        first_mask = first_idx < half_D
        second_mask = second_idx < D

        k_first = tl.load(K_ptr + k_base + first_idx, mask=mask & first_mask, other=0).to(tl.float32)
        k_second = tl.load(K_ptr + k_base + second_idx, mask=mask & second_mask, other=0).to(tl.float32)

        k_first_out = k_first * cos_vals - k_second * sin_vals
        k_second_out = k_second * cos_vals + k_first * sin_vals

        tl.store(K_out_ptr + k_base + first_idx, k_first_out.to(tl.float16), mask=mask & first_mask)
        tl.store(K_out_ptr + k_base + second_idx, k_second_out.to(tl.float16), mask=mask & second_mask)


@triton.jit
def qknorm_rope_kernel(Q_ptr, K_ptr, QW_ptr, KW_ptr, COS_ptr, SIN_ptr,
                       Q_out_ptr, K_out_ptr, pos_ptr, H, Hk, D,
                       stride_cos, eps: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    half_D = D // 2
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    pos = tl.load(pos_ptr)
    half_cols = tl.arange(0, BLOCK_D)
    half_mask = half_cols < half_D
    cos_vals = tl.load(COS_ptr + pos * stride_cos + half_cols, mask=half_mask, other=1).to(tl.float32)
    sin_vals = tl.load(SIN_ptr + pos * stride_cos + half_cols, mask=half_mask, other=0).to(tl.float32)

    if pid < H:
        q_base = pid * D
        q_raw = tl.load(Q_ptr + q_base + cols, mask=mask, other=0).to(tl.float32)
        qw = tl.load(QW_ptr + cols, mask=mask, other=0).to(tl.float32)

        q_sq = q_raw * q_raw
        mean_sq = tl.sum(q_sq, axis=0) / D
        rrms = tl.math.rsqrt(mean_sq + eps)
        q_normed = q_raw * rrms * qw

        # RoPE on normalized Q
        q_first_vals = tl.load(Q_ptr + q_base + half_cols, mask=half_mask, other=0).to(tl.float32)
        q_first_vals = q_first_vals * rrms
        qw_first = tl.load(QW_ptr + half_cols, mask=half_mask, other=0).to(tl.float32)
        q_first_vals = q_first_vals * qw_first

        q_second = tl.load(Q_ptr + q_base + half_cols + half_D, mask=half_mask, other=0).to(tl.float32)
        qw_second = tl.load(QW_ptr + half_cols + half_D, mask=half_mask, other=0).to(tl.float32)
        q_second = q_second * rrms * qw_second

        q_first_out = q_first_vals * cos_vals - q_second * sin_vals
        q_second_out = q_second * cos_vals + q_first_vals * sin_vals

        tl.store(Q_out_ptr + q_base + half_cols, q_first_out.to(tl.float16), mask=half_mask)
        tl.store(Q_out_ptr + q_base + half_cols + half_D, q_second_out.to(tl.float16), mask=half_mask)

    if pid < Hk:
        k_base = pid * D
        k_raw = tl.load(K_ptr + k_base + cols, mask=mask, other=0).to(tl.float32)
        kw = tl.load(KW_ptr + cols, mask=mask, other=0).to(tl.float32)

        k_sq = k_raw * k_raw
        mean_sq_k = tl.sum(k_sq, axis=0) / D
        rrms_k = tl.math.rsqrt(mean_sq_k + eps)
        k_normed = k_raw * rrms_k * kw

        k_first_vals = tl.load(K_ptr + k_base + half_cols, mask=half_mask, other=0).to(tl.float32)
        kw_first = tl.load(KW_ptr + half_cols, mask=half_mask, other=0).to(tl.float32)
        k_first_vals = k_first_vals * rrms_k * kw_first

        k_second = tl.load(K_ptr + k_base + half_cols + half_D, mask=half_mask, other=0).to(tl.float32)
        kw_second = tl.load(KW_ptr + half_cols + half_D, mask=half_mask, other=0).to(tl.float32)
        k_second = k_second * rrms_k * kw_second

        k_first_out = k_first_vals * cos_vals - k_second * sin_vals
        k_second_out = k_second * cos_vals + k_first_vals * sin_vals

        tl.store(K_out_ptr + k_base + half_cols, k_first_out.to(tl.float16), mask=half_mask)
        tl.store(K_out_ptr + k_base + half_cols + half_D, k_second_out.to(tl.float16), mask=half_mask)


def fused_qknorm_rope_decode(q, k, q_norm_weight, k_norm_weight, cos, sin, position, eps=1e-6):
    """Fused QK-Norm + RoPE in a single kernel launch.
    q: [1, H, 1, D], k: [1, Hk, 1, D], weights: [D]
    Returns: (q_normed_rotated, k_normed_rotated)"""
    B, H, S, D = q.shape
    _, Hk, _, _ = k.shape

    q_flat = q.reshape(H, D).contiguous()
    k_flat = k.reshape(Hk, D).contiguous()
    q_out = torch.empty_like(q_flat)
    k_out = torch.empty_like(k_flat)

    if isinstance(position, int):
        pos_tensor = torch.tensor([position], device=q.device, dtype=torch.int64)
    else:
        pos_tensor = position.reshape(-1).to(torch.int64)

    BLOCK_D = triton.next_power_of_2(D)
    stride_cos = cos.stride(0)
    grid = (max(H, Hk),)

    qknorm_rope_kernel[grid](
        q_flat, k_flat, q_norm_weight, k_norm_weight,
        cos, sin, q_out, k_out, pos_tensor,
        H, Hk, D, stride_cos, eps=eps, BLOCK_D=BLOCK_D
    )
    return q_out.reshape(B, H, S, D), k_out.reshape(B, Hk, S, D)


def fused_rope_decode(q, k, cos, sin, position):
    """
    Apply RoPE to q and k for single-token decode.
    q: [1, H, 1, D] float16
    k: [1, Hk, 1, D] float16
    cos: [max_seq, D//2] float16
    sin: [max_seq, D//2] float16
    position: int or 0-dim/1-element int64 GPU tensor
    Returns: (q_rotated, k_rotated) same shapes
    """
    B, H, S, D = q.shape
    _, Hk, _, _ = k.shape

    q_flat = q.reshape(H, D).contiguous()
    k_flat = k.reshape(Hk, D).contiguous()
    q_out = torch.empty_like(q_flat)
    k_out = torch.empty_like(k_flat)

    if isinstance(position, int):
        pos_tensor = torch.tensor([position], device=q.device, dtype=torch.int64)
    else:
        pos_tensor = position.reshape(-1).to(torch.int64)

    BLOCK_D = triton.next_power_of_2(D // 2)
    stride_cos = cos.stride(0)
    grid = (max(H, Hk),)

    rope_kernel[grid](
        q_flat, k_flat, cos, sin, q_out, k_out, pos_tensor,
        H, Hk, D, stride_cos, BLOCK_D=BLOCK_D
    )
    return q_out.reshape(B, H, S, D), k_out.reshape(B, Hk, S, D)


if __name__ == '__main__':
    print('=' * 60)
    print('Fused RMSNorm + RoPE — Triton kernels')
    print('=' * 60)

    # Quick self-test
    device = 'cuda'
    D = 5120
    x = torch.randn(1, D, device=device, dtype=torch.float16) * 0.1
    w = torch.randn(D, device=device, dtype=torch.float16)
    eps = 1e-6

    x_f = x.float()
    rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    ref = (x_f * rms).half() * w
    out = fused_rmsnorm(x, w, eps)
    cos_sim = torch.nn.functional.cosine_similarity(
        ref.flatten().float().unsqueeze(0),
        out.flatten().float().unsqueeze(0)
    ).item()
    print(f'RMSNorm D={D}: cos={cos_sim:.6f} {"PASS" if cos_sim > 0.999 else "FAIL"}')
