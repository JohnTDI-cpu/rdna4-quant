"""
INT4 Quantization v2 — Asymmetric + Fixed GPTQ.

Key changes from v1:
  1. Asymmetric quantization (zero-point offset) instead of symmetric
  2. GPTQ error clamping relaxed (Hessian-aware bounds)
  3. Better Hessian conditioning (diagonal regularization)

Dequant: val = (int4_unsigned - zero_point) * scale_fp16
Range: [0..15] mapped to [min, max] per block
"""

import torch
import torch.nn.functional as F


def quantize_to_int4_asymmetric(block_f32, block_size=32):
    """Asymmetric INT4 quantization: [0..15] with FP16 scale + zero-point.

    Args:
        block_f32: [..., block_size] tensor of FP32 weights

    Returns:
        quantized: [..., block_size] uint8 tensor with values in [0, 15]
        scale: [...] FP16 per-block scale
        zero_point: [...] uint8 per-block zero point
        reconstructed: [..., block_size] FP32 reconstructed weights
    """
    vmin = block_f32.amin(dim=-1, keepdim=True)  # [..., 1]
    vmax = block_f32.amax(dim=-1, keepdim=True)  # [..., 1]

    # Scale: maps [vmin, vmax] to [0, 15]
    scale = (vmax - vmin) / 15.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    scale = scale.clamp(max=65504.0)

    # Zero point: the integer that maps to 0.0
    # zero_point = round(-vmin / scale), clamped to [0, 15]
    zero_point = (-vmin / scale).round().clamp(0, 15)

    # Quantize
    quantized = ((block_f32 / scale) + zero_point).round().clamp(0, 15).to(torch.uint8)

    # Reconstruct
    reconstructed = (quantized.float() - zero_point) * scale

    return quantized, scale.squeeze(-1).half(), zero_point.squeeze(-1).to(torch.uint8), reconstructed


def find_optimal_scale_int4_asym(block_f32):
    """Find optimal FP16 scale and zero-point for asymmetric INT4.

    Returns:
        scale: [...] FP16
        zero_point: [...] uint8
    """
    vmin = block_f32.amin(dim=-1)
    vmax = block_f32.amax(dim=-1)

    scale = (vmax - vmin) / 15.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    scale = scale.clamp(max=65504.0)

    zero_point = (-vmin / scale).round().clamp(0, 15)

    # Convert to FP16 precision
    scale_fp16 = scale.half().float()
    zero_point_u8 = zero_point.to(torch.uint8)

    return scale_fp16.half(), zero_point_u8


def quantize_to_int4_block_asym(block_f32, scale_fp16, zero_point):
    """Quantize block given scale and zero-point (asymmetric).

    Returns:
        quantized: [..., block_size] uint8 in [0, 15]
        reconstructed: [..., block_size] FP32
    """
    scale = scale_fp16.float().unsqueeze(-1)
    zp = zero_point.float().unsqueeze(-1)
    quantized = ((block_f32 / scale) + zp).round().clamp(0, 15).to(torch.uint8)
    reconstructed = (quantized.float() - zp) * scale
    return quantized, reconstructed


def pack_int4_unsigned(quantized):
    """Pack [..., N] unsigned int4 (0..15) into [..., N//2] uint8 bytes.

    Packing: byte = lo | (hi << 4)
    """
    lo = quantized[..., 0::2] & 0x0F
    hi = quantized[..., 1::2] & 0x0F
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_int4_unsigned(packed):
    """Unpack [..., N//2] uint8 bytes into [..., N] unsigned int4 (0..15) as uint8."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    result = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return result


def dequantize_int4_asym(packed, scale_fp16, zero_point, block_size=32):
    """Dequantize packed asymmetric INT4 weights to FP16.

    Args:
        packed: [N, K//2] uint8
        scale_fp16: [N, K//block_size] FP16
        zero_point: [N, K//block_size] uint8
        block_size: 32

    Returns:
        W_fp16: [N, K] FP16
    """
    N = packed.shape[0]
    K = packed.shape[1] * 2
    num_blocks = K // block_size

    q = unpack_int4_unsigned(packed)  # [N, K] uint8 in [0..15]
    q_blocks = q.reshape(N, num_blocks, block_size).float()
    scales = scale_fp16.float().unsqueeze(-1)  # [N, num_blocks, 1]
    zps = zero_point.float().unsqueeze(-1)     # [N, num_blocks, 1]

    W = ((q_blocks - zps) * scales).reshape(N, K).half()
    return W


# ---- Keep symmetric versions for backward compat ----
def quantize_to_int4_symmetric(block_f32, block_size=32):
    """Symmetric INT4 quantization (v1 compat)."""
    max_abs = block_f32.abs().amax(dim=-1, keepdim=True)
    scale = max_abs / 7.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    scale = scale.clamp(max=65504.0)
    quantized = (block_f32 / scale).round().clamp(-8, 7).to(torch.int8)
    reconstructed = quantized.float() * scale
    return quantized, scale.squeeze(-1).half(), reconstructed


def find_optimal_scale_int4(block_f32):
    max_abs = block_f32.abs().amax(dim=-1)
    scale_exact = max_abs / 7.0
    scale_exact = torch.where(scale_exact > 0, scale_exact, torch.ones_like(scale_exact))
    scale_exact = scale_exact.clamp(max=65504.0)
    return scale_exact.half()


def quantize_to_int4_block(block_f32, scale_fp16):
    scale = scale_fp16.float().unsqueeze(-1)
    quantized = (block_f32 / scale).round().clamp(-8, 7).to(torch.int8)
    reconstructed = quantized.float() * scale
    return quantized, reconstructed


def pack_int4(quantized):
    q = quantized.to(torch.int8)
    lo = q[..., 0::2] & 0x0F
    hi = q[..., 1::2] & 0x0F
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_int4(packed):
    lo = (packed & 0x0F).to(torch.int8)
    hi = ((packed >> 4) & 0x0F).to(torch.int8)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    result = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return result


# ---- GPTQ v2: Asymmetric + Fixed Error Propagation ----

def gptq_quantize_int4_v2(W_fp16, H, block_size=32, device='cuda', asymmetric=True):
    """
    GPTQ v2: Asymmetric INT4 with fixed error propagation.

    Changes from v1:
      1. Asymmetric quantization (zero-point) by default
      2. No aggressive delta clamping — let GPTQ do its job
      3. Better Hessian conditioning (diagonal + small regularization)

    Returns: W_packed [N, K_pad//2], W_scales [N, K_pad//block_size],
             W_zeros [N, K_pad//block_size] (uint8), K_pad
    """
    W = W_fp16.float().to(device)
    N, K = W.shape
    K_pad = ((K + block_size - 1) // block_size) * block_size
    if K_pad > K:
        W = F.pad(W, (0, K_pad - K))
        H_pad = torch.zeros(K_pad, K_pad, device=device, dtype=torch.float32)
        H_pad[:K, :K] = H.float().to(device)
        H = H_pad
    else:
        H = H.float().to(device)

    num_blocks = K_pad // block_size

    # Damping — gentler: use diagonal mean with minimal regularization
    diag_mean = H.diag().mean().item()
    damp = max(0.01 * diag_mean, 1e-6)

    # Cholesky — try minimal damping first, escalate less aggressively
    H_inv = None
    for damp_mult in [1.0, 2.0, 5.0, 10.0, 50.0]:
        try:
            H_try = H + (damp * damp_mult) * torch.eye(K_pad, device=device, dtype=H.dtype)
            L = torch.linalg.cholesky(H_try)
            H_inv = torch.cholesky_inverse(L)
            if damp_mult > 1.0:
                print(f"    [INFO] Cholesky succeeded with {damp_mult}x damping")
            break
        except Exception:
            continue
    if H_inv is None:
        print("    [WARN] Cholesky failed, using diagonal Hessian")
        H_diag = H.diag().clamp(min=1e-8)
        H_inv = torch.diag(1.0 / H_diag)

    # Storage
    W_packed = torch.zeros(N, K_pad // 2, dtype=torch.uint8, device=device)
    W_scales = torch.zeros(N, num_blocks, dtype=torch.float16, device=device)
    W_zeros = torch.zeros(N, num_blocks, dtype=torch.uint8, device=device)

    # GPTQ: process blocks of columns
    for bi in range(num_blocks):
        col_start = bi * block_size
        col_end = col_start + block_size

        H_inv_block = H_inv[col_start:col_end, col_start:col_end]
        W_block = W[:, col_start:col_end].clone()  # [N, block_size]

        if asymmetric:
            # Asymmetric: [0..15] with zero-point
            scale, zp = find_optimal_scale_int4_asym(W_block)
            W_scales[:, bi] = scale
            W_zeros[:, bi] = zp
            q, W_quant = quantize_to_int4_block_asym(W_block, scale, zp)
            packed = pack_int4_unsigned(q)
        else:
            # Symmetric fallback
            scale = find_optimal_scale_int4(W_block)
            W_scales[:, bi] = scale
            W_zeros[:, bi] = 0  # No zero-point for symmetric
            q, W_quant = quantize_to_int4_block(W_block, scale)
            packed = pack_int4(q)

        W_packed[:, bi * (block_size // 2):(bi + 1) * (block_size // 2)] = packed

        # GPTQ error compensation — NO aggressive clamping
        Err = W_block - W_quant  # [N, block_size]

        if bi < num_blocks - 1:
            try:
                H_inv_block_inv = torch.linalg.inv(H_inv_block)
            except Exception:
                H_inv_block_inv = torch.linalg.pinv(H_inv_block)

            H_inv_br = H_inv[col_start:col_end, col_end:]
            delta = Err @ H_inv_block_inv @ H_inv_br  # [N, remaining]

            # Soft clamp: only clip extreme outliers (50x instead of 5x)
            # This preserves most of GPTQ's error compensation
            w_scale = W[:, col_end:].abs().amax(dim=0, keepdim=True).clamp(min=1e-6)
            delta = delta.clamp(-50 * w_scale, 50 * w_scale)

            W[:, col_end:] -= delta

    return W_packed, W_scales, W_zeros, K_pad
