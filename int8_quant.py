"""
INT8 Symmetric Quantization for mixed-precision W4A16/W8A16 inference.

Symmetric INT8: 256 levels [-128..+127], FP16 per-block scale.
Dequant: val = int8_signed * scale_fp16

Used for sensitive layers identified by sensitivity analysis.
"""

import torch
import torch.nn.functional as F


def quantize_to_int8_symmetric(block_f32, block_size=32):
    """Symmetric INT8 quantization: [-128..+127] with FP16 per-block scale.

    Args:
        block_f32: [..., block_size] tensor of FP32 weights

    Returns:
        quantized: [..., block_size] int8 tensor with values in [-128, 127]
        scale: [...] FP16 per-block scale
        reconstructed: [..., block_size] FP32 reconstructed weights
    """
    max_abs = block_f32.abs().amax(dim=-1, keepdim=True)
    scale = max_abs / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    scale = scale.clamp(max=65504.0)

    quantized = (block_f32 / scale).round().clamp(-128, 127).to(torch.int8)
    reconstructed = quantized.float() * scale

    return quantized, scale.squeeze(-1).half(), reconstructed


def find_optimal_scale_int8(block_f32):
    """Find optimal FP16 scale for INT8 symmetric quantization."""
    max_abs = block_f32.abs().amax(dim=-1)
    scale = max_abs / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale.clamp(max=65504.0).half()


def quantize_to_int8_block(block_f32, scale_fp16):
    """Quantize block to INT8 given FP16 scale."""
    scale = scale_fp16.float().unsqueeze(-1)
    quantized = (block_f32 / scale).round().clamp(-128, 127).to(torch.int8)
    reconstructed = quantized.float() * scale
    return quantized, reconstructed


def dequantize_int8(weights_int8, scale_fp16, block_size=32):
    """Dequantize INT8 weights to FP16.

    Args:
        weights_int8: [N, K] int8 weights
        scale_fp16: [N, K//block_size] FP16 per-block scales
        block_size: elements per scale block

    Returns:
        W_fp16: [N, K] FP16 weights
    """
    N, K = weights_int8.shape
    num_blocks = K // block_size

    q_blocks = weights_int8.reshape(N, num_blocks, block_size).float()
    scales = scale_fp16.float().unsqueeze(-1)

    return (q_blocks * scales).reshape(N, K).half()


def gptq_quantize_int8(W_fp16, H, block_size=32, device='cuda'):
    """GPTQ-quantize W [N, K] to INT8 symmetric using Hessian H [K, K].

    Same GPTQ algorithm as INT4 but with INT8 quantization:
    - 256 levels [-128..+127] instead of 16
    - Much lower quantization error per column
    - FP16 per-block scales

    Returns: W_int8 [N, K_pad], W_scales [N, K_pad//block_size], K_pad
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

    # Damping
    diag_mean = H.diag().mean().item()
    damp = max(0.01 * diag_mean, 1e-6)

    # Cholesky
    H_inv = None
    for damp_mult in [1.0, 10.0, 100.0, 1000.0]:
        try:
            H_try = H + (damp * damp_mult) * torch.eye(K_pad, device=device, dtype=H.dtype)
            L = torch.linalg.cholesky(H_try)
            H_inv = torch.cholesky_inverse(L)
            break
        except Exception:
            continue
    if H_inv is None:
        H_inv = (1.0 / (diag_mean + damp * 1000)) * torch.eye(K_pad, device=device, dtype=H.dtype)

    # Storage
    W_int8 = torch.zeros(N, K_pad, dtype=torch.int8, device=device)
    W_scales = torch.zeros(N, num_blocks, dtype=torch.float16, device=device)

    # GPTQ: process in blocks
    for bi in range(num_blocks):
        col_start = bi * block_size
        col_end = col_start + block_size

        H_inv_block = H_inv[col_start:col_end, col_start:col_end]
        W_block = W[:, col_start:col_end].clone()

        scale = find_optimal_scale_int8(W_block)
        W_scales[:, bi] = scale

        q, W_quant = quantize_to_int8_block(W_block, scale)
        W_int8[:, col_start:col_end] = q

        Err = W_block - W_quant

        if bi < num_blocks - 1:
            try:
                H_inv_block_inv = torch.linalg.inv(H_inv_block)
            except Exception:
                H_inv_block_inv = torch.linalg.pinv(H_inv_block)

            H_inv_br = H_inv[col_start:col_end, col_end:]
            delta = Err @ H_inv_block_inv @ H_inv_br
            w_remaining_scale = W[:, col_end:].abs().mean(dim=0, keepdim=True).clamp(min=1e-6)
            delta = delta.clamp(-5 * w_remaining_scale, 5 * w_remaining_scale)
            W[:, col_end:] -= delta

    return W_int8, W_scales, K_pad
