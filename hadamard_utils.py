"""Hadamard Block Rotation utilities for MXFP4 quantization.

Block-diagonal Hadamard rotation spreads outliers across 32-element blocks,
making weight distributions more uniform for better MXFP4 quantization.

Key property: H @ H^T = I (orthogonal), so rotation is lossless in FP16.
At inference time, we apply H to activations before each linear layer.
"""

import torch
import math


def hadamard_matrix(n):
    """Generate a normalized Hadamard matrix of size n (must be power of 2)."""
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    if n == 1:
        return torch.tensor([[1.0]])
    H_half = hadamard_matrix(n // 2)
    H = torch.cat([
        torch.cat([H_half, H_half], dim=1),
        torch.cat([H_half, -H_half], dim=1),
    ], dim=0)
    return H


_hadamard_cache = {}

def get_hadamard(block_size, device='cpu', dtype=torch.float32):
    """Get cached normalized Hadamard matrix."""
    key = (block_size, device, dtype)
    if key not in _hadamard_cache:
        H = hadamard_matrix(block_size) / math.sqrt(block_size)
        _hadamard_cache[key] = H.to(device=device, dtype=dtype)
    return _hadamard_cache[key]


def rotate_weight(W, block_size=32):
    """Apply block-diagonal Hadamard rotation to weight matrix W [N, K].

    W_rot = W @ block_diag(H, H, ..., H)

    This is equivalent to rotating each 32-column block independently:
    W_rot[:, i*32:(i+1)*32] = W[:, i*32:(i+1)*32] @ H
    """
    N, K = W.shape
    assert K % block_size == 0, f"K={K} must be divisible by block_size={block_size}"

    H = get_hadamard(block_size, device=W.device, dtype=W.dtype)

    # Reshape to blocks and apply rotation
    W_blocks = W.reshape(N, K // block_size, block_size)  # [N, num_blocks, 32]
    W_rot = torch.einsum('nbk,kj->nbj', W_blocks, H)     # [N, num_blocks, 32]
    return W_rot.reshape(N, K)


def rotate_activation(x, block_size=32):
    """Apply block-diagonal Hadamard rotation to activation x [..., K].

    x_rot = x @ block_diag(H, H, ..., H)

    Since H is orthogonal (H @ H^T = I), this is the same transform as for weights.
    At inference: y = W_rot @ x_rot = (W @ H) @ (H^T @ x)... wait, we need H^T on x.

    But H is symmetric (H = H^T for Hadamard), so H^T = H.
    Therefore: W_rot @ x_rot = (W @ H) @ (x @ H) ≠ W @ x

    Correct formulation:
    - W_rot = W @ H  (rotate columns of W)
    - x_rot = x @ H  (rotate x the same way)
    - y = W_rot @ x_rot^T = W @ H @ H^T @ x^T = W @ x^T  ✓

    Wait — for linear layers y = x @ W^T:
    - W_rot = W @ H  →  W_rot^T = H^T @ W^T = H @ W^T
    - y = x_rot @ W_rot^T = (x @ H) @ (H @ W^T) = x @ H @ H @ W^T
    - For this to equal x @ W^T, we need H @ H = I
    - Normalized Hadamard: H = H_raw / sqrt(n), so H @ H = H_raw @ H_raw / n = n*I/n = I  ✓
    """
    orig_shape = x.shape
    K = orig_shape[-1]
    assert K % block_size == 0, f"K={K} must be divisible by block_size={block_size}"

    H = get_hadamard(block_size, device=x.device, dtype=x.dtype)

    x_flat = x.reshape(-1, K)
    x_blocks = x_flat.reshape(-1, K // block_size, block_size)
    x_rot = torch.einsum('nbk,kj->nbj', x_blocks, H)
    return x_rot.reshape(orig_shape)
