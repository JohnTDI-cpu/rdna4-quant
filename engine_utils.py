"""
Shared utility classes for INT4 inference engine.

Provides: KVCache, RMSNorm, precompute_rope_freqs, apply_rope.
Used by int4_engine_v5.py and api_server.py.
"""

import torch
import torch.nn.functional as F

try:
    from fused_ops import fused_rmsnorm
except ImportError:
    fused_rmsnorm = None


# ============================================================
# Pre-allocated KV Cache
# ============================================================

class KVCache:
    """Pre-allocated KV cache — eliminates torch.cat per decode step."""

    def __init__(self, num_layers, batch_size, max_seq_len, num_kv_heads, head_dim, device, dtype=torch.float16):
        self.max_seq_len = max_seq_len
        self.current_len = 0
        self.k_caches = []
        self.v_caches = []
        for _ in range(num_layers):
            self.k_caches.append(torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype))
            self.v_caches.append(torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype))

    def update(self, layer_idx, k_new, v_new):
        """Write new K/V into cache, return full view up to current position."""
        seq_len = k_new.shape[2]
        start = self.current_len
        end = start + seq_len
        self.k_caches[layer_idx][:, :, start:end, :] = k_new
        self.v_caches[layer_idx][:, :, start:end, :] = v_new
        return self.k_caches[layer_idx][:, :, :end, :], self.v_caches[layer_idx][:, :, :end, :]

    def advance(self, n=1):
        self.current_len += n

    def reset(self):
        self.current_len = 0


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm:
    def __init__(self, weight, eps=1e-6):
        self.weight = weight.half()
        self.eps = eps

    def forward(self, x):
        if fused_rmsnorm is not None and x.is_cuda:
            return fused_rmsnorm(x, self.weight, self.eps)
        # Fallback: pure PyTorch (CPU or when Triton unavailable)
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + self.eps) * self.weight.float()).to(x.dtype)


# ============================================================
# Rotary Position Embeddings (RoPE)
# ============================================================

def precompute_rope_freqs(dim, max_seq_len, theta=1000000.0, device='cuda'):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos().half()
    sin = freqs.sin().half()
    return cos, sin  # [max_seq_len, dim//2]


def apply_rope(x, cos, sin, position_ids):
    """Apply RoPE to x: [batch, n_heads, seq_len, head_dim]
    Uses half-split convention (HuggingFace/Qwen2): pairs (i, i+d/2)."""
    # position_ids: [batch, seq_len]
    head_dim = x.shape[3]
    rope_dim = cos.shape[1] * 2  # cos has dim//2 cols

    # Only rotate first rope_dim dimensions
    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]

    # Gather cos/sin for positions
    cos_pos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, dim//2]
    sin_pos = sin[position_ids].unsqueeze(1)

    # Half-split: first half and second half (HuggingFace convention)
    half = rope_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]

    # Apply rotation
    out1 = x1 * cos_pos - x2 * sin_pos
    out2 = x2 * cos_pos + x1 * sin_pos

    x_rot_out = torch.cat([out1, out2], dim=-1)

    if x_pass.shape[-1] > 0:
        return torch.cat([x_rot_out, x_pass], dim=-1)
    return x_rot_out
