"""
MXFP4 Full Inference Engine for AMD RDNA4 (gfx1201)

Complete text generation pipeline:
1. Load HuggingFace model weights
2. Quantize all linear layers to MXFP4 on-the-fly (or load cached)
3. Run full transformer forward pass with MXFP4 GEMM kernel
4. KV-cache, RoPE, RMSNorm, SiLU — all in FP16
5. Token generation with sampling

Target model: Qwen2ForCausalLM (Qwen2.5-1.5B/3B/7B/14B)
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mxfp4_gemm_v6 import mxfp4_gemm_v6
from mxfp4_gemv import mxfp4_gemv
from mxfp4_quant_gpu import quantize_weights_gpu
from fused_ops import fused_rmsnorm, fused_rope_decode
from mxfp4_fused_gemv import fused_qkv_gemv, fused_gate_up_gemv

# Try to load HIP extension for fused ops
try:
    import mxfp4_hip_gemv as _hip_ext
except ImportError:
    _hip_ext = None

# Try to load fused WMMA GEMM (best for M <= 32)
try:
    import mxfp4_wmma_gemm as _wmma_ext
except ImportError:
    _wmma_ext = None


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
# MXFP4 Linear Layer
# ============================================================

class MXFP4Linear:
    """Drop-in replacement for nn.Linear using MXFP4 weights."""

    def __init__(self, W_packed, W_scales, N, K, bias=None):
        self.W_packed = W_packed   # [N, K//2] uint8 on GPU
        self.W_scales = W_scales   # [N, K//32] uint8 on GPU
        self.N = N
        self.K = K
        self.bias = bias           # [N] float16 on GPU or None

    @staticmethod
    def from_fp16(weight_fp16, bias=None, device='cuda'):
        """Convert FP16 weight tensor to MXFP4 on GPU."""
        W = weight_fp16.detach().half().to(device)
        W_packed, W_scales, K_pad = quantize_weights_gpu(W, device)

        b = None
        if bias is not None:
            b = bias.detach().half().to(device)
        return MXFP4Linear(W_packed, W_scales, W.shape[0], K_pad, b)

    def forward(self, x):
        """x: [batch, seq_len, K] or [batch, K] -> [batch, ..., N]"""
        orig_shape = x.shape
        K_in = orig_shape[-1]

        # Pad input if needed
        if K_in < self.K:
            x = F.pad(x, (0, self.K - K_in))

        # Flatten to 2D for GEMM/GEMV
        x_2d = x.reshape(-1, self.K)
        M = x_2d.shape[0]
        if M <= 1:
            y = mxfp4_gemv(x_2d.half(), self.W_packed, self.W_scales, self.N, self.K)
        else:
            # Triton fused dequant+GEMM — faster than WMMA HIP for all M
            y = mxfp4_gemm_v6(x_2d.half(), self.W_packed, self.W_scales, self.N, self.K)

        # Reshape back
        out_shape = list(orig_shape[:-1]) + [self.N]
        y = y.reshape(out_shape)

        if self.bias is not None:
            y = y + self.bias

        return y

    def memory_bytes(self):
        return self.W_packed.numel() + self.W_scales.numel()


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm:
    def __init__(self, weight, eps=1e-6):
        self.weight = weight.half()
        self.eps = eps

    def forward(self, x):
        return fused_rmsnorm(x, self.weight, self.eps)


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


# ============================================================
# Qwen2 Transformer Block
# ============================================================

class Qwen2Attention:
    def __init__(self, q_proj, k_proj, v_proj, o_proj,
                 num_heads, num_kv_heads, head_dim):
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self._rope_pos_buf = None  # pre-allocated position buffer for RoPE

    def forward(self, hidden, cos, sin, position_ids, kv_cache=None):
        B, S, D = hidden.shape

        if S == 1:
            # Decode: fused QKV + fused RoPE
            q, k, v = fused_qkv_gemv(hidden.reshape(B, -1), self.q_proj, self.k_proj, self.v_proj)
            # Add biases if present
            if self.q_proj.bias is not None:
                q = q + self.q_proj.bias
            if self.k_proj.bias is not None:
                k = k + self.k_proj.bias
            if self.v_proj.bias is not None:
                v = v + self.v_proj.bias
            q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
            # Use pre-allocated position buffer for RoPE (avoids tensor creation per call)
            if self._rope_pos_buf is None:
                self._rope_pos_buf = torch.empty(1, dtype=torch.int64, device=hidden.device)
            self._rope_pos_buf[0] = position_ids[0, 0]
            q, k = fused_rope_decode(q, k, cos, sin, self._rope_pos_buf)
        else:
            # Prefill: separate projections + PyTorch RoPE
            q = self.q_proj.forward(hidden).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj.forward(hidden).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj.forward(hidden).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = apply_rope(q, cos, sin, position_ids)
            k = apply_rope(k, cos, sin, position_ids)

        # KV cache
        if kv_cache is not None:
            if isinstance(kv_cache, KVCache):
                # Pre-allocated cache: write-in-place
                k_full, v_full = kv_cache.update(self._layer_idx, k, v)
            else:
                # Legacy tuple cache (prefill)
                k_cache, v_cache = kv_cache
                k_full = torch.cat([k_cache, k], dim=2)
                v_full = torch.cat([v_cache, v], dim=2)
        else:
            k_full, v_full = k, v

        new_kv = (k_full, v_full)

        # Scaled dot-product attention with native GQA support
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k_full, v_full, is_causal=(S > 1), enable_gqa=True
        )

        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj.forward(attn), new_kv


class Qwen2MLP:
    def __init__(self, gate_proj, up_proj, down_proj):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x):
        B_S = x.shape[:-1]
        M = 1
        for d in B_S:
            M *= d
        if M <= 1:
            # Decode: fused gate+up
            gate, up = fused_gate_up_gemv(x.reshape(1, -1), self.gate_proj, self.up_proj)
            if _hip_ext is not None:
                activated = _hip_ext.silu_mul(gate, up).view(*B_S, -1)
            else:
                gate = gate.view(*B_S, -1)
                up = up.view(*B_S, -1)
                activated = F.silu(gate) * up
        else:
            gate = self.gate_proj.forward(x)
            up = self.up_proj.forward(x)
            activated = F.silu(gate) * up
        return self.down_proj.forward(activated)


class Qwen2Block:
    def __init__(self, attn, mlp, input_norm, post_norm):
        self.attn = attn
        self.mlp = mlp
        self.input_norm = input_norm
        self.post_norm = post_norm

    def forward(self, hidden, cos, sin, position_ids, kv_cache=None):
        B, S, D = hidden.shape
        if S == 1 and _hip_ext is not None:
            # Decode path: fuse residual_add + RMSNorm where possible
            residual = hidden
            normed = self.input_norm.forward(hidden)
            attn_out, new_kv = self.attn.forward(normed, cos, sin, position_ids, kv_cache)

            # Fused: hidden = residual + attn_out, then RMSNorm(hidden)
            post_normed, hidden = _hip_ext.residual_rmsnorm(attn_out, residual, self.post_norm.weight, self.post_norm.eps)

            mlp_out = self.mlp.forward(post_normed)
            hidden = hidden + mlp_out
        else:
            residual = hidden
            hidden = self.input_norm.forward(hidden)
            hidden, new_kv = self.attn.forward(hidden, cos, sin, position_ids, kv_cache)
            hidden = residual + hidden

            residual = hidden
            hidden = self.post_norm.forward(hidden)
            hidden = self.mlp.forward(hidden)
            hidden = residual + hidden

        return hidden, new_kv


# ============================================================
# Qwen3 MoE Support
# ============================================================

class Qwen3Attention:
    """Qwen3 attention: like Qwen2 but with QK normalization, no bias."""
    def __init__(self, q_proj, k_proj, v_proj, o_proj,
                 num_heads, num_kv_heads, head_dim, q_norm_w, k_norm_w, rms_eps):
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.q_norm = RMSNorm(q_norm_w, rms_eps)
        self.k_norm = RMSNorm(k_norm_w, rms_eps)
        self._rope_pos_buf = None
        self._layer_idx = 0

    def forward(self, hidden, cos, sin, position_ids, kv_cache=None):
        B, S, D = hidden.shape

        q = self.q_proj.forward(hidden).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj.forward(hidden).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj.forward(hidden).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK normalization (per-head RMSNorm on head_dim)
        q = self.q_norm.forward(q)
        k = self.k_norm.forward(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # RoPE
        q = apply_rope(q, cos, sin, position_ids)
        k = apply_rope(k, cos, sin, position_ids)

        # KV cache
        if kv_cache is not None:
            if isinstance(kv_cache, KVCache):
                k_full, v_full = kv_cache.update(self._layer_idx, k, v)
            else:
                k_cache, v_cache = kv_cache
                k_full = torch.cat([k_cache, k], dim=2)
                v_full = torch.cat([v_cache, v], dim=2)
        else:
            k_full, v_full = k, v

        new_kv = (k_full, v_full)

        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k_full, v_full, is_causal=(S > 1), enable_gqa=True
        )
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj.forward(attn), new_kv


class Qwen3MoEMLP:
    """MoE feed-forward: router + N experts, top-k selection."""
    def __init__(self, router_weight, experts, num_experts_per_tok, norm_topk_prob=True):
        self.router_weight = router_weight  # [num_experts, hidden_size] FP16
        self.experts = experts              # list of (gate_proj, up_proj, down_proj) MXFP4Linear
        self.num_experts = len(experts)
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

    def forward(self, x):
        """x: [B, S, D] -> [B, S, D]"""
        orig_shape = x.shape
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)  # [M, D]
        M = x_flat.shape[0]

        # Router: compute logits and select top-k experts
        router_logits = F.linear(x_flat.float(), self.router_weight.float())  # [M, num_experts]
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)  # [M, k]

        if self.norm_topk_prob:
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        topk_probs = topk_probs.half()

        # For M=1 decode: simple loop over selected experts
        output = torch.zeros_like(x_flat)
        for token_idx in range(M):
            for k_idx in range(self.num_experts_per_tok):
                expert_idx = topk_indices[token_idx, k_idx].item()
                weight = topk_probs[token_idx, k_idx]
                gate_proj, up_proj, down_proj = self.experts[expert_idx]
                gate = gate_proj.forward(x_flat[token_idx:token_idx+1])
                up = up_proj.forward(x_flat[token_idx:token_idx+1])
                activated = F.silu(gate) * up
                expert_out = down_proj.forward(activated)
                output[token_idx] += weight * expert_out.squeeze(0)

        return output.reshape(orig_shape)


class Qwen3MoEBlock:
    """Qwen3 MoE transformer block."""
    def __init__(self, attn, moe_mlp, input_norm, post_norm):
        self.attn = attn
        self.mlp = moe_mlp  # Qwen3MoEMLP
        self.input_norm = input_norm
        self.post_norm = post_norm

    def forward(self, hidden, cos, sin, position_ids, kv_cache=None):
        residual = hidden
        hidden = self.input_norm.forward(hidden)
        hidden, new_kv = self.attn.forward(hidden, cos, sin, position_ids, kv_cache)
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_norm.forward(hidden)
        hidden = self.mlp.forward(hidden)
        hidden = residual + hidden

        return hidden, new_kv


# ============================================================
# Full Model
# ============================================================

class _MXFP4EmbedWeight:
    """Fake weight tensor for MXFP4Embedding — provides .device, .numel(), etc."""
    def __init__(self, mxfp4):
        self._mxfp4 = mxfp4
        self.data = self
    @property
    def device(self):
        return self._mxfp4.W_packed.device
    @property
    def shape(self):
        return (self._mxfp4.N, self._mxfp4.K)
    def numel(self):
        return self._mxfp4.N * self._mxfp4.K
    def size(self, dim=None):
        s = (self._mxfp4.N, self._mxfp4.K)
        return s[dim] if dim is not None else s
    def cpu(self):
        return None  # not supported for save — use MXFP4 format
    def element_size(self):
        return 2  # pretend FP16 for memory reporting


class MXFP4Embedding:
    """Embedding lookup using MXFP4 weights — dequants rows on the fly."""

    FP4_LUT = torch.tensor([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                             0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float16)

    def __init__(self, mxfp4_linear):
        self.mxfp4 = mxfp4_linear
        self.weight = _MXFP4EmbedWeight(mxfp4_linear)
        self.device = mxfp4_linear.W_packed.device

    def __call__(self, input_ids):
        """Dequant embedding rows for given token IDs. input_ids: [B, S] -> [B, S, D]"""
        B, S = input_ids.shape
        D = self.mxfp4.K
        K_half = D // 2
        nblocks = D // 32

        flat_ids = input_ids.reshape(-1)  # [B*S]
        num_tokens = flat_ids.shape[0]

        # Gather packed weights and scales for selected rows
        packed = self.mxfp4.W_packed[flat_ids]  # [num_tokens, K_half]
        scales = self.mxfp4.W_scales[flat_ids]  # [num_tokens, nblocks]

        # Dequant: unpack nibbles, apply FP4 LUT, multiply by scale
        lut = self.FP4_LUT.to(self.device)
        # Unpack low and high nibbles
        lo = (packed & 0xF).to(torch.int64)
        hi = ((packed >> 4) & 0xF).to(torch.int64)
        # Interleave: [num_tokens, K_half] -> [num_tokens, K]
        vals_lo = lut[lo]  # [num_tokens, K_half]
        vals_hi = lut[hi]  # [num_tokens, K_half]
        vals = torch.stack([vals_lo, vals_hi], dim=-1).reshape(num_tokens, D)

        # Apply block scales: each scale covers 32 elements
        scales_f = (2.0 ** (scales.float() - 127.0)).half()  # [num_tokens, nblocks]
        scales_expanded = scales_f.unsqueeze(-1).expand(-1, -1, 32).reshape(num_tokens, D)
        result = vals * scales_expanded

        return result.reshape(B, S, D)


class MXFP4Model:
    def __init__(self, config, blocks, embed_tokens, final_norm, lm_head, rope_cos, rope_sin,
                 embed_mxfp4=None):
        self.config = config
        self.blocks = blocks
        self.final_norm = final_norm
        self.lm_head = lm_head  # Can be MXFP4Linear or tied embedding
        self.rope_cos = rope_cos
        self.rope_sin = rope_sin
        # MXFP4 embedding: replaces FP16 embed_tokens to save ~467 MB
        self.embed_mxfp4 = embed_mxfp4
        if embed_mxfp4 is not None:
            self.embed_tokens = MXFP4Embedding(embed_mxfp4)
            # Free FP16 embedding (the big VRAM savings)
            if embed_tokens is not None:
                del embed_tokens
                torch.cuda.empty_cache()
        else:
            self.embed_tokens = embed_tokens
        # For tied embeddings: MXFP4-quantized version for fast LM head GEMV
        self.lm_head_mxfp4 = None
        if isinstance(lm_head, torch.Tensor) or isinstance(lm_head, torch.nn.Parameter):
            if embed_mxfp4 is not None:
                # Share MXFP4 embedding as LM head (zero extra memory)
                self.lm_head_mxfp4 = embed_mxfp4
            else:
                # Quantize for GEMV (3.3x faster than FP16 at::linear)
                self.lm_head_mxfp4 = MXFP4Linear.from_fp16(lm_head, None, lm_head.device)

    def forward(self, input_ids, position_ids, kv_caches=None):
        """
        input_ids: [B, S] int64
        position_ids: [B, S] int64
        kv_caches: KVCache object, list of (k, v) tuples, or None
        """
        hidden = self.embed_tokens(input_ids).half()
        B, S, D = hidden.shape

        if isinstance(kv_caches, KVCache):
            # Pre-allocated cache: pass same object to each block
            new_kv_caches = kv_caches
            for i, block in enumerate(self.blocks):
                hidden, _ = block.forward(hidden, self.rope_cos, self.rope_sin, position_ids, kv_caches)
            kv_caches.advance(S)
        else:
            # Legacy list-of-tuples
            new_kv_caches = []
            for i, block in enumerate(self.blocks):
                kv = kv_caches[i] if kv_caches is not None else None
                hidden, new_kv = block.forward(hidden, self.rope_cos, self.rope_sin, position_ids, kv)
                new_kv_caches.append(new_kv)

        hidden = self.final_norm.forward(hidden)

        # LM head
        if isinstance(self.lm_head, MXFP4Linear):
            logits = self.lm_head.forward(hidden)
        elif self.lm_head_mxfp4 is not None:
            # Tied embeddings with MXFP4: use MXFP4 LM head
            logits = self.lm_head_mxfp4.forward(hidden)
        else:
            # Tied embeddings: use embed_tokens weight
            logits = F.linear(hidden, self.lm_head)

        return logits, new_kv_caches

    def memory_report(self):
        total_mxfp4 = 0
        total_fp16 = 0
        for block in self.blocks:
            for proj in [block.attn.q_proj, block.attn.k_proj, block.attn.v_proj,
                         block.attn.o_proj, block.mlp.gate_proj, block.mlp.up_proj,
                         block.mlp.down_proj]:
                total_mxfp4 += proj.memory_bytes()
            for norm in [block.input_norm, block.post_norm]:
                total_fp16 += norm.weight.numel() * 2
        if self.embed_mxfp4 is not None:
            total_mxfp4 += self.embed_mxfp4.memory_bytes()
        else:
            total_fp16 += self.embed_tokens.weight.numel() * 2
        total_fp16 += self.final_norm.weight.numel() * 2
        return total_mxfp4, total_fp16


# ============================================================
# Model Loading
# ============================================================

def load_qwen2_mxfp4(model_name, device='cuda', cache_dir=None):
    """Load Qwen2 model with MXFP4 quantized linear layers."""
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"Loading {model_name}...")
    t0 = time.perf_counter()

    # Check for cached MXFP4 weights
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"{model_name.replace('/', '_')}_mxfp4.pt"
        if cache_path.exists():
            print(f"  Loading cached MXFP4 weights from {cache_path}")
            return _load_cached(cache_path, model_name, device)

    # Load FP16 model to CPU
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
    )

    cfg = config.to_dict() if hasattr(config, 'to_dict') else config.__dict__
    # Handle nested text_config for multimodal models
    if 'text_config' in cfg:
        cfg = cfg['text_config']

    hidden_size = cfg['hidden_size']
    intermediate_size = cfg['intermediate_size']
    num_layers = cfg['num_hidden_layers']
    num_heads = cfg['num_attention_heads']
    num_kv_heads = cfg.get('num_key_value_heads', num_heads)
    head_dim = cfg.get('head_dim', hidden_size // num_heads)
    vocab_size = cfg['vocab_size']
    rms_eps = cfg.get('rms_norm_eps', 1e-6)
    rope_theta = cfg.get('rope_theta', 1000000.0)
    tie_embeddings = cfg.get('tie_word_embeddings', False)

    print(f"  Config: {num_layers}L, hidden={hidden_size}, inter={intermediate_size}, "
          f"heads={num_heads}/{num_kv_heads}, head_dim={head_dim}")

    # Precompute RoPE
    rope_dim = head_dim  # Qwen2 uses full head_dim for RoPE
    rope_cos, rope_sin = precompute_rope_freqs(rope_dim, 8192, theta=rope_theta, device=device)

    # Embedding
    embed_tokens = model.model.embed_tokens.to(device)

    # Build blocks
    blocks = []
    for i in range(num_layers):
        layer = model.model.layers[i]
        print(f"  Quantizing layer {i+1}/{num_layers}...", end='\r')

        # Quantize linear layers
        q_proj = MXFP4Linear.from_fp16(layer.self_attn.q_proj.weight, layer.self_attn.q_proj.bias, device)
        k_proj = MXFP4Linear.from_fp16(layer.self_attn.k_proj.weight, layer.self_attn.k_proj.bias, device)
        v_proj = MXFP4Linear.from_fp16(layer.self_attn.v_proj.weight, layer.self_attn.v_proj.bias, device)
        o_proj = MXFP4Linear.from_fp16(layer.self_attn.o_proj.weight, getattr(layer.self_attn.o_proj, 'bias', None), device)

        gate_proj = MXFP4Linear.from_fp16(layer.mlp.gate_proj.weight, None, device)
        up_proj = MXFP4Linear.from_fp16(layer.mlp.up_proj.weight, None, device)
        down_proj = MXFP4Linear.from_fp16(layer.mlp.down_proj.weight, None, device)

        # Norms stay FP16
        input_norm = RMSNorm(layer.input_layernorm.weight.to(device), rms_eps)
        post_norm = RMSNorm(layer.post_attention_layernorm.weight.to(device), rms_eps)

        attn = Qwen2Attention(q_proj, k_proj, v_proj, o_proj, num_heads, num_kv_heads, head_dim)
        attn._layer_idx = i
        mlp = Qwen2MLP(gate_proj, up_proj, down_proj)
        blocks.append(Qwen2Block(attn, mlp, input_norm, post_norm))

        # Free CPU memory
        model.model.layers[i] = None
        torch.cuda.empty_cache()

    print(f"  Quantized {num_layers} layers in {time.perf_counter()-t0:.1f}s")

    # Final norm
    final_norm = RMSNorm(model.model.norm.weight.to(device), rms_eps)

    # LM head
    if tie_embeddings:
        lm_head = embed_tokens.weight  # tied — just use embedding weight
    else:
        lm_head = MXFP4Linear.from_fp16(model.lm_head.weight, model.lm_head.bias, device)

    # Quantize embedding to MXFP4 (saves ~467 MB for 1.5B model)
    embed_mxfp4 = MXFP4Linear.from_fp16(embed_tokens.weight, None, device)
    print(f"  MXFP4 embedding: {embed_mxfp4.W_packed.numel()/1e6:.0f} MB packed")

    # Free FP16 embedding and tied LM head (biggest VRAM saving)
    if tie_embeddings:
        lm_head = torch.empty(0, device=device)  # placeholder, will use embed_mxfp4
    del embed_tokens
    del model
    torch.cuda.empty_cache()

    mxfp4_model = MXFP4Model(cfg, blocks, None, final_norm, lm_head, rope_cos, rope_sin,
                              embed_mxfp4=embed_mxfp4)

    # Memory report
    mxfp4_bytes, fp16_bytes = mxfp4_model.memory_report()
    total = mxfp4_bytes + fp16_bytes
    print(f"  MXFP4 weights: {mxfp4_bytes/1e6:.0f} MB, FP16 other: {fp16_bytes/1e6:.0f} MB, Total: {total/1e6:.0f} MB")

    # Save cache
    if cache_path:
        print(f"  Saving MXFP4 cache to {cache_path}...")
        _save_cached(mxfp4_model, cache_path)

    return mxfp4_model


def _save_cached(model, path):
    """Save MXFP4 model weights for fast loading."""
    state = {
        'config': model.config,
        'final_norm': model.final_norm.weight.cpu(),
        'rope_cos': model.rope_cos.cpu(),
        'rope_sin': model.rope_sin.cpu(),
    }
    # Save embedding: MXFP4 if available, else FP16
    if model.embed_mxfp4 is not None:
        state['embed_mxfp4_packed'] = model.embed_mxfp4.W_packed.cpu()
        state['embed_mxfp4_scales'] = model.embed_mxfp4.W_scales.cpu()
        state['embed_mxfp4_N'] = model.embed_mxfp4.N
        state['embed_mxfp4_K'] = model.embed_mxfp4.K
        state['embed_is_mxfp4'] = True
    else:
        state['embed_tokens'] = model.embed_tokens.weight.data.cpu()
        state['embed_is_mxfp4'] = False

    if isinstance(model.lm_head, torch.Tensor):
        state['lm_head_tied'] = True
    else:
        state['lm_head_tied'] = False
        state['lm_head_packed'] = model.lm_head.W_packed.cpu()
        state['lm_head_scales'] = model.lm_head.W_scales.cpu()
        state['lm_head_N'] = model.lm_head.N
        state['lm_head_K'] = model.lm_head.K

    for i, block in enumerate(model.blocks):
        prefix = f'block_{i}'
        for name, proj in [('q', block.attn.q_proj), ('k', block.attn.k_proj),
                           ('v', block.attn.v_proj), ('o', block.attn.o_proj),
                           ('gate', block.mlp.gate_proj), ('up', block.mlp.up_proj),
                           ('down', block.mlp.down_proj)]:
            state[f'{prefix}_{name}_packed'] = proj.W_packed.cpu()
            state[f'{prefix}_{name}_scales'] = proj.W_scales.cpu()
            state[f'{prefix}_{name}_N'] = proj.N
            state[f'{prefix}_{name}_K'] = proj.K
            if proj.bias is not None:
                state[f'{prefix}_{name}_bias'] = proj.bias.cpu()

        state[f'{prefix}_input_norm'] = block.input_norm.weight.cpu()
        state[f'{prefix}_post_norm'] = block.post_norm.weight.cpu()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _load_cached(path, model_name, device):
    """Load cached MXFP4 weights."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    cfg = state['config']

    hidden_size = cfg['hidden_size']
    num_layers = cfg['num_hidden_layers']
    num_heads = cfg['num_attention_heads']
    num_kv_heads = cfg.get('num_key_value_heads', num_heads)
    head_dim = cfg.get('head_dim', hidden_size // num_heads)
    rms_eps = cfg.get('rms_norm_eps', 1e-6)

    rope_cos = state['rope_cos'].to(device)
    rope_sin = state['rope_sin'].to(device)

    # Load embedding
    embed_mxfp4 = None
    embed = None
    if state.get('embed_is_mxfp4', False):
        embed_mxfp4 = MXFP4Linear(
            state['embed_mxfp4_packed'].to(device),
            state['embed_mxfp4_scales'].to(device),
            state['embed_mxfp4_N'], state['embed_mxfp4_K']
        )
    else:
        embed = torch.nn.Embedding(cfg['vocab_size'], hidden_size)
        embed.weight.data = state['embed_tokens'].half()
        embed = embed.to(device)
        # Also create MXFP4 version if not already saved
        if 'embed_mxfp4_packed' in state:
            embed_mxfp4 = MXFP4Linear(
                state['embed_mxfp4_packed'].to(device),
                state['embed_mxfp4_scales'].to(device),
                state['embed_mxfp4_N'], state['embed_mxfp4_K']
            )

    final_norm = RMSNorm(state['final_norm'].to(device), rms_eps)

    if state['lm_head_tied']:
        if embed is not None:
            lm_head = embed.weight
        else:
            # Tied but MXFP4-only embedding — lm_head will use MXFP4 path
            lm_head = torch.empty(0, device=device)  # placeholder
    else:
        lm_head = MXFP4Linear(
            state['lm_head_packed'].to(device),
            state['lm_head_scales'].to(device),
            state['lm_head_N'], state['lm_head_K']
        )

    blocks = []
    for i in range(num_layers):
        prefix = f'block_{i}'
        projs = {}
        for name in ['q', 'k', 'v', 'o', 'gate', 'up', 'down']:
            bias = state.get(f'{prefix}_{name}_bias')
            if bias is not None:
                bias = bias.half().to(device)
            projs[name] = MXFP4Linear(
                state[f'{prefix}_{name}_packed'].to(device),
                state[f'{prefix}_{name}_scales'].to(device),
                state[f'{prefix}_{name}_N'],
                state[f'{prefix}_{name}_K'],
                bias
            )

        attn = Qwen2Attention(projs['q'], projs['k'], projs['v'], projs['o'],
                              num_heads, num_kv_heads, head_dim)
        attn._layer_idx = i
        mlp = Qwen2MLP(projs['gate'], projs['up'], projs['down'])
        input_norm = RMSNorm(state[f'{prefix}_input_norm'].to(device), rms_eps)
        post_norm = RMSNorm(state[f'{prefix}_post_norm'].to(device), rms_eps)
        blocks.append(Qwen2Block(attn, mlp, input_norm, post_norm))

    return MXFP4Model(cfg, blocks, embed, final_norm, lm_head, rope_cos, rope_sin,
                      embed_mxfp4=embed_mxfp4)


# ============================================================
# Qwen3 MoE Loading
# ============================================================

def load_qwen3_moe_mxfp4(model_name, device='cuda', cache_dir=None):
    """Load Qwen3 MoE model with MXFP4 quantized weights."""
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"Loading {model_name}...")
    t0 = time.perf_counter()

    # Check cache
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"{model_name.replace('/', '_')}_mxfp4_moe.pt"
        if cache_path.exists():
            print(f"  Loading cached MoE MXFP4 weights from {cache_path}")
            return _load_cached_moe(cache_path, model_name, device)

    config = AutoConfig.from_pretrained(model_name)
    cfg = config.to_dict()

    hidden_size = cfg['hidden_size']
    num_layers = cfg['num_hidden_layers']
    num_heads = cfg['num_attention_heads']
    num_kv_heads = cfg.get('num_key_value_heads', num_heads)
    head_dim = cfg.get('head_dim', hidden_size // num_heads)
    rms_eps = cfg.get('rms_norm_eps', 1e-6)
    rope_theta = cfg.get('rope_theta', 1000000.0)
    num_experts = cfg.get('num_experts', cfg.get('num_local_experts', 128))
    num_experts_per_tok = cfg['num_experts_per_tok']
    moe_intermediate_size = cfg['moe_intermediate_size']
    norm_topk_prob = cfg.get('norm_topk_prob', True)
    tie_embeddings = cfg.get('tie_word_embeddings', False)

    print(f"  Config: {num_layers}L, hidden={hidden_size}, experts={num_experts}×{moe_intermediate_size}, "
          f"top-{num_experts_per_tok}, heads={num_heads}/{num_kv_heads}")

    # RoPE
    rope_cos, rope_sin = precompute_rope_freqs(head_dim, 8192, theta=rope_theta, device=device)

    # Load model layer by layer (low memory)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
    )

    embed_tokens = model.model.embed_tokens.to(device)

    blocks = []
    for i in range(num_layers):
        layer = model.model.layers[i]
        print(f"  Quantizing layer {i+1}/{num_layers} ({num_experts} experts)...", end='\r')

        # Attention
        q_proj = MXFP4Linear.from_fp16(layer.self_attn.q_proj.weight, None, device)
        k_proj = MXFP4Linear.from_fp16(layer.self_attn.k_proj.weight, None, device)
        v_proj = MXFP4Linear.from_fp16(layer.self_attn.v_proj.weight, None, device)
        o_proj = MXFP4Linear.from_fp16(layer.self_attn.o_proj.weight, None, device)

        # QK norms
        q_norm_w = layer.self_attn.q_norm.weight.to(device)
        k_norm_w = layer.self_attn.k_norm.weight.to(device)

        attn = Qwen3Attention(q_proj, k_proj, v_proj, o_proj,
                              num_heads, num_kv_heads, head_dim,
                              q_norm_w, k_norm_w, rms_eps)
        attn._layer_idx = i

        # MoE experts
        experts = []
        for e in range(num_experts):
            expert = layer.mlp.experts[e]
            g = MXFP4Linear.from_fp16(expert.gate_proj.weight, None, device)
            u = MXFP4Linear.from_fp16(expert.up_proj.weight, None, device)
            d = MXFP4Linear.from_fp16(expert.down_proj.weight, None, device)
            experts.append((g, u, d))

        # Router stays FP16
        router_weight = layer.mlp.gate.weight.half().to(device)

        moe_mlp = Qwen3MoEMLP(router_weight, experts, num_experts_per_tok, norm_topk_prob)

        input_norm = RMSNorm(layer.input_layernorm.weight.to(device), rms_eps)
        post_norm = RMSNorm(layer.post_attention_layernorm.weight.to(device), rms_eps)

        blocks.append(Qwen3MoEBlock(attn, moe_mlp, input_norm, post_norm))

        # Free CPU memory
        model.model.layers[i] = None
        torch.cuda.empty_cache()

    print(f"  Quantized {num_layers} MoE layers in {time.perf_counter()-t0:.1f}s")

    final_norm = RMSNorm(model.model.norm.weight.to(device), rms_eps)

    # LM head (not tied for Qwen3 MoE)
    if tie_embeddings:
        lm_head = embed_tokens.weight
    else:
        lm_head = MXFP4Linear.from_fp16(model.lm_head.weight, None, device)

    # MXFP4 embedding
    embed_mxfp4 = MXFP4Linear.from_fp16(embed_tokens.weight, None, device)

    if tie_embeddings:
        lm_head = torch.empty(0, device=device)
    del embed_tokens, model
    torch.cuda.empty_cache()

    mxfp4_model = MXFP4Model(cfg, blocks, None, final_norm, lm_head, rope_cos, rope_sin,
                              embed_mxfp4=embed_mxfp4)

    vram = torch.cuda.memory_allocated() / 1e6
    print(f"  VRAM: {vram:.0f} MB")

    if cache_path:
        print(f"  Saving MoE cache to {cache_path}...")
        _save_cached_moe(mxfp4_model, cache_path)

    return mxfp4_model


def _save_cached_moe(model, path):
    """Save MoE MXFP4 model weights."""
    state = {
        'config': model.config,
        'model_type': 'qwen3_moe',
        'final_norm': model.final_norm.weight.cpu(),
        'rope_cos': model.rope_cos.cpu(),
        'rope_sin': model.rope_sin.cpu(),
    }

    # Embedding
    if model.embed_mxfp4 is not None:
        state['embed_mxfp4_packed'] = model.embed_mxfp4.W_packed.cpu()
        state['embed_mxfp4_scales'] = model.embed_mxfp4.W_scales.cpu()
        state['embed_mxfp4_N'] = model.embed_mxfp4.N
        state['embed_mxfp4_K'] = model.embed_mxfp4.K
        state['embed_is_mxfp4'] = True

    # LM head
    if isinstance(model.lm_head, MXFP4Linear):
        state['lm_head_tied'] = False
        state['lm_head_packed'] = model.lm_head.W_packed.cpu()
        state['lm_head_scales'] = model.lm_head.W_scales.cpu()
        state['lm_head_N'] = model.lm_head.N
        state['lm_head_K'] = model.lm_head.K
    else:
        state['lm_head_tied'] = True

    for i, block in enumerate(model.blocks):
        prefix = f'block_{i}'
        # Attention
        for name, proj in [('q', block.attn.q_proj), ('k', block.attn.k_proj),
                           ('v', block.attn.v_proj), ('o', block.attn.o_proj)]:
            state[f'{prefix}_{name}_packed'] = proj.W_packed.cpu()
            state[f'{prefix}_{name}_scales'] = proj.W_scales.cpu()
            state[f'{prefix}_{name}_N'] = proj.N
            state[f'{prefix}_{name}_K'] = proj.K

        # QK norms
        state[f'{prefix}_q_norm'] = block.attn.q_norm.weight.cpu()
        state[f'{prefix}_k_norm'] = block.attn.k_norm.weight.cpu()

        # Router
        state[f'{prefix}_router'] = block.mlp.router_weight.cpu()

        # Experts
        for e_idx, (g, u, d) in enumerate(block.mlp.experts):
            ep = f'{prefix}_e{e_idx}'
            state[f'{ep}_g_packed'] = g.W_packed.cpu()
            state[f'{ep}_g_scales'] = g.W_scales.cpu()
            state[f'{ep}_g_N'] = g.N
            state[f'{ep}_g_K'] = g.K
            state[f'{ep}_u_packed'] = u.W_packed.cpu()
            state[f'{ep}_u_scales'] = u.W_scales.cpu()
            state[f'{ep}_u_N'] = u.N
            state[f'{ep}_u_K'] = u.K
            state[f'{ep}_d_packed'] = d.W_packed.cpu()
            state[f'{ep}_d_scales'] = d.W_scales.cpu()
            state[f'{ep}_d_N'] = d.N
            state[f'{ep}_d_K'] = d.K

        state[f'{prefix}_input_norm'] = block.input_norm.weight.cpu()
        state[f'{prefix}_post_norm'] = block.post_norm.weight.cpu()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _load_cached_moe(path, model_name, device):
    """Load cached MoE MXFP4 weights."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    cfg = state['config']

    hidden_size = cfg['hidden_size']
    num_layers = cfg['num_hidden_layers']
    num_heads = cfg['num_attention_heads']
    num_kv_heads = cfg.get('num_key_value_heads', num_heads)
    head_dim = cfg.get('head_dim', hidden_size // num_heads)
    rms_eps = cfg.get('rms_norm_eps', 1e-6)
    num_experts = cfg.get('num_experts', cfg.get('num_local_experts', 128))
    num_experts_per_tok = cfg['num_experts_per_tok']
    norm_topk_prob = cfg.get('norm_topk_prob', True)

    rope_cos = state['rope_cos'].to(device)
    rope_sin = state['rope_sin'].to(device)

    # Embedding
    embed_mxfp4 = None
    if state.get('embed_is_mxfp4', False):
        embed_mxfp4 = MXFP4Linear(
            state['embed_mxfp4_packed'].to(device),
            state['embed_mxfp4_scales'].to(device),
            state['embed_mxfp4_N'], state['embed_mxfp4_K']
        )

    final_norm = RMSNorm(state['final_norm'].to(device), rms_eps)

    # LM head
    if state.get('lm_head_tied', False):
        lm_head = torch.empty(0, device=device)
    else:
        lm_head = MXFP4Linear(
            state['lm_head_packed'].to(device),
            state['lm_head_scales'].to(device),
            state['lm_head_N'], state['lm_head_K']
        )

    blocks = []
    for i in range(num_layers):
        prefix = f'block_{i}'
        print(f"  Loading layer {i+1}/{num_layers}...", end='\r')

        # Attention
        projs = {}
        for name in ['q', 'k', 'v', 'o']:
            projs[name] = MXFP4Linear(
                state[f'{prefix}_{name}_packed'].to(device),
                state[f'{prefix}_{name}_scales'].to(device),
                state[f'{prefix}_{name}_N'], state[f'{prefix}_{name}_K']
            )

        q_norm_w = state[f'{prefix}_q_norm'].to(device)
        k_norm_w = state[f'{prefix}_k_norm'].to(device)

        attn = Qwen3Attention(projs['q'], projs['k'], projs['v'], projs['o'],
                              num_heads, num_kv_heads, head_dim,
                              q_norm_w, k_norm_w, rms_eps)
        attn._layer_idx = i

        # Router
        router_weight = state[f'{prefix}_router'].half().to(device)

        # Experts
        experts = []
        for e_idx in range(num_experts):
            ep = f'{prefix}_e{e_idx}'
            g = MXFP4Linear(state[f'{ep}_g_packed'].to(device), state[f'{ep}_g_scales'].to(device),
                            state[f'{ep}_g_N'], state[f'{ep}_g_K'])
            u = MXFP4Linear(state[f'{ep}_u_packed'].to(device), state[f'{ep}_u_scales'].to(device),
                            state[f'{ep}_u_N'], state[f'{ep}_u_K'])
            d = MXFP4Linear(state[f'{ep}_d_packed'].to(device), state[f'{ep}_d_scales'].to(device),
                            state[f'{ep}_d_N'], state[f'{ep}_d_K'])
            experts.append((g, u, d))

        moe_mlp = Qwen3MoEMLP(router_weight, experts, num_experts_per_tok, norm_topk_prob)

        input_norm = RMSNorm(state[f'{prefix}_input_norm'].to(device), rms_eps)
        post_norm = RMSNorm(state[f'{prefix}_post_norm'].to(device), rms_eps)
        blocks.append(Qwen3MoEBlock(attn, moe_mlp, input_norm, post_norm))

    print(f"  Loaded {num_layers} MoE layers")

    return MXFP4Model(cfg, blocks, None, final_norm, lm_head, rope_cos, rope_sin,
                      embed_mxfp4=embed_mxfp4)


# ============================================================
# Text Generation
# ============================================================

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """Generate text using MXFP4 model with pre-allocated KV cache."""
    device = model.embed_tokens.weight.device
    cfg = model.config
    num_layers = cfg['num_hidden_layers']
    num_kv_heads = cfg.get('num_key_value_heads', cfg['num_attention_heads'])
    head_dim = cfg.get('head_dim', cfg['hidden_size'] // cfg['num_attention_heads'])

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    B, S = input_ids.shape
    position_ids = torch.arange(S, device=device).unsqueeze(0)

    # Pre-allocate KV cache
    kv_cache = KVCache(num_layers, B, S + max_new_tokens + 16, num_kv_heads, head_dim, device)

    # Prefill
    t_start = time.perf_counter()
    logits, _ = model.forward(input_ids, position_ids, kv_cache)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t_start

    # Get next token
    next_logits = logits[:, -1, :]

    # Pre-allocate decode buffers — all on GPU, no CPU syncs in loop
    pos_buf = torch.zeros((1, 1), dtype=torch.long, device=device)
    token_buf = torch.empty((max_new_tokens,), dtype=torch.long, device=device)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
    eos_tensor = torch.tensor([eos_id], dtype=torch.long, device=device)

    # Decode loop — zero CPU syncs
    t_decode_start = time.perf_counter()
    cur_pos = S
    n_generated = 0

    for step in range(max_new_tokens):
        # Sample (all on GPU)
        if temperature > 0:
            probs = F.softmax(next_logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            else:
                next_token = torch.multinomial(probs, 1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        # Store token on GPU
        token_buf[step] = next_token.view(-1)[0]
        n_generated += 1

        # EOS check on GPU (no .item() sync)
        if (next_token.view(-1) == eos_tensor).any():
            break

        # Next position (all GPU tensor ops, no .item())
        pos_buf[0, 0] = cur_pos
        next_logits, _ = model.forward(next_token, pos_buf, kv_cache)
        next_logits = next_logits[:, -1, :]
        cur_pos += 1

    torch.cuda.synchronize()
    t_decode = time.perf_counter() - t_decode_start

    # Single CPU sync: transfer all generated tokens at once
    generated_ids = token_buf[:n_generated].cpu().tolist()
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n_tokens = len(generated_ids)

    stats = {
        'prefill_ms': t_prefill * 1000,
        'prefill_tokens': S,
        'prefill_tps': S / t_prefill,
        'decode_tokens': n_tokens,
        'decode_ms': t_decode * 1000,
        'decode_tps': n_tokens / t_decode if t_decode > 0 else 0,
    }

    return output_text, stats


# ============================================================
# C++ Decode Step Generation (zero Python dispatch)
# ============================================================

def generate_cpp(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """Generate text using C++ decode step — zero Python in hot loop."""
    try:
        import mxfp4_decode_step
    except ImportError:
        print("WARNING: mxfp4_decode_step not built, falling back to Python")
        return generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p)

    device = model.embed_tokens.weight.device
    cfg = model.config
    num_layers = cfg['num_hidden_layers']
    num_heads = cfg['num_attention_heads']
    num_kv_heads = cfg.get('num_key_value_heads', num_heads)
    head_dim = cfg.get('head_dim', cfg['hidden_size'] // num_heads)

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    B, S = input_ids.shape
    position_ids = torch.arange(S, device=device).unsqueeze(0)

    # Pre-allocate KV cache
    max_seq = S + max_new_tokens + 16
    kv_cache = KVCache(num_layers, B, max_seq, num_kv_heads, head_dim, device)

    # Prefill with Python path (handles variable S)
    t_start = time.perf_counter()
    logits, _ = model.forward(input_ids, position_ids, kv_cache)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t_start

    # Prepare weight lists for C++ decode step
    q_packed, q_scales = [], []
    k_packed, k_scales = [], []
    v_packed, v_scales = [], []
    o_packed, o_scales = [], []
    gate_packed, gate_scales = [], []
    up_packed, up_scales = [], []
    down_packed, down_scales = [], []
    input_norms, post_norms = [], []
    q_biases, k_biases, v_biases = [], [], []

    for block in model.blocks:
        a = block.attn
        q_packed.append(a.q_proj.W_packed); q_scales.append(a.q_proj.W_scales)
        k_packed.append(a.k_proj.W_packed); k_scales.append(a.k_proj.W_scales)
        v_packed.append(a.v_proj.W_packed); v_scales.append(a.v_proj.W_scales)
        o_packed.append(a.o_proj.W_packed); o_scales.append(a.o_proj.W_scales)
        m = block.mlp
        gate_packed.append(m.gate_proj.W_packed); gate_scales.append(m.gate_proj.W_scales)
        up_packed.append(m.up_proj.W_packed); up_scales.append(m.up_proj.W_scales)
        down_packed.append(m.down_proj.W_packed); down_scales.append(m.down_proj.W_scales)
        input_norms.append(block.input_norm.weight)
        post_norms.append(block.post_norm.weight)
        q_biases.append(a.q_proj.bias if a.q_proj.bias is not None else torch.empty(0, device=device))
        k_biases.append(a.k_proj.bias if a.k_proj.bias is not None else torch.empty(0, device=device))
        v_biases.append(a.v_proj.bias if a.v_proj.bias is not None else torch.empty(0, device=device))

    # Dimension arrays (same for all layers)
    b0 = model.blocks[0]
    N_dims = [b0.attn.q_proj.N, b0.attn.k_proj.N, b0.attn.v_proj.N, b0.attn.o_proj.N,
              b0.mlp.gate_proj.N, b0.mlp.up_proj.N, b0.mlp.down_proj.N]
    K_dims = [b0.attn.q_proj.K, b0.attn.k_proj.K, b0.attn.v_proj.K, b0.attn.o_proj.K,
              b0.mlp.gate_proj.K, b0.mlp.up_proj.K, b0.mlp.down_proj.K]

    # LM head
    lm_head_tied = not isinstance(model.lm_head, MXFP4Linear)
    if lm_head_tied and model.lm_head_mxfp4 is not None:
        # Use MXFP4-quantized LM head (3.3x faster than FP16 at::linear)
        lm_packed = model.lm_head_mxfp4.W_packed
        lm_scales = model.lm_head_mxfp4.W_scales
        lm_N, lm_K = model.lm_head_mxfp4.N, model.lm_head_mxfp4.K
        lm_head_tied = False  # treat as non-tied since we have MXFP4 weights
    elif lm_head_tied:
        lm_packed = torch.empty(0, device=device)
        lm_scales = torch.empty(0, device=device)
        lm_N, lm_K = 0, 0
    else:
        lm_packed = model.lm_head.W_packed
        lm_scales = model.lm_head.W_scales
        lm_N, lm_K = model.lm_head.N, model.lm_head.K

    rms_eps = model.blocks[0].input_norm.eps
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1

    # Decode loop
    next_logits = logits[:, -1, :]
    token_buf = torch.empty((max_new_tokens,), dtype=torch.long, device=device)
    cur_pos = S
    n_generated = 0

    t_decode_start = time.perf_counter()

    for step in range(max_new_tokens):
        # Sample (on GPU)
        if temperature > 0:
            probs = F.softmax(next_logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            else:
                next_token = torch.multinomial(probs, 1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        token_buf[step] = next_token.view(-1)[0]
        n_generated += 1

        # Defer EOS check — only sync every 16 tokens to check
        if step % 16 == 15:
            # Batch EOS check: look at last 16 tokens
            chunk = token_buf[max(0, step-15):step+1]
            if (chunk == eos_id).any().item():
                # Find actual EOS position
                for i in range(max(0, step-15), step+1):
                    if token_buf[i].item() == eos_id:
                        n_generated = i + 1
                        break
                break

        # C++ decode step — entire forward pass, zero Python dispatch
        next_logits = mxfp4_decode_step.decode_step(
            next_token, torch.tensor(cur_pos),
            model.embed_tokens.weight,
            q_packed, q_scales, k_packed, k_scales,
            v_packed, v_scales, o_packed, o_scales,
            gate_packed, gate_scales, up_packed, up_scales,
            down_packed, down_scales,
            input_norms, post_norms,
            q_biases, k_biases, v_biases,
            N_dims, K_dims,
            num_heads, num_kv_heads, head_dim, rms_eps,
            model.rope_cos, model.rope_sin,
            kv_cache.k_caches, kv_cache.v_caches,
            cur_pos,
            model.final_norm.weight,
            lm_packed, lm_scales, lm_N, lm_K, lm_head_tied,
        )
        next_logits = next_logits[:, -1, :]
        cur_pos += 1

    torch.cuda.synchronize()
    t_decode = time.perf_counter() - t_decode_start

    generated_ids = token_buf[:n_generated].cpu().tolist()
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    stats = {
        'prefill_ms': t_prefill * 1000,
        'prefill_tokens': S,
        'prefill_tps': S / t_prefill,
        'decode_tokens': n_generated,
        'decode_ms': t_decode * 1000,
        'decode_tps': n_generated / t_decode if t_decode > 0 else 0,
    }
    return output_text, stats


# ============================================================
# C++ Decode Loop V2 (entire loop in C++, zero Python)
# ============================================================

def generate_cpp_v2(model, tokenizer, prompt, max_new_tokens=100, temperature=0.0, top_p=0.9):
    """Generate text using V2 C++ decode loop — entire decode in C++."""
    try:
        import mxfp4_decode_v2
    except ImportError:
        print("WARNING: mxfp4_decode_v2 not built, falling back to generate_cpp")
        return generate_cpp(model, tokenizer, prompt, max_new_tokens, temperature, top_p)

    device = model.embed_tokens.weight.device
    cfg = model.config
    num_layers = cfg['num_hidden_layers']
    num_heads = cfg['num_attention_heads']
    num_kv_heads = cfg.get('num_key_value_heads', num_heads)
    head_dim = cfg.get('head_dim', cfg['hidden_size'] // num_heads)

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    B, S = input_ids.shape
    position_ids = torch.arange(S, device=device).unsqueeze(0)

    # Pre-allocate KV cache
    max_seq = S + max_new_tokens + 16
    kv_cache = KVCache(num_layers, B, max_seq, num_kv_heads, head_dim, device)

    # Prefill with Python path (handles variable S, uses GEMM)
    t_start = time.perf_counter()
    logits, _ = model.forward(input_ids, position_ids, kv_cache)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t_start

    # Get first token (greedy for V2 — temperature=0 only for now)
    first_token = logits[:, -1, :].argmax(dim=-1).view(1)

    # Prepare weight lists
    q_packed, q_scales = [], []
    k_packed, k_scales = [], []
    v_packed, v_scales = [], []
    o_packed, o_scales = [], []
    gate_packed, gate_scales = [], []
    up_packed, up_scales = [], []
    down_packed, down_scales = [], []
    input_norms, post_norms = [], []
    q_biases, k_biases, v_biases = [], [], []

    for block in model.blocks:
        a = block.attn
        q_packed.append(a.q_proj.W_packed); q_scales.append(a.q_proj.W_scales)
        k_packed.append(a.k_proj.W_packed); k_scales.append(a.k_proj.W_scales)
        v_packed.append(a.v_proj.W_packed); v_scales.append(a.v_proj.W_scales)
        o_packed.append(a.o_proj.W_packed); o_scales.append(a.o_proj.W_scales)
        m = block.mlp
        gate_packed.append(m.gate_proj.W_packed); gate_scales.append(m.gate_proj.W_scales)
        up_packed.append(m.up_proj.W_packed); up_scales.append(m.up_proj.W_scales)
        down_packed.append(m.down_proj.W_packed); down_scales.append(m.down_proj.W_scales)
        input_norms.append(block.input_norm.weight)
        post_norms.append(block.post_norm.weight)
        q_biases.append(a.q_proj.bias if a.q_proj.bias is not None else torch.empty(0, device=device))
        k_biases.append(a.k_proj.bias if a.k_proj.bias is not None else torch.empty(0, device=device))
        v_biases.append(a.v_proj.bias if a.v_proj.bias is not None else torch.empty(0, device=device))

    b0 = model.blocks[0]
    N_dims = [b0.attn.q_proj.N, b0.attn.k_proj.N, b0.attn.v_proj.N, b0.attn.o_proj.N,
              b0.mlp.gate_proj.N, b0.mlp.up_proj.N, b0.mlp.down_proj.N]
    K_dims = [b0.attn.q_proj.K, b0.attn.k_proj.K, b0.attn.v_proj.K, b0.attn.o_proj.K,
              b0.mlp.gate_proj.K, b0.mlp.up_proj.K, b0.mlp.down_proj.K]

    lm_head_tied = not isinstance(model.lm_head, MXFP4Linear)
    if lm_head_tied and model.lm_head_mxfp4 is not None:
        # Use MXFP4-quantized LM head (3.3x faster than FP16 at::linear)
        lm_packed = model.lm_head_mxfp4.W_packed
        lm_scales = model.lm_head_mxfp4.W_scales
        lm_N, lm_K = model.lm_head_mxfp4.N, model.lm_head_mxfp4.K
        lm_head_tied = False  # treat as non-tied since we have MXFP4 weights
    elif lm_head_tied:
        lm_packed = torch.empty(0, device=device)
        lm_scales = torch.empty(0, device=device)
        lm_N, lm_K = 0, 0
    else:
        lm_packed = model.lm_head.W_packed
        lm_scales = model.lm_head.W_scales
        lm_N, lm_K = model.lm_head.N, model.lm_head.K

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1

    # Run entire decode loop in C++
    t_decode_start = time.perf_counter()

    output_tokens = mxfp4_decode_v2.decode_loop(
        first_token, S, max_new_tokens - 1,  # -1 because first_token is already sampled
        model.embed_tokens.weight,
        q_packed, q_scales, k_packed, k_scales,
        v_packed, v_scales, o_packed, o_scales,
        gate_packed, gate_scales, up_packed, up_scales,
        down_packed, down_scales,
        input_norms, post_norms,
        q_biases, k_biases, v_biases,
        N_dims, K_dims,
        num_heads, num_kv_heads, head_dim, model.blocks[0].input_norm.eps,
        model.rope_cos, model.rope_sin,
        kv_cache.k_caches, kv_cache.v_caches,
        model.final_norm.weight,
        lm_packed, lm_scales, lm_N, lm_K, lm_head_tied,
        eos_id,
    )

    torch.cuda.synchronize()
    t_decode = time.perf_counter() - t_decode_start

    # Prepend first token and decode
    all_tokens = torch.cat([first_token.view(-1), output_tokens]).cpu().tolist()
    output_text = tokenizer.decode(all_tokens, skip_special_tokens=True)
    n_tokens = len(all_tokens)

    stats = {
        'prefill_ms': t_prefill * 1000,
        'prefill_tokens': S,
        'prefill_tps': S / t_prefill,
        'decode_tokens': n_tokens,
        'decode_ms': t_decode * 1000,
        'decode_tps': n_tokens / t_decode if t_decode > 0 else 0,
    }
    return output_text, stats


# ============================================================
# Main
# ============================================================

def main():
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    cache_dir = Path(__file__).parent / "cache"
    device = "cuda"

    print("=" * 60)
    print("MXFP4 Full Inference Engine — AMD RDNA4")
    print("=" * 60)
    print()

    # Load model
    model = load_qwen2_mxfp4(model_name, device, cache_dir)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    cfg = model.config
    num_layers = cfg['num_hidden_layers']
    num_kv_heads = cfg.get('num_key_value_heads', cfg['num_attention_heads'])
    head_dim = cfg.get('head_dim', cfg['hidden_size'] // cfg['num_attention_heads'])

    # Warmup: run one forward pass to trigger Triton JIT compilation
    print("Warming up Triton kernels...")
    t_warmup = time.perf_counter()
    warmup_kv = KVCache(num_layers, 1, 32, num_kv_heads, head_dim, device)
    warmup_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    warmup_pos = torch.arange(4, device=device).unsqueeze(0)
    model.forward(warmup_ids, warmup_pos, warmup_kv)
    # Also warmup decode path (S=1)
    model.forward(torch.tensor([[5]], device=device),
                  torch.tensor([[4]], device=device), warmup_kv)
    torch.cuda.synchronize()
    print(f"  Warmup done in {time.perf_counter()-t_warmup:.1f}s")
    del warmup_kv

    # Test prompts
    prompts = [
        "Hello! What is the capital of Poland?",
        "Write a short Python function to compute fibonacci numbers:",
        "Explain quantum computing in simple terms:",
    ]

    print("\n" + "=" * 60)
    print("Generating text...")
    print("=" * 60)

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)

        output, stats = generate(model, tokenizer, prompt,
                                max_new_tokens=100, temperature=0.7)

        print(f"Output: {output}")
        print(f"\nStats: prefill={stats['prefill_ms']:.0f}ms ({stats['prefill_tps']:.0f} t/s), "
              f"decode={stats['decode_tokens']} tokens in {stats['decode_ms']:.0f}ms "
              f"({stats['decode_tps']:.1f} t/s)")

    # Benchmark
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    # Prefill benchmark (various lengths)
    print("\nPrefill (prompt processing):")
    for pp_len in [16, 64, 128, 512]:
        input_ids = torch.randint(100, 10000, (1, pp_len), device=device)
        position_ids = torch.arange(pp_len, device=device).unsqueeze(0)

        # Warmup
        model.forward(input_ids, position_ids)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N_RUNS = 3
        for _ in range(N_RUNS):
            model.forward(input_ids, position_ids)
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / N_RUNS
        tps = pp_len / t
        print(f"  pp{pp_len:>4}: {t*1000:.0f}ms = {tps:.0f} tokens/s")

    # Decode benchmark
    print("\nDecode (token generation):")
    prompt = "The meaning of life is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    B, S = input_ids.shape
    N_TOKENS = 128

    kv = KVCache(num_layers, B, S + N_TOKENS + 16, num_kv_heads, head_dim, device)
    position_ids = torch.arange(S, device=device).unsqueeze(0)
    logits, _ = model.forward(input_ids, position_ids, kv)
    torch.cuda.synchronize()

    next_logits = logits[:, -1, :]
    cur_pos = S
    pos_buf = torch.zeros((1, 1), dtype=torch.long, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_TOKENS):
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        pos_buf[0, 0] = cur_pos
        next_logits, _ = model.forward(next_token, pos_buf, kv)
        next_logits = next_logits[:, -1, :]
        cur_pos += 1
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t0

    tps = N_TOKENS / t_total
    ms_per_token = t_total / N_TOKENS * 1000
    print(f"  tg{N_TOKENS}: {t_total*1000:.0f}ms = {tps:.1f} tokens/s ({ms_per_token:.1f} ms/token)")

    # Memory usage
    mxfp4_bytes, fp16_bytes = model.memory_report()
    gpu_mem = torch.cuda.memory_allocated() / 1e6
    gpu_reserved = torch.cuda.memory_reserved() / 1e6
    print(f"\nMemory:")
    print(f"  Model weights: MXFP4={mxfp4_bytes/1e6:.0f}MB + FP16={fp16_bytes/1e6:.0f}MB = {(mxfp4_bytes+fp16_bytes)/1e6:.0f}MB")
    print(f"  GPU allocated: {gpu_mem:.0f} MB, reserved: {gpu_reserved:.0f} MB")


if __name__ == "__main__":
    main()
