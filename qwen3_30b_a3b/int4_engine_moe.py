"""
INT4 MoE Engine — Qwen3-30B-A3B inference with INT4 asymmetric + Hadamard + GPTQ.

Architecture: 48-layer MoE, hidden=2048, 128 experts/layer, top-8 active.

Decode strategy (batch=1):
  - Attention: existing HIP C++ kernels (gemv_warp_norm, flash_decode)
  - MoE routing: Python (2 ops: norm + topk-softmax)
  - Expert dispatch: int4_hip.moe_gemv_top_k (C++ loop over 8 experts)

Prefill strategy (batch=M):
  - Attention: rocBLAS GEMM via torch.matmul + on-the-fly INT4 dequant
  - MoE routing: Python topk + per-expert sparse dispatch
  - Expert dispatch: rocBLAS GEMM per active expert

KV cache: FP16, max_seq from meta.pt (default 4096).

Usage:
  QUANT_DIR=quantized_moe_v1 python int4_engine_moe.py --chat
  python int4_engine_moe.py --bench --ctx 128
  python int4_engine_moe.py --prompt "Hello, world"
"""
import torch
import torch.nn.functional as F
import time, sys, gc, os
from pathlib import Path

_root = Path(__file__).parent.parent  # AMD MXFP4/
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / 'hip_int4'))


def detect_gpu_arch():
    env_arch = os.environ.get('PYTORCH_ROCM_ARCH', '')
    if env_arch:
        return env_arch
    try:
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, 'gcnArchName', '')
        if arch:
            os.environ['PYTORCH_ROCM_ARCH'] = arch
            return arch
    except Exception:
        pass
    fallback = 'gfx1201'
    os.environ['PYTORCH_ROCM_ARCH'] = fallback
    return fallback


GPU_ARCH = detect_gpu_arch()

try:
    import int4_hip
except ImportError:
    print(f"int4_hip not found — building JIT for {GPU_ARCH} ...")
    from torch.utils.cpp_extension import load
    hip_dir = Path(__file__).parent / 'hip_int4'
    int4_hip = load(
        name='int4_hip',
        sources=[str(hip_dir / 'int4_decode_step.hip')],
        extra_cuda_cflags=['-O3', f'--offload-arch={GPU_ARCH}', '-std=c++17', '-Wno-unused-result'],
        extra_cflags=['-O3', '-std=c++17'],
        verbose=True,
    )
    print("JIT build complete.")

from engine_utils import RMSNorm, KVCache, precompute_rope_freqs, apply_rope
from hadamard_utils import get_hadamard
from transformers import AutoTokenizer


def _fp16_to_fp8e4m3(t: torch.Tensor) -> torch.Tensor:
    """Convert FP16 tensor to FP8 E4M3 stored as uint8 (for KV cache write in Python paths)."""
    return t.to(torch.float8_e4m3fn).view(torch.uint8)


def _fp8e4m3_to_fp16(t: torch.Tensor) -> torch.Tensor:
    """Convert FP8 E4M3 (uint8) tensor back to FP16 (for KV cache read in Python paths)."""
    return t.view(torch.float8_e4m3fn).to(torch.float16)

device = "cuda"
quant_dir = Path(os.environ.get("QUANT_DIR",
                 str(Path(__file__).parent / "quantized_moe_tiled")))
# Use FP16 attention from original model (fixes MoE router sensitivity to INT4 errors)
USE_FP16_ATTN = os.environ.get("FP16_ATTN", "1") == "1"

# ---- Load metadata ----
meta = torch.load(quant_dir / "meta.pt", weights_only=False)
model_name  = meta['model_name']
num_layers  = meta['num_layers']      # 48
hidden_size = meta['hidden_size']     # 2048
num_heads   = meta['num_heads']       # 32
num_kv_heads = meta['num_kv_heads']  # 4
head_dim    = meta['head_dim']        # 128
rms_eps     = meta['rms_eps']
rope_theta  = meta['rope_theta']
num_experts = meta['num_experts']     # 128
num_active  = meta['num_active']      # 8
moe_inter   = meta['moe_inter']       # 768
block_size  = meta.get('block_size', 32)
MAX_SEQ     = max(meta.get('max_seq', 4096), 32768)  # Support up to 32k prefill

q_dim  = num_heads * head_dim      # 4096
kv_dim = num_kv_heads * head_dim   # 512

_model_local = Path(__file__).parent / model_name.split('/')[-1]
_tokenizer_path = str(_model_local) if _model_local.exists() else model_name
tokenizer = AutoTokenizer.from_pretrained(_tokenizer_path)


def interleave_scale_zero(scales, zeros_uint8):
    """Interleave FP16 scales and FP16 zeros into [N, K/16] format."""
    zeros_fp16 = zeros_uint8.float().half()
    sz = torch.stack([scales, zeros_fp16], dim=-1).contiguous()
    return sz.view(sz.shape[0], -1)


def interleave_scale_zero_2d_transposed(scales, zeros_uint8):
    """Returns [K/32, N, 2] FP16 — transposed interleaved (scale, zero_fp16) pairs.

    Enables gemv_warp_ts to load all BLOCK_N=4 scale pairs with a single 128-bit
    load instead of 4 scattered loads, matching the expert batch kernel layout.
    """
    N, nblocks = scales.shape  # scales: [N, K/32]
    zeros_fp16 = zeros_uint8.float().half()
    sz = torch.stack([scales, zeros_fp16], dim=-1).contiguous()  # [N, K/32, 2]
    sz_t = sz.permute(1, 0, 2).contiguous()  # [K/32, N, 2]
    return sz_t


def interleave_scale_zero_2d_rowmajor(scales, zeros_uint8):
    """Returns [N, K/64, 2] FP16 — row-major interleaved (scale, zero_fp16) pairs."""
    zeros_fp16 = zeros_uint8.float().half()
    sz = torch.stack([scales, zeros_fp16], dim=-1).contiguous()  # [N, K/64, 2]
    return sz


def interleave_scale_zero_2d_transposed(scales, zeros_uint8):
    """Returns [K/64, N, 2] FP16 — transposed interleaved (scale, zero_fp16) pairs.

    For gemv_multiwave_ts_g64: vectorized uint4 reads of 4 consecutive rows'
    scale+zero pairs. Much better coalescing than row-major for multi-row GEMV.
    """
    zeros_fp16 = zeros_uint8.float().half()
    sz = torch.stack([scales, zeros_fp16], dim=-1)  # [N, K/64, 2]
    return sz.permute(1, 0, 2).contiguous()  # [K/64, N, 2]


def interleave_scale_zero_3d(scales, zeros_uint8):
    """Interleave for stacked expert tensors [E, N, K/32] -> [E, N, K/16]."""
    E, N, nblocks = scales.shape
    zeros_fp16 = zeros_uint8.float().half()
    sz = torch.stack([scales, zeros_fp16], dim=-1).contiguous()  # [E, N, K/32, 2]
    return sz.view(E, N, -1)  # [E, N, K/16]


def interleave_scale_zero_3d_rowmajor(scales, zeros_uint8):
    """Row-major interleaved scale+zero format for expert GEMV.

    Returns [E, N, K/64, 2] FP16 — coalesced across K groups within warp.
    """
    E, N, nblocks = scales.shape
    zeros_fp16 = zeros_uint8.float().half()  # [E, N, K/64]
    sz = torch.stack([scales, zeros_fp16], dim=-1).contiguous()  # [E, N, K/64, 2]
    return sz


def convert_symmetric_to_asymmetric(packed, scales, N, K):
    """Convert symmetric INT4 to asymmetric (for lm_head)."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    lo_new = (lo + 8) & 0x0F
    hi_new = (hi + 8) & 0x0F
    packed_new = (lo_new | (hi_new << 4)).to(torch.uint8)
    num_blocks = K // 32
    zeros = torch.full((N, num_blocks), 8, dtype=torch.uint8)
    sz = interleave_scale_zero(scales.half(), zeros)
    return packed_new, sz


# ---- Load weights ----
print(f"Loading MoE INT4+Had+GPTQ from {quant_dir}...")
print(f"  {num_layers} layers, {num_experts} experts/layer, top-{num_active}")
if USE_FP16_ATTN:
    print(f"  FP16 attention enabled (fixes MoE router sensitivity)")

embed_w = torch.load(quant_dir / "embed.pt", weights_only=True).half().to(device)
final_norm_w = torch.load(quant_dir / "final_norm.pt", weights_only=True).half().to(device)
rope_cos, rope_sin = precompute_rope_freqs(
    head_dim, MAX_SEQ + 128, theta=rope_theta, device=device)

# Per-layer weight lists
hip_qkv_w, hip_qkv_s, hip_qkv_N, hip_qkv_K = [], [], [], []
hip_o_w,   hip_o_s,   hip_o_N,   hip_o_K   = [], [], [], []
hip_in_norm_w, hip_post_norm_w   = [], []
hip_q_norm_w,  hip_k_norm_w      = [], []

# FP16 attention weights (loaded from original model when USE_FP16_ATTN)
fp16_qkv_w = []  # [q_dim+2*kv_dim, hidden] FP16
fp16_o_w   = []  # [hidden, q_dim] FP16

# MoE-specific per-layer
hip_gate_w   = []   # [E, hidden] FP16 router
hip_exp_gu_w = []   # [E, 2*moe_inter, hidden/2] uint8
hip_exp_gu_s = []   # [E, nit_gu, 2*moe_inter, 2] FP16 — transposed interleaved scale+zero
hip_exp_dn_w = []   # [E, hidden, moe_inter/2] uint8
hip_exp_dn_s = []   # [E, nit_dn, hidden, 2] FP16 — transposed interleaved scale+zero
hip_exp_gu_wsum = []  # W4A4 disabled — empty list triggers FP16 fallback in C++
hip_exp_dn_wsum = []
hip_exp_gu_w_tiled = []  # Not used (tiling done in C++)
hip_exp_dn_w_tiled = []  # Not used

# Load FP16 attention from original model
_sf_cache = {}
if USE_FP16_ATTN:
    import json
    _model_fp16_dir = Path(__file__).parent / model_name.split('/')[-1]
    if not (_model_fp16_dir / "model.safetensors.index.json").exists():
        print(f"  WARNING: FP16 model not found at {_model_fp16_dir}, falling back to INT4 attention")
        USE_FP16_ATTN = False
    else:
        from safetensors import safe_open
        with open(_model_fp16_dir / "model.safetensors.index.json") as f:
            _sf_index = json.load(f)
        def _load_fp16(key):
            sf = _sf_index["weight_map"][key]
            if sf not in _sf_cache:
                _sf_cache.clear()
                _sf_cache[sf] = safe_open(str(_model_fp16_dir / sf), framework="pt")
            return _sf_cache[sf].get_tensor(key).half().to(device)

for li in range(num_layers):
    if (li + 1) % 12 == 0 or li == 0:
        print(f"  Layer {li+1}/{num_layers}...", flush=True)
    ld = torch.load(quant_dir / f"layer_{li:03d}.pt", weights_only=False)

    if USE_FP16_ATTN:
        # Load FP16 attention weights from original model
        prefix = f"model.layers.{li}"
        qkv = torch.cat([
            _load_fp16(f"{prefix}.self_attn.q_proj.weight"),
            _load_fp16(f"{prefix}.self_attn.k_proj.weight"),
            _load_fp16(f"{prefix}.self_attn.v_proj.weight"),
        ], dim=0)
        fp16_qkv_w.append(qkv)
        fp16_o_w.append(_load_fp16(f"{prefix}.self_attn.o_proj.weight"))
    else:
        fp16_qkv_w.append(None)
        fp16_o_w.append(None)

    # Still load INT4 attention for C++ decode path
    hip_qkv_w.append(ld['qkv_packed'].to(device))
    hip_qkv_s.append(interleave_scale_zero_2d_rowmajor(
        ld['qkv_scales'].half(), ld['qkv_zeros']).to(device))
    hip_qkv_N.append(ld['qkv_N'])
    hip_qkv_K.append(ld['qkv_K'])

    hip_o_w.append(ld['o_packed'].to(device))
    hip_o_s.append(interleave_scale_zero_2d_rowmajor(
        ld['o_scales'].half(), ld['o_zeros']).to(device))
    hip_o_N.append(ld['o_N'])
    hip_o_K.append(ld['o_K'])

    hip_in_norm_w.append(ld['in_norm'].half().to(device))
    hip_post_norm_w.append(ld['post_norm'].half().to(device))
    hip_q_norm_w.append(ld['q_norm'].half().to(device))
    hip_k_norm_w.append(ld['k_norm'].half().to(device))

    hip_gate_w.append(ld['gate_weight'].half().to(device))

    # Stacked expert weights — keep contiguous on GPU
    egu_packed = ld['exp_gu_packed']
    # Handle both [E, N, K/2] (original) and [E, flat] (tiled) formats
    if egu_packed.dim() == 2:
        E_l = egu_packed.shape[0]
        egu_packed = egu_packed.reshape(E_l, -1)  # keep flat for tiled WMMA
    egu_packed = egu_packed.to(device)
    assert egu_packed.is_contiguous(), f"Layer {li} exp_gu_packed not contiguous!"
    hip_exp_gu_w.append(egu_packed)
    gu_sz = interleave_scale_zero_3d_rowmajor(ld['exp_gu_scales'].half(), ld['exp_gu_zeros'])
    hip_exp_gu_s.append(gu_sz.to(device))

    edn_packed = ld['exp_dn_packed']
    if edn_packed.dim() == 2:
        edn_packed = edn_packed.reshape(edn_packed.shape[0], -1)
    edn_packed = edn_packed.to(device)
    assert edn_packed.is_contiguous(), f"Layer {li} exp_dn_packed not contiguous!"
    hip_exp_dn_w.append(edn_packed)
    dn_sz = interleave_scale_zero_3d_rowmajor(ld['exp_dn_scales'].half(), ld['exp_dn_zeros'])
    hip_exp_dn_s.append(dn_sz.to(device))

    # Note: expert weights stay row-major. Tiling is done in C++ (lazy, on first prefill).

    del ld
    gc.collect()

_sf_cache.clear()  # close safetensor file handles

# LM head
lm_data = torch.load(quant_dir / "lm_head.pt", weights_only=False)
lm_N, lm_K = lm_data['N'], lm_data['K']
if 'zeros' in lm_data:
    lm_w = lm_data['packed'].to(device)
    lm_s = interleave_scale_zero_2d_rowmajor(
        lm_data['scales'].half(), lm_data['zeros']).to(device)
else:
    # Symmetric: convert to asymmetric then merge to group-64
    packed_new, sz_old = convert_symmetric_to_asymmetric(
        lm_data['packed'], lm_data['scales'], lm_N, lm_K)
    lm_w = packed_new.to(device)
    # sz_old is [N, K/16] (interleaved group-32) — reshape to [N, K/32, 2], merge pairs
    nblocks_g32 = lm_K // 32
    sz_g32 = sz_old.view(lm_N, nblocks_g32, 2)  # [N, K/32, 2]
    s_g32 = sz_g32[:, :, 0]  # [N, K/32]
    z_g32 = sz_g32[:, :, 1].round().clamp(0, 15).to(torch.uint8)
    from convert_g32_to_g64 import merge_g32_to_g64 as _merge
    s_g64, z_g64, packed_g64 = _merge(s_g32, z_g32, packed_new)
    lm_w = packed_g64.to(device)  # update packed with requantized nibbles
    lm_s = interleave_scale_zero_2d_rowmajor(s_g64, z_g64).to(device)
del lm_data

# Hadamard matrix [32, 32] FP16 on GPU
had_mat = get_hadamard(block_size, device=device, dtype=torch.float32).half().contiguous()

gpu_mb = torch.cuda.memory_allocated() / 1024**2
print(f"Model loaded: {gpu_mb:.0f} MB VRAM")
print(f"Max sequence: {MAX_SEQ} tokens\n")


# ---- Utility functions ----

def dequant_int4(packed, scales_zeros, N, K):
    """On-the-fly dequant [N, K/2] uint8 -> [N, K] FP16 (group-32 scales)."""
    return int4_hip.dequant_int4(packed, scales_zeros, N, K)


def dequant_int4_g64(packed, scales_zeros, N, K):
    """On-the-fly dequant [N, K/2] uint8 -> [N, K] FP16 (group-64 scales)."""
    return int4_hip.dequant_int4_g64(packed, scales_zeros, N, K)


def ts_to_flat_g64(sz_ts):
    """Convert transposed [K/64, N, 2] scales to flat [N, K/64*2] for dequant_int4_g64."""
    sz = sz_ts.permute(1, 0, 2).contiguous()  # [N, K/64, 2]
    return sz.contiguous().view(sz.shape[0], -1)


def rm_to_flat_g64(sz_rm):
    """Convert row-major [N, K/64, 2] scales to flat [N, K/64*2] for dequant_int4_g64."""
    return sz_rm.contiguous().view(sz_rm.shape[0], -1)


def fast_hadamard(x):
    """Block-diagonal Hadamard rotation via flat matmul."""
    shape = x.shape
    return (x.view(-1, 32) @ had_mat).view(shape)


def _expert_scale_flat(s_rm):
    """Convert expert scale from row-major [N, K/64, 2] to flat [N, K/64*2] for dequant."""
    return s_rm.reshape(s_rm.shape[0], -1)


def fast_rmsnorm(x, w):
    """RMSNorm: x [..., D], w [D]."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + rms_eps) * w.float()).half()


def head_rmsnorm(x, weight):
    """Per-head RMSNorm: x [..., D], weight [D]."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + rms_eps) * weight.float()).half()


# ---- Prefill ----

def fast_prefill_cpp(input_ids, kv):
    """C++ MoE prefill — INT4 attention (dequant+GEMM) + INT4 expert dispatch.
    Fills KV cache and returns logits for last token."""
    M = input_ids.shape[1]
    if M >= 16384:
        torch.cuda.empty_cache()  # defrag VRAM for large contexts
    qkv_s_flat = [rm_to_flat_g64(s) for s in hip_qkv_s]
    o_s_flat = [rm_to_flat_g64(s) for s in hip_o_s]
    fp16_qkv = [w for w in fp16_qkv_w if w is not None]
    fp16_o = [w for w in fp16_o_w if w is not None]
    with torch.no_grad():
        all_logits = int4_hip.prefill_moe_logits(
            input_ids.squeeze(0), embed_w, final_norm_w,
            lm_w, lm_s, lm_N, lm_K,
            fp16_qkv, fp16_o,
            hip_qkv_w, qkv_s_flat, hip_qkv_N,
            hip_o_w, o_s_flat, hip_o_N, hip_o_K,
            hip_gate_w,
            hip_exp_gu_w, hip_exp_gu_s,
            hip_exp_dn_w, hip_exp_dn_s,
            hip_in_norm_w, hip_post_norm_w,
            hip_q_norm_w, hip_k_norm_w,
            rope_cos, rope_sin,
            num_heads, num_kv_heads, head_dim,
            hidden_size, moe_inter, num_active, num_experts,
            rms_eps,
            kv.k_caches, kv.v_caches,
            had_mat,
        )
    return all_logits  # [1, vocab] — C++ returns last token only


def fast_prefill(input_ids, kv):
    """Fast prefill using rocBLAS GEMM with on-the-fly INT4 dequant.
    Fills KV cache for subsequent decode steps.
    Returns logits for last token."""
    M = input_ids.shape[1]
    pos_ids = torch.arange(M, device=device).unsqueeze(0)

    x = F.embedding(input_ids.squeeze(0), embed_w)  # [M, D]

    for li in range(num_layers):
        # Input norm
        normed_attn_raw = fast_rmsnorm(x, hip_in_norm_w[li])

        if USE_FP16_ATTN and fp16_qkv_w[li] is not None:
            # FP16 attention: no Hadamard rotation needed
            qkv = normed_attn_raw @ fp16_qkv_w[li].T
        else:
            normed_attn = fast_hadamard(normed_attn_raw)
            w = dequant_int4_g64(hip_qkv_w[li], rm_to_flat_g64(hip_qkv_s[li]), hip_qkv_N[li], hidden_size)
            qkv = normed_attn @ w.T
            del w

        # Q, K, V split + per-head norm + RoPE
        q = head_rmsnorm(
            qkv[:, :q_dim].view(M, num_heads, head_dim),
            hip_q_norm_w[li])
        k = head_rmsnorm(
            qkv[:, q_dim:q_dim+kv_dim].view(M, num_kv_heads, head_dim),
            hip_k_norm_w[li])
        v = qkv[:, q_dim+kv_dim:].view(M, num_kv_heads, head_dim)

        q = apply_rope(q.unsqueeze(0).transpose(1, 2), rope_cos, rope_sin, pos_ids)
        k = apply_rope(k.unsqueeze(0).transpose(1, 2), rope_cos, rope_sin, pos_ids)
        v = v.unsqueeze(0).transpose(1, 2)

        # Write to KV cache (FP16)
        kv.k_caches[li][:, :M, :] = k.squeeze(0).contiguous()
        kv.v_caches[li][:, :M, :] = v.squeeze(0).contiguous()

        # Attention + O projection
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        attn_out = attn.squeeze(0).transpose(0, 1).reshape(M, q_dim)
        if USE_FP16_ATTN and fp16_o_w[li] is not None:
            o_out = attn_out @ fp16_o_w[li].T
        else:
            w = dequant_int4_g64(hip_o_w[li], rm_to_flat_g64(hip_o_s[li]), hip_o_N[li], hip_o_K[li])
            o_out = fast_hadamard(attn_out) @ w.T
            del w

        x = x + o_out

        # MoE FFN
        normed_mlp_norot = fast_rmsnorm(x, hip_post_norm_w[li])
        normed_mlp = fast_hadamard(normed_mlp_norot)

        # Router must use UN-rotated input (gate_weight is stored in original space)
        router_logits = normed_mlp_norot @ hip_gate_w[li].T  # [M, E]
        router_probs = F.softmax(router_logits.float(), dim=-1).half()
        top_k_probs, top_k_ids = torch.topk(router_probs, k=num_active, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        moe_out = torch.zeros(M, hidden_size, device=device, dtype=torch.float16)

        active_experts = top_k_ids.unique().tolist()
        for expert_j in active_experts:
            mask = (top_k_ids == expert_j).any(dim=-1)
            x_j = normed_mlp[mask]

            gu_w = dequant_int4_g64(hip_exp_gu_w[li][expert_j],
                                     _expert_scale_flat(hip_exp_gu_s[li][expert_j]),
                                     2 * moe_inter, hidden_size)
            gu = x_j @ gu_w.T
            mlp_act = F.silu(gu[:, :moe_inter]) * gu[:, moe_inter:]
            mlp_rot = fast_hadamard(mlp_act)

            dn_w = dequant_int4_g64(hip_exp_dn_w[li][expert_j],
                                     _expert_scale_flat(hip_exp_dn_s[li][expert_j]),
                                     hidden_size, moe_inter)
            expert_out = mlp_rot @ dn_w.T

            w_j = (top_k_probs[mask] *
                   (top_k_ids[mask] == expert_j).half()).sum(-1)
            moe_out[mask] = moe_out[mask] + expert_out * w_j.unsqueeze(-1)

        x = x + moe_out

    kv.current_len = M
    # Final norm + LM head (last token only)
    normed_final = fast_hadamard(fast_rmsnorm(x[-1:, :], final_norm_w))
    w = dequant_int4_g64(lm_w, rm_to_flat_g64(lm_s), lm_N, lm_K)
    logits = normed_final @ w.T
    del w
    return logits


def fast_prefill_v2(input_ids, kv):
    """Optimized prefill: tiled dequant+GEMM for attention, batched expert dispatch.
    Key optimizations over fast_prefill:
    1. dequant_gemm_tiled_g64 for QKV/O (L2-friendly tiling)
    2. Batched expert dispatch: sort tokens by expert on GPU, one GEMM per expert
    3. Pre-allocated dequant buffers reused across layers
    """
    M = input_ids.shape[1]
    pos_ids = torch.arange(M, device=device).unsqueeze(0)
    x = F.embedding(input_ids.squeeze(0), embed_w)  # [M, D]
    opts = dict(device=device, dtype=torch.float16)

    # Pre-allocate dequant buffers (reused across layers)
    max_qkv_N = max(hip_qkv_N)
    max_o_K = max(hip_o_K) if hip_o_K else q_dim
    w_buf_qkv = torch.empty(max_qkv_N * hidden_size, **opts)
    w_buf_o = torch.empty(hidden_size * max_o_K, **opts)
    w_buf_gu = torch.empty(2 * moe_inter * hidden_size, **opts)
    w_buf_dn = torch.empty(hidden_size * moe_inter, **opts)
    qkv_out = torch.empty(M, max_qkv_N, **opts)
    o_out_buf = torch.empty(M, hidden_size, **opts)

    for li in range(num_layers):
        normed_attn_raw = fast_rmsnorm(x, hip_in_norm_w[li])

        if USE_FP16_ATTN and fp16_qkv_w[li] is not None:
            qkv = normed_attn_raw @ fp16_qkv_w[li].T
        else:
            normed_attn = fast_hadamard(normed_attn_raw)
            Nq = hip_qkv_N[li]
            qkv_slice = qkv_out[:, :Nq]
            int4_hip.dequant_gemm_tiled_g64(
                normed_attn, hip_qkv_w[li], rm_to_flat_g64(hip_qkv_s[li]),
                Nq, hidden_size, w_buf_qkv, qkv_slice, 4096)
            qkv = qkv_slice

        q = head_rmsnorm(qkv[:, :q_dim].view(M, num_heads, head_dim), hip_q_norm_w[li])
        k = head_rmsnorm(qkv[:, q_dim:q_dim+kv_dim].view(M, num_kv_heads, head_dim), hip_k_norm_w[li])
        v = qkv[:, q_dim+kv_dim:].view(M, num_kv_heads, head_dim)

        q = apply_rope(q.unsqueeze(0).transpose(1, 2), rope_cos, rope_sin, pos_ids)
        k = apply_rope(k.unsqueeze(0).transpose(1, 2), rope_cos, rope_sin, pos_ids)
        v = v.unsqueeze(0).transpose(1, 2)
        kv.k_caches[li][:, :M, :] = k.squeeze(0).contiguous()
        kv.v_caches[li][:, :M, :] = v.squeeze(0).contiguous()

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        attn_out = attn.squeeze(0).transpose(0, 1).reshape(M, q_dim)

        if USE_FP16_ATTN and fp16_o_w[li] is not None:
            o_proj = attn_out @ fp16_o_w[li].T
        else:
            No = hip_o_N[li]
            Ko = hip_o_K[li]
            o_proj_slice = o_out_buf[:, :No]
            int4_hip.dequant_gemm_tiled_g64(
                fast_hadamard(attn_out), hip_o_w[li], rm_to_flat_g64(hip_o_s[li]),
                No, Ko, w_buf_o, o_proj_slice, 4096)
            o_proj = o_proj_slice

        x = x + o_proj

        # MoE FFN — batched expert dispatch
        normed_mlp_norot = fast_rmsnorm(x, hip_post_norm_w[li])
        normed_mlp = fast_hadamard(normed_mlp_norot)

        router_logits = normed_mlp_norot @ hip_gate_w[li].T  # [M, E]
        router_probs = F.softmax(router_logits.float(), dim=-1).half()
        top_k_probs, top_k_ids = torch.topk(router_probs, k=num_active, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        moe_out = torch.zeros(M, hidden_size, **opts)

        # Batched expert dispatch: group tokens by expert
        active_experts = top_k_ids.unique()
        for expert_j in active_experts.tolist():
            mask = (top_k_ids == expert_j).any(dim=-1)
            n_tok = mask.sum().item()
            if n_tok == 0:
                continue
            x_j = normed_mlp[mask]  # [n_tok, D]

            # Tiled dequant+GEMM for GU (avoids materializing full weight matrix)
            gu_out = torch.empty(n_tok, 2 * moe_inter, **opts)
            gu_packed = hip_exp_gu_w[li][expert_j]
            gu_scales = _expert_scale_flat(hip_exp_gu_s[li][expert_j])
            int4_hip.dequant_gemm_tiled_g64(
                x_j, gu_packed, gu_scales,
                2 * moe_inter, hidden_size, w_buf_gu, gu_out, 1536)

            mlp_act = F.silu(gu_out[:, :moe_inter]) * gu_out[:, moe_inter:]
            mlp_rot = fast_hadamard(mlp_act)

            # Tiled dequant+GEMM for DN
            dn_out = torch.empty(n_tok, hidden_size, **opts)
            dn_packed = hip_exp_dn_w[li][expert_j]
            dn_scales = _expert_scale_flat(hip_exp_dn_s[li][expert_j])
            int4_hip.dequant_gemm_tiled_g64(
                mlp_rot, dn_packed, dn_scales,
                hidden_size, moe_inter, w_buf_dn, dn_out, 2048)

            w_j = (top_k_probs[mask] * (top_k_ids[mask] == expert_j).half()).sum(-1)
            moe_out[mask] = moe_out[mask] + dn_out * w_j.unsqueeze(-1)

        x = x + moe_out

    kv.current_len = M
    normed_final = fast_hadamard(fast_rmsnorm(x[-1:, :], final_norm_w))
    lm_out = torch.empty(1, lm_N, **opts)
    int4_hip.dequant_gemm_tiled_g64(
        normed_final, lm_w, rm_to_flat_g64(lm_s), lm_N, lm_K, w_buf_qkv, lm_out, 4096)
    return lm_out


def fast_prefill_ppl(input_ids):
    """Prefill returning ALL position logits for PPL evaluation. No KV cache."""
    M = input_ids.shape[1]
    pos_ids = torch.arange(M, device=device).unsqueeze(0)
    x = F.embedding(input_ids.squeeze(0), embed_w)  # [M, D]

    for li in range(num_layers):
        normed_attn_raw = fast_rmsnorm(x, hip_in_norm_w[li])
        if USE_FP16_ATTN and fp16_qkv_w[li] is not None:
            qkv = normed_attn_raw @ fp16_qkv_w[li].T
        else:
            normed_attn = fast_hadamard(normed_attn_raw)
            w = dequant_int4_g64(hip_qkv_w[li], rm_to_flat_g64(hip_qkv_s[li]), hip_qkv_N[li], hidden_size)
            qkv = normed_attn @ w.T; del w

        q = head_rmsnorm(qkv[:, :q_dim].view(M, num_heads, head_dim), hip_q_norm_w[li])
        k = head_rmsnorm(qkv[:, q_dim:q_dim+kv_dim].view(M, num_kv_heads, head_dim), hip_k_norm_w[li])
        v = qkv[:, q_dim+kv_dim:].view(M, num_kv_heads, head_dim)
        q = apply_rope(q.unsqueeze(0).transpose(1, 2), rope_cos, rope_sin, pos_ids)
        k = apply_rope(k.unsqueeze(0).transpose(1, 2), rope_cos, rope_sin, pos_ids)
        v = v.unsqueeze(0).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        attn_out = attn.squeeze(0).transpose(0, 1).reshape(M, q_dim)
        if USE_FP16_ATTN and fp16_o_w[li] is not None:
            o_out = attn_out @ fp16_o_w[li].T
        else:
            w = dequant_int4_g64(hip_o_w[li], rm_to_flat_g64(hip_o_s[li]), hip_o_N[li], hip_o_K[li])
            o_out = fast_hadamard(attn_out) @ w.T; del w
        x = x + o_out

        normed_mlp_norot = fast_rmsnorm(x, hip_post_norm_w[li])
        normed_mlp = fast_hadamard(normed_mlp_norot)
        router_logits = normed_mlp_norot @ hip_gate_w[li].T
        router_probs = F.softmax(router_logits.float(), dim=-1).half()
        top_k_probs, top_k_ids = torch.topk(router_probs, k=num_active, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        moe_out = torch.zeros(M, hidden_size, device=device, dtype=torch.float16)
        active_experts = top_k_ids.unique().tolist()
        for expert_j in active_experts:
            mask = (top_k_ids == expert_j).any(dim=-1)
            if not mask.any(): continue
            x_j = normed_mlp[mask]
            gu_w = dequant_int4_g64(hip_exp_gu_w[li][expert_j],
                                     _expert_scale_flat(hip_exp_gu_s[li][expert_j]),
                                     2*moe_inter, hidden_size)
            gu = x_j @ gu_w.T
            mlp_act = F.silu(gu[:, :moe_inter]) * gu[:, moe_inter:]
            mlp_rot = fast_hadamard(mlp_act)
            dn_w = dequant_int4_g64(hip_exp_dn_w[li][expert_j],
                                     _expert_scale_flat(hip_exp_dn_s[li][expert_j]),
                                     hidden_size, moe_inter)
            expert_out = mlp_rot @ dn_w.T
            w_j = (top_k_probs[mask] * (top_k_ids[mask] == expert_j).half()).sum(-1)
            moe_out[mask] = moe_out[mask] + expert_out * w_j.unsqueeze(-1)
        x = x + moe_out

    # ALL positions: final norm + LM head (chunked to avoid OOM)
    normed_final = fast_hadamard(fast_rmsnorm(x, final_norm_w))
    lm_s_flat = rm_to_flat_g64(lm_s)
    chunk_v = 16384  # dequant vocab in chunks
    all_logits = torch.empty(M, lm_N, device=device, dtype=torch.float16)
    for v0 in range(0, lm_N, chunk_v):
        v1 = min(v0 + chunk_v, lm_N)
        w_chunk = dequant_int4_g64(lm_w[v0:v1].contiguous(),
                                    lm_s_flat[v0:v1].contiguous(), v1 - v0, lm_K)
        all_logits[:, v0:v1] = normed_final @ w_chunk.T
        del w_chunk
    return all_logits


# ---- Decode ----

_half = head_dim // 2

def _apply_rope_decode(vec, pos_idx):
    """Apply RoPE at decode position. vec: [..., head_dim]."""
    cos_p = rope_cos[pos_idx]  # [head_dim/2]
    sin_p = rope_sin[pos_idx]
    v1, v2 = vec[..., :_half], vec[..., _half:]
    return torch.cat([v1 * cos_p - v2 * sin_p,
                      v2 * cos_p + v1 * sin_p], dim=-1)


def decode_step_logits(hidden, pos_idx, kv):
    """Single decode step — all 48 layers in C++, FP16 KV cache.
    Returns logits [1, 1, vocab]."""
    return int4_hip.decode_step_moe_logits(
        hidden.view(hidden_size),
        # Attention weights
        hip_qkv_w, hip_qkv_s, hip_qkv_N,
        hip_o_w,   hip_o_s,   hip_o_N,
        # MoE weights
        hip_gate_w,
        hip_exp_gu_w, hip_exp_gu_s,
        hip_exp_dn_w, hip_exp_dn_s,
        hip_exp_gu_wsum, hip_exp_dn_wsum,
        # Norms
        hip_in_norm_w, hip_post_norm_w,
        hip_q_norm_w,  hip_k_norm_w,
        # RoPE
        rope_cos, rope_sin,
        int(pos_idx),
        # Model config
        num_heads, num_kv_heads, head_dim,
        hidden_size, moe_inter, num_active,
        rms_eps,
        # FP16 KV caches
        kv.k_caches, kv.v_caches,
        # Final norm + LM head
        final_norm_w, lm_w, lm_s, lm_N, lm_K,
    )


def decode_step_logits_graph(hidden, pos_idx, kv):
    """Single decode step with HIP Graph — captures on first call, replays after.
    Uses fast replay path (minimal args) after first call to avoid pybind11 overhead."""
    # Try fast replay first (returns None if graph not yet captured or KV changed)
    result = int4_hip.graph_replay_fast(hidden.view(hidden_size), int(pos_idx), kv.k_caches[0])
    if result is not None:
        return result
    # First call: full argument list for graph capture
    return int4_hip.decode_step_moe_graph_logits(
        hidden.view(hidden_size),
        hip_qkv_w, hip_qkv_s, hip_qkv_N,
        hip_o_w,   hip_o_s,   hip_o_N,
        hip_gate_w,
        hip_exp_gu_w, hip_exp_gu_s,
        hip_exp_dn_w, hip_exp_dn_s,
        hip_exp_gu_wsum, hip_exp_dn_wsum,
        hip_in_norm_w, hip_post_norm_w,
        hip_q_norm_w,  hip_k_norm_w,
        rope_cos, rope_sin,
        int(pos_idx),
        num_heads, num_kv_heads, head_dim,
        hidden_size, moe_inter, num_active,
        rms_eps,
        kv.k_caches, kv.v_caches,
        final_norm_w, lm_w, lm_s, lm_N, lm_K,
    )


def decode_step_logits_fp16attn(hidden, pos_idx, kv):
    """C++ decode step — FP16 attention GEMV + INT4 expert GEMV. All layers in C++.
    Returns logits [1, 1, vocab]."""
    return int4_hip.decode_step_moe_fp16attn_logits(
        hidden.view(hidden_size),
        # FP16 attention weights
        fp16_qkv_w, fp16_o_w,
        # MoE weights (INT4)
        hip_gate_w,
        hip_exp_gu_w, hip_exp_gu_s,
        hip_exp_dn_w, hip_exp_dn_s,
        # Norms
        hip_in_norm_w, hip_post_norm_w,
        hip_q_norm_w,  hip_k_norm_w,
        # RoPE
        rope_cos, rope_sin,
        int(pos_idx),
        # Model config
        num_heads, num_kv_heads, head_dim,
        hidden_size, moe_inter, num_active,
        rms_eps,
        # FP16 KV caches
        kv.k_caches, kv.v_caches,
        # Final norm + LM head
        final_norm_w, lm_w, lm_s, lm_N, lm_K,
    )


def decode_step_logits_py(hidden, pos_idx, kv):
    """Python decode step with FP16 attention + INT4 MoE experts."""
    x = hidden.squeeze()
    seq_len = pos_idx + 1

    for li in range(num_layers):
        x_normed_raw = fast_rmsnorm(x, hip_in_norm_w[li])

        if USE_FP16_ATTN and fp16_qkv_w[li] is not None:
            qkv = x_normed_raw @ fp16_qkv_w[li].T
        else:
            x_normed = fast_hadamard(x_normed_raw)
            w_qkv = dequant_int4_g64(hip_qkv_w[li], rm_to_flat_g64(hip_qkv_s[li]), hip_qkv_N[li], hidden_size)
            qkv = x_normed @ w_qkv.T
            del w_qkv

        q = head_rmsnorm(qkv[:q_dim].view(num_heads, head_dim), hip_q_norm_w[li])
        k_new = head_rmsnorm(qkv[q_dim:q_dim+kv_dim].view(num_kv_heads, head_dim), hip_k_norm_w[li])
        v_new = qkv[q_dim+kv_dim:].view(num_kv_heads, head_dim)
        q = _apply_rope_decode(q, pos_idx)
        k_new = _apply_rope_decode(k_new, pos_idx)
        kv.k_caches[li][:, pos_idx, :] = k_new
        kv.v_caches[li][:, pos_idx, :] = v_new

        k_hist = kv.k_caches[li][:, :seq_len, :]
        v_hist = kv.v_caches[li][:, :seq_len, :]
        attn = F.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(2), k_hist.unsqueeze(0), v_hist.unsqueeze(0),
            is_causal=False, enable_gqa=True)
        attn_out = attn.squeeze(0).squeeze(1).reshape(q_dim)

        if USE_FP16_ATTN and fp16_o_w[li] is not None:
            x = x + attn_out @ fp16_o_w[li].T
        else:
            w_o = dequant_int4_g64(hip_o_w[li], rm_to_flat_g64(hip_o_s[li]), hip_o_N[li], hip_o_K[li])
            x = x + fast_hadamard(attn_out) @ w_o.T
            del w_o

        normed_post_norot = fast_rmsnorm(x, hip_post_norm_w[li])
        normed_post = fast_hadamard(normed_post_norot)
        router_logits = hip_gate_w[li] @ normed_post_norot
        top_k_probs, top_k_ids = torch.topk(F.softmax(router_logits.float(), dim=-1), k=num_active)
        top_k_probs = top_k_probs / top_k_probs.sum()
        # moe_gemv_top_k_v2 expects [E, N, K/16] non-transposed flat format.
        # hip_exp_gu_s[li] is [E, nit, N, 2]; convert by permuting and viewing.
        E_l = hip_exp_gu_s[li].shape[0]
        gu_nit = hip_exp_gu_s[li].shape[1]
        gu_N_l = hip_exp_gu_s[li].shape[2]
        gu_s_flat = hip_exp_gu_s[li].permute(0, 2, 1, 3).contiguous().view(E_l, gu_N_l, gu_nit * 2)
        dn_nit = hip_exp_dn_s[li].shape[1]
        dn_N_l = hip_exp_dn_s[li].shape[2]
        dn_s_flat = hip_exp_dn_s[li].permute(0, 2, 1, 3).contiguous().view(E_l, dn_N_l, dn_nit * 2)
        moe_out = int4_hip.moe_gemv_top_k_v2(
            normed_post, hip_exp_gu_w[li], gu_s_flat,
            hip_exp_dn_w[li], dn_s_flat,
            top_k_ids, top_k_probs, moe_inter, hidden_size)
        x = x + moe_out

    normed_final = fast_hadamard(fast_rmsnorm(x, final_norm_w))
    w = dequant_int4_g64(lm_w, rm_to_flat_g64(lm_s), lm_N, lm_K)
    logits = normed_final @ w.T
    del w
    return logits.unsqueeze(0).unsqueeze(0)


# ---- KV cache in FP16 (2x 32GB GPUs have ample headroom) ----
# FP16 KV cache avoids FP8 decode complexity in per-layer Python decode loop.
# VRAM: 48 layers × 2 × 4 kv_heads × 4096 × 128 × 2B ≈ 402 MB — negligible.

class MoeKVCache:
    """KV cache in FP16. Shapes: [kv_heads, max_seq, head_dim]."""
    def __init__(self, num_layers, num_kv_heads, max_seq, head_dim, device):
        self.k_caches = [
            torch.zeros(num_kv_heads, max_seq, head_dim,
                        dtype=torch.float16, device=device)
            for _ in range(num_layers)]
        self.v_caches = [
            torch.zeros(num_kv_heads, max_seq, head_dim,
                        dtype=torch.float16, device=device)
            for _ in range(num_layers)]
        self.current_len = 0


# ---- Generation ----

def _sample(logits, temperature, top_p, generated_ids=None, rep_penalty=1.15,
            freq_penalty=0.1, pres_penalty=0.1):
    logits_f = logits[0, 0, :].float() if logits.dim() == 3 else logits[0, :].float()
    # Repetition + frequency + presence penalty (llama.cpp style)
    if generated_ids and (rep_penalty != 1.0 or freq_penalty > 0 or pres_penalty > 0):
        from collections import Counter
        counts = Counter(generated_ids)
        penalty_ids = torch.tensor(list(counts.keys()), dtype=torch.long,
                                   device=logits_f.device)
        freq = torch.tensor([counts[t] for t in counts], dtype=torch.float32,
                            device=logits_f.device)
        selected = logits_f[penalty_ids]
        if rep_penalty != 1.0:
            selected = torch.where(selected > 0, selected / rep_penalty,
                                   selected * rep_penalty)
        selected -= freq_penalty * freq   # harder penalty for repeated tokens
        selected -= pres_penalty          # flat penalty for any seen token
        logits_f[penalty_ids] = selected
    if temperature > 0:
        logits_f = logits_f / temperature
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits_f, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = -float('inf')
            logits_f = torch.zeros_like(logits_f).scatter_(0, sorted_idx, sorted_logits)
        probs = F.softmax(logits_f, dim=-1)
        return torch.multinomial(probs.unsqueeze(0), 1).squeeze()
    else:
        return logits_f.argmax(-1)

# Stop token IDs for Qwen3
_STOP_IDS = set()
for _tok_str in ['<|im_end|>', '<|endoftext|>']:
    _ids = tokenizer.encode(_tok_str, add_special_tokens=False)
    if len(_ids) == 1:
        _STOP_IDS.add(_ids[0])
if tokenizer.eos_token_id is not None:
    _STOP_IDS.add(tokenizer.eos_token_id)


def generate(prompt, max_tokens=256, temperature=0.7, top_p=0.9, rep_penalty=1.15,
             freq_penalty=0.1, pres_penalty=0.1, think=True):
    """Generate from prompt. Prefill via GEMM, decode via Python/C++ MoE."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        enable_thinking=think).to(device)
    S = input_ids.shape[1]
    kv = MoeKVCache(num_layers, num_kv_heads, S + max_tokens + 16, head_dim, device)

    # Select decode function: FP16 attn uses Python path, INT4-only uses C++ path
    # Always use INT4 attention for decode (20% faster, ~same quality)
    _decode_fn = decode_step_logits

    t0 = time.time()
    with torch.no_grad():
        logits = fast_prefill_cpp(input_ids, kv)
    prefill_time = time.time() - t0

    next_id = _sample(logits, temperature, top_p)
    generated = [next_id.item()]
    pos = S

    t_dec = time.time()
    with torch.no_grad():
        for _ in range(max_tokens - 1):
            hidden = F.embedding(next_id.view(1), embed_w).view(1, 1, hidden_size)
            logits = _decode_fn(hidden, pos, kv)
            next_id = _sample(logits, temperature, top_p,
                              generated_ids=generated, rep_penalty=rep_penalty,
                              freq_penalty=freq_penalty, pres_penalty=pres_penalty)
            tok = next_id.item()
            generated.append(tok)
            pos += 1
            if tok in _STOP_IDS:
                break

    dec_time = time.time() - t_dec
    result = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\n--- Stats ---")
    print(f"Prefill: {S} tokens, {S/prefill_time:.0f} t/s")
    print(f"Decode:  {len(generated)} tokens, {len(generated)/dec_time:.1f} t/s")
    print(f"VRAM:    {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    return result


def generate_streaming(prompt, max_tokens=512, temperature=0.7, top_p=0.9, rep_penalty=1.15,
                       freq_penalty=0.1, pres_penalty=0.1, think=True):
    """Stream tokens to stdout."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        enable_thinking=think).to(device)
    S = input_ids.shape[1]
    kv = MoeKVCache(num_layers, num_kv_heads, S + max_tokens + 16, head_dim, device)

    # Always use INT4 attention for decode (20% faster, ~same quality)
    _decode_fn = decode_step_logits

    t0 = time.time()
    with torch.no_grad():
        logits = fast_prefill_cpp(input_ids, kv)
    prefill_time = time.time() - t0

    next_id = _sample(logits, temperature, top_p)
    generated = [next_id.item()]
    pos = S
    prev_text = ""
    in_think = think_done = False

    t_dec = time.time()
    with torch.no_grad():
        for _ in range(max_tokens - 1):
            hidden = F.embedding(next_id.view(1), embed_w).view(1, 1, hidden_size)
            logits = _decode_fn(hidden, pos, kv)
            next_id = _sample(logits, temperature, top_p,
                              generated_ids=generated, rep_penalty=rep_penalty,
                              freq_penalty=freq_penalty, pres_penalty=pres_penalty)
            tok = next_id.item()
            generated.append(tok)
            pos += 1

            if tok in _STOP_IDS:
                break

            raw_text = tokenizer.decode(generated, skip_special_tokens=False)
            if '<|im_end|>' in raw_text:
                break

            if not think_done:
                if '<think>' in raw_text and '</think>' not in raw_text:
                    in_think = True
                    if len(generated) < 5:
                        sys.stdout.write("\033[2m(thinking...)\033[0m ")
                        sys.stdout.flush()
                    continue
                if in_think and '</think>' in raw_text:
                    in_think = False
                    think_done = True
                    idx = raw_text.index('</think>') + len('</think>')
                    clean = raw_text[idx:].replace('<|im_end|>', '').strip()
                    if clean:
                        sys.stdout.write(clean)
                        sys.stdout.flush()
                    prev_text = clean
                    continue
                if in_think:
                    continue

            clean = tokenizer.decode(generated, skip_special_tokens=True)
            if think_done:
                raw = tokenizer.decode(generated, skip_special_tokens=False)
                idx = raw.index('</think>') + len('</think>') if '</think>' in raw else 0
                clean = raw[idx:].replace('<|im_end|>', '').replace('<|im_start|>', '')
            new_text = clean[len(prev_text):]
            if new_text:
                sys.stdout.write(new_text)
                sys.stdout.flush()
            prev_text = clean

    dec_time = time.time() - t_dec
    dec_tokens = len(generated)
    print(f"\n\n--- Stats ---")
    print(f"Prefill: {S} tokens in {prefill_time*1000:.0f}ms ({S/prefill_time:.0f} t/s)")
    print(f"Decode:  {dec_tokens} tokens in {dec_time*1000:.0f}ms ({dec_tokens/dec_time:.1f} t/s)")
    print(f"VRAM:    {torch.cuda.memory_allocated()/1024**2:.0f} MB")


def interactive_chat(max_tokens=512, temperature=0.7, top_p=0.9):
    """Interactive chat loop."""
    print("=" * 60)
    print("  INT4 MoE Engine — Qwen3-30B-A3B")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    print(f"  Settings: temp={temperature}, top_p={top_p}, max={max_tokens}")
    print("  Type 'quit' to exit, 'clear' to reset.")
    print("=" * 60)

    while True:
        try:
            print()
            user_input = input("\033[1;32mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("Bye!")
            break
        if user_input.lower() == 'clear':
            print("\033[2J\033[H", end="")
            continue

        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        print(f"\033[1;34mAssistant:\033[0m ", end="")
        generate_streaming(prompt, max_tokens=max_tokens,
                           temperature=temperature, top_p=top_p)


def benchmark_decode(n_tokens=30, ctx=128, use_graph=False):
    """Benchmark decode speed. use_graph=True uses HIP Graph path."""
    step_fn = decode_step_logits_graph if use_graph else decode_step_logits
    label = "Graph" if use_graph else "Kernel"

    kv = MoeKVCache(num_layers, num_kv_heads, ctx + n_tokens + 16, head_dim, device)
    # Fill KV cache with dummy data for realistic context benchmark
    for li in range(num_layers):
        kv.k_caches[li][:, :ctx, :] = 0.01
        kv.v_caches[li][:, :ctx, :] = 0.01
    kv.current_len = ctx

    h = torch.randn(1, 1, hidden_size, dtype=torch.float16, device=device)

    # Warmup (extra warmup for graph to capture + warm caches)
    n_warmup = 5 if use_graph else 2
    for i in range(n_warmup):
        with torch.no_grad():
            _ = step_fn(h, ctx + i, kv)
    torch.cuda.synchronize()

    # Reset KV cache for clean measurement
    kv = MoeKVCache(num_layers, num_kv_heads, ctx + n_tokens + 16, head_dim, device)
    for li in range(num_layers):
        kv.k_caches[li][:, :ctx, :] = 0.01
        kv.v_caches[li][:, :ctx, :] = 0.01

    times = []
    for i in range(n_tokens):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = step_fn(h, ctx + 2 + i, kv)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    avg = sum(times) / len(times)
    # Bulk measurement (no per-token sync)
    kv2 = MoeKVCache(num_layers, num_kv_heads, ctx + n_tokens + 16, head_dim, device)
    for li in range(num_layers):
        kv2.k_caches[li][:, :ctx, :] = 0.01
        kv2.v_caches[li][:, :ctx, :] = 0.01
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_tokens):
            _ = step_fn(h, ctx + i, kv2)
    torch.cuda.synchronize()
    bulk_avg = (time.time() - t0) / n_tokens
    print(f"[{label}] Decode ctx={ctx}: {avg*1000:.1f} ms/tok, {1/avg:.1f} t/s  (bulk: {bulk_avg*1000:.1f} ms/tok, {1/bulk_avg:.1f} t/s)")
    return 1 / avg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--ctx', type=int, default=128)
    parser.add_argument('--graph', action='store_true', help='Use HIP Graph for decode')
    args = parser.parse_args()

    if args.chat:
        interactive_chat(max_tokens=args.max_tokens,
                         temperature=args.temperature, top_p=args.top_p)
    elif args.bench:
        print("--- Kernel launch baseline ---")
        for ctx in [128, 512, 1024]:
            benchmark_decode(30, ctx, use_graph=False)
        print("\n--- HIP Graph (production) ---")
        for ctx in [128, 512, 1024]:
            benchmark_decode(30, ctx, use_graph=True)
        # Pure C++ graph replay benchmark (no Python overhead at all)
        print("\n--- C++ Graph Replay (zero Python overhead) ---")
        int4_hip.bench_graph_replay(100, 128)
    elif args.prompt:
        result = generate(args.prompt, max_tokens=args.max_tokens)
        print(f"\n{result}")
    else:
        benchmark_decode(20, 128)
        print()
        result = generate("The meaning of life is", max_tokens=50)
        print(f"\n{result}")
