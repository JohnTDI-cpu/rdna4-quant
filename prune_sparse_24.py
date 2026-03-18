#!/usr/bin/env python3
"""Prune expert weights to 2:4 sparsity + quantize to signed symmetric INT4.
Output: compressed format for V_SWMMAC_I32_16X16X64_IU4"""
import torch, json, os, time, sys
from safetensors import safe_open

model_dir = '/home/janusz/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39'
src_quant = 'qwen3_30b_a3b/quantized_moe_v2_g64'  # existing dense quantized (for non-expert weights)
dst_dir = 'qwen3_30b_a3b/quantized_moe_sparse24'
os.makedirs(dst_dir, exist_ok=True)

def hadamard_rotate(w, block_size=32):
    """Apply block-diagonal Hadamard rotation (same as quantize_moe_gptq.py)"""
    N, K = w.shape
    # Build Hadamard matrix
    H = torch.ones(1, 1)
    while H.shape[0] < block_size:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H = H / (block_size ** 0.5)
    H = H.to(w.dtype).to(w.device)
    return (w.reshape(N, K // block_size, block_size) @ H).reshape(N, K)

def prune_24_and_quantize_sym(w_fp, group_size=64):
    """BF16 weight → Hadamard rotate → 2:4 prune → signed symmetric INT4 quantize.
    Returns: packed_sparse [N, K//4] uint8, index [N, K//4] uint8, scales [N, ng] FP16"""
    N, K = w_fp.shape
    
    # 1. Hadamard rotate
    w = hadamard_rotate(w_fp.float(), 32)
    
    # 2. Prune 2:4: in each group of 4, keep top-2 by magnitude
    wg = w.reshape(N, K//4, 4)
    abs_g = wg.abs()
    _, top2 = abs_g.topk(2, dim=-1)
    mask = torch.zeros_like(wg, dtype=torch.bool)
    mask.scatter_(2, top2, True)
    pruned = wg * mask.float()
    
    # 3. Signed symmetric INT4 quantize (on pruned values)
    # Per-group scale = absmax / 7
    ng = K // group_size
    pg = pruned.reshape(N, ng, group_size)
    amax = pg.abs().max(dim=-1).values.clamp(min=1e-10)
    scale = amax / 7.0
    
    # Quantize: signed nibble = round(val / scale), clamp [-8, +7]
    scale_exp = scale.unsqueeze(-1).expand(N, ng, group_size)
    signed_nib = (pg / scale_exp).round().clamp(-8, 7).to(torch.int8)
    # Store as unsigned: stored = signed + 8 → 0..15
    stored = (signed_nib.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
    stored_flat = stored.reshape(N, K)
    
    # 4. Compress 2:4: extract 2 non-zero values + 2-bit index per group of 4
    mask_flat = mask.reshape(N, K)
    sg = stored_flat.reshape(N, K//4, 4)
    mg = mask_flat.reshape(N, K//4, 4)
    
    # Extract non-zero values (always 2 per group)
    # For each group: val0 = sg[pos0], val1 = sg[pos1]
    # Pack: (val0 & 0xF) | (val1 << 4) → 1 byte per group
    packed = torch.zeros(N, K//4, dtype=torch.uint8)
    index = torch.zeros(N, K//4, dtype=torch.uint8)
    
    for gi in range(K//4):
        group_vals = sg[:, gi, :]  # [N, 4]
        group_mask = mg[:, gi, :]  # [N, 4]
        # Find positions of non-zero elements
        # positions: [N, 2] — which 2 of 4 are kept
        pos = torch.zeros(N, 2, dtype=torch.long)
        for n in range(N):
            nz = group_mask[n].nonzero(as_tuple=True)[0]
            if len(nz) >= 2:
                pos[n, 0] = nz[0]
                pos[n, 1] = nz[1]
        
        v0 = group_vals[torch.arange(N), pos[:, 0]]
        v1 = group_vals[torch.arange(N), pos[:, 1]]
        packed[:, gi] = (v0 & 0xF) | (v1 << 4)
        # Index: 2 bits per position, idx0 in bits [0:1], idx1 in bits [2:3]
        index[:, gi] = (pos[:, 0] | (pos[:, 1] << 2)).to(torch.uint8)
    
    return packed, index, scale.half()

# Quick test on one expert
print("Testing on 1 expert...", flush=True)
with open(f'{model_dir}/model.safetensors.index.json') as f:
    sf_idx = json.load(f)

shard = sf_idx['weight_map']['model.layers.0.mlp.experts.0.gate_proj.weight']
with safe_open(f'{model_dir}/{shard}', framework='pt', device='cpu') as f:
    w = f.get_tensor('model.layers.0.mlp.experts.0.gate_proj.weight')

packed, index, scales = prune_24_and_quantize_sym(w, group_size=64)
print(f"packed: {packed.shape}, index: {index.shape}, scales: {scales.shape}")
print(f"Compression: {w.numel()*2} bytes → {packed.numel() + index.numel() + scales.numel()*2} bytes = "
      f"{(packed.numel() + index.numel() + scales.numel()*2) / (w.numel()*2) * 100:.0f}%")
print("Test OK!", flush=True)
