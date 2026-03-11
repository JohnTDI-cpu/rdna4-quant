"""
Quantization v5 — Pure INT4, NO mixed precision, GPTQ with proper calibration.

Same as v4 but ALL weights are INT4 (no INT8 groups).
Goal: ~6 GB model vs 9.2 GB v4 mixed vs 8.4 GB Q4_K_M.

Usage:
  python3 quantize_v5_pure_int4.py --save-dir quantized_v5_pure_int4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, sys, gc, json, os, argparse
from pathlib import Path

os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1201')  # Override: PYTORCH_ROCM_ARCH=gfx1100
os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_datasets'

sys.path.insert(0, str(Path(__file__).parent))
from int4_quant_v2 import (gptq_quantize_int4_v2, quantize_to_int4_asymmetric,
                            pack_int4_unsigned, unpack_int4_unsigned,
                            find_optimal_scale_int4_asym,
                            quantize_to_int4_symmetric, pack_int4)
from hadamard_utils import get_hadamard, rotate_weight
from safetensors.torch import load_file


def apply_rotation_blockdiag(W, R, block_size):
    N, K = W.shape
    return torch.einsum('nbk,kj->nbj', W.reshape(N, K // block_size, block_size), R).reshape(N, K)


def transform_hessian_rotation(H, R, block_size=32):
    K = H.shape[0]
    nb = K // block_size
    H_cpu = H.cpu().float()
    R_cpu = R.cpu().float()
    H_4d = H_cpu.reshape(nb, block_size, nb, block_size)
    H_4d = torch.einsum('ba,ibjk->iajk', R_cpu, H_4d)
    H_4d = torch.einsum('dc,ijkd->ijkc', R_cpu, H_4d)
    return H_4d.reshape(K, K).to(H.device)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight.float())
        self.eps = eps
    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + self.eps) * self.weight).to(x.dtype)


class Qwen3Layer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 intermediate_size, rms_eps, weights):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size

        self.input_layernorm = Qwen3RMSNorm(weights['in_norm'], rms_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(weights['post_norm'], rms_eps)
        self.q_norm = Qwen3RMSNorm(weights['q_norm'], rms_eps)
        self.k_norm = Qwen3RMSNorm(weights['k_norm'], rms_eps)

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.q_proj.weight = nn.Parameter(weights['q_proj'])
        self.k_proj.weight = nn.Parameter(weights['k_proj'])
        self.v_proj.weight = nn.Parameter(weights['v_proj'])
        self.o_proj.weight = nn.Parameter(weights['o_proj'])
        self.gate_proj.weight = nn.Parameter(weights['gate_proj'])
        self.up_proj.weight = nn.Parameter(weights['up_proj'])
        self.down_proj.weight = nn.Parameter(weights['down_proj'])

    def forward(self, x, rope_cos, rope_sin, pos_ids):
        residual = x
        x = self.input_layernorm(x)
        q = self.q_norm(self.q_proj(x).view(-1, x.shape[1], self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(-1, x.shape[1], self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).view(-1, x.shape[1], self.num_kv_heads, self.head_dim)
        q = apply_rope_fn(q.transpose(1, 2), rope_cos, rope_sin, pos_ids)
        k = apply_rope_fn(k.transpose(1, 2), rope_cos, rope_sin, pos_ids)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        attn = attn.transpose(1, 2).reshape(-1, x.shape[1], self.num_heads * self.head_dim)
        x = residual + self.o_proj(attn)
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return x


def apply_rope_fn(x, cos, sin, pos):
    half = x.shape[-1] // 2
    cos_pos = cos[pos].unsqueeze(1)
    sin_pos = sin[pos].unsqueeze(1)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos_pos - x2 * sin_pos, x2 * cos_pos + x1 * sin_pos], dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-14B')
    parser.add_argument('--save-dir', type=str, default='quantized_v5_pure_int4')
    parser.add_argument('--nsamples', type=int, default=256)
    parser.add_argument('--block-size', type=int, default=32)
    parser.add_argument('--cal-seqlen', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--no-rotation', action='store_true',
                        help='Skip Hadamard rotation (test raw asymmetric GPTQ)')
    parser.add_argument('--symmetric', action='store_true',
                        help='Use symmetric INT4 instead of asymmetric')
    args = parser.parse_args()

    model_name = args.model
    block_size = args.block_size
    device = args.device

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(model_name)
    cfg = config.to_dict()
    num_layers = cfg['num_hidden_layers']
    hidden_size = cfg['hidden_size']
    intermediate_size = cfg['intermediate_size']
    num_heads = cfg['num_attention_heads']
    num_kv_heads = cfg.get('num_key_value_heads', num_heads)
    head_dim = cfg.get('head_dim', hidden_size // num_heads)
    rms_eps = cfg.get('rms_norm_eps', 1e-6)
    rope_theta = cfg.get('rope_theta', 1000000.0)

    cache = Path.home() / ".cache/huggingface/hub"
    repo_dir = cache / f"models--{model_name.replace('/', '--')}"
    model_path = sorted((repo_dir / "snapshots").iterdir())[-1]
    print(f"Model path: {model_path}")

    with open(model_path / "model.safetensors.index.json") as f:
        index = json.load(f)

    _shard_cache = {}
    def load_weight(key):
        sf = index["weight_map"][key]
        if sf not in _shard_cache:
            _shard_cache.clear(); gc.collect()
            _shard_cache[sf] = load_file(str(model_path / sf))
        return _shard_cache[sf][key]

    # ---- Calibration from C4 (diverse data, well-conditioned Hessian) ----
    torch.manual_seed(42)
    cal_tokens = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    from datasets import load_dataset
    # Use RedPajama for diverse calibration data (C4 replacement)
    # Falls back to WikiText-2 if unavailable
    try:
        dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train",
                              streaming=True, cache_dir="/tmp/hf_datasets")
        text_key = 'text'
    except Exception:
        print("  RedPajama unavailable, falling back to WikiText-2")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                              cache_dir="/tmp/hf_datasets")
        text_key = 'text'
    for sample in dataset:
        text = sample[text_key].strip()
        if len(text) < 200:
            continue
        ids = tokenizer.encode(text[:args.cal_seqlen * 6])
        if len(ids) >= args.cal_seqlen:
            cal_tokens.append(torch.tensor(ids[:args.cal_seqlen]).unsqueeze(0))
        if len(cal_tokens) >= args.nsamples:
            break
    del dataset, tokenizer; gc.collect()
    print(f"Calibration: {len(cal_tokens)} samples x {args.cal_seqlen} tokens "
          f"({len(cal_tokens) * args.cal_seqlen} total tokens)")

    embed_w = load_weight("model.embed_tokens.weight").half().to(device)
    cal_hidden = []
    with torch.no_grad():
        for tok in cal_tokens:
            cal_hidden.append(F.embedding(tok.to(device), embed_w).cpu())
    del embed_w; _shard_cache.clear(); gc.collect(); torch.cuda.empty_cache()

    freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(args.cal_seqlen, device=device).float()
    angles = torch.outer(t, freqs)
    rope_cos = angles.cos().half()
    rope_sin = angles.sin().half()
    cal_pos_ids = torch.arange(args.cal_seqlen, device=device).unsqueeze(0)

    had_mat = get_hadamard(block_size, device='cpu', dtype=torch.float32)
    use_rotation = not args.no_rotation
    use_asymmetric = not args.symmetric

    rot_str = "Hadamard" if use_rotation else "NoRotation"
    quant_str = "Asymmetric" if use_asymmetric else "Symmetric"
    print(f"\nQuantize v5: PURE {quant_str} INT4 + GPTQ (proper cal) + {rot_str} (block={block_size})")
    print(f"  ALL groups INT4 (no mixed precision)")
    print(f"  Calibration: {len(cal_tokens)}×{args.cal_seqlen} tokens")
    print(f"Output: {save_dir}\n")

    weight_configs = [
        ('qkv', ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'], 'q_proj'),
        ('o', ['self_attn.o_proj'], 'o_proj'),
        ('gate_up', ['mlp.gate_proj', 'mlp.up_proj'], 'gate_proj'),
        ('down', ['mlp.down_proj'], 'down_proj'),
    ]

    t_total = time.time()
    cnt_int4 = 0

    for li in range(num_layers):
        layer_file = save_dir / f"layer_{li:03d}.pt"

        if layer_file.exists():
            print(f"Layer {li+1}/{num_layers}: EXISTS, skip (forward pass only)")
            lw = {}
            for short, full_key in [
                ('q_proj', f"model.layers.{li}.self_attn.q_proj.weight"),
                ('k_proj', f"model.layers.{li}.self_attn.k_proj.weight"),
                ('v_proj', f"model.layers.{li}.self_attn.v_proj.weight"),
                ('o_proj', f"model.layers.{li}.self_attn.o_proj.weight"),
                ('gate_proj', f"model.layers.{li}.mlp.gate_proj.weight"),
                ('up_proj', f"model.layers.{li}.mlp.up_proj.weight"),
                ('down_proj', f"model.layers.{li}.mlp.down_proj.weight"),
                ('in_norm', f"model.layers.{li}.input_layernorm.weight"),
                ('post_norm', f"model.layers.{li}.post_attention_layernorm.weight"),
                ('q_norm', f"model.layers.{li}.self_attn.q_norm.weight"),
                ('k_norm', f"model.layers.{li}.self_attn.k_norm.weight"),
            ]:
                lw[short] = load_weight(full_key)
            layer = Qwen3Layer(hidden_size, num_heads, num_kv_heads, head_dim,
                              intermediate_size, rms_eps, lw).half().to(device)
            layer.eval()
            next_hidden = []
            with torch.no_grad():
                for h in cal_hidden:
                    next_hidden.append(layer(h.to(device), rope_cos, rope_sin, cal_pos_ids).cpu())
            cal_hidden = next_hidden
            del layer, lw; _shard_cache.clear(); gc.collect(); torch.cuda.empty_cache()
            continue

        t0 = time.time()
        print(f"\nLayer {li+1}/{num_layers}: ALL INT4")

        lw = {}
        for short, full_key in [
            ('q_proj', f"model.layers.{li}.self_attn.q_proj.weight"),
            ('k_proj', f"model.layers.{li}.self_attn.k_proj.weight"),
            ('v_proj', f"model.layers.{li}.self_attn.v_proj.weight"),
            ('o_proj', f"model.layers.{li}.self_attn.o_proj.weight"),
            ('gate_proj', f"model.layers.{li}.mlp.gate_proj.weight"),
            ('up_proj', f"model.layers.{li}.mlp.up_proj.weight"),
            ('down_proj', f"model.layers.{li}.mlp.down_proj.weight"),
            ('in_norm', f"model.layers.{li}.input_layernorm.weight"),
            ('post_norm', f"model.layers.{li}.post_attention_layernorm.weight"),
            ('q_norm', f"model.layers.{li}.self_attn.q_norm.weight"),
            ('k_norm', f"model.layers.{li}.self_attn.k_norm.weight"),
        ]:
            lw[short] = load_weight(full_key)

        layer = Qwen3Layer(hidden_size, num_heads, num_kv_heads, head_dim,
                          intermediate_size, rms_eps, lw).half().to(device)
        layer.eval()

        # Collect Hessians
        hessians = {}
        h_diags = {}
        n_tokens = {}

        def make_hook(name):
            def hook_fn(module, input, output):
                x = input[0].detach().float().reshape(-1, input[0].shape[-1]).clamp(-1e4, 1e4)
                K = x.shape[1]
                if name not in hessians:
                    hessians[name] = torch.zeros(K, K, device=device, dtype=torch.float32)
                    h_diags[name] = torch.zeros(K, device=device, dtype=torch.float32)
                    n_tokens[name] = 0
                hessians[name].add_(x.T @ x)
                h_diags[name].add_(x.pow(2).sum(dim=0))
                n_tokens[name] += x.shape[0]
            return hook_fn

        handles = []
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            handles.append(getattr(layer, name).register_forward_hook(make_hook(name)))

        next_hidden = []
        with torch.no_grad():
            for h in cal_hidden:
                next_hidden.append(layer(h.to(device), rope_cos, rope_sin, cal_pos_ids).cpu())

        for handle in handles:
            handle.remove()
        for name in hessians:
            if n_tokens[name] > 0:
                hessians[name] /= n_tokens[name]
                h_diags[name] /= n_tokens[name]

        cal_hidden = next_hidden
        del next_hidden, layer; torch.cuda.synchronize(); torch.cuda.empty_cache()

        # ---- Quantize each weight group ----
        layer_data = {}

        for save_key, proj_names, hessian_name in weight_configs:
            W = torch.cat([lw[pn.split('.')[-1]] for pn in proj_names], dim=0)
            N_out, K_in = W.shape

            # Optional Hadamard rotation
            K_pad = ((K_in + block_size - 1) // block_size) * block_size
            W_f = W.float()
            if K_pad > K_in:
                W_f = F.pad(W_f, (0, K_pad - K_in))

            if use_rotation:
                W_rot = apply_rotation_blockdiag(W_f.cpu(), had_mat, block_size)
            else:
                W_rot = W_f.cpu()

            # Transform Hessian (or use raw)
            H = hessians.get(hessian_name)
            if H is not None:
                if K_pad > K_in:
                    H_pad = torch.zeros(K_pad, K_pad, device=device, dtype=torch.float32)
                    H_pad[:K_in, :K_in] = H
                    H = H_pad
                if use_rotation:
                    H_rot = transform_hessian_rotation(H, had_mat.to(device), block_size)
                else:
                    H_rot = H
            else:
                H_rot = torch.eye(K_pad, device=device, dtype=torch.float32)

            # GPTQ quantization — ALL INT4
            W_packed, W_scales, W_zeros, K_final = gptq_quantize_int4_v2(
                W_rot.half(), H_rot, block_size=block_size, device=device,
                asymmetric=use_asymmetric)
            prec_tag = 'int4a' if use_asymmetric else 'int4'
            rot_tag = 'Had' if use_rotation else 'noR'
            print(f"  {save_key:8s}: {prec_tag}+{rot_tag}+GPTQ [{N_out}x{K_final}]")
            layer_data[f'{save_key}_packed'] = W_packed.cpu()
            layer_data[f'{save_key}_scales'] = W_scales.cpu()
            layer_data[f'{save_key}_zeros'] = W_zeros.cpu()
            layer_data[f'{save_key}_precision'] = prec_tag
            cnt_int4 += 1

            layer_data[f'{save_key}_N'] = N_out
            layer_data[f'{save_key}_K'] = K_final

            del W, W_f, W_rot, H_rot; torch.cuda.empty_cache()

        layer_data['in_norm'] = lw['in_norm'].cpu().clone()
        layer_data['post_norm'] = lw['post_norm'].cpu().clone()
        layer_data['q_norm'] = lw['q_norm'].cpu().clone()
        layer_data['k_norm'] = lw['k_norm'].cpu().clone()

        torch.save(layer_data, layer_file)
        del layer_data, hessians, h_diags, n_tokens, lw
        _shard_cache.clear(); gc.collect(); torch.cuda.empty_cache()

        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s")

    # lm_head — symmetric INT4, rotation matches main weights
    lm_file = save_dir / "lm_head.pt"
    if not lm_file.exists():
        rot_label = "Hadamard" if use_rotation else "none"
        print(f"\nQuantizing lm_head ({rot_label}+INT4 symmetric)...")
        lm_W = load_weight("lm_head.weight")
        N_lm, K_lm = lm_W.shape
        K_pad = ((K_lm + block_size - 1) // block_size) * block_size
        W_f = lm_W.float()
        if K_pad > K_lm:
            W_f = F.pad(W_f, (0, K_pad - K_lm))
        if use_rotation:
            W_rot = rotate_weight(W_f, block_size)
        else:
            W_rot = W_f
        W_blocks = W_rot.reshape(N_lm, K_pad // block_size, block_size)
        q, scale, _ = quantize_to_int4_symmetric(W_blocks)
        packed = pack_int4(q.reshape(N_lm, K_pad))
        torch.save({"packed": packed.cpu(), "scales": scale.cpu(),
                    "N": N_lm, "K": K_pad,
                    "rotation": "hadamard" if use_rotation else "none",
                    "quant_format": "int4_symmetric"}, lm_file)

    embed_file = save_dir / "embed.pt"
    if not embed_file.exists():
        torch.save(load_weight("model.embed_tokens.weight").clone(), embed_file)

    norm_file = save_dir / "final_norm.pt"
    if not norm_file.exists():
        torch.save(load_weight("model.norm.weight").clone(), norm_file)

    meta = {
        'model_name': model_name, 'num_layers': num_layers,
        'hidden_size': hidden_size, 'intermediate_size': intermediate_size,
        'num_heads': num_heads, 'num_kv_heads': num_kv_heads,
        'head_dim': head_dim, 'rms_eps': rms_eps, 'rope_theta': rope_theta,
        'method': f"v5_pure_{'asym' if use_asymmetric else 'sym'}_{'had' if use_rotation else 'norot'}_gptq",
        'block_size': block_size,
        'quant_format': f"pure_{'int4a' if use_asymmetric else 'int4'}_{'' if use_rotation else 'no'}rot",
        'use_rotation': use_rotation,
        'use_asymmetric': use_asymmetric,
        'n_int4_groups': cnt_int4, 'n_int8_groups': 0,
        'nsamples': args.nsamples, 'cal_seqlen': args.cal_seqlen,
    }
    torch.save(meta, save_dir / "meta.pt")

    dt_total = time.time() - t_total
    print(f"\nQuantization complete in {dt_total:.0f}s")
    print(f"Pure INT4: {cnt_int4} groups (0 INT8)")
    print(f"Saved to {save_dir}")


if __name__ == "__main__":
    main()
