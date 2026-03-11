"""
Per-layer quantization sensitivity measurement for Qwen3-14B.

Method: For each layer's weight matrices, measure the output MSE when quantizing
to INT4 vs keeping FP16. High MSE = high sensitivity = should keep higher precision.

Two metrics per weight:
  1. Hessian trace (tr(H)) — measures how much activations amplify weight perturbations
  2. Quant MSE — direct ||W_fp16 - W_int4||^2 weighted by activation magnitude

Output: sensitivity_map.json with per-layer, per-weight scores + recommended precision.

Usage: python measure_sensitivity.py
"""
import torch
import torch.nn.functional as F
import math, time, sys, gc, json, os
from pathlib import Path

os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1201'
os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_datasets'

sys.path.insert(0, str(Path(__file__).parent))
from int4_quant import quantize_to_int4_symmetric, pack_int4, unpack_int4
from hadamard_utils import get_hadamard
from safetensors.torch import load_file

device = "cuda"
model_name = "Qwen/Qwen3-14B"

# ---- Load model config ----
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
block_size = 32

print(f"Model: {model_name}")
print(f"Layers: {num_layers}, hidden: {hidden_size}, intermediate: {intermediate_size}")
print(f"Heads: {num_heads}, KV heads: {num_kv_heads}, head_dim: {head_dim}")

# ---- Find safetensors ----
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
        _shard_cache.clear()
        gc.collect()
        _shard_cache[sf] = load_file(str(model_path / sf))
    return _shard_cache[sf][key]

# ---- Calibration data ----
NSAMPLES = 64
CAL_SEQLEN = 256
torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained(model_name)
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                      cache_dir="/tmp/hf_datasets")
cal_tokens = []
for sample in dataset:
    text = sample['text'].strip()
    if len(text) < 500:
        continue
    ids = tokenizer.encode(text[:CAL_SEQLEN * 6])
    if len(ids) >= CAL_SEQLEN:
        cal_tokens.append(torch.tensor(ids[:CAL_SEQLEN]).unsqueeze(0))
    if len(cal_tokens) >= NSAMPLES:
        break
del dataset, tokenizer
gc.collect()
print(f"Calibration: {len(cal_tokens)} samples x {CAL_SEQLEN} tokens")

# ---- Embed ----
embed_w = load_weight("model.embed_tokens.weight").half().to(device)
cal_hidden = []
with torch.no_grad():
    for tok in cal_tokens:
        h = F.embedding(tok.to(device), embed_w).cpu()
        cal_hidden.append(h)
del embed_w
_shard_cache.clear(); gc.collect(); torch.cuda.empty_cache()

# ---- RoPE ----
freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
t = torch.arange(CAL_SEQLEN, device=device).float()
angles = torch.outer(t, freqs)
rope_cos = angles.cos().half()
rope_sin = angles.sin().half()
cal_pos_ids = torch.arange(CAL_SEQLEN, device=device).unsqueeze(0)

# ---- Hadamard ----
had_mat = get_hadamard(block_size, device='cpu', dtype=torch.float32)

def apply_had(W_f):
    """Apply block-diagonal Hadamard rotation to W [N, K]."""
    N, K = W_f.shape
    return (W_f.reshape(N, K // block_size, block_size) @ had_mat.unsqueeze(0)).reshape(N, K)

def quant_error_weighted(W_fp16, H_diag):
    """Compute Hessian-weighted quantization error for INT4.

    Error = sum_k (w_k - w_hat_k)^2 * H_kk
    where H_kk is the diagonal of the Hessian (activation second moment).
    """
    W_f = W_fp16.float()
    N, K = W_f.shape
    K_pad = ((K + block_size - 1) // block_size) * block_size
    if K_pad > K:
        W_f = F.pad(W_f, (0, K_pad - K))

    # Apply Hadamard rotation (same as our quantization pipeline)
    W_rot = apply_had(W_f.cpu()).to(device)

    # Quantize
    W_blocks = W_rot.reshape(N, K_pad // block_size, block_size)
    q, scale, W_recon = quantize_to_int4_symmetric(W_blocks)
    W_recon = W_recon.reshape(N, K_pad)

    # Error per column
    err = (W_rot - W_recon).pow(2)  # [N, K_pad]

    # Pad H_diag if needed
    if H_diag.shape[0] < K_pad:
        H_pad = torch.zeros(K_pad, device=device)
        H_pad[:H_diag.shape[0]] = H_diag
        H_diag = H_pad

    # Hessian-weighted error: sum over output dims, weight by H_kk
    weighted_err = (err.sum(dim=0) * H_diag[:K_pad]).sum().item()

    # Raw MSE (unweighted)
    raw_mse = err.mean().item()

    # Relative error: ||W - W_hat|| / ||W||
    rel_err = err.sum().sqrt().item() / (W_rot.pow(2).sum().sqrt().item() + 1e-10)

    return weighted_err, raw_mse, rel_err


def apply_rope_fn(x, cos, sin, pos):
    half = x.shape[-1] // 2
    cos_pos = cos[pos].unsqueeze(1)
    sin_pos = sin[pos].unsqueeze(1)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos_pos - x2 * sin_pos, x2 * cos_pos + x1 * sin_pos], dim=-1)


# ---- Measure per-layer ----
print(f"\n{'='*70}")
print(f"Per-layer Sensitivity Measurement")
print(f"{'='*70}\n")

weight_groups = {
    'qkv': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
    'o': ['self_attn.o_proj'],
    'gate_up': ['mlp.gate_proj', 'mlp.up_proj'],
    'down': ['mlp.down_proj'],
}

all_sensitivity = {}

for li in range(num_layers):
    t0 = time.time()

    # Load FP16 weights
    lw = {}
    weight_keys = [
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
    ]
    for short, full in weight_keys:
        lw[short] = load_weight(full)

    # Build layer for forward pass + Hessian collection
    from quantize_int4_gptq import Qwen3Layer
    layer = Qwen3Layer(hidden_size, num_heads, num_kv_heads, head_dim,
                       intermediate_size, rms_eps, lw).half().to(device)
    layer.eval()

    # Collect activation statistics (Hessian diagonal = E[x^2] per input dim)
    act_stats = {}  # name -> sum of x^2 per input dim
    act_counts = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            x2d = x.reshape(-1, x.shape[-1])
            if name not in act_stats:
                act_stats[name] = torch.zeros(x2d.shape[1], device=device)
                act_counts[name] = 0
            act_stats[name] += x2d.pow(2).sum(dim=0)
            act_counts[name] += x2d.shape[0]
        return hook_fn

    handles = []
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        handles.append(getattr(layer, name).register_forward_hook(make_hook(name)))

    # Forward pass through all calibration data
    next_hidden = []
    with torch.no_grad():
        for h in cal_hidden:
            out = layer(h.to(device), rope_cos, rope_sin, cal_pos_ids)
            next_hidden.append(out.cpu())

    for handle in handles:
        handle.remove()

    # Compute H_diag = E[x_k^2]
    h_diags = {}
    for name in act_stats:
        h_diags[name] = act_stats[name] / act_counts[name]

    # Measure quantization error per weight group
    layer_sens = {}
    for group_name, proj_names in weight_groups.items():
        # Concatenate weights
        weights = []
        for pn in proj_names:
            short = pn.split('.')[-1]
            weights.append(lw[short].to(device))
        W = torch.cat(weights, dim=0)

        # Get H_diag for this group (use first proj's Hessian)
        hessian_key = proj_names[0].split('.')[-1]
        H_diag = h_diags.get(hessian_key, torch.ones(W.shape[1], device=device))

        weighted_err, raw_mse, rel_err = quant_error_weighted(W, H_diag)

        # Hessian trace (sum of H_kk) — measures activation magnitude
        h_trace = H_diag.sum().item()

        layer_sens[group_name] = {
            'weighted_err': weighted_err,
            'raw_mse': raw_mse,
            'rel_err': rel_err,
            'h_trace': h_trace,
            'shape': list(W.shape),
        }

        del W

    all_sensitivity[li] = layer_sens
    cal_hidden = next_hidden

    dt = time.time() - t0
    # Print summary for this layer
    qkv_s = layer_sens['qkv']['weighted_err']
    o_s = layer_sens['o']['weighted_err']
    gu_s = layer_sens['gate_up']['weighted_err']
    d_s = layer_sens['down']['weighted_err']
    total_s = qkv_s + o_s + gu_s + d_s
    print(f"Layer {li:2d}: total={total_s:10.2f}  qkv={qkv_s:8.2f}  o={o_s:8.2f}  gate_up={gu_s:8.2f}  down={d_s:8.2f}  ({dt:.1f}s)")

    del layer, lw, act_stats, act_counts, h_diags, next_hidden
    _shard_cache.clear(); gc.collect(); torch.cuda.empty_cache()

# ---- Analysis: rank layers and recommend precision ----
print(f"\n{'='*70}")
print(f"Sensitivity Ranking & Precision Recommendations")
print(f"{'='*70}\n")

# Collect all scores
all_scores = []
for li in range(num_layers):
    for group_name in ['qkv', 'o', 'gate_up', 'down']:
        score = all_sensitivity[li][group_name]['weighted_err']
        all_scores.append((li, group_name, score))

# Sort by sensitivity (highest = most sensitive)
all_scores.sort(key=lambda x: x[2], reverse=True)

# Determine threshold for INT8: top 20% most sensitive → INT8
total_weights = len(all_scores)
n_int8 = max(1, total_weights // 5)  # ~20% → INT8

# Also compute per-layer total sensitivity for layer-level ranking
layer_totals = []
for li in range(num_layers):
    total = sum(all_sensitivity[li][g]['weighted_err'] for g in ['qkv', 'o', 'gate_up', 'down'])
    layer_totals.append((li, total))
layer_totals.sort(key=lambda x: x[1], reverse=True)

print("Top 10 most sensitive weights:")
for i, (li, group, score) in enumerate(all_scores[:10]):
    print(f"  {i+1:2d}. Layer {li:2d} {group:8s}: {score:.2f}")

print(f"\nTop 10 most sensitive layers:")
for i, (li, total) in enumerate(layer_totals[:10]):
    print(f"  {i+1:2d}. Layer {li:2d}: {total:.2f}")

# Build precision map
precision_map = {}
int8_set = set()
for i, (li, group, score) in enumerate(all_scores[:n_int8]):
    int8_set.add((li, group))

for li in range(num_layers):
    precision_map[li] = {}
    for group in ['qkv', 'o', 'gate_up', 'down']:
        if (li, group) in int8_set:
            precision_map[li][group] = 'int8'
        else:
            precision_map[li][group] = 'int4'

# Summary
n_int4_total = sum(1 for li in precision_map for g in precision_map[li] if precision_map[li][g] == 'int4')
n_int8_total = sum(1 for li in precision_map for g in precision_map[li] if precision_map[li][g] == 'int8')

print(f"\nPrecision assignment:")
print(f"  INT4: {n_int4_total} weight groups ({n_int4_total / total_weights * 100:.0f}%)")
print(f"  INT8: {n_int8_total} weight groups ({n_int8_total / total_weights * 100:.0f}%)")

# Estimate VRAM impact
# Each weight group: INT4 = K/2 bytes packed + K/bs * 2 bytes scales
# INT8 = K bytes + K/bs * 2 bytes scales (or no block scales for INT8)
# Qwen3-14B weight sizes:
#   qkv: [7680, 5120], o: [5120, 5120], gate_up: [34816, 5120], down: [5120, 17408]
print(f"\nINT8 layers:")
for li in sorted(precision_map.keys()):
    int8_groups = [g for g in precision_map[li] if precision_map[li][g] == 'int8']
    if int8_groups:
        print(f"  Layer {li:2d}: {', '.join(int8_groups)}")

# Save results
results = {
    'sensitivity': {str(k): v for k, v in all_sensitivity.items()},
    'precision_map': {str(k): v for k, v in precision_map.items()},
    'ranking': [(li, group, score) for li, group, score in all_scores],
    'layer_ranking': [(li, total) for li, total in layer_totals],
}

out_path = str(Path(__file__).parent / 'sensitivity_map.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
