"""
INT4 Asymmetric Engine v5 — HIP C++ decode + rocBLAS prefill.

Decode: All 40 layers in C++ (HIP GEMV + fused ops), zero Python overhead.
Prefill: On-the-fly INT4→FP16 dequant + rocBLAS GEMM (torch.matmul) + PyTorch SDPA.
Asymmetric INT4: val = (unsigned_nibble - zero_point) * scale
Scales+zeros stored as interleaved [N, K/32, 2] FP16 pairs.

Weights from: quantized_v5_pure_int4/
"""

import torch
import torch.nn.functional as F
import time
import sys
import gc
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'hip_int4'))


def detect_gpu_arch():
    """Auto-detect AMD GPU architecture (e.g. gfx1201, gfx1100, gfx942).
    Priority: PYTORCH_ROCM_ARCH env > PyTorch device query > rocminfo > default."""
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
    try:
        import subprocess
        out = subprocess.check_output(['rocminfo'], stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            if 'gfx' in line and 'Name:' in line:
                arch = line.split()[-1].strip()
                if arch.startswith('gfx'):
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
    print(f"int4_hip not found — building JIT for {GPU_ARCH} from hip_int4/int4_decode_step.hip ...")
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
from fused_ops import fused_rmsnorm
import hadamard_utils
from hadamard_utils import get_hadamard
from transformers import AutoTokenizer

device = "cuda"
quant_dir = Path(os.environ.get("QUANT_DIR", str(Path(__file__).parent / "quantized_v5_pure_int4")))

meta = torch.load(quant_dir / "meta.pt", weights_only=False)
model_name = meta['model_name']
num_layers = meta['num_layers']
hidden_size = meta['hidden_size']
intermediate_size = meta['intermediate_size']
num_heads = meta['num_heads']
num_kv_heads = meta['num_kv_heads']
head_dim = meta['head_dim']
rms_eps = meta['rms_eps']
rope_theta = meta['rope_theta']
block_size = meta.get('block_size', 32)

tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_SEQ = 2048


def interleave_scale_zero(scales, zeros_uint8):
    """Interleave FP16 scales and FP16 zeros into [N, K/32, 2] format.
    Input: scales [N, K/32] FP16, zeros [N, K/32] uint8
    Output: [N, K/16] FP16 (interleaved scale, zero pairs)"""
    zeros_fp16 = zeros_uint8.float().half()  # uint8 [0-15] → FP16
    # Stack along last dim: [N, K/32, 2] then view as [N, K/16]
    sz = torch.stack([scales, zeros_fp16], dim=-1).contiguous()
    return sz.view(sz.shape[0], -1)  # [N, 2*K/32] = [N, K/16]


def convert_symmetric_to_asymmetric(packed, scales, N, K):
    """Convert symmetric INT4 (signed two's complement) to asymmetric unsigned format.
    Symmetric: val = int4_decode(nibble) * scale  (nibble in two's complement)
    Asymmetric: val = (nibble_unsigned - 8) * scale  (zp=8, nibble in [0,15])
    Conversion: each nibble n → (n + 8) & 0xF, zero_point = 8 for all blocks."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    lo_new = (lo + 8) & 0x0F
    hi_new = (hi + 8) & 0x0F
    packed_new = (lo_new | (hi_new << 4)).to(torch.uint8)
    num_blocks = K // 32
    zeros = torch.full((N, num_blocks), 8, dtype=torch.uint8)
    sz = interleave_scale_zero(scales.half(), zeros)
    return packed_new, sz


# ---- Load model ----
print(f"Loading v5 INT4+Had+GPTQ from {quant_dir}...")

embed_w = torch.load(quant_dir / "embed.pt", weights_only=True).half().to(device)
final_norm_w = torch.load(quant_dir / "final_norm.pt", weights_only=True).half().to(device)
rope_cos, rope_sin = precompute_rope_freqs(head_dim, MAX_SEQ + 128, theta=rope_theta, device=device)

# Per-layer weight vectors for HIP C++ decode
hip_qkv_w, hip_qkv_s, hip_qkv_N = [], [], []
hip_o_w, hip_o_s, hip_o_N = [], [], []
hip_gu_w, hip_gu_s, hip_gu_N = [], [], []
hip_down_w, hip_down_s, hip_down_N = [], [], []
hip_in_norm_w, hip_post_norm_w = [], []
hip_q_norm_w, hip_k_norm_w = [], []

for li in range(num_layers):
    if (li + 1) % 10 == 0 or li == 0:
        print(f"  Layer {li+1}/{num_layers}...", flush=True)
    ld = torch.load(quant_dir / f"layer_{li:03d}.pt", weights_only=False)

    hip_qkv_w.append(ld['qkv_packed'].to(device))
    hip_qkv_s.append(interleave_scale_zero(ld['qkv_scales'].half(), ld['qkv_zeros']).to(device))
    hip_qkv_N.append(ld['qkv_N'])

    hip_o_w.append(ld['o_packed'].to(device))
    hip_o_s.append(interleave_scale_zero(ld['o_scales'].half(), ld['o_zeros']).to(device))
    hip_o_N.append(ld['o_N'])

    hip_gu_w.append(ld['gate_up_packed'].to(device))
    hip_gu_s.append(interleave_scale_zero(ld['gate_up_scales'].half(), ld['gate_up_zeros']).to(device))
    hip_gu_N.append(ld['gate_up_N'])

    hip_down_w.append(ld['down_packed'].to(device))
    hip_down_s.append(interleave_scale_zero(ld['down_scales'].half(), ld['down_zeros']).to(device))
    hip_down_N.append(ld['down_N'])

    hip_in_norm_w.append(ld['in_norm'].half().to(device))
    hip_post_norm_w.append(ld['post_norm'].half().to(device))
    hip_q_norm_w.append(ld['q_norm'].half().to(device))
    hip_k_norm_w.append(ld['k_norm'].half().to(device))

    del ld; gc.collect()

# lm_head — may be symmetric (no zeros) or asymmetric
lm_data = torch.load(quant_dir / "lm_head.pt", weights_only=False)
lm_N = lm_data['N']
lm_K = lm_data['K']
if 'zeros' in lm_data:
    # Asymmetric format
    lm_w = lm_data['packed'].to(device)
    lm_s = interleave_scale_zero(lm_data['scales'].half(), lm_data['zeros']).to(device)
else:
    # Symmetric format — convert to asymmetric (unsigned + zp=8)
    packed_new, sz = convert_symmetric_to_asymmetric(
        lm_data['packed'], lm_data['scales'], lm_N, lm_K)
    lm_w = packed_new.to(device)
    lm_s = sz.to(device)
del lm_data

# Precompute Hadamard matrix [32, 32] FP16 on GPU
had_mat = get_hadamard(block_size, device=device, dtype=torch.float32).half().contiguous()

gpu_mb = torch.cuda.memory_allocated() / 1024**2
print(f"Model loaded: {gpu_mb:.0f} MB VRAM\n")


# ---- INT4 dequant + fast prefill ----

def dequant_int4(packed, scales_zeros, N, K):
    """Dequant [N, K/2] uint8 + [N, K/32, 2] interleaved FP16 → [N, K] FP16.
    Single HIP kernel launch — on-the-fly, temporary."""
    return int4_hip.dequant_int4(packed, scales_zeros, N, K)


def head_rmsnorm(x, weight, eps):
    """Per-head RMSNorm: x [..., D], weight [D]."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x.float() * torch.rsqrt(variance + eps)
    return (x * weight.float()).half()


def fast_hadamard(x):
    """Fast block-diagonal Hadamard rotation via flat matmul."""
    shape = x.shape
    return (x.view(-1, 32) @ had_mat).view(shape)


def fast_rmsnorm(x, w):
    """RMSNorm: x [..., D], w [D]."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + rms_eps) * w.float()).half()


def fast_prefill(input_ids, kv):
    """Fast prefill using rocBLAS GEMM (torch.matmul) with on-the-fly INT4 dequant.
    Fills KV cache for subsequent decode steps. Returns logits for last token."""
    M = input_ids.shape[1]  # sequence length
    pos_ids = torch.arange(M, device=device).unsqueeze(0)  # [1, M]

    x = F.embedding(input_ids.squeeze(0), embed_w)  # [M, D]

    for li in range(num_layers):
        # Input RMSNorm + Hadamard rotation (fast path)
        normed = fast_hadamard(fast_rmsnorm(x, hip_in_norm_w[li]))

        # QKV projection (CUDA dequant + rocBLAS GEMM)
        w = dequant_int4(hip_qkv_w[li], hip_qkv_s[li], hip_qkv_N[li], hidden_size)
        qkv = normed @ w.T; del w

        # Split Q, K, V + per-head RMSNorm + RoPE
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        q = head_rmsnorm(qkv[:, :q_dim].view(M, num_heads, head_dim), hip_q_norm_w[li], rms_eps)
        k = head_rmsnorm(qkv[:, q_dim:q_dim+kv_dim].view(M, num_kv_heads, head_dim), hip_k_norm_w[li], rms_eps)
        v = qkv[:, q_dim+kv_dim:].view(M, num_kv_heads, head_dim)
        q = apply_rope(q.unsqueeze(0).transpose(1,2), rope_cos, rope_sin, pos_ids)
        k = apply_rope(k.unsqueeze(0).transpose(1,2), rope_cos, rope_sin, pos_ids)
        v = v.unsqueeze(0).transpose(1,2)

        # Write K, V to KV cache (convert FP16 → FP8 E4M3)
        kv.k_caches[li][:, :, :M, :] = int4_hip.fp16_to_fp8(k.squeeze(0).contiguous()).view(k.squeeze(0).shape)
        kv.v_caches[li][:, :, :M, :] = int4_hip.fp16_to_fp8(v.squeeze(0).contiguous()).view(v.squeeze(0).shape)

        # SDPA + Hadamard + O projection
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        attn_out = attn.squeeze(0).transpose(0, 1).reshape(M, q_dim)
        w = dequant_int4(hip_o_w[li], hip_o_s[li], hip_o_N[li], hidden_size)
        o_out = fast_hadamard(attn_out) @ w.T; del w

        # Residual + post norm + Hadamard + GateUp
        x = x + o_out
        normed = fast_hadamard(fast_rmsnorm(x, hip_post_norm_w[li]))
        w = dequant_int4(hip_gu_w[li], hip_gu_s[li], hip_gu_N[li], hidden_size)
        gu = normed @ w.T; del w

        # SiLU(gate) * up + Hadamard + Down
        mlp_out = F.silu(gu[:, :intermediate_size]) * gu[:, intermediate_size:]
        w = dequant_int4(hip_down_w[li], hip_down_s[li], hip_down_N[li], intermediate_size)
        x = x + fast_hadamard(mlp_out) @ w.T; del w

    # Final norm + LM head (only last token for efficiency)
    kv.current_len = M
    normed = fast_hadamard(fast_rmsnorm(x[-1:, :], final_norm_w))
    w = dequant_int4(lm_w, lm_s, lm_N, lm_K)
    logits = normed @ w.T; del w
    return logits


def decode_step_logits(hidden, pos_idx, kv):
    """Single decode step via HIP C++ — all 40 layers + final_norm + lm_head.
    Returns logits [1, 1, vocab]."""
    return int4_hip.decode_step_logits(
        hidden,
        hip_qkv_w, hip_qkv_s, hip_qkv_N,
        hip_o_w, hip_o_s, hip_o_N,
        hip_gu_w, hip_gu_s, hip_gu_N,
        hip_down_w, hip_down_s, hip_down_N,
        hip_in_norm_w, hip_post_norm_w,
        hip_q_norm_w, hip_k_norm_w,
        rope_cos, rope_sin,
        pos_idx,
        num_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size,
        rms_eps,
        [kv.k_caches[li] for li in range(num_layers)],
        [kv.v_caches[li] for li in range(num_layers)],
        final_norm_w,
        lm_w, lm_s, lm_N, lm_K,
        had_mat
    )


def prefill_hip(input_ids):
    """HIP C++ prefill — all layers, returns logits [M, vocab]."""
    return int4_hip.prefill_logits(
        input_ids.squeeze(0),  # [M]
        embed_w,               # [vocab, D]
        final_norm_w,          # [D]
        lm_w, lm_s, lm_N, lm_K,
        hip_qkv_w, hip_qkv_s, hip_qkv_N,
        hip_o_w, hip_o_s, hip_o_N,
        hip_gu_w, hip_gu_s, hip_gu_N,
        hip_down_w, hip_down_s, hip_down_N,
        hip_in_norm_w, hip_post_norm_w,
        hip_q_norm_w, hip_k_norm_w,
        rope_cos, rope_sin,
        num_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size,
        rms_eps,
        had_mat
    )


def generate(prompt, max_tokens=100, temperature=0.7, top_p=0.9):
    """Generate text from prompt.
    Prefill via rocBLAS GEMM (fast_prefill), decode via HIP C++ GEMV."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    S = input_ids.shape[1]
    kv = KVCache(num_layers, 1, S + max_tokens + 16, num_kv_heads, head_dim, device, dtype=torch.uint8)

    # Fast prefill: rocBLAS GEMM + on-the-fly INT4 dequant
    t0 = time.time()
    with torch.no_grad():
        logits = fast_prefill(input_ids, kv)
    prefill_time = time.time() - t0
    prefill_tps = S / prefill_time

    # Sample first token from last prefill logits
    logits_last = logits[0, :].float()
    if temperature > 0:
        logits_last = logits_last / temperature
        probs = F.softmax(logits_last, dim=-1)
        next_id = torch.multinomial(probs.unsqueeze(0), 1)
    else:
        next_id = logits_last.argmax(-1, keepdim=True).unsqueeze(0)

    generated = [next_id.item()]
    pos = S

    # Decode
    t_decode_start = time.time()
    with torch.no_grad():
        for _ in range(max_tokens - 1):
            hidden = F.embedding(next_id.view(1), embed_w).view(1, 1, hidden_size)
            logits = decode_step_logits(hidden, pos, kv)
            logits_last = logits[0, 0, :].float()

            if temperature > 0:
                logits_last = logits_last / temperature
                probs = F.softmax(logits_last, dim=-1)
                next_id = torch.multinomial(probs.unsqueeze(0), 1)
            else:
                next_id = logits_last.argmax(-1, keepdim=True).unsqueeze(0)

            tok = next_id.item()
            generated.append(tok)
            pos += 1

            if tok == tokenizer.eos_token_id:
                break

    decode_time = time.time() - t_decode_start
    decode_tokens = len(generated)
    decode_tps = decode_tokens / decode_time if decode_time > 0 else 0

    result = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\n--- Stats ---")
    print(f"Prefill: {S} tokens in {prefill_time*1000:.0f}ms ({prefill_tps:.0f} t/s)")
    print(f"Decode:  {decode_tokens} tokens in {decode_time*1000:.0f}ms ({decode_tps:.1f} t/s)")
    print(f"VRAM:    {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    return prompt + result


def generate_streaming(prompt, max_tokens=512, temperature=0.7, top_p=0.9):
    """Generate text with token-by-token streaming to stdout."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    S = input_ids.shape[1]
    kv = KVCache(num_layers, 1, S + max_tokens + 16, num_kv_heads, head_dim, device, dtype=torch.uint8)

    # Prefill
    t0 = time.time()
    with torch.no_grad():
        logits = fast_prefill(input_ids, kv)
    prefill_time = time.time() - t0

    # Sample first token
    logits_last = logits[0, :].float()
    if temperature > 0:
        logits_last = logits_last / temperature
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits_last, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = -float('inf')
            logits_last = torch.zeros_like(logits_last).scatter_(0, sorted_idx, sorted_logits)
        probs = F.softmax(logits_last, dim=-1)
        next_id = torch.multinomial(probs.unsqueeze(0), 1)
    else:
        next_id = logits_last.argmax(-1, keepdim=True).unsqueeze(0)

    generated = [next_id.item()]
    pos = S
    in_think = False
    think_done = False
    prev_text = ""

    # Decode loop with streaming
    t_decode_start = time.time()
    with torch.no_grad():
        for _ in range(max_tokens - 1):
            hidden = F.embedding(next_id.view(1), embed_w).view(1, 1, hidden_size)
            logits = decode_step_logits(hidden, pos, kv)
            logits_last = logits[0, 0, :].float()

            if temperature > 0:
                logits_last = logits_last / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits_last, descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[mask] = -float('inf')
                    logits_last = torch.zeros_like(logits_last).scatter_(0, sorted_idx, sorted_logits)
                probs = F.softmax(logits_last, dim=-1)
                next_id = torch.multinomial(probs.unsqueeze(0), 1)
            else:
                next_id = logits_last.argmax(-1, keepdim=True).unsqueeze(0)

            tok = next_id.item()
            generated.append(tok)
            pos += 1

            # Check for EOS and special tokens
            if tok == tokenizer.eos_token_id:
                break

            # Decode full text to detect <think> blocks and <|im_end|>
            raw_text = tokenizer.decode(generated, skip_special_tokens=False)

            # Stop on <|im_end|>
            if '<|im_end|>' in raw_text:
                break

            # Handle <think>...</think> — skip thinking, show only answer
            if not think_done:
                if '<think>' in raw_text and '</think>' not in raw_text:
                    in_think = True
                    # Show thinking indicator once
                    if len(generated) < 5:
                        sys.stdout.write("\033[2m(thinking...)\033[0m ")
                        sys.stdout.flush()
                    continue
                if in_think and '</think>' in raw_text:
                    in_think = False
                    think_done = True
                    # Reset prev_text to start printing after </think>
                    idx = raw_text.index('</think>') + len('</think>')
                    clean = raw_text[idx:].replace('<|im_end|>', '').strip()
                    if clean:
                        sys.stdout.write(clean)
                        sys.stdout.flush()
                    prev_text = clean
                    continue
                if in_think:
                    continue

            # Stream visible text
            clean = tokenizer.decode(generated, skip_special_tokens=True)
            # After think block, only show text after the block
            if think_done:
                raw = tokenizer.decode(generated, skip_special_tokens=False)
                idx = raw.index('</think>') + len('</think>') if '</think>' in raw else 0
                clean = raw[idx:].replace('<|im_end|>', '').replace('<|im_start|>', '')
            new_text = clean[len(prev_text):]
            if new_text:
                sys.stdout.write(new_text)
                sys.stdout.flush()
            prev_text = clean

    decode_time = time.time() - t_decode_start
    decode_tokens = len(generated)
    decode_tps = decode_tokens / decode_time if decode_time > 0 else 0
    print(f"\n\n--- Stats ---")
    print(f"Prefill: {S} tokens in {prefill_time*1000:.0f}ms ({S/prefill_time:.0f} t/s)")
    print(f"Decode:  {decode_tokens} tokens in {decode_time*1000:.0f}ms ({decode_tps:.1f} t/s)")
    print(f"VRAM:    {torch.cuda.memory_allocated()/1024**2:.0f} MB")


def interactive_chat(max_tokens=512, temperature=0.7, top_p=0.9):
    """Interactive chat loop in terminal."""
    print("=" * 60)
    print("  INT4 Engine v5 — Interactive Chat")
    print(f"  Model: Qwen3-14B (INT4+Hadamard+GPTQ)")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    print(f"  Settings: temp={temperature}, top_p={top_p}, max={max_tokens}")
    print("  Type 'quit' or 'exit' to end. 'clear' to reset.")
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
            print("\033[2J\033[H", end="")  # clear terminal
            continue

        # Format as chat prompt (Qwen3 ChatML format)
        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        print(f"\033[1;34mAssistant:\033[0m ", end="")
        generate_streaming(prompt, max_tokens=max_tokens,
                          temperature=temperature, top_p=top_p)


def benchmark_decode(n_tokens=50, ctx=128):
    """Benchmark pure decode speed."""
    kv = KVCache(num_layers, 1, ctx + n_tokens + 16, num_kv_heads, head_dim, device, dtype=torch.uint8)
    h = torch.randn(1, 1, hidden_size, dtype=torch.float16, device=device)

    # Warmup
    for i in range(3):
        with torch.no_grad():
            _ = decode_step_logits(h, ctx + i, kv)
    torch.cuda.synchronize()

    # Measure
    times = []
    for i in range(n_tokens):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = decode_step_logits(h, ctx + 3 + i, kv)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    avg = sum(times) / len(times)
    print(f"Decode ctx={ctx}: {avg*1000:.2f} ms/tok, {1/avg:.1f} t/s")
    return 1/avg


def benchmark_decode_graph(n_tokens=50, ctx=128):
    """Benchmark decode with HIP Graph — eliminates kernel launch overhead."""
    kv = KVCache(num_layers, 1, ctx + n_tokens + 16, num_kv_heads, head_dim, device, dtype=torch.uint8)
    h = torch.randn(1, 1, hidden_size, dtype=torch.float16, device=device)

    # Warmup (non-graph, fills KV cache with some data)
    for i in range(3):
        with torch.no_grad():
            _ = decode_step_logits(h, ctx + i, kv)
    torch.cuda.synchronize()

    # Graph benchmark
    torch.cuda.synchronize()
    t0 = time.time()
    int4_hip.decode_bench_graph(
        h, embed_w, n_tokens,
        hip_qkv_w, hip_qkv_s, hip_qkv_N,
        hip_o_w, hip_o_s, hip_o_N,
        hip_gu_w, hip_gu_s, hip_gu_N,
        hip_down_w, hip_down_s, hip_down_N,
        hip_in_norm_w, hip_post_norm_w,
        hip_q_norm_w, hip_k_norm_w,
        rope_cos, rope_sin,
        ctx,
        num_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size,
        rms_eps,
        [kv.k_caches[li] for li in range(num_layers)],
        [kv.v_caches[li] for li in range(num_layers)],
        final_norm_w,
        lm_w, lm_s, lm_N, lm_K,
        had_mat
    )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tps = n_tokens / elapsed
    ms_per_tok = elapsed * 1000 / n_tokens
    print(f"Graph decode ctx={ctx}: {ms_per_tok:.2f} ms/tok, {tps:.1f} t/s ({n_tokens} tokens)")
    return tps


def benchmark_decode_cpp(n_tokens=50, ctx=128):
    """Benchmark decode using C++ tight loop (no graph, baseline)."""
    kv = KVCache(num_layers, 1, ctx + n_tokens + 16, num_kv_heads, head_dim, device, dtype=torch.uint8)
    h = torch.randn(1, 1, hidden_size, dtype=torch.float16, device=device)

    # Warmup
    for i in range(3):
        with torch.no_grad():
            _ = decode_step_logits(h, ctx + i, kv)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.time()
    int4_hip.decode_bench(
        h, embed_w, n_tokens,
        hip_qkv_w, hip_qkv_s, hip_qkv_N,
        hip_o_w, hip_o_s, hip_o_N,
        hip_gu_w, hip_gu_s, hip_gu_N,
        hip_down_w, hip_down_s, hip_down_N,
        hip_in_norm_w, hip_post_norm_w,
        hip_q_norm_w, hip_k_norm_w,
        rope_cos, rope_sin,
        ctx,
        num_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size,
        rms_eps,
        [kv.k_caches[li] for li in range(num_layers)],
        [kv.v_caches[li] for li in range(num_layers)],
        final_norm_w,
        lm_w, lm_s, lm_N, lm_K,
        had_mat
    )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tps = n_tokens / elapsed
    ms_per_tok = elapsed * 1000 / n_tokens
    print(f"C++ bench ctx={ctx}: {ms_per_tok:.2f} ms/tok, {tps:.1f} t/s ({n_tokens} tokens)")
    return tps


def profile_decode(ctx=128):
    """Profile 1 token decode — returns [total_ms, gemv_ms, attn_ms, norm_ms, rest_ms]."""
    kv = KVCache(num_layers, 1, ctx + 16, num_kv_heads, head_dim, device, dtype=torch.uint8)
    h = torch.randn(1, 1, hidden_size, dtype=torch.float16, device=device)
    # Warmup
    for i in range(3):
        with torch.no_grad():
            _ = decode_step_logits(h, ctx + i, kv)
    torch.cuda.synchronize()
    return int4_hip.decode_profile(
        h, embed_w,
        hip_qkv_w, hip_qkv_s, hip_qkv_N,
        hip_o_w, hip_o_s, hip_o_N,
        hip_gu_w, hip_gu_s, hip_gu_N,
        hip_down_w, hip_down_s, hip_down_N,
        hip_in_norm_w, hip_post_norm_w,
        hip_q_norm_w, hip_k_norm_w,
        rope_cos, rope_sin,
        ctx - 1,
        num_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size,
        rms_eps,
        [kv.k_caches[li] for li in range(num_layers)],
        [kv.v_caches[li] for li in range(num_layers)],
        final_norm_w,
        lm_w, lm_s, lm_N, lm_K,
        had_mat
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', action='store_true', help='Run decode benchmark')
    parser.add_argument('--bench-graph', action='store_true', help='Run HIP Graph decode benchmark')
    parser.add_argument('--bench-compare', action='store_true', help='Compare graph vs non-graph decode')
    parser.add_argument('--chat', action='store_true', help='Interactive chat mode')
    parser.add_argument('--prompt', type=str, default=None, help='Generate from prompt')
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--ctx', type=int, default=128, help='Context length for benchmark')
    args = parser.parse_args()

    if args.chat:
        interactive_chat(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
    elif args.bench_compare:
        for ctx in [128, 512, 1024]:
            print(f"\n--- ctx={ctx} ---")
            tps_cpp = benchmark_decode_cpp(50, ctx)
            tps_graph = benchmark_decode_graph(50, ctx)
            print(f"  Speedup: {tps_graph/tps_cpp:.2f}x")
    elif args.bench_graph:
        for ctx in [128, 512, 1024, 2048]:
            benchmark_decode_graph(50, ctx)
    elif args.bench:
        for ctx in [128, 512, 1024, 2048]:
            benchmark_decode(50, ctx)
    elif args.prompt:
        result = generate(args.prompt, max_tokens=args.max_tokens)
        print(f"\n{result}")
    else:
        # Default: quick bench + sample generate
        benchmark_decode(50, 128)
        print()
        result = generate("The meaning of life is", max_tokens=50)
        print(f"\n{result}")
