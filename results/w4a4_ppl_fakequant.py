#!/usr/bin/env python3
"""
W4A4 fake-quant PPL evaluation on real Bielik-11B-v2.6-Instruct.

Simulates (numerically, in PyTorch) the W4A4 prefill quantization scheme that the
HIP kernels (ggml-johnv8/prefill/gemm_w4a4.hip) implement, and measures END-TO-END
perplexity loss vs FP16 on real corpora (English wikitext-2 + Polish wikipedia).

Variants measured on the 5 prefill GEMMs (q_proj,k_proj,v_proj,o_proj,gate,up,down):
  1. FP16 baseline                                   (reference PPL)
  2. Q4_K weight-only + int8 activation              (USER'S CURRENT BASELINE)
  3. W4A4: Q4_K weight + int4 act + Hadamard         (NEW prefill scheme)
  4. W4A4 no Hadamard (Q4_K weight + int4 act)       (what Hadamard buys)
  5. Q4_0 weight + int4 act + Hadamard               (symmetric speed-winner)

Quant scheme exactly mirrors the HIP kernel:
  - per-32-block, symmetric int4: scale = amax/7, round-nearest, clamp [-8,7]   (Q4_0 / activation)
  - per-32-block, asymmetric int4 (Q4_K-like): scale=(max-min)/15, zp, q in [0,15], clamp
  - int8 activation: per-32-block, scale=amax/127
  - Hadamard: per-32-block normalized Walsh-Hadamard H_32 on the K/feature dim.
    (X·H)(Hᵀ·W) == X·W in fp; H is orthogonal so it only redistributes outliers.

Module is applied as a forward-hook monkeypatch on nn.Linear forward of the 7
target projections in every decoder layer. Embedding, final norm, lm_head, RMSNorm,
attention softmax, RoPE all run in fp16 (only the GEMM operands are fake-quantized,
exactly like the prefill kernel which only touches those weight GEMMs).
"""
import os, sys, json, time, math, argparse
import torch
import torch.nn as nn

torch.set_grad_enabled(False)

BLK = 32

# ---------------------------------------------------------------------------
# Hadamard H_32 (normalized, orthogonal): H @ H.T = I
# ---------------------------------------------------------------------------
def build_hadamard(n, device, dtype=torch.float32):
    assert (n & (n - 1)) == 0, "n must be power of 2"
    H = torch.ones((1, 1), dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H = H / math.sqrt(n)             # normalized -> orthogonal
    return H.to(device=device, dtype=dtype)

# ---------------------------------------------------------------------------
# per-32-block quantizers (operate on last dim, which must be multiple of 32)
# ---------------------------------------------------------------------------
def _blocks(x):
    *lead, K = x.shape
    assert K % BLK == 0, f"K={K} not multiple of {BLK}"
    return x.reshape(*lead, K // BLK, BLK), lead, K

def quant_sym_int4(x):
    """Q4_0-style symmetric: scale=amax/7, q in [-8,7]. Returns dequantized."""
    xb, lead, K = _blocks(x)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    sc = torch.where(amax > 1e-12, amax / 7.0, torch.full_like(amax, 1e-12))
    q = torch.clamp(torch.round(xb / sc), -8, 7)
    return (q * sc).reshape(*lead, K)

def quant_asym_int4(x):
    """Q4_K-style asymmetric per-32: scale=(max-min)/15, zp, q in [0,15]. Dequantized."""
    xb, lead, K = _blocks(x)
    xmax = xb.amax(dim=-1, keepdim=True)
    xmin = xb.amin(dim=-1, keepdim=True)
    sc = (xmax - xmin) / 15.0
    sc = torch.where(sc > 1e-12, sc, torch.full_like(sc, 1e-12))
    q = torch.clamp(torch.round((xb - xmin) / sc), 0, 15)
    return (q * sc + xmin).reshape(*lead, K)

def quant_sym_int8(x):
    """int8 symmetric per-32: scale=amax/127, q in [-127,127]. Dequantized."""
    xb, lead, K = _blocks(x)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    sc = torch.where(amax > 1e-12, amax / 127.0, torch.full_like(amax, 1e-12))
    q = torch.clamp(torch.round(xb / sc), -127, 127)
    return (q * sc).reshape(*lead, K)

# ---------------------------------------------------------------------------
# Hadamard rotate on last dim (per-32 block-diagonal)
# ---------------------------------------------------------------------------
def hadamard_rotate(x, H):
    """Apply block-diagonal H_32 to last dim. x:[...,K] -> [...,K]."""
    *lead, K = x.shape
    xb = x.reshape(*lead, K // BLK, BLK)
    out = torch.matmul(xb, H)   # [...,nb,32] @ [32,32]
    return out.reshape(*lead, K)

# ---------------------------------------------------------------------------
# Fake-quant linear forward
#   mode:
#     'fp16'      -> pristine
#     'q4k_int8'  -> weight Q4_K asym (pre-quantized), activation int8 per-32
#     'w4a4_qk_h' -> weight Q4_K rotated (pre), activation int4 + Hadamard
#     'w4a4_qk'   -> weight Q4_K (pre, no rot), activation int4, no Hadamard
#     'w4a4_q40_h'-> weight Q4_0 sym rotated (pre), activation int4 + Hadamard
# ---------------------------------------------------------------------------
def fake_quant_weight(weight, mode, H):
    """Return the fake-quantized weight (same device/dtype). Quant computed on CPU
    in fp32 (the 11B model nearly fills VRAM; avoid a big GPU fp32 intermediate)."""
    dev = weight.device; dt = weight.dtype
    w = weight.detach().to('cpu', torch.float32)
    Hc = H.to('cpu')
    if mode == 'q4k_int8':
        wq = quant_asym_int4(w)
    elif mode == 'w4a4_qk_h':
        # rotate weight on K (cols): W·H ; consistent with activation X·H so (XH)(WH)ᵀ=XWᵀ
        wq = quant_asym_int4(hadamard_rotate(w, Hc))
    elif mode == 'w4a4_qk':
        wq = quant_asym_int4(w)
    elif mode == 'w4a4_q40_h':
        wq = quant_sym_int4(hadamard_rotate(w, Hc))
    else:
        raise ValueError(mode)
    return wq.to(device=dev, dtype=dt)

def make_forward(mod, mode, H):
    """Activation-side fake-quant forward; weight already quantized IN-PLACE in mod.weight."""
    W = mod.weight  # already fake-quantized (or fp16 for baseline)
    b = mod.bias
    if mode == 'fp16':
        def fwd(x):
            return torch.nn.functional.linear(x, W, b)
        return fwd
    def fwd(x):
        xf = x.float()
        if mode == 'q4k_int8':
            xa = quant_sym_int8(xf)
        elif mode in ('w4a4_qk_h', 'w4a4_q40_h'):
            xa = quant_sym_int4(hadamard_rotate(xf, H))   # X·H then int4
        elif mode == 'w4a4_qk':
            xa = quant_sym_int4(xf)
        out = torch.nn.functional.linear(xa.to(W.dtype), W, b)
        return out.to(x.dtype)
    return fwd

# ---------------------------------------------------------------------------
# Patch the model: quantize target weights IN-PLACE (orig stashed on CPU) and
# wrap forward for the activation-side fake quant. Restores exactly on unpatch.
# ---------------------------------------------------------------------------
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

def patch_model(model, mode, H):
    saved = []
    n = 0
    for name, mod in model.named_modules():
        short = name.split('.')[-1]
        if short in TARGETS and isinstance(mod, nn.Linear):
            orig_fwd = mod.forward
            orig_w_cpu = None
            if mode != 'fp16':
                # stash original on CPU, overwrite GPU weight in-place (flat VRAM)
                orig_w_cpu = mod.weight.data.detach().to('cpu').clone()
                wq = fake_quant_weight(mod.weight.data, mode, H)
                mod.weight.data.copy_(wq)
                del wq
            saved.append((mod, orig_fwd, orig_w_cpu))
            mod.forward = make_forward(mod, mode, H)
            n += 1
    torch.cuda.empty_cache()
    return saved, n

def unpatch(saved):
    for mod, orig_fwd, orig_w_cpu in saved:
        mod.forward = orig_fwd
        if orig_w_cpu is not None:
            mod.weight.data.copy_(orig_w_cpu.to(mod.weight.device))
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# PPL eval (sliding non-overlapping windows of ctx tokens)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_ppl(model, input_ids, ctx, device, max_windows=None):
    nll_sum = 0.0
    ntok = 0
    nwin = input_ids.shape[1] // ctx
    if max_windows:
        nwin = min(nwin, max_windows)
    for i in range(nwin):
        chunk = input_ids[:, i*ctx:(i+1)*ctx].to(device)
        out = model(chunk)
        logits = out.logits[:, :-1, :].float()
        tgt = chunk[:, 1:]
        ll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction='sum')
        nll_sum += ll.item()
        ntok += tgt.numel()
        del out, logits
    return math.exp(nll_sum / ntok), ntok, nwin

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/home/janusz/models/Bielik-11B-v2.6-Instruct')
    ap.add_argument('--ctx', type=int, default=2048)
    ap.add_argument('--max-windows-en', type=int, default=40)
    ap.add_argument('--max-windows-pl', type=int, default=40)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='/home/janusz/AMD MXFP4/results/w4a4_ppl_results.json')
    args = ap.parse_args()

    dev = args.device
    print(f"[load] {args.model} -> {dev}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(dev).eval()
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    H = build_hadamard(BLK, dev, torch.float32)
    # sanity: orthogonality
    err = (H @ H.t() - torch.eye(BLK, device=dev)).abs().max().item()
    print(f"[had] H_32 orthogonality max err = {err:.2e}", flush=True)

    # ---- corpora ----
    print("[corpus] tokenizing EN wikitext-2 + PL wiki", flush=True)
    en_text = open('/home/janusz/llama.cpp/wikitext-2-raw/wiki.test.raw').read()
    pl_lines = open('/home/janusz/Polish_LLM_data/moje/source_corpus/wiki_passages_long_1500_raw.jsonl')
    pl_chunks = []
    for i, l in enumerate(pl_lines):
        if i >= 800:
            break
        r = json.loads(l)
        pl_chunks.append(r['text'])
    pl_text = "\n\n".join(pl_chunks)

    en_ids = tok(en_text, return_tensors='pt').input_ids
    pl_ids = tok(pl_text, return_tensors='pt').input_ids
    print(f"[corpus] EN tokens={en_ids.shape[1]}  PL tokens={pl_ids.shape[1]}", flush=True)

    variants = [
        ('1_fp16',        'fp16'),
        ('2_q4k_int8act', 'q4k_int8'),
        ('3_w4a4_qk_had', 'w4a4_qk_h'),
        ('4_w4a4_qk_noH', 'w4a4_qk'),
        ('5_w4a4_q40_had','w4a4_q40_h'),
    ]

    results = {'ctx': args.ctx, 'model': args.model, 'variants': {}}
    for vname, mode in variants:
        saved, n = patch_model(model, mode, H)
        t0 = time.time()
        ppl_en, nt_en, nw_en = eval_ppl(model, en_ids, args.ctx, dev, args.max_windows_en)
        ppl_pl, nt_pl, nw_pl = eval_ppl(model, pl_ids, args.ctx, dev, args.max_windows_pl)
        unpatch(saved)
        torch.cuda.empty_cache()
        dt = time.time() - t0
        results['variants'][vname] = {
            'mode': mode, 'n_patched': n,
            'ppl_en': ppl_en, 'ppl_pl': ppl_pl,
            'tok_en': nt_en, 'win_en': nw_en, 'tok_pl': nt_pl, 'win_pl': nw_pl,
            'sec': dt}
        print(f"[{vname:16s}] patched={n}  PPL_en={ppl_en:.4f}  PPL_pl={ppl_pl:.4f}  ({dt:.1f}s)", flush=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)

    # ---- summary deltas ----
    V = results['variants']
    base = V['1_fp16']
    print("\n==================== SUMMARY ====================")
    print(f"{'variant':18s} {'PPL_en':>9s} {'%vsFP16':>8s} {'PPL_pl':>9s} {'%vsFP16':>8s}")
    for vname, _ in variants:
        v = V[vname]
        de = 100*(v['ppl_en']/base['ppl_en']-1)
        dp = 100*(v['ppl_pl']/base['ppl_pl']-1)
        print(f"{vname:18s} {v['ppl_en']:9.4f} {de:7.2f}% {v['ppl_pl']:9.4f} {dp:7.2f}%")
    # critical delta: W4A4(#3) - Q4K_int8act(#2)  == extra int4-activation loss
    w4a4 = V['3_w4a4_qk_had']; q4ki8 = V['2_q4k_int8act']
    add_en = 100*(w4a4['ppl_en']/q4ki8['ppl_en']-1)
    add_pl = 100*(w4a4['ppl_pl']/q4ki8['ppl_pl']-1)
    print("\n*** ADDITIONAL int4-activation loss = W4A4(#3) vs Q4K-int8act(#2) ***")
    print(f"    EN: {add_en:+.2f}%    PL: {add_pl:+.2f}%   (must be < 1%)")
    # Hadamard benefit: #4 (noH) vs #3 (had)
    noH = V['4_w4a4_qk_noH']
    h_en = 100*(noH['ppl_en']/w4a4['ppl_en']-1)
    h_pl = 100*(noH['ppl_pl']/w4a4['ppl_pl']-1)
    print(f"\n*** Hadamard benefit: no-H(#4) is worse than +H(#3) by EN {h_en:+.2f}%  PL {h_pl:+.2f}% ***")
    results['deltas'] = {'add_int4act_en_pct': add_en, 'add_int4act_pl_pct': add_pl,
                         'hadamard_benefit_en_pct': h_en, 'hadamard_benefit_pl_pct': h_pl}
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {args.out}")

if __name__ == '__main__':
    main()
