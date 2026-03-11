# AMD MXFP4/NVFP4 Quantization Research — Complete Notes

**Project**: Custom 4-bit quantization engine for AMD RDNA4 (gfx1201)
**Model**: Qwen3-14B (14.77B params, 40 layers, hidden=5120, intermediate=17408)
**Hardware**: AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32GB VRAM, 576 GB/s bandwidth)
**Stack**: ROCm 7.1.0, PyTorch 2.10.0+rocm7.1, hipcc (clang-19)
**Date**: March 2026

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Quantization Methods Tested](#quantization-methods-tested)
3. [Perplexity Results (WikiText-2)](#perplexity-results)
4. [Benchmark Results (lm-evaluation-harness)](#benchmark-results)
5. [Speed & VRAM Results](#speed--vram-results)
6. [What Worked Well](#what-worked-well)
7. [What Did NOT Work / Not Worth Pursuing](#what-did-not-work)
8. [What Is Worth Pursuing Further](#what-is-worth-pursuing-further)
9. [Critical Technical Discoveries](#critical-technical-discoveries)
10. [File Structure](#file-structure)
11. [How to Use](#how-to-use)

---

## Executive Summary

We built a complete FP4 quantization + inference pipeline from scratch for AMD RDNA4,
targeting the first consumer GPU with WMMA (Wave Matrix Multiply-Accumulate) matrix cores.
Two quantization formats were explored:

- **MXFP4** (OCP MX v1.0): E2M1 data + E8M0 block scales, block_size=32
- **NVFP4**: E2M1 data + E4M3 block scales, block_size=16

Key finding: **NVFP4 format is strictly better than MXFP4** for quality at similar speed,
because E4M3 scales (256 values) are far more expressive than E8M0 (power-of-2 only, 32 values),
and block_size=16 captures finer-grained distributions.

Best achieved quality: **PPL 7.692** (INT4 Asymmetric + Hadamard + GPTQ with proper calibration),
which **BEATS llama.cpp Q4_K_M** (PPL 7.715) by 0.023 on identical hardware and evaluation methodology.
BF16 baseline is PPL 7.556 (all measured with sliding window, ctx=2048, stride=512).

**CRITICAL: PPL Evaluation Methodology (discovered 2026-03-11)**
All PPL numbers in the "MXFP4 Methods" table below used non-overlapping chunks (stride=ctx), which
INFLATES PPL by ~1.1 points. These numbers are only valid for relative comparison within the table.
The INT4 v4 results use the CORRECT sliding-window methodology matching llama.cpp and lm-eval-harness.

---

## Quantization Methods Tested

### MXFP4 Format (E2M1 + E8M0, block=32)

| # | Method | PPL | vs FP16 | Time | Notes |
|---|--------|-----|---------|------|-------|
| 1 | RTN (round-to-nearest) | 9.81 | +13.8% | ~5 min | Baseline, simplest method |
| 2 | MSE (optimal clipping) | 9.67 | +12.1% | ~5 min | Small improvement over RTN |
| 3 | GPTQ (Cholesky error compensation) | 9.51 | +10.4% | ~15 min | Modest improvement, heavyweight |
| 4 | Hadamard rotation + MSE | 9.37 | +8.7% | ~6 min | Block-diagonal Hadamard spreads outliers |
| 5 | Hadamard + GPTQ | 9.14 | +6.0% | ~25 min | Best simple combo |
| 6 | Learned rotation (150 steps) | 9.23 | +7.1% | ~1h | SPSA on Stiefel manifold |
| 7 | Learned rotation (500 steps, seq) | 9.58 | +11.1% | ~3h | Overfitted! More steps != better |
| 8 | **Learned rot + GPTQ hybrid** | **9.07** | **+5.2%** | ~2h | **Best MXFP4 result** |
| 9 | SmoothQuant + Hadamard + GPTQ | ~9.1 | ~+5.7% | ~30 min | No better than pure Hadamard+GPTQ |

### NVFP4 Format (E2M1 + E4M3 scales, block=16)

| # | Method | PPL | vs FP16 | Notes |
|---|--------|-----|---------|-------|
| 10 | NVFP4 RTN | ~8.9 | ~+3.2% | Already better than best MXFP4! |
| 11 | NVFP4 + AWQ | ~8.7 | ~+0.9% | Activation-aware scaling |
| 12 | NVFP4 + AWQ + GPTQ | ~8.6 | ~+0% | Approaches FP16 quality |

### INT4 Symmetric (uniform levels [-8..+7], FP16 scales)

| # | Method | PPL | Notes |
|---|--------|-----|-------|
| 13 | INT4 + GPTQ | ~9.3 | Similar to MXFP4 Hadamard+MSE |
| 14 | INT4 + Learned rotation + GPTQ | ~9.1 | Similar to best MXFP4 |

### INT4 Asymmetric + Hadamard (FP16 block scales, block=32) — BEST RESULTS

All measured with CORRECT sliding-window eval (ctx=2048, stride=512).

| # | Method | PPL | Δ vs BF16 | Δ vs Q4_K_M | Notes |
|---|--------|-----|-----------|-------------|-------|
| 15 | v3 asym+Had+noGPTQ | 7.718 | +0.162 | +0.003 | Simple block quantization, no GPTQ |
| 16 | **v4 asym+Had+GPTQ (mixed INT4/INT8)** | **7.692** | **+0.136** | **-0.023** | **BEATS Q4_K_M!** 256 cal × 256 tok, 20% INT8 |
| 17 | v5 pure INT4+Had+GPTQ | 7.787 | +0.231 | +0.072 | All INT4, no mixed precision. 7.7 GB vs 9.2 GB |

### Reference Baselines (Sliding Window, ctx=2048, stride=512)

| Method | PPL | BPW | Notes |
|--------|-----|-----|-------|
| BF16 (our engine) | 7.556 | 16.0 | Sliding window baseline |
| **Q4_K_M (llama-perplexity, same GPU)** | **7.715** | **4.87** | Same hardware, same eval |
| FP16 (our engine, non-overlapping) | 8.62 | 16.0 | Old methodology — INFLATED |

Note: The old "FP16=8.62 vs Q4_K_M=7.66" comparison was apples-to-oranges due to different evaluation
methodologies (non-overlapping chunks vs sliding window). With identical methodology on identical hardware,
our v4 INT4 engine **beats Q4_K_M** by 0.023 PPL.

---

## Benchmark Results

### lm-evaluation-harness (NVFP4+AWQ, 100 samples)

| Benchmark | Accuracy | Normalized |
|-----------|----------|------------|
| ARC Challenge | 57% | 54% |
| HellaSwag | 52% | 68% |
| TruthfulQA MC2 | 56% | — |
| WinoGrande | 75% | — |

---

## Speed & VRAM Results

### Qwen3-14B on single R9700

| Engine | Prefill (t/s) | Decode (t/s) | VRAM (MB) |
|--------|---------------|--------------|-----------|
| MXFP4 (Triton GEMM + HIP GEMV) | 2287 | ~15 | 8648 |
| MXFP4 (Triton GEMM, PPL eval) | 2857 | — | 8648 |
| GGUF Q4_K_M (llama.cpp ROCm) | 541 (pp512) | 52 | 8788 |
| GGUF Q4_K_M (llama.cpp Vulkan) | 535 (pp512) | 7 (broken) | — |

Key observations:
- Our Triton GEMM prefill is **4.2x faster** than llama.cpp ROCm for pp512
- Decode speed (single-token GEMV) is the bottleneck — bandwidth-limited
- llama.cpp Vulkan decode is broken on RDNA4 (warp_size=64 misdetect for Wave32)
- WMMA GEMM kernel achieves **40.8 TFLOPS** (53% of FP16 theoretical)

---

## What Worked Well

### 1. NVFP4 format (E4M3 scales, block=16)
NVFP4 RTN alone outperforms the best MXFP4 method. The improvement comes from:
- E4M3 scales (256 distinct values) vs E8M0 (32 power-of-2 values)
- block_size=16 vs 32 — finer granularity catches local outliers
- **Recommendation: Always use NVFP4 over MXFP4**

### 2. AWQ (Activation-Weighted Quantization)
Cheapest quality improvement per compute cost. AWQ rescales channels by activation magnitude
before quantization, protecting important channels. ~5 min extra, significant quality gain.
- **Recommendation: Always combine with AWQ**

### 3. Hadamard rotation
Block-diagonal Hadamard rotation (orthogonal transform) spreads weight outliers across the block,
reducing quantization error. Fast to compute (~6 min total), good quality/cost ratio.
- Works because LLM weights have heavy-tailed distributions with per-channel outliers
- Block-diagonal (block=32) is sufficient — full rotation is impractical

### 4. GPTQ error compensation
Cholesky-based second-order error compensation. Worthwhile when combined with other methods.
Solo GPTQ improvement is modest, but it stacks well with Hadamard/AWQ.
- Crashed on layer 30 of Hadamard+GPTQ due to HIP kernel failure (transform_hessian)
- **Workaround: save weights per-layer to survive crashes**

### 5. Fused WMMA GEMM kernel
Our custom HIP kernel fuses MXFP4 dequantization + FP16 WMMA multiplication in one kernel.
Eliminates the global memory round-trip for dequantized weights.
- 3.8x faster than separate dequant + hipBLAS for batch size <= 32
- Key insight: LDS-based LUT for FP4 decode (1 read vs 6-8 ALU ops)
- Key insight: ldexpf for E8M0 scale application (1 ALU op)

### 6. HIP C++ decode loop (zero Python overhead)
Moving the entire 40-layer decode loop into C++ eliminates ~2ms per-token Python overhead.
Critical for single-token decode where each layer takes ~1.5ms.

---

## What Did NOT Work

### 1. Learned rotation (SPSA on Stiefel manifold) — diminishing returns
- 150 steps: 9.23 PPL (decent)
- 500 steps: 9.58 PPL (WORSE — overfitted!)
- The Stiefel manifold constraint makes optimization unstable beyond ~200 steps
- Combined with GPTQ ("hybrid"): 9.07 PPL — best MXFP4 result, but 2h compute
- **Not worth it**: Hadamard+GPTQ gets 9.14 in 25 min (same ballpark, 5x cheaper)
- **Not worth it vs NVFP4**: NVFP4 RTN gets ~8.9 in 5 min

### 2. SmoothQuant + Hadamard
Per-channel activation smoothing (migrating quantization difficulty from activations to weights).
- No improvement over plain Hadamard+GPTQ
- Adds complexity without benefit for weight-only quantization
- SmoothQuant was designed for activation quantization — less relevant here

### 3. MXFP4 E8M0 scales in general
The fundamental bottleneck: E8M0 can only represent powers of 2 (2^-127 to 2^127).
Real weight distributions need continuous scales. This adds ~1 bit of quantization noise
compared to E4M3 or FP16 scales.
- **Root cause of the PPL gap vs GGUF Q4_K_M** (which uses FP16 block scales)
- No amount of rotation/calibration can fully compensate for scale quantization noise

### 4. Multi-step iterative quantization
More optimization steps do NOT necessarily help:
- 500 steps learned rotation = worse than 150 steps (overfitting)
- The calibration set (128 samples, 256 tokens) is small — easy to overfit
- **Recommendation: Keep calibration light (128-256 samples, 150-200 steps max)**

### 5. Codebook/asymmetric quantization experiments (MXFP4 era)
Various experimental approaches within MXFP4 format that showed no meaningful improvement:
- Asymmetric INT4 within MXFP4 framework — slight quality gain but breaks WMMA packing
- Mixed precision (FP8 for sensitive layers) — complicated, marginal benefit
- Custom codebook optimization — too slow, negligible gain

**NOTE**: Asymmetric INT4 with FP16 scales (NOT MXFP4) turned out to be excellent — see v3/v4 results.
The key insight: asymmetric + Hadamard rotation + proper GPTQ calibration BEATS Q4_K_M.

### 6. llama.cpp Vulkan backend on RDNA4
- Decode completely broken: 7.22 t/s vs 51.88 t/s ROCm
- Root cause: RADV misdetects RDNA4 as Wave64 (it's Wave32)
- Must use ROCm backend or our custom engine

---

## What Is Worth Pursuing Further

### 1. INT4 Asymmetric + Hadamard + GPTQ engine (PRIORITY: HIGHEST)
Our v4 quantization **beats Q4_K_M** on quality (PPL 7.692 vs 7.715). Now the priority shifts
to building a fast inference engine for this format:
- INT4 GEMV decode kernel (fork from NVFP4 GEMV, simpler dequant: no LUT, just multiply)
- INT4 WMMA GEMM for prefill (v_wmma_i32_16x16x32_iu4 native instruction)
- Combined with HIP C++ decode loop for zero Python overhead
- Target: 60+ t/s decode, 5000+ t/s prefill
- Key files: `quantized_v4_gptq/` (best weights), `quantize_v4_gptq.py` (quantizer)

### 2. GPTQ calibration optimization
v4 used WikiText-2 train (fallback from C4 due to API change) with 256 samples × 256 tokens.
Could improve further with:
- C4/RedPajama diverse calibration data (if API is fixed)
- Longer seqlen (512) with enough samples (need >128 for well-conditioned Hessian)
- Larger INT8 percentage for sensitive layers (currently 20%)

### 3. KV-cache quantization (FP8 E4M3)
Already implemented and tested for NVFP4 engine. Port to INT4 engine.
- VRAM savings: 320 MB at ctx=4096, 1.28 GB at ctx=16K
- Quality: verified stable PPL

### 4. Faster decode kernels
Single-token decode is bandwidth-limited at ~60 t/s. Potential optimizations:
- Persistent kernels that keep weights in L2 cache
- Fused multi-layer decode (combine adjacent layers)
- Speculative decoding (batch verify multiple tokens)

### 5. Expanding to other models
Engine currently supports Qwen2/Qwen3 architecture. Extending to LLaMA, Mistral,
DeepSeek would increase utility. Architecture is modular enough for this.

---

## Critical Technical Discoveries

### WMMA Lane Mapping on RDNA4 (gfx12)

**This is documented in detail in [DISCOVERY_GFX12_WMMA_OUTPUT.md](DISCOVERY_GFX12_WMMA_OUTPUT.md)**

The `v_wmma_f32_16x16x16_f16` instruction uses a **column-distributed fragment layout**:
```
VGPR[lane][j] = matrix[(lane / 16) * 8 + j][lane % 16]
```
Lanes index COLUMNS (not rows). Getting this wrong produces transposed tiles.
No AMD documentation clearly explained this — we derived it empirically and verified
against CK's `wmma_gemm.hpp` diagram.

### E8M0 Scale Limitation (MXFP4)

E8M0 can only represent 32 distinct scale values (powers of 2). For a block of 32 weights,
the optimal continuous scale is `max(abs(block)) / 6.0` (E2M1 range). When this is rounded
to the nearest power of 2, the relative error can be up to 41% (sqrt(2) - 1).

This is the single largest source of quantization error in MXFP4. NVFP4's E4M3 scales
(256 distinct values with sub-exponent precision) reduce this error by ~4x.

### RoPE Must Use Original Positions (not cache-relative)

When implementing RoPE for the KV-cache decode path, position indices must be the
absolute token position, not the position within the cache. Getting this wrong produces
correct-looking output for short sequences but degrades quality for longer contexts.

### HIP Kernel Crashes During GPTQ

Large matrix operations (Hessian transformation for 34816x5120 gate_up projections)
can trigger `hipErrorLaunchFailure` on ROCm 7.1. Workaround:
- Save per-layer weights immediately after quantization
- Use smaller Cholesky block sizes (64 instead of 128)
- The crash is likely a ROCm memory management bug, not a hardware limit

### Triton on RDNA4 Quirks

- `tl.dot` maps to WMMA — works well for GEMM (M >= 16)
- GEMV (M=1) cannot use WMMA — must use scalar FP32 accumulation
- `tl.atomic_add` with FP16 is not supported — must use FP32 atomics
- Block sizes must be powers of 2 and >= 16 for WMMA path

---

## File Structure

After refactoring, the project is organized as:

```
AMD MXFP4/
  # Documentation
  RESEARCH_NOTES.md          — This file (comprehensive research summary)
  DISCOVERY_GFX12_WMMA_OUTPUT.md — WMMA lane mapping guide for RDNA4
  benchmark_results.json     — All benchmark data in structured format

  # Main entry points
  convert.py                 — High-level model converter (HF → MXFP4)

  # Inference engines
  mxfp4_engine.py            — MXFP4 inference engine (Qwen2/Qwen3)
  nvfp4_awq_hip_engine.py    — NVFP4+AWQ fastest engine (HIP C++ decode)
  nvfp4_awq_fast_engine.py   — NVFP4+AWQ Triton-based engine
  nvfp4_awq_graph_engine.py  — NVFP4+AWQ with CUDA Graph
  qwen35_engine.py           — Qwen3.5-4B engine (GatedDeltaNet hybrid)

  # Triton kernels
  mxfp4_gemm_v6.py           — MXFP4 GEMM (prefill, fused dequant+matmul)
  mxfp4_gemv.py              — MXFP4 GEMV (decode, bandwidth-optimized)
  mxfp4_fused_gemv.py        — Fused QKV/GateUp GEMV (fewer launches)
  nvfp4_triton.py            — NVFP4 GEMV/GEMM kernels
  nvfp4_gemm_fast.py         — NVFP4 GEMM kernel (fast path)
  fused_ops.py               — Fused RMSNorm + RoPE (Triton)

  # Quantization core
  mxfp4_quantize.py          — CPU MXFP4 quantization
  mxfp4_quant_gpu.py         — GPU MXFP4 quantization (Triton)
  int4_quant.py              — INT4 symmetric quantization
  hadamard_utils.py          — Hadamard rotation utilities

  # Quantization scripts (produce quantized_*/ directories)
  quantize_gptq.py           — MXFP4 + GPTQ
  quantize_hadamard.py       — MXFP4 + Hadamard rotation + MSE
  quantize_hadamard_gptq.py  — MXFP4 + Hadamard + GPTQ
  quantize_learned_rotation.py — MXFP4 + Learned rotation (SPSA/Stiefel)
  quantize_nvfp4_qwen3.py    — NVFP4 RTN
  quantize_nvfp4_awq.py      — NVFP4 + AWQ
  quantize_nvfp4_awq_gptq.py — NVFP4 + AWQ + GPTQ
  quantize_int4_gptq.py      — INT4 + GPTQ
  quantize_v3_nogptq.py      — INT4 asym + Hadamard, NO GPTQ (v3 baseline)
  quantize_v4_gptq.py        — INT4 asym + Hadamard + GPTQ with proper calibration (BEST QUALITY)
  quantize_qwen35.py         — Qwen3.5-4B specific quantization

  # Perplexity evaluation
  bench_ppl_fp16.py          — FP16 baseline PPL
  bench_ppl_gptq.py          — MXFP4 GPTQ PPL
  bench_ppl_hadamard.py      — MXFP4 Hadamard PPL
  bench_ppl_learned.py       — MXFP4 Learned rotation PPL
  bench_ppl_nvfp4.py         — NVFP4 PPL
  bench_ppl_int4.py          — INT4 PPL
  bench_ppl_qwen35.py        — Qwen3.5-4B PPL
  nvfp4_lm_eval.py           — lm-evaluation-harness wrapper (NVFP4)
  gguf_lm_eval.py            — lm-evaluation-harness wrapper (GGUF)

  # Speed benchmarks
  bench_nvfp4_final.py       — NVFP4 final speed benchmark
  benchmark_comparison.py    — Comparative benchmarks (GGUF vs custom)
  bench_full.py              — Full forward pass timing

  # HIP C++ kernels
  hip_mxfp4/                 — MXFP4 HIP source files
  hip_nvfp4/                 — NVFP4 HIP source files + compiled .so
  *.so                       — Compiled HIP extensions (root dir)

  # Quantized weights (use quantized_v4_gptq for best quality)
  quantized_v4_gptq/         — INT4 asym+Had+GPTQ weights (BEST QUALITY, PPL 7.692)
  quantized_v3_had/          — INT4 asym+Had+noGPTQ weights (PPL 7.718)
  quantized_nvfp4_awq/       — NVFP4+AWQ weights (best NVFP4 quality)
  quantized_nvfp4_awq_gptq/  — NVFP4+AWQ+GPTQ weights (best absolute quality)
  quantized_nvfp4/           — NVFP4 RTN weights
  quantized_learned_gptq/    — MXFP4 Learned+GPTQ weights (best MXFP4)
  quantized_hadamard/        — MXFP4 Hadamard+MSE weights
  quantized_hadamard_gptq/   — MXFP4 Hadamard+GPTQ weights
  quantized_gptq/            — MXFP4 GPTQ weights (baseline)
  quantized_int4_gptq/       — INT4+GPTQ weights
  quantized_qwen35_gptq/     — Qwen3.5-4B quantized weights

  # Data
  wikitext-2-test.txt        — WikiText-2 test set for perplexity evaluation
  Qwen3.5-4B-GGUF/           — GGUF model for comparison benchmarks

  # Shell scripts
  run_multigpu_sequential.sh — Multi-GPU quantization orchestrator
  run_multigpu.sh            — Parallel multi-GPU variant
```

---

## How to Use

### Quick start — best quality quantization
```bash
# Quantize with INT4 + Hadamard + GPTQ (BEST QUALITY, ~20 min)
python3 quantize_v4_gptq.py --save-dir quantized_v4_gptq

# Evaluate perplexity (sliding window, CORRECT methodology)
python3 bench_ppl_v4_proper.py

# Alternative: NVFP4+AWQ (faster decode engine available)
python3 quantize_nvfp4_awq.py --model Qwen/Qwen3-14B
python3 nvfp4_awq_hip_engine.py
```

### Using the converter (MXFP4)
```bash
# RTN (fastest, ~5 min)
python convert.py --model Qwen/Qwen3-14B --method rtn

# Hadamard+GPTQ (best simple method, ~15 min)
python convert.py --model Qwen/Qwen3-14B --method hadamard

# Learned rotation (best MXFP4 quality, ~2h)
python convert.py --model Qwen/Qwen3-14B --method learned --steps 200

# Multi-GPU learned rotation (~1h with 2 GPUs)
python convert.py --model Qwen/Qwen3-14B --method learned --steps 200 --multi-gpu
```

### Build HIP extensions
```bash
cd hip_nvfp4 && python setup.py build_ext --inplace
cd hip_mxfp4 && python setup.py build_ext --inplace  # if present
```

---

## Lessons for Future AI Assistants

1. **INT4 asymmetric + Hadamard + proper GPTQ is the winning combo** — beats Q4_K_M on quality
2. **GPTQ calibration quality matters enormously** — 37 samples = hurts PPL; 256 samples = helps PPL
3. **Always use sliding-window PPL eval** (stride=512, ctx=2048) — non-overlapping chunks inflate PPL by ~1.1
4. **Hadamard rotation is 90% of learned rotation quality at 10% cost** — don't over-optimize rotations
5. **NVFP4 is better than MXFP4 but INT4+FP16 scales is even better** — FP16 scales > E4M3 > E8M0
6. **Watch for ROCm HIP crashes** — save intermediate results, use per-layer checkpointing
7. **WMMA lane mapping is column-distributed** — see DISCOVERY_GFX12_WMMA_OUTPUT.md
8. **llama.cpp Vulkan is broken on RDNA4** — use ROCm backend only
9. **More calibration steps can overfit** — 150-200 steps is the sweet spot
10. **Decode is bandwidth-limited** — kernel fusion matters more than compute optimization
11. **Compare apples to apples** — always use same eval methodology when comparing models
12. **WikiText-2 train has limited long samples** — at seqlen=512 only 37 samples; use seqlen=256 for 256+ samples
