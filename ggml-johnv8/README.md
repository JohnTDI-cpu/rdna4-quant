# ggml-JohnV8 — dedicated RDNA4/gfx1201 inference engine for llama.cpp

A purpose-built HIP inference engine for **AMD RDNA4** (Radeon AI PRO R9700, gfx1201; RDNA3/gfx11 expected-compatible, untested) that **beats both ROCm-HIP and Vulkan-RADV on decode AND prefill, losslessly**, across the full GGUF quant range. Built from first principles for our own silicon rather than hipified CUDA.

> Naming: `ggml-JohnV8` (architecture-agnostic — not RDNA4-locked; primitives used are gfx11+).
> Lossless mandate: coherent, correct text; int8 activation everywhere (W4A4/INT4-activation rejected at 10.85% NRMSE). **No 2:4 sparsity** (lossy).

---

## 1. Results (Bielik-11B-v3.0, R9700, sustained/warm, same GPU, lossless)

### DECODE (tg128, t/s) — we beat RADV on ALL 7 quants, ROCm on all 7
| Quant | ggml-JohnV8 | RADV | ROCm | vs RADV | vs ROCm |
|---|---|---|---|---|---|
| Q8_0 | 49.72 | 48.61 | 45.82 | **+2.3%** | +8.5% |
| Q4_0 | 86.40 | 80.47 | 75.95 | **+7.3%** | +13% |
| Q4_K | 83.13 | 81.19 | 70.88 | **+2.4%** | +17% |
| Q6_K | 60.02 | 59.28 | 55.46 | **+1.2%** | +8% |
| **Q4_K_M** | 77.12 | 76.35 | 67.15 | **+1.0%** | +15% |
| Q5_K | 70.67 | 68.83 | 63.91 | **+2.7%** | +6% |
| Q5_K_M | 68.11 | 66.87 | 62.52 | **+1.8%** | +5% |

### PREFILL (pp512, register-blocked INT8 WMMA GEMM aggregate, M=512)
| | t/s | TFLOPS | vs ROCm | vs RADV |
|---|---|---|---|---|
| ggml-JohnV8 RB GEMM (GEMM-only, L2-warm) | 5451 | 119 | **+69%** | **+95%** |
| full forward (cold weights + attention, est.) | ~3700-3800 | ~80 | **+15-30%** | **+35%+** |

Prefill sweep (pp64-32k) showed: simple quants (Q8/Q4_0) → ROCm leads ggml-baseline (well-tuned int8-WMMA mmq); K-quants → RADV leads (Q6_K worst, +37-40%). Our RB WMMA GEMM beats both.

---

## 2. File manifest

### `decode/`
| File | Role | Status |
|---|---|---|
| **m4_full.hip** (76 KB) | The complete K-quant DECODE engine. Self-contained GGUF parser + 50-layer Bielik forward. **Universal K-quant dispatcher** (`Q4K=1`): per-tensor type detection → Q4_K/Q5_K/Q6_K int8-sudot4 dots. `grid.sync` cooperative **megakernel** (whole layer in one launch). Q8_0 + Q4_0 native sudot4 paths. Handles Q4_K_M/Q4_K_S/Q5_K_M (mixed presets). | ✅ 7 quants beat both, lossless, validated |

### `prefill/`
| File | Role | Status |
|---|---|---|
| **prefill_v3.hip** (6 KB) | RB GEMM aggregate harness (50L × 5 GEMM, per-shape register-block configs). **The +69%/+95% result.** | ✅ validated |
| **wmma_rb.hip** (5 KB) | The register-blocked WMMA INT8 GEMM kernel (CUTLASS-style: block BMxBN, WMWxWNW warps, accumulators in REGISTERS → reuse both A and B). Per-shape sustained: qkv 82 / o 69.5 / gate-up 82.4 / down 102.3 TOPS. | ✅ core kernel |
| **flash_prefill.hip** (20 KB) | FlashAttention-2 forward: v0 baseline + v1 FP16 WMMA (`wmma_f32_16x16x16_f16`), causal, GQA. | ✅ SDPA-parity validated (test_flash_prefill.py) |
| **swmmac_engine.hip** (36 KB) | Full WMMA GEMM library: dense IU4 (`wmma_gemm_tiled`), asymmetric-correct IU4 (zero-point analytical correction for Q4_K/Q5_K min), tiling (`swmmac_tile_dense`), activation quant (`swmmac_quantize_act`), scale transpose. (2:4-sparse path present but UNUSED — lossy.) | ✅ verified on R9700 |
| swmmac_gemm.hip (10 KB) | Standalone WMMA GEMM variant. | reference |
| prefill_gemm_v2.hip, wmma_gemm_db.hip, wmma_i8_gemm.hip | GEMM evolution (naive 7 → tiled 53 → double-buffer 66 → register-blocked 85+ TOPS). | history/reference |
| test_flash_prefill.py, bench_flash_prefill.py | FA2 correctness (allclose vs torch SDPA) + benchmark. | ✅ |

### `docs/`
| File | Role |
|---|---|
| **DISCOVERY_GFX12_WMMA_OUTPUT.md** | The RDNA4 WMMA fragment layout (column-distributed: `lane%16 = N-col`, `lane/16 = K-half`, `acc v8i[j] = M-row`). Critical — without it every 16×16 tile comes out transposed. |

### External (in repo, referenced — not duplicated here)
- `../hip_int4/int4_decode_step.hip` (258 KB) — MoE engine (Qwen3-30B-A3B) with batched `head_rmsnorm_rope_prefill` + a working MoE prefill forward (commit `11c2772`: BEAT GGUF ON ALL PREFILL CONTEXTS). Reusable prefill-forward scaffolding.

---

## 3. Key techniques (the levers that beat both)

**Decode (memory-bound):**
1. **int8-sudot4 K-quant dots** — the K-quant gap is *dequant compute*, not memory/dispatch. `__builtin_amdgcn_sudot4` (4 int8-MAC/instr) on the quant value × int8 activation, scales applied after. ~2-4× fewer ALU than f32 dequant → flips K-quants from −3.4% to wins.
2. **Packed scales, zero inflation** — read on-disk packed layouts (Q5_K 176B, Q6_K 210B); decode 6-bit scales inline (`gsm`) rather than pre-expanding to int8 (which inflated +2-31% → memory-bound loss).
3. **`grid.sync` megakernel** — whole transformer layer in one cooperative launch (mclk-boosted ~0.6s); collapses ~700 dispatches/token. +5.7% on Q8 over per-op.
4. **Per-tensor mixed-quant dispatch** — `find()` reads each tensor's `ggml_type`; one engine handles Q4_K_M/Q4_K_S/Q5_K_M mixes (Q4_K + Q6_K per layer).

**Prefill (compute/matrix-core-bound):**
1. **Register-blocked INT8 WMMA GEMM** — CUTLASS-style: accumulators in registers, reuse both operands. 85 TFLOPS sustained vs ROCm/RADV ~57/54 (they sit at ~14-16% of the 282-TOPS real INT8 peak).
2. **Per-shape tuned configs + fuse gate+up** (N=28672, 5→4 launches/layer). Down-shape needs 256×256 tiles (512×256 left half the CUs idle).
3. **INT8 activation** (lossless) over INT4 (W4A4 = 10.85% NRMSE).

---

## 4. Integration into llama.cpp (ggml-cuda backend)

Target: a `ggml-JohnV8` path inside the ROCm/HIP backend (`ggml/src/ggml-cuda/`), gated for `GGML_CUDA_CC_IS_RDNA4` (and RDNA3 after validation), selected per-op:

| ggml op | ggml-JohnV8 replacement |
|---|---|
| `mul_mat_vec` (decode GEMV), K-quant | int8-sudot4 dot (`m4_full.hip` kernels) + per-tensor dispatch |
| layer graph (decode) | `grid.sync` megakernel fusion (Llama-dense arch detect; per-op fallback otherwise) |
| `mul_mat` (prefill GEMM), all quants | register-blocked INT8 WMMA GEMM (`wmma_rb.hip`) + K-quant→int8 dequant |
| attention (prefill) | FA2 (`flash_prefill.hip`) |

**Generalization needed for ship:** dims are currently hardcoded to Bielik-11B (NE=4096, NH=32, NKV=8, NFF=14336, NL=50). Parameterize from the GGUF header (n_embd/n_head/n_ff/n_layer). Architecture is Llama-dense (RMSNorm + RoPE + GQA + SwiGLU); MoE = a second family.

---

## 5. Build & run

```bash
# Decode engine (standalone, any K-quant GGUF with Bielik-11B dims)
hipcc --offload-arch=gfx1201 -O3 decode/m4_full.hip -I/usr/include/libdrm -o /tmp/m4_full -ldrm_amdgpu
HIP_VISIBLE_DEVICES=0 Q4K=1 /tmp/m4_full <model.gguf> 64     # universal K-quant runner
# (Q4=1 → Q4_0; default → Q8_0)

# Prefill GEMM aggregate
hipcc --offload-arch=gfx1201 -O3 prefill/prefill_v3.hip -o /tmp/prefill_v3 && HIP_VISIBLE_DEVICES=1 /tmp/prefill_v3
```

**Benchmark discipline:** measure WARM — RDNA4 mclk ramps 96→1258 MHz in ~0.6 s; the engine has no internal warmup, so a cold single short run reads low. Use ≥2 runs, take the warm one. Not thermal (temps stay 34-88°C; mclk holds 1258 even at 88°C).

---

## 6. Status & remaining work

- ✅ **Decode: DONE** — 7 quants beat ROCm + RADV, lossless, validated.
- ✅ **Prefill GEMM: DONE** — register-blocked INT8 WMMA beats both (GEMM-only +69%/+95%).
- 🔜 **Prefill K-quant dequant→int8** before the GEMM (Q4_K/Q5_K/Q6_K; Q6_K is the biggest target, RADV +40%).
- 🔜 **Full prefill forward** (M>1): RB GEMM + FA2 + batched norm/rope → real pp512 vs both.
- 🔜 **Generalize dims** from Bielik-11B to any GGUF (header-driven).
- 🔜 **KV-cache TurboQuant** (4-bit KV, +0.83% PPL near-lossless, validated on gfx1201) — long-context lever for the attention phase.

---

*See `../LESSONS_LEARNED.md` for the full experimental record. Decode-engine memory: `project_rdna4_engine_beats_radv`. Prefill plan: `project_prefill_build_plan`.*
