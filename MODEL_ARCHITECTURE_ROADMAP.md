# ggml-JohnV8 — Model Architecture Roadmap

Which model families the engine should support, ordered by value. **Key insight: our wins (K-quant
decode GEMV int8-sudot4, prefill int8 WMMA GEMM + mem-op fusion) are architecture-AGNOSTIC** — they
apply to *any* model using GGUF quants. The only architecture-specific parts are the **attention
module** and the **FFN module** (dense vs MoE). So: engine = shared quant core + pluggable modules.

## Engine = core + pluggable modules

**Shared core (done / in progress, reused by ALL architectures):**
- K-quant decode GEMV (int8-sudot4, packed scales, megakernel) — ✅ 7 quants beat both
- K-quant prefill GEMM (dequant→re-quant per-32 int8 → WMMA, + adaptive mem-op fusion) — ✅ winning
- Per-tensor mixed-quant dispatch — ✅

**Attention modules (the main differentiator):**
| Mod | Mechanism | Models | Status |
|---|---|---|---|
| **A1** | GQA full + RoPE | Llama, Mistral, Qwen2/3, Phi, Command-R | ✅ FA2 v1 (have) |
| **A2** | Sliding-window (local/global alt.) | Gemma 2/3, Mistral SWA, Llama-4 | 🔜 FA2 windowed variant |
| **A3** | **MLA** (latent/compressed KV) | DeepSeek V2/V3/R1 | 🔜 NEW — distinct |
| **A4** | **DeltaNet / gated-delta** linear attn | Qwen3-Next, Qwen3.5 | 🔜 NEW — chunked recurrent scan (refs in int4_decode_step) |
| **A5** | Mamba-2 SSM scan | Jamba, Zamba, Nemotron-H, Granite-4 | 🔜 NEW — lower priority |

**FFN modules:**
| Mod | Type | Models | Status |
|---|---|---|---|
| **F1** | Dense SwiGLU / GeGLU | Llama, Mistral, Qwen-dense, Gemma, Phi | ✅ have |
| **F2** | **MoE** (top-k route + experts + shared) | Qwen-MoE, Mixtral, DeepSeek, Llama-4, GLM, GPT-OSS, Granite-MoE | 🔶 Qwen-MoE exists → generalize |

**Norm/PE small-adds:** QK-norm (Qwen3, Gemma3), pre+post-norm (Gemma), logit soft-cap (Gemma2), partial-RoPE/NoPE-layers (some), QKV-bias (Qwen2).

---

## Roadmap by value (popularity × our relevance × incremental cost)

### Phase 0 — DONE
- **Llama / Mistral dense** (Bielik-11B). `A1 + F1`. The reference. ✅

### Phase 1 — small deltas, high coverage (cheap, do first)
1. **Qwen2.5 / Qwen3 dense** — `A1 + F1` + QK-norm + QKV-bias. Tiny delta over Bielik. **Your coder/agent base.**
2. **Phi-3 / Phi-4 dense** — `A1 + F1`. Near-trivial (Llama-like).
3. **Command-R, Yi, InternLM, GLM-4 dense** — `A1 + F1` + per-model config. Free once Qwen done.

### Phase 2 — MoE module (ONE module unlocks MANY popular models) ⭐ highest leverage
- **F2 general MoE** (top-k routing + expert GEMM batching + shared expert). The project already has
  Qwen3-30B-A3B MoE (`qwen3_30b_a3b/int4_engine_moe.py`, `int4_decode_step.hip`) — **generalize it.**
- Unlocks: **Qwen3-MoE (30B-A3B, 235B), Mixtral 8x7B/8x22B, DeepSeek-MoE, Llama-4 Scout/Maverick,
  GLM-4-MoE, GPT-OSS, Granite-MoE, Qwen3-Coder-MoE.** Huge popularity per unit work.
- The quant win matters MOST here: MoE is memory-bound (many experts), K-quant decode GEMV win shines.

### Phase 3 — distinct attention (the real architectural builds)
3a. **DeepSeek-V3 / R1 (MLA + MoE)** ⭐ — the SOTA open model, extremely popular. `A3 MLA` (compressed
    KV, absorbed projections — saves huge KV cache) + Phase-2 MoE. Big effort, biggest single-model payoff.
3b. **Gemma 3 (sliding window)** — `A2` (alternating local-512/global) + GeGLU + QK-norm. **Your Polish/Gemma work.**
3c. **Qwen3-Next / Qwen3.5 (DeltaNet hybrid)** — `A4` gated-delta linear attention (most layers) +
    `A1` full attention (every Nth layer) + MoE. **Your coder/agent direction.** Linear-attn = O(N) long ctx win.

### Phase 4 — specialized / lower priority
- **Mamba-2 / hybrid (Jamba, Zamba, Nemotron-H, Granite-4)** — `A5` SSM scan. Distinct, less GGUF usage.
- **MiniMax-01 (lightning attention)**, **Kimi-K2 (huge MoE)** — niche / very large.

---

## Recommended order (engine modules, not models)

1. **Phase 1 dense configs** (Qwen3, Phi) — days, validates the "config not rewrite" thesis. Your coder base.
2. **Phase 2 MoE module** — the leverage multiplier (Mixtral + Qwen-MoE + DeepSeek-MoE + Llama-4 + GPT-OSS).
3. **A2 sliding-window** (Gemma 3) — moderate, your Polish models.
4. **A4 DeltaNet** (Qwen3-Next) — your coder/agent direction, long-context linear-attn win.
5. **A3 MLA** (DeepSeek-V3) — biggest single payoff, biggest effort.
6. **A5 Mamba** — last.

## Your concrete model set (maps to phases)
| Model | Arch | Phase |
|---|---|---|
| Bielik-11B / fit-6B | Llama/Mistral dense | ✅ 0 |
| Qwen3-Coder dense, Qwen2.5 | Qwen dense | 1 |
| Qwen3-30B-A3B | Qwen MoE | 2 |
| Mixtral, DeepSeek-MoE | MoE | 2 |
| Gemma-3/4 (PL bench) | sliding-window | 3b |
| Qwen3.5 / Qwen3-Next | DeltaNet hybrid | 3c |
| DeepSeek-V3 / R1 | MLA + MoE | 3a |

**Strategic note:** Phase 1+2 (dense configs + MoE module) cover ~80% of popular GGUF models with
~20% of the work, because the quant core is shared and most models are "Llama + (MoE or a norm tweak)".
The genuinely new attention (MLA, DeltaNet, Mamba) is where the engineering is — prioritize by your usage.
