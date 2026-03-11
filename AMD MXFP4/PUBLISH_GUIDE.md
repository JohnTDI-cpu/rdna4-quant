# Gdzie i jak opublikować rdna4-quant

---

## 1. GitHub (PIERWSZY KROK)

Utwórz repo: `github.com/TWOJ_USERNAME/rdna4-quant`

```bash
cd "/home/janusz/AMD MXFP4"
# Repo jest już zainicjalizowane, wystarczy:
git remote add origin https://github.com/TWOJ_USERNAME/rdna4-quant.git
git push -u origin main
```

---

## 2. Reddit r/LocalLLaMA (NAJWAŻNIEJSZY, ~500k+ subscribers)

**Tytuł:**
```
Custom INT4 engine for AMD RDNA4 — 61 t/s decode on Qwen3-14B, beats llama.cpp Q4_K_M by 28% (open source)
```

**Body:**
```markdown
I built a custom INT4 quantization + inference engine from scratch for consumer AMD GPUs
(RDNA3/4). It runs entirely on a single AMD Radeon AI PRO R9700 ($600 GPU).

## Results vs llama.cpp GGUF Q4_K_M (same GPU, same model)

| Metric | My Engine | llama.cpp Q4_K_M | Difference |
|---|---|---|---|
| **Decode** (ctx=128) | **61 t/s** | 47.7 t/s | +28% faster |
| **Decode** (ctx=2048) | **57.5 t/s** | 41.0 t/s | +40% faster |
| **Prefill** (pp512) | **2,076 t/s** | 559 t/s | 3.7x faster |
| **PPL** (WikiText-2) | **7.692** | 7.715 | Better quality |
| **ARC-Challenge** | **92.8%** | 90.8% | +2.0% |
| **MMLU** | **75.6%** | 72.8% | +2.8% |
| Context scaling (128→2048) | -6% | -14% | Degrades less |
| VRAM | 9.9 GB | 8.8 GB | ~1 GB more |

## What's different from llama.cpp?

1. **INT4 asymmetric + Hadamard rotation + GPTQ** — the Hadamard rotation spreads
   weight outliers evenly before quantization. Combined with GPTQ calibration,
   this achieves lower PPL than GGUF Q4_K_M despite being simpler 4-bit format.

2. **Custom HIP GEMV kernels** — hand-written for RDNA4 with fused INT4 dequant,
   128-bit vectorized loads, and inline Hadamard transform (FWHT in 32 registers).

3. **Zero Python overhead** — entire 40-layer decode loop runs in C++ (HIP).
   Eliminates ~2ms per token Python overhead.

4. **FP8 KV cache** — 1 byte per element instead of 2, so context scaling degrades
   only 6% from ctx=128 to ctx=2048 (vs 14% for GGUF).

## How to use

```bash
git clone https://github.com/TWOJ_USERNAME/rdna4-quant
cd rdna4-quant
pip install -r requirements.txt
cd hip_int4 && python setup.py build_ext --inplace && cd ..

# Quantize (~30 min)
python quantize_v4_gptq.py --model Qwen/Qwen3-14B

# Chat
python int4_engine_v5.py --chat
```

## Interesting findings

- **Scale format matters more than data format**: FP16 scales with INT4 data (PPL 7.69)
  beats E4M3 scales with FP4 data (PPL 8.7) and E8M0 scales (PPL 9.1).
  GGUF Q4_K_M also uses FP16-class scales — this is why it's good.

- **We're at 88% of peak HBM bandwidth** — the engine is near-optimal for single-GPU.
  To go faster, you'd need tensor parallelism or speculative decoding.

- **RDNA4 WMMA lane mapping is undocumented** — we reverse-engineered it and documented it.
  See DISCOVERY_GFX12_WMMA_OUTPUT.md if you're writing GPU kernels for gfx12.

**GitHub:** https://github.com/TWOJ_USERNAME/rdna4-quant

Happy to answer questions. This is MIT licensed — use it however you want.
```

**Kiedy postować:** Wtorek-Czwartek, 14:00-17:00 UTC (rano w USA)

---

## 3. Reddit r/AMD (~1.2M subscribers)

**Tytuł:**
```
Built a custom LLM inference engine for RDNA4 — 61 t/s decode on 14B model, 28% faster than llama.cpp [open source]
```

**Body:** Skrócona wersja postu z r/LocalLLaMA. Fokus na:
- "Consumer AMD GPU achieving competitive performance"
- "No NVIDIA required"
- Prosta instrukcja instalacji
- Link do GitHub

---

## 4. Hacker News — "Show HN"

**Tytuł:**
```
Show HN: INT4 LLM engine for AMD Radeon – beats llama.cpp by 28% with custom HIP kernels
```

**Body (krótki, techniczny):**
```
I wrote a production INT4 quantization + inference engine for consumer AMD GPUs (RDNA3/4).

Key results on Qwen3-14B with AMD R9700:
- 61 t/s decode (vs 47.7 for llama.cpp, +28%)
- 2076 t/s prefill (vs 559, 3.7x)
- PPL 7.692 (vs 7.715, slightly better quality)

The secret sauce: INT4 asymmetric quantization with block-diagonal Hadamard
rotation + GPTQ error compensation, served by custom HIP GEMV kernels with
fused dequant and Fast Walsh-Hadamard Transform in registers.

We also reverse-engineered the RDNA4 WMMA lane mapping (AMD doesn't document it).

GitHub: https://github.com/TWOJ_USERNAME/rdna4-quant
```

---

## 5. ROCm GitHub Discussions

Wejdź na: https://github.com/ROCm/ROCm/discussions

**Tytuł:**
```
Custom INT4 LLM inference engine for RDNA4 (gfx1201) — 61 t/s, open source
```

Fokus: techniczny, wspomnieć o WMMA discovery, poprosić o feedback od AMD engineers.

---

## 6. X/Twitter

```
Built a custom INT4 LLM engine for AMD Radeon RDNA4:

61 t/s decode (vs 47.7 llama.cpp, +28%)
2076 t/s prefill (3.7x faster)
PPL 7.69 (beats GGUF Q4_K_M)

Custom HIP kernels + Hadamard rotation + GPTQ
Open source: github.com/TWOJ_USERNAME/rdna4-quant

@AMDRadeon @ROCmSoftware
```

---

## 7. HuggingFace Model Card

Upload skwantyzowanych wag jako model:
- `TWOJ_USERNAME/Qwen3-14B-INT4-Hadamard-GPTQ`
- Model card z benchmarkami
- Link do repozytorium

```bash
# Upload za pomocą huggingface-cli:
pip install huggingface_hub
huggingface-cli upload TWOJ_USERNAME/Qwen3-14B-INT4-Hadamard-GPTQ ./quantized_v4_gptq/
```

---

## 8. Opcjonalne

| Forum | Kiedy | Uwagi |
|-------|-------|-------|
| r/RDNA4 | Po r/LocalLLaMA | Mały subreddit, ale targetowany |
| LocalLLaMA Discord | Po Reddit | Crosspost link do Reddit |
| dev.to / Medium | Tydzień później | Bardziej tutorial-style blog post |
| arxiv | Jeśli chcesz formalny paper | Technical report format |

---

## Kolejność publikacji

1. **Dzień 1:** GitHub repo (musi być gotowe zanim cokolwiek postujesz)
2. **Dzień 1:** HuggingFace model card z wagami
3. **Dzień 2:** Reddit r/LocalLLaMA (rano USA)
4. **Dzień 2:** Hacker News "Show HN"
5. **Dzień 3:** Reddit r/AMD
6. **Dzień 3:** ROCm GitHub Discussions
7. **Dzień 3:** X/Twitter
8. **Tydzień 2:** Blog post (dev.to/Medium) — bardziej tutorial
