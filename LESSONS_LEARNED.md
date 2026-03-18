# Lessons Learned — AMD RDNA4 INT4 Quantization Research

Dokument opisuje wszystkie metody które testowaliśmy przy kwantyzacji modeli LLM.
Powstał żeby nie powtarzać tych samych eksperymentów przy następnych modelach.

**Modele:** Qwen3-14B (dense), Qwen3-30B-A3B (MoE), Qwen3.5-27B (dense, hybrid DeltaNet+FullAttn)
**GPU:** AMD Radeon AI PRO R9700 (RDNA4, gfx1201, 32 GB, 640 GB/s peak / ~507 GB/s effective)

---

## Chronologia podejść (od najgorszego do najlepszego)

### 1. MXFP4 E2M1 + E8M0 scales — PORZUCONE ❌

**Co to:** FP4 format E2M1 (8 nieuniformowych poziomów), skale per-blok w formacie E8M0 (potęgi 2), blok=32.

**Wyniki PPL (WikiText-2, sliding window, ctx=2048):**
| Wariant | PPL |
|---------|-----|
| RTN (round-to-nearest) | 9.8102 |
| MSE scale search | 9.6680 |
| + GPTQ | 9.5137 |
| + Hadamard + GPTQ | 9.1405 |
| FP16 baseline (nasz engine) | 8.6209 |
| GGUF Q4_K_M baseline | 7.5423 |

**Dlaczego porzucone:**
- MXFP4 E2M1 ma tylko 8 nieuniformowych poziomów kwantyzacji
- E8M0 skale (format power-of-2, 0-bit mantysa) są zbyt zgrubne
- Nawet GPTQ + Hadamard nie może zniwelować błędu formatu
- PPL 9.14 to +22% vs GGUF — nie do przyjęcia
- **INT4 symmetric ma 16 uniformowych poziomów i FP16 skale (11-bit mantysa) — jest obiektywnie lepszy**

---

### 2. NVFP4 E2M1 + E4M3 scales + AWQ — PORZUCONE ❌

**Co to:** Ulepszona wersja MXFP4: blok=16 (zamiast 32), skale E4M3 (3-bit mantysa, znacznie lepsze niż E8M0), MSE-optimal scale search, 4/6 adaptive scaling (MIT Han Lab), RaZeR (remap -0 → ~5.0). Bez GPTQ (badania pokazały +34.6% degradację z GPTQ dla NVFP4).

**Wyniki speed:**
- Decode: ~41 t/s (GORSZE niż GGUF: 47 t/s!)
- Prefill 1024: 2517 t/s
- VRAM: ~9 GB

**Dlaczego porzucone:**
- Decode speed 41 t/s gorszy od GGUF 47 t/s — porażka
- Jakość też nie lepsza od GGUF
- Brak GPTQ oznacza wyższy błąd kwantyzacji
- Format FP4 E2M1 nadal limituje (8 poziomów) mimo lepszych skal
- **Wniosek: AWQ bez GPTQ nie działa dobrze. Na AMD lepiej użyć INT4 + GPTQ.**

---

### 3. Learned Rotation (SPSA na Stiefel Manifold) — PORZUCONE ❌

**Co to:** Zamiast stałej rotacji Hadamarda — uczenie optymalnej ortogonalnej macierzy rotacji per-warstwa, metodą SPSA (gradient-free, finite difference) z Cayley retraction na manifoldzie Stiefel.

**Wyniki PPL (MXFP4 base):**
| Wariant | PPL |
|---------|-----|
| Fixed Hadamard + GPTQ | 9.1405 |
| Learned rotation 150 steps MSE | 9.2313 |
| Learned rotation 500 steps seq MSE | 9.5780 |
| Learned rotation + GPTQ hybrid | 9.0667 |

**Dlaczego porzucone:**
- Poprawa vs stały Hadamard: tylko **0.074 PPL** (0.8%)
- Koszt obliczeniowy: ~10x dłużej niż Hadamard
- STE (Straight-Through Estimator) daje zero gradientów dla ||Q(WR) - WR||² — stąd potrzeba SPSA
- 500 kroków dało gorszy wynik niż 150 (overfitting do kalibracji)
- **Wniosek: Stały Hadamard jest 99% tak dobry jak learned rotation. Nie warto.**

---

### 4. Mixed Precision v1 (symmetric INT4) — POŚREDNI ETAP ⚠️

**Co to:** quantize_mixed_precision.py. Mapa sensitivity → top 20% najwrażliwszych wag → INT8 + GPTQ, reszta → INT4 symmetric + GPTQ + Hadamard.

**Problem:** Używał starego `int4_quant.py` (symmetric INT4 = zakres [-8..+7]). Asymmetric jest lepszy.

**Wniosek:** Idea mixed precision jest dobra (używana w v4), ale implementation trzeba było przepisać na asymmetric INT4.

---

### 5. Stiefel Mixed Precision — PORZUCONE ❌

**Co to:** quantize_mixed_stiefel.py. Kombinacja Stiefel rotation (per-blok) + mixed INT4/INT8. Najcięższy obliczeniowo wariant.

**Dlaczego porzucone:**
- Per-block Stiefel rotation: ogromny koszt obliczeniowy (godziny vs minuty dla Hadamarda)
- Marginalna poprawa vs stały Hadamard (tak jak samodzielna learned rotation)
- **Wniosek: Per-block rotation nie warta zachodu. Jedna globalna Hadamard per layer wystarczy.**

---

### 6. INT4 Symmetric (v1) — POŚREDNI ETAP ⚠️

**Co to:** int4_quant.py + quantize_int4_gptq.py. Symmetric INT4: zakres [-8..+7], FP16 skale per blok.
Następnie absorb_hadamard.py: absorbowanie rotacji Hadamarda w wagi (eliminuje runtime rotation).

**Problem:** Symmetric INT4 gorzej radzi sobie z niesymetrycznym rozkładem wag.

**Wniosek dobry:** Absorbowanie rotacji w wagi (W_absorbed = W_rot @ H) eliminuje runtime overhead. Używamy tego w v4/v5.

---

### 7. INT4 Asymmetric v2 + GPTQ + Hadamard — ZWYCIĘZCA ✅

**Co to:** int4_quant_v2.py (asymmetric: zero-point offset, zakres [0..15] ze scale+offset) + GPTQ Cholesky + Hadamard rotation.

**Ewolucja:**
- quantize_v2.py: pierwsza wersja asymmetric + GPTQ
- quantize_v4_gptq.py: mixed INT4/INT8 (sensitivity map → top 20% INT8)
- quantize_v5_pure_int4.py: pure INT4 (mniejszy rozmiar, minimalnie gorszy PPL)

**Wyniki końcowe:**
| Model | PPL | VRAM | Decode | ARC | MMLU |
|-------|-----|------|--------|-----|------|
| **v4 Mixed INT4/INT8** | **7.692** | 10.5 GB | 61 t/s | 92.8% | 75.6% |
| **v5 Pure INT4** | **7.787** | 8.5 GB | 62.7 t/s | 91.2% | 74.3% |
| GGUF Q4_K_M | 7.657 | 8.8 GB | 47 t/s | 90.8% | 72.8% |
| FP16 (nasz engine) | 8.621 | — | — | — | — |

**Kluczowe wnioski:**
- Asymmetric INT4 > symmetric INT4 dla modeli z niesymetrycznym rozkładem wag
- GPTQ Hessian compensation niezbędny do dobrej jakości
- Hadamard rotation redukuje outliers, wyrównuje rozkład wag
- Mixed precision (top 20% wrażliwych warstw w INT8): +0.1 PPL przy +20% VRAM
- Nasz v4 BIJE GGUF Q4_K_M zarówno jakością jak i szybkością

---

## Kluczowe technikalia

### Ocena jakości (PPL)
- **Zawsze używaj sliding window**: stride=512, ctx=2048, loss tylko na ostatnich 512 tokenach
- To samo co llama.cpp i lm-eval-harness — inaczej wyniki nieporównywalne
- Nasz engine FP16 daje PPL=8.62, GGUF FP16 daje PPL=7.49 — **różne preprocessing/tokenizer** nie porównuj bezpośrednio
- Do wiarygodnego porównania z GGUF: uruchom oba na tym samym tescie

### Wrażliwość warstw (sensitivity map)
- Warstwy 0-2 i ostatnie 2-3 są najbardziej wrażliwe (embed, lm_head, early/final layers)
- gate/up projekcje w FFN mniej wrażliwe niż q/k/v/o w attention
- sensitivity_map.json zawiera Hessian traces dla wszystkich 40 warstw Qwen3-14B
- **Ten map jest specyficzny dla Qwen3-14B** — dla nowego modelu trzeba zmierzyć od nowa (measure_sensitivity.py)

### GPU arch detection
- Zawsze auto-detect: env var → PyTorch CUDA props → rocminfo → fallback do gfx1201
- Nigdy hardcode --offload-arch w setup.py
- gfx1201 = RDNA4 (RX 9700 / R9700 PRO), gfx1100 = RDNA3 (RX 7900)

### NVFP4 na AMD — dlaczego NIE
- NVIDIA NVFP4 jest natywny tylko na Blackwell (B100/B200)
- Na AMD brak natywnych NVFP4 tensorcore instrukcji
- Triton/HIP emulacja NVFP4 jest wolna
- **INT4 WMMA (wave matrix multiply) jest natywny na gfx1201** → używaj INT4

### Calibration data
- 512 samples × 512 tokens = 262K tokenów — wystarczy dla Qwen3-14B
- Większy dataset (1024×512) marginalnie lepszy, nie wart 2x czasu
- WikiText-2 jako dane kalibracyjne jest OK

---

## 8. MoE: Qwen3-30B-A3B (INT4 g64 + Hadamard + GPTQ) — W TRAKCIE

**Model:** 48 warstw, hidden=2048, 128 ekspertów/warstwa, top-8, moe_inter=768, shared expert
**Kwantyzacja:** INT4 asymmetric g64 + Hadamard + GPTQ (quantize_moe_gptq.py)
**Engine:** int4_engine_moe.py + rozszerzony int4_decode_step.hip

### Wyniki vs GGUF Q4_K_M (Vulkan na R9700)

**Uwaga:** GGUF Vulkan baseline 174.6 t/s zmierzony przed aktualizacją Mesa 25.2.8, która zepsuła RADV
(raportuje warp_size=64 zamiast 32 dla RDNA4 gfx1201). Do czasu naprawy RADV/llama.cpp, GGUF
Vulkan daje 8-47 t/s — nie jest porównywalny.

**Decode — aktualny stan (pomiar 2026-03-16, GPU 0 = pierwszy R9700):**

| Ścieżka | CTX=128 | CTX=512 | CTX=1024 |
|---------|---------|---------|----------|
| Kernel launch (bulk) | 170.6 t/s | 165.8 t/s | 160.4 t/s |
| HIP Graph (bulk) | **176.7 t/s** | **172.2 t/s** | **165.6 t/s** |
| C++ Graph replay (200iter) | **175.9 t/s** | — | — |
| GGUF Q4_K_M (stary baseline) | 174.6 t/s | 174.9 t/s | 174.9 t/s |

**Bijemy stary GGUF baseline przy CTX=128 o +1.2% (176.7 vs 174.6 t/s).**
Przy CTX=512 gap zmniejszony do -1.5% (172.2 vs 174.9 t/s).

**Prefill (C++ prefill z GPU-only routing + K-outer fused INT4 GEMM):**

| PP | Nasz INT4 | GGUF Q4_K_M (stary) | Różnica |
|----|----------|----------------------|---------|
| 128 | 872 t/s | **1,302 t/s** | -33% |
| 256 | 934 t/s | **2,103 t/s** | -56% |
| 512 | 993 t/s | **2,992 t/s** | -67% |

| Metryka | Nasz INT4 | GGUF Q4_K_M | Różnica |
|---------|-----------|-------------|---------|
| MMLU | **80.1%** | 78.9% | +1.2pp |
| VRAM (model) | 18,552 MB | ~18,000 MB | similar |

**Kluczowe optymalizacje decode (łącznie +14% vs stary kernel, +3.5% vs poprzedni graph):**
1. **Fused resnorm+FWHT+router GEMV** — 1 dispatch zamiast 2, każdy blok redundantnie liczy resnorm (D=2048 mieści się w L2). Dodane do WSZYSTKICH ścieżek (produkcja, graph, persistent graph).
2. **block_s=32 w flash decode** — więcej mniejszych bloków = lepszy load balancing na 32 CU × 2 SIMD. Eliminuje dyskontynuację na granicy ctx=256.
3. **Partial+reduce flash decode (BEZ selfreduce)** — selfreduce używa __threadfence (~1µs × 960 per token = ~1ms overhead na RDNA4). Partial+reduce nie ma fence.
4. **HIP Graph capture+replay** — eliminuje dispatch overhead (~3-5µs × ~480 kerneli/tok). Daje +3.5% vs kernel launch.

**Kluczowa obserwacja**: warmup GPU ma ogromny wpływ. Bez warmup: 139-167 t/s. Z 5× warmup (20 tokenów): 170 t/s. Z 200-iter bench: 175.9 t/s. Pełne nasycenie memory controller wymaga ~50 iteracji.

### Co zadziałało w MoE ✅

1. **GPTQ + Hadamard dla ekspertów MoE** — Jakość MMLU 80.1% (vs 78.9% GGUF). Metoda kwantyzacji z dense modelu przenosi się dobrze na MoE.
2. **Stacked tensor format** — `[num_experts, 2*moe_inter, hidden//2]` uint8. Zero-copy slicing per expert, contiguous w pamięci. Kluczowe dla wydajności.
3. **Fused router GEMV + softmax + topk** — `router_gemv_softmax_topk` kernel eliminuje 1 launch per layer. Atomic last-block-wins pattern.
4. **Combined attention kernel** — `flash_decode_combined_fp16` łączy partial + reduce w 1 launch (zamiast 2). Grid=(num_heads=32), shared memory reduction.
5. **Group size g64 zamiast g32** — Mniej metadanych (scales/zeros), mniejszy model na dysku. Transposed scales format `[K/64, N, 2]` dla lepszego coalescing.
6. **DN GEMV BLOCK_N=4 zamiast 2** — Zmniejsza liczbę bloków z 8192 do 4096, mniej kernel launch overhead.

7. **`__launch_bounds__(32, 2)` na DN GEMV** — 149.6 → 154.0 t/s (+4 t/s, +3%). Zmniejsza VGPRs z 140 → 92 (4 wave/SIMD zamiast 2). Działa bo DN ma BLOCK_N=2 (mniej rejestrów per wątek), kompilator może zmieścić się w 92. Na GU (BLOCK_N=4) ta sama sztuczka nie działa — za dużo danych per wątek.

8. **Cooperative kernels na gfx1201** — `hipLaunchCooperativeKernel` + `cooperative_groups::this_grid().sync()` działa poprawnie. Test (test_coop4.hip): 40-layer pipeline: individual=8.537ms vs persistent=8.297ms (2.8% savings, ~3µs per transition). Potencjał: mega-kernel eliminujący ~580 launchów/token.

9. **Self-reducing flash decode (`flash_decode_selfreduce_fp16`)** — Partial + reduce w 1 dispatch zamiast 2. Atomic last-block-reduces pattern: każdy blok pisze partials, `__threadfence()`, `atomicAdd(counter)`. Ostatni blok (counter == n_splits-1) robi redukcję + FWHT inline, resetuje counter. Bezpieczne bo HIP stream ordering gwarantuje sekwencyjne wykonanie kerneli per warstwa. Oszczędza 1 dispatch/warstwa × 48 warstw = 96 dispatchów. Zysk: ~3.5 t/s na C++ loop, ~6 t/s na graph path.

10. **Fused resnorm+FWHT+router GEMV (`fused_resnorm_router_gemv`)** — Każdy blok (32 wątków) redundantnie liczy: residual A+B → RMSNorm → FWHT + router GEMV(normed, gate_W). D=2048 mieści się w L2 (4 MB), więc redundantne odczyty są tanie. Bloki FWHT output dzielą zapis równomiernie (distributed writes). Oszczędza 1 dispatch/warstwa. Zysk: ~2 t/s.

11. **block_s=32 zawsze w flash decode** — Eliminuje dyskontynuację 128→256 ctx (block_s 32→64 boundary). Analiza: przy block_s=32, n_splits rośnie wolniej, a więcej mniejszych bloków lepiej balansuje się na 32 CU × 2 SIMD. Przy CTX=256 z block_s=64: 20 bloków × 64 iter = wolne. Z block_s=32: 36 bloków × 32 iter = 2× mniej pracy per blok, te same fale. Zysk na CTX=256: **159.7 → 175.1 t/s (+9.6%)**, CTX=1024: **155.4 → 167.1 t/s (+7.5%)**.

12. **C++ prefill z GPU-only routing** — Eliminacja CPU sync (`expert_counts.cpu()`) i pre-alokacja buforów. Routing robi GPU-only: `topk → sort → scatter_add → cumsum` na GPU. Launchuje E=128 ekspertów (puste early-exit w kernelu). Pre-allocated: `rot_buf`, `moe_out`, `qkv_out`, `router_out`, `ones_buf`, `flat_token`, `d_expert_ids_all`. Zysk vs Python prefill: **116 → 872 t/s** (+651%, pp128). Vs GGUF: nadal -33%.

13. **K-outer fused INT4 GEMM (`gemm_int4_g64_ts_fused<4>`)** — Dla prefill expert GU/DN: K-outer loop, M-inner TILE_M=4 loop. Weights loaded ONCE per K-chunk, reused across 4 tokens. Redukuje bandwidth o ~4× vs M-outer (where weights re-read for each token). Zysk: pp512: **906 → 993 t/s (+10%)**.

### Co NIE zadziałało w MoE ❌

1. **Multi-warp GU GEMV (`gemv_mw_batch_g64<2>`)** — 37.6µs vs 36.6µs (WOLNIEJ). VGPR pressure przy K=2048 z 4 warpami. Mały K w MoE (2048 vs 5120 w dense) = za mało pracy per warp.

2. **Fused GU+SiLU+FWHT kernel (`gemv_batch_g64_fused_silu`)** — 155 t/s vs 162 t/s (7 t/s WOLNIEJ!). `hipMemsetAsync` (counter reset per expert) + `__threadfence` w każdym bloku kosztuje więcej niż zaoszczędzony 1 kernel launch. Atomic counter pattern + fence per block = za dużo overhead dla małych macierzy MoE.

3. **HIP Graph replay** — 133 t/s vs 150 t/s (WOLNIEJ!). Testowane 2026-03-14 na ROCm. HIP Graph capture+replay na AMD gfx1201 dodaje overhead zamiast go usuwać. Prawdopodobnie immaturny driver/runtime. **GGUF Vulkan używa pre-recorded command buffers (natywne Vulkan) które mają near-zero dispatch overhead — to inna technologia niż HIP Graph.**

4. **Fused router softmax+topk (single-block, 128 threads)** — 140 t/s vs 154 t/s (WOLNIEJ). `fused_router_softmax_topk` — 1 blok × 128 wątków serializuje router GEMV na 1 CU. Oryginał używa 32 bloków (1 per CU) dla GEMV. Oszczędność 1 launcha (~2µs) nie równoważy straty bandwidth.

5. **Fused router GEMV+topk (multi-block, atomic last-block-wins)** — 148 t/s vs 154 t/s (WOLNIEJ). `fp16_gemv_topk<4>` — 32 bloków GEMV + atomicAdd counter + `__threadfence` + softmax+topk w ostatnim bloku. Overhead: threadfence per blok + hipMemsetAsync counter per warstwa × 48 = więcej niż oszczędzone 48 launchów.

6. **GU GEMV BLOCK_N=2 zamiast 4** — 140 t/s vs 154 t/s (WOLNIEJ). Zmniejsza VGPRs (140→92) ale podwaja liczbę bloków (3072→6144), co zwiększa dispatch overhead i redundantne odczyty X. Przy bandwidth-bound workload dodatkowe occupancy nie pomaga.

7. **`__launch_bounds__(32, 2)` na GU GEMV** — BEZ EFEKTU. Kompilator i tak alokuje 140 VGPRs dla BLOCK_N=4 — nie da się zmieścić poniżej 128 bez poważnego spilling.

8. **FP16 KV cache** — Zabija wydajność przy długim kontekście. 96 KB/token vs ~48 KB w GGUF Q8. Przy ctx=4096 nasz decode spada do 67 t/s (GGUF: 155 t/s).

9. **FP8 E4M3 KV cache** — Testowane 2 podejścia, oba WOLNIEJSZE niż FP16:
   - **(a) Hardware `dot4_fp8_fp8` z non-coalesced byte loads:** Attn=29.2µs vs FP16 25.0µs (+17%). Problem: `dot4_fp8_fp8` wymaga 4 bajtów w 4 oddzielnych pozycjach pamięci (stride=head_dim), co powoduje non-coalesced byte loads. RDNA4 ma 128-byte cache lines, więc 4 byte loads z stride=128B to 4 cache-line fetche zamiast 1.
   - **(b) Coalesced scalar `fp8→float` konwersja:** Attn=37.3µs (+49% vs FP16). Problem: `fp8e4m3_to_f32()` ALU overhead — 128 konwersji per pozycja (branch + shift + compose per value). Coalesced loads ale za dużo ALU.
   - **Wniosek: Przy D=128 i wave32, FP8 KV nie daje zysku. Dane per-pozycja (128×1B=128B) to zaledwie 2 cache lines — oszczędność bandwidth jest znikoma vs overhead konwersji. FP8 KV miałby sens przy D≥512 lub batch>1. Zostawiamy FP16 KV.**

10. **Python prefill z pętlą per-expert** — 106 t/s przy pp128 vs 1451 t/s GGUF. Dequantyzacja on-the-fly + torch.matmul per expert w Pythonie jest tragicznie wolna. **Rozwiązane**: C++ prefill z GPU-only routing + fused kernel (→ 872 t/s).

11. **C++ prefill — NAPRAWIONE** ✅ — Wcześniej produkowało garbage (bug layout). Naprawione: FP16 `at::mm_out` dla QKV/O proj, fused INT4 `gemm_int4_g64_ts_fused<4>` K-outer kernel dla GU/DN, GPU-only routing (sort+scatter_add+cumsum bez CPU sync). KL divergence vs Python reference: 0.000015 (prawidłowe).

12. **WMMA INT4 dla decode** — NIE pomaga. Decode (batch=1 GEMV) jest bandwidth-bound, nie compute-bound. WMMA daje 4x compute ale decode nie potrzebuje więcej compute. WMMA ma sens TYLKO dla prefill (GEMM, batch>1).

13. **Fused ResNorm+Router GEMV (`fused_resnorm_router_gemv<4>`)** — 141-143 t/s vs 150 t/s (WOLNIEJ). Testowano 2026-03-15.
   - Kernel: 32 bloków × 32 wątków, single-pass trick (`rrms` factored out).
   - Oszczędza 1 dispatch per warstwa (ResNorm+FWHT + Router GEMV → 1 kernel).
   - **Root cause regresji:** Każdy z 32 bloków redundantnie liczy rrms (redukuje D=2048 elementów). Blok 0 dodatkowo pisze ResOut (D elementów) + FWHT z 32 wątkami (zamiast 1024 w oryginale). Load imbalance + redundant compute > zaoszczędzony 1 dispatch (~2µs).
   - **Wniosek:** Fusion oszczędzający 1 dispatch nie warta gdy wymaga redundant reduction lub load imbalance. Lepiej zostawić oddzielne kernele.

14. **W4A4 V_DOT8 (`__builtin_amdgcn_udot8`) dla decode expert GEMV** — 14% REGRESJA (151 t/s vs 176 t/s). Testowano 2026-03-15.
   - **Izolowany test (test_vdot8_batched.hip):** GU FP32=20.81µs → V_DOT8=17.46µs (16% szybciej), ALE quantize overhead=2.85µs. Netto: ±0%.
   - **Integracja produkcyjna:** GU+DN = 73.6µs (z W4A4+quantize) vs ~35µs (FP32). 2× wolniej!
   - **Root cause:** FP32 expert GEMV przy 140 VGPRs (10 waves/SIMD) już osiąga ~95% peak memory bandwidth. Więcej occupancy (V_DOT8: 28 VGPRs, 16 waves) NIE pomaga bo problem jest bandwidth-bound, nie compute-bound. Dodatkowy ruch pamięci (w_sum reads, quantize scratch writes/reads) tylko dodaje overhead.
   - **VGPR comparison (z .so):** FP32 GU/DN=140 VGPRs, W4A4 GU/DN=28 VGPRs, quantize_u4_v2=13 VGPRs.
   - **Wniosek: W4A4 (kwantyzacja aktywacji do INT4 + V_DOT8) jest DEAD END dla decode. Bandwidth-bound workload nie zyskuje na lepszym occupancy. Jedyne co pomaga to redukcja ilości danych do przeczytania (lepsze BPW) lub szybsza pamięć.**

15. **Single-dispatch flash decode (`flash_decode_single_fp16`)** — 123 t/s vs 150 t/s (18% WOLNIEJ). Testowano 2026-03-15.
   - Kernel: grid=(num_heads=32), block=(32). Jeden warp per głowa, online softmax iterujący po wszystkich pozycjach, fused FWHT.
   - Cel: zamiana 2 dispatchów (partial+reduce) na 1 dispatch per warstwa (oszczędność ~48×2µs = ~96µs).
   - **Root cause regresji:** 32 warp-sized bloków = 32 wave occupancy na 32 CU GPU. Oryginał: GQA-grouped split-K z `dim3(num_kv_heads=4, n_splits) × dim3(32, gqa_ratio=8)` = 16+ bloków × 8 warpów z L1 cache sharing K/V reads. Nowy kernel nie może korzystać z GQA grouping ani split-K.
   - **Wniosek:** Flash decode wymaga split-K + GQA grouping na RDNA4. Jeden warp per głowa jest zbyt mało parallelism.

16. **Fused ResNorm+Router+TopK mega-kernel (`fused_resnorm_router_topk_mega`)** — 146 t/s vs 150 t/s (WOLNIEJ). Testowano 2026-03-15.
   - 1 blok × 1024 wątków. Fuses ResNorm+FWHT + Router GEMV + Softmax + TopK.
   - **Root cause:** 1 CU bandwidth ogranicza router GEMV (128 experts × 2048 × 2B = 512 KB, ale 1 CU BW ~18 GB/s vs 32 CU łącznie ~576 GB/s).
   - **Wniosek:** Single-block fusion NIE działa dla compute-intensive kroków. Tylko dla bardzo lekkich operacji.

17. **Fused ResNorm+Router+TopK multi-block (`fused_resnorm_router_topk<4>`)** — 137-139 t/s vs 150 t/s (DUŻO WOLNIEJ). Testowano 2026-03-15.
   - Multi-block z `__threadfence()` + atomic last-block-wins pattern.
   - **Root cause:** `__threadfence()` jest BARDZO drogi na RDNA4. Kosztuje ~3-5µs per invocation × 32 bloków × 48 warstw = massive overhead.
   - **Wniosek: `__threadfence()` jest zabójczy na gfx1201. Nigdy nie używać w hot-path kernelach.**

18. **Cooperative grid mega-kernel (`decode_generate_moe_coop`)** — 129.6 t/s vs 150 t/s (13% WOLNIEJ). Testowano 2026-03-15.
   - `hipLaunchCooperativeKernel` + `cooperative_groups::this_grid().sync()` w jednym mega-kernelu.
   - **Root cause:** `grid.sync()` implementowany przez atomics na device memory — tak samo drogi jak `__threadfence()`.
   - **Wniosek:** Cooperative grids NIE są odpowiednim rozwiązaniem dla dispatch overhead na RDNA4.

19. **Persistent mega-kernel z atomic barriers (`decode_generate_moe_persistent`)** — 106.7 t/s vs 150 t/s (29% WOLNIEJ). Testowano 2026-03-15.
   - Persistent kernel z atomicAdd barrier synchronization.
   - **Root cause:** Wielokrotne `__threadfence()` + atomic barriers = najgorszy wariant. Synchronizacja między blokami jest fundamentalnie droga na RDNA4.
   - **Wniosek:** Persistent kernels z atomic sync = dead end na RDNA4. HIP dispatch (~1.7µs) jest tańszy niż JAKAKOLWIEK forma inter-block sync.

20. **Warp-level GEMV dla QKV/O (`gemv_warp_rm_g64<2>`)** — 150 t/s (BEZ ZMIANY). Testowano 2026-03-15.
   - Zamiana `gemv_multiwave_rm_g64<2>` (128 threads, 4 warps/block) na `gemv_warp_rm_g64<2>` (32 threads, 1 warp/block, 4× więcej bloków).
   - Cel: więcej bloków = więcej parallelism = lepsze occupancy.
   - **Wynik:** Identyczna prędkość. Multiwave kernel (4 warpy współdzielą X read) vs warp kernel (każdy blok czyta X niezależnie) — tradeoff jest neutralny dla tych rozmiarów macierzy.
   - **Wniosek:** Dla GEMV z K=2048-7168, zarówno 4-warp/block jak i 1-warp/block dają ten sam wynik.

21. **NT (non-temporal) cache hints na expert GEMVs** — BEZ EFEKTU. Testowano 2026-03-15.
   - `__builtin_nontemporal_load` na weight reads w GU/DN GEMV kernelach.
   - **Wynik:** Identyczna prędkość. RDNA4 cache controller już efektywnie zarządza eviction policy.
   - **Wniosek:** NT hints nie pomagają na gfx1201 dla streaming weight reads w GEMV.

22. **GU multi-warp RPW=1 (`gemv_mw_batch_g64<1>`)** — 150 → 158 t/s (+5.3%) ✅. Testowano 2026-03-15.
   - Zamiana `gemv_warp_batch_g64<4>` (32 thr/block, RPW=4) na `gemv_mw_batch_g64<1>` (128 thr/block, 4 warpy, RPW=1) dla GU GEMV we WSZYSTKICH 7 launch sites.
   - RPW=1 = 1 row per warp, 4 warpy/block = 4 rows/block. Więcej bloków = więcej wave occupancy = lepsze ukrywanie latencji pamięci.
   - 4 warpy współdzielą odczyt X (input) z LDS — redukuje bandwidth.
   - **Wniosek:** RPW=1 multi-warp jest OPTYMALNY dla GU GEMV (K=2048, N=1536).

23. **GU multi-warp RPW=2 (`gemv_mw_batch_g64<2>`)** — 153.5 t/s (WOLNIEJ niż RPW=1). Testowano 2026-03-15.
   - RPW=2 = 2 rows per warp = mniej bloków = mniej parallelism.
   - **Wniosek:** RPW=1 > RPW=2 > RPW=4 dla GU GEMV. Więcej bloków = lepiej.

24. **DN multi-warp RPW=1 (`gemv_mw_batch_xbatch_g64<1>`)** — 155.9 t/s (WOLNIEJ). Testowano 2026-03-15.
   - DN: K=768, nit=24. Tylko 24/32 lanes aktywnych (75% utilization).
   - Multi-warp dodaje overhead (LDS sync, shared X management) bez zysku bo K za małe.
   - **Wniosek:** Multi-warp NIE pomaga dla małych K (K<1024). DN zostawić przy `gemv_warp_batch_xbatch_g64<2>`.

25. **Light fences (`s_wait_storecnt + global_inv`) dla Router+TopK fusion** — 152.6 t/s (WOLNIEJ). Testowano 2026-03-15.
   - Próba fused Router GEMV + TopK z lekkimi fencami RDNA4 zamiast `__threadfence()`.
   - `asm volatile("s_wait_storecnt 0x0" ::: "memory")` + `asm volatile("global_inv" ::: "memory")` zamiast `__threadfence()`.
   - **Wynik:** Nadal ~5µs overhead per invocation (32 bloków × 48 warstw = ~7.7ms!).
   - **Wniosek:** Na gfx1201, KAŻDA forma memory fence (threadfence, s_wait_storecnt+global_inv, cooperative grid.sync) kosztuje ~5µs. NIE MA taniej synchronizacji międzyblokowej.

26. **gfx1201 asm instructions — cheatsheet:**
   - ✅ `global_inv` — invalidate L1 cache (acquire fence)
   - ✅ `s_wait_storecnt 0x0` — wait for all stores to complete (release fence)
   - ✅ `s_waitcnt vmcnt(0)` — wait for vector memory operations
   - ❌ `buffer_gl1_inv` — **NOT supported on gfx1201** (compilation error)
   - Koszt: `global_inv` + `s_wait_storecnt` razem ≈ `__threadfence()` ≈ ~5µs

27. **NT weight loads na QKV/O GEMV** — BEZ EFEKTU. Testowano 2026-03-15.
   - `__builtin_nontemporal_load` na weight reads w `gemv_multiwave_rm_g64` dla QKV i O.
   - 158.8 vs 158.5 t/s — w szumie pomiarowym.
   - **Wniosek:** NT hints nie pomagają na QKV/O (tak samo jak na expert GEMVs, #21).

28. **Transposed scale format `[K/64, N, 2]` dla QKV/O** — REGRESJA (153.7 vs 158.2 t/s, -2.8%). Testowano 2026-03-16.
   - Hipoteza: GU expert GEMV osiąga 70% BW z transposed `[K/64, N, 2]` scales → przenieś na QKV/O.
   - **BŁĘDNA ANALIZA:** Fokus na cross-warp coalescing (różne warpy = różne N → coalescing po N).
   - **PRAWDZIWY PROBLEM — within-warp coalescing:** W wave32 warpu, 32 lanes mają TAKI SAM `n` ale RÓŻNE grupy `g` (via `it = lane`). Z `[N, K/64, 2]` (row-major): stride=4B między grupami → **1 cache line** (128B pokrywa 32 grupy). Z `[K/64, N, 2]` (transposed): stride=N×4B → **16 osobnych cache line fetchy** per warp per iterację.
   - **GU expert 70% BW:** GU osiąga 70% BW NIE dzięki transposed scales, ale z innych powodów (batched dispatch, mniejszy dispatch overhead, wiele ekspertów = więcej bloków).
   - **Wniosek:** `[N, K/64, 2]` (row-major) jest OPTYMALNY dla within-warp coalescing w GEMV. Nigdy nie transponować skali w kernelach gdzie lane=group.

29. **Separate SiLU+FWHT + DN GEMV vs fused `gemv_warp_silu_had_batch_g64`** — ODZYSKANIE 158.5 t/s (z 141.6 t/s w backupie). Testowano 2026-03-16.
   - Backup code używał fused SiLU+FWHT+DN w jednym kernelu (`gemv_warp_silu_had_batch_g64<2>`).
   - Zamiana na 2 oddzielne kernele: `silu_fwht_batch` (K_dn/32 bloków, 32 thr/blk) + `gemv_warp_batch_xbatch_g64<2>` (N_dn/2 bloków × 8 ekspertów, 32 thr/blk, `__launch_bounds__(32, 2)`).
   - **Dlaczego fused był wolniejszy:** Fused kernel łączy SiLU+FWHT (compute-heavy, niskie VGPR) z DN GEMV (bandwidth-bound, wysokie VGPR). Wyższe VGPR w połączeniu → gorsza occupancy → gorsze latency hiding.
   - **Wniosek:** Separation > fusion gdy dwa etapy mają różne profile (compute vs bandwidth, niskie vs wysokie VGPR). `__launch_bounds__(32, 2)` na DN GEMV działa TYLKO gdy jest oddzielny.

30. **Parallel-M prefill kernel (`gemm_int4_g64_ts_batched_par`)** — REGRESJA (531 vs 820 t/s, pp128). Testowano 2026-03-16.
   - Grid: `(N/BN, E, M_max)` — 3D grid z blockIdx.z per token. Cel: M_j=16 tokens parallel zamiast sequential.
   - Grid size: ~6.9M blocks (384×128×141), >50% early-exit. Scheduling overhead overwhelms parallelism gains.
   - **Wniosek:** 3D grid z wieloma early-exit blokami jest gorszy niż sequential M loop. GPU scheduler nie radzi sobie z milionami no-op bloków.

31. **LUT-based INT4 GEMM (`gemm_int4_g64_lut`)** — REGRESJA (694 vs 820 t/s, pp128). Testowano 2026-03-16.
   - 256-entry LDS lookup table: byte → half2, then `__builtin_amdgcn_fdot2` for dot product.
   - **Root cause:** LDS bank conflicts na 256-entry tabeli + `__syncthreads()` overhead. fdot2 oszczędza ALU ale LDS penalty jest większy.
   - **Wniosek:** LUT + fdot2 nie opłaca się dla GEMV-style kerneli. Lepszy scalar FP32 dequant.

32. **Fused K-outer kernel z BN=8 (`gemm_int4_g64_ts_fused<8>`)** — REGRESJA (568 vs 834 t/s, pp128). Testowano 2026-03-16.
   - BN=8 z TILE_M=4: 8 acc × 4 tokens = 32 float accumulators + 8 uint4 weight regs + 32 xv regs ≈ 90+ VGPRs.
   - **Root cause:** Register pressure kills occupancy. Z BN=8+TILE_M=4: <6 waves/SIMD → nie wystarcza do ukrycia memory latency.
   - **Wniosek:** BN > 4 nie pomaga w fused kernel z TILE_M>1. Register budget per warp jest ograniczony na RDNA4 (512 VGPRs/SIMD). BN=4 + TILE_M=4 = 16 acc + 4 uint4 + 32 xv ≈ 60 VGPRs — optimum.

33. **Prefill bottleneck analysis (pp128):** Testowano 2026-03-16.
   - Per-layer breakdown: norm=0.01ms, qkv+attn=0.18ms, route=0.17ms, **GU=1.59ms (54%)**, silu=0.02ms, **DN=0.93ms (31%)**, scatter=0.07ms. TOTAL=2.97ms.
   - **GU+DN = 84% czasu per layer.** Theoretical minimum (bandwidth): 96 MB GU + 48 MB DN = 144 MB at 507 GB/s = 0.28ms. Actual 2.52ms = **9× wolniej**.
   - **Root cause:** Scalar INT4 dequant w inner loop jest ALU-bound mimo niskiej arithmetic intensity (7.75 FLOP/byte < 93.7 crossover). Problem: ~70 ALU ops per 16B weight load (16 nibble extracts + 32 int2float + 32 fmul + 32 fadd). Z M_j=16 sequential tokens, occupancy spada bo warpy są zajęte dłużej.
   - **Wniosek:** Dalszy postęp wymaga WMMA (v_wmma_f16_16x16x16_f16) z fused INT4 dequant lub dequant+rocBLAS batched GEMM. Scalar GEMV jest dead end dla prefill.

### Analiza bottlenecków MoE decode (profiling ctx=128)

```
Per-layer breakdown (x48 layers, bench_moe_profile_kernels.py):
QKV GEMV:      22.4µs  (12.4%)  — 264 GB/s (46% peak)
Head norm+RoPE: 7.5µs  (4.2%)
Flash decode:  18.6µs  (10.3%)
O GEMV:        20.8µs  (11.5%)  — 227 GB/s (39% peak) ← WORST BW
Res+Norm1:      7.8µs  (4.3%)
Router GEMV:    9.8µs  (5.4%)   — 53 GB/s (9% peak, FP16 weights, tiny matrix)
Softmax+TopK:  10.8µs  (6.0%)
Expert GU:     35.2µs  (19.6%)  — 402 GB/s (70% peak) ← BEST BW
SiLU+FWHT:     6.7µs   (3.7%)
Expert DN:     22.6µs  (12.5%)  — 313 GB/s (54% peak)
Res+Norm2:      7.7µs  (4.3%)
LM head:        6.5µs  (3.6%)
TOTAL:        180.2µs  (×48 = 8.65ms → 116 t/s from pure kernel time)
Actual:       150 t/s (6.63ms/tok) — dispatches add ~1ms
```

**Kluczowe obserwacje:**
- GU GEMV (70% BW) jest najefektywniejszy — duże macierze, wiele bloków, dobry streaming
- O GEMV (39% BW) jest najgorszy — za mały K=2048 na occupancy hiding memory latency
- Total GEMV: 5.32ms, 1.554 GB → 292 GB/s (51% peak)
- **Dispatch overhead:** 150 t/s actual vs ~116 t/s pure kernel = overhead jest NEGATYWNY (kernel profiling double-counts event overhead). Realne dispatches: ~580 × 1.7µs ≈ 1.0ms overhead.

### Priorytety optymalizacji MoE (stan na 2026-03-15)

**Obecny wynik:** 158 t/s (C++ loop, ctx=128, po GU RPW=1 opt). **Cel:** ≥177 t/s (GGUF Q4_K_M Vulkan).

**Gap analysis:** 6.32ms/tok → potrzeba 5.65ms/tok. Różnica: ~0.67ms (~10.6%).

**Wszystkie testowane metody eliminacji dispatch overhead ZAWIODŁY:**
- ~~HIP Graph~~ — WOLNIEJSZE (153 t/s)
- ~~Cooperative mega-kernel~~ — WOLNIEJSZE (130 t/s), grid.sync() drogi
- ~~Persistent mega-kernel~~ — WOLNIEJSZE (107 t/s), atomics najgorsze
- ~~Kernel fusion (5 wariantów)~~ — WOLNIEJSZE (137-152 t/s), threadfence/load imbalance/light fence
- ~~Single-dispatch flash decode~~ — WOLNIEJSZE (123 t/s), brak GQA parallelism
- ~~W4A4 V_DOT8~~ — REGRESJA (151 t/s), bandwidth-bound nie zyskuje na compute
- ~~NT cache hints~~ — BEZ EFEKTU (ani expert, ani QKV/O)
- ~~Warp-level QKV/O GEMV~~ — BEZ EFEKTU
- ~~DN multi-warp~~ — REGRESJA (155.9 t/s, K=768 za małe)
- ~~GU RPW=2~~ — REGRESJA (153.5 t/s, za mało bloków)
- ~~Light fences (s_wait_storecnt+global_inv)~~ — tak samo drogie jak __threadfence (~5µs)

**Fundamentalny problem:** Na RDNA4 gfx1201, HIP dispatch overhead (~1.7µs) jest TAŃSZY niż jakakolwiek forma inter-block synchronizacji. GGUF Vulkan ma przewagę dzięki pre-recorded command buffers (natywna Vulkan feature) z near-zero dispatch overhead — HIP nie ma odpowiednika tej technologii.

**Pozostałe kierunki:**
1. **`rocprof` hardware counters** — zrozumieć prawdziwy bottleneck O GEMV (39% BW) i QKV GEMV (46% BW)
2. **Software prefetch hints** — `__builtin_amdgcn_s_prefetch_data` w inner loop GEMV
3. **Lepszy GEMV layout** — transposed weights [K,N/2] zamiast [N,K/2] dla QKV/O (lepszy coalescing)
4. **Reduced precision scales** — FP8 zamiast FP16 skale w g64 → mniej bandwidth na metadane
5. **WMMA fused INT4 dequant GEMM dla prefill** — C++ prefill działa (872 t/s pp128) ale scalar INT4 GEMV jest ALU-bound. WMMA v_wmma_f16_16x16x16_f16 z fused INT4 dequant mógłby dać ~3-5× speedup na expert GU/DN w prefill
6. **Split-K O GEMV** — O ma tylko 512 bloków (najniższy parallelism z dużych GEMVów), split-K może poprawić BW utilization
7. ~~**Interleaved scale format dla QKV/O**~~ — PRZETESTOWANE (#28): REGRESJA -2.8%. Row-major `[N, K/64, 2]` jest optymalny.
8. **Direct HSA queue manipulation** — lower-level dispatch API niż HIP, potencjalnie niższy overhead

---

## 9. Hybrid Dense: Qwen3.5-27B (Gated DeltaNet + Full Attention) — W TRAKCIE

**Model:** 64 warstwy (48× Gated DeltaNet + 16× Full Attention), hidden=5120, FFN=17408
**Architektura:** Hybrid — DeltaNet O(1) per token + Full Attention (GQA 24Q/4KV, D=256)
**Kwantyzacja:** INT4 asymmetric g32 + Hadamard + GPTQ (quantize_hybrid.py)
**Engine:** int4_engine_hybrid.py (Python ref) + rozszerzony int4_decode_step.hip

### GGUF Q4_K_M Baseline (Vulkan1, R9700, pomiar 2026-03-15)

| Test | t/s | Uwagi |
|------|-----|-------|
| pp128 | 641.85 ± 27.32 | Prefill |
| pp256 | 667.03 ± 0.84 | |
| pp512 | 739.71 ± 0.92 | |
| pp1024 | 735.04 ± 1.22 | |
| tg128 | **25.98 ± 0.07** | Baseline decode |
| tg256 | **25.93 ± 0.04** | |
| tg512 | **25.84 ± 0.04** | Stabilny (pp=0) |

**GGUF model size:** 15.58 GiB (Q4_K_M)
**UWAGA:** Pierwszy test z pp128-pp1024 + tg512 dawał 5.73 t/s — to VRAM overflow od dużych promptów. Re-test z pp=0 potwierdza stabilne ~26 t/s do tg512.

**Kluczowe obserwacje:**
- GGUF Vulkan: **~26 t/s** stabilnie do min. tg512 (pp=0)
- GGUF traktuje WSZYSTKIE 64 warstwy z KV cache → przy dłuższych kontekstach ryzyko VRAM overflow
- **Nasza przewaga:** DeltaNet state (48 warstw) = O(1), tylko 72 MB niezależnie od context
- KV cache tylko dla 16 warstw Full Attention → ~4x mniej niż GGUF
- **Cel decode:** ≥33 t/s (bijemy GGUF o ≥27%) + stabilność na bardzo długim kontekście

### Architektura plików

| Plik | Co robi |
|------|---------|
| qwen3_5_27b/ARCHITECTURE.md | Pełna specyfikacja architektury, tensor paths |
| qwen3_5_27b/quantize_hybrid.py | Kwantyzacja INT4 + GPTQ + Hadamard (hybrid) |
| qwen3_5_27b/int4_engine_hybrid.py | Python inference engine (reference) |
| qwen3_5_27b/bench_gguf_baseline.sh | Skrypt benchmarku GGUF |

---

## Co warto zbadać przy następnych modelach

1. **FP8 wagi** (zamiast INT4) — gfx1201 ma natywne FP8 GEMM. Może lepsza jakość przy tej samej prędkości?
2. **Łączenie kwantyzacji wag z kwantyzacją aktywacji (W4A8)** — aktivacje w INT8, wagi w INT4
3. **SmoothQuant dla aktivacji** — przed kwantyzacją wag przenieść outliers z aktivacji w wagi
4. **Modele z MoE** (Mixture of Experts jak Qwen3-MoE) — osobna sensitivity per expert?
5. **Quantization-aware fine-tuning** (QLoRA style) po kwantyzacji — może odzyska 0.1-0.2 PPL

---

## Pliki referencyjne

### Dense (Qwen3-14B)
| Plik | Co robi |
|------|---------|
| quantize_v4_gptq.py | **Główny** — mixed INT4/INT8, Hadamard, GPTQ |
| quantize_v5_pure_int4.py | Pure INT4, mniejszy rozmiar |
| int4_engine_v5.py | Inference engine dense (production) |

### MoE (Qwen3-30B-A3B)
| Plik | Co robi |
|------|---------|
| qwen3_30b_a3b/quantize_moe_gptq.py | Kwantyzacja MoE — GPTQ + Hadamard per expert |
| qwen3_30b_a3b/int4_engine_moe.py | Inference engine MoE (decode + prefill) |

### Wspólne
| Plik | Co robi |
|------|---------|
| int4_quant_v2.py | Asymmetric INT4 quantization functions |
| hadamard_utils.py | Hadamard rotation matrices |
| measure_sensitivity.py | Hessian-based sensitivity |
| engine_utils.py | RMSNorm, KVCache, RoPE utilities |
| hip_int4/int4_decode_step.hip | HIP kernels (GEMV, attention, MoE routing) |
| hip_int4/setup.py | Build script |

### Benchmarking
| Plik | Co robi |
|------|---------|
| llama-bench (Vulkan) | `/home/janusz/llama.cpp/build_vulkan/bin/llama-bench -dev Vulkan1` |
| GGUF modele | `/home/janusz/drugi_dysk/GGUF/` |

---

## Qwen3.5-27B: Dense Hybrid (DeltaNet + Full Attention)

### Architektura
- 27B parametrów, 64 warstwy: 48× Gated DeltaNet (linear attention) + 16× Full Attention
- Wzór: 3 DeltaNet + 1 FullAttn × 16 bloków
- Hidden=5120, FFN=17408, FullAttn: 24Q/4KV heads D=256, DeltaNet: 16K/48V heads k_dim=v_dim=128
- Partial RoPE: rotary_factor=0.25 → rotary_dim=64, n_freqs=32 (tylko w 16 FullAttn warstwach)

### Kwantyzacja
- Metoda: INT4 asymmetric + Hadamard(32×32) + GPTQ Cholesky, block_size=32
- Kalibracja: 256 samples × 256 tokens (RedPajama)
- Co INT4: q/k/v/o_proj, FFN gate_up/down (we WSZYSTKICH 64 warstwach)
- Co FP16: delta_gate_a/b, short_conv, A_log, dt_bias, attn_norm, norm weights, q/k_norm
- VRAM: 18022 MB / 32768 MB (14.7 GB headroom)

### Ważne konwencje (trudne do debugowania!)

1. **RMSNorm (1+weight)**: Regularne normy (in_norm, post_norm, final_norm) używają `(1+w)*x*rrms`.
   Init=0, kernel C++ używa `W[i]` → **trzeba pre-add 1.0** do wag norm zanim podamy do kerneli.
   ALE: q_norm/k_norm (w head_rmsnorm_partial_rope_kv) mają `1+W` wbudowane w kernel.
   ALE: dn_attn_norm (w rmsnorm_gated_silu) używa `W` bezpośrednio (init=1).

2. **Query/Gate interleaved layout**: FullAttn q_proj daje [H, head_dim*2] z interleave per głowa:
   `[q_h0[256], gate_h0[256], q_h1[256], gate_h1[256], ...]`, NIE flat `[all_q, all_gate]`.
   Split: `q_gate.view(H, hd*2)` → `[:, :hd]` = query, `[:, hd:]` = gate.

3. **FP16 weights bez Hadamard**: in_proj_a/b [48,5120] potrzebują pre-Hadamard input (`norot`),
   bo INT4 wagi mają Hadamard wchłonięty. Trzeba osobno śledzić `rot` (post-FWHT) i `norot` (pre-FWHT).

4. **Partial RoPE COS/SIN**: Kernel oczekuje `COS[pos_idx * n_freqs : (pos_idx+1) * n_freqs]`,
   NIE pełnej tablicy [max_seq, n_freqs]. Przy graph capture: kernel `_g` czyta pozycję z `g_graph_pos`.

5. **repeat_interleave**: DeltaNet ma 16 K heads i 48 V heads → repeat factor=3.
   Kernel `repeat_interleave_heads` oczekuje `num_v_heads=48` (output), NIE `num_k=16` (input).

### Wyniki decode (2026-03-15)

| Mode | ctx=128 | ctx=256 | ctx=512 | ctx=1024 |
|------|---------|---------|---------|----------|
| Python (reference) | ~3.8 t/s | - | - | - |
| Python HIP kernels | 22.5 t/s | - | - | - |
| C++ non-graph | 28.1 t/s | 27.2 t/s | 26.1 t/s | 24.2 t/s |
| **HIP Graph** | **29.2 t/s** | **28.0 t/s** | **26.8 t/s** | **24.7 t/s** |
| GGUF Q4_K_M Vulkan | 26.0 t/s | 25.9 t/s | 25.8 t/s | ~25.7 t/s* |

*GGUF ctx=1024 estymowane z trendu (brak pliku GGUF na dysku).

**Podsumowanie: +12.3% szybciej niż GGUF przy ctx=128, +8% przy ctx=256.**

### Analiza wydajności

**Budżet bandwidth per token:**
| Komponent | Rozmiar |
|-----------|---------|
| INT4 weights (27B × 0.5B/param) | 13.5 GB |
| Scales g32 (K/32 × N × 4B per matrix) | ~1.5 GB |
| DeltaNet states (48 layers × 48 heads × 128² × 4B, R+W) | 0.29 GB |
| KV cache ctx=128 (16 layers × 4 heads × 128 × 256 × 2B × 2) | 0.03 GB |
| Norms, buffers, LM head scales, misc | ~0.1 GB |
| **TOTAL** | **~15.4 GB** |

Theoretical min: 15.4 GB / 507 GB/s = 30.4 ms = **32.9 t/s** (ctx=128)
Actual (graph): 34.3 ms = 29.2 t/s → **88.6% bandwidth efficiency**

**Bottleneck:** ~5 ms overhead from kernel serialization and GEMV sub-optimal bandwidth utilization.
NOT kernel launch overhead (graph only saved 1.3 ms).

### Context scaling problem

Degradacja 29.2→24.7 t/s z ctx 128→1024 = 4.5 ms overhead.
Źródło: `flash_decode_fused_fp16_d256` jest latency-bound (sequential iteration over positions).
16 FullAttn layers × 4 KV heads × 1024 pos × ~400ns/pos = ~6.6 ms theoretical latency.
**Fix:** split-K flash decode (jak w MoE: `launch_flash_decode_fp16_g`).

### Wnioski dot. DeltaNet

- **DeltaNet state update jest tani**: 48 layers × 3 MB = 144 MB read + 144 MB write = 288 MB total.
  Przy 507 GB/s = 0.57 ms — zaledwie ~1.6% czasu decode.
- **Prawdziwa przewaga DeltaNet**: stały koszt O(1) per token niezależnie od kontekstu.
  ALE nasz flash_decode w 16 warstwach FullAttn niweluje tę przewagę.
- **Z split-K flash decode**: degradacja powinna być znacznie mniejsza niż GGUF na długich kontekstach.

### Prefill (2026-03-15)

**Optymalizacje zastosowane:**
1. `dequant_gemm_out` — pre-allocated flat buffer [max_N*K] FP16, eliminuje alokacje per-call
2. `hybrid_deltanet_recurrence_batch_v2` — batched pre/post-processing (conv1d, L2norm, repeat, SSM params, RMSNorm+gating), reducing kernel launches from ~12×M to ~M+10
3. `gated_delta_net_step_batch` — single kernel launch for all M tokens per head (looping inside kernel), eliminates M-1 launch overhead

| Prefill | pp128 | pp256 | pp512 | pp1024 |
|---------|-------|-------|-------|--------|
| Custom INT4 v1 | 276 t/s | 397 t/s | 471 t/s | 507 t/s |
| Custom INT4 v2 (LDS+fused) | **408 t/s** | **673 t/s** | **933 t/s** | **1095 t/s** |
| GGUF Q4_K_M Vulkan | 642 t/s | 667 t/s | 740 t/s | 833 t/s |

**v1→v2 optymalizacje (2026-03-15):**
1. `gated_delta_net_step_batch_lds` — DeltaNet state cached in LDS as FP16 (32 KB fits in 64 KB LDS). State loaded from VRAM FP32 once, all M tokens processed from fast LDS (~640 GB/s per CU), stored back once. Recurrence: 8 ms → 1.75 ms per layer. **Update (2026-03-16):** half2 packing eliminates 2-way bank conflicts → **1.05 ms per layer (40% faster).**
2. `rmsnorm_hadamard_batched` — fused RMSNorm + Walsh-Hadamard(32) via `__shfl_xor` butterfly (5 stages). Qwen3.5 uses `(1+W)` convention. Single kernel replaces rmsnorm + matmul. 128 calls: 216 ms → 2.6 ms.
3. `hadamard32_batched` — standalone Walsh-Hadamard via butterfly shuffles, no shared mem. Used for O-proj and down-proj pre-rotation.

**Wynik:** pp256 już bije GGUF (+1%), pp512 bije o **+26%**, pp1024 o **+31%**. pp128 nadal -36% (GEMM-bound przy małym batch).
**Remaining bottleneck:** At pp128, GEMMs (dequant+rocBLAS) dominate at ~300 ms. DeltaNet recurrence only ~38 ms. Norm+Had only ~2.6 ms.

### Jakość (2026-03-15)

| Metric | Custom INT4 | GGUF Q4_K_M (expected) |
|--------|-------------|------------------------|
| **MMLU (456 q)** | **80.5%** | ~79-81%* |
| **WikiText-2 PPL** | **6.42** | ~6.0-6.5* |

*GGUF nie ma na dysku, wartości szacowane z typowych wyników Q4_K_M dla 27B.

MMLU at 2.3 questions/sec, PPL eval in 39 min (297K tokens, ctx=2048, stride=512).

### Podsumowanie Qwen3.5-27B

| Aspekt | Custom INT4 | GGUF Q4_K_M | Wynik |
|--------|-------------|-------------|-------|
| Decode (ctx=128) | **28.0 t/s** | ~26.0 t/s | **+8%** ✓ |
| Decode (ctx=512) | **26.3 t/s** | ~26.0 t/s | **+1%** ✓ |
| Prefill (pp128) | ~457 t/s* | 642 t/s | -29% ✗ |
| Prefill (pp256) | **673 t/s** | 667 t/s | **+1%** ✓ |
| Prefill (pp512) | **933 t/s** | 740 t/s | **+26%** ✓ |
| Prefill (pp1024) | **1095 t/s** | 833 t/s | **+31%** ✓ |
| MMLU | 80.5% | ~80% | **Porównywalny** ✓ |
| PPL | 6.42 | ~6.2* | **Porównywalny** ✓ |
| VRAM | 18.0 GB | 15.6 GB | +2.4 GB (scales g32) |
| Long ctx scaling | O(1) DeltaNet 48/64 warstw | O(n) all 64 layers | **Przewaga** ✓ |

*pp128 estimated with half2 DeltaNet fix (1.75→1.05ms/layer, 34ms saved from 314ms baseline).

**Główna przewaga:** Decode +8%, Prefill +26-31% at pp512+, O(1) DeltaNet for context scaling.
**Główna słabość:** Prefill pp128 still -29% (DeltaNet recurrence 49ms + GEMM overhead).

### Możliwe optymalizacje (TODO)

1. **Split-K flash decode D=256** — fix context degradation, biggest win for long ctx
2. **g32 → g64 conversion** — reduce scale data 50% (~0.75 GB), enable multi-warp GEMV
3. ~~Chunkwise-parallel DeltaNet for prefill~~ — **SOLVED** via LDS-cached FP16 state (v2 kernel)
4. ~~Fused norm+FWHT+GEMV for g32~~ — **SOLVED** via `rmsnorm_hadamard_batched` butterfly kernel
5. ~~**Fused dequant+GEMM kernel**~~ — See section 9 below for complete analysis.

---

### 9. Prefill pp128 Optimization: Fused Dequant+GEMM i W4A4 INT4 WMMA

**Problem:** Prefill pp128 = 418 t/s vs GGUF 642 t/s (−35%). Bottleneck: osobna dequantyzacja + hipBLAS GEMM.

#### 9a. Transposed dequant + contiguous GEMM ✓

**Odkrycie:** hipBLAS daje 73.5 TFLOPS z wagami [K,N] (contiguous) vs 48.3 TFLOPS z wagami [N,K].t() — **52% speedup** samym layoutem pamięci!

**Implementacja:**
1. `dequant_kn_vec8_kernel` — dequantyzacja INT4 → FP16 do layoutu [K,N], vec8 (uint64) loads
2. `direct_hgemm_contiguous` — hipBLAS HGEMM z HIPBLAS_OP_N (bez transpozycji)

**Wyniki (M=128, pełny model):**
- dequant_vec8: 94.1ms total (598 GB/s, ~98% peak BW)
- contiguous GEMM: 102.4ms total
- **Total: 196.5ms → 652 t/s** — bije GGUF 642 t/s o +1.5%!
- Z HIP Graph (~20ms savings): ~672 t/s (+4.7%)

**Status:** Kernele dodane do `int4_decode_step.hip`, oczekują na pybind export i integrację z `prefill_fast`.

#### 9b. Fused FP16 WMMA dequant+GEMM (v9c) — CZĘŚCIOWO ✓

**Co:** Dequantyzacja INT4→FP16 fused z FP16 WMMA w jednym kernelu.

**Wyniki per GEMM (M=128):**
| GEMM | v9c fused | dq+BLAS | Ratio |
|------|-----------|---------|-------|
| gate_up | **1.29ms (35.5 TF)** | 1.91ms (23.9 TF) | **+48%** |
| down | 1.09ms (20.9 TF) | 0.89ms (25.6 TF) | −18% |
| dn_qkv | 0.43ms (31.4 TF) | 0.51ms (26.4 TF) | +19% |

**Wniosek:** Fused kernel wygrywa na dużych N (gate_up), przegrywa na dużych K (down). Hybridowe dispatch mogłoby pomóc ale total jest porównywalny z dq+BLAS.

#### 9c. W4A4 INT4 WMMA — EKSPERYMENTALNE 🔬

**Podejście:** Natywna instrukcja `wmma_i32_16x16x16_iu4` (INT4×INT4 → INT32).

**Kluczowe odkrycia:**

1. **Raw WMMA throughput (pure compute, no memory):**

| | Throughput | Cycles/WMMA |
|---|---|---|
| FP16 WMMA | 213 TFLOPS | ~14 |
| **INT4 WMMA** | **423 TOPS** | ~7 |
| Ratio | **2.0×** | — |

2. **Coalesced B layout jest kluczowy:**
   - B stored as [K/8, N] (transposed) → lanes read consecutive N addresses
   - Speedup: **10× vs non-coalesced** [N, K/8] layout
   - gate_up: 4.62ms → 0.46ms (98.8 TFLOPS!)

3. **Per-group rescaling jest konieczny:**
   - Wagi z per-group(32) scales → reset INT32 acc co 2 K-steps
   - Z per-group: 75.8ms total (coalesced) vs 405.6ms (non-coalesced)

4. **Pełne wyniki (M=128, coalesced, per-group(32) rescaling):**

| GEMM | W4A4 INT4 WMMA | dq_vec8+BLAS | Ratio |
|------|----------------|--------------|-------|
| gate_up | **0.462ms (98.8 TF)** | 1.72ms | **3.7×** |
| down | **0.329ms (69.4 TF)** | 0.89ms | **2.7×** |
| dn_qkv | **0.200ms (67.0 TF)** | 0.51ms | **2.5×** |
| dn_z | **0.109ms (61.8 TF)** | 0.28ms | **2.6×** |
| **TOTAL** | **75.8ms** | **196.5ms** | **2.6×** |
| **t/s (z overhead)** | **~1336** | **~652** | **2.0×** |

5. **Poprawność:**
   - GPU vs W4A4 reference: NRMSE = 1.11% (kernel jest poprawny)
   - W4A4 vs FP32: NRMSE = 10.85% per-GEMM (inherent INT4 activation quantization error)

6. **Ryzyko jakości:**
   - INT4 aktywacje (per-row symmetric, range [-7,7]) tracą precyzję
   - 10.85% NRMSE per-layer × 64 warstwy = potencjalnie znacząca degradacja
   - Wymaga testów PPL/MMLU przed wdrożeniem
   - Porównanie: nasza kwantyzacja wag INT4 GPTQ daje ~0.5% NRMSE per-layer

**Wniosek:** W4A4 INT4 WMMA daje **2.0× speedup vs dq+BLAS** i **2.1× vs GGUF** w surowym prefill compute, ale jakość INT4 aktywacji jest ryzykowna dla LLM inference. Bezpieczna opcja (dq_vec8+BLAS) już bije GGUF. W4A4 powinno być używane selektywnie (np. tylko FFN gate_up/down) lub z wyższą precyzją aktywacji (W4A8).

**Pliki testowe:** `/tmp/test_w4a4_coalesced.hip`, `/tmp/test_w4a4_v5.hip`, `/tmp/test_wmma_tp4.hip`

---

### 8. W4A8 WMMA GEMM — INT4 wagi × INT8 aktywacje — W TRAKCIE 🔧

**Data:** 2026-03-16

**Co to:** Zamiast W4A4 (INT4 aktywacje, 10.85% NRMSE — zbyt stratne), używamy INT8 aktywacji (per-token symmetric, range [-127,127]). WMMA `v_wmma_i32_16x16x16_iu8` na gfx1201 daje 179 TOPS peak.

**Kluczowe techniki (opracowane w v1-v17):**

1. **Tiled weight layout** `[K/8, N/16, 64]` — 16 N-lanes × 4 packed bytes = 64B, coalesced per cache line. **~2× speedup** (v6→v7)
2. **Group-major scales** `[ng, N, 2]` zamiast `[N, ng, 2]` — coalesced na N. **~1.3-3× speedup** zależnie od N (v7→v8)
3. **Tiled activations** `[K/8, M/16, 128]` — coalesced A access. **~21% speedup** (v9a→v11a)
4. **INT4→INT8 expansion** via `v_perm_b32` — 5 ALU ops vs 23 scalar (2× AND + 2× perm)
5. **4 N-tiles per wave** — 4 independent WMMAs per K-step fill pipeline (1 WMMA not enough)
6. **Factored zero-point correction** (v15b/v17b):
   - Main loop: only `fp_acc += ws * gacc` (fast, no wz)
   - Post-loop: `corr = Σ_g wz_combined * asum` (small O(ng) loop)
   - v17b: interleaved correction every group (best for medium N, scales in L1)
   - **~13% speedup for medium N** (v11a→v15b/v17b)

**Wyniki per GEMM (M=128, GS=32, best kernel per shape):**

| Shape | Czas | TOPS | Kernel | % peak |
|-------|------|------|--------|--------|
| QKV-DN [5120→10240] | 0.290ms | 46.3 | v17b | 25.9% |
| QKV-FA [5120→8192] | 0.228ms | 47.1 | v15b | 26.3% |
| O-proj [5120→5120] | 0.213ms | 31.5 | v15b | 17.6% |
| O-proj [6144→5120] | 0.258ms | 31.3 | v15b | 17.5% |
| Gate+Up [5120→34816] | 1.333ms | 34.2 | v11a | 19.1% |
| Down [17408→5120] | 0.812ms | 28.1 | v11a | 15.7% |

**Wyniki full 64-layer prefill (pp128):**

| Metoda | Czas | t/s |
|--------|------|-----|
| GGUF Q4_K_M Vulkan | 199ms | 642 |
| Dequant + rocBLAS | 270ms | 474 |
| **W4A8 WMMA (C++ standalone)** | **191ms** | **670** |
| **W4A8 WMMA (Python dispatch)** | **195ms** | **655** |
| **Estimated with norms/attn** | **~200ms** | **~640** |

**Progresja wersji kerneli:**

| Wersja | TOPS (N=5120) | TOPS (N=10240) | Kluczowa zmiana |
|--------|---------------|----------------|-----------------|
| v3 | 4.7 | — | Baseline |
| v4 | 14.4 | — | Multi-wave |
| v6 | 5.9 | 18.4 | 4 N-tiles/wave |
| v7 | 19.8 | 31.2 | + Tiled weights |
| v8 | 23.0 | 40.1 | + Group-major scales |
| v11a | **27.9** | **42.0** | + Tiled activations |
| v15b | **31.6** | **47.9** | + Factored correction |
| v17b | 31.4 | **46.3** | + Interleaved correction |

**Bottleneck analysis:**
- Gate+Up (34816 columns) = 50% of per-layer GEMM time
- Memory-bandwidth bound: 89MB weights + 22MB scales at 507 GB/s = 0.22ms theoretical, actual 1.33ms (6×)
- Scale/zero-point correction = 73% of inner loop instructions (from v12 analysis)
- Only 2 waves/SIMD (93 VGPRs for 4 N-tiles) limits memory latency hiding
- Higher occupancy (v10, v14, v16) tested but always worse — WMMA pipeline fill > latency hiding

**Wnioski:**
1. W4A8 WMMA eliminuje 122ms overhead dequantu, ale sam WMMA jest wolniejszy od rocBLAS dla surowego GEMM
2. Net improvement vs dq+rocBLAS: 270ms → ~200ms = **26% szybciej**
3. vs GGUF pp128: zasadniczo **remis** (~200ms vs 199ms)
4. vs GGUF pp256+: nadal **wygrywamy** (714 vs 669 t/s z dq+rocBLAS, WMMA powinno być jeszcze lepsze)
5. Jakość INT8 aktywacji: ~1.68% NRMSE (znacznie lepsza niż INT4's 10.85%)

**WMMA vs dq+rocBLAS crossover (GEMM-only, 2026-03-16):**

| M | WMMA | dq+rocBLAS | Winner |
|---|------|------------|--------|
| 128 | 195ms (657 t/s) | 228ms (562 t/s) | **WMMA +15%** |
| 256 | 337ms (760 t/s) | 242ms (1059 t/s) | **dq+BLAS +28%** |
| 512 | 648ms (790 t/s) | 310ms (1653 t/s) | **dq+BLAS +52%** |

**Wniosek:** WMMA wygrywa TYLKO przy pp128 (M≤~160). Przy pp256+ rocBLAS efektywniej tileuje M×N. Strategia: WMMA dla decode/pp128, dq+rocBLAS dla pp256+.

**DeltaNet recurrence: half2 LDS optimization (2026-03-16):**
- Problem: `__half state_lds[128][128]` miał 2-way bank conflicts (2-bajtowe elementy, 4-bajtowe banki)
- Fix: `__half2 state_lds[128][64]` — pakowanie par sąsiednich elementów, bank=tid%32 → zero conflicts
- Wynik: **1.75ms → 1.05ms/layer = 40% szybciej**
- 48 warstw: **84ms → 50ms** (34ms saved)

**Full pp128 breakdown (po optymalizacjach, 2026-03-16):**

| Komponent | Czas | Udział |
|-----------|------|--------|
| WMMA GEMMs (64 layers) | 195ms | 78% |
| DeltaNet recurrence (48 layers) | 49ms | 20% |
| SDPA attention (16 layers) | 0.6ms | 0.2% |
| Norms + Hadamard + SiLU + dispatch | ~5ms | 2% |
| **TOTAL estimated** | **~250ms** | → **512 t/s** |
| GGUF pp128 baseline | 199ms | → 642 t/s |

**Gap: 51ms (25% behind GGUF).** Bottleneck: GEMMs (195ms) + DeltaNet (49ms). GGUF nie ma DeltaNet overhead.

**Następne kroki:**
- Chunked-parallel DeltaNet prefill (reduce 49ms → ~5-10ms potential)
- C++ prefill loop (eliminacja Python dispatch overhead ~5ms)
- Hybrid strategy: WMMA @pp128, dq+rocBLAS @pp256+ (automatic M threshold)

**Pliki:** `/tmp/wmma_gemm_w4a8_v{1..17}.hip`, `qwen3_5_27b/hip_int4_wmma/int4_decode_step.hip`

---

### 14. Dispatch reduction — fused kernels ✅

**Co to:** Redukcja liczby kernel dispatch per layer w decode loop (MoE path, Qwen3-30B-A3B).

**Kontekst:** Dispatch overhead na RDNA4 to ~2µs per launch. Przy 12 dispatches × 48 layers = 576 dispatches = ~1.15ms overhead. Przy 5.81ms/token to 20% czasu.

**Przetestowane fuzje:**

| Fuzja | Dispatche zaoszczędzone | Wynik | Status |
|-------|------------------------|-------|--------|
| ResNorm+FWHT+Router GEMV (fused_resnorm_router_gemv) | 1/layer (48 total) | 168.5 → 172.5 t/s (+4 t/s) | ✅ W PRODUKCJI |
| Flash decode partial+reduce (flash_decode_selfreduce_fp16) | 1/layer (48 total) | 172.0 → 175.5 t/s (+3.5 t/s) | ✅ W PRODUKCJI |
| SiLU+FWHT+DN GEMV (gemv_silu_fwht_xbatch_g64) | 1/layer | 168.5 → 152.0 t/s (-16.5 t/s) REGRESJA | ❌ PORZUCONE |
| Flash decode fused threshold 32→256 | 1/layer | 172.0 → 160 t/s REGRESJA | ❌ PORZUCONE |

**Aktualna ścieżka decode (10 dispatches/layer):**
1. QKV GEMV
2. HeadNorm+RoPE+KV write
3. Flash decode (self-reducing: partial+reduce w 1 dispatch)
4. O GEMV
5. Fused ResNorm+FWHT+Router GEMV
6. Softmax+TopK
7. GU GEMV
8. SiLU+FWHT
9. DN GEMV
10. MoE reduce+norm+FWHT

**Kluczowe techniki:**

1. **fused_resnorm_router_gemv**: Każdy blok Router GEMV redundantnie oblicza ResNorm i FWHT. Ponieważ D=2048 mieści się w L2 cache (4MB), redundantne odczyty są tanie (~25 cykli z L2). FWHT distribution: zamiast blok-0-only (serial bottleneck, 158 t/s), FWHT writes rozdzielone po WSZYSTKICH blokach (172.5 t/s).

2. **flash_decode_selfreduce_fp16**: Atomowy "last-block-reduces" pattern. Każdy blok partial zapisuje wynik, robi `__threadfence()` + `atomicAdd(counter)`. Ostatni blok (counter == n_splits-1) robi redukcję + FWHT inline. Counter resetowany przez last block — bezpieczne bo HIP stream ordering gwarantuje sekwencyjne uruchomienie per layer. Oszczędza 1 dispatch/layer.

3. **SiLU+FWHT+DN fuzja — DLACZEGO ZAWIODŁA**: DN kernel ma `__launch_bounds__(32,2)` = max 2 wave/SIMD. Dodanie float xv[32] (32 VGPRs do FWHT) + SiLU expf spowolniło compute per thread ponad limit co occupancy mogło ukryć. Occupancy-limited kernele nie tolerują dodatkowego ALU.

4. **Flash decode fused threshold 256 — DLACZEGO ZAWIODŁO**: Fused flash decode ma 4 bloki (num_kv_heads=4) × 256 threads = bardzo niska occupancy (0.5 wave/SIMD). Każdy warp loopuje po 256 pozycjach seryjnie — słabe ukrywanie latencji pamięci.

**Wynik końcowy:**
- **175.5 t/s** (C++ loop, tg=100, CTX=128)
- vs GGUF Q4_K_M Vulkan: **171.8 t/s** (Vulkan2, ten sam GPU)
- **Bijemy GGUF o 3.7 t/s (2.2%)** 🎉

**Profil rozkładu czasu (per layer, szacowany):**

| Kategoria | µs/layer | % |
|-----------|----------|---|
| QKV GEMV | 14.5 | 12.3% |
| HeadNorm+RoPE | 3.9 | 3.3% |
| Flash decode (self-reduce) | ~18 | 15.3% |
| O GEMV | 12.2 | 10.4% |
| Fused ResNorm+Router | ~5 | 4.3% |
| Softmax+TopK | 6.9 | 5.9% |
| GU GEMV | 27.4 | 23.3% |
| SiLU+FWHT | 2.5 | 2.1% |
| DN GEMV | 14.5 | 12.3% |
| MoE reduce+norm+FWHT | 7.0 | 6.0% |
| Dispatch overhead (10×~2µs) | ~20 | 5.0% (est.) |

**Pliki:** `hip_int4/int4_decode_step.hip` (kernele ~8391-8490: selfreduce; ~9234-9340: fused_resnorm_router)

---

## Qwen3.5-27B — Hybrid DeltaNet+FullAttn INT4

### Architektura

| Parametr | Wartość |
|----------|---------|
| Parametry | 27B (dense) |
| Warstwy | 64 (48× DeltaNet + 16× FullAttn, wzór: 3+1 × 16) |
| Hidden dim | 5120 |
| FFN intermediate | 17408 (SiLU-gated) |
| Full Attention | 24 Q heads, 4 KV heads, head_dim=256 |
| DeltaNet | 48 V heads, 16 QK heads, head_dim=128, short_conv k=4 |

### 8. Qwen3.5-27B g32 baseline

**Kwantyzacja:** INT4 asymmetric + Hadamard rotation (32×32) + GPTQ Cholesky, group_size=32

**Wyniki:**

| Metryka | Nasza INT4 g32 | GGUF Q4_K_M Vulkan |
|---------|----------------|-------------------|
| PPL (WikiText-2) | 6.42 | TBD |
| Decode ctx=128 | 27.9 t/s | 26.1 t/s |
| Decode ctx=256 | 27.3 t/s | ~26 t/s |
| Decode ctx=512 | 26.1 t/s | ~26 t/s |
| Decode ctx=1024 | 24.1 t/s | ~26 t/s |
| Prefill pp128 | ~512 t/s | 641 t/s |
| VRAM | ~17.1 GB | 15.6 GB |

**Kluczowa obserwacja:** Przy ctx=128 bijemy GGUF o **+7%**, ale przy ctx=1024 jesteśmy **-7%** wolniejsi.
Przyczyna: 16 warstw FullAttn z head_dim=256 wymaga flash_decode z większą liczbą rejestrów → niższe occupancy.
DeltaNet warstwy (48/64) mają stały koszt ~1.05 ms/warstwa niezależnie od kontekstu.

### 9. Optymalizacja DeltaNet kernel — PLATEAU ⚠️

**Cel:** Przyspieszyć DeltaNet recurrence (50ms dla 48 warstw @ M=128 prefill, ~1.05ms/warstwa decode)

**Próby:**

| Wariant | Opis | Czas (48 warstw, M=128) | Wynik |
|---------|------|------------------------|-------|
| v1 (bazowy) | half2 LDS, bank conflict fix | 50.3 ms | baseline |
| Multi-wave 64 threads | 2 warpy per CU, podwójny throughput? | 49.5 ms | ❌ brak poprawy |
| FP32 LDS | Eliminacja half↔float konwersji | 50.3 ms | ❌ brak poprawy, LDS overflow fixed |
| Cleaned half2 | Delta w rejestrach, mniej barier | 50.4 ms | ❌ brak poprawy |

**Wniosek:** Kernel jest LDS-throughput-limited. 128×128 state matrix (32 KB half2) wymaga 2 pełnych odczyt+zapis LDS per token per głowę. Na RDNA4 z 128 KB LDS/SIMD i ~2 TB/s LDS BW per SIMD, to jest hardware limit.

**Dlaczego 64 threads nie pomogło:** 2 warpy na jednym SIMD nie podwajają LDS throughput (LDS jest per-SIMD, nie per-warp). Kernel jest instruction-limited, nie latency-limited.

### 10. g32 → g64 native GEMV — SUKCES ✅

**Cel:** Zmniejszyć bandwidth GEMV przez redukcję danych skali o 50%

**Implementacja:**
1. Konwersja wag: `convert_g32_to_g64.py` — requantyzacja z parowania sąsiednich bloków g32
2. Native g64 GEMV: parametr `GS` w template `gemv_warp<BLOCK_N, GS>`, `scale_idx = it / (GS/32)`
3. Osobny `lm_group_size` dla LM head (zostaje g32 bo symmetric quant)
4. `dequant_int4_g64` kernel do prefill path

**Wyniki:**

| Metryka | g32 | g64 | Zmiana |
|---------|-----|-----|--------|
| PPL (WikiText-2) | 6.42 | 6.52 | +0.10 (akceptowalne) |
| Decode ctx=128 | 27.9 t/s | 29.4 t/s | **+5.4%** |
| Decode ctx=256 | 27.3 t/s | 28.7 t/s | **+5.1%** |
| Decode ctx=512 | 26.1 t/s | 27.5 t/s | **+5.4%** |
| Decode ctx=1024 | 24.1 t/s | 25.2 t/s | **+4.6%** |
| VRAM | ~17.1 GB | 16.5 GB | -600 MB |

**vs GGUF Q4_K_M Vulkan (26.1 t/s @ tg128):**

| Kontekst | Nasza g64 | GGUF Q4_K_M | Różnica |
|----------|-----------|-------------|---------|
| 128 | 29.4 t/s | 26.1 t/s | **+12.6%** |
| 256 | 28.7 t/s | ~26 t/s | **+10.4%** |
| 512 | 27.5 t/s | ~26 t/s | **+5.8%** |
| 1024 | 25.2 t/s | ~26 t/s | **-3.1%** |

**Problem:** Przy ctx≥1024 tracimy z GGUF z powodu flash_decode — oryginalny kernel uruchamia tylko 24 warpy (4 KV heads × 6 GQA ratio = 37.5% GPU).

**Pliki:**
- `hip_int4/int4_decode_step.hip` — gemv_warp/gemv_warp_had z parametrem GS, launch_gemv z gs, hybrid_decode_step_logits/graph z group_size + lm_group_size
- `qwen3_5_27b/int4_engine_hybrid.py` — auto-detect group_size, _dequant z gs param, lm_group_size
- `convert_g32_to_g64.py` — konwersja wag
- `qwen3_5_27b/quantized_hybrid_g64/` — skonwertowane wagi

---

## 11. Split-K Flash Decode D=256

**Problem:** `flash_decode_fused_fp16_d256` launches Grid(Hk=4), Block(32, gqa_ratio=6) = only 24 warps (37.5% GPU utilization). At ctx≥512 this becomes the bottleneck — each warp sequentially iterates over all seq_len tokens.

**Rozwiązanie:** Split-K — parallelize across sequence length:
- `flash_decode_splitk_fp16_d256`: Grid(Hk, num_splits), Block(32, gqa_ratio). Each block handles seq_len/num_splits tokens. Writes partial output (float) + meta (max_score, sum_exp) per split.
- `flash_decode_splitk_reduce_d256`: Grid(Hk), Block(32, gqa_ratio). Combines partial results with online softmax correction.
- `num_splits = min((seq_len + 63) / 64, 16)` — auto-selected in non-graph path
- Graph path: always uses MAX_SPLITS=16 (empty splits exit early)

**Standalone kernel benchmark (isolated flash_decode only):**

| seq_len | Original | Split-K | Speedup |
|---------|----------|---------|---------|
| 128 | 0.048ms | 0.029ms | 1.7x |
| 256 | 0.085ms | 0.026ms | 3.2x |
| 512 | 0.163ms | 0.027ms | 6.0x |
| 1024 | 0.324ms | 0.029ms | 11.0x |
| 2048 | 0.654ms | 0.051ms | 12.8x |
| 4096 | 1.540ms | 0.091ms | 16.9x |

**Full model decode (C++ path, g64 wagi):**

| Kontekst | Przed split-K | Po split-K | Poprawa |
|----------|---------------|------------|---------|
| 128 | 29.4 t/s | 29.8 t/s | +1.4% |
| 256 | 28.7 t/s | 30.1 t/s | +4.9% |
| 512 | 27.5 t/s | 30.0 t/s | +9.1% |
| 1024 | 25.2 t/s | 29.9 t/s | +18.7% |
| 2048 | ~22 t/s | 29.6 t/s | +34.5% |

**Full model decode (HIP Graph path, g64 wagi):**

| Kontekst | HIP Graph + split-K | GGUF Q4_K_M Vulkan | Różnica |
|----------|---------------------|-------------------|---------|
| 128 | 31.0 t/s | 26.1 t/s | **+18.8%** |
| 256 | 30.9 t/s | ~26 t/s | **+18.8%** |
| 512 | 30.8 t/s | ~26 t/s | **+18.5%** |
| 1024 | 30.5 t/s | ~26 t/s | **+17.3%** |
| 2048 | 30.2 t/s | ~24 t/s | **+26.3%** |

**Kluczowy wniosek:** Split-K sprawia że flash_decode jest praktycznie stały (~0.03ms) dla ctx≤1024. Razem z DeltaNet (stały koszt per token) osiągamy **prawie zero degradacji z kontekstem** — od 31.0 t/s (ctx=128) do 30.2 t/s (ctx=2048), tylko 2.6% spadek. GGUF degraduje 26.1→24 t/s = 8% spadek w tym samym zakresie.

**Numerical precision:** Split-K daje max abs diff ~0.000122 vs oryginał (FP16 rounding from float partial → half output). Negligible — PPL identyczny.

**Pliki:**
- `hip_int4/int4_decode_step.hip` — flash_decode_splitk_fp16_d256, flash_decode_splitk_reduce_d256, graph-compat _g variants. Auto-dispatch: seq_len≤256 uses original, >256 uses split-K. Graph always uses split-K (fixed grid).
- Buffers: splitk_out [MAX_SPLITS*H*D] float, splitk_meta [MAX_SPLITS*H*2] float (~300KB)

---

### 12. Fused INT4→FP16 WMMA GEMM dla prefill — CZĘŚCIOWY SUKCES ⚠️

**Problem:** Prefill pp128 był 39% wolniejszy od GGUF (395 vs 641 t/s). Profiling per-operacja pokazał że gate_up [34816×5120] zajmuje 1.96ms per warstwa (dequant: 0.94ms + rocBLAS GEMM: 0.96ms), co daje 126ms/64 warstw = 39% całego pp128.

**Próby odrzucone:**
1. **W4A8 WMMA (INT4 weights × INT8 activations):** 1.41x szybszy GEMM, ale kwantyzacja aktywacji do INT8 powoduje max_diff=7.4. Niedopuszczalna jakość.
2. **Stream overlap (dequant na stream1, GEMM na stream0):** 0.95x — obie operacje saturują GPU, overlap nie pomaga.

**Rozwiązanie: fused INT4→FP16 WMMA GEMM (`fused_int4_wmma_f16`)**

Kernel dequantyzuje INT4 do FP16 w rejestrach i od razu wykonuje `v_wmma_f32_16x16x16_f16`. Eliminuje osobny dequant pass.

- 1 wave per block, 4 N-tiles (64 output columns)
- Grid: (ceil(N_pad/64), ceil(M/16)), Block: (32)
- INT4→FP16 dequant: nibble extraction + `(val - zero) * scale` per element
- Reuses w4a8_prepare_weights tiling format (pre-tiled during model loading)
- Poprawiony `w4a8_prepare_weights` — teraz automatycznie wykrywa group_size ze skali (nie hardcoded g32)

**Kluczowy bug fix:** RDNA WMMA C output layout: row=(lane/16)*8+j, col=lane%16. Oryginalny write miał M i N indeksy zamienione.

**Wyniki kernel benchmark M=128 g64:**

| Projekcja | Rozmiar | dequant+rocBLAS | fused WMMA | Speedup |
|-----------|---------|----------------|------------|---------|
| gate_up | 34816×5120 | 1.96ms | 1.24ms | **1.56x** |
| down | 5120×17408 | 0.87ms | 0.73ms | **1.20x** |
| qkv | 6144×5120 | 0.24ms | 0.24ms | 1.01x |
| o_proj | 5120×6144 | 0.27ms | 0.24ms | 1.12x |

**Ograniczenia:**
- Wymaga pre-tiled weights (+6.2 GB VRAM dla gate_up × 64 warstw)
- Dla M>128 kernel jest wolniejszy od rocBLAS → dispatch M≤128 only
- Kernel osiąga ~15% peak compute (register pressure → niska occupancy)
- v2 z 2 N-tiles (zamiast 4) miał wyższą occupancy ale był wolniejszy — amortyzacja ładowania A jest ważniejsza

**Wynik na modelu pp128:**

| Wersja | pp128 | pp256 | pp512 |
|--------|-------|-------|-------|
| Baseline (dequant+rocBLAS) | 325ms (393 t/s) | 398ms (644 t/s) | 590ms (867 t/s) |
| Fused WMMA (gate_up only) | **282ms (454 t/s)** | 399ms (641 t/s) | 595ms (861 t/s) |
| Improvement | **+15%** | ±0% | ±0% |
| GGUF Q4_K_M Vulkan | 199ms (641 t/s) | 277ms (924 t/s) | 449ms (1140 t/s) |

**Wnioski:**
1. Fused dequant+WMMA eliminuje 50% dequant overhead dla dużych macierzy (gate_up)
2. Zysk ograniczony do M≤128, bo przy większym M rocBLAS jest bardziej efektywny (lepsza tile-based parallelizacja)
3. +6 GB VRAM za 15% pp128 improvement — opłacalny tradeoff (22.7 vs 16.5 GB, mieścimy się w 32 GB)
4. Dalsze optymalizacje prefill wymagają tiling ALL projekcji (+13.5 GB, za dużo) lub C++ prefill eliminujący Python overhead

**Pliki:**
- `hip_int4/int4_decode_step.hip` — `fused_int4_wmma_f16<GS>`, `fused_int4_wmma_f16_v2<GS>` (2-tile), `fused_int4_wmma_gemm` wrapper
- `qwen3_5_27b/int4_engine_hybrid.py` — `use_wmma_prefill` flag, pre-tiling w `__init__`, `gemm_wmma` helper, M≤128 dispatch

### 13. Dalsza optymalizacja prefill pp128 — SUKCES ✅

**Kontekst:** Sekcja 12 dała pp128 282ms (+15% vs baseline 325ms). Dalsze optymalizacje skupiły się na eliminacji pozostałych wąskich gardeł.

**Profiling pp128 per-kernel (M=128, g64, RDNA4 R9700):**

| Kernel | per-layer (ms) | total (ms) | % |
|--------|---------------|------------|---|
| gate_up WMMA | 1.27 × 64 | 81.6 | 29% |
| down dequant+GEMM | 0.93 × 64 | 59.3 | 21% |
| DeltaNet recurrence | 0.99 × 48 | 47.4 | 17% |
| dn_qkv dequant+GEMM | 0.49 × 48 | 23.7 | 8% |
| fa_qkv dequant+GEMM | 0.77 × 16 | 12.3 | 4% |
| dn_o + fa_o | 0.28 × 64 | 18.0 | 6% |
| dn_z | 0.24 × 48 | 11.5 | 4% |
| LM head (dequant+mm) | 11.69 × 1 | 11.7 | 4% |
| norms, SiLU, had, etc | | ~15 | 5% |

**Optymalizacja 1: GEMV dla LM head — 10ms oszczędności**

LM head w prefill dequantyzował cały [248320, 5120] → 2.37 GB FP16, żeby pomnożyć z 1 wektorem (last token). Zmiana na `dbg_gemv` (GEMV kernel z decode): 11.69ms → 1.51ms (**7.8x**). Max diff vs FP16 dequant: 0.0039.

**Optymalizacja 2: WMMA pre-tiling dla QKV — 7ms oszczędności**

WMMA vs dequant+rocBLAS benchmark po rozmiarach:

| Projekcja | N | K | dequant+rocBLAS | fused WMMA | Speedup |
|-----------|------|------|----------------|------------|---------|
| fa_qkv | 14336 | 5120 | 0.77ms | 0.52ms | **1.47x** |
| dn_qkv | 10240 | 5120 | 0.49ms | 0.42ms | **1.19x** |
| dn_z | 6144 | 5120 | 0.25ms | 0.28ms | 0.88x ❌ |
| dn_o | 5120 | 6144 | 0.27ms | 0.30ms | 0.91x ❌ |
| fa_o | 5120 | 6144 | 0.27ms | 0.29ms | 0.92x ❌ |
| down | 5120 | 17408 | 0.93ms | 0.85ms | 1.07x |

**Kluczowy wniosek:** WMMA opłaca się TYLKO dla N ≥ 10240. Przy N ≤ 6144 kernel jest WOLNIEJSZY od rocBLAS (za mało N-tiles do amortyzacji overhead). Pre-tiling dodatkowych QKV: +2.08 GB VRAM za ~7ms oszczędności.

**Odrzucone podejścia:**
1. **Stream-overlapped tiling** — tile weights na async stream podczas attention/recurrence, reusable 150 MB buffers zamiast 6.2 GB pre-tiled. Wynik: **396ms (GORZEJ!)**. Oba operacje (tiling + compute) saturują ten sam memory bus. Na RDNA4 z jednym kontrolerem pamięci, overlap dwóch bandwidth-bound operacji nie działa.
2. **Direct packed format reading (`fused_int4_wmma_f16_direct`)** — czyta z oryginalnego [N, K/2] bez tiling. Wynik: 16.85ms vs 1.24ms tiled (**14x wolniej**). Non-coalesced access (stride=K/2 między wierszami) zabija cache.
3. **Down projection WMMA** — 1.07x speedup, 3.6ms savings za 3.1 GB extra VRAM. Nie warto.
4. **Python C++ prefill loop** — zmierzone 0.6ms overhead na 64 iteracji. Nie warto pisać C++ loopa.

**Końcowy wynik pp128 z wszystkimi optymalizacjami:**

| Wersja | pp128 | pp256 | pp512 | VRAM |
|--------|-------|-------|-------|------|
| Baseline (dequant+rocBLAS) | 325ms (393 t/s) | 398ms (644 t/s) | 590ms (867 t/s) | 16.5 GB |
| + gate_up WMMA | 282ms (454 t/s) | ~398ms | ~590ms | 22.7 GB |
| + GEMV LM head | 273ms (469 t/s) | ~388ms | ~580ms | 22.7 GB |
| + QKV WMMA | **267ms (479 t/s)** | 390ms (657 t/s) | 589ms (870 t/s) | 25.9 GB |
| **Improvement** | **+22%** | +2% | ±0% | +9.4 GB |
| GGUF Q4_K_M Vulkan | 199ms (641 t/s) | 277ms (924 t/s) | 449ms (1140 t/s) | ~15 GB |

**Decode (bez zmian):** 30.1 t/s graph @ ctx=128 (GGUF: 26.1 t/s → **+15% szybciej**)

**Roofline analysis decode:**
- Total weight reads per step: ~12.8 GB
- Theoretical min @ 507 GB/s: 25.2ms
- Actual (graph): 33.5ms → **76% bandwidth efficiency**
- Dalsze optymalizacje decode wymagałyby szybszego GEMV (VOPD dual-issue) — marginalny zysk

**Wnioski:**
1. **GEMV for LM head** to "free lunch" — 10ms oszczędności, 0 dodatkowego VRAM, trywialna zmiana
2. **WMMA ma próg opłacalności N ≥ ~10K** — poniżej tego rocBLAS jest szybszy (lepszy tile scheduler)
3. **Stream overlap nie działa na RDNA4** — single memory controller, obie operacje bandwidth-bound
4. **Prefill bottleneck = gate_up GEMM (31%)** — już zoptymalizowany WMMA, dalsze zyski wymagają szybszego kernela
5. **Decode jest memory-bound @ 76%** — bliski limitowi, główna przewaga to INT4 GEMV (mniej danych niż GGUF)

**Pliki:**
- `hip_int4/int4_decode_step.hip` — `w4a8_prepare_weights_into()` (tiling do pre-allocated buforów), `fused_int4_wmma_f16_direct<GS>` (odrzucone)
- `qwen3_5_27b/int4_engine_hybrid.py` — `wmma_stream_tile` flag, `hip_qkv_tw/tsz`, `hip_dn_qkv_tw/tsz`, GEMV LM head

---

### 17. MoE Prefill W4A4 V_DOT8 — SUKCES ✅

**Co to:** Zamiana scalar FP32 dequant (`gemm_int4_g64_ts_fused`) na W4A4 z `V_DOT8_I32_IU4` (`__builtin_amdgcn_udot8`) w prefill MoE GEMM (gate_up + down). Aktywacje kwantyzowane online do uint4 (per-group-32 asymetryczne), wagi już INT4 w pamięci.

**Motywacja:**
- Scalar FP32 dequant: 32 nibble extractions + 32 FMA = ~96 ALU ops per 128-bit load, ~70 VGPRs
- V_DOT8: 4× `udot8` per 128-bit load = 4 instrukcje, ~30 VGPRs
- Mniej instrukcji + lepsze occupancy → lepsze ukrywanie latencji pamięci

**Kernel `gemv_w4a4_udot8_v3<BN=4, BM=4>`:** K-outer M-inner loop.
- Ładuje wagi raz per K-chunk, przetwarza BM=4 tokenów w inner loop
- Per-iteration: quantize acts `quantize_activations_u4_v2`, potem V_DOT8 GEMV
- Formuła: `acc += w_sc * (a_sc * raw_u + a_mn * w_sum_u - w_zp * xz_term)`
- Grid: (N/BN, num_experts), Block: 32 (1 warp)

**Wyniki — Qwen3-30B-A3B prefill:**
| Metryka | Przed (scalar) | Po (W4A4) | GGUF Q4_K_M Vulkan |
|---------|---------------|-----------|-------------------|
| pp128 | 872 t/s | **1904 t/s** | 1302 t/s |
| pp256 | — | **2290 t/s** | — |
| pp512 | — | **2320 t/s** | — |
| Speedup | — | **2.18×** | **1.46× vs GGUF** |

**Jakość:** Cosine=0.897 vs Python FP16 reference (identyczne jak scalar baseline). Generuje poprawny tekst (np. "2+2=4", poprawne definicje stack/queue).

**Bug znaleziony i naprawiony:**
- `quantize_activations_u4_v2`: `__shfl(q, lane - 1)` wewnątrz `if (lane & 1)` — na RDNA4 kompilator/hardware może nie propagować wartości z even lanes, bo shuffle jest w conditional branch. Efekt: all even-indexed elements kwantyzowane do 0 (cosine 0.24).
- **Fix:** `int q_neighbor = __shfl(q, lane ^ 1)` POZA conditionem — unconditional shuffle, potem conditional write.
- **Lekcja: Na RDNA4 (gfx1201), `__shfl()` MUSI być unconditional — nie umieszczać wewnątrz if/else które dzieli warp.**

**Próba dequant+rocBLAS bmm — PORZUCONE ❌:**
- Dequantyzacja INT4→FP16 wszystkich 128 expertów pisze 768 MB FP16 @ 507 GB/s = 1.5ms/layer
- bmm batch=128, M=25 osiąga tylko 3.5 TFLOPS (5.8% peak) — za mały M
- `.transpose(1,2)` powoduje non-contiguous bmm → 3× penalty
- **Wniosek: FP16 dequant bandwidth cost > ALU savings for MoE with small per-expert M**

**Dlaczego W4A4 działa a W4A8 nie:**
- W4A8 wymagał repacking INT4→INT8 (~80 int ops overhead per load)
- W4A4 używa V_DOT8_I32_IU4 bezpośrednio z naszego formatu INT4 wag — zero overhead

**Pliki:**
- `hip_int4/int4_decode_step.hip` — `quantize_activations_u4_v2`, `gemv_w4a4_udot8_v3<BN,BM>`, `prefill_moe_logits` (linie 5716-5773)
- `qwen3_30b_a3b/int4_engine_moe.py` — `fast_prefill_cpp()` wywołuje C++ prefill

---

### 14. Profilowanie decode Qwen3.5-27B — analiza overhead i próba fuzji

**Kontekst:** Decode 30.1 t/s z HIP Graph (ctx=128). Roofline: 14.7 GB/token → 34.5 t/s @507 GB/s, 43.5 t/s @640 GB/s peak. Overhead: ~13% ponad efektywny BW roofline.

**Per-kernel GEMV profiling (syntetyczne wagi):**

| Kernel | N | K | ms/call | GB/s | ×layers | total ms |
|--------|-----|------|---------|------|---------|----------|
| dn_qkv | 10240 | 5120 | 0.034 | 850 | 48 | 1.65 |
| dn_z | 6144 | 5120 | 0.023 | 763 | 48 | 1.10 |
| dn_o | 5120 | 6144 | 0.025 | 710 | 48 | 1.20 |
| dn_gate_up | 34816 | 5120 | ~0.17* | ~580 | 48 | ~8.0 |
| dn_down | 5120 | 17408 | ~0.09* | ~580 | 48 | ~4.1 |
| fa_* (16L) | - | - | - | - | - | ~5.0 |
| lm_head | 248320 | 5120 | ~1.2* | ~580 | 1 | ~1.2 |
| **TOTAL GEMV** | | | | | | **~22 ms** |

*estymowane na podstawie BW dużych macierzy

**Obserwacje:**
- Małe macierze (N≤16384) osiągają 700-920 GB/s dzięki cache'owaniu X w L2
- Duże macierze (N≥34816) spadają do ~580 GB/s — czyste DRAM reads
- GEMV sum: ~22 ms, actual decode: ~33 ms → **~11 ms overhead (33%)** z DeltaNet state I/O, flash attention, małe kernele

**Bug: GPU hang dla N > ~20000 z dbg_gemv (standalone)**
- `gemv_warp<4, 64>` z BLOCK_N=4 → N=34816 daje 8704 bloków
- RDNA4 (gfx1201) hanguje przy >~5000 bloków z 32 threads (1 warp) per block
- **NIE dotyczy decode z HIP Graph** — graph capture obsługuje duże dispatche poprawnie
- Dotyczy tylko standalone benchmark (dbg_gemv)
- Workaround: benchmarkowanie z N≤19904, ekstrapolacja BW dla większych

**Próba fuzji FFN kerneli — PORZUCONE ❌**

Idea: Zastąpić 4 kernele FFN (res_norm + GEMV_gate_up + SiLU_Had + GEMV_down) dwoma fused kernelami:
- `gemv_warp_res_norm_had<BLOCK_N, GS>` — fused residual+RMSNorm+FWHT+GEMV
- `gemv_warp_silu_had<BLOCK_N, GS>` — fused SiLU+Hadamard+GEMV

Dodano template `GS` parameter (domyślnie 32, nowe: 64) do obu kerneli + update launch functions.

**Wyniki:** Decode spadł z ~30 t/s do 23.5 t/s (−21.7%)!

**Dlaczego fuzja jest wolniejsza:**
- `gemv_warp_res_norm_had` wykonuje DWA przebiegi przez K: Pass 1 (sum-of-squares), Pass 2 (norm+FWHT+dot)
- Podwaja czas per-blok, a workload jest memory-bound → GPU nie może overlappować
- Input (A, B, NormW) czytane z L2 cache (10KB), ale podwójny loop w każdym bloku blokuje pipe na dłużej
- Niezfuzowana wersja: norm kernel (1 blok, szybki) pisze `rot` → GEMV czyta `rot` (kolejny kernel, pełne memory BW)
- **Lekcja: Fuzja kerneli NIE pomaga gdy fused kernel ma 2× loop overhead i jest memory-bound. Lepiej pozwolić GPU pipelinować oddzielne kernele.**

**Zmiany zachowane (kompatybilność wsteczna):**
- Template `<int BLOCK_N, int GS=32>` w `gemv_warp_res_norm_had` i `gemv_warp_silu_had`
- Launch functions `launch_gemv_res_norm_had(..., gs)` i `launch_gemv_silu_had(..., gs)` z g64 support
- Decode loop cofnięty do oddzielnych kerneli (nie fused)

---

### 15. KV cache scaling — hybrid DeltaNet+FullAttn advantage

**Architektura Qwen3.5-27B:** 48 warstw DeltaNet (O(1) state) + 16 warstw FullAttn (KV cache skaluje z kontekstem).

**Koszt KV cache (16 warstw × 4 KV heads × head_dim=256 × FP16):**

| max_seq | KV cache | Total VRAM | Dodatkowe vs 2048 |
|---------|----------|------------|-------------------|
| 2048 | 134 MB | 17.3 GB | baseline |
| 8192 | 537 MB | 17.4 GB | +0.4 GB |
| 32768 | 2.1 GB | 17.4 GB* | +2.0 GB |
| 65536 | 4.3 GB | ~19.5 GB | +4.1 GB |

*lazy allocation — actual VRAM zależy od used context

**Decode speed vs context length:**

| Context | Decode t/s | Komentarz |
|---------|-----------|-----------|
| ~10 | 29.6 | baseline |
| ~60 | 29.4 | negligible degradation |
| ~210 | 29.0 | still stable |

**Kluczowa przewaga:** 48/64 warstw (75%) to DeltaNet z O(1) kosztem per token. Tylko 16 warstw (25%) ma KV cache. W porównaniu z dense transformerem, degradacja z kontekstem jest ~4× mniejsza.

**Flash decode D=256:** split-K wariant aktywuje się dla seq_len > 256. Przy ctx=4096 16 warstw × 4 heads × 4096 seq × 256 dim × 2B × 2 (K+V) = 1 GB → ~1.7 ms @580 GB/s, co daje ~95% of non-context baseline speed.

**Parametr:** `max_seq` w konstruktorze `Qwen35HybridEngine`. Default=2048, max testowane=32768.

---

## 16. DeltaNet Kernel Fusion (decode optimization)

**Problem:** DeltaNet decode path miał 17 kernel launches per layer × 48 warstw = 816 launches. Wiele z nich to drobne operacje (l2norm na 16×128 elementach, repeat_interleave, fp16↔fp32 konwersje) — dominował launch overhead, nie compute.

**Rozwiązanie:** 3 fused kernele zastępujące 8 oddzielnych:

1. **`deltanet_prep_qkv_fp32`** — fused l2norm + repeat_interleave + fp16→fp32 dla Q, K, V. Jeden blok (32 wątki) per output head. Redundantnie liczy l2norm dla shared source heads (3× dla rep=3), ale taniej niż osobne launche.

2. **`gated_delta_net_step_fused`** — fused compute_ssm_params (decay, beta z a_in, b_in, A_log, dt_bias) + gated_delta_net_step. Eliminuje 1 launch per layer i intermediate bufory.

3. **`deltanet_post_rmsnorm`** — fused fp32→fp16 + rmsnorm + gated SiLU. Czyta FP32 bezpośrednio, liczy RMSNorm in-register, aplikuje gate.

**Wynik:** 17→9 launches per DeltaNet layer. 384 launchów mniej total.
- Decode: ~29 t/s → **30.5 t/s** (+5%)
- Każdy zaoszczędzony launch ≈ 2 µs → 384 × 2 µs ≈ 0.77 ms saved

**Lekcja:** Przy >800 kernel launches per token, nawet drobne kernele mają znaczący koszt. Fuzja compute-bound kerneli (nie bandwidth-bound!) daje realne zyski.

---

## 17. BLOCK_N=8 GEMV dla dużych macierzy

**Problem:** `gemv_warp<4>` dla macierzy z N≥8192 (gate_up: 34816, lm_head: 248320) generuje dużo bloków, a wektor X jest redundantnie czytany z DRAM per blok.

**Rozwiązanie:** `BLOCK_N=8` dla N≥8192 — każdy warp przetwarza 8 wierszy zamiast 4. Podwaja amortyzację X vectora, zmniejsza block count o 2×.

```cpp
if (N >= 8192) {
    hipLaunchKernelGGL((gemv_warp<8, 64>), dim3((N+7)/8), dim3(32), 0, st, ...);
} else {
    hipLaunchKernelGGL((gemv_warp<4, 64>), dim3((N+3)/4), dim3(32), 0, st, ...);
}
```

**Wynik:** 30.5 → **31.1 t/s** (32.1 ms/tok), kolejne +2%.

**Lekcja:** Przy decode (batch=1), GEMV jest czysto bandwidth-bound, więc redukcja redundantnego czytania X to darmowa optymalizacja.

---

## 18. Tiled dequant+GEMM dla prefill (L2 cache friendly)

**Problem:** Prefill dequantyzuje całą macierz wag INT4→FP16 do DRAM, potem GEMM czyta ją ponownie z DRAM. Dla gate_up (34816×5120): dequant pisze 340 MB FP16, GEMM czyta 340 MB → 680 MB dodatkowego ruchu DRAM.

**Rozwiązanie:** Tiled dequant+GEMM z N_tile=4096:
1. Dequant tylko N_tile=4096 wierszy → 40 MB FP16 (mieści się w L2 cache gfx1201)
2. GEMM natychmiast czyta z L2 (hit rate ~100%)
3. Następny tile nadpisuje ten sam bufor

**Testowane tile sizes:**
| N_tile | gate_up ms | vs baseline |
|--------|-----------|-------------|
| full (34816) | 2.28 ms | baseline |
| 8192 | 1.62 ms | -29% |
| 4096 | 1.45 ms | **-37%** |
| 2048 | 1.51 ms | -34% |

**Wynik prefill:** pp128: 405 → **443 t/s** (+9%)

**Lekcja:** L2 cache (4 MB na gfx1201) to klucz do prefill. Tile size musi być dobrany tak, żeby FP16 tile mieścił się w L2 (4096×5120×2B = 40 MB → za duże? ale GEMM czyta kolumnami, więc working set jest mniejszy). Zbyt małe tile (2048) dają overhead z wielu rocBLAS wywołań.

---

## 19. Failed approaches: stream overlap i fused WMMA direct

### 19a. Overlapped dequant (HIP stream overlap)
**Idea:** Dequant na stream B podczas gdy GEMM na stream A przetwarza poprzedni tile.
**Wynik:** 0% speedup.
**Powód:** Oba streamy konkurują o ten sam kontroler DRAM bandwidth. GPU ma 1 memory controller — overlapping nie daje nic gdy bottleneck to DRAM BW, nie compute.

### 19b. Fused INT4 WMMA GEMM direct
**Idea:** Pominąć dequant, czytać INT4 packed bezpośrednio w WMMA GEMM kernelu. Dequant inline per tile.
**Wynik:** 7× wolniejsze niż dequant+rocBLAS.
**Powód:** Weight layout [N, K/2] powoduje strided (non-coalesced) access w K-dimension. rocBLAS operuje na contiguous FP16 tiles zoptymalizowanych pod coalesced access. Żeby to naprawić, trzeba by zmienić weight layout na tile-friendly (np. [N_tile, K_tile/2] z padding), ale to komplikuje cały pipeline kwantyzacji.

**Lekcja:** Nie próbować bić rocBLAS na GEMM — to lata optymalizacji. Lepiej minimalizować I/O dookoła rocBLAS (tiling, L2 reuse).

---

## 20. Aktualny stan wydajności (2026-03-16)

### Decode (HIP Graph, ctx~100)
| Metryka | Wartość |
|---------|---------|
| Decode speed | **30.9 t/s** (32.4 ms/tok) |
| Effective BW | 454 GB/s (70.9% peak 640) |
| GGUF Q4_K_M Vulkan | 26.15 t/s |
| **Przewaga vs GGUF** | **+18%** |
| Dane per token | 14.7 GB |
| Roofline @507 GB/s | 29.0 ms → 34.5 t/s |
| Roofline @640 GB/s | 23.0 ms → 43.5 t/s |

### Prefill (pp128)
| Metryka | Wartość |
|---------|---------|
| Prefill speed | **443 t/s** |
| GGUF Q4_K_M Vulkan | 657 t/s |
| **vs GGUF** | **-33%** (still behind) |

### VRAM
| Metryka | Wartość |
|---------|---------|
| Total | **17.3 GB** / 32 GB |
| Wagi INT4 g64 | ~14 GB |
| DeltaNet states | 72 MB |
| KV cache (default 2048) | 134 MB |

### Jakość
| Metryka | Wartość |
|---------|---------|
| PPL WikiText-2 (g64) | 6.52 |

### Remaining decode gap analysis (updated 2026-03-16)
- Roofline limit @640 GB/s: 22.6 ms (44.2 t/s)
- Achieved: 31.9 ms (31.4 t/s)
- Gap: 9.3 ms
  - GEMV not reaching peak BW (~500 vs 640): **~5 ms** (per-category: gate_up 93%, QKV 80-87%, O/Z/down 79-83%)
  - Pipeline transitions GEMV↔small kernels: **~3 ms** (128 transitions × ~23 µs)
  - DeltaNet state I/O + compute: **~0.6 ms** (302 MB at 500 GB/s)
  - HIP Graph dispatch overhead: **~0.8 ms** (851 kernels × ~1 µs)

### 20. Adaptive BLOCK_N for GEMV — NO EFFECT ⚠️

**Hipoteza:** Mniejsze macierze (O projection N=5120, Z N=6144) mają za mało bloków z BLOCK_N=4. BLOCK_N=2 daje 2× więcej bloków = lepszy memory parallelism.

**Wyniki (syntetyczne, Infinity Cache inflated):**
| Macierz | BN=4 GB/s | BN=2 GB/s | Zmiana |
|---------|-----------|-----------|--------|
| O_proj 5120×6144 | 419 | 447 | +7% |
| fa_o 5120×6144 | 419 | 451 | +8% |
| down 5120×17408 | 570 | 541 | **-5%** |

**Wyniki (real decode, HIP Graph):** 32.1 ms → 32.1 ms = **ZERO różnicy** w real decode!

**Dlaczego:** Syntetyczne benchmarki z back-to-back identycznym GEMV influją BW przez 64 MB Infinity Cache. W real decode, między GEMVami są inne operacje (norm, attention, state update) które evictują cache. BLOCK_N nie wpływa na cold-cache BW.

**Wniosek:** Infinity Cache inflation makes synthetic GEMV benchmarks unreliable. Always measure end-to-end decode. Zostawiony BLOCK_N=8 dla N≥8192 (mniejszy grid = mniej header overhead), BLOCK_N=4 reszta.

### 21. FFN GEMV fusion (norm+GEMV, SiLU+GEMV) — REGRESJA 🔴

**Hipoteza:** Pipeline transitions between small kernels (norm, SiLU) and GEMV cost ~3ms total (128 transitions × ~23µs). Fusing norm INTO the GEMV (gemv_warp_res_norm_had, gemv_warp_silu_had) eliminates transitions.

**Wyniki:** 31.9 ms → **37.2 ms** (+5.3 ms = **17% regresja**)

**Dlaczego:** Fused kernels need many more VGPRs:
- gemv_warp_res_norm_had: 2 passes (sum-of-squares + normalized GEMV), extra registers for A/B residual, NormW, rrms, FWHT
- gemv_warp_silu_had: inline SiLU + FWHT computation per iteration
- Result: occupancy drops from ~6 waves/SIMD to ~4 waves/SIMD
- Lower occupancy = fewer outstanding memory requests = worse BW utilization
- The BW loss (~20%) completely overwhelms the transition savings (~3 ms)

**Wniosek:** Na RDNA4 (32 VGPRs per lane, Wave32) nie wolno dodawać compute do memory-bound GEMV kernels — occupancy jest zbyt ważny. Podejście to było testowane DWA RAZY (sekcja 14 i tutaj) — za każdym razem regresja. **NIGDY WIĘCEJ nie próbować fusji compute→GEMV na RDNA4.**

### 22. Analiza wydajności GEMV per rozmiar macierzy (cold cache)

Pomiary z `bench_gemv_decode.py` z cache flush, gs=64:

| Macierz | N | K | ms | GB/s | % peak | Uwagi |
|---------|------|-------|--------|------|--------|-------|
| gate_up | 34816 | 5120 | 0.169 | 594 | 93% | Duży N = dużo bloków |
| down | 5120 | 17408 | 0.094 | 534 | 83% | Duży K, mało bloków |
| fa_qkv | 14336 | 5120 | 0.079 | 521 | 81% | OK |
| dn_qkv | 10240 | 5120 | 0.057 | 515 | 80% | Średni |
| O_proj | 5120 | 6144 | 0.035 | 513 | 80% | Mały N |
| Z_proj | 6144 | 5120 | 0.035 | 507 | 79% | Mały N |
| lm_head | 248320 | 5120 | 1.369 | 522 | 82% | Ogromny ale nie najszybszy |

**Obserwacja:** Gate_up (93%) jest jedynym GEMV blisko peak. Mniejsze macierze ~80%. LM head też 82% mimo ogromnego N — prawdopodobnie memory controller page switching overhead.

### 23. QKV+Z weight concatenation (DeltaNet) — ZREALIZOWANE ✅

**Idea:** Połączyć macierze QKV (N=10240, K=5120) i Z (N=6144, K=5120) w jedną macierz (N=16384, K=5120) dla DeltaNet warstw. Jeden GEMV zamiast dwóch → 48 mniej launches per decode.

**Implementacja:**
- C++ graph decode: combined GEMV, Z output czytane z offsetu `qkv + conv_dim`
- Python engine: `hip_dn_qkvz_w/s/N` dicts z combined tensorami
- Zero-copy views: QKV i Z jako slices combined tensora (brak duplikacji VRAM)
- `qkv_buf` powiększony: `max(q_dim*2 + kv_dim*2, conv_dim + val_dim)`

**Wyniki:**
- VRAM: -2.5 GB dzięki zero-copy views (27.1 GB vs 29.6 GB z duplikatami)
- Decode: ~0.3 ms oszczędności z 48 mniej launches (w szumie pomiarowym)

### 24. g32 vs g64 — porównanie group sizes

| Metryka | g32 | g64 |
|---------|-----|-----|
| Decode (graph) | 33.4 ms = 29.9 t/s | **32.4 ms = 30.9 t/s** |
| VRAM | 27.1 GB | **24.6 GB** |
| vs GGUF | +14% | **+18%** |

g64 jest szybsze o 1 ms dzięki 2× mniejszym tensorom scale (mniej danych do czytania per GEMV).

### 25. Persistent kernel benchmark — UMIARKOWANY EFEKT ⚠️

Test z `test_coop4.hip` (cooperative groups, grid sync):
- 40 warstw: 0.22 ms saved (2.5%)
- 80 warstw: 0.49 ms saved (2.9%)
- Per-transition: ~16 µs saved

**Wniosek:** Persistent kernel eliminuje launch overhead ale BW dominuje czas. Dla naszego modelu (64 warstw × ~13 kerneli = ~820 launches w graph) szacunkowa oszczędność: ~0.5-1 ms. Nie warto ogromnego nakładu pracy.

### 26. Szczegółowy rozbicie decode overhead (2026-03-17)

Pomiary z prawdziwymi wagami modelu (g32, 257 GEMVs po QKV+Z concat):

| Komponent | Czas | % |
|-----------|------|---|
| GEMV (real weights, 257 calls) | 28.9 ms | 86.5% |
|  - Python dispatch overhead | ~2.0 ms | — |
|  - Czyste GPU GEMV | ~26.9 ms | — |
| Non-GEMV kernels | ~4.5 ms | 13.5% |
|  - rmsnorm_had (128×) | ~0.7 ms | — |
|  - silu_hadamard_ds (64×) | ~0.5 ms | — |
|  - DeltaNet (conv+prep+delta+post, 48×) | ~1.2 ms | — |
|  - FP16 small GEMVs (proj_a/b, 96×) | ~0.5 ms | — |
|  - Flash decode (16×) | ~0.5 ms | — |
|  - Residual adds + other | ~1.1 ms | — |
| Graph dispatch (~820 nodes × ~2 µs) | ~1.6 ms | — |
| **Total graph decode** | **33.4 ms** | **100%** |

**Wniosek:** 86.5% czasu to GEMV. Non-GEMV overhead (4.5 ms) to głównie compute-bound małe kernele + pipeline transitions. Graph dispatch dodaje ~1.6 ms. Nie ma łatwej ścieżki do dalszej optymalizacji — jesteśmy blisko hardware ceiling.

### Aktualny stan performance (2026-03-17)

| Metryka | g32 | g64 |
|---------|-----|-----|
| **Decode (graph)** | **33.4 ms = 29.9 t/s** | **32.4 ms = 30.9 t/s** |
| GGUF Q4_K_M Vulkan | 26.13 t/s | 26.13 t/s |
| **Przewaga vs GGUF** | **+14%** | **+18%** |
| VRAM | 27.1 GB | 24.6 GB |
| GEMV BW utilization | 77.5% peak | ~80% peak |
| PPL WikiText-2 | ~6.4 | 6.52 |

### Pozostałe ścieżki optymalizacji

1. **g128 re-kwantyzacja:** kolejne -0.5 ms ze scale reduction. Ryzyko jakości.
2. ~~**Prefill optimization:** 443 t/s vs GGUF 657 t/s (-33%).~~ → Patrz Lekcja 27.
3. ~~**Fused INT4 WMMA prefill:**~~ → Już aktywne dla gate_up + QKV (Lekcja 27).
4. **Realistyczny ceil:** ~30 ms = 33 t/s (85% BW + 3 ms overhead). Obecne 32.4 ms (g64) jest 8% od tego.

---

### 27. Prefill WMMA profiling i optymalizacja (2026-03-17)

#### Problem

Prefill pp128 = 454 t/s vs GGUF 651 t/s (-30%). Potrzebna analiza gdzie idzie czas i czy WMMA można rozszerzyć.

#### Profiling: GEMM method comparison (fused WMMA vs tiled dequant+rocBLAS)

**gate_up GEMM [M, 5120] × [5120, 34816]:**

| M | Tiled dequant+rocBLAS | Fused WMMA (pre-tiled) | Speedup |
|---|---|---|---|
| 32 | 3.27 ms | 0.39 ms | **8.5×** |
| 64 | 1.39 ms | 0.66 ms | **2.1×** |
| 128 | 1.41 ms | 1.34 ms | 1.05× |

**down_proj GEMM [M, 17408] × [17408, 5120]:**

| M | Tiled dequant+rocBLAS | Fused WMMA (pre-tiled) | Speedup |
|---|---|---|---|
| 32 | 0.89 ms | 0.48 ms | **1.9×** |
| 64 | 0.90 ms | 0.46 ms | **2.0×** |
| 128 | 1.01 ms | 0.80 ms | 1.3× |

**Wniosek:** WMMA jest masywnie szybsze przy małym M (do 8.5× dla gate_up przy M=32), ale konwerguje z tiled dequant przy M=128.

#### Per-layer time breakdown (DeltaNet, M=128)

| Komponent | Czas (ms) | % |
|-----------|-----------|---|
| gate_up WMMA [N=34816] | 1.344 | 26% |
| DeltaNet recurrence | ~1.75 | 34% |
| down tiled [N=5120] | 0.792-1.01 | 16-20% |
| QKV WMMA [N=10240] | 0.419 | 8% |
| o_proj tiled [N=5120] | 0.333 | 7% |
| z_proj tiled [N=6144] | 0.260 | 5% |
| norms + hadamard + silu | ~0.2 | 4% |
| **Total per layer** | **~5.1 ms** | |

#### Próba: WMMA dla down_proj

**Problem:** Pre-tiling down_proj kosztuje 3.57 GB VRAM (55.7 MB × 64 layers), co podnosi VRAM z 27.1 do 30.5 GB. Decode (LM head dequant 2.4 GB) powoduje OOM.

**Próba 1: Permanent pre-tiling at load** → OOM during decode ❌
**Próba 2: On-demand per-layer tiling** → tiling cost 0.44 ms/layer ≈ WMMA savings → net zero ❌
**Próba 3: Batch pre-tile at prefill start, free after** → 28 ms overhead + memory pressure → neutral ❌

**Wniosek:** `w4a8_prepare_weights_into` kosztuje 0.44 ms per down_proj layer — za dużo na on-demand. Pre-tiling wymaga 3.57 GB VRAM — za dużo na permanentne przechowywanie z 27.1 GB base.

#### Porównanie trybów prefill (Qwen3.5-27B, g32)

| Seq | No WMMA | Default WMMA | Stream-tiled | GGUF Vulkan |
|-----|---------|--------------|--------------|-------------|
| pp32 | 138 t/s | **212 t/s** | 124 t/s | 264 t/s |
| pp64 | 255 t/s | **337 t/s** | 216 t/s | 394 t/s |
| pp128 | 445 t/s | **456 t/s** | 326 t/s | 651 t/s |
| pp256 | 703 t/s | **689 t/s** | 709 t/s | 665 t/s |
| pp512 | 904 t/s | **897 t/s** | 911 t/s | 739 t/s |
| VRAM | 18,024 MB | 27,115 MB | 20,279 MB | — |

**Default WMMA** (pre-tiled gate_up + QKV, tiled dequant for rest) = najlepszy kompromis.
**Stream-tiled** = katastrofa przy krótkich sekwencjach (tiling na async stream nie nadąża).

#### Dlaczego pp128 nadal -30% vs GGUF

**DeltaNet recurrence = 34% czasu prefill** (~1.75 ms × 48 layers = 84 ms). Jest to sekwencyjne przetwarzanie stanu (state update zależy od poprzedniego tokena), którego nie da się zparalelizować jak standardowy attention.

GGUF z pełnym attention: cały prefill jest parallel (matmul). Nasz hybrid DeltaNet + FullAttn: 48/64 warstw ma sekwencyjny overhead. To jest architektoniczna cena za szybki decode (O(1) per token vs O(n) dla attention).

**Trade-off:** DeltaNet daje ~14-18% szybszy decode kosztem ~30% wolniejszego short-prefill. Przy pp≥256 nasz prefill bije GGUF (+3-21%) bo GEMM dominuje nad recurrence.

#### Aktualny stan performance (2026-03-17)

| Metryka | Nasz INT4 (g32) | GGUF Q4_K_M Vulkan | Różnica |
|---------|-----------------|--------------------| --------|
| **Decode (graph)** | **29.9 t/s** | 26.13 t/s | **+14%** |
| **Decode (g64)** | **30.9 t/s** | 26.13 t/s | **+18%** |
| pp32 | 212 t/s | 264 t/s | -20% |
| pp64 | 337 t/s | 394 t/s | -14% |
| pp128 | 456 t/s | 651 t/s | -30% |
| pp256 | 689 t/s | 665 t/s | **+4%** |
| pp512 | 897 t/s | 739 t/s | **+21%** |
| VRAM | 27,115 MB | — | — |

---

## Lekcja 18: W4A4 V_DOT8 GEMV — regresja, nie optymalizacja (2026-03-16)

### Hipoteza

Zamiana FP16 scalar dequant (~96 ALU ops) na W4A4 udot8 (4 instrukcje V_DOT8_I32_IU4) w Expert GU/DN GEMV zmniejszy ALU i VGPR → lepsze occupancy → lepsze ukrywanie latencji pamięci → wyższy throughput.

### Implementacja

1. **Nowe kernele**: `gemv_mw_batch_g64_w4a4<1>` (GU, multi-warp 128 threads) i `gemv_batch_xbatch_g64_w4a4<2>` (DN, 32 threads)
2. **Quantize activations**: `quantize_activations_u4_v2` — FP16→uint4 asymmetric per-group-32
3. **Precomputed w_sum**: `precompute_wsum_kernel` — sum(weight_nibbles) per N×(K/32), +1.74 GB VRAM
4. **96 dodatkowych kernel launchów** per token (48 layers × 2 quantize)

### Wynik

| Metryka | FP16 scalar | W4A4 udot8 | Zmiana |
|---------|-------------|------------|--------|
| ctx=128 graph | **175.6 t/s** | **163.4 t/s** | **-7.0%** |
| ctx=512 graph | ~172 t/s | 159.5 t/s | -7.3% |
| VRAM | 18.5 GB | 20.3 GB | +1.7 GB |

### Analiza root cause

**GEMV (M=1) jest 100% memory-bound.** Redukcja ALU nie pomaga bo GPU i tak czeka na pamięć.

**Bandwidth per row (GU, K=2048)**:
- FP16 scalar: W (1024B) + S (128B) = 1152 B/row
- W4A4 z precomputed w_sum: W (1024B) + S (128B) + WS (128B) = **1280 B/row (+11%)**
- W4A4 inline w_sum: W (1024B) + S (128B) = 1152 B/row (identycznie)

**Dodatkowy overhead**:
- w_sum precompute loads: +11% bandwidth → +0.168 ms/tok
- 96 quantize kernel launches (pipeline bubbles): +0.288 ms/tok
- **Łączny overhead: ~0.456 ms/tok** (zmierzone: 0.461 ms — idealnie się zgadza)

### Kluczowa lekcja

**Dla bandwidth-bound GEMV (M=1), jedyne co może pomóc to zmniejszenie ILOŚCI danych do załadowania (mniejsze wagi, większe grupy kwantyzacji) lub lepsze wykorzystanie bandwidth (memory access patterns, prefetch).** Zmniejszenie ALU nie daje nic, bo ALU jest za darmo w shadow memory latency.

W4A4 ma sens TYLKO dla:
- **Prefill (M>1)**: ALU-bound, udot8 daje realny speedup
- **Matmul (GEMM)**: większe tile, ALU-bound fragmenty
- **Nie** dla decode GEMV z M=1

### Status

Reverted. Kernele W4A4 zachowane w kodzie (używane w prefill dispatch). Produkcyjny decode używa FP16 scalar dequant.

---

## Lekcja 19: Transponowanie skal QKV/O + RPW=2 — bez efektu (2026-03-16)

### Hipoteza 1: Transponowanie skal

Profiler pokazał QKV GEMV na 26.5% BW, O-proj na 37.2%, vs Expert GU na 77%. Podejrzenie: row-major skale `[N, K/64, 2]` powodują scattered access, a Expert GU używa transponowanych `[K/64, N, 2]` (coalesced).

**Wynik**: Transponowanie skal QKV/O do `[K/64, N, 2]` — **brak mierzalnej różnicy** w graph replay (175.4 t/s ≈ 175.6 t/s poprzednio).

**Dlaczego nie pomogło**: Profiler HIP events dodają ~2-4µs overhead per event × 48 layers = 96-192µs per kernel type. To sztucznie zawyża zmierzony czas małych kerneli (QKV, O-proj), powodując pozornie niski BW utilization. W graph replay kernele pipeline'ują się i overhead znika.

**Wniosek**: Transponowanie zostawione (nie szkodzi, jest poprawniejsze semantycznie). Profiler HIP event measurements trzeba traktować z rezerwą — nie odzwierciedlają graph performance.

### Hipoteza 2: Multi-row per warp (RPW=2)

Jeden warp ładuje X raz, przetwarza 2 rzędy W → ~40% mniej memory traffic per output.

**Wynik**: **163.4 t/s** — regresja 7% vs 176 t/s (RPW=1).

**Dlaczego regresja**: `xv[32]` float array = 32 VGPRs dla cached X. Łącznie ~50+ VGPRs → max 5 wavefronts/SIMD vs 12 przy RPW=1 (~20 VGPRs). Occupancy 2.5× gorsza → dużo gorsza latency hiding → wolniejsze.

**Kluczowa lekcja RDNA4**: Na architekturze z 256 VGPRs/SIMD, occupancy jest krytyczna. Kernel z ~20 VGPRs (12 wavefronts) bije kernel z ~50 VGPRs (5 wavefronts) nawet jeśli ten drugi ma 40% mniej memory traffic. **Optymalizacja GEMV na RDNA4 = minimalizacja VGPRs.**

### Aktualny stan (2026-03-16)

| Metryka | Wynik |
|---------|-------|
| **Custom INT4 decode (graph, ctx=128)** | **175-176 t/s** |
| GGUF Q4_K_M Vulkan (llama-bench, tg128) | **142 t/s** |
| **Przewaga** | **+24%** |
| VRAM | 18.5 GB |
| Model | Qwen3-30B-A3B, 48 MoE layers, 128 exp, top-8 |
| GPU | AMD R9700 (RDNA4, gfx1201, 32 CU, 32GB) |

### Profil per-kernel (48 layers total, HIP events, ctx=128)

| Kernel | Czas | % | BW util (per-layer) |
|--------|------|---|-----|
| Expert GU | 1.54 ms | 17.2% | 77% |
| Attention | 1.26 ms | 14.1% | — |
| Expert DN | 0.92 ms | 10.3% | 64% |
| QKV | 0.91 ms | 10.2% | ~34%* |
| O-proj | 0.80 ms | 8.9% | ~37%* |
| TopK | 0.58 ms | 6.4% | — |
| Router | 0.50 ms | 5.6% | — |
| Other | 2.47 ms | 27.3% | — |

\* Inflated by HIP event overhead. Graph execution is faster.

### Możliwe dalsze optymalizacje

1. **Persistent mega-kernel** (1 kernel per layer): eliminuje ~0.5 ms dispatch overhead. Trudna implementacja.
2. **Cooperative groups**: grid-level sync zamiast osobnych kernel launchów.
3. ~~**Fused TopK+Router**: jeden kernel zamiast dwóch, oszczędność 48 dispatches.~~ → **DONE** (Lekcja 20)
4. **Dalsze eksperymenty z flash decode**: bloc

---

## Lekcja 20: Transpozycja skal, flash decode block_s, cooperative kernel (2026-03-17)

### Eksperymenty

**1. Transpozycja skal [K/64, N, 2] vs row-major [N, K/64, 2]:**
- Hipoteza: transposed coalesces scale reads across N → lepszy BW
- Wynik: **brak mierzalnej różnicy** (166.6 vs 166.6 t/s)
- Analiza: scale data to ~6-12% total danych. Nawet 16x poprawa coalescing daje <2% total BW
- Wniosek: scale access pattern nie jest bottleneckiem. Zmieniono na row-major bo logicznie czystsze

**2. Flash decode block_s tuning:**
- block_s=32: 16 blocks × 6 warps = 96 wavefronts → 1.5/SIMD (baseline)
- block_s=16: 32 blocks × 6 warps = 192 wavefronts → 3/SIMD → **+0.8% (168 t/s)**
- block_s=8: 64 blocks → crash + regresja (za dużo splits, overhead reduce)
- Wniosek: block_s=16 to optimum dla ctx=128

**3. Cooperative mega-kernel:**
- hipLaunchCooperativeKernel z grid.sync() zamiast kernel dispatches
- Wynik: **100.9 t/s** (ctx=128) — ~40% WOLNIEJ od graph!
- Analiza: grid.sync() bariery na RDNA4 są bardzo kosztowne. Atomic barriers ~200-500ns × 480/token = ~100µs+
- Wniosek: HIP Graph >>> cooperative kernel na RDNA4

**4. Split-K GEMV:**
- QKV (N=5120, K=2048): split-K=2 → 18.83µs vs original 10.91µs — **72% WOLNIEJ**
- O (N=2048, K=3072): split-K=2 → 16.14µs vs original 13.16µs — **23% WOLNIEJ**
- Analiza: hipMemsetAsync + atomicAdd + float→half konwersja dodaje więcej overhead niż parallelism zyskuje
- Wniosek: Split-K nie opłaca się na RDNA4 z INT4 GEMV (already good occupancy)

### Bandwidth analysis

| Buffer | Size (MB) | Peak BW test |
|--------|-----------|-------------|
| 64 MB (L2 hit) | 64 | 1236 GB/s |
| 256 MB | 256 | 537 GB/s |
| 1024 MB | 1024 | 598 GB/s |
| 1600 MB (model) | 1600 | 599 GB/s |

Achievable peak DRAM read BW: **~599 GB/s**. Theoretical: 576 GB/s.
Per-token data: ~1600 MB. Minimum time: 1600/599 = 2.67 ms.
Our 5.73 ms = **46.6% BW efficiency** (rest: dispatch, flash decode, small kernels).
GGUF 5.88 ms = **45.4% BW efficiency** at ~1700 MB data.

### Status końcowy

| Engine | ctx=128 t/s | Rozmiar |
|--------|------------|---------|
| **Custom INT4 (HIP Graph)** | **174.6 t/s** | **16.8 GB VRAM** |
| GGUF Q4_K_M Vulkan | 167-174 t/s | 17.3 GB |
| GGUF Q4_K_M ROCm | 96.3 t/s | 17.3 GB |

**Cel osiągnięty: Custom engine = GGUF Vulkan (±3%), przy mniejszym VRAM.**

## Lekcja 21: C++ prefill z INT4 attention (2026-03-17)

### Problem
Python prefill (fast_prefill) osiągał 100-370 t/s — 8-13x wolniej od GGUF Vulkan (1444-2971 t/s).
Bottleneck: per-expert Python loop z dequant+GEMM (960 małych GEMMs × Python overhead).

### Rozwiązanie
Zmodyfikowano C++ `prefill_moe_logits` aby akceptował INT4 attention weights:
- Dodano parametry `int4_qkv_w/s/N`, `int4_o_w/s/N/K`
- Fallback: gdy `fp16_qkv_w` puste → Hadamard rotate + `dequant_gemm_tiled_g64`
- Expert dispatch: W4A4 `gemv_w4a4_udot8_v3` z GPU-sorted batching (1 kernel per GU/DN)

### Wyniki

| ctx | C++ Engine | Python old | GGUF Vulkan | vs GGUF |
|-----|-----------|-----------|-------------|---------|
| pp128 | **1719 t/s** | 99 t/s | 1444 t/s | **+19%** |
| pp256 | **2184 t/s** | 201 t/s | 2176 t/s | **+0.4%** |
| pp512 | crash | 343 t/s | 2971 t/s | needs fix |

### Znany bug
pp512+ crash (HIP launch failure): W4A4 kernel przy M*k=4096 tokens.
Prawdopodobnie overflow w grid/buffer calculation.

---

## Qwen3.5-27B Optimization Session (2026-03-17)

### Context
Hybrid model: 48 DeltaNet + 16 FullAttention layers, D=5120, 24 Q heads, 4 KV heads (GQA=6).
INT4 g32 z FP16 scales, ~17.9 GB model weights.

### Prefill optimizations

**1. Tiled dequant+GEMM (Infinity Cache friendly):** `dequant_gemm_tiled_g32` z tile=4096
- Dequantyzuje N w pasmach po 4096 wierszy → 40MB FP16 mieści się w 64MB Infinity Cache
- pp128: +14% (366→419 t/s), pp512: +2%
- Najlepsze dla dużych N>8192 (GU 34816, QKV+Z 16384, FA_QKV 14336)
- Małe macierze (O 5120, Down 5120): wolniejsze z tiled → threshold N>8192

**2. Fused INT4 WMMA GEMM (pre-tiled weights):** `fused_int4_wmma_gemm`
- **42% szybszy** niż dequant+rocBLAS przy M≤256 (1.35ms vs 1.92ms GU)
- Zero BW overhead: czyta INT4 bezpośrednio, dequant w rejestrach, WMMA 16×16×16
- **Zero VRAM overhead**: tiled format ma identyczny rozmiar jak original packed
- **PROBLEM**: wymaga pre-tiled weights (`w4a8_prepare_weights`) — na 32GB GPU z 27B modelem (~17.9GB) nie ma miejsca na drugą kopię wag (~17.9GB) + KV cache + state
- **PROBLEM**: on-the-fly tiling = 2ms overhead per projection → netto wolniej
- **Bug found**: `w4a8_prepare_weights_into` transponuje scales z stride N zamiast N_pad → crash dla N≥14336

**3. F.linear vs @ dq.T — MISLEADING benchmark**
- Izolowany bench: `F.linear(x, dq)` = 0.94ms vs `x @ dq.T` = 7.71ms (8×!)
- W pipeline z dequant: `F.linear` = 8.48ms vs `x @ dq.T` = 1.91ms (odwrotnie!)
- Powód: PyTorch overlaps dequant with GEMM w pipeline; F.linear blokuje

**4. Bandwidth analysis (pp128):**
- Dequant pipeline: read 17.9GB INT4 + write 71.6GB FP16 + read 71.6GB FP16 = **161 GB** per pass
- At 644 GB/s → minimum 250ms. Actual: 298ms = **86% BW efficiency**
- Fused WMMA: 20 GB total → 31ms → ~4000 t/s (theoretical, blocked by VRAM)

### Decode optimizations

**5. C++ fused decode step:** `hybrid_decode_step_logits`
- Zero Python overhead (0.02ms vs 12ms Python loop)
- 34.14ms per step = **29.3 t/s** (81% BW utilization, theoretical max 36 t/s)
- Kernel launch overhead: ~320 launches × 5μs = 1.6ms (~5% of total)

**6. Flash decode LDS + V_DOT2_F32_F16:** `flash_decode_lds_dot2_d256`
- Cooperative K/V load: all 6 GQA warps load K/V together → LDS → private compute
- Eliminates 5× redundant K/V reads (GQA=6, 1 load instead of 6)
- `__builtin_amdgcn_fdot2` for 2× FMA throughput in dot product
- **+18% on flash decode**, +0.3% on total decode (attention is ~2% of total)

**7. HIP Graph capture:** Tested, **marginal gain** (+1.4%). Graph replay overhead on RDNA4 ≈ dispatch overhead. Not worth complexity.

### Final results vs GGUF Q4_K_M Vulkan

| Metryka | INT4 Engine | GGUF Vulkan | vs GGUF |
|---------|-------------|-------------|---------|
| Decode (tg128) | **29.0 t/s** | 26.13 t/s | **+11%** |
| pp128 | 341 t/s | 651 t/s | -48% |
| pp256 | 645 t/s | — | — |
| pp512 | **844 t/s** | 739 t/s | **+14%** |
| pp1024 | 980 t/s | — | — |
| VRAM | 18.7 GB | ~18 GB | = |

### Key insight: dequant→FP16→GEMM is 8× more BW than fused INT4→GEMM
Jedyne rozwiązanie dla pp128: fused INT4 WMMA kernel z in-place weight retiling (replace packed weights, modify C++ decode GEMV to read tiled format). Potrzebuje ~2 dni pracy na C++ kernel redesign.

### WMMA throughput benchmarks (gfx1201, 64 CU, 1 wave/CU)

| Instrukcja | Format | K per op | Throughput | vs FP16 |
|------------|--------|----------|-----------|---------|
| `v_wmma_f32_16x16x16_f16` | FP16→FP32 | 16 | 40.3 TFLOPS | 1.00× |
| `v_wmma_i32_16x16x32_iu4` | **INT4→INT32** | **32** | **81.1 TOPS** | **2.01×** |
| `v_wmma_f32_16x16x16_fp8_fp8` | FP8→FP32 | 16 | 40.5 TFLOPS | 1.00× |
| `v_wmma_i32_16x16x16_iu8` | INT8→INT32 | 16 | — | — |

**Kluczowe: INT4 WMMA = 2× throughput vs FP16 WMMA!**
- K=32 per instruction (vs K=16 for FP16/FP8)
- Wymaga aktywacje w INT4 → potrzeba on-the-fly quantization (W4A4 path)
- INT32 accumulator → post-multiply by scale after K-loop
- **Potencjalny gain: 4-8× vs obecny FP16 WMMA** (eliminuje scalar dequant + 2× K depth)
- Builtin: `__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(neg_a, A_int2, neg_b, B_int2, C_int8, clamp)`

### Obecny WMMA bottleneck: scalar dequant (26% peak compute)
Nasz `fused_int4_wmma_f16_direct/tiled` kernel:
1. Ładuje INT4 packed z VRAM/LDS
2. **Dequantyzuje do FP16 per-lane** (8 scalar float→half konwersji per K-step) ← BOTTLENECK
3. WMMA f32_16x16x16_f16

Scalar dequant = ~40 VALU cykli vs WMMA ~4 cykli → 10:1 overhead ratio.
INT4 WMMA eliminuje krok 2 całkowicie.
---

## Lekcja 22: Occupancy Limiter — przełom w attention scaling (2026-03-18)

### Odkrycie

Jedna linijka `__launch_bounds__(160, 2)` daje **+40% na decode ctx=4096**.

### Problem

Flash decode kernel (split-K + LDS) na RDNA4 przy dużym ctx:
- ctx=4096, block_s=64: 72 splits × 8 KV heads = 576 bloków
- Bez limitu: scheduler ładuje 6-8 bloków/CU = 30-40 wavefronts
- L0 cache: ~16 KB per WGP — za mało na 40 wavefrontów czytających różne K/V tiles
- **96% L0 cache miss** → kernel czeka na VRAM → attention staje się BW-bound zamiast compute-bound

### Rozwiązanie

```cpp
// Low-occupancy variant: max 2 bloki/CU
__global__ __launch_bounds__(160, 2)
void flash_decode_partial_fp16_g_lo(...)
```

Efekt:
- 2 bloki/CU × 5 warpów = 10 wavefrontów (zamiast 40)
- LDS working set per CU: 2 × 32KB = 64KB (mieści się w LDS!)
- L0 cachuje LDS dostępy → Q·K dot i V accumulate czytają z cache
- **Zero L0 thrashing**

### Kluczowa reguła

**Occupancy limiter pomaga TYLKO kernelom z data reuse (attention, GEMM z LDS tiling).**
**GEMV jest streaming (BW-bound) → więcej wavefrontów = lepiej → NIE limituj.**

| Kernel | Typ | launch_bounds | Uzasadnienie |
|--------|-----|---------------|--------------|
| `gemv_multiwave_rm_g64` | Streaming | (128, 4) — dużo waves | BW-bound, brak reuse |
| `flash_decode_partial_fp16_g_lo` | Data reuse | (160, 2) — mało waves | LDS reuse, L0 hot |
| `gemv_mw_batch_g64` | Streaming | brak limitu | Expert GEMV, BW-bound |

### Adaptive dispatch

```cpp
bool use_lo = (max_n_splits >= 16);  // ~1024+ ctx
if (use_lo) LAUNCH(flash_decode_partial_fp16_g_lo);
else        LAUNCH(flash_decode_partial_fp16_g);
```

Krótki ctx (mało splits) → standard kernel (potrzebuje occupancy).
Długi ctx (dużo splits) → low-occupancy kernel (potrzebuje cache reuse).

### Wyniki

| ctx | Przed | Po | Zmiana | vs GGUF |
|-----|-------|----|--------|---------|
| tg128 | 170 | 170 | 0% | -3% |
| tg512 | 185 | 187 | +1% | **+9%** |
| tg1024 | 179 | **217** | **+21%** | **+30%** |
| tg2048 | 154 | **191** | **+24%** | **+18%** |
| tg4096 | 134 | **187** | **+40%** | **+21%** |
| tg8192 | 129 | **187** | **+45%** | **+34%** |

### Analogia z GGUF Vulkan

GGUF robi to samo w shaderze Vulkan:
```glsl
limit_occupancy_shmem = 26KB;  // sztuczna alokacja LDS → mniej workgroups/CU
```
Nasz `__launch_bounds__` jest czystszy — kompilator optymalizuje VGPR allocation.

---

## Lekcja 23: Parallel Reduce — 4-warp flash decode reduce (2026-03-18)

### Problem

Reduce kernel iterował splits **serialnie** w jednym warpie.
ctx=4096: 72 splits × (load + expf + FMA) = 72 sequential iterations per Q head.

### Rozwiązanie

4 warpy per block, każdy procesuje chunk splits równolegle:
```cpp
__global__ __launch_bounds__(128)
void flash_decode_reduce(...) {
    int sp_start = (num_splits * warp_id) / 4;
    int sp_end = (num_splits * (warp_id + 1)) / 4;
    // ... each warp handles 1/4 of splits
    // Inter-warp reduce via shared memory
}
```

### Wynik

+7-13% na ctx=2048-8192 (reduce overhead cut 4×).

---

## Lekcja 24: FP8 KV Cache — nie opłaca się na RDNA4 (2026-03-18)

### Eksperyment

Zamiana KV cache FP16 → FP8 E4M3 (uint8):
- 50% mniej danych do przeczytania
- `v_cvt_f32_fp8` hardware conversion (1 cycle)

### Wynik

**WOLNIEJSZE** — 165 t/s vs 187 t/s z FP16 (tg512)

### Przyczyna

FP8 ładuje po 1 bajcie (byte-level load) vs FP16 ładuje po 4 bajty (__half2).
Konwersja `v_cvt_f32_fp8` per-element dodaje latency.
Żeby FP8 było szybsze, potrzebne `V_DOT4_F32_FP8_FP8` (4 elementy naraz) z przepakowanym Q.

### Wniosek

FP8 KV daje oszczędność **VRAM** (50%) ale nie **prędkości** na RDNA4.
Przydatne tylko dla 32k+ kontekstu gdzie VRAM jest limitem.

---

## Lekcja 25: Multi-tile (S_BARRIER) nie działa na RDNA4 (2026-03-18)

### Eksperyment

Jeden block procesuje 8-16 LDS tiles sekwencyjnie (online softmax cross-tile):
- TILES_PER_SPLIT=8, block_s=64 → effective 512 per split
- Mniej splits → mniej reduce overhead

### Wynik

**Katastrofa** — 97 t/s vs 185 t/s (tg512)

### Przyczyna

`__syncthreads()` (S_BARRIER) z 5 warpami (160 threads) kosztuje ~500ns na RDNA4.
8 barier × 500ns = 4µs per split. × 48 layers × 10 splits = 1.9ms overhead.
Plus: mniej bloków (64 vs 576) → gorsza GPU utilization.

### Wniosek

Na RDNA4 S_BARRIER jest DROGI. Każda bariera to ~500ns z 5+ warpami.
Preferuj dużo małych bloków z jedną barierą niż mało dużych z wieloma barierami.

---

## Lekcja 26: GPU Runtime PM — silent performance killer (2026-03-18)

### Problem

R9700 AI PRO spada z 175 → 141 t/s bez widocznej przyczyny.
Temperatura OK (36°C), power cap OK (300W), rocm-smi pokazuje idle clocks.

### Przyczyna

Linux runtime power management (runtime PM) automatycznie usypia GPU do stanu D3hot:
```
power_state: D3hot
runtime_status: suspended
power/control: auto
```

Po wybudzeniu z D3hot, GPU nie wraca do pełnych zegarów pamięci:
- mclk: 1124 MHz (level 4) zamiast 1258 MHz (level 5)
- sclk: 14 MHz pod obciążeniem (!)
- VDDGFX: 90 mV — minimalne napięcie

### Fix (permanentny)

```bash
# Immediate
echo "on" > /sys/class/drm/card1/device/power/control

# Permanent udev rule
echo 'SUBSYSTEM=="pci", ATTR{vendor}=="0x1002", ATTR{device}=="0x7551", ATTR{power/control}="on"' \
  > /etc/udev/rules.d/99-amdgpu-no-runtime-pm.rules

# Backup: systemd service
systemctl enable amdgpu-no-suspend.service
```

### KRYTYCZNE

Sprawdzaj `cat /sys/class/drm/card*/device/power_state` PRZED każdym benchmarkiem.
Jeśli D3hot → wyniki są bezwartościowe.

---

## Lekcja 27: Finalne wyniki decode po optymalizacjach (2026-03-18)

### Zastosowane optymalizacje (kumulatywne)

1. **LDS-tiled flash decode** — cooperative K/V load, horizontal sharing (Lekcja 18-19)
2. **Parallel reduce** — 4-warp reduce zamiast serial (Lekcja 23)
3. **Adaptive block_s** — bs=64 short ctx, bs=128 long ctx, threshold=4608 (Lekcja 22)
4. **Occupancy limiter** — `__launch_bounds__(160, 2)` dla attention przy ≥16 splits (Lekcja 22)

### Finalne wyniki decode — Qwen3-30B-A3B, R9700 #1

| ctx | Nasz engine | GGUF Q4_K_M Vulkan1 | vs GGUF |
|-----|------------|---------------------|---------|
| tg128 | 170 t/s | 175 t/s | -3% |
| tg512 | **187 t/s** | 172 t/s | **+9%** |
| tg1024 | **217 t/s** | 167 t/s | **+30%** |
| tg2048 | **191 t/s** | 162 t/s | **+18%** |
| tg4096 | **187 t/s** | 154 t/s | **+21%** |
| tg8192 | **187 t/s** | ~140 t/s | **+34%** |

**GGUF przebity na każdym kontekście ≥512.** Jedyne -3% na tg128 (GPU clock warmup).

### Prefill — wymaga refaktoru

| ctx | Nasz | GGUF | vs GGUF | Problem |
|-----|------|------|---------|---------|
| pp128 | **1546** | 1413 | **+9%** | OK |
| pp512 | 1885 | **2983** | -37% | 91% overhead |
| pp1024 | 2031 | **2918** | -30% | 91% overhead |
| pp2048 | 2593 | **2668** | -3% | Catching up |
| pp4096 | **3086** | 2306 | **+34%** | Wygrywamy |

Profiling pp1024: kernels=44ms, total=504ms. **91% to overhead** (MoE dispatch, hipStreamSynchronize per layer, PyTorch tensor ops w hot loop). Potrzebny czysty C++ prefill.

### VRAM

| Komponent | Rozmiar |
|-----------|---------|
| Model (INT4 g64) | 16.4 GB |
| KV cache (FP16, ctx=32k) | 3.2 GB |
| Bufory robocze | ~2 GB |
| **Total max** | **~22 GB / 32 GB** |

### Kluczowa lekcja architekturalna

**RDNA4 GEMV vs Attention — diametralnie różne strategie occupancy:**

```
GEMV (streaming):     więcej wavefrontów = lepiej (BW-bound)
Attention (LDS reuse): mniej wavefrontów = lepiej (cache-bound)
```

Jedna linijka `__launch_bounds__` dała +40% na attention. To samo co GGUF robi z `limit_occupancy_shmem`.

---

## Lekcja 28: SWMMAC A fragment = ROWS nie columns (2026-03-18)

### Odkrycie empiryczne

SWMMAC `V_SWMMAC_I32_16X16X64_IU4`:
- **A fragment (sparse): lane nl = ROW nl** (nie kolumna!)
- **B fragment (dense): lane nl = COLUMN nl**  
- **Output C: lane nl = COLUMN nl** (jak dense WMMA, DISCOVERY confirmed)

To jest ODWROTNE od output layout! A = wiersze, C = kolumny.

### Implikacja dla tiling

Tiled sparse A: `[K_group][N_tile][16_rows × 8_bytes]` = `[K/64][N/16][128]`
Lane nl reads bytes `[nl*8 .. nl*8+7]` → row nl of N-tile → COALESCED (16 × 8 = 128 consecutive)

### Raw SWMMAC throughput progression

| Approach | TOPS | Bottleneck |
|----------|------|------------|
| Row-major scattered | 31 | Memory latency (scattered reads) |
| Python tiled | 67 | Partially coalesced |
| **Correct row-based tiled** | **TBD** | Should approach 477 TOPS |
| Raw register-only | 477 | Hardware peak |
