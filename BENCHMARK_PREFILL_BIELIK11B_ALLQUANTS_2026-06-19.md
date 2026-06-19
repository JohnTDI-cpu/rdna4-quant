# Prefill Benchmark — Bielik-11B, all quants, ROCm vs RADV (2026-06-19)

Definitive baseline of **what ggml-JohnV8 must beat** on prefill. Measured fresh, best settings.

## Environment

| | |
|---|---|
| **Model** | Bielik-11B-v3.0-Instruct (11.17 B params), `/home/janusz/teacher_q8/Bielik-11B-*.gguf` |
| **GPU** | 2× AMD Radeon AI PRO R9700 (RDNA4, gfx1201, 32 GB), Wave32 |
| **ROCm** | 7.2.3 (`/opt/rocm-7.2.3`) |
| **llama.cpp ROCm** | `/home/janusz/llama_new/build_hip/bin/llama-bench` (build cze 2026, newest local) |
| **llama.cpp Vulkan** | `/home/janusz/llama_new/build_vulkan/bin/llama-bench` (RADV) |
| **Vulkan ICD** | RADV (`/usr/share/vulkan/icd.d/radeon_icd.json`) |

## Settings (best, validated)

- **Power: `auto`** — NOT `high`/`profile_peak`. Confirmed: `high` THROTTLES sclk on this workload (Q6_K pp512 high=1323 vs auto=1638, **−24%**). `auto` boosts sclk ~3359 under sustained load vs high pins ~2330. `rocm-smi --setperflevel auto`.
- **Sequential, NOT parallel** — ROCm on GPU1, RADV on GPU0, run one at a time (parallel was tested: no contention here, but sequential is the clean reference).
- **Commands:**
  ```bash
  # ROCm (HIP), GPU1
  HIP_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/opt/rocm-7.2.3/lib:/opt/rocm-7.2.3/lib/llvm/lib \
    llama_new/build_hip/bin/llama-bench -m <gguf> -p 256,512,1024,2048,4096,8192 -n 0 -r 3
  # RADV (Vulkan), GPU0
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json GGML_VK_VISIBLE_DEVICES=1 \
    llama_new/build_vulkan/bin/llama-bench -m <gguf> -p 256,512,1024,2048,4096,8192 -n 0 -r 3
  ```
- Prefill only (`-n 0`), `-r 3` reps (`-r 2` for 16k/32k). t/s = prompt tokens/sec.

## Results — prefill t/s (higher = better)

| Quant | Backend | pp256 | pp512 | pp1024 | pp2048 | pp4096 | pp8192 |
|---|---|---|---|---|---|---|---|
| **Q4_0** | ROCm | 3063 | 3156 | **3223** | 3121 | 2936 | 2616 |
| | RADV | 2751 | **3224** | 3168 | 3064 | 2875 | 2547 |
| **Q4_K_M** | ROCm | 2195 | 2504 | 2545 | 2479 | 2355 | 2138 |
| | RADV | **2443** | **2696** | **2655** | **2584** | **2445** | **2203** |
| **Q5_K_M** | ROCm | 2281 | 2442 | 2417 | 2356 | 2246 | 2050 |
| | RADV | **2384** | **2608** | **2569** | **2502** | **2375** | **2140** |
| **Q5_K** | ROCm | **2455** | **2661** | **2629** | **2557** | **2428** | **2203** |
| | RADV | 2430 | 2657 | 2621 | 2551 | 2417 | 2172 |
| **Q6_K** | ROCm | 1551 | 1644 | 1633 | 1604 | 1546 | 1446 |
| | RADV | **2183** | **2341** | **2319** | **2265** | **2158** | **1962** |
| **Q8_0** | ROCm | **2951** | **3214** | **3166** | **3067** | **2887** | **2581** |
| | RADV | 2557 | 2878 | 2838 | 2757 | 2597 | 2322 |

*(Bold = faster backend.)*

### Long context (pp16384, pp32768; -r 2)

| Quant | ROCm 16k | RADV 16k | ROCm 32k | RADV 32k |
|---|---|---|---|---|
| Q4_0 | **2155** | 2097 | 1532 | 1532 |
| Q4_K_M | 1829 | **1867** | 1360 | **1400** |
| Q5_K_M | 1766 | **1827** | 1323 | **1376** |
| Q5_K | **1876** | 1849 | 792¹ | **1389** |
| Q6_K | 1301 | **1694** | 1042 | **1298** |
| Q8_0 | **2138** | 1958 | **1523** | 1449 |

¹ Q5_K ROCm 32k = glitch (-r 2 outlier).

At long context the attention O(N²) dominates → all drop. Pattern holds: ROCm wins simple (Q4_0/Q8), RADV wins K-quants. Q6_K RADV still +30-25% over ROCm at 16k/32k.

## Best-of-both = the bar to beat (@ pp512)

| Quant | best-of-both | winner | margin |
|---|---|---|---|
| Q4_0 | **3224** | ~tie (RADV) | +2% |
| Q8_0 | **3214** | ROCm | +12% |
| Q5_K | **2661** | ~tie (ROCm) | +0.2% |
| Q4_K_M | **2696** | RADV | +8% |
| Q5_K_M | **2608** | RADV | +7% |
| Q6_K | **2341** | RADV | +42% |

## Pattern

- **Simple / dense-int quants (Q4_0, Q8_0): ~3200 t/s, both backends strong** — they have efficient int8/int4 paths. Hard to beat.
- **K-quants (Q4_K_M, Q5_K_M, Q5_K, Q6_K): 2341–2696** — backends pay the 4/5/6-bit dequant cost in the matmul loop. **Q6_K worst for ROCm** (1644, the mmq 6-bit dequant). This is the opening.
- **ggml-JohnV8 angle:** dequant K-quant→int8 ONCE at load (re-quant per-32, lossless-grade) → pure int8 WMMA GEMM (no per-forward dequant). Measured GEMM-only: Q8=2928, Q6_K=2746. Above the K-quant bars, below the simple-quant bars.
- **Caveat:** engine numbers are GEMM-only (no attention/norm). Full forward needed for a valid claim; K-quant margins are tight (~0–5% after attention overhead), Q6_K the most solid.

## Notes / lessons

- Old memory numbers (ROCm 1612 / RADV 2255 for Q6_K) were ~correct (auto-era). The `505`/`1323` readings were `high`-throttled artifacts — do not use `high` for max-t/s benchmarks.
- Always measure fresh, `auto` power, newest ROCm + llama.cpp.
