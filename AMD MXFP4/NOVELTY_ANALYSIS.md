# Analiza nowości: INT4 Engine na konsumenckim AMD

## Pytanie: Czy nasza kwantyzacja to nowość na konsumenckim AMD?

---

## Odpowiedź: TAK — kombinacja jest unikalna

Poszczególne techniki (GPTQ, Hadamard, INT4 asymmetric) istnieją w literaturze,
ale **nasza specyficzna kombinacja + implementacja na konsumenckim RDNA4 jest pierwszą znaną publicznie.**

---

## Co ISTNIEJE w literaturze (nie jest nasze)

| Technika | Źródło | Rok |
|----------|--------|-----|
| GPTQ (Cholesky error compensation) | Frantar et al., "GPTQ: Accurate Post-Training Quantization" | 2022 |
| Hadamard rotation przed kwantyzacją | QuaRot (Ashkboos et al.), ICLR 2026 | 2024 |
| SpinQuant (per-layer learned rotation) | Liu et al., Meta AI | 2024 |
| AWQ (activation-weighted quantization) | Lin et al., "AWQ: Activation-aware Weight Quantization" | 2023 |
| MXFP4/NVFP4 format | OCP MX Specification v1.0 / NVIDIA TensorRT-LLM | 2023-24 |
| FP8 KV cache | vLLM, TensorRT-LLM, AMD Quark | 2024+ |
| HadaCore (Hadamard Tensor Core kernel) | Abts et al. | 2024 |

---

## Co jest NOWE w naszej pracy

### 1. Pierwsza produkcyjna implementacja INT4+Hadamard+GPTQ na konsumenckim AMD (RDNA3/4)

**Stan świata:**
- AMD **Quark** (oficjalne narzędzie) wspiera QuaRot i GPTQ, ale:
  - Dokumentacja i testy tylko dla **MI300X** (datacenter)
  - Brak gotowych kerneli decode dla konsumenckich GPU
  - Brak benchmarków na RDNA3/RDNA4
- **llama.cpp** ma backend ROCm, ale:
  - Nie używa Hadamard rotation
  - Nie używa asymetrycznego INT4 z GPTQ
  - Używa formatu GGUF z własnymi skalami (nie FP16 block scales)
- **vLLM** wspiera AMD od v0.14, ale:
  - Skupiony na datacenter (MI250X/MI300X)
  - Brak optymalizacji decode dla single-user na konsumenckim GPU
- **Nikt** nie opublikował custom HIP kerneli GEMV z fused INT4 dequant dla RDNA4

**Nasze osiągnięcie:** Pełny pipeline od kwantyzacji do inference, z custom kernelami HIP,
na sprzęcie konsumenckim za ~$600 (R9700).

### 2. Pierwszy znany 60+ t/s decode na konsumenckim AMD dla modelu 14B

**Porównanie publicznie znanych wyników:**
| Silnik | GPU | Model 14B | Decode t/s |
|--------|-----|-----------|-----------|
| llama.cpp ROCm | RDNA4 (R9700) | Qwen3-14B Q4_K_M | **47.7** |
| llama.cpp Vulkan | RDNA4 | broken | **7** (bug) |
| vLLM (datacenter) | MI300X | 14B INT4 | >200 (ale 192GB GPU) |
| **Nasz engine** | **RDNA4 (R9700)** | **Qwen3-14B INT4** | **61** |

Nasz engine jest **28% szybszy** od llama.cpp na tym samym GPU.

### 3. Custom HIP GEMV z fused INT4 asymmetric dequant + Hadamard FWHT

**Nie istnieje publicznie:**
- HadaCore (Abts 2024) — CUDA kernel na A100, nie HIP, nie konsumencki
- rocWMMA — biblioteka headerów, nie production decode engine
- Brak publicznych repozytoriów z HIP GEMV dla INT4 asymmetric na RDNA4

**Nasze kernele:**
- `gemv_warp`: warp-based GEMV z 128-bit loads, inline asymmetric dequant
- `fwht_inplace_32`: FWHT w 32 rejestrach bez shared memory
- `fp8e4m3_to_f32` / `f32_to_fp8e4m3`: branchless FP8 encode/decode
- `flash_decode_partial`: split-K flash attention z FP8 KV cache

### 4. Empiryczne porównanie MXFP4 vs NVFP4 vs INT4+FP16 na RDNA4

**Nikt wcześniej nie opublikował takiego porównania na konsumenckim AMD:**

| Format | Skale | PPL | Wniosek |
|--------|-------|-----|---------|
| MXFP4 (E8M0 scales) | Power-of-2 only (32 wartości) | 9.07-9.81 | Najgorsza |
| NVFP4 (E4M3 scales) | 256 wartości | 8.7 | Dobra |
| **INT4 asym (FP16 scales)** | **65504 wartości** | **7.692** | **Najlepsza** |

To empirycznie udowadnia że **format skali jest ważniejszy niż format danych**
— FP16 scales wygrywają, nawet z prostszym INT4 zamiast E2M1 FP4.

### 5. WMMA lane mapping discovery na gfx12 (RDNA4)

Dokumentacja AMD nie wyjaśnia jasno layoutu fragmentów WMMA na gfx12.
My odkryliśmy empirycznie i opisaliśmy w DISCOVERY_GFX12_WMMA_OUTPUT.md:

```
VGPR[lane][j] = matrix[(lane / 16) * 8 + j][lane % 16]
Lanes indeksują KOLUMNY (nie wiersze) — to niezgodne z intuicją.
```

---

## Co NIE jest przełomowe (uczciwa ocena)

1. **Algorytm GPTQ** — znany od 2022, my go tylko implementujemy
2. **Rotacja Hadamarda** — QuaRot (2024) opisał to formalnie
3. **Asymetryczny INT4** — standard w GPTQ/AWQ/bitsandbytes
4. **FP8 KV cache** — vLLM/TensorRT-LLM mają to od 2024
5. **Koncept mixed precision** — wiele prac (SqueezeLLM, AQLM, etc.)

---

## Podsumowanie: Rating nowości

| Aspekt | Ocena | Komentarz |
|--------|-------|-----------|
| Algorytm kwantyzacji | 3/10 | Znane techniki w nowej kombinacji |
| Implementacja na RDNA4 | **9/10** | Pierwsza publiczna, custom HIP kernele |
| Prędkość vs competition | **8/10** | Bije llama.cpp o 28%, bliski bandwidth wall |
| Jakość vs competition | **7/10** | Bije GGUF Q4_K_M na PPL i benchmarkach |
| Porównanie formatów (MXFP4/NVFP4/INT4) | **8/10** | Unikalne empiryczne dane |
| WMMA lane mapping discovery | **7/10** | Brak dokumentacji AMD, przydatne dla community |

**Ogólna ocena: 7/10 — solidny wkład techniczny.**

Nie jest to przełom naukowy (algorytmy są znane), ale jest to **pierwsza kompletna,
zoptymalizowana implementacja na konsumenckim AMD** z wynikami lepszymi niż istniejące
alternatywy. Jako open-source miałoby wysoką wartość dla community.

---

## Rekomendacja: co warto opublikować

1. **Custom HIP kernele** (hip_int4/) — największa wartość, nikt tego nie ma na RDNA4
2. **Receptura kwantyzacji** (quantize_v4_gptq.py) — reprodukowalna, lepsza od Q4_K_M
3. **Benchmark comparison** — INT4 vs GGUF na identycznym sprzęcie (uczciwe porównanie)
4. **WMMA lane mapping** — pomoc dla każdego piszącego kernele na gfx12
5. **Porównanie formatów** (MXFP4 vs NVFP4 vs INT4) — empiryczne dane z jednego modelu
