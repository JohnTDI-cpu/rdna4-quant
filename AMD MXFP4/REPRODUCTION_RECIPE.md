# Kompletna Receptura Kwantyzacji INT4 + Hadamard + GPTQ

## Cel dokumentu
Pełna reprodukcja najlepszej kwantyzacji INT4 (PPL 7.692, bije GGUF Q4_K_M 7.715)
na dowolnym PC z AMD GPU (RDNA3/4) lub NVIDIA GPU z ROCm/CUDA.

Model: **Qwen3-14B** (14.77B parametrów, 40 warstw, hidden=5120)

---

## 1. Wymagania sprzętowe i programowe

### Hardware (minimum)
- GPU z ≥24 GB VRAM (kwantyzacja potrzebuje FP16 model + Hessian w pamięci)
- RAM ≥32 GB (ładowanie safetensors)
- ~30 GB dysku na model FP16 + ~8 GB na wynik kwantyzacji

### Testowano na
- 2× AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32GB VRAM)
- ROCm 7.1.0, PyTorch 2.10.0+rocm7.1, hipcc clang-19
- CPU: AMD Ryzen 9 9900X, 64 GB DDR5

### Software
```bash
pip install torch transformers safetensors datasets
# Opcjonalnie (do inference):
pip install triton   # Triton 3.x z obsługą ROCm
```

---

## 2. Formalna definicja formatu kwantyzacji

### 2.1. Format wag: INT4 Asymmetric z zero-pointem

Każda macierz wag W[N, K] jest dzielona na bloki po 32 elementów wzdłuż wymiaru K.

**Dla każdego bloku [N, 32]:**

```
vmin = min(block)                    # minimum bloku
vmax = max(block)                    # maximum bloku
scale = (vmax - vmin) / 15.0         # FP16 skala
zero_point = round(-vmin / scale)    # uint8 ∈ [0..15]

quantized[i] = round(W[i] / scale + zero_point)   # uint8 ∈ [0..15]
dequantized[i] = (quantized[i] - zero_point) * scale  # ≈ W[i]
```

**Przechowywanie:**
- Dane: 2 nibbly w jednym bajcie uint8: `byte = (lo & 0xF) | ((hi & 0xF) << 4)`
- Skale: FP16 per blok 32 elementów → `[N, K/32]`
- Zero-pointy: uint8 per blok → `[N, K/32]`

**Efektywne BPW (bits-per-weight):**
- Dane: 4 bity/wagę
- Metadane: (16 bit skala + 8 bit zero) / 32 = 0.75 bit/wagę
- Razem: ~4.75 BPW

### 2.2. Preprocessing: Rotacja Hadamarda (blokowa)

Przed kwantyzacją, wagi są obracane macierzą Hadamarda:

```
H = sylvester_hadamard(32) / sqrt(32)   # [32, 32] znormalizowana macierz ortogonalna
W_rot[:, i*32:(i+1)*32] = W[:, i*32:(i+1)*32] @ H    # per-blok rotacja kolumn
```

**Cel:** Rozkłada outlier-y (ciężkie ogony dystrybucji wag) równomiernie po bloku.
Kluczowa właściwość: H @ Hᵀ = I (ortogonalna), więc rotacja jest bezstratna w FP16.

**Przy inferencji:** aktywacje też przechodzą tę samą rotację:
```
x_rot = x @ block_diag(H, H, ..., H)
y = W_rot @ x_rot^T = W @ x^T   ✓  (bo H @ H = I)
```

W kernelach HIP realizowane jako **FWHT** (Fast Walsh-Hadamard Transform) — 5 etapów butterfly na 32 rejestrach float, bez shared memory.

### 2.3. Mixed precision: INT4 vs INT8 (na podstawie wrażliwości)

**20% najbardziej wrażliwych grup wagowych** (warstwy × projekcje) jest kwantyzowane do INT8 zamiast INT4.

Wrażliwość mierzona jako:
```
sensitivity_score = Σ_k (W_error[k])² × H_diag[k]
```
gdzie H_diag to diagonala macierzy Hessiana (= kowariancja aktywacji).

Głębsze warstwy (31-39) są zwykle bardziej wrażliwe → częściej INT8.

### 2.4. KV cache: FP8 E4M3

KV cache przy inferencji przechowywany w FP8 E4M3 (1 bajt/element zamiast 2 bajtów FP16):
```
FP8 E4M3: 1 bit znaku + 4 bity wykładnika (bias=7) + 3 bity mantysy
Zakres: ±448, 256 unikalnych wartości
Encode: float32 → clamp[-448, 448] → rebias exponent → round mantissa
```
Oszczędność: 50% VRAM na KV cache. Przy ctx=4096: ~320 MB mniej.

---

## 3. Algorytm GPTQ (kompensacja błędu)

### 3.1. Zbieranie danych kalibracyjnych

```python
# Źródło: RedPajama-1T-Sample (zróżnicowany tekst)
# Ilość: 256 próbek × 256 tokenów = 65,536 tokenów
# Fallback: WikiText-2 train (ale ma tylko ~37 długich próbek przy seqlen=512)
dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train", streaming=True)
```

### 3.2. Zbieranie macierzy Hessiana

Dla każdej macierzy wag w każdej warstwie, podczas forward pass zbieramy:
```python
# Hook na input każdego nn.Linear:
H[i,j] += Σ_tokens x_i × x_j    # outer product sum
# Po przetworzeniu:
H /= n_tokens                     # średnia kowariancja
```

Macierz Hessiana **transformujemy** rotacją Hadamarda (musi pasować do obróconych wag):
```python
H_rot = R^T @ H @ R   # blokowa transformacja, R = block_diag(H32, H32, ...)
```

### 3.3. GPTQ blokowa kompensacja błędu (Cholesky)

Dla każdego bloku 32 kolumn (od lewej do prawej):

```python
1. H_inv = cholesky_inverse(H + damping * I)     # damping = 0.01 × diag_mean
2. scale, zero_point = find_optimal(W_block)       # asymetryczne
3. W_quant = quantize(W_block, scale, zero_point)  # INT4 [0..15]
4. Error = W_block - W_quant                        # [N, 32]
5. delta = Error @ inv(H_inv_block) @ H_inv_remaining  # kompensacja Hessianu
6. delta = clamp(delta, -50×w_scale, +50×w_scale)  # miękki clamp (nie 5× jak w oryginalnym GPTQ)
7. W[:, remaining_cols] -= delta                    # aktualizacja pozostałych kolumn
```

**Kluczowa modyfikacja vs oryginalny GPTQ:**
- Miękki clamp (50× zamiast 5×) — zachowuje więcej kompensacji błędu
- Eskalacja dampingu: próbuje 1×, 2×, 5×, 10×, 50× aż Cholesky się powiedzie
- Fallback na diagonalny Hessian jeśli Cholesky zawsze failuje

---

## 4. Krok po kroku: reprodukcja

### Krok 0: Pobranie modelu
```bash
# Model automatycznie ściągany z HuggingFace
# Wymaga ~28 GB na dysku
python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('Qwen/Qwen3-14B')"
```

### Krok 1: Pomiar wrażliwości (opcjonalny, ~15 min)
```bash
python measure_sensitivity.py
# Output: sensitivity_map.json — ranking grup wagowych po wrażliwości
```

Co robi:
1. Ładuje model FP16 warstwa-po-warstwie
2. Forward pass kalibracyjny (64 próbek × 256 tokenów)
3. Kwantyzuje każdą macierz do INT4 (prosta MSE, bez GPTQ)
4. Oblicza błąd × diagonalę Hessiana
5. Ranking od najwrażliwszej do najstabilniejszej grupy

### Krok 2: Kwantyzacja główna (~20-60 min, zależy od GPU)
```bash
python quantize_v4_gptq.py \
  --model Qwen/Qwen3-14B \
  --save-dir quantized_v4_gptq \
  --sensitivity-map sensitivity_map.json \
  --nsamples 256 \
  --cal-seqlen 256 \
  --block-size 32 \
  --int8-pct 20 \
  --device cuda:0
```

Co robi per warstwa (×40):
1. Ładuje wagi FP16 z safetensors
2. Forward pass z hookami → zbiera Hessian [K, K]
3. Obraca wagi: `W_rot = W @ block_diag(H32, ...)`
4. Transformuje Hessian: `H_rot = R^T @ H @ R`
5. GPTQ kwantyzacja (Cholesky + kompensacja) → packed INT4/INT8
6. Zapisuje `layer_XXX.pt` natychmiast (odporność na crash)

Osobno kwantyzuje:
- `lm_head` — symetryczny INT4 (bez zero-pointu) z rotacją Hadamarda
- `embed.pt` — embeddingi FP16 bez zmian
- `final_norm.pt` — końcowy RMSNorm FP16

### Krok 3: Budowa kerneli HIP (inference, ~1 min)
```bash
cd hip_int4 && python setup.py build_ext --inplace
```

### Krok 4: Uruchomienie inference
```bash
python int4_engine_v5.py
# Interaktywny chat z benchmarkiem:
# Decode: ~61 t/s, Prefill: ~2000 t/s (na R9700 RDNA4)
```

---

## 5. Format wyjściowy (quantized_v4_gptq/)

### Struktura katalogowa
```
quantized_v4_gptq/
├── meta.pt              # Dict z konfiguracją modelu + precision_map
├── embed.pt             # [151936, 5120] FP16 — token embeddings
├── final_norm.pt        # [5120] FP16/BF16 — końcowy RMSNorm
├── lm_head.pt           # INT4 symmetric + Hadamard — vocab head
├── layer_000.pt         # Pierwsza warstwa
├── layer_001.pt         # ...
└── layer_039.pt         # Ostatnia warstwa
```

### Zawartość meta.pt
```python
{
    'model_name': 'Qwen/Qwen3-14B',
    'num_layers': 40,
    'hidden_size': 5120,
    'intermediate_size': 17408,
    'num_heads': 40,
    'num_kv_heads': 8,
    'head_dim': 128,
    'rms_eps': 1e-06,
    'rope_theta': 1000000.0,
    'method': 'v4_asym_had_gptq_c4cal',
    'block_size': 32,
    'use_rotation': True,
    'use_asymmetric': True,
    'precision_map': { '0': {'qkv':'int4','o':'int4','gate_up':'int4','down':'int4'}, ... },
    'nsamples': 256,
    'cal_seqlen': 256,
}
```

### Zawartość layer_XXX.pt
```python
{
    # QKV projection (7168 out × 5120 in) — INT4 asymmetric
    'qkv_packed': [7168, 2560] uint8,       # 2 nibbly/bajt
    'qkv_scales': [7168, 160] FP16,         # 160 bloków = 5120/32
    'qkv_zeros':  [7168, 160] uint8,        # zero-point per blok
    'qkv_N': 7168, 'qkv_K': 5120,
    'qkv_precision': 'int4a',               # lub 'int8'

    # O projection (5120 × 5120)
    'o_packed': [5120, 2560] uint8,
    'o_scales': [5120, 160] FP16,
    'o_zeros':  [5120, 160] uint8,
    'o_N': 5120, 'o_K': 5120,

    # Gate+Up (34816 × 5120) — gate i up skonkatenowane
    'gate_up_packed': [34816, 2560] uint8,
    'gate_up_scales': [34816, 160] FP16,
    'gate_up_zeros':  [34816, 160] uint8,
    'gate_up_N': 34816, 'gate_up_K': 5120,

    # Down (5120 × 17408)
    'down_packed': [5120, 8704] uint8,
    'down_scales': [5120, 544] FP16,        # 544 bloków = 17408/32
    'down_zeros':  [5120, 544] uint8,

    # Normy (FP16/BF16, bez kwantyzacji)
    'in_norm':   [5120] BF16,     # input_layernorm
    'post_norm': [5120] BF16,     # post_attention_layernorm
    'q_norm':    [128]  BF16,     # query norm
    'k_norm':    [128]  BF16,     # key norm
}
```

---

## 6. Architektura inference engine

### Decode (single-token, HIP C++)

Cały 40-warstwowy loop w C++ — zero narzutu Python.

Per warstwa:
```
1. Residual + RMSNorm (fused HIP kernel, 1 blok × 1024 wątków)
2. FWHT: aktywacja × Hadamard (inline w GEMV lub osobny kernel)
3. GEMV: QKV projection (INT4 → FP16 on-the-fly dequant)
4. Q_norm + K_norm (RMSNorm per head)
5. RoPE (compute cos/sin z pozycji, apply rotation)
6. KV cache write (FP8 E4M3 encode)
7. Flash Decode attention (split-K, partial softmax, reduce)
8. GEMV: O projection
9. Residual add
10. Residual + RMSNorm (fused)
11. FWHT
12. GEMV: Gate+Up projection
13. SiLU * mul (fused activation)
14. FWHT
15. GEMV: Down projection
16. Residual add
```

Final: RMSNorm → lm_head GEMV → argmax/sample

**GEMV kernel (HIP):**
- Warp-based: każdy warp liczy BLOCK_N wierszy wyjściowych
- 128-bitowe ładowanie wag (uint4 = 32 nibbly na raz)
- Asymmetric dequant inline: `(float)nibble - zp) * scale`
- Skale+zero przeplecione w pamięci: `[N, K/32, 2]` FP16

### Prefill (batch tokens, PyTorch + rocBLAS)

```
1. On-the-fly INT4→FP16 dequant (HIP kernel)
2. torch.matmul (rocBLAS GEMM) dla mnożenia macierzy
3. PyTorch SDPA (F.scaled_dot_product_attention) z is_causal=True
```

---

## 7. Kluczowe parametry (podsumowanie)

| Parametr | Wartość | Opis |
|----------|---------|------|
| **Typ kwantyzacji** | INT4 asymmetric | [0..15] z zero-pointem |
| **Rozmiar bloku** | 32 | Per 32 elementów wzdłuż K |
| **Format skali** | FP16 | 65504 unikalnych wartości |
| **Zero-point** | uint8 [0..15] | Per blok |
| **Rotacja** | Hadamard 32×32 | Block-diagonal, znormalizowana |
| **GPTQ kalibracja** | 256 próbek × 256 tokenów | RedPajama diverse text |
| **GPTQ damping** | 0.01 × diag_mean | Eskalacja jeśli Cholesky fail |
| **GPTQ delta clamp** | 50× weight_scale | Miękki (nie agresywny) |
| **Mixed precision** | 20% INT8, 80% INT4 | Na podstawie sensitivity map |
| **KV cache** | FP8 E4M3 | 1 bajt/element |
| **Packing** | `lo \| (hi << 4)` | 2 nibbly na uint8 |

---

## 8. Wyniki

### Jakość (PPL WikiText-2, sliding window ctx=2048, stride=512)
| Metoda | PPL | Δ vs BF16 |
|--------|-----|-----------|
| BF16 (baseline) | 7.556 | — |
| **INT4 v4 (nasz najlepszy)** | **7.692** | +0.136 |
| GGUF Q4_K_M (llama.cpp) | 7.715 | +0.159 |
| INT4 v5 (pure INT4, bez INT8 mix) | 7.787 | +0.231 |

### Benchmarki (250 próbek per task)
| Benchmark | INT4 v5 | GGUF Q4_K_M |
|-----------|---------|-------------|
| ARC-Challenge | **92.8%** | 90.8% |
| MMLU | **75.6%** | 72.8% |

### Prędkość (AMD R9700, Qwen3-14B)
| Metryka | INT4 Engine | GGUF (llama.cpp ROCm) |
|---------|-------------|----------------------|
| Decode (ctx=128) | **61 t/s** | 47.7 t/s |
| Prefill (pp512) | **2076 t/s** | 559 t/s |
| Context scaling (128→2048) | **-6%** | -14% |
| VRAM | 9.9 GB | 8.8 GB |

---

## 9. Pliki źródłowe

| Plik | Rola |
|------|------|
| `quantize_v4_gptq.py` | Główny skrypt kwantyzacji (GPTQ + Hadamard + mixed) |
| `int4_quant_v2.py` | Core: asymmetryczna kwantyzacja INT4, GPTQ v2 |
| `int8_quant.py` | INT8 kwantyzacja (dla wrażliwych warstw) |
| `hadamard_utils.py` | Generacja macierzy Hadamarda, rotacja wag i aktywacji |
| `measure_sensitivity.py` | Pomiar wrażliwości → sensitivity_map.json |
| `int4_engine_v5.py` | Silnik inferencji (HIP decode + PyTorch prefill) |
| `hip_int4/int4_decode_step.hip` | Kernele HIP: GEMV, FWHT, FP8 encode/decode, attention |
| `fused_ops.py` | Triton: fused RMSNorm + RoPE |
| `RESEARCH_NOTES.md` | Pełne notatki z eksperymentów |

---

## 10. Wskazówki dla reprodukcji na innym sprzęcie

### AMD RDNA3 (RX 7900 XTX, 24GB)
- gfx1100 zamiast gfx1201 — zmienić `PYTORCH_ROCM_ARCH`
- WMMA działa na gfx11 (Wave32)
- Mniej VRAM: użyć `quantized_v5_pure_int4` (czysty INT4, 7.7 GB) zamiast mixed INT4/INT8

### AMD MI300X (datacenter)
- Pełna kompatybilność ROCm
- Większy bandwidth → szybszy decode
- Tensor parallelism na 2+ chipletach

### NVIDIA (RTX 3090/4090/5090)
- Zmienić HIP → CUDA w kernelach (lub użyć PyTorch fallback)
- `hipLaunchKernelGGL` → `<<<blocks, threads>>>`
- `__shared__` i reszta API identyczne
- Alternatywnie: użyć samej kwantyzacji (quantize_v4_gptq.py działa na CUDA)
  i inny silnik inferencji (np. vLLM z GPTQ)

### Minimalna reprodukcja (bez custom kerneli)
```python
# Tylko kwantyzacja — działa na dowolnym GPU z PyTorch:
python quantize_v4_gptq.py --save-dir my_quant --device cuda:0

# Inference z PyTorch (wolne, ale działa wszędzie):
# Załaduj layer_XXX.pt, dequantyzuj on-the-fly, torch.matmul
```
