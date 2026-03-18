# RDNA4 ISA - Skrocone odniesienie dla projektu INT4 LLM

> Zrodlo: [AMD RDNA4 Instruction Set Architecture Reference Guide](https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture) (7-April-2025, 707 stron)
> Lokalna kopia: `~/Pobrane/rdna4-instruction-set-architecture.pdf`

---

## 1. Architektura sprzetu (s.3-8)

### WGP (Work-Group Processor) - podstawowa jednostka obliczeniowa
- Kazdy WGP zawiera **2 CU** (Compute Units), kazdy CU ma **2 SIMD32**
- Razem = **4 SIMD32 na WGP** (4 x 32 = 128 lane'ow)
- SIMD32 = jednostka VALU przetwarzajaca 32 work-items rownolegle

### Rejestry na SIMD (s.15-18)
| Zasob | Rozmiar | Uwagi |
|-------|---------|-------|
| **SGPR** | 128 x 32-bit na SIMD | Wspoldzielone przez wave (skalarne) |
| **VGPR** | 1536 x 32-bit na SIMD (wave32) | Prywatne per lane, alokowane w blokach po 12 (dynamicznie) |
| **VGPR** | 768 x 32-bit na SIMD (wave64) | Polowa bo 2 pasy |

**Kluczowe dla occupancy:** Im mniej VGPR uzywa kernel, tym wiecej wave'ow moze byc aktywnych.
Alokacja VGPR w blokach po 12 (wave32) lub 24 (wave64). Max 256 VGPR per wave.

### LDS - Local Data Share (s.7, s.144)
- **128 KB na WGP** (64 KB per CU w trybie CU, 128 KB shared w trybie WGP)
- **64 banki**, kazdy 512 wpisow x 4 bajty
- Max **64 KB na work-group**
- Dwa tryby: **CU mode** (LDS podzielone na polowy per CU) vs **WGP mode** (128KB wspolne)
- Bank conflicts: unikaj dostepow do tego samego banku z roznych lane'ow

### Hierarchia cache (s.7-8)
```
VGPR/SGPR -> L0 (per WGP, R/W Texture) -> GL1 (write-combining) -> L2 (R/W, per memory channel) -> VRAM
                                                                                    ^
                                           Constant Cache (read-only) ------+       |
                                           Instruction Cache (per SIMD) ----+       |
```
- **L0** - per WGP, cache textowy R/W
- **GL1** - bufor write-combining (scatter/store)
- **L2** - R/W cache z atomics, per kana pamietowy
- Cache-less load mozliwy (bypass do device memory)

> **Dla nas:** R9700 AI PRO ma ~640 GB/s bandwidth. Nasze GEMV kernele osiagaja ~88% (545 GB/s).

---

## 2. Wave32 vs Wave64 (s.9-10)

| Cecha | Wave32 | Wave64 |
|-------|--------|--------|
| Work-items | 32 | 64 |
| VALU issue | 1 cykl | 2 cykle (low half + high half) |
| VOPD (dual-issue) | TAK | NIE |
| EXEC mask | 32-bit | 64-bit |
| WMMA | TAK | TAK (wiecej danych per instrukcja) |

> **Dla nas:** Uzywamy **wave32** - lepsze dla decode (warp shuffle reduction, VOPD dual-issue). Wave64 daje 2x dane per WMMA ale kosztem 2x cykli VALU.

---

## 3. Instrukcje DOT product (s.80-82) - KLUCZOWE dla INT4

### V_DOT - dot product w jednym VGPR per lane

| Instrukcja | Opis | Zastosowanie |
|-----------|------|-------------|
| `V_DOT4_I32_IU4` | 4x nibble(4-bit) dot -> I32 | **Nasz glowny candidate! INT4 dot product** |
| `V_DOT8_I32_IU4` | 8x nibble(4-bit) dot -> I32 | **8 elementow INT4 naraz!** |
| `V_DOT4_I32_IU8` | 4x byte dot -> I32 | Dla warstw INT8 (sensitive layers) |
| `V_DOT4_U32_U8` | 4x unsigned byte dot -> U32 | Wariant unsigned |
| `V_DOT8_U32_U4` | 8x unsigned nibble dot -> U32 | Unsigned INT4 x8 |
| `V_DOT2_F32_F16` | 2x FP16 dot -> F32 | Dla aktywacji FP16 |
| `V_DOT2_F32_BF16` | 2x BF16 dot -> F32 | BF16 wariant |
| `V_DOT4_F32_FP8_FP8` | 4x FP8 dot -> F32 | Dla FP8 KV cache dot products |
| `V_DOT4_F32_BF8_FP8` | 4x mixed FP8/BF8 -> F32 | Mixed precision |

**NEG[1:0] repurposed:** Dla DOT...IU: NEG[0]=signed A, NEG[1]=signed B (0=unsigned, 1=signed)

> **Dla nas:** `V_DOT8_I32_IU4` przetwarza **8 elementow INT4 w jednym cyklu** per lane.
> Przy 32 lane'ach wave32 = **256 INT4 MAC per cykl per SIMD**.
> 4 SIMD per WGP = 1024 INT4 MAC per cykl per WGP.

### Inline constants z DOT (s.82)
- `DOT4_I32_IU8`, `DOT8_I32_IU4` etc. uzywaja 32-bit inline src0/1 (ignore OPSEL)
- 8-bit i 4-bit integer inline constants dzialaja normalnie

---

## 4. WMMA - Wave Matrix Multiply Accumulate (s.89-96) - DLA PREFILL

### Dostepne instrukcje WMMA (Table 41, s.90)

| Instrukcja | Matrix A | Matrix B | Matrix C | Result |
|-----------|----------|----------|----------|--------|
| `V_WMMA_I32_16X16X16_IU4` | 16x16 IU4 | 16x16 IU4 | 16x16 I32 | 16x16 I32 |
| `V_WMMA_I32_16X16X32_IU4` | 16x32 IU4 | 32x16 IU4 | 16x16 I32 | 16x16 I32 |
| `V_WMMA_I32_16X16X16_IU8` | 16x16 IU8 | 16x16 IU8 | 16x16 I32 | 16x16 I32 |
| `V_WMMA_F32_16X16X16_F16` | 16x16 F16 | 16x16 F16 | 16x16 F32 | 16x16 F32 |
| `V_WMMA_F32_16X16X16_BF16` | 16x16 BF16 | 16x16 BF16 | 16x16 F32 | 16x16 F32 |
| `V_WMMA_F32_16X16X16_FP8_*` | 16x16 FP8 | 16x16 FP8/BF8 | 16x16 F32 | 16x16 F32 |
| `V_WMMA_F16_16X16X16_F16` | 16x16 F16 | 16x16 F16 | 16x16 F16 | 16x16 F16 |

### Sparse WMMA (SWMMAC) - 2:4 structured sparsity!

| Instrukcja | A (sparse) | B (dense) | C | Result |
|-----------|------------|-----------|---|--------|
| `V_SWMMAC_I32_16X16X32_IU4` | 16x32 IU4 | 32x16 IU4 | 16x16 I32 | 16x16 I32 |
| `V_SWMMAC_I32_16X16X64_IU4` | **16x64 IU4** | **64x16 IU4** | 16x16 I32 | 16x16 I32 |
| `V_SWMMAC_I32_16X16X32_IU8` | 16x32 IU8 | 32x16 IU8 | 16x16 I32 | 16x16 I32 |
| `V_SWMMAC_F32_16X16X32_FP8_*` | 16x32 FP8 | 32x16 FP8/BF8 | 16x16 F32 | 16x16 F32 |

> **Dla nas (prefill):** `V_WMMA_I32_16X16X32_IU4` robi 16x16x32 = **8192 INT4 MAC per instrukcja**.
> SWMMAC z IU4 64-wide: **16x16x64 = 16384 INT4 MAC** per instrukcja (ze sparsity).

### WMMA data hazards (s.90-91)
- Miedzy dwoma WMMA z tym samym A/B/indeksem: **wymagany 1 V_NOP** (lub niezalezna VALU)
- WMMA -> WMMA z tym samym D jako C: **stall jesli rozny typ lub ABS/NEG**
- WMMA -> VALU czytajacy D: **moze stallowac**
- Matryca jest **rozlozona na wszystkie lane** (nie jedna matryca per lane!)

### Matrix layout w VGPR (s.91-92)
- **Macierz A:** jeden wiersz rozlozony w VGPRach jednego lane'a
- **Macierze B, C, D:** jeden wiersz rozlozony **miedzy lane'ami** w jednym VGPR
- Dla IU4, wave32: lane = `{col[3], row[3:0]}`, vgpr = `0`, startPosn = `col[2:0]`

> **Nasz DISCOVERY_GFX12_WMMA_OUTPUT.md** potwierdza:
> `VGPR[lane][j] = matrix[(lane/16)*8 + j][lane % 16]` - lane'y indeksuja KOLUMNY!

---

## 5. Cache controls: SCOPE i Temporal-Hint (s.39-42)

### SCOPE (zasiag widocznosci)
| Wartosc | Nazwa | Opis |
|---------|-------|------|
| 0 | SE (Shader Engine) | Widocznosc w ramach SE |
| 1 | SA | Shader Array |
| 2 | WGP | Tylko ten WGP |
| 3 | DEVICE | Cale GPU |

### TH (Temporal Hint) - kontrola cachowania
| TH | Opis | Zastosowanie |
|----|------|-------------|
| 0 | RT (regular temporal) | Normalne cachowanie |
| 1 | NT (non-temporal) | Nie cache'uj (streaming) |
| 2 | HT (high-temporal) | Cache agresywnie w L0 |
| 3 | LU (last-use) | Dane do wyrzucenia po uzyciu |
| 4 | NT_RT | Non-temporal for near, RT for far scope |

> **Dla nas:** Wagi ladujemy z `TH=NT` (streaming, nie brudzic cache). Aktywacje z `TH=HT` (reuse miedzy warstwami).

---

## 6. Packed Math i 16-bit operacje (s.77-82)

### 16-bit Math (s.77)
- VGPR trzyma **2 x FP16** (V0.L = bits[15:0], V0.H = bits[31:16])
- OPSEL wybiera high/low polowke
- Packed math operuje na **obu polowkach rownolegle**

### Packed Math ops (s.80)
- `V_PK_MUL_F16`, `V_PK_ADD_F16`, `V_PK_FMA_F16` - 2x FP16 per cykl
- `V_PK_ADD_I16`, `V_PK_MUL_LO_U16` - integer packed
- `V_DOT2_F32_F16` / `V_DOT2_F32_BF16` - 2-element dot product

### 8-bit Math / FP8 (s.78)
- **FP8 (E4M3):** zakres +-448, min normal 0.0078, brak INF (max = NaN)
- **BF8 (E5M2):** zakres +-57344, min normal 3.05e-5
- Konwersje: `CVT_PK_FP8_F32`, `CVT_SR_FP8_F32` (stochastic rounding!)

> **Dla nas:** FP8 E4M3 uzywamy w KV cache. `CVT_PK_FP8_F32` pakuje 2 wartosci F32->FP8 w 16 bitach.
> Stochastic rounding (`CVT_SR_*`) moze poprawic jakosc kwantyzacji.

---

## 7. Dual-Issue VALU - VOPD (s.82-84)

VOPD koduje **2 niezalezne operacje VALU w 1 instrukcji** (tylko wave32!).

### OPX opcodes
`V_DUAL_FMAC_F32`, `V_DUAL_MUL_F32`, `V_DUAL_ADD_F32`, `V_DUAL_MOV_B32`, ...

### OPY opcodes (superset)
Jak OPX + `V_DUAL_DOT2ACC_F32_F16`, `V_DUAL_DOT2ACC_F32_BF16`, `V_DUAL_CNDMASK_B32`, ...

### Restrykcje
- Dest VGPRs: jeden musi byc **even**, drugi **odd**
- SRCX0 i SRCY0 musza byc z **roznych bankow VGPR** (bank = VGPR % 4)
- Max 2 SGPR, max 1 literal
- Operacje musza byc **niezalezne**

> **Dla nas:** Mozemy robic `FMAC + MOV` albo `MUL + ADD` w jednym cyklu.
> `V_DUAL_DOT2ACC_F32_F16` - **dual-issue dot product!** Podwaja throughput FP16 dot.

---

## 8. Cross-Lane: DPP i PERMLANE (s.84-88)

### DPP16 - swizzle w grupach 16 lane'ow
- `DPP_ROW_SL{1:15}` - shift left
- `DPP_ROW_SR{1:15}` - shift right (nasz warp shuffle!)
- `DPP_ROW_RR{1:15}` - rotate right
- `DPP_ROW_SHARE{0:15}` - broadcast
- `DPP_ROW_XMASK{0:15}` - XOR mask

### DPP8 - arbitrary swizzle w grupach 8 lane'ow
- 8 x 3-bit selektory (SEL0..SEL7): kazdy lane wybiera zrodlo z lane'ow 0-7

### PERMLANE
| Instrukcja | Opis |
|-----------|------|
| `V_PERMLANE16_B32` | Gather 16-lane groups z uniform control |
| `V_PERMLANE16_VAR_B32` | j.w. ale unique select per lane |
| `V_PERMLANEX16_B32` | Cross-group (0-15 czyta 16-31 i odwrotnie) |
| `V_PERMLANE64_B32` | Swap upper/lower 32 lanes (NOP w wave32) |

> **Dla nas:** Uzywamy `DPP_ROW_SR` do warp-shuffle reduction (6 shuffle per accumulator).
> `V_READLANE` do ekstrakcji skalarnej z dowolnego lane.

---

## 9. Global Memory Operations (s.135-141)

### Typy adresowania
- **GLOBAL** - bezposredni dostep, nie blokuje LDS zasobow (preferowane!)
- **FLAT** - generyczny, automatycznie rozpoznaje global/LDS/scratch
- **BUFFER** (VBUFFER) - przez deskryptor bufora (128-bit resource)

### Instrukcje GLOBAL load/store
- `GLOBAL_LOAD_{UBYTE,USHORT,DWORD,DWORDX2,DWORDX3,DWORDX4}`
- `GLOBAL_LOAD_BLOCK` - **block load** do wielu VGPRs

### Kluczowe dla bandwidth
- Max load per instrukcja: **DWORDX4 = 128 bit = 16 bajtow per lane**
- Wave32 x 128-bit = **512 bajtow per load** (16 cache lines jesli coalesced)
- **Coalescing:** sasiednie lane'y powinny czytac sasiednie adresy

### SCOPE i TH na loads
```
GLOBAL_LOAD_DWORDX4 vdst, vaddr, saddr SCOPE:SE TH:NT  // streaming, no cache pollution
GLOBAL_LOAD_DWORDX4 vdst, vaddr, saddr SCOPE:SE TH:HT  // cache aggressively in L0
```

> **Dla nas:** 128-bit vectorized loads (DWORDX4) = 32 wartosci INT4 naraz per lane.
> Przy 32 lane'ach: **1024 INT4 wartosci per load instrukcja**.

---

## 10. WMMA Matrix Load with Transpose (s.142-143)

### GLOBAL_LOAD_TR instrukcje (NOWE w RDNA4!)
Laduja dane z pamieci globalnej i transponuja do formatu WMMA fragment.

| Instrukcja | Opis |
|-----------|------|
| `GLOBAL_LOAD_TR_B128_w32` | Load + transpose 128-bit (wave32) |
| `GLOBAL_LOAD_TR_B64_w32` | Load + transpose 64-bit (wave32) |

> **Potencjalne zastosowanie:** Ladowanie macierzy B (wag) bezposrednio do WMMA fragmentow
> bez recznego przeukladania danych. Krytyczne dla prefill kerneli WMMA.

---

## 11. LDS Operations (s.144-154)

### Kluczowe instrukcje DS
- `DS_READ_B32/B64/B128` - load z LDS
- `DS_WRITE_B32/B64/B128` - store do LDS
- `DS_READ2_B32/B64` - **2 niezalezne loady** (rozne offsets)
- `DS_SWIZZLE_B32` - swizzle w 32 lane'ach (rotate, broadcast, swap)
- `DS_PERMUTE_B32` / `DS_BPERMUTE_B32` - permutacja miedzy lane'ami

### Bank conflicts
- 64 banki, 4-byte granularity
- 2+ lane'y czytajace ten sam bank = **serializacja** (wyjtek: broadcast jesli ten sam adres)

> **Dla nas:** DS_SWIZZLE do redukcji w shared memory (flash attention partial reduce).
> DS_READ2 do jednoczesnego ladowania scale+zero w kwantyzacji.

---

## 12. Barriers i synchronizacja (s.48-53)

### S_BARRIER / S_BARRIER_SIGNAL / S_BARRIER_WAIT
- Synchronizacja wave'ow w work-group
- Max **32 work-groups** per WGP

### Memory dependency counters (s.52)
| Counter | Tracks |
|---------|--------|
| LOADcnt | Outstanding VMEM loads |
| STOREcnt | Outstanding VMEM stores |
| DScnt | Outstanding LDS operations |
| KMcnt | Outstanding scalar memory |
| BVHcnt | Outstanding ray-tracing |
| SAMPLEcnt | Outstanding texture samples |

```
S_WAIT_LOADcnt 0    // czekaj az wszystkie VMEM loads sie zakoncza
S_WAIT_DScnt 0      // czekaj az wszystkie LDS ops sie zakoncza
```

> **Dla nas:** `S_WAIT_LOADcnt` po ladowaniu wag, przed rozpoczeciem obliczen DOT/WMMA.

---

## 13. Instrukcje konwersji danych (s.78-79)

| Instrukcja | Opis |
|-----------|------|
| `V_CVT_F32_I32` | INT32 -> FP32 |
| `V_CVT_F16_F32` | FP32 -> FP16 |
| `V_CVT_PK_FP8_F32` | 2x F32 -> 2x FP8 packed |
| `V_CVT_SR_FP8_F32` | F32 -> FP8 ze stochastic rounding |
| `V_CVT_PK_BF8_F32` | 2x F32 -> 2x BF8 packed |
| `V_CVT_F32_FP8` | FP8 -> F32 |
| `V_CVT_PK_I16_F32` | F32 -> I16 packed |
| `V_CVT_OFF_F32_I4` | **INT4 offset -> F32** (dequant helper!) |

> **Dla nas:** `V_CVT_OFF_F32_I4` - konwersja 4-bit int do float! Moze przyspieszyc dequantyzacje.

---

## 14. Scalar ALU - przydatne instrukcje (s.58-64)

| Instrukcja | Opis | Zastosowanie |
|-----------|------|-------------|
| `S_ADD_I32/U32` | Scalar add | Wskazniki, liczniki |
| `S_MUL_I32` | Scalar multiply | Obliczanie offsetow |
| `S_LSHL_B32/B64` | Shift left | Mnozenie przez potegi 2 |
| `S_BFE_U32/I32` | Bit field extract | Wydobywanie nibble'ow z INT4! |
| `S_PACK_HH/HL/LH/LL_B32_B16` | Pack 16-bit halves | Pakowanie FP16 par |
| `S_GETREG_B32` | Read HW register | Odczyt cycle counter (profiling) |
| `S_SETREG_B32` | Write HW register | Ustawianie rounding mode |

---

## 15. Mapowanie na nasz projekt

### Decode (GEMV) - terazniejszosc
```
Nasz aktualny flow per warp:
1. GLOBAL_LOAD_DWORDX4  -> 128-bit load wag INT4 (32 wartosci)
2. GLOBAL_LOAD_USHORT   -> load scale+zero FP16
3. V_BFE_U32            -> ekstrakcja nibble'ow (dequant)
4. V_CVT_F32_I32        -> konwersja na float
5. V_FMA_F32            -> scale * (nibble - zero)
6. V_FMA_F32            -> akumulacja dot product
7. DPP_ROW_SR           -> warp shuffle reduction
8. V_READFIRSTLANE_B32  -> wynik skalarny
```

### Potencjalne optymalizacje z ISA
| Optymalizacja | Instrukcja ISA | Zysk |
|--------------|---------------|------|
| INT4 dot product zamiast recznej dequant | `V_DOT8_I32_IU4` | **8x mniej instrukcji** per element |
| Dual-issue dla niezaleznych operacji | `VOPD (V_DUAL_*)` | **2x throughput** wybranych ops |
| WMMA dla prefill batched GEMM | `V_WMMA_I32_16X16X32_IU4` | **Masywny throughput** |
| Sparse WMMA jesli wagi pruned | `V_SWMMAC_I32_16X16X64_IU4` | **2x throughput** vs dense |
| Transpose load dla WMMA | `GLOBAL_LOAD_TR_B128` | **Zero-cost transpose** |
| FP8 dot dla KV attention | `V_DOT4_F32_FP8_FP8` | Natywne FP8 compute |
| Stochastic rounding przy kwantyzacji | `V_CVT_SR_FP8_F32` | Lepsza jakosc |

### Prefill (GEMM) - przyszlosc
```
Optymalny flow z WMMA:
1. GLOBAL_LOAD_TR_B128   -> load + transpose wag do WMMA fragment
2. GLOBAL_LOAD_DWORDX4   -> load aktywacji
3. V_WMMA_I32_16X16X32_IU4 -> 8192 INT4 MAC per instrukcja!
4. V_CVT_F32_I32         -> dequant wynikow
5. Repeat for K tiles
```

---

## 16. Narzedzia

- **AMD Matrix Instruction Calculator:** https://github.com/ROCm/amd_matrix_instruction_calculator
  - Throughput, register usage, lane mappings per WMMA
- **GPUOpen WMMA blog:** https://gpuopen.com/learn/wmma_on_rdna3/ (RDNA3, ale layout zblizone)
- **ROCm on Radeon:** https://rocm.docs.amd.com/projects/radeon/en/latest/index.html
- **LLVM AMDGPU docs:** https://llvm.org/docs/AMDGPUUsage.html

---

## Quick-reference: kluczowe numery stron w PDF

| Temat | Strony PDF |
|-------|-----------|
| Hardware overview, block diagram | 5-8 |
| Wave32/Wave64 | 9-10 |
| SGPR/VGPR limits, occupancy | 15-18 |
| LDS modes (CU vs WGP) | 11, 144 |
| Cache SCOPE/TH hints | 39-42 |
| DOT product instructions | 80-82 |
| WMMA instructions (Table 41) | 89-90 |
| WMMA data hazards | 90-91 |
| WMMA matrix layout in VGPRs | 91-96 |
| Sparse WMMA (SWMMAC) | 90, 96 |
| Packed math (FP16 dual) | 80-82 |
| VOPD dual-issue | 82-84 |
| DPP cross-lane (shuffle) | 84-88 |
| FP8/BF8 data types | 78-79 |
| Data conversion (CVT) | 78-79 |
| Global/Flat memory ops | 135-141 |
| WMMA Load-Transpose | 142-143 |
| LDS operations | 144-154 |
| Memory dependency (S_WAIT) | 52-56 |
| Microcode formats (VOP3P etc.) | 162-205 |
| Full instruction reference | 206-697 |

---

## 17. SWMMAC 2:4 Sparsity — empiryczne odkrycia (2026-03-18)

### V_SWMMAC_I32_16X16X64_IU4 na gfx1201

Potwierdzone działanie! Raw throughput: **477 TOPS** (vs 261 TOPS dense WMMA = 1.83×).

### Index format (częściowo zdekodowany)

```
Sygnatura: __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(neg_a, A, neg_b, B, C, idx, clamp)
  A: v2i (8 bytes) — compressed sparse weights (16 non-zero nibbles z 32 original)
  B: v4i (16 bytes) — dense activations (32 nibbles)
  C: v8i (32 bytes) — INT32 accumulator
  idx: short (16 bits) — sparsity pattern index

Per lane (kg = lane/16, nl = lane%16):
  kg selects K-half: 0→first 32 K elements, 1→second 32 K elements
  Each K-half: 32 elements → 8 groups of 4
  idx encodes 2 bits per group → 16 bits total per lane
```

### Empiryczne wyniki

```
A = [1,1,0,...] (first group non-zero)
B = [1,2,3,4,0,...] (first group has unique values)

idx_bits=0: result=465 → selects B positions (0,1): 1*1 + 1*2 = 3
idx_bits=1: result=466 → selects B positions (0,2): 1*1 + 1*3 = 4
idx_bits=2: result=467 → selects B positions (0,3): 1*1 + 1*4 = 5
idx_bits=3: result=468 → selects B positions (1,2): 1*2 + 1*3 = 5

Mapping 2-bit → pair of positions:
  0 → (0,1)
  1 → (0,2)
  2 → (0,3)
  3 → (1,2)
  
Missing pairs: (1,3) and (2,3) — ISA constraint "idx0 < 2"
```

### Constraint

Z ISA strony 96: pierwsza pozycja non-zero MUSI być w dolnej połowie grupy (pozycja 0 lub 1).
To ogranicza 2:4 pattern do 4 z 6 możliwych kombinacji.

Pruning musi respektować ten constraint: w każdej grupie 4, jedno z 2 zachowanych elementów 
musi być na pozycji 0 lub 1. Nigdy nie może być (2,3) ani (1,3).
