# Complete Quantization Recipe: INT4 + Hadamard + GPTQ

## Purpose

Full reproduction of the best INT4 quantization (PPL 7.692, beats GGUF Q4_K_M 7.715)
on any PC with an AMD GPU (RDNA3/4) or NVIDIA GPU with ROCm/CUDA.

Model: **Qwen3-14B** (14.77B parameters, 40 layers, hidden=5120)

---

## 1. Hardware and Software Requirements

### Hardware (minimum)
- GPU with ≥24 GB VRAM (quantization needs FP16 model + Hessian in memory)
- RAM ≥32 GB (loading safetensors)
- ~30 GB disk for FP16 model + ~8 GB for quantized output

### Tested on
- 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32GB VRAM)
- ROCm 7.1.0, PyTorch 2.10.0+rocm7.1, hipcc clang-19
- CPU: AMD Ryzen 9 9900X, 64 GB DDR5

### Software
```bash
pip install torch transformers safetensors datasets
# Optional (for inference):
pip install triton   # Triton 3.x with ROCm support
```

---

## 2. Formal Quantization Format Definition

### 2.1. Weight format: INT4 Asymmetric with zero-point

Each weight matrix W[N, K] is split into blocks of 32 elements along the K dimension.

**For each block [N, 32]:**

```
vmin = min(block)                    # block minimum
vmax = max(block)                    # block maximum
scale = (vmax - vmin) / 15.0         # FP16 scale
zero_point = round(-vmin / scale)    # uint8 in [0..15]

quantized[i] = round(W[i] / scale + zero_point)   # uint8 in [0..15]
dequantized[i] = (quantized[i] - zero_point) * scale  # ≈ W[i]
```

**Storage:**
- Data: 2 nibbles in one uint8 byte: `byte = (lo & 0xF) | ((hi & 0xF) << 4)`
- Scales: FP16 per 32-element block → `[N, K/32]`
- Zero-points: uint8 per block → `[N, K/32]`

**Effective BPW (bits-per-weight):**
- Data: 4 bits/weight
- Metadata: (16-bit scale + 8-bit zero) / 32 = 0.75 bits/weight
- Total: ~4.75 BPW

### 2.2. Preprocessing: Hadamard Rotation (block-diagonal)

Before quantization, weights are rotated by a Hadamard matrix:

```
H = sylvester_hadamard(32) / sqrt(32)   # [32, 32] normalized orthogonal matrix
W_rot[:, i*32:(i+1)*32] = W[:, i*32:(i+1)*32] @ H    # per-block column rotation
```

**Purpose:** Spreads outliers (heavy tails in weight distribution) evenly across the block.
Key property: H @ H^T = I (orthogonal), so rotation is lossless in FP16.

**At inference:** activations also undergo the same rotation:
```
x_rot = x @ block_diag(H, H, ..., H)
y = W_rot @ x_rot^T = W @ x^T   (because H @ H = I)
```

In HIP kernels this is implemented as **FWHT** (Fast Walsh-Hadamard Transform) — 5 butterfly stages on 32 float registers, no shared memory needed.

### 2.3. Mixed precision: INT4 vs INT8 (sensitivity-based)

**The 20% most sensitive weight groups** (layers x projections) are quantized to INT8 instead of INT4.

Sensitivity is measured as:
```
sensitivity_score = sum_k (W_error[k])^2 * H_diag[k]
```
where H_diag is the diagonal of the Hessian matrix (= activation covariance).

Deeper layers (31-39) are typically more sensitive → more often INT8.

### 2.4. KV cache: FP8 E4M3

KV cache during inference is stored in FP8 E4M3 (1 byte/element instead of 2 bytes FP16):
```
FP8 E4M3: 1 sign bit + 4 exponent bits (bias=7) + 3 mantissa bits
Range: +/-448, 256 unique values
Encode: float32 → clamp[-448, 448] → rebias exponent → round mantissa
```
Savings: 50% VRAM on KV cache. At ctx=4096: ~320 MB less.

---

## 3. GPTQ Algorithm (Error Compensation)

### 3.1. Collecting calibration data

```python
# Source: RedPajama-1T-Sample (diverse text)
# Amount: 256 samples x 256 tokens = 65,536 tokens
# Fallback: WikiText-2 train (but only ~37 long samples at seqlen=512)
dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train", streaming=True)
```

### 3.2. Collecting the Hessian matrix

For each weight matrix in each layer, during forward pass we collect:
```python
# Hook on input of each nn.Linear:
H[i,j] += sum_tokens x_i * x_j    # outer product sum
# After processing:
H /= n_tokens                      # mean covariance
```

The Hessian matrix is **transformed** by Hadamard rotation (must match rotated weights):
```python
H_rot = R^T @ H @ R   # block transformation, R = block_diag(H32, H32, ...)
```

### 3.3. GPTQ block error compensation (Cholesky)

For each block of 32 columns (left to right):

```python
1. H_inv = cholesky_inverse(H + damping * I)     # damping = 0.01 * diag_mean
2. scale, zero_point = find_optimal(W_block)       # asymmetric
3. W_quant = quantize(W_block, scale, zero_point)  # INT4 [0..15]
4. Error = W_block - W_quant                        # [N, 32]
5. delta = Error @ inv(H_inv_block) @ H_inv_remaining  # Hessian compensation
6. delta = clamp(delta, -50*w_scale, +50*w_scale)  # soft clamp (not 5x as in original GPTQ)
7. W[:, remaining_cols] -= delta                    # update remaining columns
```

**Key modifications vs original GPTQ:**
- Soft clamp (50x instead of 5x) — preserves more error compensation
- Damping escalation: tries 1x, 2x, 5x, 10x, 50x until Cholesky succeeds
- Fallback to diagonal Hessian if Cholesky always fails

---

## 4. Step-by-Step Reproduction

### Step 0: Download model
```bash
# Model is automatically downloaded from HuggingFace
# Requires ~28 GB disk space
python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('Qwen/Qwen3-14B')"
```

### Step 1: Sensitivity measurement (optional, ~15 min)
```bash
python measure_sensitivity.py
# Output: sensitivity_map.json — ranking of weight groups by sensitivity
```

What it does:
1. Loads FP16 model layer-by-layer
2. Calibration forward pass (64 samples x 256 tokens)
3. Quantizes each matrix to INT4 (simple MSE, no GPTQ)
4. Computes error x Hessian diagonal
5. Ranks from most sensitive to most stable group

### Step 2: Main quantization (~20-60 min, depends on GPU)
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

What it does per layer (x40):
1. Loads FP16 weights from safetensors
2. Forward pass with hooks → collects Hessian [K, K]
3. Rotates weights: `W_rot = W @ block_diag(H32, ...)`
4. Transforms Hessian: `H_rot = R^T @ H @ R`
5. GPTQ quantization (Cholesky + compensation) → packed INT4/INT8
6. Saves `layer_XXX.pt` immediately (crash-resilient)

Separately quantizes:
- `lm_head` — symmetric INT4 (no zero-point) with Hadamard rotation
- `embed.pt` — FP16 embeddings unchanged
- `final_norm.pt` — final RMSNorm FP16

### Step 3: Build HIP kernels (inference, ~1 min)
```bash
cd hip_int4 && python setup.py build_ext --inplace
```

### Step 4: Run inference
```bash
python int4_engine_v5.py
# Interactive chat with benchmark:
# Decode: ~61 t/s, Prefill: ~2000 t/s (on R9700 RDNA4)
```

---

## 5. Output Format (quantized_v4_gptq/)

### Directory structure
```
quantized_v4_gptq/
├── meta.pt              # Dict with model config + precision_map
├── embed.pt             # [151936, 5120] FP16 — token embeddings
├── final_norm.pt        # [5120] FP16/BF16 — final RMSNorm
├── lm_head.pt           # INT4 symmetric + Hadamard — vocab head
├── layer_000.pt         # First layer
├── layer_001.pt         # ...
└── layer_039.pt         # Last layer
```

### Contents of meta.pt
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

### Contents of layer_XXX.pt
```python
{
    # QKV projection (7168 out x 5120 in) — INT4 asymmetric
    'qkv_packed': [7168, 2560] uint8,       # 2 nibbles/byte
    'qkv_scales': [7168, 160] FP16,         # 160 blocks = 5120/32
    'qkv_zeros':  [7168, 160] uint8,        # zero-point per block
    'qkv_N': 7168, 'qkv_K': 5120,
    'qkv_precision': 'int4a',               # or 'int8'

    # O projection (5120 x 5120)
    'o_packed': [5120, 2560] uint8,
    'o_scales': [5120, 160] FP16,
    'o_zeros':  [5120, 160] uint8,
    'o_N': 5120, 'o_K': 5120,

    # Gate+Up (34816 x 5120) — gate and up concatenated
    'gate_up_packed': [34816, 2560] uint8,
    'gate_up_scales': [34816, 160] FP16,
    'gate_up_zeros':  [34816, 160] uint8,
    'gate_up_N': 34816, 'gate_up_K': 5120,

    # Down (5120 x 17408)
    'down_packed': [5120, 8704] uint8,
    'down_scales': [5120, 544] FP16,        # 544 blocks = 17408/32
    'down_zeros':  [5120, 544] uint8,

    # Norms (FP16/BF16, not quantized)
    'in_norm':   [5120] BF16,     # input_layernorm
    'post_norm': [5120] BF16,     # post_attention_layernorm
    'q_norm':    [128]  BF16,     # query norm
    'k_norm':    [128]  BF16,     # key norm
}
```

---

## 6. Inference Engine Architecture

### Decode (single-token, HIP C++)

Full 40-layer loop in C++ — zero Python overhead.

Per layer:
```
1. Residual + RMSNorm (fused HIP kernel, 1 block x 1024 threads)
2. FWHT: activation x Hadamard (inline in GEMV or separate kernel)
3. GEMV: QKV projection (INT4 → FP16 on-the-fly dequant)
4. Q_norm + K_norm (RMSNorm per head)
5. RoPE (compute cos/sin from position, apply rotation)
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
- Warp-based: each warp computes BLOCK_N output rows
- 128-bit weight loads (uint4 = 32 nibbles at once)
- Asymmetric dequant inline: `(float)nibble - zp) * scale`
- Scales+zeros interleaved in memory: `[N, K/32, 2]` FP16

### Prefill (batch tokens, PyTorch + rocBLAS)

```
1. On-the-fly INT4→FP16 dequant (HIP kernel)
2. torch.matmul (rocBLAS GEMM) for matrix multiply
3. PyTorch SDPA (F.scaled_dot_product_attention) with is_causal=True
```

---

## 7. Key Parameters (Summary)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Quantization type** | INT4 asymmetric | [0..15] with zero-point |
| **Block size** | 32 | Per 32 elements along K |
| **Scale format** | FP16 | 65504 unique values |
| **Zero-point** | uint8 [0..15] | Per block |
| **Rotation** | Hadamard 32x32 | Block-diagonal, normalized |
| **GPTQ calibration** | 256 samples x 256 tokens | RedPajama diverse text |
| **GPTQ damping** | 0.01 x diag_mean | Escalation if Cholesky fails |
| **GPTQ delta clamp** | 50x weight_scale | Soft (not aggressive) |
| **Mixed precision** | 20% INT8, 80% INT4 | Based on sensitivity map |
| **KV cache** | FP8 E4M3 | 1 byte/element |
| **Packing** | `lo \| (hi << 4)` | 2 nibbles per uint8 |

---

## 8. Results

### Quality (PPL WikiText-2, sliding window ctx=2048, stride=512)
| Method | PPL | Delta vs BF16 |
|--------|-----|---------------|
| BF16 (baseline) | 7.556 | — |
| **INT4 v4 (our best)** | **7.692** | +0.136 |
| GGUF Q4_K_M (llama.cpp) | 7.715 | +0.159 |
| INT4 v5 (pure INT4, no INT8 mix) | 7.787 | +0.231 |

### Benchmarks (250 samples per task)
| Benchmark | INT4 v5 | GGUF Q4_K_M |
|-----------|---------|-------------|
| ARC-Challenge | **92.8%** | 90.8% |
| MMLU | **75.6%** | 72.8% |

### Speed (AMD R9700, Qwen3-14B)
| Metric | INT4 Engine | GGUF (llama.cpp ROCm) |
|--------|-------------|----------------------|
| Decode (ctx=128) | **61 t/s** | 47.7 t/s |
| Prefill (pp512) | **2076 t/s** | 559 t/s |
| Context scaling (128→2048) | **-6%** | -14% |
| VRAM | 9.9 GB | 8.8 GB |

---

## 9. Source Files

| File | Role |
|------|------|
| `quantize_v4_gptq.py` | Main quantization script (GPTQ + Hadamard + mixed) |
| `int4_quant_v2.py` | Core: asymmetric INT4 quantization, GPTQ v2 |
| `int8_quant.py` | INT8 quantization (for sensitive layers) |
| `hadamard_utils.py` | Hadamard matrix generation, weight and activation rotation |
| `measure_sensitivity.py` | Sensitivity measurement → sensitivity_map.json |
| `int4_engine_v5.py` | Inference engine (HIP decode + PyTorch prefill) |
| `hip_int4/int4_decode_step.hip` | HIP kernels: GEMV, FWHT, FP8 encode/decode, attention |
| `fused_ops.py` | Triton: fused RMSNorm + RoPE |
| `RESEARCH_NOTES.md` | Full experiment notes |

---

## 10. Reproduction Tips for Other Hardware

### AMD RDNA3 (RX 7900 XTX, 24GB)
- gfx1100 instead of gfx1201 — change `PYTORCH_ROCM_ARCH`
- WMMA works on gfx11 (Wave32)
- Less VRAM: use `quantized_v5_pure_int4` (pure INT4, 7.7 GB) instead of mixed INT4/INT8

### AMD MI300X (datacenter)
- Full ROCm compatibility
- Higher bandwidth → faster decode
- Tensor parallelism across 2+ chiplets

### NVIDIA (RTX 3090/4090/5090)
- Change HIP → CUDA in kernels (or use PyTorch fallback)
- `hipLaunchKernelGGL` → `<<<blocks, threads>>>`
- `__shared__` and the rest of the API are identical
- Alternatively: use only the quantization (quantize_v4_gptq.py works on CUDA)
  and a different inference engine (e.g. vLLM with GPTQ)

### Minimal reproduction (no custom kernels)
```python
# Quantization only — works on any GPU with PyTorch:
python quantize_v4_gptq.py --save-dir my_quant --device cuda:0

# Inference with PyTorch (slow but works everywhere):
# Load layer_XXX.pt, dequantize on-the-fly, torch.matmul
```
