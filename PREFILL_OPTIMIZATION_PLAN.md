# Prefill Optimization Plan — ported tricks from NVIDIA/AMD/RADV research (2026-06-19)

Synthesis of 4 parallel research agents (Blackwell/Hopper GEMM, fused-quant/CUTLASS, how-RADV-does-it,
FlashAttention-3/4). **The headroom is real and the path is known**: AMD's own Gluon GEMM tutorial goes
25%→99% peak (CDNA); the portable subset reaches ~70% on RDNA4. Our prefill GEMM sits at ~25% (=ROCm/RADV
level) because we're missing the software pipeline + swizzle + the RADV inner-loop structure.

## The core diagnosis (source-verified)

- **RDNA4 is in the async-copy dead zone**: no cp.async/TMA (gfx9/10 had direct-to-LDS, gfx1250+ has async; gfx1201 has neither). Load path = `global_load → VGPR → ds_store → LDS → ds_load → VGPR → WMMA`. This is why the matrix cores starve at 25%.
- **RDNA4 NEW vs RDNA3** (the levers that make software pipelining work): out-of-order memory returns + **split wait counters** (`vmcnt`/`dscnt`/`kmcnt`) + halved A/B VGPR budget (no lane-duplication).
- **Why RADV beats ROCm on Q6_K**: identical DP4A matmul, but ROCm carries the `ql|qh` recombine + **16 int8 sub-block scales as integer multiplies INSIDE the dot** → VGPR spill >256 → throughput halves. RADV does **dequant-once-to-LDS** (centered int8 `qs[8]+d_scales`), inner loop = pure DP4A + 2 float scales **outside** the dot. **Structural, replicable — not driver magic.**
- **Our ~16% activation-quant gap**: we run a separate quant pass (or re-quant per output-tile). RADV/vLLM/TRT-LLM **fuse the activation quant into the preceding RMSNorm op** (writes int8 directly). Zero separate pass, zero redundancy.
- **No MXFP4 hardware on RDNA4**: no block-scaled WMMA (that's CDNA4/Blackwell). Element types: fp16/bf16/int8/int4/**fp8(e4m3)/bf8**. For MXFP4 stay at FP8 WMMA (2× fp16) + per-32-K E8M0 scale as accumulator exponent-add; don't dequant FP4→FP16 (loses the compute win, measured 40.8 TFLOPS = 53% only).

## Prioritized port plan

### Phase 1 — GEMM software pipeline (25% → ~70%, the biggest win)
| # | Trick | Source | Gain | Notes |
|---|---|---|---|---|
| **P1** | **Multi-stage LDS pipeline (2-3 stages)** | Blackwell/Gluon | **25%→~70%** | prefetch tile N+1 (`global_load→ds_store`) while WMMA on N. Exploit split `vmcnt`/`dscnt` + out-of-order memory. 2-3 stages NOT 7 (no async DMA, VGPR budget). |
| **P2** | **XOR-swizzled LDS layout** | Blackwell/IREE | **~28%** | bank-conflict-free `ds_load_b128` across 32 lanes/banks. Removing it cost IREE 28%. |
| **P3** | **Dequant-once-to-LDS, scales OUTSIDE the dot** | RADV-how | the Q6_K cliff | stage weight as centered int8 in LDS once; inner loop pure WMMA + per-group float scale. NOT 16 scales inside the dot (ROCm's spill). |
| **P4** | **Accumulator tiling ≥2×2 (avoid WMMA RAW `v_nop` hazard)** | RADV-how/ACO | free | consecutive WMMAs must write different VGPRs, or interleave the next tile's dequant between them (clears hazard + overlaps). |
| **P5** | **gfx12 transposing loads** (`global_load_tr_b128`/`ds_read_*_tr`) | RADV-how | shuffle-free | lands 16×16 tile in WMMA register layout with zero `v_perm`. RADV emits these; naive HIP wastes work. |
| **P6** | **Symmetric ping-pong (NOT producer/consumer)** | HipKittens | latency hide | 2 waves/SIMD alternate load/WMMA, `s_barrier`. Producer/consumer **regresses on AMD** (893 vs 1610 TFLOPS — static register partition). |
| **P7** | **int4 WMMA** (`iu4_w32_gfx12`, K=32, raw builtin) for 4-bit weights | RADV-how | 4× int | only native K=32 dense op; rocWMMA doesn't expose it. |

### Phase 2 — activation quant (the ~16% in-forward gap)
| **P8** | **Fuse activation-quant into the RMSNorm op** (write int8+scale directly) | fused-quant/vLLM | ~16% | the vLLM/TRT-LLM pattern (`rms_norm_dynamic_per_token_quant`). NOT the GEMM prologue (that's redundant N/BN×M/BM — our earlier failure). Per-K-block scales so it streams. |

### Phase 3 — FA2 attention (long-ctx)
| **P9** | **Software exp2 polynomial off the SFU** (Horner 3rd-order + Cody-Waite) | FA-4 | >+3% | exp/TLU is 1/4-rate but co-executes with WMMA; emulate on full-rate VALU. (our exp2 was the easy half.) |
| **P10** | **Intra-warp GEMM-softmax pipeline** (`sched_group_barrier`/`s_setprio`) | FA-3/HipKittens | past 50% peak | overlap exp(N-1) with WMMA(N) **in-warp, no `__syncthreads`** (multi-warp failed — sync tax). |
| **P11** | **GQA packing** (fold 4-way head group into Q tile) | FA-4/FlashInfer | 4× K/V reuse | load K/V once for the 4 Q-heads sharing a KV-head (H=32,Hk=8). Watch VGPR. |
| **P12** | **Conditional softmax rescaling** (skip if Δmax<τ) | FA-4 | fewer non-matmul ops | pure algorithm. |

### Phase 4 — structural / long-term
| **P13** | **Persistent + Stream-K (hybrid) scheduler** | Blackwell/FlashInfer | wave-quant + causal balance | cooperativeLaunch works gfx1201/ROCm7.2.3 (our measurement), or global-workspace atomics. |
| **P14** | **Sparse / sliding-window attention** | FA-4 | O(M²) wall | the only thing that breaks quadratic; fits our quant/sparsity track (long-term). |

## What NOT to port (proven traps)
- Producer/consumer warp specialization (FA-3/Hopper) — **regresses on AMD** (static register partition).
- TMA / wgmma / tcgen05 / TMEM / 128×128 tiles / 2-CTA MMA / block-scaled MMA — Hopper/Blackwell/CDNA4 hardware we don't have.
- Async global→LDS — gfx1250+, not gfx1201.

## References
- AMD Gluon GEMM tutorial (25%→99%): rocm.blogs.amd.com/.../gluon-gemm-tutorial
- llama.cpp PR #17156 (HIP WMMA-MMQ for RDNA4) — closest reference kernel
- llama.cpp Vulkan `mul_mmq.comp` / `mul_mmq_funcs.glsl` (RADV's fast path) — local /home/janusz/llama_new
- HipKittens (arXiv 2511.08083), FlashAttention-4 (arXiv 2603.05451), FA-3 (2407.08608)
- rdna4-wmma-guide (ours): github.com/JohnTDI-cpu/rdna4-wmma-guide

## Expected outcome
GEMM 25%→~70% peak ≈ ~140 TFLOPS standalone → **beats RADV (51) and ROCm (71) decisively**. The RMSNorm-quant
fusion closes the ~16% in-forward gap. FA2 (exp2 + intra-warp pipeline + GQA-pack) pushes past 50% and extends
the win to long context. **First strike: P1+P2+P3 (pipeline + swizzle + dequant-once) on the gate-shape GEMM.**
