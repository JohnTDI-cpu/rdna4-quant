"""Benchmark FA2 prefill v0 vs SDPA na shape'ach Qwen3-30B-A3B (H=32, Hk=4, D=128)."""
import os
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")

import time
import torch
import torch.nn.functional as F
import int4_hip

DEV = "cuda:0"
torch.manual_seed(0)


def time_fn(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters  # ms


def bench(M, H=32, Hk=4, D=128):
    q = torch.randn(1, H, M, D, device=DEV, dtype=torch.float16) * 0.5
    k = torch.randn(1, Hk, M, D, device=DEV, dtype=torch.float16) * 0.5
    v = torch.randn(1, Hk, M, D, device=DEV, dtype=torch.float16) * 0.5

    sdpa = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    v0   = lambda: int4_hip.flash_prefill_v0(q, k, v)
    v1   = lambda: int4_hip.flash_prefill_v1(q, k, v)

    t_sdpa = time_fn(sdpa)
    t_v0   = time_fn(v0)
    t_v1   = time_fn(v1)
    flops = 2.0 * M * M * D * H
    tf_s = flops / (t_sdpa * 1e-3) / 1e12
    tf_0 = flops / (t_v0   * 1e-3) / 1e12
    tf_1 = flops / (t_v1   * 1e-3) / 1e12
    print(f"M={M:<5}  SDPA={t_sdpa:7.3f}ms ({tf_s:5.1f}TF)  v0={t_v0:7.3f}ms ({tf_0:5.1f}TF, {t_sdpa/t_v0:.2f}x)  v1={t_v1:7.3f}ms ({tf_1:5.1f}TF, {t_sdpa/t_v1:.2f}x)")


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()
    for M in [128, 256, 512, 1024, 2048, 4096, 8192]:
        bench(M)
