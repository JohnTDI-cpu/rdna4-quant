"""Test correctness FA2 prefill v0 vs F.scaled_dot_product_attention.

Wymaga GPU 0 (HIP_VISIBLE_DEVICES=0).
"""
import os
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F
import int4_hip

DEV = "cuda:0"
VERSION = os.environ.get("FA_VERSION", "v1")
torch.manual_seed(0)


def ref_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)


def check(M, H, Hk, D, atol=2e-3, rtol=2e-3):
    q = torch.randn(1, H, M, D, device=DEV, dtype=torch.float16) * 0.5
    k = torch.randn(1, Hk, M, D, device=DEV, dtype=torch.float16) * 0.5
    v = torch.randn(1, Hk, M, D, device=DEV, dtype=torch.float16) * 0.5

    out_ref = ref_sdpa(q, k, v)
    fn = int4_hip.flash_prefill_v1 if VERSION == "v1" else int4_hip.flash_prefill_v0
    out_ours = fn(q, k, v)

    diff = (out_ref - out_ours).float().abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_max = (diff / (out_ref.float().abs() + 1e-6)).max().item()

    ok = torch.allclose(out_ref, out_ours, atol=atol, rtol=rtol)
    tag = "OK " if ok else "FAIL"
    print(f"[{tag}] M={M:<5} H={H:<3} Hk={Hk:<3} D={D}  max_abs={max_err:.4e}  mean_abs={mean_err:.4e}  max_rel={rel_max:.3e}")
    return ok


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Torch:  {torch.__version__}")
    print(f"Testuję wersję: {VERSION}")

    cases = [
        # (M, H, Hk, D)
        (4,    4, 1, 128),    # toy
        (16,   8, 2, 128),    # GQA ratio 4
        (64,  32, 4, 128),    # Qwen3 ratio
        (256, 32, 4, 128),
        (1024, 32, 4, 128),
        (2048, 32, 4, 128),
    ]
    fails = 0
    for M, H, Hk, D in cases:
        if not check(M, H, Hk, D):
            fails += 1
    print(f"\n{'PASS' if fails == 0 else f'FAIL ({fails} cases)'}")
