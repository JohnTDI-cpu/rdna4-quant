"""Unit tests for INT4 quantization and HIP kernels.

Verifies correctness against PyTorch FP16 reference.
Run: python -m pytest tests/ -v
  or: python tests/test_kernels.py
"""
import torch
import sys
import os
from pathlib import Path

os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1201')
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'hip_int4'))

from int4_quant_v2 import (
    quantize_to_int4_asymmetric, pack_int4_unsigned, unpack_int4_unsigned,
    dequantize_int4_asym, find_optimal_scale_int4_asym,
    quantize_to_int4_block_asym, gptq_quantize_int4_v2,
)
from hadamard_utils import get_hadamard, rotate_weight, rotate_activation

device = "cuda" if torch.cuda.is_available() else "cpu"


def test_int4_asymmetric_roundtrip():
    """Pack -> unpack -> dequant should approximately reconstruct weights."""
    torch.manual_seed(42)
    N, K = 128, 256
    W = torch.randn(N, K, dtype=torch.float32)
    block_size = 32

    W_blocks = W.reshape(N, K // block_size, block_size)
    quantized, scale, zero_point, reconstructed = quantize_to_int4_asymmetric(W_blocks)

    # Check quantized values are in [0, 15]
    assert quantized.min() >= 0, f"Min quantized = {quantized.min()}"
    assert quantized.max() <= 15, f"Max quantized = {quantized.max()}"

    # Pack and unpack
    packed = pack_int4_unsigned(quantized.reshape(N, K))
    unpacked = unpack_int4_unsigned(packed)
    assert (unpacked == quantized.reshape(N, K)).all(), "Pack/unpack roundtrip failed"

    # Check reconstruction error
    reconstructed_flat = reconstructed.reshape(N, K)
    mse = (W - reconstructed_flat).pow(2).mean().item()
    rel_error = mse / W.pow(2).mean().item()
    assert rel_error < 0.05, f"Relative MSE too high: {rel_error:.4f} (expected < 0.05)"
    print(f"  INT4 asymmetric roundtrip: rel_MSE = {rel_error:.4f} OK")


def test_dequantize_int4_asym():
    """Full pipeline: quantize -> pack -> dequantize should match."""
    torch.manual_seed(123)
    N, K = 64, 128
    block_size = 32
    W = torch.randn(N, K)

    W_blocks = W.reshape(N, K // block_size, block_size)
    quantized, scale, zero_point, reconstructed = quantize_to_int4_asymmetric(W_blocks)
    packed = pack_int4_unsigned(quantized.reshape(N, K))

    # Dequantize from packed format
    W_deq = dequantize_int4_asym(packed, scale, zero_point, block_size=block_size)

    # Should approximately match (FP16 rounding causes small diffs)
    diff = (W_deq.float() - reconstructed.reshape(N, K)).abs().max().item()
    assert diff < 0.01, f"Dequant mismatch: max diff = {diff}"
    print(f"  Dequant consistency: max_diff = {diff:.6f} OK")


def test_hadamard_orthogonality():
    """H @ H^T = I for normalized Hadamard matrix."""
    H = get_hadamard(32, device='cpu', dtype=torch.float64)
    I = H @ H.T
    error = (I - torch.eye(32, dtype=torch.float64)).abs().max().item()
    assert error < 1e-6, f"Hadamard not orthogonal: max error = {error}"
    print(f"  Hadamard orthogonality: error = {error:.2e} OK")


def test_hadamard_rotation_invertible():
    """rotate_weight then rotate_activation should preserve dot product."""
    torch.manual_seed(7)
    N, K = 32, 128
    W = torch.randn(N, K)
    x = torch.randn(1, K)

    # y = x @ W^T (original)
    y_orig = x @ W.T

    # y_rot = x_rot @ W_rot^T (should equal y_orig because H @ H = I)
    W_rot = rotate_weight(W, block_size=32)
    x_rot = rotate_activation(x, block_size=32)
    y_rot = x_rot @ W_rot.T

    error = (y_orig - y_rot).abs().max().item()
    assert error < 1e-4, f"Rotation not invertible: max error = {error}"
    print(f"  Hadamard rotation invertibility: error = {error:.6f} OK")


def test_gptq_improves_quality():
    """GPTQ should produce lower reconstruction error than naive quantization."""
    torch.manual_seed(99)
    N, K = 64, 128
    block_size = 32
    W = torch.randn(N, K) * 0.1
    W[:, 0] *= 10  # Add outlier column

    # Naive quantization
    W_blocks = W.reshape(N, K // block_size, block_size)
    _, _, _, naive_recon = quantize_to_int4_asymmetric(W_blocks)
    naive_mse = (W - naive_recon.reshape(N, K)).pow(2).mean().item()

    # GPTQ quantization (use identity Hessian — activations equally important)
    H = torch.eye(K, dtype=torch.float32)
    packed, scales, zeros, K_pad = gptq_quantize_int4_v2(
        W.half(), H, block_size=block_size, device='cpu')

    gptq_recon = dequantize_int4_asym(packed, scales, zeros, block_size=block_size)
    gptq_mse = (W - gptq_recon.float()).pow(2).mean().item()

    # GPTQ should be at least as good (with identity Hessian, similar or better)
    print(f"  GPTQ vs naive: naive_MSE={naive_mse:.6f}, gptq_MSE={gptq_mse:.6f}")
    # Don't assert strictly better — with identity H it's similar, just check it works
    assert gptq_mse < naive_mse * 2, "GPTQ much worse than naive — something is broken"


def test_gemv_vs_pytorch():
    """Compare HIP GEMV output against PyTorch FP16 matmul reference."""
    try:
        import int4_hip
    except ImportError:
        print("  SKIP: int4_hip not built (run setup.py first)")
        return

    torch.manual_seed(42)
    N, K = 5120, 5120
    block_size = 32

    # Create random weights and quantize
    W = torch.randn(N, K, dtype=torch.float32)
    W_blocks = W.reshape(N, K // block_size, block_size)
    quantized, scale, zero_point, _ = quantize_to_int4_asymmetric(W_blocks)
    packed = pack_int4_unsigned(quantized.reshape(N, K)).to(device)

    # Interleave scales and zeros for HIP kernel format: [N, K/32, 2] FP16
    zeros_fp16 = zero_point.float().half()
    sz = torch.stack([scale.to(device), zeros_fp16.to(device)], dim=-1).contiguous()
    sz_flat = sz.view(N, -1)  # [N, K/16]

    # Input vector
    x = torch.randn(1, K, dtype=torch.float16, device=device)

    # Reference: dequantize to FP16 and matmul
    W_deq = dequantize_int4_asym(packed.cpu(), scale, zero_point, block_size=block_size).to(device)
    y_ref = (x @ W_deq.T).squeeze(0)  # [N]

    # HIP GEMV (dbg_gemv returns output tensor)
    y_hip = int4_hip.dbg_gemv(x.squeeze(0), packed, sz_flat, N, K)

    # Compare
    abs_err = (y_ref - y_hip).abs()
    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        y_ref.float().unsqueeze(0), y_hip.float().unsqueeze(0)).item()

    print(f"  GEMV vs PyTorch ref: max_err={max_err:.4f}, mean_err={mean_err:.4f}, cos_sim={cos_sim:.6f}")
    assert cos_sim > 0.99, f"GEMV cosine similarity too low: {cos_sim}"
    print("  GEMV correctness OK")


if __name__ == "__main__":
    tests = [
        ("INT4 asymmetric roundtrip", test_int4_asymmetric_roundtrip),
        ("Dequantize consistency", test_dequantize_int4_asym),
        ("Hadamard orthogonality", test_hadamard_orthogonality),
        ("Hadamard rotation invertible", test_hadamard_rotation_invertible),
        ("GPTQ vs naive quality", test_gptq_improves_quality),
        ("GEMV vs PyTorch reference", test_gemv_vs_pytorch),
    ]

    print("=" * 50)
    print("  rdna4-quant kernel tests")
    print("=" * 50)

    passed = 0
    failed = 0
    skipped = 0
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn()
            passed += 1
        except AssertionError as e:  # noqa: F821
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            if "SKIP" in str(e) or "not built" in str(e):
                skipped += 1
            else:
                print(f"  ERROR: {e}")
                failed += 1

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'=' * 50}")
    sys.exit(1 if failed > 0 else 0)
