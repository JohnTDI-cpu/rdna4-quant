#!/usr/bin/env python3
"""
MXFP4 Model Converter — HuggingFace → MXFP4 format for AMD RDNA4.

Converts a HuggingFace Qwen3 model to our MXFP4 format with optional
learned rotation optimization. Produces a directory of .pt files ready
for inference with mxfp4_engine.py.

Output format:
  quantized_dir/
    meta.pt          — model metadata
    embed.pt         — FP16 embedding weights
    final_norm.pt    — FP16 final RMSNorm weights
    lm_head.pt       — MXFP4 lm_head (Hadamard rotation)
    layer_000.pt     — MXFP4 layer 0 (packed weights + scales + rotations)
    ...
    layer_039.pt     — MXFP4 layer 39

Usage:
  # Quick RTN quantization (~5 min):
  python convert.py --model Qwen/Qwen3-14B --method rtn

  # Learned rotation (~3h on 2× R9700):
  python convert.py --model Qwen/Qwen3-14B --method learned --steps 500

  # Multi-GPU sequential (fastest for learned rotation):
  python convert.py --model Qwen/Qwen3-14B --method learned --steps 500 --multi-gpu

Requirements:
  - ROCm 7.1+, PyTorch 2.10+
  - HuggingFace model cached locally
  - ~16 GB VRAM per GPU
"""

import os, sys, subprocess, time, argparse
from pathlib import Path

os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1201'

def main():
    parser = argparse.ArgumentParser(description='Convert HuggingFace model to MXFP4')
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name (e.g. Qwen/Qwen3-14B)')
    parser.add_argument('--method', type=str, default='learned',
                        choices=['rtn', 'hadamard', 'learned'],
                        help='Quantization method (default: learned)')
    parser.add_argument('--steps', type=int, default=200,
                        help='Optimization steps for learned rotation (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for learned rotation (default: 0.01)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: quantized_<method>/)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='GPU device (default: cuda:0)')
    parser.add_argument('--multi-gpu', action='store_true',
                        help='Use 2 GPUs sequentially (layers 0-19 on GPU0, 20-39 on GPU1)')
    parser.add_argument('--block-size', type=int, default=32,
                        help='Rotation block size (default: 32)')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Calibration samples (default: 128)')
    args = parser.parse_args()

    # Determine output directory
    if args.output:
        save_dir = Path(args.output)
    else:
        method_name = args.method
        if method_name == 'learned':
            method_name = f'learned_{args.steps}'
        save_dir = Path(f'quantized_{method_name}')
    save_dir.mkdir(exist_ok=True)

    script_dir = Path(__file__).parent
    env_activate = ""  # Set to "source /path/to/venv/bin/activate" if using a virtualenv
    env_vars = "HSA_OVERRIDE_GFX_VERSION=12.0.1"

    print(f"MXFP4 Converter")
    print(f"  Model:   {args.model}")
    print(f"  Method:  {args.method}")
    if args.method == 'learned':
        print(f"  Steps:   {args.steps}")
        print(f"  LR:      {args.lr}")
    print(f"  Output:  {save_dir}")
    print(f"  Device:  {args.device}")
    print()

    t_start = time.time()

    if args.method == 'rtn':
        # Simple RTN quantization — fast, ~5 min
        _run_rtn(args, save_dir, script_dir)

    elif args.method == 'hadamard':
        # Hadamard rotation + MSE/GPTQ — ~15 min
        _run_hadamard(args, save_dir, script_dir)

    elif args.method == 'learned':
        if args.multi_gpu:
            _run_learned_multigpu(args, save_dir, script_dir)
        else:
            _run_learned_single(args, save_dir, script_dir)

    total_time = time.time() - t_start
    print(f"\nConversion complete in {total_time/60:.1f} minutes")
    print(f"Output: {save_dir}")
    print(f"\nTo evaluate perplexity:")
    print(f"  python bench_ppl_learned.py --quant-dir {save_dir}")
    print(f"\nTo run inference:")
    print(f"  python generate_mxfp4.py --quant-dir {save_dir} --prompt 'Hello'")


def _run_rtn(args, save_dir, script_dir):
    """Simple Round-To-Nearest quantization with Hadamard rotation."""
    print("Running Hadamard+MSE (RTN) quantization...")
    cmd = [
        sys.executable, str(script_dir / 'quantize_hadamard.py'),
        '--model', args.model,
        '--save-dir', str(save_dir),
        '--block-size', str(args.block_size),
    ]
    _run(cmd)


def _run_hadamard(args, save_dir, script_dir):
    """Hadamard rotation + GPTQ quantization."""
    print("Running Hadamard+GPTQ quantization...")
    cmd = [
        sys.executable, str(script_dir / 'quantize_hadamard_gptq.py'),
        '--model', args.model,
        '--save-dir', str(save_dir),
        '--block-size', str(args.block_size),
        '--nsamples', str(args.nsamples),
    ]
    _run(cmd)


def _run_learned_single(args, save_dir, script_dir):
    """Learned rotation on single GPU."""
    print(f"Running learned rotation ({args.steps} steps) on {args.device}...")
    cmd = [
        sys.executable, str(script_dir / 'quantize_learned_rotation.py'),
        '--model', args.model,
        '--steps', str(args.steps),
        '--lr', str(args.lr),
        '--save-dir', str(save_dir),
        '--device', args.device,
        '--block-size', str(args.block_size),
        '--nsamples', str(args.nsamples),
    ]
    _run(cmd)


def _run_learned_multigpu(args, save_dir, script_dir):
    """Learned rotation on 2 GPUs sequentially."""
    import torch
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"WARNING: Only {num_gpus} GPU(s) detected, falling back to single GPU")
        _run_learned_single(args, save_dir, script_dir)
        return

    hidden_path = '/tmp/cal_hidden_after_layer19.pt'
    quant_script = str(script_dir / 'quantize_learned_rotation.py')

    # Phase 1: GPU 0, layers 0-19
    print(f"Phase 1: GPU 0, layers 0-19 ({args.steps} steps each)...")
    cmd1 = [
        sys.executable, quant_script,
        '--model', args.model,
        '--steps', str(args.steps),
        '--lr', str(args.lr),
        '--save-dir', str(save_dir),
        '--device', 'cuda:0',
        '--start-layer', '0', '--end-layer', '20',
        '--save-hidden', hidden_path,
        '--block-size', str(args.block_size),
        '--nsamples', str(args.nsamples),
    ]
    _run(cmd1)

    # Phase 2: GPU 1, layers 20-39
    print(f"\nPhase 2: GPU 1, layers 20-39 ({args.steps} steps each)...")
    env = os.environ.copy()
    env['HIP_VISIBLE_DEVICES'] = '1'
    cmd2 = [
        sys.executable, quant_script,
        '--model', args.model,
        '--steps', str(args.steps),
        '--lr', str(args.lr),
        '--save-dir', str(save_dir),
        '--device', 'cuda:0',  # cuda:0 because HIP_VISIBLE_DEVICES=1 remaps
        '--start-layer', '20', '--end-layer', '40',
        '--load-hidden', hidden_path,
        '--block-size', str(args.block_size),
        '--nsamples', str(args.nsamples),
    ]
    _run(cmd2, env=env)


def _run(cmd, env=None):
    """Run a subprocess, streaming output."""
    if env is None:
        env = os.environ.copy()
    env['HSA_OVERRIDE_GFX_VERSION'] = '12.0.1'

    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)
    proc.wait()
    if proc.returncode != 0:
        print(f"ERROR: Process exited with code {proc.returncode}")
        sys.exit(1)


if __name__ == '__main__':
    main()
