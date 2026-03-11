"""Build INT4 HIP extension with decode step for PyTorch ROCm.

Auto-detects GPU architecture. Override with:
  PYTORCH_ROCM_ARCH=gfx1100 python setup.py build_ext --inplace
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os, subprocess


def detect_arch():
    """Detect GPU arch from env, PyTorch, or rocminfo."""
    arch = os.environ.get('PYTORCH_ROCM_ARCH', '')
    if arch:
        return arch
    try:
        import torch
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, 'gcnArchName', '')
        if arch:
            return arch
    except Exception:
        pass
    try:
        out = subprocess.check_output(['rocminfo'], stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            if 'gfx' in line and 'Name:' in line:
                a = line.split()[-1].strip()
                if a.startswith('gfx'):
                    return a
    except Exception:
        pass
    return 'gfx1201'


GPU_ARCH = detect_arch()
os.environ['PYTORCH_ROCM_ARCH'] = GPU_ARCH
print(f"Building for GPU architecture: {GPU_ARCH}")

setup(
    name='int4_hip',
    ext_modules=[
        CUDAExtension(
            'int4_hip',
            ['int4_decode_step.hip'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', f'--offload-arch={GPU_ARCH}', '-std=c++17',
                         '-Wno-unused-result'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
