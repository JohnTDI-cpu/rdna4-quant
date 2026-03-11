"""Build W4A4 WMMA GEMM extension for PyTorch ROCm."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1201')

setup(
    name='wmma_gemm',
    ext_modules=[
        CUDAExtension(
            'wmma_gemm',
            ['int4_wmma_gemm_v2.hip'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '--offload-arch=gfx1201', '-std=c++17',
                         '-Wno-unused-result'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
