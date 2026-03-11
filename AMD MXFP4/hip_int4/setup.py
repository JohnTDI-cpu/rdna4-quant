"""Build INT4 HIP extension with decode step for PyTorch ROCm."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1201')

setup(
    name='int4_hip',
    ext_modules=[
        CUDAExtension(
            'int4_hip',
            ['int4_decode_step.hip'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '--offload-arch=gfx1201', '-std=c++17',
                         '-Wno-unused-result'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
