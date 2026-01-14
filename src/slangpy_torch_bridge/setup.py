# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Build script for slangpy_torch_bridge

This is a separate PyTorch native extension that provides fast tensor
access without Python API overhead. It's intentionally built separately
from the main slangpy extension to avoid libtorch dependency issues.

Usage:
    pip install ./src/slangpy_torch_bridge

Or for development:
    pip install -e ./src/slangpy_torch_bridge
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="slangpy_torch_bridge",
    version="0.1.0",
    description="Fast PyTorch tensor access bridge for slangpy",
    author="Shader Slang",
    ext_modules=[
        CUDAExtension(
            name="slangpy_torch_bridge",
            sources=["torch_bridge_impl.cpp"],
            include_dirs=["."],  # For tensor_bridge_api.h
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch>=2.0.0"],
)
