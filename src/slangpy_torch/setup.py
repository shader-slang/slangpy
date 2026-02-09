# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# pyright: reportMissingImports=none
"""
Build script for slangpy_torch

This is a separate PyTorch native extension that provides fast tensor
access without Python API overhead. It's intentionally built separately
from the main slangpy extension to avoid libtorch dependency issues.

Usage:
    pip install ./src/slangpy_torch --no-build-isolation

Or for development:
    pip install -e ./src/slangpy_torch --no-build-isolation

Note: --no-build-isolation is required to ensure the extension is compiled
against your installed PyTorch version for ABI compatibility.
"""

import os
import sys

# =============================================================================
# Pre-flight checks with helpful error messages
# =============================================================================


def _check_build_isolation():
    """Check if running in build isolation (which we don't support)."""
    # When pip uses build isolation, it creates a temporary venv without torch.
    # We can detect this by checking if we're in a temp directory or if certain
    # isolation markers are present.
    import sysconfig

    # Check for common isolation indicators
    scripts_path = sysconfig.get_path("scripts", "posix_prefix") or ""
    purelib_path = sysconfig.get_path("purelib") or ""

    # Build isolation typically uses paths containing 'pip-build-env' or similar
    isolation_markers = ["pip-build-env", "build-env", ".tmp"]
    in_isolation = any(
        marker in scripts_path or marker in purelib_path for marker in isolation_markers
    )

    return in_isolation


def _check_torch_available():
    """Check if PyTorch is available and provide helpful error if not."""
    try:
        import torch

        return torch
    except ImportError:
        # Check if we might be in build isolation
        likely_isolation = _check_build_isolation()

        error_msg = """
================================================================================
ERROR: PyTorch not found!

slangpy-torch requires PyTorch to be installed BEFORE building.
"""
        if likely_isolation:
            error_msg += """
It looks like you may be running in build isolation mode.
This package MUST be installed with --no-build-isolation:

    pip install slangpy-torch --no-build-isolation
"""
        else:
            error_msg += """
Please install PyTorch first:

    pip install torch

Then install slangpy-torch with --no-build-isolation:

    pip install slangpy-torch --no-build-isolation
"""
        error_msg += """
The --no-build-isolation flag is required to ensure slangpy-torch is compiled
against your installed PyTorch version for ABI compatibility.

For more information, see: https://github.com/shader-slang/slangpy
================================================================================
"""
        print(error_msg, file=sys.stderr)
        sys.exit(1)


def _is_sdist_only():
    """Check if we're only building an sdist (no compilation needed)."""
    # Check command line for sdist-only operations
    # 'egg_info' is called during sdist creation
    # 'sdist' is the explicit sdist command
    sdist_commands = {"sdist", "egg_info", "--version"}
    return any(cmd in sys.argv for cmd in sdist_commands)


# =============================================================================
# Build configuration
# =============================================================================

# Required for Windows MSVC builds with PyTorch extensions
# Must be set before importing torch.utils.cpp_extension
if sys.platform == "win32":
    os.environ.setdefault("DISTUTILS_USE_SDK", "1")

# Only import torch and set up extensions if we're actually building
if _is_sdist_only():
    # For sdist, we don't need torch - just package the source
    from setuptools import setup

    ext_modules = []
    cmdclass = {}
else:
    # For wheel/install, we need torch
    torch = _check_torch_available()

    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # Detect if this is a development/source install
    # Check for .git in repo root or SLANGPY_TORCH_DEBUG environment variable
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(os.path.dirname(_script_dir))  # src/slangpy_torch -> src -> repo
    _is_dev_install = os.path.exists(os.path.join(_repo_root, ".git")) or os.environ.get(
        "SLANGPY_TORCH_DEBUG", ""
    ).lower() in ("1", "true", "yes")

    # Configure compile args based on install type and platform
    if sys.platform == "win32":
        if _is_dev_install:
            # Debug info for development
            cxx_args = ["/O2", "/Zi"]
            link_args = ["/DEBUG"]
        else:
            # Release build for PyPI
            cxx_args = ["/O2"]
            link_args = []
        nvcc_args = ["-O3"]
    else:
        if _is_dev_install:
            # Debug info for development (GCC/Clang)
            cxx_args = ["-O3", "-g"]
        else:
            # Release build for PyPI
            cxx_args = ["-O3"]
        link_args = []
        nvcc_args = ["-O3"]

    ext_modules = [
        CUDAExtension(
            name="slangpy_torch",
            sources=["torch_bridge_impl.cpp"],
            include_dirs=["."],  # For tensor_bridge_api.h
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
            extra_link_args=link_args,
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="slangpy-torch",
    version="0.1.0",
    description="Fast PyTorch tensor access for slangpy",
    author="Shader Slang",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
)
