# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Smoke tests for slangpy_torch benchmark functions (torch_bridge_impl.cpp).
"""

import sys
import pytest

if sys.platform == "darwin":
    pytest.skip("Torch tests require CUDA", allow_module_level=True)

try:
    import torch
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


def test_benchmark_c_api():
    from slangpy_torch import benchmark_c_api

    t = torch.randn(4, 4, device="cuda")
    result = benchmark_c_api(t, iterations=100)
    assert "per_call_ns" in result
    assert "data_ptr" in result
    assert "ndim" in result
    assert "numel" in result
    assert result["ndim"] == 2
    assert result["numel"] == 16


def test_benchmark_libtorch():
    from slangpy_torch import benchmark_libtorch

    t = torch.randn(3, 5, device="cuda")
    result = benchmark_libtorch(t, iterations=100)
    assert "per_call_ns" in result
    assert "data_ptr" in result
    assert "numel" in result
    assert result["numel"] == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
