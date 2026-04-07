# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Tests targeting C++ coverage gaps in torch integration (Tier 2).

Covers:
- torch_bridge_impl.cpp benchmark_c_api / benchmark_libtorch (~41 lines)
- slangpytorchtensor.cpp create_output rare scalar types (~12 lines)
- slangpytorchtensor.cpp create_dispatchdata (~20 lines)
- slangpytorchtensor.cpp DiffPair read_signature / get_shape (~25 lines)
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

import numpy as np
import slangpy as spy
from slangpy.testing.helpers import get_device, create_function_from_module


@pytest.fixture(scope="module")
def device():
    return get_device(spy.DeviceType.cuda)


# ============================================================================
# Benchmark smoke tests (torch_bridge_impl.cpp, ~41 lines)
# ============================================================================


class TestBenchmarks:
    """Smoke tests for slangpy_torch benchmark functions."""

    def test_benchmark_c_api(self):
        from slangpy_torch import benchmark_c_api

        t = torch.randn(4, 4, device="cuda")
        result = benchmark_c_api(t, iterations=100)
        assert "per_call_ns" in result
        assert "data_ptr" in result
        assert "ndim" in result
        assert "numel" in result
        assert result["ndim"] == 2
        assert result["numel"] == 16

    def test_benchmark_libtorch(self):
        from slangpy_torch import benchmark_libtorch

        t = torch.randn(3, 5, device="cuda")
        result = benchmark_libtorch(t, iterations=100)
        assert "per_call_ns" in result
        assert "data_ptr" in result
        assert "numel" in result
        assert result["numel"] == 15


# ============================================================================
# create_output rare scalar types (slangpytorchtensor.cpp, ~12 lines)
# Already covered: int32, int64, float32, float64
# ============================================================================


@pytest.mark.parametrize(
    "slang_type,slang_func,torch_dtype",
    [
        ("uint8_t", "uint8_t inc(uint8_t x) { return x + uint8_t(1); }", torch.uint8),
        ("int8_t", "int8_t neg(int8_t x) { return -x; }", torch.int8),
        ("int16_t", "int16_t dbl(int16_t x) { return x * int16_t(2); }", torch.int16),
        ("bool", "bool flip(bool x) { return !x; }", torch.bool),
    ],
    ids=["uint8", "int8", "int16", "bool"],
)
def test_torch_output_rare_scalar(device, slang_type, slang_func, torch_dtype):
    """Auto-create output tensor for rare scalar types."""
    func = create_function_from_module(device, slang_func.split("(")[0].split()[-1], slang_func)
    if torch_dtype == torch.bool:
        inp = torch.tensor([True, False, True], device="cuda", dtype=torch_dtype)
        result = func(inp)
        assert result.dtype == torch.bool
        assert torch.equal(result, torch.tensor([False, True, False], device="cuda"))
    elif torch_dtype == torch.uint8:
        inp = torch.tensor([1, 2, 3], device="cuda", dtype=torch_dtype)
        result = func(inp)
        assert result.dtype == torch.uint8
        assert torch.equal(result, torch.tensor([2, 3, 4], device="cuda", dtype=torch.uint8))
    elif torch_dtype == torch.int8:
        inp = torch.tensor([1, -2, 3], device="cuda", dtype=torch_dtype)
        result = func(inp)
        assert result.dtype == torch.int8
        assert torch.equal(result, torch.tensor([-1, 2, -3], device="cuda", dtype=torch.int8))
    else:
        inp = torch.tensor([5, 10], device="cuda", dtype=torch_dtype)
        result = func(inp)
        assert result.dtype == torch_dtype
        assert torch.equal(result, torch.tensor([10, 20], device="cuda", dtype=torch_dtype))


# ============================================================================
# create_dispatchdata for torch tensors (slangpytorchtensor.cpp, ~20 lines)
# Triggered by raw dispatch path (function with dispatchThreadID first param)
# ============================================================================


RAW_DISPATCH_SRC = r"""
import "slangpy";

void copy_vals(uint3 dispatchThreadID, RWTensor<float,1> output, Tensor<float,1> input) {
    output[dispatchThreadID.x] = input[dispatchThreadID.x] * 2.0;
}
"""


def test_torch_raw_dispatch(device):
    """Raw dispatch with torch tensor triggers create_dispatchdata."""
    import slangpy.torchintegration.torchtensormarshall  # noqa: F401 — register torch.Tensor handler

    mod = spy.Module.load_from_source(device, "raw_torch_dispatch", RAW_DISPATCH_SRC)

    inp = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=torch.float32)
    out = torch.zeros(4, device="cuda", dtype=torch.float32)
    mod.copy_vals.dispatch(spy.uint3(4, 1, 1), output=out, input=inp)
    expected = torch.tensor([2.0, 4.0, 6.0, 8.0], device="cuda", dtype=torch.float32)
    assert torch.allclose(out, expected)


# ============================================================================
# DiffPair paths (slangpytorchtensor.cpp, ~25 lines)
# - read_signature builds cache key from primal+grad signatures
# - get_shape grad-only fallback
# ============================================================================


DIFF_SRC = r"""
[Differentiable]
float square(float x) { return x * x; }
"""


def test_diffpair_read_signature(device):
    """Calling a function with a torch DiffPair triggers read_signature (~16 lines).

    The DiffPair flows through the type registry as NativeTorchTensorDiffPair,
    and when the call signature is built, read_signature is invoked to create
    a unique cache key combining primal and grad tensor signatures.
    """
    from slangpy.torchintegration import diff_pair

    func = create_function_from_module(device, "square", DIFF_SRC)

    primal = torch.tensor([2.0, 3.0, 4.0], device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.ones(3, device="cuda", dtype=torch.float32)
    pair = diff_pair(primal, grad)

    result = func(pair)
    assert result is not None


def test_diffpair_get_shape_grad_only(device):
    """get_shape with primal=None falls back to grad tensor for shape (~3 lines)."""
    from slangpy.torchintegration import diff_pair

    grad = torch.ones(5, device="cuda", dtype=torch.float32)
    pair = diff_pair(None, grad)

    func = create_function_from_module(device, "square", DIFF_SRC)
    result = func(pair)
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
