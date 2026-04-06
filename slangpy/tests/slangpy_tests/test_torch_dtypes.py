# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for torch tensor output auto-creation with different scalar types.

Exercises the create_output dtype branches in slangpytorchtensor.cpp
by calling Slang functions that return non-float32 types with torch.Tensor inputs.
"""

import pytest
import torch

import slangpy as spy
from slangpy.testing.helpers import get_device, create_function_from_module


@pytest.fixture(scope="module")
def device():
    return get_device(spy.DeviceType.cuda)


def test_torch_output_int32(device):
    """Auto-create int32 output tensor from Slang int return type."""
    func = create_function_from_module(
        device, "double_it", "int double_it(int x) { return x * 2; }"
    )
    inp = torch.tensor([3, 7, 11], device="cuda", dtype=torch.int32)
    result = func(inp)
    assert result.dtype == torch.int32
    assert torch.equal(result, torch.tensor([6, 14, 22], device="cuda", dtype=torch.int32))


def test_torch_output_int64(device):
    """Auto-create int64 output tensor from Slang int64_t return type."""
    func = create_function_from_module(
        device, "add64", "int64_t add64(int64_t x, int64_t y) { return x + y; }"
    )
    a = torch.tensor([100, 200], device="cuda", dtype=torch.int64)
    b = torch.tensor([300, 400], device="cuda", dtype=torch.int64)
    result = func(a, b)
    assert result.dtype == torch.int64
    assert torch.equal(result, torch.tensor([400, 600], device="cuda", dtype=torch.int64))


def test_torch_output_float64(device):
    """Auto-create float64 output tensor from Slang double return type."""
    func = create_function_from_module(
        device, "add_d", "double add_d(double a, double b) { return a + b; }"
    )
    a = torch.tensor([1.5, 3.0], device="cuda", dtype=torch.float64)
    b = torch.tensor([2.5, 4.0], device="cuda", dtype=torch.float64)
    result = func(a, b)
    assert result.dtype == torch.float64
    assert torch.allclose(result, torch.tensor([4.0, 7.0], device="cuda", dtype=torch.float64))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
