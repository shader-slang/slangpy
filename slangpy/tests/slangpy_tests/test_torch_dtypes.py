# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for torch tensor output auto-creation with different scalar types.

Exercises the create_output dtype branches in slangpytorchtensor.cpp
by calling Slang functions that return non-float32 types with torch.Tensor inputs.
"""

import sys

import pytest

torch = pytest.importorskip("torch")

import slangpy as spy
from slangpy.testing.helpers import get_device, create_function_from_module

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="Torch tests require CUDA")


@pytest.fixture(scope="module")
def device() -> spy.Device:
    return get_device(spy.DeviceType.cuda)


def test_torch_output_int32(device: spy.Device) -> None:
    """Auto-create int32 output tensor from Slang int return type."""
    func = create_function_from_module(
        device, "double_it", "int double_it(int x) { return x * 2; }"
    )
    inp = torch.tensor([3, 7, 11], device="cuda", dtype=torch.int32)
    result = func(inp)
    assert result.dtype == torch.int32
    assert torch.equal(result, torch.tensor([6, 14, 22], device="cuda", dtype=torch.int32))


def test_torch_output_int64(device: spy.Device) -> None:
    """Auto-create int64 output tensor from Slang int64_t return type."""
    func = create_function_from_module(
        device, "add64", "int64_t add64(int64_t x, int64_t y) { return x + y; }"
    )
    a = torch.tensor([100, 200], device="cuda", dtype=torch.int64)
    b = torch.tensor([300, 400], device="cuda", dtype=torch.int64)
    result = func(a, b)
    assert result.dtype == torch.int64
    assert torch.equal(result, torch.tensor([400, 600], device="cuda", dtype=torch.int64))


def test_torch_output_float64(device: spy.Device) -> None:
    """Auto-create float64 output tensor from Slang double return type."""
    func = create_function_from_module(
        device, "add_d", "double add_d(double a, double b) { return a + b; }"
    )
    a = torch.tensor([1.5, 3.0], device="cuda", dtype=torch.float64)
    b = torch.tensor([2.5, 4.0], device="cuda", dtype=torch.float64)
    result = func(a, b)
    assert result.dtype == torch.float64
    assert torch.allclose(result, torch.tensor([4.0, 7.0], device="cuda", dtype=torch.float64))


@pytest.mark.parametrize(
    "slang_func,torch_dtype",
    [
        ("uint8_t inc(uint8_t x) { return x + uint8_t(1); }", torch.uint8),
        ("int8_t neg(int8_t x) { return -x; }", torch.int8),
        ("int16_t dbl(int16_t x) { return x * int16_t(2); }", torch.int16),
        ("bool flip(bool x) { return !x; }", torch.bool),
    ],
    ids=["uint8", "int8", "int16", "bool"],
)
def test_torch_output_rare_scalar(device: spy.Device, slang_func: str, torch_dtype: torch.dtype) -> None:
    """Auto-create output tensor for rare scalar types."""
    func_name = slang_func.split("(")[0].split()[-1]
    func = create_function_from_module(device, func_name, slang_func)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
