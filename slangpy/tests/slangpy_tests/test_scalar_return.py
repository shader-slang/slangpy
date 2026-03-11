# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for scalar (0D) return values from functions that accept tensor inputs.

Regression tests for shader-slang/slangpy#827: when torch integration is active
and the function returns a scalar, the return type must NOT default to
torch.Tensor. A 0D torch.Tensor return would produce zero-length arrays that
crash the Slang compiler during type legalization on non-CUDA targets.

These tests verify:
  - Tensor input + scalar return (0D call) works without crashing
  - Tensor input + tensor return (>0D call) produces the correct return type
  - Both torch and non-torch (numpy/slangpy.Tensor) paths behave correctly
"""

import sys
import pytest
import numpy as np

from slangpy import DeviceType, ValueRef
from slangpy.testing import helpers

SCALAR_RETURN_SOURCE = """
import slangpy;

[Differentiable]
float read_element(ITensor<float, 1> data, float idx)
{
    return data[int(idx)];
}

[Differentiable]
float add_scalars(float a, float b)
{
    return a + b;
}

[Differentiable]
float elementwise_double(float x)
{
    return x * 2.0;
}
"""


# ---------------------------------------------------------------------------
# Non-torch tests (slangpy.Tensor / numpy)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scalar_return_with_tensor_input_no_torch(device_type: DeviceType):
    """A function receiving a Tensor but returning a scalar should produce a
    plain scalar result, not a Tensor."""
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, SCALAR_RETURN_SOURCE)

    from slangpy import Tensor

    data = Tensor.from_numpy(device, np.array([10.0, 20.0, 30.0], dtype=np.float32))
    result = module.read_element(data=data, idx=1.0)
    assert result == pytest.approx(20.0), f"Expected 20.0, got {result}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scalar_return_plain_scalars_no_torch(device_type: DeviceType):
    """Pure scalar call with no tensor inputs."""
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, SCALAR_RETURN_SOURCE)

    result = module.add_scalars(3.0, 4.0)
    assert result == pytest.approx(7.0), f"Expected 7.0, got {result}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vectorized_return_with_tensor_input_no_torch(device_type: DeviceType):
    """Calling a scalar function with a Tensor should vectorize and return a Tensor."""
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, SCALAR_RETURN_SOURCE)

    from slangpy import Tensor

    data = Tensor.from_numpy(device, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    result = module.elementwise_double(data)
    assert not isinstance(result, (int, float)), (
        f"Expected a tensor-like return for vectorized call, got {type(result)}"
    )
    result_np = np.asarray(result.to_numpy()).flatten()
    np.testing.assert_allclose(result_np, [2.0, 4.0, 6.0])


# ---------------------------------------------------------------------------
# Torch tests
# ---------------------------------------------------------------------------

if sys.platform == "darwin":
    pytest.skip("PyTorch CUDA interop not available on macOS", allow_module_level=True)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    pytest.skip("PyTorch not installed", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

TORCH_DEVICE_TYPES = [
    dt for dt in helpers.DEFAULT_DEVICE_TYPES if dt != DeviceType.metal
]


@pytest.mark.parametrize("device_type", TORCH_DEVICE_TYPES)
def test_scalar_return_with_torch_tensor_input(device_type: DeviceType):
    """Regression test for #827: a torch tensor input with a scalar-returning
    function (0D call) must NOT attempt to create a torch.Tensor return.

    The return should be a plain Python scalar (via ValueRef), not a
    torch.Tensor. Creating a 0D torch.Tensor result would crash the Slang
    compiler on non-CUDA backends.
    """
    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, SCALAR_RETURN_SOURCE)

    data = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32, device="cuda")
    result = module.read_element(data=data, idx=1.0)

    assert not isinstance(result, torch.Tensor), (
        f"Expected a scalar return, got torch.Tensor. "
        f"0D calls with torch inputs should not default return_type to torch.Tensor."
    )
    assert result == pytest.approx(20.0), f"Expected 20.0, got {result}"


@pytest.mark.parametrize("device_type", TORCH_DEVICE_TYPES)
def test_vectorized_return_with_torch_tensor_input(device_type: DeviceType):
    """When a scalar function is vectorized over a torch tensor (call_dimensionality > 0),
    the result should be a torch.Tensor."""
    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, SCALAR_RETURN_SOURCE)

    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda")
    result = module.elementwise_double(data)

    assert isinstance(result, torch.Tensor), (
        f"Expected torch.Tensor return for vectorized call, got {type(result)}"
    )
    expected = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float32, device="cuda")
    assert torch.allclose(result.cpu(), expected.cpu()), (
        f"Expected {expected}, got {result}"
    )


@pytest.mark.parametrize("device_type", TORCH_DEVICE_TYPES)
def test_scalar_return_torch_scalars_only(device_type: DeviceType):
    """Pure scalar call where torch is imported but no torch tensors are passed.
    Should behave identically to the non-torch case."""
    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, SCALAR_RETURN_SOURCE)

    result = module.add_scalars(3.0, 4.0)
    assert not isinstance(result, torch.Tensor)
    assert result == pytest.approx(7.0), f"Expected 7.0, got {result}"


@pytest.mark.parametrize("device_type", TORCH_DEVICE_TYPES)
def test_scalar_return_with_torch_itensor_multiple_calls(device_type: DeviceType):
    """Calling a scalar-returning function with a torch tensor multiple times
    should not crash or leak (quick smoke test for repeated invocations)."""
    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, SCALAR_RETURN_SOURCE)

    data = torch.tensor([5.0, 15.0, 25.0, 35.0], dtype=torch.float32, device="cuda")
    for i in range(4):
        result = module.read_element(data=data, idx=float(i))
        expected = data[i].item()
        assert result == pytest.approx(expected), (
            f"Iteration {i}: expected {expected}, got {result}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
