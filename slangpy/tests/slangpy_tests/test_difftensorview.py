# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys
import numpy as np
from pathlib import Path

from slangpy import DeviceType, Tensor, diff_pair
from slangpy.core.module import Module
from slangpy.testing import helpers

# Skip all tests in this file if running on MacOS
if sys.platform == "darwin":
    pytest.skip("DiffTensorView requires CUDA, not available on macOS", allow_module_level=True)

# DiffTensorView only works with CUDA device type
DEVICE_TYPES = [DeviceType.cuda] if DeviceType.cuda in helpers.DEFAULT_DEVICE_TYPES else []
if not DEVICE_TYPES:
    pytest.skip("DiffTensorView requires CUDA device type", allow_module_level=True)


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def load_module_torch(device_type: DeviceType):
    device = helpers.get_torch_device(device_type)
    return Module.load_from_file(
        device,
        str(Path(__file__).parent / "test_difftensorview.slang"),
    )


def load_module(device_type: DeviceType):
    device = helpers.get_device(type=device_type)
    return Module.load_from_file(
        device,
        str(Path(__file__).parent / "test_difftensorview.slang"),
    )


# ============================================================================
# Tests with slangpy.Tensor (exercises tensorcommon.py DiffTensorView path)
# ============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_difftensorview_copy_tensor(device_type: DeviceType):
    """Test copy_difftensorview with slangpy.Tensor — exercises tensorcommon resolve_types for DiffTensorView."""
    module = load_module(device_type)
    device = helpers.get_device(type=device_type)

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_tensor = Tensor.from_numpy(device, data)
    output_tensor = Tensor.zeros(device, (5,), "float")

    module.copy_difftensorview(input_tensor, output_tensor)

    result = output_tensor.to_numpy()
    assert np.allclose(result, data, atol=1e-5)


# ============================================================================
# Tests with torch.Tensor
# ============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_difftensorview_copy_torch(device_type: DeviceType):
    """Test copy_difftensorview with torch.Tensor arguments."""
    module = load_module_torch(device_type)

    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=torch.float32)
    output_tensor = torch.zeros(5, device="cuda", dtype=torch.float32)

    module.copy_difftensorview(input_tensor, output_tensor)
    torch.cuda.synchronize()

    assert torch.allclose(
        input_tensor, output_tensor
    ), f"Expected {input_tensor}, got {output_tensor}"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_difftensorview_add_torch(device_type: DeviceType):
    """Test add_difftensorview with torch.Tensor arguments."""
    module = load_module_torch(device_type)

    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=torch.float32)
    b = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], device="cuda", dtype=torch.float32)
    output_tensor = torch.zeros(5, device="cuda", dtype=torch.float32)

    module.add_difftensorview(a, b, output_tensor)
    torch.cuda.synchronize()

    expected = a + b
    assert torch.allclose(expected, output_tensor), f"Expected {expected}, got {output_tensor}"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_difftensorview_diff_square_torch(device_type: DeviceType):
    """Test backward pass of diff_square with torch.Tensor: f(x) = x^2, df/dx = 2x."""
    module = load_module_torch(device_type)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=torch.float32)
    x_grad = torch.zeros(5, device="cuda", dtype=torch.float32)
    output = torch.zeros(5, device="cuda", dtype=torch.float32)
    output_grad = torch.ones(5, device="cuda", dtype=torch.float32)

    module.diff_square.bwds(diff_pair(x, x_grad), diff_pair(output, output_grad))
    torch.cuda.synchronize()

    # df/dx = 2x
    expected_grad = 2.0 * x
    assert torch.allclose(
        x_grad, expected_grad, atol=1e-5
    ), f"Expected grad {expected_grad}, got {x_grad}"


# ============================================================================
# Tests with slangpy.Tensor
# ============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_difftensorview_copy_slangpy_tensor(device_type: DeviceType):
    """Test copy_difftensorview with slangpy.Tensor arguments."""
    module = load_module(device_type)

    input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_tensor = Tensor.from_numpy(module.device, input_data)
    output_tensor = Tensor.zeros(module.device, (5,), "float")

    module.copy_difftensorview(input_tensor, output_tensor)

    output_data = output_tensor.to_numpy()
    assert np.array_equal(input_data, output_data), f"Expected {input_data}, got {output_data}"


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_difftensorview_add_slangpy_tensor(device_type: DeviceType):
    """Test add_difftensorview with slangpy.Tensor arguments."""
    module = load_module(device_type)

    a_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    b_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    a = Tensor.from_numpy(module.device, a_data)
    b = Tensor.from_numpy(module.device, b_data)
    output = Tensor.zeros(module.device, (5,), "float")

    module.add_difftensorview(a, b, output)

    output_data = output.to_numpy()
    expected = a_data + b_data
    assert np.array_equal(expected, output_data), f"Expected {expected}, got {output_data}"


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_difftensorview_diff_square_slangpy_tensor(device_type: DeviceType):
    """Test backward pass of diff_square with slangpy.Tensor: f(x) = x^2, df/dx = 2x."""
    module = load_module(device_type)

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_tensor = Tensor.from_numpy(module.device, data).with_grads()
    output_tensor = Tensor.zeros(module.device, (5,), "float").with_grads()
    output_tensor.grad_in.copy_from_numpy(np.ones(5, dtype=np.float32))

    module.diff_square.bwds(input_tensor, output_tensor)

    # df/dx = 2x
    grad = input_tensor.grad_out.to_numpy()
    expected_grad = 2.0 * data
    assert np.allclose(grad, expected_grad, atol=1e-5), f"Expected grad {expected_grad}, got {grad}"


# ============================================================================
# Tests for _thread_count with CUDAKernel + Differentiable
# ============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_difftensorview_kernel_forward(device_type: DeviceType):
    """Test forward pass of CUDAKernel diff_square_kernel with _thread_count."""
    module = load_module_torch(device_type)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=torch.float32)
    output = torch.zeros(5, device="cuda", dtype=torch.float32)
    count = x.numel()

    module.diff_square_kernel(count=count, input=x, output=output, _thread_count=count)
    torch.cuda.synchronize()

    expected = x * x
    assert torch.allclose(output, expected), f"Expected {expected}, got {output}"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_difftensorview_kernel_backward(device_type: DeviceType):
    """Test backward pass of CUDAKernel diff_square_kernel: f(x) = x^2, df/dx = 2x."""
    module = load_module_torch(device_type)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=torch.float32)
    x_grad = torch.zeros(5, device="cuda", dtype=torch.float32)
    output = torch.zeros(5, device="cuda", dtype=torch.float32)
    output_grad = torch.ones(5, device="cuda", dtype=torch.float32)
    count = x.numel()

    module.diff_square_kernel.bwds(
        count=count,
        input=diff_pair(x, x_grad),
        output=diff_pair(output, output_grad),
        _thread_count=count,
    )
    torch.cuda.synchronize()

    expected_grad = 2.0 * x
    assert torch.allclose(
        x_grad, expected_grad, atol=1e-5
    ), f"Expected grad {expected_grad}, got {x_grad}"
