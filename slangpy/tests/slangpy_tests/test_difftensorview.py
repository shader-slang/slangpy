# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys
import numpy as np
from pathlib import Path
from typing import Any

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
# Non-contiguous / sliced tensor tests
# ============================================================================
DIFFTV_SLICE_CASES = [
    pytest.param(6, lambda t: t[:3], id="prefix"),
    pytest.param(6, lambda t: t[1:4], id="offset"),
    pytest.param(6, lambda t: t[::2], id="strided"),
    pytest.param(3, lambda t: t.flip(0), id="flipped"),
    pytest.param(9, lambda t: t.reshape(3, 3).diagonal(), id="diagonal"),
]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("source_size,slicer", DIFFTV_SLICE_CASES)
def test_difftensorview_sliced_input(device_type: DeviceType, source_size: int, slicer: Any):
    """Test that DiffTensorView correctly reads from a sliced input."""
    module = load_module_torch(device_type)

    input_full = torch.arange(1.0, source_size + 1.0, device="cuda", dtype=torch.float32)
    input_sliced = slicer(input_full)
    output = torch.zeros(3, device="cuda", dtype=torch.float32)

    module.copy_difftensorview(input_sliced, output)
    torch.cuda.synchronize()

    assert torch.allclose(output, input_sliced), f"Expected {input_sliced}, got {output}"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("full_size,slicer", DIFFTV_SLICE_CASES)
def test_difftensorview_sliced_output(device_type: DeviceType, full_size: int, slicer: Any):
    """Test that DiffTensorView write-back correctly targets only the sliced region."""
    module = load_module_torch(device_type)

    input_data = torch.tensor([10.0, 20.0, 30.0], device="cuda", dtype=torch.float32)

    sentinel = -1.0
    output_full = torch.full((full_size,), sentinel, device="cuda", dtype=torch.float32)
    output_sliced = slicer(output_full)

    module.copy_difftensorview(input_data, output_sliced)
    torch.cuda.synchronize()

    expected_full = torch.full_like(output_full, sentinel)
    slicer(expected_full)[:] = input_data
    assert torch.allclose(
        output_full, expected_full
    ), f"Expected {expected_full}, got {output_full}"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("source_size,slicer", DIFFTV_SLICE_CASES)
def test_difftensorview_diff_square_sliced(device_type: DeviceType, source_size: int, slicer: Any):
    """Test backward pass of diff_square with sliced DiffTensorView inputs."""
    module = load_module_torch(device_type)

    x_full = torch.arange(1.0, source_size + 1.0, device="cuda", dtype=torch.float32)
    x = slicer(x_full)
    x_grad = torch.zeros(3, device="cuda", dtype=torch.float32)
    output = torch.zeros(3, device="cuda", dtype=torch.float32)
    output_grad = torch.ones(3, device="cuda", dtype=torch.float32)

    module.diff_square.bwds(diff_pair(x, x_grad), diff_pair(output, output_grad))
    torch.cuda.synchronize()

    expected_grad = 2.0 * x
    assert torch.allclose(
        x_grad, expected_grad, atol=1e-5
    ), f"Expected grad {expected_grad}, got {x_grad}"


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
