# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys
import numpy as np
from pathlib import Path

from slangpy import DeviceType, Tensor
from slangpy.core.module import Module
from slangpy.experimental.gridarg import grid
from slangpy.testing import helpers

# Skip all tests in this file if running on MacOS
if sys.platform == "darwin":
    pytest.skip("TensorView requires CUDA, not available on macOS", allow_module_level=True)

# TensorView only works with CUDA device type
DEVICE_TYPES = [DeviceType.cuda] if DeviceType.cuda in helpers.DEFAULT_DEVICE_TYPES else []
if not DEVICE_TYPES:
    pytest.skip("TensorView requires CUDA device type", allow_module_level=True)


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def load_module(device):
    return Module.load_from_file(
        device,
        str(Path(__file__).parent / "test_tensorview.slang"),
    )


# ============================================================================
# Tests with torch.Tensor
# ============================================================================
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_tensorview_copy_torch(device_type: DeviceType):
    """Test copy_tensorview with torch.Tensor arguments."""
    device = helpers.get_torch_device(device_type)
    module = load_module(device)

    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=torch.float32)
    output_tensor = torch.zeros(5, device="cuda", dtype=torch.float32)

    module.copy_tensorview(input_tensor, output_tensor)
    torch.cuda.synchronize()

    assert torch.allclose(
        input_tensor, output_tensor
    ), f"Expected {input_tensor}, got {output_tensor}"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_tensorview_add_torch(device_type: DeviceType):
    """Test add_tensorview with torch.Tensor arguments."""
    device = helpers.get_torch_device(device_type)
    module = load_module(device)

    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=torch.float32)
    b = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], device="cuda", dtype=torch.float32)
    output_tensor = torch.zeros(5, device="cuda", dtype=torch.float32)

    module.add_tensorview(a, b, output_tensor)
    torch.cuda.synchronize()

    expected = a + b
    assert torch.allclose(expected, output_tensor), f"Expected {expected}, got {output_tensor}"


# ============================================================================
# Tests with slangpy Tensor
# ============================================================================
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_tensorview_copy_slangpy_tensor(device_type: DeviceType):
    """Test copy_tensorview with slangpy Tensor arguments."""
    device = helpers.get_device(type=device_type)
    module = load_module(device)

    input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_tensor = Tensor.from_numpy(device, input_data)
    output_tensor = Tensor.zeros(device, dtype="float", shape=(5,))

    module.copy_tensorview(input_tensor, output_tensor)

    output_data = output_tensor.to_numpy()
    assert np.array_equal(input_data, output_data), f"Expected {input_data}, got {output_data}"


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_tensorview_add_slangpy_tensor(device_type: DeviceType):
    """Test add_tensorview with slangpy Tensor arguments."""
    device = helpers.get_device(type=device_type)
    module = load_module(device)

    a_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    b_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    a = Tensor.from_numpy(device, a_data)
    b = Tensor.from_numpy(device, b_data)
    output = Tensor.zeros(device, dtype="float", shape=(5,))

    module.add_tensorview(a, b, output)

    output_data = output.to_numpy()
    expected = a_data + b_data
    assert np.array_equal(expected, output_data), f"Expected {expected}, got {output_data}"


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_tensorview_grid_dispatch(device_type: DeviceType):
    """Test that TensorView works with grid dispatch.

    grid((5,)) explicitly dispatches 5 threads. Each thread receives its
    grid coordinate as 'coord' and writes 1 at that position in the markers
    TensorView.
    """
    device = helpers.get_device(type=device_type)
    module = load_module(device)

    markers = Tensor.zeros(device, dtype="int", shape=(5,))

    module.mark_thread(grid((5,)), markers)

    result = markers.to_numpy()
    expected = np.ones(5, dtype=np.int32)
    assert np.array_equal(result, expected), f"Expected all markers to be 1, got {result}"


# ============================================================================
# Tests for TensorView<float2> / <float4> (vector element types)
# ============================================================================
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_tensorview_float2_torch(device_type: DeviceType):
    """Test TensorView<float2> with a float32 torch tensor (2 floats per element)."""
    device = helpers.get_torch_device(device_type)
    module = load_module(device)

    input_tensor = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="cuda", dtype=torch.float32
    )
    output_tensor = torch.zeros(3, 2, device="cuda", dtype=torch.float32)

    module.copy_tensorview_float2(input_tensor, output_tensor, _thread_count=1)
    torch.cuda.synchronize()

    assert torch.allclose(
        input_tensor, output_tensor
    ), f"Expected {input_tensor}, got {output_tensor}"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_tensorview_float4_torch(device_type: DeviceType):
    """Test TensorView<float4> with a float32 torch tensor (4 floats per element)."""
    device = helpers.get_torch_device(device_type)
    module = load_module(device)

    input_tensor = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device="cuda", dtype=torch.float32
    )
    output_tensor = torch.zeros(2, 4, device="cuda", dtype=torch.float32)

    module.copy_tensorview_float4(input_tensor, output_tensor, _thread_count=1)
    torch.cuda.synchronize()

    assert torch.allclose(
        input_tensor, output_tensor
    ), f"Expected {input_tensor}, got {output_tensor}"


# ============================================================================
# Tests for _thread_count (CUDAKernel dispatch)
# ============================================================================
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_thread_count_fill_ids(device_type: DeviceType):
    """Test _thread_count with a CUDAKernel that fills thread IDs."""
    device = helpers.get_torch_device(device_type)
    module = load_module(device)

    count = 64
    output = torch.zeros(count, device="cuda", dtype=torch.int32)

    module.fill_thread_ids(count=count, output=output, _thread_count=count)
    torch.cuda.synchronize()

    expected = torch.arange(count, device="cuda", dtype=torch.int32)
    assert torch.equal(output, expected), f"Expected {expected}, got {output}"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_thread_count_append_to(device_type: DeviceType):
    """Test _thread_count goes through the append_to dispatch path."""
    device = helpers.get_torch_device(device_type)
    module = load_module(device)

    count = 64
    output = torch.zeros(count, device="cuda", dtype=torch.int32)

    command_encoder = device.create_command_encoder()
    module.fill_thread_ids.append_to(
        command_encoder, count=count, output=output, _thread_count=count
    )

    # Nothing should have executed yet
    assert torch.all(output == 0), "Output should be zero before command buffer submission"

    device.submit_command_buffer(command_encoder.finish())
    torch.cuda.synchronize()

    expected = torch.arange(count, device="cuda", dtype=torch.int32)
    assert torch.equal(output, expected), f"Expected {expected}, got {output}"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available")
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_thread_count_error_on_auto_vectorized(device_type: DeviceType):
    """_thread_count must not be passed to an auto-vectorized kernel (call_dimensionality > 0)."""
    device = helpers.get_torch_device(device_type)
    module = load_module(device)

    # mark_thread(int coord, TensorView<int>) is auto-vectorized when coord is a tensor:
    # the thread count is inferred from the shape of coord, so _thread_count is invalid.
    coords = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    markers = torch.zeros(4, device="cuda", dtype=torch.int32)

    with pytest.raises(Exception, match="_thread_count"):
        module.mark_thread(coords, markers, _thread_count=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
