# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pytest
from slangpy import DeviceType, Device
from slangpy.types import Tensor
from . import helpers
import numpy as np
from typing import Any
import os


def get_test_tensors(device: Device, din: int = 5, dout: int = 8, N: int = 4):
    np.random.seed(0)

    np_weights = np.random.randn(din, dout).astype(np.float32)
    np_biases = np.random.randn(din).astype(np.float32)
    np_x = np.random.randn(dout).astype(np.float32)
    np_result = np.tile(np_weights.dot(np_x) + np_biases, (N, 1))

    biases = Tensor.from_numpy(device, np_biases).broadcast_to((N, din))
    x = Tensor.from_numpy(device, np_x).broadcast_to((N, dout))
    weights = Tensor.from_numpy(
        device,
        np_weights,
    ).broadcast_to((N, din, dout))

    return weights, biases, x, np_result


def get_func(device: Device, name: str):
    path = os.path.split(__file__)[0] + "/test_tensor.slang"
    return helpers.create_function_from_module(device, name, open(path, "r").read())


def compare_tensors(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]):
    assert a.shape == b.shape, f"Tensor shape {a.shape} does not match expected shape {b.shape}"
    err = np.max(np.abs(a - b))
    assert err < 1e-4, f"Tensor deviates by {err} from reference"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_arguments(device_type: DeviceType):
    device = helpers.get_device(device_type)

    func = get_func(device, "matrix_vector_direct").return_type(Tensor)
    weights, biases, x, np_result = get_test_tensors(device)
    y = func(weights, biases, x)
    compare_tensors(y.to_numpy(), np_result)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_parameters(device_type: DeviceType):
    device = helpers.get_device(device_type)

    func = get_func(device, "matrix_vector_tensorized").return_type(Tensor)
    weights, biases, x, np_result = get_test_tensors(device)
    y = func(weights, biases, x)
    compare_tensors(y.to_numpy(), np_result)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_matrix_parameters(device_type: DeviceType):
    device = helpers.get_device(device_type)

    func = get_func(device, "matrix_vector_matrices").return_type(Tensor)
    weights, biases, x, np_result = get_test_tensors(device, din=3, dout=4)
    y = func(weights, biases, x)
    compare_tensors(y.to_numpy(), np_result)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_interface_parameters(device_type: DeviceType):
    device = helpers.get_device(device_type)

    func = get_func(device, "matrix_vector_interfaces").return_type(Tensor)
    weights, biases, x, np_result = get_test_tensors(device)
    y = func(weights, biases, x)
    compare_tensors(y.to_numpy(), np_result)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_generic_call(device_type: DeviceType):
    device = helpers.get_device(device_type)

    func = get_func(device, "matrix_vector_generic<8, 5>").return_type(Tensor)
    weights, biases, x, np_result = get_test_tensors(device)
    y = func(weights, biases, x)
    compare_tensors(y.to_numpy(), np_result)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_view(device_type: DeviceType):
    device = helpers.get_device(device_type)

    # Create a 1D tensor with 100 elements
    np.random.seed(42)
    original_data = np.random.randn(100).astype(np.float32)
    original_tensor = Tensor.from_numpy(device, original_data)

    # Test 1: Basic view with different shape (100,) -> (10, 10)
    view_10x10 = original_tensor.view((10, 10))
    assert view_10x10.shape.as_tuple() == (10, 10)
    assert view_10x10.offset == 0

    # Verify the data is the same when flattened
    view_data = view_10x10.to_numpy().flatten()
    compare_tensors(view_data, original_data)

    # Test 2: View with offset - view a subset of the data starting at element 20
    offset = 20
    subset_shape = (8, 10)  # 80 elements starting from offset 20
    view_with_offset = original_tensor.view(subset_shape, offset=offset)
    assert view_with_offset.shape.as_tuple() == subset_shape
    assert view_with_offset.offset == offset

    # Verify the offset data matches
    view_offset_data = view_with_offset.to_numpy().flatten()
    expected_offset_data = original_data[offset : offset + 80]
    compare_tensors(view_offset_data, expected_offset_data)

    # Test 3: View with custom strides (transposed view)
    matrix_tensor = Tensor.from_numpy(device, np.arange(24, dtype=np.float32))
    transposed_view = matrix_tensor.view((4, 6), strides=(1, 4))  # Column-major access
    assert transposed_view.shape.as_tuple() == (4, 6)
    assert transposed_view.strides.as_tuple() == (1, 4)

    # Test 4: Multiple chained views
    chained_view = view_10x10.view((5, 20))
    assert chained_view.shape.as_tuple() == (5, 20)
    chained_data = chained_view.to_numpy().flatten()
    compare_tensors(chained_data, original_data)

    # Test 5: View back to original shape
    back_to_original = view_10x10.view((100,))
    assert back_to_original.shape.as_tuple() == (100,)
    back_data = back_to_original.to_numpy()
    compare_tensors(back_data, original_data)

    # Test 6: Different data starting from different positions
    # Create a larger tensor for more comprehensive offset testing
    large_data = np.arange(200, dtype=np.float32)
    large_tensor = Tensor.from_numpy(device, large_data)
    # View middle section as 2D
    middle_view = large_tensor.view((10, 8), offset=50)
    assert middle_view.shape.as_tuple() == (10, 8)
    assert middle_view.offset == 50

    middle_data = middle_view.to_numpy().flatten()
    expected_middle = large_data[50 : 50 + 80]
    compare_tensors(middle_data, expected_middle)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
