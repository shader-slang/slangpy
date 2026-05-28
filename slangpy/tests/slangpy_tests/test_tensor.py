# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

from slangpy import DeviceType, Device, grid, Module
from slangpy.types import Tensor
from slangpy.testing import helpers

from typing import Any


def numpy_sample_scalar(data: np.ndarray[Any, Any], u: float, v: float) -> float:
    """Bilinear interpolation matching Slang _get_sample_indices / _bilerp logic.

    The Slang sample() does:
        int2 shape = int2(shape[1], shape[0])   ->  (cols, rows)
        texel = int2(uv * shape)                ->  (col, row)
        interp = frac(uv * shape)
    Then load(int2(col, row)) indexes data[row, col] because the vector _idx
    overload reverses components (int2.x -> last stride, int2.y -> first stride).
    """
    rows = data.shape[0]
    cols = data.shape[1]

    tx = np.float32(u * cols)
    ty = np.float32(v * rows)

    ix = int(np.floor(tx))
    iy = int(np.floor(ty))
    fx = np.float32(tx - np.floor(tx))
    fy = np.float32(ty - np.floor(ty))

    col0 = ix
    row0 = iy
    col1 = min(ix + 1, cols - 1)
    row1 = min(iy + 1, rows - 1)

    # load(int2(a, b)) -> data[b, a] due to vector _idx reversal
    c00 = np.float32(data[row0, col0])
    c10 = np.float32(data[row0, col1])
    c01 = np.float32(data[row1, col0])
    c11 = np.float32(data[row1, col1])

    # lerp(lerp(c00, c10, fx), lerp(c01, c11, fx), fy)
    top = np.float32(c00 * (1 - fx) + c10 * fx)
    bot = np.float32(c01 * (1 - fx) + c11 * fx)
    return float(np.float32(top * (1 - fy) + bot * fy))


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
    return Module.load_from_file(device, "test_tensor.slang").require_function(name)


def compare_tensors(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]):
    assert a.shape == b.shape, f"Tensor shape {a.shape} does not match expected shape {b.shape}"
    err = np.max(np.abs(a - b))
    assert err < 1e-4, f"Tensor deviates by {err} from reference"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "func_name",
    [
        "copy_itensor_each_index",
        "copy_itensor_array_index",
        "copy_itensor_vector_index",
        "copy_itensor_each_loadstore",
        "copy_itensor_array_loadstore",
        "copy_itensor_vector_loadstore",
        "copy_idifftensor_each_index",
        "copy_idifftensor_array_index",
        "copy_idifftensor_vector_index",
        "copy_idifftensor_each_loadstore",
        "copy_idifftensor_array_loadstore",
        "copy_idifftensor_vector_loadstore",
        "copy_tensor_each_index",
        "copy_tensor_array_index",
        "copy_tensor_vector_index",
        "copy_tensor_each_loadstore",
        "copy_tensor_array_loadstore",
        "copy_tensor_vector_loadstore",
        "copy_difftensor_each_index",
        "copy_difftensor_array_index",
        "copy_difftensor_vector_index",
        "copy_difftensor_each_loadstore",
        "copy_difftensor_array_loadstore",
        "copy_difftensor_vector_loadstore",
        "copy_primaltensor_each_index",
        "copy_primaltensor_array_index",
        "copy_primaltensor_vector_index",
        "copy_primaltensor_each_loadstore",
        "copy_primaltensor_array_loadstore",
        "copy_primaltensor_vector_loadstore",
    ],
)
def test_simple_copy(device_type: DeviceType, func_name: str):
    device = helpers.get_device(device_type)

    func = get_func(device, func_name)

    din = np.random.randn(1000, 3).astype(np.float32)
    tin = Tensor.from_numpy(device, din)
    tout = Tensor.empty_like(tin)

    if "difftensor" in func_name:
        tin = tin.with_grads(None, Tensor.empty_like(tin))
        tout = tout.with_grads(Tensor.empty_like(tout), None)

    func(grid(tin.shape), tin, tout)

    dout = tout.to_numpy()
    assert np.array_equal(din, dout), "Copied tensor does not match original"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_simple_copy_by_element(device_type: DeviceType):
    device = helpers.get_device(device_type)

    func = get_func(device, "copy_tensor_by_element")

    din = np.random.randn(1000, 3).astype(np.float32)
    tin = Tensor.from_numpy(device, din)
    tout = Tensor.empty_like(tin)

    func(tin, tout)

    dout = tout.to_numpy()
    assert np.array_equal(din, dout), "Copied tensor does not match original"


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
def test_existential_type_bug(device_type: DeviceType):
    """Tests for a slang issue that causes compilation error when using Tensor accessors"""

    device = helpers.get_device(device_type)

    program = device.load_program("test_existential_bug.slang", ["build_importance_map"])
    kernel = device.create_compute_kernel(program=program)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_sample_scalar(device_type: DeviceType):
    """Test bilinear sampling on a Tensor<float, 2>."""
    device = helpers.get_device(device_type)
    func = get_func(device, "sample_tensor_scalar")

    np.random.seed(42)
    data = np.random.rand(8, 6).astype(np.float32)
    tensor = Tensor.from_numpy(device, data)

    test_uvs = [(0.25, 0.35), (0.5, 0.5), (0.1, 0.9), (0.0, 0.0), (0.75, 0.125)]
    for u, v in test_uvs:
        result = func(tensor, u, v)
        expected = numpy_sample_scalar(data, u, v)
        assert (
            abs(float(result) - float(expected)) < 1e-5
        ), f"Sample mismatch at uv=({u},{v}): got {result}, expected {expected}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_rwtensor_sample_scalar(device_type: DeviceType):
    """Test bilinear sampling on an RWTensor<float, 2>."""
    device = helpers.get_device(device_type)
    func = get_func(device, "sample_rwtensor_scalar")

    np.random.seed(42)
    data = np.random.rand(8, 6).astype(np.float32)
    tensor = Tensor.from_numpy(device, data)

    u, v = 0.3, 0.4
    result = func(tensor, u, v)
    expected = numpy_sample_scalar(data, u, v)
    assert abs(float(result) - float(expected)) < 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.skip("TODO: DiffTensor sample")
def test_difftensor_sample_scalar(device_type: DeviceType):
    """Test bilinear sampling on a DiffTensor<float, 2>."""
    device = helpers.get_device(device_type)
    func = get_func(device, "sample_difftensor_scalar")

    np.random.seed(42)
    data = np.random.rand(8, 6).astype(np.float32)
    tensor = Tensor.from_numpy(device, data)
    tensor = tensor.with_grads(None, Tensor.empty_like(tensor))

    u, v = 0.3, 0.4
    result = func(tensor, u, v)
    expected = numpy_sample_scalar(data, u, v)
    assert abs(float(result) - float(expected)) < 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.skip("TODO: DiffTensor sample")
def test_primaltensor_sample_scalar(device_type: DeviceType):
    """Test bilinear sampling on a PrimalTensor<float, 2>."""
    device = helpers.get_device(device_type)
    func = get_func(device, "sample_primaltensor_scalar")

    np.random.seed(42)
    data = np.random.rand(8, 6).astype(np.float32)
    tensor = Tensor.from_numpy(device, data)

    u, v = 0.3, 0.4
    result = func(tensor, u, v)
    expected = numpy_sample_scalar(data, u, v)
    assert abs(float(result) - float(expected)) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
