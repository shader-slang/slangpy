# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

import slangpy as spy
from slangpy import DeviceType
from slangpy.testing import helpers
from slangpy.types import Tensor


MODULE = r"""
import "slangpy";

half[NumChannels] tensor_test_channels<let NumChannels : int>(half[NumChannels] data)
{
    [ForceUnroll]
    for (int i = 0; i < NumChannels; ++i)
    {
        data[i] = 2.h * data[i];
    }
    return data;
}
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_return_array_value_from_python_list(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    # Scalar call (no vectorization): return type is half[8] as a Python value.
    func = module.find_function("tensor_test_channels<8>")
    assert func is not None
    out = func([1.0] * 8)
    assert isinstance(out, list)
    assert len(out) == 8
    assert np.allclose(np.array(out, dtype=np.float32), np.array([2.0] * 8, dtype=np.float32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_return_array_value_generic_resolution(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    # Verify SlangPy can resolve tensor_test_channels<NumChannels> from the Python argument length.
    func = module.find_function("tensor_test_channels")
    assert func is not None
    out = func([1.0] * 8)
    assert isinstance(out, list)
    assert len(out) == 8
    assert np.allclose(np.array(out, dtype=np.float32), np.array([2.0] * 8, dtype=np.float32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_return_array_value_from_tensor_scalar(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    # Scalar call: tensor has shape (8,), which matches the element shape of half[8],
    # so the call dimensionality is 0 and the return is a Python array value.
    t = Tensor.from_numpy(device, np.ones((8,), dtype=np.float16))
    func = module.find_function("tensor_test_channels<8>")
    assert func is not None
    out = func(t)
    assert isinstance(out, list)
    assert len(out) == 8
    assert np.allclose(np.array(out, dtype=np.float32), np.array([2.0] * 8, dtype=np.float32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_coercion_trailing_dim_into_array_element(device_type: DeviceType):
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device,
        "tensor_test_channels<8>",
        MODULE,
    ).return_type(Tensor)

    x_np = (np.random.rand(4, 4, 8).astype(np.float16) - np.float16(0.5)) * np.float16(2.0)
    x = Tensor.from_numpy(device, x_np)

    y = func(x)
    y_np = y.to_numpy()
    assert y_np.shape == (4, 4, 8)
    assert np.allclose(y_np.astype(np.float32), (x_np.astype(np.float32) * 2.0), atol=1e-3)


VECTOR_ARRAY_MODULE = r"""
import "slangpy";

[Differentiable]
float2[6] return_vector_array(int coord) {
    float2 outputs[6];
    for (int i = 0; i < 6; ++i) {
        outputs[i] = float2(coord, coord + i);
    }
    return outputs;
}

[Differentiable]
float2[4] scale_vector_array(float2[4] input, float scale) {
    float2 outputs[4];
    for (int i = 0; i < 4; ++i) {
        outputs[i] = input[i] * scale;
    }
    return outputs;
}
"""


def _expected_vector_array(n: int) -> np.ndarray:
    expected = np.zeros((n, 6, 2), dtype=np.float32)
    for c in range(n):
        for i in range(6):
            expected[c, i] = [c, c + i]
    return expected


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_return_array_of_vectors_tensor(device_type: DeviceType):
    """Regression test for https://github.com/shader-slang/slangpy/issues/638"""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "return_vector_array", VECTOR_ARRAY_MODULE)
    result = func(coord=spy.grid((13,)))
    result_np = result.to_numpy()
    assert result_np.shape == (13, 6, 2)
    np.testing.assert_allclose(result_np, _expected_vector_array(13))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_return_array_of_vectors_numpy(device_type: DeviceType):
    """Regression test for https://github.com/shader-slang/slangpy/issues/638"""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "return_vector_array", VECTOR_ARRAY_MODULE)
    result = func(coord=spy.grid((13,)), _result="numpy")
    assert isinstance(result, np.ndarray)
    assert result.shape == (13, 6, 2)
    np.testing.assert_allclose(result, _expected_vector_array(13))


@pytest.mark.parametrize("device_type", [DeviceType.cuda])
def test_return_array_of_vectors_torch(device_type: DeviceType):
    """Regression test for https://github.com/shader-slang/slangpy/issues/638"""
    torch = pytest.importorskip("torch")
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "return_vector_array", VECTOR_ARRAY_MODULE)
    coord = torch.arange(13, dtype=torch.int32, device="cuda")
    result = func(coord=coord)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([13, 6, 2])
    np.testing.assert_allclose(result.cpu().numpy(), _expected_vector_array(13))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_input_array_of_vectors_scalar_call(device_type: DeviceType):
    """Pass array-of-vectors as Python list input (scalar call, no vectorization)."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "scale_vector_array", VECTOR_ARRAY_MODULE)
    input_val = [[1, 2], [3, 4], [5, 6], [7, 8]]
    result = func(input=input_val, scale=2.0)
    assert isinstance(result, list)
    expected = [[2, 4], [6, 8], [10, 12], [14, 16]]
    for i in range(4):
        assert list(result[i]) == expected[i]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_result_preallocated_array_of_vectors(device_type: DeviceType):
    """Write array-of-vector result into a pre-allocated Tensor."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "return_vector_array", VECTOR_ARRAY_MODULE)
    result_tensor = Tensor.empty(device, shape=(13, 6, 2), dtype=float)
    func(coord=spy.grid((13,)), _result=result_tensor)
    result_np = result_tensor.to_numpy()
    assert result_np.shape == (13, 6, 2)
    np.testing.assert_allclose(result_np, _expected_vector_array(13))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_input_array_of_vectors_numpy(device_type: DeviceType):
    """Pass array-of-vectors as numpy input and return numpy result."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "scale_vector_array", VECTOR_ARRAY_MODULE)
    input_np = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    result = func(input=input_np, scale=2.0, _result="numpy")
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 2)
    np.testing.assert_allclose(result, input_np * 2.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_diff_array_of_vectors_bwds(device_type: DeviceType):
    """Backward pass through a differentiable function returning array-of-vectors."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device, "scale_vector_array", VECTOR_ARRAY_MODULE
    ).return_type(Tensor)

    # Input tensor (batch=8, array=4, vector=2) with gradients
    input_np = np.ones((8, 4, 2), dtype=np.float32) * 2.0
    input_t = Tensor.from_numpy(device, input_np).with_grads()

    # Forward: output = input * scale
    result = func(input=input_t, scale=3.0)
    result_np = result.to_numpy()
    assert result_np.shape == (8, 4, 2)
    np.testing.assert_allclose(result_np, input_np * 3.0)

    # Backward: d_result = all ones
    result_w_grads = result.with_grads()
    result_w_grads.grad.storage.copy_from_numpy(np.ones((8, 4, 2), dtype=np.float32))
    func.bwds(input=input_t, scale=3.0, _result=result_w_grads)

    # d(output)/d(input) = scale = 3.0
    input_grad = input_t.grad.to_numpy()
    np.testing.assert_allclose(input_grad, np.full((8, 4, 2), 3.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
