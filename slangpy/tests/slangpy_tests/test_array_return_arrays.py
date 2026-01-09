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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
