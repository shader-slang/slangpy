# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

import slangpy.tests.helpers as helpers
from slangpy.backend import DeviceType

NUMPY_MODULE = r"""
import "slangpy";

float add_floats(float a, float b) {
    return a + b;
}

float3 add_float3s(float3 a, float3 b) {
    return a + b;
}
"""


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return helpers.create_module(device, NUMPY_MODULE)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pass_numpy_floats(device_type: DeviceType):

    module = load_test_module(device_type)

    a = np.random.rand(2, 2).astype(np.float32)
    b = np.random.rand(2, 2).astype(np.float32)

    res_buffer = module.add_floats(a, b)
    res = res_buffer.to_numpy().view(np.float32).reshape(*res_buffer.shape)

    res_expected = a + b

    assert np.allclose(res, res_expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_return_numpy_floats(device_type: DeviceType):

    module = load_test_module(device_type)

    a = np.random.rand(2, 2).astype(np.float32)
    b = np.random.rand(2, 2).astype(np.float32)
    res = np.zeros_like(a)

    module.add_floats(a, b, _result=res)
    res_expected = a + b

    assert np.allclose(res, res_expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_return_numpy_floats_auto(device_type: DeviceType):

    module = load_test_module(device_type)

    a = np.random.rand(2, 2).astype(np.float32)
    b = np.random.rand(2, 2).astype(np.float32)

    res = module.add_floats(a, b, _result='numpy')
    res_expected = a + b

    assert np.allclose(res, res_expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_returntype_numpy_floats(device_type: DeviceType):

    module = load_test_module(device_type)

    a = np.random.rand(2, 2).astype(np.float32)
    b = np.random.rand(2, 2).astype(np.float32)

    res = module.add_floats.return_type(np.ndarray)(a, b)
    res_expected = a + b

    assert np.allclose(res, res_expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pass_numpy_float3s(device_type: DeviceType):

    module = load_test_module(device_type)

    a = np.random.rand(2, 2, 3).astype(np.float32)
    b = np.random.rand(2, 2, 3).astype(np.float32)

    res = module.add_float3s.return_type(np.ndarray)(a, b)
    res_expected = a + b

    assert np.allclose(res, res_expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_fail_numpy_float3s(device_type: DeviceType):

    module = load_test_module(device_type)

    a = np.random.rand(2, 2, 2).astype(np.float32)
    b = np.random.rand(2, 2, 2).astype(np.float32)

    with pytest.raises(RuntimeError, match="does not match the expected shape"):
        module.add_float3s.return_type(np.ndarray)(a, b)

# Ensure numpy array kernels are cached correctly


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_cache(device_type: DeviceType):

    module = load_test_module(device_type)

    a = np.random.rand(3, 2, 2).astype(np.float32)
    b = np.random.rand(3, 2, 2).astype(np.float32)

    res = module.add_floats.return_type(np.ndarray)(a, b)
    res_expected = a + b

    assert np.allclose(res, res_expected)

    a = np.random.rand(2, 2).astype(np.float32)
    b = np.random.rand(2, 2).astype(np.float32)

    res = module.add_floats.return_type(np.ndarray)(a, b)
    res_expected = a + b

    assert np.allclose(res, res_expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
