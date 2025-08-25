# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import pytest

import slangpy as spy
from slangpy import DeviceType
from slangpy.testing import helpers


NUMPY_MODULE = r"""
import "slangpy";

float add_floats(float a, float b) {
    return a + b;
}

float3 add_float3s(float3 a, float3 b) {
    return a + b;
}


matrix<float, R, C> matFunc<int R, int C>(){
    return matrix<float, R, C>(1);
}

struct Wrapper
{
    float[16] data;
}

Wrapper flattenMatrix<int R, int C>(matrix<float, R, C> mat){
    Wrapper res;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            res.data[i * C + j] = mat[i][j];
        }
    }
    return res;
}

matrix<float, R, C> matFunc1<int R, int C>(matrix<float, R, C> input){
    return input + 2.0f;
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

    res = module.add_floats(a, b, _result="numpy")
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


# test that we handle the matrix alignment correctly when reading the matrix from the output buffer
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_return_numpy_matrix(device_type: DeviceType):

    module = load_test_module(device_type)

    for R in range(2, 5):
        for C in range(2, 5):
            funName = f"matFunc<{R}, {C}>"
            func = module.find_function(funName)
            assert func is not None
            res = func().to_numpy()
            assert res is not None
            assert res.shape == (R, C)
            assert np.allclose(res, np.ones((R, C)))


# test that we handle the matrix alignment correctly when writing the matrix to the input buffer
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_setup_numpy_matrix(device_type: DeviceType):

    module = load_test_module(device_type)
    for R in range(2, 5):
        for C in range(2, 5):
            funName = f"flattenMatrix<{R}, {C}>"
            func = module.find_function(funName)
            assert func is not None
            matType = getattr(spy, f"float{R}x{C}")
            res = func(matType(np.ones((R, C))))

            assert res is not None
            assert np.allclose(res["data"][0 : R * C], np.ones(R * C))


# test that we can use "numpy" as the result type for a function that returns a matrix
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_numpy_matrix_as_result(device_type: DeviceType):

    module = load_test_module(device_type)

    for R in range(2, 5):
        for C in range(2, 5):
            funName = f"matFunc1<{R}, {C}>"
            func = module.find_function(funName)
            assert func is not None
            matType = getattr(spy, f"float{R}x{C}")
            N = R * C
            res = func(matType(np.arange(1, N + 1).reshape(R, C)), _result="numpy")

            assert res is not None
            assert res.shape == (R, C)
            assert np.allclose(res, (np.arange(1, N + 1) + 2.0).reshape(R, C))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
