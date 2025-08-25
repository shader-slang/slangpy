# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

from slangpy import DeviceType, pack, Tensor
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_simple_int_call(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "copy",
        r"""
int copy(int val) {
    return val;
}
""",
    )

    fv = pack(function.module, 42)

    # just verify it can be called with no exceptions
    result = function(fv)
    assert result == 42


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_simple_struct_call(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "copy",
        r"""
struct Val
{
    int value;
}
int copy(Val val) {
    return val.value;
}
""",
    )

    fv = pack(function.module, {"_type": "Val", "value": 42})

    # just verify it can be called with no exceptions
    result = function(fv)
    assert result == 42


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vectorize_struct_array(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "inc",
        r"""
struct Val {
    int x;
}
int inc(Val val) {
    return val.x+1;
}
""",
    )

    fv = pack(
        function.module,
        [
            {"x": 1, "_type": "Val"},
            {"x": 2, "_type": "Val"},
            {"x": 3, "_type": "Val"},
            {"x": 4, "_type": "Val"},
        ],
    )

    # just verify it can be called with no exceptions
    results = function(fv, _result="numpy")
    assert isinstance(results, np.ndarray)
    assert results.shape == (4,)
    assert results.dtype == np.int32
    assert np.array_equal(results, np.array([2, 3, 4, 5], dtype=np.int32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vectorize_struct_with_tensor_array(device_type: DeviceType):
    if device_type == DeviceType.metal:
        # https://github.com/shader-slang/slang/issues/7606
        pytest.skip("Crash in the slang compiler")

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "inc",
        r"""
struct Val {
    Tensor<float,1> x;
}
float inc(Val val) {
    return val.x[0]+1;
}
""",
    )

    vals = []
    for i in range(4):
        buffer = Tensor.zeros(device, dtype=function.module.float, shape=(1,))
        buffer.copy_from_numpy(np.array([i + 1], dtype=np.float32))
        vals.append({"x": buffer, "_type": "Val"})
    fv = pack(function.module, vals)

    # just verify it can be called with no exceptions
    results = function(fv, _result="numpy")
    assert isinstance(results, np.ndarray)
    assert results.shape == (4,)
    assert results.dtype == np.float32
    assert np.array_equal(results, np.array([2, 3, 4, 5], dtype=np.float32))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
