# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np
from slangpy import DeviceType, BufferUsage, Tensor
from slangpy.types import NDBuffer
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_simple_array(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "sum",
        r"""
int sum(int[4] data) {
    int sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += data[i];
    }
    return sum;
}
""",
    )

    # just verify it can be called with no exceptions
    result = function([1, 2, 3, 4])
    assert result == 10


@pytest.mark.skip(
    reason="Skipped due to removal of the assert. See https://github.com/shader-slang/slangpy/issues/255"
)
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_untyped_struct_array(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "sum",
        r"""
struct Val {
    int x;
}

int sum(Val[4] vals) {
    int sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += vals[i].x;
    }
    return sum;
}
""",
    )

    # just verify it can be called with no exceptions
    with pytest.raises(Exception, match=".*Element type must be fully defined.*"):
        result = function([{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}])
        assert result == 10


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_typed_struct_array(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "sum",
        r"""
struct Val {
    int x;
}

int sum(Val[4] vals) {
    int sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += vals[i].x;
    }
    return sum;
}
""",
    )

    # just verify it can be called with no exceptions
    result = function(
        [
            {"x": 1, "_type": "Val"},
            {"x": 2, "_type": "Val"},
            {"x": 3, "_type": "Val"},
            {"x": 4, "_type": "Val"},
        ]
    )
    assert result == 10


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vectorize_array(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "inc",
        r"""
int inc(int val) {
    return val+1;
}
""",
    )

    # just verify it can be called with no exceptions
    results = function([1, 2, 3, 4], _result="numpy")
    assert isinstance(results, np.ndarray)
    assert results.shape == (4,)
    assert results.dtype == np.int32
    assert np.array_equal(results, np.array([2, 3, 4, 5], dtype=np.int32))


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

    # just verify it can be called with no exceptions
    results = function(
        [
            {"x": 1, "_type": "Val"},
            {"x": 2, "_type": "Val"},
            {"x": 3, "_type": "Val"},
            {"x": 4, "_type": "Val"},
        ],
        _result="numpy",
    )
    assert isinstance(results, np.ndarray)
    assert results.shape == (4,)
    assert results.dtype == np.int32
    assert np.array_equal(results, np.array([2, 3, 4, 5], dtype=np.int32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vectorize_struct_with_resource_array(device_type: DeviceType):
    if device_type == DeviceType.metal:
        # https://github.com/shader-slang/slang/issues/7606
        pytest.skip("Crash in the slang compiler")

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "inc",
        r"""
struct Val {
    StructuredBuffer<int> x;
}
int inc(Val val) {
    return val.x[0]+1;
}
""",
    )

    vals = []
    for i in range(4):
        buffer = device.create_buffer(
            element_count=1,
            struct_size=4,
            data=np.array([i + 1], dtype=np.int32),
            usage=BufferUsage.shader_resource,
        )
        vals.append({"x": buffer, "_type": "Val"})

    # just verify it can be called with no exceptions
    results = function(vals, _result="numpy")
    assert isinstance(results, np.ndarray)
    assert results.shape == (4,)
    assert results.dtype == np.int32
    assert np.array_equal(results, np.array([2, 3, 4, 5], dtype=np.int32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vectorize_struct_with_ndbuffer_array(device_type: DeviceType):
    if device_type == DeviceType.metal:
        # https://github.com/shader-slang/slang/issues/7606
        pytest.skip("Crash in the slang compiler")

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "inc",
        r"""
struct Val {
    NDBuffer<int,1> x;
}
int inc(Val val) {
    return val.x[0]+1;
}
""",
    )

    vals = []
    for i in range(4):
        buffer = NDBuffer.zeros(
            device,
            dtype=function.module.int,
            shape=(1,),
            usage=BufferUsage.shader_resource | BufferUsage.unordered_access,
        )
        buffer.copy_from_numpy(np.array([i + 1], dtype=np.int32))
        vals.append({"x": buffer, "_type": "Val"})

    # just verify it can be called with no exceptions
    results = function(vals, _result="numpy")
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

    # just verify it can be called with no exceptions
    results = function(vals, _result="numpy")
    assert isinstance(results, np.ndarray)
    assert results.shape == (4,)
    assert results.dtype == np.float32
    assert np.array_equal(results, np.array([2, 3, 4, 5], dtype=np.float32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_2d_mapped_vectorize_struct_with_tensor_array(device_type: DeviceType):
    if device_type == DeviceType.metal:
        # https://github.com/shader-slang/slang/issues/7606
        pytest.skip("Crash in the slang compiler")

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add",
        r"""
struct Val {
    Tensor<float,1> x;
}
float add(Val val, float val2) {
    return val.x[0]+val2;
}
""",
    )

    # Build a set of 4 structures, each containing a tensor with a single value
    vals = []
    for i in range(4):
        buffer = Tensor.zeros(device, dtype=function.module.float, shape=(1,))
        buffer.copy_from_numpy(np.array([i + 1], dtype=np.float32))
        vals.append({"x": buffer, "_type": "Val"})

    # Map the tensor list to the first dimension, and a list of floats to the second dimensions
    results = function.map((0,), (1,))(vals, [5.0, 10.0, 15.0, 20.0], _result="numpy")

    # Should end up with 4x4 matrix, where each row corresponds to a tensor and each column corresponds to the float value added
    assert isinstance(results, np.ndarray)
    assert results.shape == (4, 4)
    assert results.dtype == np.float32
    assert np.array_equal(
        results,
        np.array(
            [[6, 11, 16, 21], [7, 12, 17, 22], [8, 13, 18, 23], [9, 14, 19, 24]], dtype=np.float32
        ),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
