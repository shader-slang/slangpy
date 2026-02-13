# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np
from slangpy import DeviceType, BufferUsage, Tensor
from slangpy.types import Tensor
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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_struct_with_scalar_array_field(device_type: DeviceType):
    """A struct whose field is a fixed-size array of scalars."""

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "sum_inner",
        r"""
struct Foo {
    int vals[4];
}

int sum_inner(Foo foo) {
    int s = 0;
    for (int i = 0; i < 4; i++) {
        s += foo.vals[i];
    }
    return s;
}
""",
    )

    result = function({"vals": [1, 2, 3, 4], "_type": "Foo"})
    assert result == 10


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_struct_with_struct_array_field(device_type: DeviceType):
    """A struct whose field is a fixed-size array of structs."""

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "sum_inner",
        r"""
struct Inner {
    int x;
}

struct Outer {
    Inner items[4];
}

int sum_inner(Outer outer) {
    int s = 0;
    for (int i = 0; i < 4; i++) {
        s += outer.items[i].x;
    }
    return s;
}
""",
    )

    result = function(
        {
            "items": [
                {"x": 10, "_type": "Inner"},
                {"x": 20, "_type": "Inner"},
                {"x": 30, "_type": "Inner"},
                {"x": 40, "_type": "Inner"},
            ],
            "_type": "Outer",
        }
    )
    assert result == 100


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_array_of_structured_buffers(device_type: DeviceType):
    """A function parameter that is an array of StructuredBuffer<T>."""

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "sum_buffers",
        r"""
int sum_buffers(StructuredBuffer<int> buffers[4]) {
    int s = 0;
    for (int i = 0; i < 4; i++) {
        s += buffers[i][0];
    }
    return s;
}
""",
    )

    buffers = []
    for i in range(4):
        buf = device.create_buffer(
            element_count=1,
            struct_size=4,
            data=np.array([(i + 1) * 10], dtype=np.int32),
            usage=BufferUsage.shader_resource,
        )
        buffers.append(buf)

    result = function(buffers)
    assert result == 100


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_nested_struct_with_array_field(device_type: DeviceType):
    """A struct containing another struct that itself has an array field."""

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "sum_nested",
        r"""
struct Middle {
    int vals[4];
}

struct Outer {
    Middle m;
    int scale;
}

int sum_nested(Outer outer) {
    int s = 0;
    for (int i = 0; i < 4; i++) {
        s += outer.m.vals[i];
    }
    return s * outer.scale;
}
""",
    )

    result = function(
        {
            "m": {"vals": [1, 2, 3, 4], "_type": "Middle"},
            "scale": 2,
            "_type": "Outer",
        }
    )
    assert result == 20


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_broadcast_array_to_scalar(device_type: DeviceType):
    """Broadcasting: pass an array of values where each element maps to a scalar parameter."""

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "double_val",
        r"""
struct Pair {
    int vals[2];
}

int double_val(Pair p) {
    return p.vals[0] + p.vals[1];
}
""",
    )

    # Pass multiple Pair structs and vectorize over them
    results = function.map((0,))(
        [
            {"vals": [1, 2], "_type": "Pair"},
            {"vals": [3, 4], "_type": "Pair"},
            {"vals": [5, 6], "_type": "Pair"},
        ],
        _result="numpy",
    )
    assert isinstance(results, np.ndarray)
    assert results.shape == (3,)
    assert results.dtype == np.int32
    assert np.array_equal(results, np.array([3, 7, 11], dtype=np.int32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_array_of_rw_structured_buffers(device_type: DeviceType):
    """A function parameter that is an array of RWStructuredBuffer<T>."""

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "double_buffers",
        r"""
void double_buffers(RWStructuredBuffer<int> buffers[4]) {
    for (int i = 0; i < 4; i++) {
        buffers[i][0] = buffers[i][0] * 2;
    }
}
""",
    )

    buffers = []
    for i in range(4):
        buf = device.create_buffer(
            element_count=1,
            struct_size=4,
            data=np.array([(i + 1) * 10], dtype=np.int32),
            usage=BufferUsage.shader_resource | BufferUsage.unordered_access,
        )
        buffers.append(buf)

    function(buffers)

    # Verify each buffer was doubled
    for i, buf in enumerate(buffers):
        result = np.frombuffer(buf.to_numpy(), dtype=np.int32)
        assert result[0] == (i + 1) * 20


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
