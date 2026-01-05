# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for structured numpy dtype support in NDBuffer.
Related to: https://github.com/shader-slang/slangpy/issues/636
"""

import numpy as np
import pytest

import slangpy as spy
from slangpy import BufferCursor, DeviceType
from slangpy.testing import helpers
from slangpy.types import NDBuffer


STRUCTURED_DTYPE_MODULE = r"""
import "slangpy";

// Basic struct - two floats
struct Point2D {
    float x;
    float y;
};

// Single field struct - edge case
struct SingleField {
    float value;
};

// Integer struct - tests signed and unsigned
struct IntStruct {
    int a;
    int b;
    uint c;  // Mix signed and unsigned
};

// Vec3 struct - Important: float3 has alignment issues on Metal (16B vs 12B)
struct Vec3Struct {
    float3 position;
    float3 normal;
};

// 64-bit types
struct Double64Struct {
    double value;
    int64_t big_int;
};

// Mixed everything - comprehensive test (from issue #636)
struct TrainingSample {
    float valid;
    int material_id;
    float2 uv;
    float3 wi;
    float3 wo;
    int mip_level;
    float3 target;
};

// Struct with array
struct ArrayStruct {
    float values[4];
    int count;
};
"""


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return helpers.create_module(device, STRUCTURED_DTYPE_MODULE)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_copy_from_numpy_structured_dtype(device_type: DeviceType):
    """Test that copy_from_numpy handles structured dtypes correctly."""
    device = helpers.get_device(device_type)
    module = load_test_module(device_type)

    # Create a structured numpy dtype matching Point2D
    point_dtype = np.dtype([("x", np.float32), ("y", np.float32)])

    # Create test data
    num_points = 10
    data = np.zeros(num_points, dtype=point_dtype)
    data["x"] = np.arange(num_points, dtype=np.float32)
    data["y"] = np.arange(num_points, dtype=np.float32) * 2.0

    # Create NDBuffer with explicit Slang type
    Point2D = module.Point2D
    buffer = NDBuffer(
        device,
        dtype=Point2D,
        shape=(num_points,),
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    # This should work now - structured dtype is converted to bytes internally
    buffer.copy_from_numpy(data)

    # Read back and verify
    result = buffer.to_numpy()
    # Result comes back as bytes, view as structured dtype and flatten
    result_view = result.view(point_dtype).flatten()

    assert np.allclose(result_view["x"], data["x"])
    assert np.allclose(result_view["y"], data["y"])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_from_numpy_with_explicit_dtype(device_type: DeviceType):
    """Test that from_numpy works with structured dtypes when dtype is provided."""
    device = helpers.get_device(device_type)
    module = load_test_module(device_type)

    # Create a structured numpy dtype matching Point2D
    point_dtype = np.dtype([("x", np.float32), ("y", np.float32)])

    # Create test data
    num_points = 5
    data = np.zeros(num_points, dtype=point_dtype)
    data["x"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    data["y"] = [10.0, 20.0, 30.0, 40.0, 50.0]

    # Create NDBuffer using from_numpy with explicit dtype
    Point2D = module.Point2D
    buffer = NDBuffer.from_numpy(device, data, dtype=Point2D, shape=(num_points,))

    # Read back and verify
    result = buffer.to_numpy()
    result_view = result.view(point_dtype).flatten()

    assert np.allclose(result_view["x"], data["x"])
    assert np.allclose(result_view["y"], data["y"])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_from_numpy_structured_dtype_requires_explicit_type(
    device_type: DeviceType,
):
    """Test that from_numpy raises an error for structured dtypes without explicit dtype."""
    device = helpers.get_device(device_type)

    # Create a structured numpy dtype
    point_dtype = np.dtype([("x", np.float32), ("y", np.float32)])
    data = np.zeros(10, dtype=point_dtype)

    # This should raise an error because structured dtypes can't be auto-converted
    with pytest.raises(ValueError, match="Structured numpy dtype"):
        NDBuffer.from_numpy(device, data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_from_numpy_raw_bytes_workaround(device_type: DeviceType):
    """
    Test the traditional workaround of viewing structured dtype as bytes.
    This demonstrates that users can still use the explicit byte view approach.
    """
    device = helpers.get_device(device_type)
    module = load_test_module(device_type)

    # Create a simple structured dtype
    point_dtype = np.dtype([("x", np.float32), ("y", np.float32)])
    num_points = 5
    data = np.zeros(num_points, dtype=point_dtype)
    data["x"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    data["y"] = [10.0, 20.0, 30.0, 40.0, 50.0]

    # Get the Slang type
    Point2D = module.Point2D

    # Create buffer
    buffer = NDBuffer(
        device,
        dtype=Point2D,
        shape=(num_points,),
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    # Old workaround still works - view as bytes explicitly
    buffer.copy_from_numpy(data.view(np.uint8))

    # Verify data was copied correctly
    result = buffer.to_numpy().view(point_dtype).flatten()
    assert np.allclose(result["x"], data["x"])
    assert np.allclose(result["y"], data["y"])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_simple_dtype_still_works(device_type: DeviceType):
    """Ensure simple dtypes still work after the changes."""
    device = helpers.get_device(device_type)

    # Simple float32 array
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buffer = NDBuffer.from_numpy(device, data)

    result = buffer.to_numpy().view(np.float32)
    assert np.allclose(result, data)

    # Simple int32 array
    data_int = np.array([1, 2, 3, 4], dtype=np.int32)
    buffer_int = NDBuffer.from_numpy(device, data_int)

    result_int = buffer_int.to_numpy().view(np.int32)
    assert np.all(result_int == data_int)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_issue_636_structured_dtype(device_type: DeviceType):
    """
    Test with a structured dtype containing multiple field types.
    https://github.com/shader-slang/slangpy/issues/636

    This test verifies that copy_from_numpy accepts structured dtypes
    without requiring a manual .view(np.uint8) workaround.
    """
    device = helpers.get_device(device_type)
    module = load_test_module(device_type)

    # A structured dtype with multiple field types (from issue #636)
    training_sample_dtype = np.dtype(
        [
            ("valid", np.float32, (1,)),
            ("material_id", np.int32, (1,)),
            ("uv", np.float32, (2,)),
            ("wi", np.float32, (3,)),
            ("wo", np.float32, (3,)),
            ("mip_level", np.int32, (1,)),
            ("target", np.float32, (3,)),
        ]
    )

    num_samples = 10
    data = np.zeros(num_samples, dtype=training_sample_dtype)
    data["valid"] = 1.0
    data["material_id"] = np.arange(num_samples).reshape(-1, 1)
    data["uv"][:, 0] = 0.5
    data["uv"][:, 1] = 0.75
    data["target"][:, 0] = 1.0
    data["target"][:, 1] = 2.0
    data["target"][:, 2] = 3.0

    # Create NDBuffer with explicit Slang type
    TrainingSample = module.TrainingSample
    buffer = NDBuffer(
        device,
        dtype=TrainingSample,
        shape=(num_samples,),
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    # Previously this would fail with a nanobind error
    buffer.copy_from_numpy(data)

    # Verify each field roundtrips correctly using BufferCursor
    # (can't use .view() because Metal has different float3 alignment)
    cursor = BufferCursor(buffer.dtype.buffer_layout.reflection, buffer.storage)
    for i in range(num_samples):
        elem = cursor[i]
        assert np.isclose(elem["valid"].read(), data["valid"][i])
        assert elem["material_id"].read() == data["material_id"][i]
        assert np.allclose(elem["uv"].read(), data["uv"][i])
        assert np.allclose(elem["wi"].read(), data["wi"][i])
        assert np.allclose(elem["wo"].read(), data["wo"][i])
        assert elem["mip_level"].read() == data["mip_level"][i]
        assert np.allclose(elem["target"].read(), data["target"][i])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_single_field_struct(device_type: DeviceType):
    """Test edge case: struct with only one field."""
    device = helpers.get_device(device_type)
    module = load_test_module(device_type)

    dtype = np.dtype([("value", np.float32)])
    num_elements = 8
    data = np.zeros(num_elements, dtype=dtype)
    data["value"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    buffer = NDBuffer(
        device,
        dtype=module.SingleField,
        shape=(num_elements,),
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    buffer.copy_from_numpy(data)

    result = buffer.to_numpy().view(dtype).flatten()
    assert np.allclose(result["value"], data["value"])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_integer_struct(device_type: DeviceType):
    """Test struct with integer fields (signed and unsigned, no floats)."""
    device = helpers.get_device(device_type)
    module = load_test_module(device_type)

    # Mix of signed (int32) and unsigned (uint32)
    dtype = np.dtype([("a", np.int32), ("b", np.int32), ("c", np.uint32)])
    num_elements = 5
    data = np.zeros(num_elements, dtype=dtype)
    data["a"] = [-1, -2, 3, 4, 5]  # Signed values
    data["b"] = [10, 20, 30, 40, 50]
    data["c"] = [0xFFFFFFFF, 0x12345678, 0xDEADBEEF, 100, 0]  # Unsigned values

    buffer = NDBuffer(
        device,
        dtype=module.IntStruct,
        shape=(num_elements,),
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    buffer.copy_from_numpy(data)

    result = buffer.to_numpy().view(dtype).flatten()
    assert np.all(result["a"] == data["a"])
    assert np.all(result["b"] == data["b"])
    assert np.all(result["c"] == data["c"])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vec3_struct(device_type: DeviceType):
    """
    Test struct with float3 vectors.

    This is particularly important because float3 has different alignment
    on Metal (16 bytes) vs CUDA/Vulkan (12 bytes). BufferCursor handles this.
    """
    device = helpers.get_device(device_type)
    module = load_test_module(device_type)

    dtype = np.dtype([("position", np.float32, (3,)), ("normal", np.float32, (3,))])
    num_elements = 4
    data = np.zeros(num_elements, dtype=dtype)
    data["position"] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    data["normal"] = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.577, 0.577, 0.577]]

    buffer = NDBuffer(
        device,
        dtype=module.Vec3Struct,
        shape=(num_elements,),
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    buffer.copy_from_numpy(data)

    # Verify using BufferCursor to read back with correct alignment
    cursor = BufferCursor(buffer.dtype.buffer_layout.reflection, buffer.storage)
    for i in range(num_elements):
        elem = cursor[i]
        pos = elem["position"].read()
        norm = elem["normal"].read()
        assert np.allclose(pos, data["position"][i])
        assert np.allclose(norm, data["normal"][i])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_64bit_types_struct(device_type: DeviceType):
    """Test struct with 64-bit types (double and int64)."""
    device = helpers.get_device(device_type)
    module = load_test_module(device_type)

    dtype = np.dtype([("value", np.float64), ("big_int", np.int64)])
    num_elements = 4
    data = np.zeros(num_elements, dtype=dtype)
    data["value"] = [3.14159265358979, 2.71828182845904, 1.41421356237309, 1.61803398874989]
    data["big_int"] = [2**60, 2**61, 2**62, 2**63 - 1]

    buffer = NDBuffer(
        device,
        dtype=module.Double64Struct,
        shape=(num_elements,),
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    buffer.copy_from_numpy(data)

    result = buffer.to_numpy().view(dtype).flatten()
    assert np.allclose(result["value"], data["value"])
    assert np.all(result["big_int"] == data["big_int"])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_array_in_struct(device_type: DeviceType):
    """Test struct containing a fixed-size array."""
    device = helpers.get_device(device_type)
    module = load_test_module(device_type)

    dtype = np.dtype([("values", np.float32, (4,)), ("count", np.int32)])
    num_elements = 3
    data = np.zeros(num_elements, dtype=dtype)
    data["values"] = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    data["count"] = [4, 3, 2]

    buffer = NDBuffer(
        device,
        dtype=module.ArrayStruct,
        shape=(num_elements,),
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    buffer.copy_from_numpy(data)

    result = buffer.to_numpy().view(dtype).flatten()
    assert np.allclose(result["values"], data["values"])
    assert np.all(result["count"] == data["count"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
