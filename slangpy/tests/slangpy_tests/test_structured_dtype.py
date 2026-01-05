# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for structured numpy dtype support in NDBuffer.
Related to: https://github.com/shader-slang/slangpy/issues/636
"""

import numpy as np
import pytest

import slangpy as spy
from slangpy import DeviceType
from slangpy.testing import helpers
from slangpy.types import NDBuffer


STRUCTURED_DTYPE_MODULE = r"""
import "slangpy";

struct Point2D {
    float x;
    float y;
};

struct TrainingSample {
    float valid;
    int material_id;
    float2 uv;
    float3 wi;
    float3 wo;
    int mip_level;
    float3 target;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void double_points(uint3 tid: SV_DispatchThreadID, RWStructuredBuffer<Point2D> points, uniform uint count) {
    uint idx = tid.x;
    if (idx >= count) return;
    points[idx].x *= 2.0;
    points[idx].y *= 2.0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void process_sample(uint3 tid: SV_DispatchThreadID, RWStructuredBuffer<TrainingSample> samples, uniform uint count) {
    uint idx = tid.x;
    if (idx >= count) return;
    samples[idx].target = samples[idx].target * 2.0;
}
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
