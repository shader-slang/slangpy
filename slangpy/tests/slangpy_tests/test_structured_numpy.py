# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for issue #636: Numpy arrays created with np.dtype can't be passed to slangpy.

Verifies that structured numpy arrays (with named fields) can be:
  - Copied into NDBuffer via copy_from_numpy
  - Created via NDBuffer.from_numpy with explicit dtype
  - Created via Tensor.from_numpy with explicit dtype
  - Round-tripped (write → read → compare)
"""

import numpy as np
import pytest

import slangpy as spy
from slangpy import DeviceType
from slangpy.testing import helpers
from slangpy.types import NDBuffer, Tensor


MODULE = r"""
import "slangpy";

struct TrainingSample {
    float valid;
    int material_id;
    float2 uv;
    float3 wi;
    float3 wo;
    int mip_level;
    float3 target;
};

struct SimpleVec2 {
    float x;
    float y;
};

float sum_vec2(SimpleVec2 v) {
    return v.x + v.y;
}

void copy_training_sample(TrainingSample input, out TrainingSample output) {
    output = input;
}
"""


def _make_simple_vec2_dtype():
    return np.dtype([("x", np.float32), ("y", np.float32)])


def _make_training_sample_dtype():
    """Matches the Slang TrainingSample struct layout (std430 packing)."""
    return np.dtype(
        [
            ("valid", np.float32),
            ("material_id", np.int32),
            ("uv", np.float32, (2,)),
            ("wi", np.float32, (3,)),
            ("wo", np.float32, (3,)),
            ("mip_level", np.int32),
            ("target", np.float32, (3,)),
        ]
    )


# ---------------------------------------------------------------------------
# NDBuffer.copy_from_numpy with structured dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_copy_from_structured_numpy(device_type: DeviceType):
    """NDBuffer.copy_from_numpy accepts a structured numpy array."""
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    SimpleVec2 = module.SimpleVec2

    np_dtype = _make_simple_vec2_dtype()
    data = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=np_dtype)

    buf = NDBuffer(device, dtype=SimpleVec2, shape=(2,))
    buf.copy_from_numpy(data)

    result = buf.to_numpy()
    expected_bytes = data.tobytes()
    actual_bytes = result.tobytes()
    assert expected_bytes == actual_bytes


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_copy_from_structured_numpy_training_sample(device_type: DeviceType):
    """NDBuffer.copy_from_numpy handles larger structured dtypes matching a Slang struct.

    GPU backends may pad struct fields differently (e.g. Metal aligns float3
    to 16 bytes). When the numpy itemsize doesn't match the Slang stride,
    copy_from_numpy must raise ValueError rather than silently corrupt data.
    """
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    TrainingSample = module.TrainingSample
    np_dtype = _make_training_sample_dtype()
    slang_stride = TrainingSample.buffer_layout.stride

    N = 8
    data = np.zeros(N, dtype=np_dtype)
    for i in range(N):
        data[i]["valid"] = float(i)
        data[i]["material_id"] = i * 10
        data[i]["uv"] = [float(i), float(i + 1)]
        data[i]["wi"] = [1.0, 0.0, 0.0]
        data[i]["wo"] = [0.0, 1.0, 0.0]
        data[i]["mip_level"] = i
        data[i]["target"] = [0.5, 0.5, 0.5]

    buf = NDBuffer(device, dtype=TrainingSample, shape=(N,))

    if np_dtype.itemsize == slang_stride:
        buf.copy_from_numpy(data)
        result = buf.to_numpy()
        assert data.tobytes() == result.tobytes()
    else:
        with pytest.raises(ValueError, match="byte size"):
            buf.copy_from_numpy(data)


# ---------------------------------------------------------------------------
# NDBuffer.from_numpy with structured dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_from_numpy_structured_with_dtype(device_type: DeviceType):
    """NDBuffer.from_numpy creates buffer from structured array when dtype is given."""
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    SimpleVec2 = module.SimpleVec2

    np_dtype = _make_simple_vec2_dtype()
    data = np.array([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)], dtype=np_dtype)

    buf = NDBuffer.from_numpy(device, data, dtype=SimpleVec2)

    assert tuple(buf.shape) == (3,)

    result = buf.to_numpy()
    assert data.tobytes() == result.tobytes()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_from_numpy_structured_without_dtype_raises(device_type: DeviceType):
    """NDBuffer.from_numpy raises ValueError for structured dtype without explicit dtype."""
    device = helpers.get_device(device_type)

    np_dtype = _make_simple_vec2_dtype()
    data = np.array([(1.0, 2.0)], dtype=np_dtype)

    with pytest.raises(ValueError, match="Structured numpy dtype"):
        NDBuffer.from_numpy(device, data)


# ---------------------------------------------------------------------------
# Tensor.from_numpy with structured dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_from_numpy_structured_with_dtype(device_type: DeviceType):
    """Tensor.from_numpy creates tensor from structured array when dtype is given."""
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    SimpleVec2 = module.SimpleVec2

    np_dtype = _make_simple_vec2_dtype()
    data = np.array([(10.0, 20.0), (30.0, 40.0)], dtype=np_dtype)

    tensor = Tensor.from_numpy(device, data, dtype=SimpleVec2)

    assert tuple(tensor.shape) == (2,)

    result = tensor.to_numpy()
    assert data.tobytes() == result.tobytes()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_from_numpy_structured_without_dtype_raises(device_type: DeviceType):
    """Tensor.from_numpy raises ValueError for structured dtype without explicit dtype."""
    device = helpers.get_device(device_type)

    np_dtype = _make_simple_vec2_dtype()
    data = np.array([(1.0, 2.0)], dtype=np_dtype)

    with pytest.raises(ValueError, match="Structured numpy dtype"):
        Tensor.from_numpy(device, data)


# ---------------------------------------------------------------------------
# Size mismatch detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_copy_from_structured_numpy_size_mismatch(device_type: DeviceType):
    """copy_from_numpy detects byte size mismatch between numpy struct and Slang struct."""
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    TrainingSample = module.TrainingSample

    wrong_dtype = _make_simple_vec2_dtype()
    data = np.array([(1.0, 2.0)], dtype=wrong_dtype)

    buf = NDBuffer(device, dtype=TrainingSample, shape=(1,))
    with pytest.raises(ValueError, match="byte size"):
        buf.copy_from_numpy(data)


# ---------------------------------------------------------------------------
# Tensor.from_numpy size mismatch detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_from_numpy_structured_size_mismatch(device_type: DeviceType):
    """Tensor.from_numpy detects itemsize mismatch between numpy struct and Slang struct."""
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    TrainingSample = module.TrainingSample

    wrong_dtype = _make_simple_vec2_dtype()
    data = np.array([(1.0, 2.0)], dtype=wrong_dtype)

    with pytest.raises(ValueError, match="does not match"):
        Tensor.from_numpy(device, data, dtype=TrainingSample)


# ---------------------------------------------------------------------------
# Direct function call with structured numpy raises helpful error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_direct_call_with_structured_numpy_raises(device_type: DeviceType):
    """Passing a structured numpy array directly to a Slang function raises a clear error."""
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    np_dtype = _make_simple_vec2_dtype()
    data = np.array([(1.0, 2.0)], dtype=np_dtype)

    with pytest.raises(Exception, match="Structured numpy dtype"):
        module.sum_vec2(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
