# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from slangpy import DeviceType, BufferUsage
from slangpy.types import Tensor, Tensor
from slangpy.testing import helpers

from typing import Any, Union, Type

import numpy as np
import sys

MODULE = r"""
struct RGB {
    float x;
    float y;
    float z;
};
"""

TEST_INDICES = [
    # Partial indexing
    3,
    (3, 4, 2, 1),
    # Ellipses
    (3, 4, ...),
    (..., 1),
    (3, ..., 1),
    # Singleton dimension
    (None,),
    (2, 6, None, None, 2, None),
    # Slices
    (2, slice(4, None, None)),
    (1, slice(None, -3, None)),
    (slice(None, None, 2), ..., 3),
    (slice(4, None, 3),),
    # Full indexing
    (0, 0, 0, 0, 0),
]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("index", TEST_INDICES)
def test_indexing(
    device_type: DeviceType,
    index: tuple[Any, ...],
):

    device = helpers.get_device(device_type)

    shape = (10, 8, 5, 3, 5)
    rng = np.random.default_rng()
    numpy_ref = rng.random(shape, np.float32)
    buffer = Tensor.zeros(device, dtype="float", shape=shape)
    buffer.copy_from_numpy(numpy_ref)

    indexed_buffer = buffer.__getitem__(index)
    indexed_ndarray = numpy_ref.__getitem__(index)

    if isinstance(indexed_ndarray, np.number):
        # Result is a scalar
        assert indexed_buffer.shape.as_tuple() == (1,)
        assert indexed_buffer.strides.as_tuple() == (1,)
    else:
        # Result is an array slice
        spy_byte_strides = tuple(numpy_ref.itemsize * s for s in indexed_buffer.strides)
        spy_byte_offset = numpy_ref.itemsize * indexed_buffer.offset
        np_byte_offset = indexed_ndarray.ctypes.data - numpy_ref.ctypes.data
        assert indexed_buffer.shape.as_tuple() == indexed_ndarray.shape
        assert spy_byte_strides == indexed_ndarray.strides
        assert spy_byte_offset == np_byte_offset


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_view(device_type: DeviceType):

    device = helpers.get_device(device_type)
    buffer = Tensor.zeros(device, dtype="float", shape=(32 * 32 * 32,))

    # Transposed view of the 3rd 32x32 slice
    view_offset = 64
    view_size = (32, 32)
    view_strides = (1, 32)
    view = buffer.view(view_size, view_strides, view_offset)
    assert view.offset == view_offset
    assert view.shape.as_tuple() == view_size
    assert view.strides.as_tuple() == view_strides

    # Adjust view to original buffer
    reversed_view = view.view(buffer.shape, offset=-view_offset)
    assert reversed_view.offset == buffer.offset
    assert reversed_view.shape.as_tuple() == buffer.shape.as_tuple()
    assert reversed_view.strides.as_tuple() == buffer.strides.as_tuple()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_view_errors(device_type: DeviceType):

    device = helpers.get_device(device_type)
    buffer = Tensor.zeros(device, dtype="float", shape=(32 * 32 * 32,))

    with pytest.raises(Exception, match=r"Shape dimensions \([0-9]\) must match stride dimensions"):
        buffer.view((5, 4), (5,))

    with pytest.raises(Exception, match=r"Strides must be positive"):
        buffer.view((5, 4), (-5, 1))

    with pytest.raises(Exception, match=r"Buffer view offset is negative"):
        buffer.view((5, 4), offset=-100)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_point_to(device_type: DeviceType):

    device = helpers.get_device(device_type)

    # Create two identical buffers with different random data
    data_a = np.random.rand(16, 16, 16)
    data_b = np.random.rand(16, 16, 16)

    tensor_a = Tensor.from_numpy(device, data_a)
    tensor_b = Tensor.from_numpy(device, data_b)

    # Take a slice from the first buffer
    slice_a = tensor_a[5]
    assert slice_a.shape == (16, 16)
    assert slice_a.offset == 5 * 16 * 16

    # Sanity check: Verify slice matches expected data
    assert np.all(slice_a.to_numpy() == data_a[5])

    # Retarget slice_a to point to a slice of data_b
    slice_b = tensor_b[2]
    slice_a.point_to(slice_b)

    # Very shape is unchanged, but data reflects new view
    assert slice_a.shape == (16, 16)
    assert np.all(slice_a.to_numpy() == data_b[2])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_broadcast_to(device_type: DeviceType):

    device = helpers.get_device(device_type)
    buffer = Tensor.zeros(
        device,
        dtype="float",
        shape=(
            32,
            1,
            1,
        ),
    )

    new_shape = (64, 64, 32, 54, 5)
    broadcast_buffer = buffer.broadcast_to(new_shape)
    assert broadcast_buffer.shape == new_shape

    with pytest.raises(Exception, match=r"Broadcast shape must be larger than tensor shape"):
        buffer.broadcast_to((32,))

    with pytest.raises(Exception, match=r"Current dimension"):
        buffer.broadcast_to((16, 5, 5))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_full_numpy_copy(device_type: DeviceType):

    device = helpers.get_device(device_type)
    shape = (5, 4)

    numpy_ref = np.random.default_rng().random(shape, np.float32)
    buffer = Tensor.zeros(device, dtype="float", shape=shape)

    buffer.copy_from_numpy(numpy_ref)
    buffer_to_np = buffer.to_numpy()
    assert (buffer_to_np == numpy_ref).all()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_full_torch_copy(device_type: DeviceType):
    try:
        import torch
    except ImportError:
        pytest.skip("Pytorch not installed")

    if sys.platform == "darwin":
        pytest.skip("PyTorch requires CUDA, that is not available on macOS")

    device = helpers.get_torch_device(device_type)
    shape = (5, 4)

    torch_ref = torch.randn(shape, dtype=torch.float32).cuda()
    usage = BufferUsage.shader_resource | BufferUsage.unordered_access | BufferUsage.shared
    buffer = Tensor.zeros(device, dtype="float", shape=shape, usage=usage)

    # Wait for buffer_type.zeros() to complete
    device.sync_to_device()

    buffer.copy_from_torch(torch_ref)

    buffer_to_torch = buffer.to_torch()
    assert torch.allclose(buffer_to_torch, torch_ref)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_partial_numpy_copy(device_type: DeviceType):

    device = helpers.get_device(device_type)
    shape = (5, 4)

    numpy_ref = np.random.default_rng().random(shape, np.float32)
    buffer = Tensor.zeros(device, dtype="float", shape=shape)

    for i in range(shape[0]):
        buffer[i].copy_from_numpy(numpy_ref[i])

    buffer_to_np = buffer.to_numpy()
    assert (buffer_to_np == numpy_ref).all()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_partial_torch_copy(device_type: DeviceType):
    try:
        import torch
    except ImportError:
        pytest.skip("Pytorch not installed")

    if sys.platform == "darwin":
        pytest.skip("PyTorch requires CUDA, that is not available on macOS")

    device = helpers.get_torch_device(device_type)
    shape = (5, 4)

    torch_ref = torch.randn(shape, dtype=torch.float32).cuda()
    usage = BufferUsage.shader_resource | BufferUsage.unordered_access | BufferUsage.shared
    buffer = Tensor.zeros(device, dtype="float", shape=shape, usage=usage)

    # Wait for buffer_type.zeros() to complete
    device.sync_to_device()

    for i in range(shape[0]):
        buffer[i].copy_from_torch(torch_ref[i])

    buffer_to_torch = buffer.to_torch()
    assert torch.allclose(buffer_to_torch, torch_ref)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_numpy_copy_errors(device_type: DeviceType):

    device = helpers.get_device(device_type)
    shape = (5, 4)

    buffer = Tensor.zeros(device, dtype="float", shape=shape)

    with pytest.raises(Exception, match=r"Numpy array is larger"):
        ndarray = np.zeros((shape[0], shape[1] + 1), dtype=np.float32)
        buffer.copy_from_numpy(ndarray)

    buffer_view = buffer.view(shape, (1, shape[0]))
    with pytest.raises(Exception, match=r"Destination buffer view must be contiguous"):
        ndarray = np.zeros(shape, dtype=np.float32)
        buffer_view.copy_from_numpy(ndarray)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_torch_copy_errors(device_type: DeviceType):
    try:
        import torch
    except ImportError:
        pytest.skip("Pytorch not installed")

    if sys.platform == "darwin":
        pytest.skip("PyTorch requires CUDA, that is not available on macOS")

    device = helpers.get_torch_device(device_type)
    shape = (5, 4)

    usage = BufferUsage.shader_resource | BufferUsage.unordered_access | BufferUsage.shared
    buffer = Tensor.zeros(device, dtype="float", shape=shape, usage=usage)

    # Wait for buffer.zeros() to complete
    device.sync_to_device()

    with pytest.raises(Exception, match=r"Tensor is larger"):
        tensor = torch.zeros((shape[0], shape[1] + 1), dtype=torch.float32)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        buffer.copy_from_torch(tensor)

    buffer_view = buffer.view(shape, (1, shape[0]))
    with pytest.raises(Exception, match=r"Destination buffer view must be contiguous"):
        tensor = torch.zeros(shape, dtype=torch.float32)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        buffer_view.copy_from_torch(tensor)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_negative_index(device_type: DeviceType):
    """Negative index wrapping on StridedBufferView."""
    device = helpers.get_device(device_type)
    tensor = Tensor.empty(device, dtype="float", shape=(4,))
    data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    tensor.copy_from_numpy(data)
    device.wait_for_idle()
    assert tensor[-1].to_numpy().item() == pytest.approx(40.0)
    assert tensor[-2].to_numpy().item() == pytest.approx(30.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_out_of_bounds_index(device_type: DeviceType):
    """Out-of-bounds index raises IndexError on StridedBufferView."""
    device = helpers.get_device(device_type)
    tensor = Tensor.empty(device, dtype="float", shape=(4,))
    with pytest.raises(IndexError):
        _ = tensor[4]
    with pytest.raises(IndexError):
        _ = tensor[-5]


@pytest.mark.skipif(sys.platform == "darwin", reason="Torch tests require CUDA")
def test_copy_from_torch_cpu_fallback():
    """copy_from_torch with CPU tensor falls back through numpy."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    device = helpers.get_device(DeviceType.cuda)
    tensor = Tensor.empty(device, dtype="float", shape=(4,))
    cpu_data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    tensor.copy_from_torch(cpu_data)
    device.wait_for_idle()
    result = tensor.to_numpy()
    np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0])


# ============================================================================
# point_to error guards
# ============================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_point_to_shape_mismatch(device_type: DeviceType):
    """point_to rejects views with different shapes."""
    device = helpers.get_device(device_type)
    a = Tensor.zeros(device, dtype="float", shape=(4, 4))
    b = Tensor.zeros(device, dtype="float", shape=(8, 2))
    with pytest.raises(Exception, match="Shape"):
        a.point_to(b)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_point_to_element_stride_mismatch(device_type: DeviceType):
    """point_to rejects views with different element strides (different dtypes)."""
    device = helpers.get_device(device_type)
    a = Tensor.zeros(device, dtype="float", shape=(4,))
    b = Tensor.zeros(device, dtype="double", shape=(4,))
    with pytest.raises(Exception, match="Element size"):
        a.point_to(b)


# ============================================================================
# Indexing error paths
# ============================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_index_illegal_argument_type(device_type: DeviceType):
    """Indexing with an unsupported type (e.g. string) raises an error."""
    device = helpers.get_device(device_type)
    tensor = Tensor.zeros(device, dtype="float", shape=(4, 4))
    with pytest.raises(Exception, match="Illegal argument"):
        tensor["bad"]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_index_too_many_indices(device_type: DeviceType):
    """More real indices than dimensions raises an error."""
    device = helpers.get_device(device_type)
    tensor = Tensor.zeros(device, dtype="float", shape=(4, 4))
    with pytest.raises(Exception, match="Too many indices"):
        tensor[0, 0, 0]


# ============================================================================
# copy_from_numpy non-contiguous source
# ============================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_from_numpy_non_contiguous(device_type: DeviceType):
    """copy_from_numpy rejects a non-contiguous numpy array."""
    device = helpers.get_device(device_type)
    tensor = Tensor.zeros(device, dtype="float", shape=(4,))
    data = np.zeros((8,), dtype=np.float32)
    non_contiguous = data[::2]
    with pytest.raises(Exception, match="contiguous"):
        tensor.copy_from_numpy(non_contiguous)


# ============================================================================
# clear with explicit command encoder
# ============================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_clear_with_command_encoder(device_type: DeviceType):
    """clear() accepts an explicit CommandEncoder."""
    device = helpers.get_device(device_type)
    tensor = Tensor.empty(device, dtype="float", shape=(4,))
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    tensor.copy_from_numpy(data)
    device.wait_for_idle()

    cmd = device.create_command_encoder()
    tensor.clear(cmd)
    device.submit_command_buffer(cmd.finish())
    device.wait_for_idle()

    result = tensor.to_numpy()
    np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 0.0])


# ============================================================================
# is_contiguous with singleton dimension
# ============================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_is_contiguous_singleton_dim(device_type: DeviceType):
    """is_contiguous skips size-1 dimensions regardless of their stride.

    None indexing inserts a size-1 dim with stride 0, which would fail
    the normal contiguity check if the singleton skip were missing.
    """
    device = helpers.get_device(device_type)
    tensor = Tensor.zeros(device, dtype="float", shape=(4, 3))
    view_with_singleton = tensor[None, :, :]
    assert view_with_singleton.shape == (1, 4, 3)
    assert view_with_singleton.strides.as_tuple()[0] == 0
    assert view_with_singleton.is_contiguous()


# ============================================================================
# maybe_pad_data: numpy vector padding for aligned float3
# ============================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_from_numpy_float3_padding(device_type: DeviceType):
    """copy_from_numpy pads (N,3) float32 data when float3 layout stride > 12.

    Exercises maybe_pad_data vector padding path. If the backend uses a
    12-byte float3 stride (no padding needed), the test is skipped since
    the padding branch won't be reached.
    """
    device = helpers.get_device(device_type)
    n = 4
    tensor = Tensor.zeros(device, dtype="float3", shape=(n,))

    buf_size = tensor.storage.size
    elem_count = tensor.element_count
    stride = buf_size // elem_count
    if stride <= 12:
        pytest.skip("float3 layout stride is 12 (no padding needed on this backend)")

    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
    tensor.copy_from_numpy(data)
    device.wait_for_idle()
    result = tensor.to_numpy()
    np.testing.assert_array_almost_equal(result[:, :3], data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
