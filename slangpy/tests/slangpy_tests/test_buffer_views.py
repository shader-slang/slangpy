# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from slangpy import DeviceType, BufferUsage
from slangpy.types import NDBuffer, Tensor
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
@pytest.mark.parametrize("buffer_type", [Tensor, NDBuffer])
@pytest.mark.parametrize("index", TEST_INDICES)
def test_indexing(
    device_type: DeviceType,
    buffer_type: Union[Type[Tensor], Type[NDBuffer]],
    index: tuple[Any, ...],
):

    device = helpers.get_device(device_type)

    shape = (10, 8, 5, 3, 5)
    rng = np.random.default_rng()
    numpy_ref = rng.random(shape, np.float32)
    buffer = buffer_type.zeros(device, dtype="float", shape=shape)
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
@pytest.mark.parametrize("buffer_type", [Tensor, NDBuffer])
def test_view(device_type: DeviceType, buffer_type: Union[Type[Tensor], Type[NDBuffer]]):

    device = helpers.get_device(device_type)
    buffer = buffer_type.zeros(device, dtype="float", shape=(32 * 32 * 32,))

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
@pytest.mark.parametrize("buffer_type", [Tensor, NDBuffer])
def test_view_errors(device_type: DeviceType, buffer_type: Union[Type[Tensor], Type[NDBuffer]]):

    device = helpers.get_device(device_type)
    buffer = buffer_type.zeros(device, dtype="float", shape=(32 * 32 * 32,))

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
@pytest.mark.parametrize("buffer_type", [Tensor, NDBuffer])
def test_broadcast_to(device_type: DeviceType, buffer_type: Union[Type[Tensor], Type[NDBuffer]]):

    device = helpers.get_device(device_type)
    buffer = buffer_type.zeros(
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
@pytest.mark.parametrize("buffer_type", [Tensor, NDBuffer])
def test_full_numpy_copy(device_type: DeviceType, buffer_type: Union[Type[Tensor], Type[NDBuffer]]):

    device = helpers.get_device(device_type)
    shape = (5, 4)

    numpy_ref = np.random.default_rng().random(shape, np.float32)
    buffer = buffer_type.zeros(device, dtype="float", shape=shape)

    buffer.copy_from_numpy(numpy_ref)
    buffer_to_np = buffer.to_numpy()
    assert (buffer_to_np == numpy_ref).all()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("buffer_type", [Tensor, NDBuffer])
def test_full_torch_copy(device_type: DeviceType, buffer_type: Union[Type[Tensor], Type[NDBuffer]]):
    try:
        import torch
    except ImportError:
        pytest.skip("Pytorch not installed", allow_module_level=True)

    if sys.platform == "darwin":
        pytest.skip(
            "PyTorch requires CUDA, that is not available on macOS", allow_module_level=True
        )

    device = helpers.get_torch_device(device_type)
    shape = (5, 4)

    torch_ref = torch.randn(shape, dtype=torch.float32).cuda()
    usage = BufferUsage.shader_resource | BufferUsage.unordered_access | BufferUsage.shared
    buffer = buffer_type.zeros(device, dtype="float", shape=shape, usage=usage)

    # Wait for buffer_type.zeros() to complete
    device.sync_to_device()

    buffer.copy_from_torch(torch_ref)

    buffer_to_torch = buffer.to_torch()
    assert torch.allclose(buffer_to_torch, torch_ref)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("buffer_type", [Tensor, NDBuffer])
def test_partial_numpy_copy(
    device_type: DeviceType, buffer_type: Union[Type[Tensor], Type[NDBuffer]]
):

    device = helpers.get_device(device_type)
    shape = (5, 4)

    numpy_ref = np.random.default_rng().random(shape, np.float32)
    buffer = buffer_type.zeros(device, dtype="float", shape=shape)

    for i in range(shape[0]):
        buffer[i].copy_from_numpy(numpy_ref[i])

    buffer_to_np = buffer.to_numpy()
    assert (buffer_to_np == numpy_ref).all()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("buffer_type", [Tensor, NDBuffer])
def test_partial_torch_copy(
    device_type: DeviceType, buffer_type: Union[Type[Tensor], Type[NDBuffer]]
):
    try:
        import torch
    except ImportError:
        pytest.skip("Pytorch not installed", allow_module_level=True)

    if sys.platform == "darwin":
        pytest.skip(
            "PyTorch requires CUDA, that is not available on macOS", allow_module_level=True
        )

    device = helpers.get_torch_device(device_type)
    shape = (5, 4)

    torch_ref = torch.randn(shape, dtype=torch.float32).cuda()
    usage = BufferUsage.shader_resource | BufferUsage.unordered_access | BufferUsage.shared
    buffer = buffer_type.zeros(device, dtype="float", shape=shape, usage=usage)

    # Wait for buffer_type.zeros() to complete
    device.sync_to_device()

    for i in range(shape[0]):
        buffer[i].copy_from_torch(torch_ref[i])

    buffer_to_torch = buffer.to_torch()
    assert torch.allclose(buffer_to_torch, torch_ref)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("buffer_type", [Tensor, NDBuffer])
def test_numpy_copy_errors(
    device_type: DeviceType, buffer_type: Union[Type[Tensor], Type[NDBuffer]]
):

    device = helpers.get_device(device_type)
    shape = (5, 4)

    buffer = buffer_type.zeros(device, dtype="float", shape=shape)

    with pytest.raises(Exception, match=r"Numpy array is larger"):
        ndarray = np.zeros((shape[0], shape[1] + 1), dtype=np.float32)
        buffer.copy_from_numpy(ndarray)

    buffer_view = buffer.view(shape, (1, shape[0]))
    with pytest.raises(Exception, match=r"Destination buffer view must be contiguous"):
        ndarray = np.zeros(shape, dtype=np.float32)
        buffer_view.copy_from_numpy(ndarray)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("buffer_type", [Tensor, NDBuffer])
def test_torch_copy_errors(
    device_type: DeviceType, buffer_type: Union[Type[Tensor], Type[NDBuffer]]
):
    try:
        import torch
    except ImportError:
        pytest.skip("Pytorch not installed", allow_module_level=True)

    if sys.platform == "darwin":
        pytest.skip(
            "PyTorch requires CUDA, that is not available on macOS", allow_module_level=True
        )

    device = helpers.get_torch_device(device_type)
    shape = (5, 4)

    usage = BufferUsage.shader_resource | BufferUsage.unordered_access | BufferUsage.shared
    buffer = buffer_type.zeros(device, dtype="float", shape=shape, usage=usage)

    # Wait for buffer_type.zeros() to complete
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
