# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pytest

from slangpy import Struct
from slangpy.core.native import Shape
from slangpy import DeviceType, BufferUsage
from . import helpers
from slangpy.types import NDBuffer, Tensor

from typing import Any, Optional, Union, Type, cast

import numpy as np
import math
import sys

try:
    import torch
except ImportError:
    pytest.skip("Pytorch not installed", allow_module_level=True)

# Skip all tests in this file if running on MacOS
if sys.platform == "darwin":
    pytest.skip("PyTorch requires CUDA, that is not available on macOS", allow_module_level=True)

MODULE = r"""
struct RGB {
    float x;
    float y;
    float z;
};
"""

TEST_DTYPES = [
    ("half", torch.half, np.float16, ()),
    ("float", torch.float, np.float32, ()),
    ("double", torch.double, np.float64, ()),
    ("uint8_t", torch.uint8, np.uint8, ()),
    ("uint16_t", None, np.uint16, ()),
    ("uint32_t", None, np.uint32, ()),
    ("uint64_t", None, np.uint64, ()),
    ("int8_t", torch.int8, np.int8, ()),
    ("int16_t", torch.int16, np.int16, ()),
    ("int32_t", torch.int32, np.int32, ()),
    ("int64_t", torch.int64, np.int64, ()),
    ("float2", torch.float, np.float32, (2,)),
    ("float3", torch.float, np.float32, (3,)),
    ("float[3]", torch.float, np.float32, (3,)),
    ("float[2][3]", torch.float, np.float32, (3, 2)),
    ("float3[2]", torch.float, np.float32, (2, 3)),
    ("RGB", torch.uint8, np.uint8, (12,)),
]


TEST_INDICES = [
    # Partial indexing
    (3,),
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
@pytest.mark.parametrize("test_dtype", TEST_DTYPES)
def test_to_numpy(
    device_type: DeviceType,
    buffer_type: Union[Type[Tensor], Type[NDBuffer]],
    test_dtype: tuple[str, Optional[torch.dtype], Type[Any], tuple[int, ...]],
):

    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    slang_dtype, _, np_type, dtype_shape = test_dtype

    np_dtype = np.dtype(np_type)
    shape = (5, 4)
    unravelled_shape = shape + dtype_shape

    rng = np.random.default_rng()
    if np_type in (np.float16, np.float32, np.float64):
        numpy_ref = rng.random(unravelled_shape, np.double).astype(np_dtype)
    else:
        iinfo = np.iinfo(np_type)
        numpy_ref = rng.integers(iinfo.min, iinfo.max, unravelled_shape, np_dtype)

    buffer = buffer_type.zeros(device, dtype=module[slang_dtype], shape=shape)

    assert buffer.shape == shape
    assert buffer.strides == Shape(shape).calc_contiguous_strides()
    assert buffer.offset == 0

    buffer.copy_from_numpy(numpy_ref)

    strides = Shape(unravelled_shape).calc_contiguous_strides()
    byte_strides = tuple(s * np_dtype.itemsize for s in strides)

    ndarray = buffer.to_numpy()
    assert ndarray.shape == unravelled_shape
    assert ndarray.strides == byte_strides
    assert ndarray.dtype == np_dtype
    assert (ndarray == numpy_ref).all()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("buffer_type", [Tensor, NDBuffer])
@pytest.mark.parametrize("test_dtype", TEST_DTYPES)
def test_to_torch(
    device_type: DeviceType,
    buffer_type: Union[Type[Tensor], Type[NDBuffer]],
    test_dtype: tuple[str, Optional[torch.dtype], Type[Any], tuple[int, ...]],
):

    device = helpers.get_device(device_type, cuda_interop=True)
    module = helpers.create_module(device, MODULE)

    slang_dtype, torch_dtype, _, dtype_shape = test_dtype
    if torch_dtype is None:
        pytest.skip()
    slang_type = cast(Struct, module[slang_dtype]).struct

    shape = (5, 4)
    unravelled_shape = shape + dtype_shape

    rng = np.random.default_rng()
    if torch_dtype.is_floating_point:
        torch_ref = torch.randn(unravelled_shape, dtype=torch_dtype).cuda()
    else:
        iinfo = torch.iinfo(torch_dtype)
        torch_ref = torch.randint(iinfo.min, iinfo.max, unravelled_shape, dtype=torch_dtype).cuda()

    usage = BufferUsage.shader_resource | BufferUsage.unordered_access | BufferUsage.shared
    if buffer_type == Tensor:
        storage = device.create_buffer(
            element_count=math.prod(shape),
            struct_size=slang_type.buffer_layout.reflection.size,
            usage=usage,
        )
        buffer = Tensor(storage, slang_type, shape)
    else:
        buffer = NDBuffer(device, dtype=slang_type, shape=shape, usage=usage)
    buffer.clear()

    assert buffer.shape == shape
    assert buffer.strides == Shape(shape).calc_contiguous_strides()
    assert buffer.offset == 0

    buffer.copy_from_numpy(torch_ref.cpu().numpy())

    strides = Shape(unravelled_shape).calc_contiguous_strides()

    tensor = buffer.to_torch()
    assert tensor.shape == unravelled_shape
    assert tensor.stride() == strides.as_tuple()
    assert tensor.dtype == torch_dtype
    assert (tensor == torch_ref).all().item()


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

    indexed_buffer = buffer.__getitem__(*index)
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
