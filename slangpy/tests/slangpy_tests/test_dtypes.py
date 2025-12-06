# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys
import numpy as np
import math

from slangpy import Struct
from slangpy.core.native import Shape
from slangpy import DeviceType, BufferUsage
from slangpy.types import NDBuffer, Tensor
from slangpy.testing import helpers

from typing import Any, Optional, Union, Type, cast


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
    device = helpers.get_torch_device(device_type)
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

    device.sync_to_device()

    strides = Shape(unravelled_shape).calc_contiguous_strides()

    tensor = buffer.to_torch()
    assert tensor.shape == unravelled_shape
    assert tensor.stride() == strides.as_tuple()
    assert tensor.dtype == torch_dtype
    assert (tensor == torch_ref).all().item()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
