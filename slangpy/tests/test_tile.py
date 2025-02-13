# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import pytest

from slangpy.backend import DeviceType, ResourceUsage, Format, int2
from slangpy.core.module import Module
from slangpy.experimental.gridarg import grid
from slangpy.experimental.tilearg import tile
from slangpy.tests import helpers
from slangpy.types.buffer import NDBuffer
from slangpy.types.callidarg import call_id
from slangpy.types.tensor import Tensor


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return Module(device.load_module("test_tile.slang"))


def create_random_float4_storage(module: Module, width: int, height: int, type: str):
    data = np.random.random((height, width, 4)).astype(np.float32)
    if type == 'texture':
        storage = module.device.create_texture(width=width, height=height, usage=ResourceUsage.shader_resource |
                                               ResourceUsage.unordered_access, format=Format.rgba32_float, data=data.flatten())
    elif type == 'ndbuffer':
        storage = NDBuffer(module.device, dtype=module.float4, shape=(height, width))
        storage.copy_from_numpy(data.flatten())
    elif type == 'tensor':
        storage = Tensor.empty(module.device, dtype=module.float4, shape=(height, width))
        storage.storage.copy_from_numpy(data.flatten())
    elif type == 'difftensor':
        storage = Tensor.empty(module.device, dtype=module.float4,
                               shape=(height, width)).with_grads(zero=True)
        storage.storage.copy_from_numpy(data.flatten())
    else:
        raise ValueError(f"Unknown storage type {type}")
    return (data, storage)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("storage_type", ['texture', 'ndbuffer', 'tensor', 'difftensor'])
def test_tile_centre_vector_pixel(device_type: DeviceType, storage_type: str):

    module = load_test_module(device_type)

    width = 8
    height = 4

    (data, storage) = create_random_float4_storage(module, width, height, storage_type)

    pixels = module.get_centre_pixel(tile(storage), _result='numpy')

    assert np.allclose(pixels, data)


def create_uint_storage(module: Module, width: int, height: int, type: str):
    data = np.indices((height, width), dtype=np.uint32).transpose(1, 2, 0)
    if type == 'texture':
        storage = module.device.create_texture(width=width, height=height, usage=ResourceUsage.shader_resource |
                                               ResourceUsage.unordered_access, format=Format.rg32_uint, data=data.flatten())
    elif type == 'ndbuffer':
        storage = NDBuffer(module.device, dtype=module.uint2, shape=(height, width))
        storage.copy_from_numpy(data.flatten())
    elif type == 'tensor':
        raise NotImplementedError("Tensor storage not implemented")
    else:
        raise ValueError(f"Unknown storage type {type}")
    return (data, storage)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("storage_type", ['texture', 'ndbuffer'])
def test_tile_centre_uint_pixel(device_type: DeviceType, storage_type: str):

    module = load_test_module(device_type)

    width = 8
    height = 4

    (data, storage) = create_uint_storage(module, width, height, storage_type)

    pixels = module.get_centre_pixel_uint(tile(storage), _result='numpy')

    assert np.all(pixels == data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tile_offset_uint_pixel(device_type: DeviceType):

    module = load_test_module(device_type)

    width = 8
    height = 4

    (data, storage) = create_uint_storage(module, width, height, 'texture')

    for x_offset in range(-2, 3):
        for y_offset in range(-2, 3):
            pixels = module.get_offset_pixel_uint(
                tile(storage), int2(x_offset, y_offset), _result='numpy')

            # copy source data and apply offsets
            # NOTE: np.indices will have put the y coordinate in channel 0, and x coordinate in channel 1
            expected = np.copy(data).astype(np.int32)
            expected[:, :, 0] += y_offset
            expected[:, :, 1] += x_offset

            # set values to -1 where out of bounds
            expected[expected[:, :, 0] > height-1] = [-1, -1]
            expected[expected[:, :, 0] < 0] = [-1, -1]
            expected[expected[:, :, 1] > width-1] = [-1, -1]
            expected[expected[:, :, 1] < 0] = [-1, -1]

            # reinterpret as uint for comparisons
            expected = expected.view(np.uint32)

            # compare to shader results
            assert np.all(pixels == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tile_offset_write_uint_pixel(device_type: DeviceType):

    module = load_test_module(device_type)

    width = 8
    height = 4

    (data, storage) = create_uint_storage(module, width, height, 'texture')

    for x_offset in range(-2, 3):
        for y_offset in range(-2, 3):
            module.clear(storage)

            module.set_offset_pixel_uint(
                tile(storage), int2(x_offset, y_offset), call_id(), _result='numpy')

            pixels = storage.to_numpy().view(np.uint32).reshape((height, width, 2))

            # copy source data and apply offsets
            # NOTE: np.indices will have put the y coordinate in channel 0, and x coordinate in channel 1
            expected = np.copy(data).astype(np.int32)
            expected[:, :, 0] -= y_offset
            expected[:, :, 1] -= x_offset

            # set values to -1 where out of bounds
            expected[expected[:, :, 0] > height-1] = [-1, -1]
            expected[expected[:, :, 0] < 0] = [-1, -1]
            expected[expected[:, :, 1] > width-1] = [-1, -1]
            expected[expected[:, :, 1] < 0] = [-1, -1]

            # reinterpret as uint for comparisons
            expected = expected.view(np.uint32)

            # compare to shader results
            assert np.all(pixels == expected)

# Friendly to the brain sanity check manipulating pixels with uint2s


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("storage_type", ['ndbuffer', 'tensor'])
def test_blur(device_type: DeviceType, storage_type: str):

    module = load_test_module(device_type)
    width = 8
    height = 4

    # 2 striped black/white patterns, where the first alternates
    # in x and the 2nd alternates in y
    x_alternating = module.generate_x_checkboard(grid((height, width)), _result=storage_type)
    y_alternating = module.generate_y_checkboard(grid((height, width)), _result=storage_type)

    # run horizontal blur on both
    xx_blurred = module.blur_x(tile(x_alternating), _result='numpy')
    xy_blurred = module.blur_x(tile(y_alternating), _result='numpy')

    # we expect the horizontal blur to have generated softened values
    # on stripes alternating in x, and make no difference to stripes
    # alternating in y
    assert np.all(np.logical_or(xx_blurred > 4, xx_blurred < 6))
    assert np.all(np.logical_or(xy_blurred == 0, xy_blurred == 1))

    # do the inverse checks with a vertical blur
    yx_blurred = module.blur_y(tile(x_alternating), _result='numpy')
    yy_blurred = module.blur_y(tile(y_alternating), _result='numpy')
    assert np.all(np.logical_or(yx_blurred == 0,  yx_blurred == 1))
    assert np.all(np.logical_or(yy_blurred > 4, yy_blurred < 6))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
