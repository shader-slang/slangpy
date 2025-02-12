# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import pytest
from sgl import int2

from slangpy.backend import DeviceType, float3, int3, uint3, ResourceUsage, Format
from slangpy.core.module import Module
from slangpy.experimental.tilearg import tile
from slangpy.reflection.reflectiontypes import ArrayType
from slangpy.tests import helpers
from slangpy.types.buffer import NDBuffer
from slangpy.types.randfloatarg import RandFloatArg
from slangpy.types.threadidarg import ThreadIdArg
from slangpy.types.wanghasharg import WangHashArg


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return Module(device.load_module("test_tile.slang"))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tile_centre_vector_pixel(device_type: DeviceType):

    module = load_test_module(device_type)

    width = 8
    height = 4

    tex_data = np.random.random((height, width, 4)).astype(np.float32)
    tex = module.device.create_texture(width=width, height=height, usage=ResourceUsage.shader_resource |
                                       ResourceUsage.unordered_access, format=Format.rgba32_float, data=tex_data)

    pixels = module.get_centre_pixel(tile(tex), _result='numpy')

    assert np.allclose(pixels, tex_data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tile_centre_uint_pixel(device_type: DeviceType):

    module = load_test_module(device_type)

    width = 8
    height = 4

    tex_data = np.indices((height, width), dtype=np.uint32).transpose(1, 2, 0)
    tex = module.device.create_texture(width=width, height=height, usage=ResourceUsage.shader_resource |
                                       ResourceUsage.unordered_access, format=Format.rg32_uint, data=tex_data.flatten())

    pixels = module.get_centre_pixel_uint(tile(tex), _result='numpy')

    assert np.all(pixels == tex_data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tile_offset_uint_pixel(device_type: DeviceType):

    module = load_test_module(device_type)

    width = 8
    height = 4

    tex_data = np.indices((height, width), dtype=np.uint32).transpose(1, 2, 0)
    tex = module.device.create_texture(width=width, height=height, usage=ResourceUsage.shader_resource |
                                       ResourceUsage.unordered_access, format=Format.rg32_uint, data=tex_data.flatten())

    for x_offset in range(-2, 3):
        for y_offset in range(-2, 3):
            pixels = module.get_offset_pixel_uint(
                tile(tex), int2(x_offset, y_offset), _result='numpy')

            # copy source data and apply offsets
            # NOTE: np.indices will have put the y coordinate in channel 0, and x coordinate in channel 1
            expected = np.copy(tex_data).astype(np.int32)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
