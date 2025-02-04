# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from slangpy.backend import DeviceType, float3, int3, uint3, ResourceUsage, Format
from slangpy.core.module import Module
from slangpy.experimental.tilearg import tile
from slangpy.tests import helpers
from slangpy.types.buffer import NDBuffer
from slangpy.types.randfloatarg import RandFloatArg
from slangpy.types.threadidarg import ThreadIdArg
from slangpy.types.wanghasharg import WangHashArg


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return Module(device.load_module("test_tile.slang"))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tile_(device_type: DeviceType):

    module = load_test_module(device_type)

    width = 8
    height = 4

    tex_data = np.random.random((height, width, 4)).astype(np.float32)
    tex = module.device.create_texture(width=width, height=height, usage=ResourceUsage.shader_resource |
                                       ResourceUsage.unordered_access, format=Format.rgba32_float, data=tex_data)

    pixels = module.get_centre_pixel(tile(tex), _result='numpy')

    assert np.allclose(pixels, tex_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
