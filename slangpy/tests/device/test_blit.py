# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys
import numpy as np

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("mode", ["render", "compute"])
def test_generate_mips(device_type: spy.DeviceType, mode: str):
    if mode == "render" and device_type == spy.DeviceType.metal:
        pytest.skip("Currently fails on Metal, needs investigation.")

    device = helpers.get_device(device_type)

    if mode == "render" and not spy.Feature.rasterization in device.features:
        pytest.skip("Device does not support rasterization")

    mip0 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 8], [5, 6, 7, 8]], dtype=np.float32)
    mip1 = np.array([[3.5, 5.5], [7.5, 8.5]], dtype=np.float32)
    mip2 = np.array([[6.25]], dtype=np.float32)

    usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
    if mode == "render":
        usage |= spy.TextureUsage.render_target

    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.r32_float,
        width=4,
        height=4,
        mip_count=spy.ALL_MIPS,
        usage=usage,
        data=mip0,
    )

    encoder = device.create_command_encoder()
    encoder.generate_mips(texture)
    device.submit_command_buffer(encoder.finish())

    assert np.allclose(texture.to_numpy(mip=0), mip0)
    assert np.allclose(texture.to_numpy(mip=1), mip1)
    assert np.allclose(texture.to_numpy(mip=2), mip2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
