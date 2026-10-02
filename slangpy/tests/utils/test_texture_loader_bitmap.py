# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import subprocess
import sys

import numpy as np
import pytest

import slangpy as spy
from slangpy.testing import helpers


def _load_bitmaps(device_type: spy.DeviceType) -> None:
    device = spy.Device(type=device_type)
    loader = spy.TextureLoader(device)
    pixels = [np.full((8, 8, 4), i, dtype=np.uint8) for i in range(33)]
    bitmaps = [spy.Bitmap(data) for data in pixels]
    options = spy.TextureLoader.Options()
    options.srgb_mode = spy.SRGBMode.linear

    # Both load_textures overloads and the array overload retain bitmaps on workers.
    for load_options in (options, [options] * len(bitmaps)):
        textures = loader.load_textures(bitmaps, load_options)
        assert len(textures) == len(bitmaps)
        for texture, expected in zip(textures, pixels):
            np.testing.assert_array_equal(texture.to_numpy(), expected)

    texture_array = loader.load_texture_array(bitmaps, options)
    assert texture_array.array_length == len(bitmaps)
    for layer, expected in enumerate(pixels):
        np.testing.assert_array_equal(texture_array.to_numpy(layer), expected)
    device.close()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_bulk_load_python_bitmaps(device_type: spy.DeviceType) -> None:
    helpers.get_device(type=device_type)  # Skip unsupported backends in the parent.
    # A GIL deadlock must fail this test instead of hanging the entire test runner.
    result = subprocess.run(
        [sys.executable, "-m", "slangpy.tests.utils.test_texture_loader_bitmap", device_type.name],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _load_bitmaps(getattr(spy.DeviceType, sys.argv[1]))
