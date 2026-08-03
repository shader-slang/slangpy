# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for Tensor.load_from_image (types/common.py load_buffer_data_from_image).

Exercises channel-count branches, flip_y, linearize, scale/offset, and greyscale.
"""

import pytest
import numpy as np
import tempfile
import os

import slangpy as spy
from slangpy.testing import helpers


@pytest.fixture(scope="module")
def device() -> spy.Device:
    try:
        return helpers.get_device(spy.DeviceType.cuda)
    except Exception as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")


@pytest.fixture(scope="module")
def image_dir():
    d = os.path.join(tempfile.gettempdir(), "slangpy_test_images")
    os.makedirs(d, exist_ok=True)
    return d


def _save_image(path: str, data: np.ndarray, pixel_format: spy.Bitmap.PixelFormat):
    b = spy.Bitmap(data, pixel_format=pixel_format)
    b.write(path)


@pytest.fixture(scope="module")
def rgb_image(image_dir):
    data = np.array(
        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 0.0, 0.5]]], dtype=np.float32
    )
    path = os.path.join(image_dir, "test_rgb.exr")
    _save_image(path, data, spy.Bitmap.PixelFormat.rgb)
    return path, data


@pytest.fixture(scope="module")
def rgba_image(image_dir):
    data = np.array(
        [
            [[0.1, 0.2, 0.3, 1.0], [0.4, 0.5, 0.6, 0.8]],
            [[0.7, 0.8, 0.9, 0.5], [1.0, 0.0, 0.5, 0.2]],
        ],
        dtype=np.float32,
    )
    path = os.path.join(image_dir, "test_rgba.exr")
    _save_image(path, data, spy.Bitmap.PixelFormat.rgba)
    return path, data


@pytest.fixture(scope="module")
def grey_image(image_dir):
    data = np.array([[0.1, 0.4], [0.7, 1.0]], dtype=np.float32)
    path = os.path.join(image_dir, "test_grey.exr")
    _save_image(path, data, spy.Bitmap.PixelFormat.y)
    return path, data


@pytest.fixture(scope="module")
def rg_image(image_dir):
    data = np.array([[[0.1, 0.2], [0.4, 0.5]], [[0.7, 0.8], [1.0, 0.0]]], dtype=np.float32)
    path = os.path.join(image_dir, "test_rg.exr")
    _save_image(path, data, spy.Bitmap.PixelFormat.ya)
    return path, data


def test_load_rgb(device, rgb_image):
    """Load a 3-channel RGB image via Tensor.load_from_image."""
    path, expected = rgb_image
    tensor = spy.Tensor.load_from_image(device, path)
    result = tensor.to_numpy()
    assert result.shape == expected.shape
    assert np.allclose(result, expected, atol=1e-5)


def test_load_rgba(device, rgba_image):
    """Load a 4-channel RGBA image."""
    path, expected = rgba_image
    tensor = spy.Tensor.load_from_image(device, path)
    result = tensor.to_numpy()
    assert result.shape == expected.shape
    assert np.allclose(result, expected, atol=1e-5)


def test_load_greyscale_single_channel(device, grey_image):
    """Load a single-channel image."""
    path, expected = grey_image
    tensor = spy.Tensor.load_from_image(device, path)
    result = tensor.to_numpy()
    assert result.shape == expected.shape
    assert np.allclose(result, expected, atol=1e-5)


def test_load_two_channel(device, rg_image):
    """Load a 2-channel image and verify it produces float2 tensor."""
    path, expected = rg_image
    tensor = spy.Tensor.load_from_image(device, path)
    result = tensor.to_numpy()
    assert result.ndim == 3
    assert result.shape[2] == 2


def test_load_flip_y(device, rgb_image):
    """Load with flip_y=True - rows should be vertically reversed."""
    path, expected = rgb_image
    tensor = spy.Tensor.load_from_image(device, path, flip_y=True)
    result = tensor.to_numpy()
    assert result.shape == expected.shape
    assert np.allclose(result, np.flipud(expected), atol=1e-5)


def test_load_scale_offset(device, rgb_image):
    """Load with scale and offset applied."""
    path, expected = rgb_image
    tensor = spy.Tensor.load_from_image(device, path, scale=2.0, offset=0.5)
    result = tensor.to_numpy()
    assert np.allclose(result, expected * 2.0 + 0.5, atol=1e-5)


def test_load_greyscale_flag(device, rgb_image):
    """Load an RGB image with grayscale=True to force single-channel output."""
    path, _ = rgb_image
    tensor = spy.Tensor.load_from_image(device, path, grayscale=True)
    result = tensor.to_numpy()
    assert result.ndim == 2


def test_load_linearize(device, rgb_image):
    """Load with linearize=True to disable sRGB gamma correction."""
    path, _ = rgb_image
    tensor = spy.Tensor.load_from_image(device, path, linearize=True)
    result = tensor.to_numpy()
    assert result.shape[0] == 2 and result.shape[1] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
