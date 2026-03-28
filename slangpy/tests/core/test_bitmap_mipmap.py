# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np
from slangpy import Bitmap, BoxFilter, KaiserFilter, MitchellFilter


def make_float32_bitmap(
    width: int,
    height: int,
    channels: int = 4,
    value: float = 0.0,
    srgb_gamma: bool = False,
) -> Bitmap:
    """Create a float32 bitmap filled with a constant value."""
    pixel_format = {1: Bitmap.PixelFormat.y, 2: Bitmap.PixelFormat.rg, 3: Bitmap.PixelFormat.rgb, 4: Bitmap.PixelFormat.rgba}[channels]
    if channels == 1:
        data = np.full((height, width), value, dtype=np.float32)
    else:
        data = np.full((height, width, channels), value, dtype=np.float32)
    return Bitmap(data, srgb_gamma=srgb_gamma)


# ---------------------------------------------------------------------------
# Resample tests
# ---------------------------------------------------------------------------


def test_resample_identity():
    data = np.random.rand(16, 16, 4).astype(np.float32)
    bmp = Bitmap(data)
    result = bmp.resample(16, 16)
    assert result.width == 16
    assert result.height == 16
    out = np.array(result, copy=False)
    np.testing.assert_allclose(out, data, atol=1e-6)


def test_resample_downscale():
    bmp = make_float32_bitmap(256, 256)
    result = bmp.resample(100, 100)
    assert result.width == 100
    assert result.height == 100
    assert result.channel_count == 4


def test_resample_upscale():
    bmp = make_float32_bitmap(64, 64)
    result = bmp.resample(256, 256)
    assert result.width == 256
    assert result.height == 256


def test_resample_non_square():
    bmp = make_float32_bitmap(200, 100)
    result = bmp.resample(50, 25, MitchellFilter())
    assert result.width == 50
    assert result.height == 25


def test_resample_box_solid_color():
    """4x4 solid color -> 2x2 should preserve the color."""
    data = np.full((4, 4, 4), [0.8, 0.4, 0.2, 1.0], dtype=np.float32)
    bmp = Bitmap(data, srgb_gamma=False)
    result = bmp.resample(2, 2)
    out = np.array(result, copy=False)
    np.testing.assert_allclose(out, data[:2, :2], atol=1e-6)


def test_resample_rejects_uint8():
    """resample should reject non-float types."""
    data = np.zeros((16, 16, 4), dtype=np.uint8)
    bmp = Bitmap(data)
    with pytest.raises(Exception, match="float"):
        bmp.resample(8, 8)


def test_resample_float16():
    data = np.random.rand(32, 32, 4).astype(np.float16)
    bmp = Bitmap(data)
    result = bmp.resample(16, 16)
    assert result.width == 16
    assert result.height == 16
    assert result.component_type == Bitmap.ComponentType.float16


# ---------------------------------------------------------------------------
# Mip generation tests
# ---------------------------------------------------------------------------


def test_generate_mip_dimensions():
    assert make_float32_bitmap(64, 64).generate_mip().width == 32
    assert make_float32_bitmap(64, 64).generate_mip().height == 32
    assert make_float32_bitmap(63, 63).generate_mip().width == 31
    assert make_float32_bitmap(63, 63).generate_mip().height == 31
    assert make_float32_bitmap(1, 1).generate_mip().width == 1
    assert make_float32_bitmap(1, 1).generate_mip().height == 1
    assert make_float32_bitmap(4, 3).generate_mip().width == 2
    assert make_float32_bitmap(4, 3).generate_mip().height == 1


def test_generate_mip_chain_length():
    bmp = make_float32_bitmap(64, 64)
    chain = bmp.generate_mip_chain()
    # 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 = 6 levels
    assert len(chain) == 6


def test_generate_mip_chain_dimensions():
    bmp = make_float32_bitmap(64, 64)
    chain = bmp.generate_mip_chain()
    expected_w, expected_h = 32, 32
    for mip in chain:
        assert mip.width == expected_w
        assert mip.height == expected_h
        expected_w = max(1, expected_w // 2)
        expected_h = max(1, expected_h // 2)


def test_box_filter_checkerboard():
    """2x2 checkerboard -> 1x1 should average to 0.5."""
    data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32).reshape(2, 2, 1)
    bmp = Bitmap(data, pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    mip = bmp.generate_mip()
    out = np.array(mip, copy=False)
    np.testing.assert_allclose(out.flat[0], 0.5, atol=0.01)


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------


def test_all_filters_produce_valid_output():
    bmp = make_float32_bitmap(64, 64)
    for f in [BoxFilter(), KaiserFilter(), MitchellFilter()]:
        r = bmp.generate_mip(f)
        assert r.width == 32
        assert r.height == 32


def test_custom_filter_parameters():
    bmp = make_float32_bitmap(64, 64)
    r1 = bmp.generate_mip(KaiserFilter(alpha=3.0, width=2.0))
    assert r1.width == 32
    r2 = bmp.generate_mip(MitchellFilter(b=0.5, c=0.5))
    assert r2.width == 32


# ---------------------------------------------------------------------------
# Non-power-of-2 and edge cases
# ---------------------------------------------------------------------------


def test_non_power_of_2():
    bmp = make_float32_bitmap(100, 60)
    chain = bmp.generate_mip_chain()
    w, h = 100, 60
    for mip in chain:
        w = max(1, w // 2)
        h = max(1, h // 2)
        assert mip.width == w
        assert mip.height == h
    assert chain[-1].width == 1
    assert chain[-1].height == 1


def test_1xN_and_Nx1():
    # 1xN
    chain = make_float32_bitmap(1, 16).generate_mip_chain()
    assert chain[-1].width == 1
    assert chain[-1].height == 1
    assert len(chain) == 4  # 8, 4, 2, 1

    # Nx1
    chain = make_float32_bitmap(16, 1).generate_mip_chain()
    assert chain[-1].width == 1
    assert chain[-1].height == 1
    assert len(chain) == 4


# ---------------------------------------------------------------------------
# Channel count and format preservation
# ---------------------------------------------------------------------------


def test_channel_counts():
    for ch in [1, 2, 3, 4]:
        bmp = make_float32_bitmap(32, 32, channels=ch)
        result = bmp.resample(16, 16)
        assert result.channel_count == ch


def test_srgb_flag_preservation():
    bmp = make_float32_bitmap(32, 32, srgb_gamma=True)
    assert bmp.srgb_gamma is True
    mip = bmp.generate_mip()
    assert mip.srgb_gamma is True
    for level in bmp.generate_mip_chain():
        assert level.srgb_gamma is True
