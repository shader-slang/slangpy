# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
from typing import Tuple, Union

import pytest
import numpy as np
from slangpy import (
    Bitmap,
    BoxFilter,
    TentFilter,
    GaussianFilter,
    MitchellFilter,
    LanczosFilter,
    FilterBoundaryCondition,
)

# Type alias for any filter object that exposes eval() and radius.
FilterType = Union[BoxFilter, TentFilter, GaussianFilter, MitchellFilter, LanczosFilter]

ALL_FILTERS = [BoxFilter(), TentFilter(), GaussianFilter(), MitchellFilter(), LanczosFilter()]

CHANNEL_FORMAT_PAIRS = [
    (1, Bitmap.PixelFormat.y),
    (2, Bitmap.PixelFormat.rg),
    (3, Bitmap.PixelFormat.rgb),
    (4, Bitmap.PixelFormat.rgba),
]

# ---------------------------------------------------------------------------
# Numpy reference resampler — builds weights from native filter.eval(), then
# applies them with numpy operations.  This is *not* a reimplementation of the
# C++ resampler; it uses the same filter weights but an independent application
# path so bugs in the C++ weighted-sum / boundary / separable logic are caught.
# ---------------------------------------------------------------------------


def _bc_lookup(source: np.ndarray, pos: int, n: int, bc: FilterBoundaryCondition) -> float:
    """Boundary-condition-aware sample lookup."""
    if 0 <= pos < n:
        return float(source[pos])
    if bc == FilterBoundaryCondition.clamp:
        return float(source[max(0, min(pos, n - 1))])
    if bc == FilterBoundaryCondition.repeat:
        return float(source[pos % n])
    if bc == FilterBoundaryCondition.mirror:
        pos = pos % (2 * n - 2)
        if pos >= n - 1:
            pos = 2 * n - 2 - pos
        return float(source[pos])
    if bc == FilterBoundaryCondition.zero:
        return 0.0
    if bc == FilterBoundaryCondition.one:
        return 1.0
    raise ValueError(f"Unknown boundary condition: {bc}")


def _reference_resample_1d(
    source: np.ndarray,
    target_res: int,
    filt: FilterType,
    bc: FilterBoundaryCondition = FilterBoundaryCondition.clamp,
    clamp_range: Tuple[float, float] = (-math.inf, math.inf),
) -> np.ndarray:
    """Resample a 1D array using weights obtained from ``filt.eval()``."""
    source_res = len(source)
    filter_radius = filt.radius
    is_box = isinstance(filt, BoxFilter)
    inv_scale = 1.0

    if target_res < source_res:
        scale = source_res / target_res
        inv_scale = 1.0 / scale
        filter_radius *= scale

    taps = math.ceil(filter_radius * 2)
    if source_res == target_res and (taps % 2) != 1:
        taps -= 1
    if filt.radius < 1:
        taps = min(taps, source_res)

    do_clamp = clamp_range != (-math.inf, math.inf)
    target = np.empty(target_res, dtype=np.float64)

    if source_res == target_res:
        # Filtering mode (same resolution).
        half_taps = taps // 2
        weights = np.array([filt.eval(float(j - half_taps)) for j in range(taps)], dtype=np.float64)
        weights /= weights.sum()

        for i in range(target_res):
            offset = i - half_taps
            samples = np.array(
                [_bc_lookup(source, offset + j, source_res, bc) for j in range(taps)],
                dtype=np.float64,
            )
            val = np.dot(samples, weights)
            if do_clamp:
                val = np.clip(val, clamp_range[0], clamp_range[1])
            target[i] = val
    else:
        # Resampling mode (different resolution).
        for i in range(target_res):
            center = (i + 0.5) / target_res * source_res
            start = int(math.floor(center - filter_radius + 0.5))

            raw_weights = np.empty(taps, dtype=np.float64)
            for j in range(taps):
                pos = start + j + 0.5 - center
                w = filt.eval(pos * inv_scale)
                if is_box and target_res > source_res:
                    w = 1.0
                raw_weights[j] = w
            raw_weights /= raw_weights.sum()

            samples = np.array(
                [_bc_lookup(source, start + j, source_res, bc) for j in range(taps)],
                dtype=np.float64,
            )
            val = np.dot(samples, raw_weights)
            if do_clamp:
                val = np.clip(val, clamp_range[0], clamp_range[1])
            target[i] = val

    return target.astype(np.float32)


def reference_resample_2d(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    filt: FilterType,
    bc: Tuple[FilterBoundaryCondition, FilterBoundaryCondition] = (
        FilterBoundaryCondition.clamp,
        FilterBoundaryCondition.clamp,
    ),
    clamp_range: Tuple[float, float] = (-math.inf, math.inf),
) -> np.ndarray:
    """Separable 2D resample. Horizontal pass then vertical pass.

    Input is (H, W) or (H, W, C). Returns the same shape convention.
    """
    squeeze = image.ndim == 2
    if squeeze:
        image = image[:, :, np.newaxis]

    src_h, src_w, channels = image.shape
    needs_h = src_w != target_width
    needs_v = src_h != target_height

    if not needs_h and not needs_v:
        return image[:, :, 0].copy() if squeeze else image.copy()

    # Clamp only on the final pass (matches C++ logic).
    h_clamp = clamp_range if not needs_v else (-math.inf, math.inf)
    current = image

    if needs_h:
        inter = np.empty((src_h, target_width, channels), dtype=np.float32)
        for y in range(src_h):
            for ch in range(channels):
                inter[y, :, ch] = _reference_resample_1d(
                    current[y, :, ch], target_width, filt, bc[0], h_clamp
                )
        current = inter

    if needs_v:
        result = np.empty((target_height, target_width, channels), dtype=np.float32)
        for x in range(target_width):
            for ch in range(channels):
                result[:, x, ch] = _reference_resample_1d(
                    current[:, x, ch], target_height, filt, bc[1], clamp_range
                )
        current = result

    return current[:, :, 0] if squeeze else current


def make_float32_bitmap(
    width: int,
    height: int,
    channels: int = 4,
    value: float = 0.0,
    srgb_gamma: bool = False,
) -> Bitmap:
    """Create a float32 bitmap filled with a constant value."""
    pixel_format = {
        1: Bitmap.PixelFormat.y,
        2: Bitmap.PixelFormat.rg,
        3: Bitmap.PixelFormat.rgb,
        4: Bitmap.PixelFormat.rgba,
    }[channels]
    if channels == 1:
        data = np.full((height, width), value, dtype=np.float32)
    else:
        data = np.full((height, width, channels), value, dtype=np.float32)
    return Bitmap(data, pixel_format=pixel_format, srgb_gamma=srgb_gamma)


# ---------------------------------------------------------------------------
# Basic resample tests
# ---------------------------------------------------------------------------


def test_resample_identity():
    """Identity resample (same dimensions) should be an exact copy."""
    data = np.random.rand(32, 32, 4).astype(np.float32)
    bmp = Bitmap(data, srgb_gamma=False)
    target = make_float32_bitmap(32, 32, channels=4)
    bmp.resample(target)
    np.testing.assert_array_equal(np.array(target, copy=False), data)


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
# Format and metadata preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "channels,pixel_format", CHANNEL_FORMAT_PAIRS, ids=["y", "rg", "rgb", "rgba"]
)
def test_channel_format_preservation(channels: int, pixel_format: Bitmap.PixelFormat):
    """resample should preserve channel count and pixel_format."""
    bmp = make_float32_bitmap(32, 32, channels=channels)
    result = bmp.resample(16, 16)
    assert result.channel_count == channels
    assert result.pixel_format == pixel_format


def test_srgb_flag_preservation():
    bmp = make_float32_bitmap(32, 32, srgb_gamma=True)
    assert bmp.srgb_gamma is True
    result = bmp.resample(16, 16)
    assert result.srgb_gamma is True


# ---------------------------------------------------------------------------
# Boundary condition tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bc_val,value",
    [
        (FilterBoundaryCondition.clamp, 0.7),
        (FilterBoundaryCondition.repeat, 0.5),
        (FilterBoundaryCondition.mirror, 0.3),
        (FilterBoundaryCondition.one, 1.0),
    ],
    ids=["clamp", "repeat", "mirror", "one"],
)
def test_boundary_solid_color(bc_val: FilterBoundaryCondition, value: float):
    """Solid color image should be preserved regardless of boundary mode."""
    data = np.full((8, 8, 1), value, dtype=np.float32)
    bmp = Bitmap(data, srgb_gamma=False)
    result = bmp.resample(4, 4, LanczosFilter(), bc=(bc_val, bc_val))
    out = np.array(result, copy=False)
    np.testing.assert_allclose(out, value, atol=1e-5)


def test_boundary_asymmetric():
    """Asymmetric H/V boundary conditions should match the numpy reference."""
    rng = np.random.RandomState(22)
    data = rng.rand(8, 8).astype(np.float32)
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    filt = LanczosFilter()
    bc = (FilterBoundaryCondition.clamp, FilterBoundaryCondition.zero)
    result = bmp.resample(12, 12, filt, bc=bc)
    out = np.array(result, copy=False)
    ref = reference_resample_2d(data, 12, 12, filt, bc=bc)
    np.testing.assert_allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Edge case and error tests
# ---------------------------------------------------------------------------


def test_resample_zero_dimensions():
    """resample with zero target dimensions should raise."""
    bmp = make_float32_bitmap(16, 16)
    with pytest.raises(Exception):
        bmp.resample(0, 10)
    with pytest.raises(Exception):
        bmp.resample(10, 0)


def test_resample_empty_bitmap():
    """resample on an empty bitmap should raise."""
    bmp = Bitmap(Bitmap.PixelFormat.rgba, Bitmap.ComponentType.float32, 0, 0)
    with pytest.raises(Exception):
        bmp.resample(8, 8)


def test_lanczos_gradient_quality():
    """Lanczos upscale of a gradient should match the numpy reference."""
    gradient = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(1, 64)
    data = np.broadcast_to(gradient, (64, 64)).copy()
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    filt = LanczosFilter()
    result = bmp.resample(128, 128, filt)
    out = np.array(result, copy=False)
    ref = reference_resample_2d(data, 128, 128, filt)
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_large_downscale():
    """Extreme downscale (1024 -> 1) should not crash and produce valid output."""
    bmp = make_float32_bitmap(1024, 1024, channels=1, value=0.42)
    result = bmp.resample(1, 1)
    assert result.width == 1
    assert result.height == 1
    out = np.array(result, copy=False)
    np.testing.assert_allclose(out.flat[0], 0.42, atol=0.1)


# ---------------------------------------------------------------------------
# Resample into pre-allocated target tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src_size,dst_size,channels,filt",
    [
        ((64, 64), (32, 32), 4, LanczosFilter()),
        ((16, 16), (64, 64), 3, MitchellFilter()),
    ],
    ids=["downscale-4ch", "upscale-3ch"],
)
def test_resample_into_target_matches_allocating(
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
    channels: int,
    filt: FilterType,
):
    """Resample into pre-allocated target should match the allocating version."""
    pixel_format = {3: Bitmap.PixelFormat.rgb, 4: Bitmap.PixelFormat.rgba}[channels]
    data = np.random.rand(src_size[0], src_size[1], channels).astype(np.float32)
    bmp = Bitmap(data, pixel_format=pixel_format, srgb_gamma=False)

    expected = bmp.resample(dst_size[0], dst_size[1], filt)
    target = make_float32_bitmap(dst_size[0], dst_size[1], channels=channels)
    bmp.resample(target, filt)

    np.testing.assert_allclose(
        np.array(target, copy=False), np.array(expected, copy=False), atol=1e-6
    )


def test_resample_into_target_float16():
    """Resample into pre-allocated float16 target."""
    data = np.random.rand(32, 32, 4).astype(np.float16)
    bmp = Bitmap(data)

    expected = bmp.resample(16, 16)
    target = Bitmap(
        np.zeros((16, 16, 4), dtype=np.float16),
        pixel_format=Bitmap.PixelFormat.rgba,
    )
    bmp.resample(target)

    np.testing.assert_allclose(
        np.array(target, copy=False), np.array(expected, copy=False), atol=1e-3
    )


def test_resample_into_target_rejects_format_mismatch():
    """Resample into target with different format should raise an error."""
    src = make_float32_bitmap(32, 32, channels=4)
    target = make_float32_bitmap(16, 16, channels=3)
    with pytest.raises(Exception):
        src.resample(target)


# ---------------------------------------------------------------------------
# Output clamping tests
# ---------------------------------------------------------------------------


def test_resample_clamp_restricts_range():
    """Clamped resample should match the numpy reference with clamp_range."""
    data = np.zeros((16, 16), dtype=np.float32)
    data[8, :] = 10.0  # Sharp spike that will cause ringing
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    filt = LanczosFilter()

    clamped = bmp.resample(16, 32, filt, clamp=(0.0, 10.0))
    out_clamped = np.array(clamped, copy=False)
    ref = reference_resample_2d(data, 16, 32, filt, clamp_range=(0.0, 10.0))
    np.testing.assert_allclose(out_clamped, ref, atol=1e-5)
    # Verify clamping actually did something.
    assert out_clamped.min() >= 0.0 - 1e-6
    assert out_clamped.max() <= 10.0 + 1e-6


def test_resample_clamp_default_is_no_clamp():
    """Default clamp should not alter results."""
    data = np.random.rand(32, 32, 4).astype(np.float32)
    bmp = Bitmap(data, srgb_gamma=False)

    result_default = bmp.resample(16, 16, LanczosFilter())
    result_explicit = bmp.resample(16, 16, LanczosFilter(), clamp=(-float("inf"), float("inf")))

    np.testing.assert_allclose(
        np.array(result_default, copy=False),
        np.array(result_explicit, copy=False),
        atol=1e-7,
    )


def test_resample_clamp_into_target():
    """Clamp should work with the pre-allocated target overload."""
    data = np.zeros((16, 16, 1), dtype=np.float32)
    data[8, :, 0] = 10.0
    bmp = Bitmap(data, pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)

    target = make_float32_bitmap(16, 32, channels=1)
    bmp.resample(target, LanczosFilter(), clamp=(0.0, 10.0))
    out = np.array(target, copy=False)
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 10.0 + 1e-6


# ---------------------------------------------------------------------------
# Single-axis resample tests
# ---------------------------------------------------------------------------


def test_resample_width_only():
    """Width-only resample of solid color should preserve values."""
    data = np.full((16, 32, 1), 0.6, dtype=np.float32)
    bmp = Bitmap(data, pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    result = bmp.resample(64, 16)
    assert result.width == 64
    assert result.height == 16
    out = np.array(result, copy=False)
    np.testing.assert_allclose(out, 0.6, atol=1e-6)


def test_resample_height_only():
    """Height-only resample of solid color should preserve values."""
    data = np.full((32, 16, 1), 0.4, dtype=np.float32)
    bmp = Bitmap(data, pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    result = bmp.resample(16, 64)
    assert result.width == 16
    assert result.height == 64
    out = np.array(result, copy=False)
    np.testing.assert_allclose(out, 0.4, atol=1e-6)


# ---------------------------------------------------------------------------
# Filter reconstruction quality tests
# ---------------------------------------------------------------------------


def test_box_filter_downscale():
    """Box filter 2x downscale should match the numpy reference."""
    rng = np.random.RandomState(10)
    data = rng.rand(16, 16).astype(np.float32)
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    filt = BoxFilter()
    result = bmp.resample(8, 8, filt)
    out = np.array(result, copy=False)
    ref = reference_resample_2d(data, 8, 8, filt)
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_tent_filter_upscale():
    """Tent filter 2x upscale should match the numpy reference."""
    rng = np.random.RandomState(11)
    data = rng.rand(8, 8).astype(np.float32)
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    filt = TentFilter()
    result = bmp.resample(16, 16, filt)
    out = np.array(result, copy=False)
    ref = reference_resample_2d(data, 16, 16, filt)
    np.testing.assert_allclose(out, ref, atol=1e-5)


@pytest.mark.parametrize(
    "filter_fn",
    ALL_FILTERS,
    ids=["box", "tent", "gaussian", "mitchell", "lanczos"],
)
@pytest.mark.parametrize("scale", [0.5, 1.5, 2.0, 3.0], ids=["0.5x", "1.5x", "2x", "3x"])
def test_all_filters_preserve_dc(filter_fn: FilterType, scale: float):
    """All normalized filters must preserve a constant (DC) signal at any scale."""
    size = 32
    value = 0.73
    data = np.full((size, size, 1), value, dtype=np.float32)
    bmp = Bitmap(data, pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    tw = max(1, int(size * scale))
    th = max(1, int(size * scale))
    result = bmp.resample(tw, th, filter_fn)
    out = np.array(result, copy=False)
    np.testing.assert_allclose(out, value, atol=1e-4)


def test_lanczos_ringing_on_step_edge():
    """Lanczos upscale of step function should match the numpy reference."""
    data = np.zeros((32, 32), dtype=np.float32)
    data[:, 16:] = 1.0
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    filt = LanczosFilter()
    result = bmp.resample(128, 32, filt)
    out = np.array(result, copy=False)
    ref = reference_resample_2d(data, 128, 32, filt)
    np.testing.assert_allclose(out, ref, atol=1e-5)
    # Verify ringing actually occurs (values outside [0, 1]).
    assert ref.min() < -0.01, "Expected negative ringing from Lanczos on step edge"
    assert ref.max() > 1.01, "Expected positive ringing from Lanczos on step edge"


def test_mitchell_less_ringing_than_lanczos():
    """Mitchell filter should produce less ringing than Lanczos on a step edge."""
    data = np.zeros((1, 32, 1), dtype=np.float32)
    data[0, 16:, 0] = 1.0
    data = np.broadcast_to(data, (32, 32, 1)).copy()
    bmp = Bitmap(data, pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)

    lanczos_result = bmp.resample(128, 32, LanczosFilter())
    mitchell_result = bmp.resample(128, 32, MitchellFilter())
    out_l = np.array(lanczos_result, copy=False)
    out_m = np.array(mitchell_result, copy=False)

    # Mitchell(1/3,1/3) should have a tighter value range than Lanczos.
    lanczos_range = out_l.max() - out_l.min()
    mitchell_range = out_m.max() - out_m.min()
    assert (
        mitchell_range < lanczos_range
    ), f"Mitchell range {mitchell_range:.4f} should be smaller than Lanczos range {lanczos_range:.4f}"


def test_gaussian_point_spread():
    """Gaussian filter on a single bright pixel should match the numpy reference."""
    data = np.zeros((16, 16), dtype=np.float32)
    data[8, 8] = 1.0
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    filt = GaussianFilter()
    result = bmp.resample(32, 32, filt)
    out = np.array(result, copy=False)
    ref = reference_resample_2d(data, 32, 32, filt)
    np.testing.assert_allclose(out, ref, atol=1e-5)
    # Verify the point spread actually happened.
    assert ref.max() < 0.9, "Gaussian should diminish the peak of a point source"


# ---------------------------------------------------------------------------
# Boundary conditions on non-trivial images
# ---------------------------------------------------------------------------


def test_boundary_repeat_tiled_pattern():
    """Repeat BC should match the numpy reference on a periodic signal."""
    x = np.linspace(0, 2 * np.pi, 16, endpoint=False, dtype=np.float32)
    row = 0.5 + 0.5 * np.cos(x)
    data = np.tile(row.reshape(1, 16), (16, 1))
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    filt = LanczosFilter()
    bc = (FilterBoundaryCondition.repeat, FilterBoundaryCondition.clamp)
    result = bmp.resample(32, 16, filt, bc=bc)
    out = np.array(result, copy=False)
    ref = reference_resample_2d(data, 32, 16, filt, bc=bc)
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_boundary_mirror_gradient():
    """Mirror BC should match the numpy reference on a gradient."""
    gradient = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(1, 16)
    data = np.broadcast_to(gradient, (16, 16)).copy()
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    filt = LanczosFilter()
    bc = (FilterBoundaryCondition.mirror, FilterBoundaryCondition.mirror)
    result = bmp.resample(16, 16, filt, bc=bc)
    out = np.array(result, copy=False)
    ref = reference_resample_2d(data, 16, 16, filt, bc=bc)
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_boundary_zero_vs_clamp_edge_pixels():
    """Zero vs clamp BC should match their respective numpy references."""
    rng = np.random.RandomState(20)
    data = rng.rand(8, 8).astype(np.float32)
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    filt = LanczosFilter(5)
    for bc_val in [FilterBoundaryCondition.clamp, FilterBoundaryCondition.zero]:
        bc = (bc_val, bc_val)
        result = bmp.resample(12, 12, filt, bc=bc)
        out = np.array(result, copy=False)
        ref = reference_resample_2d(data, 12, 12, filt, bc=bc)
        np.testing.assert_allclose(out, ref, atol=1e-5, err_msg=f"BC={bc_val}")


# ---------------------------------------------------------------------------
# Filter parameter variation tests
# ---------------------------------------------------------------------------


def test_tent_custom_radius():
    """TentFilter with non-default radius should match the numpy reference."""
    rng = np.random.RandomState(30)
    data = rng.rand(16, 16).astype(np.float32)
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    filt = TentFilter(3.0)
    result = bmp.resample(32, 32, filt)
    out = np.array(result, copy=False)
    ref = reference_resample_2d(data, 32, 32, filt)
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_gaussian_stddev_effect():
    """GaussianFilter with non-default stddev should match the numpy reference."""
    rng = np.random.RandomState(31)
    data = rng.rand(16, 16).astype(np.float32)
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    for stddev in [0.25, 0.5, 1.0]:
        filt = GaussianFilter(stddev)
        result = bmp.resample(32, 32, filt)
        out = np.array(result, copy=False)
        ref = reference_resample_2d(data, 32, 32, filt)
        np.testing.assert_allclose(out, ref, atol=1e-5, err_msg=f"stddev={stddev}")


def test_lanczos_lobes_variation():
    """LanczosFilter with different lobe counts should match the numpy reference."""
    rng = np.random.RandomState(32)
    data = rng.rand(16, 16).astype(np.float32)
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    for lobes in [1, 2, 3, 5]:
        filt = LanczosFilter(lobes)
        result = bmp.resample(32, 32, filt)
        out = np.array(result, copy=False)
        ref = reference_resample_2d(data, 32, 32, filt)
        np.testing.assert_allclose(out, ref, atol=1e-5, err_msg=f"lobes={lobes}")


def test_mitchell_catmull_rom_interpolation():
    """MitchellFilter with non-default b/c should match the numpy reference."""
    rng = np.random.RandomState(33)
    data = rng.rand(16, 16).astype(np.float32)
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    # Catmull-Rom: b=0, c=0.5
    filt = MitchellFilter(0.0, 0.5)
    result = bmp.resample(32, 16, filt)
    out = np.array(result, copy=False)
    ref = reference_resample_2d(data, 32, 16, filt)
    np.testing.assert_allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Channel independence and precision tests
# ---------------------------------------------------------------------------


def test_channel_independence():
    """Each channel should be resampled independently with no cross-channel leakage."""
    size = 16
    data = np.zeros((size, size, 3), dtype=np.float32)
    # Channel 0: horizontal gradient
    data[:, :, 0] = np.linspace(0, 1, size, dtype=np.float32).reshape(1, size)
    # Channel 1: vertical gradient
    data[:, :, 1] = np.linspace(0, 1, size, dtype=np.float32).reshape(size, 1)
    # Channel 2: constant
    data[:, :, 2] = 0.5

    bmp_rgb = Bitmap(data, pixel_format=Bitmap.PixelFormat.rgb, srgb_gamma=False)
    result_rgb = bmp_rgb.resample(32, 32, LanczosFilter())
    out_rgb = np.array(result_rgb, copy=False)

    # Resample each channel individually as single-channel bitmaps.
    for ch in range(3):
        ch_data = data[:, :, ch : ch + 1].copy()
        bmp_ch = Bitmap(ch_data, pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
        result_ch = bmp_ch.resample(32, 32, LanczosFilter())
        out_ch = np.array(result_ch, copy=False)
        np.testing.assert_allclose(
            out_rgb[:, :, ch],
            out_ch,
            atol=1e-6,
            err_msg=f"Channel {ch} differs between RGB and single-channel resample",
        )


def test_float16_vs_float32_precision():
    """float16 resample should be within expected precision of float32 resample."""
    data32 = np.random.RandomState(42).rand(16, 16, 4).astype(np.float32)
    data16 = data32.astype(np.float16)

    bmp32 = Bitmap(data32, srgb_gamma=False)
    bmp16 = Bitmap(data16)

    result32 = bmp32.resample(32, 32, MitchellFilter())
    result16 = bmp16.resample(32, 32, MitchellFilter())

    out32 = np.array(result32, copy=False)
    out16 = np.array(result16, copy=False).astype(np.float32)

    # float16 has ~3 decimal digits of precision; after resampling the error
    # accumulates somewhat, so we allow ~2e-2.
    np.testing.assert_allclose(out16, out32, atol=2e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# Small / edge-case resolution tests
# ---------------------------------------------------------------------------


def test_single_pixel_upscale():
    """Upscaling a 1x1 image should produce uniform output equal to the pixel value."""
    for filt in ALL_FILTERS:
        data = np.full((1, 1, 1), 0.42, dtype=np.float32)
        bmp = Bitmap(data, pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
        result = bmp.resample(16, 16, filt)
        out = np.array(result, copy=False)
        np.testing.assert_allclose(
            out, 0.42, atol=1e-4, err_msg=f"Failed for {type(filt).__name__}"
        )


def test_single_row_resample():
    """A 1×N image should match the numpy reference (horizontal-only path)."""
    rng = np.random.RandomState(50)
    data = rng.rand(1, 32).astype(np.float32)
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    filt = LanczosFilter()
    result = bmp.resample(64, 1, filt)
    out = np.array(result, copy=False)
    ref = reference_resample_2d(data, 64, 1, filt)
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_single_column_resample():
    """An N×1 image should match the numpy reference (vertical-only path)."""
    rng = np.random.RandomState(51)
    data = rng.rand(32, 1).astype(np.float32)
    bmp = Bitmap(data[:, :, np.newaxis], pixel_format=Bitmap.PixelFormat.y, srgb_gamma=False)
    filt = LanczosFilter()
    result = bmp.resample(1, 64, filt)
    out = np.array(result, copy=False)
    ref = reference_resample_2d(data, 1, 64, filt)
    np.testing.assert_allclose(out, ref, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
