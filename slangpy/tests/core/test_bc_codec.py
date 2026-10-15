# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path

import numpy as np
import pytest

import slangpy as spy


@pytest.fixture(
    params=[
        spy.BCEncoderBackend.software,
        spy.BCEncoderBackend.nvtt_cpu,
        spy.BCEncoderBackend.nvtt_gpu,
    ]
)
def encoder(request: pytest.FixtureRequest) -> spy.BCEncoder:
    backend = request.param
    if not spy.BCEncoder.is_backend_available(backend):
        pytest.skip(f"{backend} is unavailable")
    return spy.BCEncoder(backend)


def test_backends() -> None:
    assert spy.BCEncoder.is_backend_available(spy.BCEncoderBackend.automatic)
    assert spy.BCEncoder.is_backend_available(spy.BCEncoderBackend.software)
    assert spy.BCEncoder().backend != spy.BCEncoderBackend.automatic
    for backend in [spy.BCEncoderBackend.nvtt_cpu, spy.BCEncoderBackend.nvtt_gpu]:
        if not spy.BCEncoder.is_backend_available(backend):
            with pytest.raises(RuntimeError):
                spy.BCEncoder(backend)
    encoder = spy.BCEncoder(spy.BCEncoderBackend.software)
    assert encoder.backend == spy.BCEncoderBackend.software
    assert encoder.can_encode(spy.Format.bc7_unorm)
    assert not encoder.can_encode(spy.Format.bc6h_ufloat)
    assert not encoder.can_encode(spy.Format.bc4_snorm)
    assert not encoder.can_encode(spy.Format.rgba8_unorm)


@pytest.mark.parametrize(
    "format,block_size", [(spy.Format.bc1_unorm, 8), (spy.Format.bc7_unorm_srgb, 16)]
)
def test_mips_and_dds(
    encoder: spy.BCEncoder, tmp_path: Path, format: spy.Format, block_size: int
) -> None:
    pixels = np.full((7, 13, 4), [64, 128, 192, 255], dtype=np.uint8)
    bitmap = spy.Bitmap(pixels, srgb_gamma=format == spy.Format.bc7_unorm_srgb)
    source = tmp_path / "source.png"
    bitmap.write(source)
    bitmap = spy.Bitmap(source)
    image = encoder.encode(
        bitmap, format, {"generate_mipmaps": True, "quality": spy.BCEncodeQuality.fastest}
    )
    assert image.format == format
    assert (image.width, image.height, image.mip_count) == (13, 7, 4)
    for mip, (width, height) in zip(image.mip_levels, [(13, 7), (6, 3), (3, 1), (1, 1)]):
        assert (mip.width, mip.height) == (width, height)
        assert len(mip.data) == ((width + 3) // 4) * ((height + 3) // 4) * block_size
        assert mip.data.dtype == np.uint8
    path = tmp_path / "encoded.dds"
    image.write_dds(path)
    decoded = spy.Bitmap(path)
    assert (decoded.width, decoded.height) == (13, 7)
    assert decoded.srgb_gamma == (format == spy.Format.bc7_unorm_srgb)
    assert path.read_bytes().endswith(b"".join(bytes(mip.data) for mip in image.mip_levels))


@pytest.mark.parametrize("filter", [spy.BoxFilter(), spy.TentFilter(), spy.MitchellFilter()])
def test_options(encoder: spy.BCEncoder, filter: object) -> None:
    options = spy.BCEncodeOptions({"generate_mipmaps": True, "mip_filter": filter})
    options.quality = spy.BCEncodeQuality.production
    options.channel_weights = [2, 3, 4, 1]
    options.has_alpha = False
    assert list(options.channel_weights) == [2, 3, 4, 1]
    bitmap = spy.Bitmap(np.full((5, 9, 3), 0.5, dtype=np.float32))
    assert encoder.encode(bitmap, spy.Format.bc7_unorm, options).mip_count == 4
    assert encoder.encode(bitmap, spy.Format.bc7_unorm).mip_count == 1
    assert encoder.encode(bitmap, spy.Format.bc7_unorm, None).mip_count == 1


@pytest.mark.parametrize(
    "srgb_source,srgb_destination", [(True, False), (False, True), (True, True)]
)
def test_color_space(
    encoder: spy.BCEncoder, tmp_path: Path, srgb_source: bool, srgb_destination: bool
) -> None:
    bitmap = spy.Bitmap(np.full((4, 4, 3), 0.5, dtype=np.float32), srgb_gamma=srgb_source)
    format = spy.Format.bc7_unorm_srgb if srgb_destination else spy.Format.bc7_unorm
    image = encoder.encode(bitmap, format)
    path = tmp_path / "color.dds"
    image.write_dds(path)
    decoded = np.array(spy.Bitmap(path)) / 255.0
    expected = 0.5
    if srgb_source and not srgb_destination:
        expected = ((0.5 + 0.055) / 1.055) ** 2.4
    elif not srgb_source and srgb_destination:
        expected = 1.055 * 0.5 ** (1.0 / 2.4) - 0.055
    np.testing.assert_allclose(decoded[:, :, :3], expected, atol=0.03)


@pytest.mark.parametrize("pixel_format", [spy.Bitmap.PixelFormat.y, spy.Bitmap.PixelFormat.ya])
def test_luminance(
    encoder: spy.BCEncoder, tmp_path: Path, pixel_format: spy.Bitmap.PixelFormat
) -> None:
    pixels = np.full((4, 4), 0.25, dtype=np.float32)
    if pixel_format == spy.Bitmap.PixelFormat.ya:
        pixels = np.stack([pixels, np.full_like(pixels, 0.75)], axis=-1)
    image = encoder.encode(spy.Bitmap(pixels, pixel_format=pixel_format), spy.Format.bc3_unorm)
    path = tmp_path / "luminance.dds"
    image.write_dds(path)
    decoded = np.array(spy.Bitmap(path)) / 255.0
    np.testing.assert_allclose(decoded[:, :, :3], 0.25, atol=0.03)
    np.testing.assert_allclose(
        decoded[:, :, 3], 0.75 if pixel_format == spy.Bitmap.PixelFormat.ya else 1, atol=0.03
    )


@pytest.mark.parametrize("dtype", [np.uint32, np.int32, np.uint64, np.int64])
@pytest.mark.parametrize(
    "pixel_format,channels",
    [
        (spy.Bitmap.PixelFormat.y, 1),
        (spy.Bitmap.PixelFormat.ya, 2),
        (spy.Bitmap.PixelFormat.rgb, 3),
        (spy.Bitmap.PixelFormat.rgba, 4),
        (spy.Bitmap.PixelFormat.r, 1),
        (spy.Bitmap.PixelFormat.rg, 2),
        (spy.Bitmap.PixelFormat.multi_channel, 4),
    ],
)
@pytest.mark.parametrize(
    "srgb_source,srgb_destination", [(False, False), (True, False), (False, True), (True, True)]
)
def test_wide_integer_bitmaps(
    encoder: spy.BCEncoder,
    dtype: type[np.generic],
    pixel_format: spy.Bitmap.PixelFormat,
    channels: int,
    srgb_source: bool,
    srgb_destination: bool,
) -> None:
    maximum = np.iinfo(dtype).max
    pixels = np.full((8, 8, channels), maximum // 2, dtype=dtype)
    if channels in (2, 4):
        pixels[:, :, -1] = maximum // 4
    bitmap = spy.Bitmap(pixels, pixel_format=pixel_format, srgb_gamma=srgb_source)
    reference = spy.Bitmap(
        (pixels.astype(np.float64) / maximum).astype(np.float32),
        pixel_format=pixel_format,
        srgb_gamma=srgb_source,
    )
    format = spy.Format.bc7_unorm_srgb if srgb_destination else spy.Format.bc7_unorm
    options = {"generate_mipmaps": True}
    actual = encoder.encode(bitmap, format, options)
    expected = encoder.encode(reference, format, options)
    assert actual.mip_count == expected.mip_count == 4
    for actual_mip, expected_mip in zip(actual.mip_levels, expected.mip_levels):
        np.testing.assert_array_equal(actual_mip.data, expected_mip.data)
    np.testing.assert_array_equal(np.array(bitmap).reshape(pixels.shape), pixels)
    assert bitmap.srgb_gamma == srgb_source


@pytest.mark.parametrize(
    "pixel_format,format,channels",
    [
        (spy.Bitmap.PixelFormat.r, spy.Format.bc4_unorm, 1),
        (spy.Bitmap.PixelFormat.rg, spy.Format.bc5_unorm, 2),
        (spy.Bitmap.PixelFormat.multi_channel, spy.Format.bc5_unorm, 2),
    ],
)
def test_raw_channels(
    encoder: spy.BCEncoder,
    tmp_path: Path,
    pixel_format: spy.Bitmap.PixelFormat,
    format: spy.Format,
    channels: int,
) -> None:
    pixels = np.empty((4, 4, channels), dtype=np.float32)
    pixels[:, :, 0] = 0.25
    if channels == 2:
        pixels[:, :, 1] = 0.75
    bitmap = spy.Bitmap(pixels, pixel_format=pixel_format, srgb_gamma=False)
    path = tmp_path / "raw.dds"
    encoder.encode(bitmap, format).write_dds(path)
    decoded = np.array(spy.Bitmap(path)).reshape(pixels.shape) / 255.0
    np.testing.assert_allclose(decoded, pixels, atol=0.02)


def test_result_lifetime_and_readonly() -> None:
    encoder = spy.BCEncoder(spy.BCEncoderBackend.software)
    image = encoder.encode(spy.Bitmap(np.ones((4, 4, 4), dtype=np.float32)), spy.Format.bc7_unorm)
    mip = image.mip_levels[0]
    data = mip.data
    expected = bytes(data)
    with pytest.raises(AttributeError):
        image.format = spy.Format.bc1_unorm
    with pytest.raises(AttributeError):
        mip.width = 1
    with pytest.raises(ValueError):
        data[0] = 0
    with pytest.raises(ValueError):
        data.setflags(write=True)
    image.mip_levels.clear()
    assert image.mip_count == 1
    del image, encoder, mip
    gc.collect()
    assert bytes(data) == expected
    assert memoryview(data).readonly


def test_invalid_inputs() -> None:
    encoder = spy.BCEncoder(spy.BCEncoderBackend.software)
    bitmap = spy.Bitmap(np.ones((4, 4, 4), dtype=np.float32))
    with pytest.raises(TypeError):
        encoder.encode(None, spy.Format.bc7_unorm)
    for format in [spy.Format.rgba8_unorm, spy.Format.bc4_snorm, spy.Format.bc6h_ufloat]:
        with pytest.raises(RuntimeError, match="cannot encode"):
            encoder.encode(bitmap, format)
    with pytest.raises(RuntimeError, match="channel_weights"):
        encoder.encode(bitmap, spy.Format.bc7_unorm, {"channel_weights": [0, 1, 1, 1]})
    with pytest.raises((TypeError, RuntimeError)):
        spy.BCEncodeOptions({"unknown_option": True})
    with pytest.raises((TypeError, RuntimeError)):
        spy.BCEncodeOptions({"mip_filter": spy.GaussianFilter()})
    with pytest.raises(RuntimeError, match="channel_count"):
        encoder.encode(spy.Bitmap(np.ones((4, 4, 5), dtype=np.float32)), spy.Format.bc7_unorm)


def test_shared_encoder_threads() -> None:
    encoder = spy.BCEncoder(spy.BCEncoderBackend.software)
    bitmap = spy.Bitmap(np.full((16, 16, 4), 0.5, dtype=np.float32))

    def encode(index: int) -> bytes:
        return bytes(encoder.encode(bitmap, spy.Format.bc7_unorm).mip_levels[0].data)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(encode, range(8)))
    assert all(data == results[0] for data in results)
