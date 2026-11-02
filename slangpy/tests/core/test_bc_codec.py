# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import struct

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


@pytest.mark.parametrize(
    "filename,format,pixel_format,component_type",
    [
        (
            "bc1-unorm.dds",
            spy.Format.bc1_unorm,
            spy.Bitmap.PixelFormat.rgba,
            spy.DataStruct.Type.uint8,
        ),
        (
            "bc2-unorm.dds",
            spy.Format.bc2_unorm,
            spy.Bitmap.PixelFormat.rgba,
            spy.DataStruct.Type.uint8,
        ),
        (
            "bc3-unorm.dds",
            spy.Format.bc3_unorm,
            spy.Bitmap.PixelFormat.rgba,
            spy.DataStruct.Type.uint8,
        ),
        (
            "bc4-unorm.dds",
            spy.Format.bc4_unorm,
            spy.Bitmap.PixelFormat.r,
            spy.DataStruct.Type.uint8,
        ),
        (
            "bc5-unorm.dds",
            spy.Format.bc5_unorm,
            spy.Bitmap.PixelFormat.rg,
            spy.DataStruct.Type.uint8,
        ),
        (
            "bc6h-uf16.dds",
            spy.Format.bc6h_ufloat,
            spy.Bitmap.PixelFormat.rgb,
            spy.DataStruct.Type.float16,
        ),
        (
            "bc7-unorm-odd.dds",
            spy.Format.bc7_unorm,
            spy.Bitmap.PixelFormat.rgba,
            spy.DataStruct.Type.uint8,
        ),
        (
            "bc7-unorm-srgb.dds",
            spy.Format.bc7_unorm_srgb,
            spy.Bitmap.PixelFormat.rgba,
            spy.DataStruct.Type.uint8,
        ),
    ],
)
def test_load_and_decode_dds(
    tmp_path: Path,
    filename: str,
    format: spy.Format,
    pixel_format: spy.Bitmap.PixelFormat,
    component_type: spy.DataStruct.Type,
) -> None:
    path = spy.platform.project_directory() / "data" / "test_images" / "dds" / filename
    image = spy.BCCompressedImage(path)
    assert image.format == format
    base = image.decode()
    np.testing.assert_array_equal(np.array(base), np.array(spy.Bitmap(path)))
    assert (image.width, image.height) == (base.width, base.height)
    for level, mip in enumerate(image.mip_levels):
        bitmap = image.decode(level)
        assert (bitmap.width, bitmap.height) == (mip.width, mip.height)
        assert bitmap.pixel_format == pixel_format
        assert bitmap.component_type == component_type
        assert bitmap.srgb_gamma == (format == spy.Format.bc7_unorm_srgb)
    output = tmp_path / "copy.dds"
    image.write_dds(output)
    reloaded = spy.BCCompressedImage(path=str(output))
    assert reloaded.format == image.format
    assert reloaded.mip_count == image.mip_count
    for original, copied in zip(image.mip_levels, reloaded.mip_levels):
        np.testing.assert_array_equal(original.data, copied.data)


@pytest.mark.parametrize(
    "format,channels,dxgi_format",
    [(spy.Format.bc4_snorm, 1, 81), (spy.Format.bc5_snorm, 2, 84)],
)
def test_decode_signed_dds(
    tmp_path: Path, format: spy.Format, channels: int, dxgi_format: int
) -> None:
    # Use a valid single-block DX10 DDS header, then supply signed BC4/5 blocks.
    encoder = spy.BCEncoder(spy.BCEncoderBackend.software)
    unorm_format = spy.Format.bc4_unorm if channels == 1 else spy.Format.bc5_unorm
    path = tmp_path / "signed.dds"
    encoder.encode(spy.Bitmap(np.zeros((4, 4, 4), dtype=np.float32)), unorm_format).write_dds(path)
    data = bytearray(path.read_bytes())
    struct.pack_into("<I", data, 128, dxgi_format)
    # Endpoint -127, all indices select the first endpoint.
    data[148:] = bytes([129, 0, 0, 0, 0, 0, 0, 0]) * channels
    path.write_bytes(data)
    image = spy.BCCompressedImage(path)
    assert image.format == format
    bitmap = image.decode()
    assert bitmap.component_type == spy.DataStruct.Type.int8
    assert bitmap.channel_count == channels
    assert not bitmap.srgb_gamma
    np.testing.assert_array_equal(np.array(bitmap), -127)


@pytest.mark.parametrize(
    "format",
    [
        spy.Format.bc1_unorm,
        spy.Format.bc2_unorm,
        spy.Format.bc3_unorm,
        spy.Format.bc5_unorm,
        spy.Format.bc7_unorm,
    ],
)
def test_convert_authored_normal_mips(tmp_path: Path, format: spy.Format) -> None:
    encoder = spy.BCEncoder(spy.BCEncoderBackend.software)
    # A partial, odd-sized chain with independently authored colors, not a downsampled base.
    dimensions = [(13, 7), (6, 3), (3, 1)]
    mips = []
    for level, (width, height) in enumerate(dimensions):
        pixels = np.full((height, width, 4), [64, 48 + 64 * level, 192, 255], dtype=np.uint8)
        mips.append(encoder.encode(spy.Bitmap(pixels, srgb_gamma=False), format).mip_levels[0])
    source_path = tmp_path / "source.dds"
    spy.BCCompressedImage(format, mips).write_dds(source_path)
    source = spy.BCCompressedImage(source_path)
    assert source.format == format
    assert source.mip_count == len(dimensions)
    expected = []
    encoded_mips = []
    for level in range(source.mip_count):
        bitmap = source.decode(level)
        pixels = np.asarray(bitmap)
        pixels[:, :, 1] = 255 - pixels[:, :, 1]
        expected.append(pixels.copy())
        encoded_mips.append(encoder.encode(bitmap, source.format).mip_levels[0])
    output_path = tmp_path / "converted.dds"
    spy.BCCompressedImage(source.format, encoded_mips).write_dds(output_path)
    result = spy.BCCompressedImage(output_path)
    assert result.format == format
    assert result.mip_count == len(dimensions)
    for level, ((width, height), pixels) in enumerate(zip(dimensions, expected)):
        decoded = result.decode(level)
        assert (decoded.width, decoded.height) == (width, height)
        assert not decoded.srgb_gamma
        np.testing.assert_allclose(np.array(decoded), pixels, atol=8)
        # Editing the decoded bitmap must not change the compressed source.
        np.testing.assert_allclose(np.array(source.decode(level))[:, :, 1], 48 + 64 * level, atol=8)


def test_constructed_image_and_decoded_bitmap_lifetime() -> None:
    encoder = spy.BCEncoder(spy.BCEncoderBackend.software)
    source = encoder.encode(
        spy.Bitmap(np.full((4, 4, 4), 0.5, dtype=np.float32)), spy.Format.bc7_unorm
    )
    mips = source.mip_levels
    image = spy.BCCompressedImage(format=source.format, mip_levels=mips)
    expected = bytes(mips[0].data)
    mips.clear()
    del source, encoder
    gc.collect()
    assert bytes(image.mip_levels[0].data) == expected
    bitmap = image.decode()
    del image
    gc.collect()
    np.testing.assert_allclose(np.array(bitmap), 128, atol=2)


def test_dds_editing_invalid_inputs(tmp_path: Path) -> None:
    encoder = spy.BCEncoder(spy.BCEncoderBackend.software)
    image = encoder.encode(spy.Bitmap(np.ones((8, 8, 4), dtype=np.float32)), spy.Format.bc7_unorm)
    with pytest.raises(RuntimeError, match="out of range"):
        image.decode(image.mip_count)
    with pytest.raises(TypeError):
        image.decode(-1)
    with pytest.raises(RuntimeError, match="no mip levels"):
        spy.BCCompressedImage(image.format, [])
    with pytest.raises(RuntimeError, match="expected a BC format"):
        spy.BCCompressedImage(spy.Format.rgba8_unorm, image.mip_levels)
    with pytest.raises(RuntimeError, match="invalid data size"):
        spy.BCCompressedImage(spy.Format.bc1_unorm, image.mip_levels)
    with pytest.raises(RuntimeError, match="invalid dimensions"):
        spy.BCCompressedImage(image.format, image.mip_levels * 2)
    with pytest.raises(TypeError):
        spy.BCCompressedImage(image.format, [None])
    with pytest.raises(RuntimeError):
        spy.BCCompressedImage(tmp_path / "missing.dds")
    dds_dir = spy.platform.project_directory() / "data" / "test_images" / "dds"
    for filename in ["bc7-unorm-broken.dds", "bc7-unorm-3d.dds"]:
        with pytest.raises(RuntimeError):
            spy.BCCompressedImage(dds_dir / filename)
