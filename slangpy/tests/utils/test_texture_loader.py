# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np
import numpy.typing as npt
import enum
from dataclasses import dataclass
from pathlib import Path

import slangpy as spy
from slangpy import TextureLoader, Bitmap, Format, DataStruct, FormatSupport
from slangpy.testing import helpers


PixelFormat = Bitmap.PixelFormat
ComponentType = DataStruct.Type


class Flags(enum.Flag):
    none = 0
    load_as_normalized = 1
    load_as_srgb = 2
    extend_alpha = 4


@dataclass
class FormatEntry:
    pixel_format: PixelFormat
    component_type: ComponentType
    format: Format
    flags: Flags


# fmt: off
FORMATS = [
    # PixelFormat.y
    FormatEntry(PixelFormat.y, ComponentType.uint8, Format.r8_unorm, Flags.load_as_normalized),
    FormatEntry(PixelFormat.y, ComponentType.uint8, Format.r8_uint, Flags.load_as_srgb),
    # PixelFormat.r
    FormatEntry(PixelFormat.r, ComponentType.int8, Format.r8_sint, Flags.none),
    FormatEntry(PixelFormat.r, ComponentType.int8, Format.r8_snorm, Flags.load_as_normalized),
    FormatEntry(PixelFormat.r, ComponentType.int16, Format.r16_sint, Flags.none),
    FormatEntry(PixelFormat.r, ComponentType.int16, Format.r16_snorm, Flags.load_as_normalized),
    FormatEntry(PixelFormat.r, ComponentType.int32, Format.r32_sint, Flags.none),
    FormatEntry(PixelFormat.r, ComponentType.uint8, Format.r8_uint, Flags.none),
    FormatEntry(PixelFormat.r, ComponentType.uint8, Format.r8_unorm, Flags.load_as_normalized),
    FormatEntry(PixelFormat.r, ComponentType.uint16, Format.r16_uint, Flags.none),
    FormatEntry(PixelFormat.r, ComponentType.uint16, Format.r16_unorm, Flags.load_as_normalized),
    FormatEntry(PixelFormat.r, ComponentType.uint32, Format.r32_uint, Flags.none),
    FormatEntry(PixelFormat.r, ComponentType.float16, Format.r16_float, Flags.none),
    FormatEntry(PixelFormat.r, ComponentType.float32, Format.r32_float, Flags.none),
    # PixelFormat.rg
    FormatEntry(PixelFormat.rg, ComponentType.int8, Format.rg8_sint, Flags.none),
    FormatEntry(PixelFormat.rg, ComponentType.int8, Format.rg8_snorm, Flags.load_as_normalized),
    FormatEntry(PixelFormat.rg, ComponentType.int16, Format.rg16_sint, Flags.none),
    FormatEntry(PixelFormat.rg, ComponentType.int16, Format.rg16_snorm, Flags.load_as_normalized),
    FormatEntry(PixelFormat.rg, ComponentType.int32, Format.rg32_sint, Flags.none),
    FormatEntry(PixelFormat.rg, ComponentType.uint8, Format.rg8_uint, Flags.none),
    FormatEntry(PixelFormat.rg, ComponentType.uint8, Format.rg8_unorm, Flags.load_as_normalized),
    FormatEntry(PixelFormat.rg, ComponentType.uint16, Format.rg16_uint, Flags.none),
    FormatEntry(PixelFormat.rg, ComponentType.uint16, Format.rg16_unorm, Flags.load_as_normalized),
    FormatEntry(PixelFormat.rg, ComponentType.uint32, Format.rg32_uint, Flags.none),
    FormatEntry(PixelFormat.rg, ComponentType.float16, Format.rg16_float, Flags.none),
    FormatEntry(PixelFormat.rg, ComponentType.float32, Format.rg32_float, Flags.none),
    # PixelFormat.rgb
    # TODO(slang-rhi) fails on vulkan
    # FormatEntry(PixelFormat.rgb, ComponentType.int32, Format.rgb32_sint, Flags.none),
    # FormatEntry(PixelFormat.rgb, ComponentType.uint32, Format.rgb32_uint, Flags.none),
    # FormatEntry(PixelFormat.rgb, ComponentType.float32, Format.rgb32_float, Flags.none),
    # PixelFormat.rgba
    FormatEntry(PixelFormat.rgba, ComponentType.int8, Format.rgba8_sint, Flags.none),
    FormatEntry(PixelFormat.rgba, ComponentType.int8, Format.rgba8_snorm, Flags.load_as_normalized),
    FormatEntry(PixelFormat.rgba, ComponentType.int16, Format.rgba16_sint, Flags.none),
    FormatEntry(PixelFormat.rgba, ComponentType.int16, Format.rgba16_snorm, Flags.load_as_normalized),
    FormatEntry(PixelFormat.rgba, ComponentType.int32, Format.rgba32_sint, Flags.none),
    FormatEntry(PixelFormat.rgba, ComponentType.uint8, Format.rgba8_uint, Flags.none),
    FormatEntry(PixelFormat.rgba, ComponentType.uint8, Format.rgba8_unorm, Flags.load_as_normalized),
    FormatEntry(PixelFormat.rgba, ComponentType.uint16, Format.rgba16_uint, Flags.none),
    FormatEntry(PixelFormat.rgba, ComponentType.uint16, Format.rgba16_unorm, Flags.load_as_normalized),
    FormatEntry(PixelFormat.rgba, ComponentType.uint32, Format.rgba32_uint, Flags.none),
    FormatEntry(PixelFormat.rgba, ComponentType.float16, Format.rgba16_float, Flags.none),
    FormatEntry(PixelFormat.rgba, ComponentType.float32, Format.rgba32_float, Flags.none),
    # sRGB handling
    FormatEntry(PixelFormat.rgba, ComponentType.uint8, Format.rgba8_unorm_srgb, Flags.load_as_srgb),
    # alpha extension
    FormatEntry(PixelFormat.rgb, ComponentType.uint8, Format.rgba8_uint, Flags.extend_alpha),
    FormatEntry(PixelFormat.rgb, ComponentType.uint8, Format.rgba8_unorm, Flags.load_as_normalized | Flags.extend_alpha),
    FormatEntry(PixelFormat.rgb, ComponentType.uint8, Format.rgba8_unorm_srgb, Flags.load_as_srgb | Flags.extend_alpha),
    # ya handling
    FormatEntry(PixelFormat.ya, ComponentType.int8, Format.rgba8_sint, Flags.none),
]
# fmt: on

PIXEL_FORMAT_TO_CHANNELS = {
    PixelFormat.y: 1,
    PixelFormat.ya: 2,
    PixelFormat.r: 1,
    PixelFormat.rg: 2,
    PixelFormat.rgb: 3,
    PixelFormat.rgba: 4,
}

COMPONENT_TYPE_TO_DTYPE = {
    ComponentType.uint8: np.uint8,
    ComponentType.uint16: np.uint16,
    ComponentType.uint32: np.uint32,
    ComponentType.uint64: np.uint64,
    ComponentType.int8: np.int8,
    ComponentType.int16: np.int16,
    ComponentType.int32: np.int32,
    ComponentType.int64: np.int64,
    ComponentType.float16: np.float16,
    ComponentType.float32: np.float32,
    ComponentType.float64: np.float64,
}

TEST_IMAGE_DIR = spy.platform.project_directory() / "data" / "test_images"
TEST_DDS_DIR = TEST_IMAGE_DIR / "dds"

TEST_BITMAP_FILES = [
    "albert.jpg",
    "monalisa.jpg",
]

TEST_DDS_FILES = [
    "bc1-unorm.dds",
    # "bc1-unorm-srgb.dds",
    # "bc2-unorm.dds",
    # "bc2-unorm-srgb.dds",
    # "bc2-unorm-srgb-tiny.dds",
    # "bc3-unorm.dds",
    # "bc3-unorm-alpha.dds",
    # "bc3-unorm-alpha-tiny.dds",
    # "bc3-unorm-srgb.dds",
    # "bc3-unorm-srgb-odd.dds",
    # "bc3-unorm-srgb-tiny.dds",
    # "bc4-unorm.dds",
    # "bc5-unorm.dds",
    # "bc5-unorm-tiny.dds",
    # "bc6h-uf16.dds",
    # "bc7-unorm.dds",
    # "bc7-unorm-odd.dds",
    # "bc7-unorm-srgb.dds",
    # "bc7-unorm-tiny.dds",
]


def create_test_array(
    width: int,
    height: int,
    channels: int,
    dtype: npt.DTypeLike,  # type: ignore
    type_range: tuple[float, float],
):
    img = np.zeros((height, width, channels), dtype)
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                value = (i + j + k) / (width + height + channels)
                value = type_range[0] + value * (type_range[1] - type_range[0])
                img[i, j, k] = value
    if channels == 1:
        img = img.reshape((height, width))
    return img


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_texture_from_bitmap(device_type: spy.DeviceType, format: FormatEntry):
    device = helpers.get_device(type=device_type)

    # Check if format is supported
    format_support = device.get_format_support(format.format)
    if not FormatSupport.shader_load in format_support:
        pytest.skip("Format not supported as shader resource")

    if device_type == spy.DeviceType.metal and format.pixel_format == PixelFormat.rgb:
        pytest.skip(
            "Metal does not support rgb format: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf"
        )

    # Create empty bitmap
    bitmap = Bitmap(
        pixel_format=format.pixel_format,
        component_type=format.component_type,
        width=100,
        height=50,
    )

    # Create test image
    channels = PIXEL_FORMAT_TO_CHANNELS[format.pixel_format]
    dtype = COMPONENT_TYPE_TO_DTYPE[format.component_type]
    if DataStruct.is_float(format.component_type):
        type_range = (0.0, 1.0)
    else:
        type_range = DataStruct.type_range(format.component_type)
    image = create_test_array(bitmap.width, bitmap.height, channels, dtype, type_range)

    # Fill bitmap with test image
    a = np.array(bitmap, copy=False)
    a[:] = image

    # Extend alpha channel if necessary
    if format.flags & Flags.extend_alpha:
        image = np.concatenate(
            (
                image,
                np.full((bitmap.height, bitmap.width, 1), 255, dtype=dtype),
            ),
            axis=2,
        )

    if format.pixel_format == PixelFormat.ya:
        image = np.concatenate(
            (
                image[:, :, :1],  # y
                image[:, :, :1],  # y
                image[:, :, :1],  # y
                image[:, :, 1:],  # a
            ),
            axis=2,
        )

    # Load the bitmap as a texture
    loader = TextureLoader(device)
    texture = loader.load_texture(
        bitmap=bitmap,
        options={
            "load_as_normalized": bool(format.flags & Flags.load_as_normalized),
            "load_as_srgb": bool(format.flags & Flags.load_as_srgb),
        },
    )

    assert texture.format == format.format
    assert texture.width == bitmap.width
    assert texture.height == bitmap.height
    assert texture.mip_count == 1

    data = texture.to_numpy()
    assert data.shape == image.shape
    assert np.allclose(data, image, atol=1e-6)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("filename", TEST_BITMAP_FILES)
def test_load_texture_from_bitmap_file(device_type: spy.DeviceType, filename: str):
    device = helpers.get_device(type=device_type)

    path = TEST_IMAGE_DIR / filename

    loader = TextureLoader(device)
    texture = loader.load_texture(path)

    bitmap = texture.to_bitmap()
    bitmap_ref = Bitmap(path).convert(pixel_format=Bitmap.PixelFormat.rgba)

    assert np.all(np.array(bitmap, copy=False) == np.array(bitmap_ref, copy=False))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("load_as_srgb", [False, True])
@pytest.mark.parametrize("load_as_normalized", [False, True])
def test_luminance_handling_is_explicit(
    tmp_path: Path,
    device_type: spy.DeviceType,
    load_as_srgb: bool,
    load_as_normalized: bool,
) -> None:
    device = helpers.get_device(type=device_type)
    values = np.array([[64, 192]], dtype=np.uint8)
    bitmap = Bitmap(values, pixel_format=PixelFormat.y, srgb_gamma=True)
    path = tmp_path / "luminance.png"
    bitmap.write(path)
    loader = TextureLoader(device)
    assert TextureLoader.Options().y_handling == spy.YHandling.preserve_as_r
    rgba = np.concatenate(
        [np.repeat(values[:, :, None], 3, axis=2), np.full((1, 2, 1), 255, dtype=np.uint8)],
        axis=2,
    )
    for source in (bitmap, path):
        # Default sRGB loading alone must keep the historical single-channel texture.
        default = loader.load_texture(source)
        assert default.format == Format.r8_unorm
        np.testing.assert_array_equal(default.to_numpy(), values)
        for handling in (None, spy.YHandling.preserve_as_r, spy.YHandling.expand_to_rgba):
            options = TextureLoader.Options(
                {"load_as_srgb": load_as_srgb, "load_as_normalized": load_as_normalized}
            )
            if handling is not None:
                options.y_handling = handling
            texture = loader.load_texture(source, options=options)
            if handling == spy.YHandling.expand_to_rgba:
                expected_format = (
                    Format.rgba8_unorm_srgb
                    if load_as_srgb
                    else Format.rgba8_unorm if load_as_normalized else Format.rgba8_uint
                )
                expected = rgba
            else:
                expected_format = Format.r8_unorm if load_as_normalized else Format.r8_uint
                expected = values
            assert texture.format == expected_format
            assert texture.mip_count == 1
            np.testing.assert_array_equal(texture.to_numpy(), expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("one_bit", [False, True])
def test_expand_luminance_to_rgba(
    tmp_path: Path, device_type: spy.DeviceType, one_bit: bool
) -> None:
    device = helpers.get_device(type=device_type)
    path = tmp_path / "luminance.png"
    if one_bit:
        # A 2x1, one-bit grayscale PNG containing black and white pixels.
        path.write_bytes(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d4948445200000002000000010100000000dc594227"
                "0000000a49444154789c63700000004200412937f4ef0000000049454e44ae426082"
            )
        )
        values = np.array([[0, 255]], dtype=np.uint8)
    else:
        values = np.array([[64, 192]], dtype=np.uint8)
        Bitmap(values, pixel_format=PixelFormat.y, srgb_gamma=True).write(path)

    loader = TextureLoader(device)
    options = TextureLoader.Options(
        {"load_as_srgb": False, "y_handling": spy.YHandling.expand_to_rgba}
    )
    assert options.y_handling == spy.YHandling.expand_to_rgba
    assert TextureLoader.Options().y_handling == spy.YHandling.preserve_as_r
    expected = np.concatenate(
        [np.repeat(values[:, :, None], 3, axis=2), np.full((1, 2, 1), 255, dtype=np.uint8)],
        axis=2,
    )
    for source in (path, Bitmap(path)):
        texture = loader.load_texture(source, options=options)
        assert texture.format == Format.rgba8_unorm
        assert texture.mip_count == 1
        np.testing.assert_array_equal(texture.to_numpy(), expected)
        scalar = loader.load_texture(source, options={"load_as_srgb": False})
        assert scalar.format == Format.r8_unorm
        np.testing.assert_array_equal(scalar.to_numpy(), values)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_luminance_expansion_preserves_components(
    device_type: spy.DeviceType, dtype: npt.DTypeLike
) -> None:
    device = helpers.get_device(type=device_type)
    values = np.array([[16384, 49152]] if dtype == np.uint16 else [[0.25, 0.75]], dtype=dtype)
    bitmap = Bitmap(values, pixel_format=PixelFormat.y, srgb_gamma=False)
    texture = TextureLoader(device).load_texture(
        bitmap, options={"load_as_srgb": False, "y_handling": spy.YHandling.expand_to_rgba}
    )
    assert texture.format == (Format.rgba16_unorm if dtype == np.uint16 else Format.rgba32_float)
    actual = texture.to_numpy()
    np.testing.assert_array_equal(actual[:, :, :3], np.repeat(values[:, :, None], 3, axis=2))
    np.testing.assert_array_equal(actual[:, :, 3], 65535 if dtype == np.uint16 else 1.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "pixel_format",
    [PixelFormat.r, PixelFormat.rg, PixelFormat.rgb, PixelFormat.rgba, PixelFormat.ya],
)
def test_y_handling_preserves_other_channels(
    device_type: spy.DeviceType, pixel_format: PixelFormat
) -> None:
    device = helpers.get_device(type=device_type)
    channels = PIXEL_FORMAT_TO_CHANNELS[pixel_format]
    values = np.arange(2 * channels, dtype=np.uint8).reshape(1, 2, channels)
    if channels == 1:
        values = values[:, :, 0]
    bitmap = Bitmap(values, pixel_format=pixel_format, srgb_gamma=False)
    loader = TextureLoader(device)
    for ya_handling in (spy.YAHandling.expand_to_rgba, spy.YAHandling.preserve_as_rg):
        options = {"load_as_srgb": False, "ya_handling": ya_handling}
        original = loader.load_texture(bitmap, options=options)
        promoted = loader.load_texture(
            bitmap, options={**options, "y_handling": spy.YHandling.expand_to_rgba}
        )
        assert promoted.format == original.format
        np.testing.assert_array_equal(promoted.to_numpy(), original.to_numpy())


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_rgb_float_texture_with_generated_mips_extends_to_rgba(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    format_support = device.get_format_support(Format.rgba32_float)
    if (
        FormatSupport.texture not in format_support
        or FormatSupport.shader_load not in format_support
    ):
        pytest.skip("RGBA32 float format not supported")
    if spy.Feature.rasterization in device.features:
        if FormatSupport.render_target not in format_support:
            pytest.skip("RGBA32 float render target format not supported")
    elif FormatSupport.shader_uav_store not in format_support:
        pytest.skip("RGBA32 float UAV format not supported")

    bitmap = Bitmap(
        pixel_format=PixelFormat.rgb,
        component_type=ComponentType.float32,
        width=16,
        height=8,
    )
    image = create_test_array(bitmap.width, bitmap.height, 3, np.float32, (0.0, 1.0))
    np.array(bitmap, copy=False)[:] = image

    loader = TextureLoader(device)
    options = TextureLoader.Options()
    options.generate_mips = True
    texture = loader.load_texture(bitmap=bitmap, options=options)

    assert texture.format == Format.rgba32_float
    assert texture.width == bitmap.width
    assert texture.height == bitmap.height
    assert texture.mip_count > 1

    data = texture.to_numpy()
    assert data.shape == (bitmap.height, bitmap.width, 4)
    assert np.allclose(data[:, :, :3], image, atol=1e-6)
    assert np.allclose(data[:, :, 3], 1.0, atol=1e-6)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("filename", TEST_DDS_FILES)
def test_load_texture_from_dds_file(device_type: spy.DeviceType, filename: str):
    device: spy.Device = helpers.get_device(type=device_type)

    path = TEST_DDS_DIR / filename

    loader = TextureLoader(device)
    texture = loader.load_texture(path)

    data = {}
    for subresource in range(texture.subresource_count):
        layer = subresource // texture.mip_count
        mip = subresource % texture.mip_count
        subresource_data = texture.to_numpy(layer, mip)
        data[f"subresource_{subresource}"] = subresource_data

    # Uncomment this to dump reference data.
    # np.savez_compressed(path.with_name(path.stem + "-ref.npz"), **data)

    ref_data = np.load(path.with_name(path.stem + "-ref.npz"))
    for subresource in range(texture.subresource_count):
        key = f"subresource_{subresource}"
        sr_data = data[key]
        sr_ref_data = ref_data[key]
        assert sr_data.shape == sr_ref_data.shape
        assert np.all(sr_data == sr_ref_data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_textures(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    loader = TextureLoader(device)
    paths = [TEST_IMAGE_DIR / f for f in TEST_BITMAP_FILES]
    textures = loader.load_textures(paths)
    assert len(textures) == 2


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_texture_array(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    loader = TextureLoader(device)
    paths = [TEST_IMAGE_DIR / f for f in TEST_BITMAP_FILES]
    texture = loader.load_texture_array(paths)
    assert texture.array_length == 2


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ya_handling_default_expands_to_rgba(device_type: spy.DeviceType):
    """Test that default YA handling expands to RGBA (Y to RGB, A to A)."""
    device = helpers.get_device(type=device_type)

    bitmap = Bitmap(
        pixel_format=PixelFormat.ya,
        component_type=ComponentType.uint8,
        width=32,
        height=16,
    )
    a = np.array(bitmap, copy=False)
    a[:, :, 0] = 100  # Y (luminance)
    a[:, :, 1] = 200  # A (alpha)

    loader = TextureLoader(device)
    options = TextureLoader.Options()
    options.load_as_normalized = True
    options.load_as_srgb = False
    texture = loader.load_texture(bitmap=bitmap, options=options)

    assert texture.format == Format.rgba8_unorm

    data = texture.to_numpy()
    assert data.shape == (16, 32, 4)
    assert np.allclose(data[:, :, 0], 100, atol=1)  # R = Y
    assert np.allclose(data[:, :, 1], 100, atol=1)  # G = Y
    assert np.allclose(data[:, :, 2], 100, atol=1)  # B = Y
    assert np.allclose(data[:, :, 3], 200, atol=1)  # A = A


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ya_handling_preserve_as_rg(device_type: spy.DeviceType):
    """Test that preserve_as_rg keeps YA as 2-channel RG (Y to R, A to G)."""
    device = helpers.get_device(type=device_type)

    format_support = device.get_format_support(Format.rg8_unorm)
    if not FormatSupport.shader_load in format_support:
        pytest.skip("RG format not supported")

    bitmap = Bitmap(
        pixel_format=PixelFormat.ya,
        component_type=ComponentType.uint8,
        width=48,
        height=24,
    )
    a = np.array(bitmap, copy=False)
    a[:, :, 0] = 100  # Y (luminance)
    a[:, :, 1] = 200  # A (alpha)

    loader = TextureLoader(device)
    options = TextureLoader.Options()
    options.load_as_normalized = True
    options.load_as_srgb = False
    options.ya_handling = spy.YAHandling.preserve_as_rg
    texture = loader.load_texture(bitmap=bitmap, options=options)

    assert texture.format == Format.rg8_unorm

    data = texture.to_numpy()
    assert data.shape == (24, 48, 2)
    assert np.allclose(data[:, :, 0], 100, atol=1)  # R = Y
    assert np.allclose(data[:, :, 1], 200, atol=1)  # G = A


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ya_handling_preserve_as_rg_float32(device_type: spy.DeviceType):
    """Test that preserve_as_rg works with float32 component type."""
    device = helpers.get_device(type=device_type)

    format_support = device.get_format_support(Format.rg32_float)
    if not FormatSupport.shader_load in format_support:
        pytest.skip("RG32 float format not supported")

    bitmap = Bitmap(
        pixel_format=PixelFormat.ya,
        component_type=ComponentType.float32,
        width=64,
        height=32,
    )
    a = np.array(bitmap, copy=False)
    a[:, :, 0] = 0.5  # Y (luminance)
    a[:, :, 1] = 0.8  # A (alpha)

    loader = TextureLoader(device)
    options = TextureLoader.Options()
    options.ya_handling = spy.YAHandling.preserve_as_rg
    texture = loader.load_texture(bitmap=bitmap, options=options)

    assert texture.format == Format.rg32_float

    data = texture.to_numpy()
    assert data.shape == (32, 64, 2)
    assert np.allclose(data[:, :, 0], 0.5, atol=0.01)  # R = Y
    assert np.allclose(data[:, :, 1], 0.8, atol=0.01)  # G = A


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
