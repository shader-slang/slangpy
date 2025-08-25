# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np
import numpy.typing as npt
import enum
from dataclasses import dataclass

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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_texture_array(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    loader = TextureLoader(device)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
