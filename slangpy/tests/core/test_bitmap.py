# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import struct
from typing import Any, Optional, Sequence
import zlib
import pytest
from slangpy import Bitmap, DataStruct
import numpy as np
import numpy.typing as npt

PIXEL_FORMAT_TO_CHANNELS = {
    Bitmap.PixelFormat.y: 1,
    Bitmap.PixelFormat.ya: 2,
    Bitmap.PixelFormat.rgb: 3,
    Bitmap.PixelFormat.rgba: 4,
    Bitmap.PixelFormat.multi_channel: 8,
}

COMPONENT_TYPE_TO_DTYPE = {
    Bitmap.ComponentType.uint8: np.uint8,
    Bitmap.ComponentType.uint16: np.uint16,
    Bitmap.ComponentType.uint32: np.uint32,
    Bitmap.ComponentType.uint64: np.uint64,
    Bitmap.ComponentType.int8: np.int8,
    Bitmap.ComponentType.int16: np.int16,
    Bitmap.ComponentType.int32: np.int32,
    Bitmap.ComponentType.int64: np.int64,
    Bitmap.ComponentType.float16: np.float16,
    Bitmap.ComponentType.float32: np.float32,
    Bitmap.ComponentType.float64: np.float64,
}


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


def create_test_image(
    width: int,
    height: int,
    pixel_format: Bitmap.PixelFormat,
    component_type: DataStruct.Type,
):
    channels = PIXEL_FORMAT_TO_CHANNELS[pixel_format]
    dtype = COMPONENT_TYPE_TO_DTYPE[component_type]
    if DataStruct.is_float(component_type):
        type_range = (0.0, 1.0)
    else:
        type_range = DataStruct.type_range(component_type)
    return create_test_array(width, height, channels, dtype, type_range)


def write_read_test(
    directory: Path,
    ext: str,
    width: int,
    height: int,
    pixel_format: Bitmap.PixelFormat,
    component_type: DataStruct.Type,
    quality: Optional[int] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
):
    path = directory / f"test_{width}x{height}_{pixel_format}_{component_type}.{ext}"

    img = create_test_image(width, height, pixel_format, component_type)

    b1 = Bitmap(img)
    if ext in ("jpg", "bmp", "tga"):
        # These writers do not store transfer metadata; match the reader's fallback.
        b1.srgb_gamma = component_type == Bitmap.ComponentType.uint8 and pixel_format in (
            Bitmap.PixelFormat.rgb,
            Bitmap.PixelFormat.rgba,
        )
    b1.write(path, quality=quality if quality else -1)

    b2 = Bitmap(path)

    assert b1.pixel_format == b2.pixel_format
    assert b1.component_type == b2.component_type
    assert b1.width == b2.width
    assert b2.height == b2.height
    assert b1.channel_count == b2.channel_count
    assert b1.srgb_gamma == b2.srgb_gamma

    a1 = np.array(b1, copy=False)
    a2 = np.array(b2, copy=False)

    if rtol:
        assert np.allclose(a1, a2, rtol=rtol)
    elif atol:
        assert np.allclose(a1, a2, atol=atol)
    else:
        assert np.all(a1 == a2)
        assert b1 == b2


def test_bitmap_empty():
    b = Bitmap(
        pixel_format=Bitmap.PixelFormat.y,
        component_type=Bitmap.ComponentType.uint8,
        width=0,
        height=0,
    )
    assert b.width == 0
    assert b.height == 0
    assert b.pixel_count == 0
    assert b.buffer_size == 0
    assert b.empty()


def test_bitmap_clear():
    img = create_test_image(100, 100, Bitmap.PixelFormat.rgb, Bitmap.ComponentType.float32)
    b = Bitmap(img)
    a = np.array(b, copy=False)
    assert np.any(a == img)
    b.clear()
    assert np.all(a == 0)


def test_bitmap_vflip():
    img = create_test_image(50, 100, Bitmap.PixelFormat.y, Bitmap.ComponentType.float32)
    b = Bitmap(img)
    a = np.array(b, copy=False)
    assert np.all(a == img)
    b.vflip()
    assert np.all(a == np.flip(img, 0))


def test_bitmap_from_non_contiguous_array():
    a = create_test_array(100, 100, 4, np.float32, (0.0, 1.0))
    # Strided 2-dimensional array
    b = Bitmap(a[:, :, 0].reshape((100, 100)))
    assert np.all(np.array(b, copy=False) == a[:, :, 0])
    # Strided 3-dimensional array with 1 channel
    b = Bitmap(a[:, :, 0])
    assert np.all(np.array(b, copy=False) == a[:, :, 0])
    # Strided 3-dimensional array with 3 channels
    b = Bitmap(a[:, :, 0:2])
    assert np.all(np.array(b, copy=False) == a[:, :, 0:2])


EXR_LAYOUTS = [
    (5, 10, Bitmap.PixelFormat.y, Bitmap.ComponentType.float16),
    (10, 20, Bitmap.PixelFormat.ya, Bitmap.ComponentType.float16),
    (50, 100, Bitmap.PixelFormat.rgb, Bitmap.ComponentType.float16),
    (100, 200, Bitmap.PixelFormat.rgba, Bitmap.ComponentType.float16),
    (50, 50, Bitmap.PixelFormat.multi_channel, Bitmap.ComponentType.float16),
    (5, 10, Bitmap.PixelFormat.y, Bitmap.ComponentType.float32),
    (10, 20, Bitmap.PixelFormat.ya, Bitmap.ComponentType.float32),
    (50, 100, Bitmap.PixelFormat.rgb, Bitmap.ComponentType.float32),
    (100, 200, Bitmap.PixelFormat.rgba, Bitmap.ComponentType.float32),
    (50, 50, Bitmap.PixelFormat.multi_channel, Bitmap.ComponentType.float32),
    (5, 10, Bitmap.PixelFormat.y, Bitmap.ComponentType.uint32),
    (10, 20, Bitmap.PixelFormat.ya, Bitmap.ComponentType.uint32),
    (50, 100, Bitmap.PixelFormat.rgb, Bitmap.ComponentType.uint32),
    (100, 200, Bitmap.PixelFormat.rgba, Bitmap.ComponentType.uint32),
    (50, 50, Bitmap.PixelFormat.multi_channel, Bitmap.ComponentType.uint32),
]


@pytest.mark.parametrize("layout", EXR_LAYOUTS)
def test_exr_io(tmp_path: Path, layout: Sequence[Any]):
    extra = layout[4] if len(layout) > 4 else {}
    write_read_test(tmp_path, "exr", layout[0], layout[1], layout[2], layout[3], **extra)


BMP_LAYOUTS = [
    (50, 100, Bitmap.PixelFormat.rgb, Bitmap.ComponentType.uint8),
    (100, 200, Bitmap.PixelFormat.rgba, Bitmap.ComponentType.uint8),
]


@pytest.mark.parametrize("layout", BMP_LAYOUTS)
def test_bmp_io(tmp_path: Path, layout: Sequence[Any]):
    extra = layout[4] if len(layout) > 4 else {}
    write_read_test(tmp_path, "bmp", layout[0], layout[1], layout[2], layout[3], **extra)


TGA_LAYOUTS = [
    (5, 10, Bitmap.PixelFormat.y, Bitmap.ComponentType.uint8),
    (50, 100, Bitmap.PixelFormat.rgb, Bitmap.ComponentType.uint8),
    (100, 200, Bitmap.PixelFormat.rgba, Bitmap.ComponentType.uint8),
]


@pytest.mark.parametrize("layout", TGA_LAYOUTS)
def test_tga_io(tmp_path: Path, layout: Sequence[Any]):
    extra = layout[4] if len(layout) > 4 else {}
    write_read_test(tmp_path, "tga", layout[0], layout[1], layout[2], layout[3], **extra)


PNG_LAYOUTS = [
    (1, 2, Bitmap.PixelFormat.y, Bitmap.ComponentType.uint8),
    (5, 10, Bitmap.PixelFormat.ya, Bitmap.ComponentType.uint8),
    (50, 100, Bitmap.PixelFormat.rgb, Bitmap.ComponentType.uint8),
    (100, 200, Bitmap.PixelFormat.rgba, Bitmap.ComponentType.uint8),
    (1, 2, Bitmap.PixelFormat.y, Bitmap.ComponentType.uint16),
    (5, 10, Bitmap.PixelFormat.ya, Bitmap.ComponentType.uint16),
    (50, 100, Bitmap.PixelFormat.rgb, Bitmap.ComponentType.uint16),
    (100, 200, Bitmap.PixelFormat.rgba, Bitmap.ComponentType.uint16),
    (100, 200, Bitmap.PixelFormat.rgb, Bitmap.ComponentType.uint8, {"quality": 0}),
    (100, 200, Bitmap.PixelFormat.rgb, Bitmap.ComponentType.uint8, {"quality": 9}),
]


@pytest.mark.parametrize("layout", PNG_LAYOUTS)
def test_png_io(tmp_path: Path, layout: Sequence[Any]):
    extra = layout[4] if len(layout) > 4 else {}
    write_read_test(tmp_path, "png", layout[0], layout[1], layout[2], layout[3], **extra)


@pytest.mark.skipif(not Bitmap.supports_png_metadata(), reason="Requires the libpng decoder")
@pytest.mark.parametrize("channels,color_type", [(1, 0), (2, 4), (3, 2), (4, 6)])
@pytest.mark.parametrize("bit_depth", [8, 16])
@pytest.mark.parametrize(
    "metadata,expected",
    [
        ([], None),
        ([(b"sRGB", b"\0")], True),
        ([(b"gAMA", struct.pack(">I", 100000))], False),
        ([(b"gAMA", struct.pack(">I", 45455))], True),
        ([(b"gAMA", struct.pack(">I", 50000))], False),
        ([(b"gAMA", struct.pack(">I", 45455)), (b"sRGB", b"\0")], True),
    ],
    ids=["untagged", "srgb", "linear", "gamma-2.2", "other-gamma", "srgb-and-gamma"],
)
def test_png_transfer_metadata(
    tmp_path: Path,
    channels: int,
    color_type: int,
    bit_depth: int,
    metadata: list[tuple[bytes, bytes]],
    expected: Optional[bool],
) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    value = 128 if bit_depth == 8 else 0x80AB
    row = value.to_bytes(bit_depth // 8, "big") * channels
    path = tmp_path / "metadata.png"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, bit_depth, color_type, 0, 0, 0))
        + b"".join(chunk(kind, data) for kind, data in metadata)
        + chunk(b"IDAT", zlib.compress(b"\0" + row))
        + chunk(b"IEND", b"")
    )
    bitmap = Bitmap(path)
    assert bitmap.srgb_gamma is (
        (bit_depth == 8 and channels in (3, 4)) if expected is None else expected
    )
    assert np.asarray(bitmap).dtype == (np.uint8 if bit_depth == 8 else np.uint16)
    np.testing.assert_array_equal(np.asarray(bitmap).reshape(-1), [value] * channels)
    # Transfer tags affect color fields, never the alpha channel or stored samples.
    for field in bitmap.pixel_struct:
        assert bool(field.flags & DataStruct.Flags.srgb_gamma) is (
            bitmap.srgb_gamma and field.name != "A"
        )


@pytest.mark.skipif(not Bitmap.supports_png_metadata(), reason="Requires the libpng decoder")
def test_png_linear_roundtrip(tmp_path: Path) -> None:
    data = np.full((2, 2, 3), 0.5, dtype=np.float32)
    bitmap = Bitmap(data).convert(component_type=Bitmap.ComponentType.uint16, srgb_gamma=False)
    path = tmp_path / "linear.png"
    bitmap.write(path)
    reread = Bitmap(path)
    assert reread.srgb_gamma is False
    np.testing.assert_array_equal(np.asarray(reread), np.asarray(bitmap))


JPG_LAYOUTS = [
    (1, 2, Bitmap.PixelFormat.y, Bitmap.ComponentType.uint8, {"atol": 5}),
    (50, 100, Bitmap.PixelFormat.rgb, Bitmap.ComponentType.uint8, {"atol": 5}),
    (
        50,
        100,
        Bitmap.PixelFormat.rgb,
        Bitmap.ComponentType.uint8,
        {"quality": 20, "atol": 20},
    ),
]


@pytest.mark.parametrize("layout", JPG_LAYOUTS)
def test_jpg_io(tmp_path: Path, layout: Sequence[Any]):
    extra = layout[4] if len(layout) > 4 else {}
    write_read_test(tmp_path, "jpg", layout[0], layout[1], layout[2], layout[3], **extra)


HDR_LAYOUTS = [
    (100, 200, Bitmap.PixelFormat.rgb, Bitmap.ComponentType.float32, {"rtol": 1e-2}),
]


@pytest.mark.parametrize("layout", HDR_LAYOUTS)
def test_hdr_io(tmp_path: Path, layout: Sequence[Any]):
    extra = layout[4] if len(layout) > 4 else {}
    write_read_test(tmp_path, "hdr", layout[0], layout[1], layout[2], layout[3], **extra)


DDS_READ_ITEMS = [
    ("bc1-unorm.dds", 256, 256, Bitmap.PixelFormat.rgba, Bitmap.ComponentType.uint8, False),
    ("bc1-unorm-srgb.dds", 256, 256, Bitmap.PixelFormat.rgba, Bitmap.ComponentType.uint8, True),
    ("bc3-unorm.dds", 256, 256, Bitmap.PixelFormat.rgba, Bitmap.ComponentType.uint8, False),
    ("bc4-unorm.dds", 256, 256, Bitmap.PixelFormat.r, Bitmap.ComponentType.uint8, False),
    ("bc5-unorm.dds", 256, 256, Bitmap.PixelFormat.rg, Bitmap.ComponentType.uint8, False),
    ("bc6h-uf16.dds", 409, 204, Bitmap.PixelFormat.rgb, Bitmap.ComponentType.float16, False),
    ("bc7-unorm.dds", 256, 256, Bitmap.PixelFormat.rgba, Bitmap.ComponentType.uint8, False),
    ("bc7-unorm-srgb.dds", 256, 256, Bitmap.PixelFormat.rgba, Bitmap.ComponentType.uint8, True),
    ("bc7-unorm-odd.dds", 127, 127, Bitmap.PixelFormat.rgba, Bitmap.ComponentType.uint8, False),
]


@pytest.mark.parametrize("item", DDS_READ_ITEMS, ids=[x[0] for x in DDS_READ_ITEMS])
def test_dds_read(item: tuple[str, int, int, Bitmap.PixelFormat, Bitmap.ComponentType, bool]):
    import slangpy.platform as platform

    dds_dir = platform.project_directory() / "data" / "test_images" / "dds"
    filename, width, height, pixel_format, component_type, srgb = item

    bmp = Bitmap(dds_dir / filename)
    assert bmp.width == width
    assert bmp.height == height
    assert bmp.pixel_format == pixel_format
    assert bmp.component_type == component_type
    assert bmp.srgb_gamma == srgb
    assert not bmp.empty()
    assert bmp.buffer_size > 0

    a = np.array(bmp, copy=False)
    assert a.shape[0] == height
    assert a.shape[1] == width


def test_dds_detect_format():
    """Verify DDS files are auto-detected by Bitmap constructor."""
    import slangpy.platform as platform

    dds_dir = platform.project_directory() / "data" / "test_images" / "dds"
    bmp = Bitmap(dds_dir / "bc7-unorm.dds")
    assert bmp.pixel_format == Bitmap.PixelFormat.rgba
    assert bmp.component_type == Bitmap.ComponentType.uint8


def test_dds_write_rejected_before_open(tmp_path: Path):
    path = tmp_path / "existing.dds"
    original = b"keep this data"
    path.write_bytes(original)
    bmp = Bitmap(np.zeros((2, 2, 4), dtype=np.uint8))

    with pytest.raises(RuntimeError, match="writing DDS files is not supported"):
        bmp.write(path)

    assert path.read_bytes() == original

    with pytest.raises(RuntimeError, match="writing DDS files is not supported"):
        bmp.write(path, format=Bitmap.FileFormat.dds)

    assert path.read_bytes() == original


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
