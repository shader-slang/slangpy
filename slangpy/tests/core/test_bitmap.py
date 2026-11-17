# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from concurrent.futures import ThreadPoolExecutor
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
    info = Bitmap.read_info(path)
    assert info.width == b2.width
    assert info.height == b2.height
    assert info.pixel_format == b2.pixel_format
    assert info.component_type == b2.component_type
    assert info.channel_count == b2.channel_count

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


def test_numpy_pixel_format_mismatch() -> None:
    data = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="Pixel format does not match"):
        Bitmap(data, pixel_format=Bitmap.PixelFormat.rgba)
    assert Bitmap(data, pixel_format=Bitmap.PixelFormat.rgb).channel_count == 3


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


EXR_UINT, EXR_HALF, EXR_FLOAT = 0, 1, 2

# Rec. 709 primaries and D65 white point (red, green, blue, white xy), the EXR default.
EXR_REC709_CHROMATICITIES = (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)
EXR_REC2020_CHROMATICITIES = (0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290)


def write_exr_channels(
    path: Path,
    channels: dict[str, tuple[int, float]],
    chromaticities: Optional[Sequence[float]] = None,
):
    """
    Write a 2x2 uncompressed scanline EXR in which every channel holds a constant value.

    The file is assembled by hand so that decoder tests do not depend on ``Bitmap.write``.

    :param path: Output path.
    :param channels: Map from channel name to (EXR pixel type, value).
    :param chromaticities: Optional red, green, blue and white xy coordinates.
    """

    def attribute(name: str, kind: str, data: bytes) -> bytes:
        return name.encode() + b"\0" + kind.encode() + b"\0" + struct.pack("<I", len(data)) + data

    channels = dict(sorted(channels.items()))
    chlist = b"".join(
        name.encode() + b"\0" + struct.pack("<iB3xii", kind, 0, 1, 1)
        for name, (kind, _) in channels.items()
    )
    box = struct.pack("<4i", 0, 0, 1, 1)
    header = struct.pack("<II", 20000630, 2)
    header += attribute("channels", "chlist", chlist + b"\0")
    if chromaticities is not None:
        header += attribute("chromaticities", "chromaticities", struct.pack("<8f", *chromaticities))
    header += attribute("compression", "compression", b"\0")
    header += attribute("dataWindow", "box2i", box)
    header += attribute("displayWindow", "box2i", box)
    header += attribute("lineOrder", "lineOrder", b"\0")
    header += attribute("pixelAspectRatio", "float", struct.pack("<f", 1))
    header += attribute("screenWindowCenter", "v2f", struct.pack("<2f", 0, 0))
    header += attribute("screenWindowWidth", "float", struct.pack("<f", 1))
    header += b"\0"
    row = b"".join(
        struct.pack("<2" + "Ief"[kind], *([int(value) if kind == EXR_UINT else value] * 2))
        for kind, value in channels.values()
    )
    chunks = [struct.pack("<iI", y, len(row)) + row for y in range(2)]
    offset = len(header) + 8 * len(chunks)
    offsets = b""
    for chunk in chunks:
        offsets += struct.pack("<Q", offset)
        offset += len(chunk)
    path.write_bytes(header + offsets + b"".join(chunks))


def luminance_chroma_to_rgb(
    y: float, ry: float, by: float, chromaticities: Sequence[float]
) -> list[float]:
    xy = np.array(chromaticities, dtype=np.float64).reshape(4, 2)
    xyz = np.column_stack([xy[:, 0], xy[:, 1], 1 - xy[:, 0] - xy[:, 1]])
    white = xyz[3] / xyz[3, 1]
    scale = np.linalg.solve(xyz[:3].T, white)
    weights = scale * xyz[:3, 1]
    r = (ry + 1) * y
    b = (by + 1) * y
    g = (y - r * weights[0] - b * weights[2]) / weights[1]
    return [r, g, b]


@pytest.mark.parametrize("prefix", ["", "layer."])
@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize(
    "chromaticities", [None, EXR_REC2020_CHROMATICITIES], ids=["default", "rec2020"]
)
def test_exr_luminance_chroma(
    tmp_path: Path, prefix: str, alpha: bool, chromaticities: Optional[Sequence[float]]
):
    channels = {
        prefix + "Y": (EXR_FLOAT, 0.5),
        prefix + "RY": (EXR_FLOAT, 0.5),
        prefix + "BY": (EXR_FLOAT, -0.5),
    }
    if alpha:
        channels[prefix + "A"] = (EXR_FLOAT, 0.25)
    path = tmp_path / "yc.exr"
    write_exr_channels(path, channels, chromaticities)

    b = Bitmap(path)

    expected = luminance_chroma_to_rgb(
        0.5, 0.5, -0.5, chromaticities or EXR_REC709_CHROMATICITIES
    ) + ([0.25] if alpha else [])
    assert b.pixel_format == (Bitmap.PixelFormat.rgba if alpha else Bitmap.PixelFormat.rgb)
    assert b.channel_names == [prefix + name for name in "RGBA"[: len(expected)]]
    assert np.allclose(np.array(b), expected, rtol=1e-5, atol=1e-6)
    assert Bitmap.read_info(path).pixel_format == b.pixel_format


@pytest.mark.parametrize(
    "channels,pixel_format",
    [
        ({"a.R": 0.75, "a.G": 0.5, "a.B": 0.25}, Bitmap.PixelFormat.rgb),
        ({"a.B": 0.25, "b.R": 0.75, "c.G": 0.5}, Bitmap.PixelFormat.multi_channel),
        ({"A": 1.0, "a.R": 0.75, "a.G": 0.5, "a.B": 0.25}, Bitmap.PixelFormat.multi_channel),
    ],
    ids=["coherent", "mixed", "bare_alpha"],
)
def test_exr_layer_prefixes(
    tmp_path: Path, channels: dict[str, float], pixel_format: Bitmap.PixelFormat
):
    path = tmp_path / "layers.exr"
    write_exr_channels(path, {name: (EXR_FLOAT, value) for name, value in channels.items()})

    b = Bitmap(path)

    assert b.pixel_format == pixel_format
    if pixel_format == Bitmap.PixelFormat.rgb:
        assert b.channel_names == ["a.R", "a.G", "a.B"]
    assert sorted(b.channel_names) == sorted(channels)
    pixels = np.array(b)
    for i, name in enumerate(b.channel_names):
        assert np.all(pixels[..., i] == channels[name]), name
    assert Bitmap.read_info(path).pixel_format == pixel_format


@pytest.mark.parametrize("g_type,g_value", [(EXR_UINT, 1), (EXR_HALF, 0.5)], ids=["uint", "half"])
def test_exr_mixed_component_types(tmp_path: Path, g_type: int, g_value: float):
    path = tmp_path / "mixed.exr"
    write_exr_channels(
        path, {"R": (EXR_FLOAT, 0.75), "G": (g_type, g_value), "B": (EXR_FLOAT, 0.25)}
    )

    b = Bitmap(path)

    assert b.pixel_format == Bitmap.PixelFormat.rgb
    assert b.component_type == Bitmap.ComponentType.float32
    assert np.all(np.array(b) == [0.75, g_value, 0.25])


def test_exr_mixed_component_types_uint_first(tmp_path: Path):
    path = tmp_path / "mixed_uint_first.exr"
    write_exr_channels(path, {"B": (EXR_UINT, 3), "G": (EXR_FLOAT, 1.75), "R": (EXR_FLOAT, -0.25)})

    b = Bitmap(path)

    assert b.pixel_format == Bitmap.PixelFormat.rgb
    # The destination type comes from file-order channel 0 (B, UINT here), so the float
    # channels convert into uint32: negative clamps to 0 and 1.75 truncates to 1.
    assert b.component_type == Bitmap.ComponentType.uint32
    assert np.all(np.array(b) == [0, 1, 3])


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


@pytest.mark.parametrize("bit_depth", [8, 16])
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("from_memory", [False, True])
def test_png_read_bit_depth(bit_depth: int, channels: int, from_memory: bool):
    import slangpy.platform as platform

    png_dir = platform.project_directory() / "data" / "test_images" / "png"
    path = png_dir / f"uint{bit_depth}-{channels}ch.png"
    dtype = np.uint16 if bit_depth == 16 else np.uint8
    values = np.array(
        (
            [0, 1, 255, 256, 257, 32768, 65534, 65535]
            if bit_depth == 16
            else [0, 1, 2, 127, 128, 253, 254, 255]
        ),
        dtype=dtype,
    )
    expected = np.stack([np.roll(values, channel) for channel in range(channels)], axis=-1).reshape(
        2, 4, channels
    )
    if from_memory:
        bitmap = Bitmap.load_from_numpy(np.fromfile(path, dtype=np.uint8))
    else:
        bitmap = Bitmap(path)

    assert bitmap.component_type == (
        Bitmap.ComponentType.uint16 if bit_depth == 16 else Bitmap.ComponentType.uint8
    )
    assert (
        bitmap.pixel_format
        == [
            Bitmap.PixelFormat.y,
            Bitmap.PixelFormat.ya,
            Bitmap.PixelFormat.rgb,
            Bitmap.PixelFormat.rgba,
        ][channels - 1]
    )
    assert bitmap.width == 4
    assert bitmap.height == 2
    assert bitmap.channel_count == channels
    actual = np.array(bitmap, copy=False)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, expected[:, :, 0] if channels == 1 else expected)


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


def test_threaded_writes_roundtrip_and_report_errors(tmp_path: Path) -> None:
    png_data = np.arange(16 * 16 * 4, dtype=np.uint8).reshape(16, 16, 4)
    exr_data = np.arange(16 * 16 * 4, dtype=np.float32).reshape(16, 16, 4) / 10
    png = Bitmap(png_data)
    exr = Bitmap(exr_data)
    png_path = tmp_path / "threaded.png"
    exr_path = tmp_path / "threaded.exr"

    with ThreadPoolExecutor(max_workers=3) as pool:
        png_write = pool.submit(png.write, png_path)
        exr_write = pool.submit(exr.write, exr_path)
        bad_write = pool.submit(png.write, tmp_path / "unsupported.dds")
        png_write.result()
        exr_write.result()
        with pytest.raises(RuntimeError, match="writing DDS files is not supported"):
            bad_write.result()

    np.testing.assert_array_equal(np.asarray(Bitmap(png_path)), png_data)
    np.testing.assert_array_equal(np.asarray(Bitmap(exr_path)), exr_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
