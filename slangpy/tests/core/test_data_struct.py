# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Optional
import pytest
from slangpy import DataStruct, DataStructConverter
import struct
import numpy as np
import itertools
import numpy.typing as npt

TSupportedType = tuple[str, DataStruct.Type, npt.DTypeLike]

supported_types: list[TSupportedType] = [
    ("b", DataStruct.Type.int8, np.int8),
    ("B", DataStruct.Type.uint8, np.uint8),
    ("h", DataStruct.Type.int16, np.int16),
    ("H", DataStruct.Type.uint16, np.uint16),
    ("i", DataStruct.Type.int32, np.int32),
    ("I", DataStruct.Type.uint32, np.uint32),
    ("q", DataStruct.Type.int64, np.int64),
    ("Q", DataStruct.Type.uint64, np.uint64),
    ("e", DataStruct.Type.float16, np.float16),
    ("f", DataStruct.Type.float32, np.float32),
    ("d", DataStruct.Type.float64, np.float64),
]


def from_srgb(x: float):
    if x < 0.04045:
        return x / 12.92
    else:
        return ((x + 0.055) / 1.055) ** 2.4


def to_srgb(x: float):
    if x < 0.0031308:
        return x * 12.92
    else:
        return 1.055 * (x ** (1.0 / 2.4)) - 0.055


def check_conversion(
    converter: DataStructConverter,
    src_fmt: str | bytes,
    dst_fmt: str | bytes,
    src_values: npt.ArrayLike,  # type: ignore
    ref_values: Optional[npt.ArrayLike] = None,  # type: ignore
    err_thresh: float = 1e-6,
):
    # print("\nsrc_values: " + str(src_values))
    src_data = struct.pack(src_fmt, *src_values)
    # print("src_data: " + binascii.hexlify(src_data).decode("utf8"))
    dst_data = converter.convert(src_data)
    # print("dst_data: " + binascii.hexlify(dst_data).decode("utf8"))
    dst_values = struct.unpack(dst_fmt, dst_data)
    # print("dst_values: " + str(dst_values))
    ref = ref_values if ref_values is not None else src_values
    # print("ref: " + str(ref))
    for i in range(len(dst_values)):
        abs_err = float(dst_values[i]) - float(ref[i])  # type: ignore (numpy pain)
        assert np.abs(abs_err / (ref[i] + 1e-6)) < err_thresh  # type: ignore (numpy pain)


def test_fields_unpacked():
    s = DataStruct()
    assert s.size == 0
    assert s.alignment == 1

    s.append("float32", DataStruct.Type.float32)
    assert s.size == 4
    assert s.alignment == 4
    assert s[0].name == "float32"
    assert s[0].type == DataStruct.Type.float32
    assert s[0].offset == 0
    assert s[0].size == 4

    s.append("uint8", DataStruct.Type.uint8)
    assert s.size == 8
    assert s.alignment == 4
    assert s[1].name == "uint8"
    assert s[1].type == DataStruct.Type.uint8
    assert s[1].offset == 4
    assert s[1].size == 1

    s.append("uint16", DataStruct.Type.uint16)
    assert s.size == 8
    assert s.alignment == 4
    assert s[2].name == "uint16"
    assert s[2].type == DataStruct.Type.uint16
    assert s[2].offset == 6
    assert s[2].size == 2


def test_fields_packed():
    s = DataStruct(pack=True)
    assert s.size == 0
    assert s.alignment == 1

    s.append("float32", DataStruct.Type.float32)
    assert s.size == 4
    assert s.alignment == 1
    assert s[0].name == "float32"
    assert s[0].type == DataStruct.Type.float32
    assert s[0].offset == 0
    assert s[0].size == 4

    s.append("uint8", DataStruct.Type.uint8)
    assert s.size == 5
    assert s.alignment == 1
    assert s[1].name == "uint8"
    assert s[1].type == DataStruct.Type.uint8
    assert s[1].offset == 4
    assert s[1].size == 1

    s.append("uint16", DataStruct.Type.uint16)
    assert s.size == 7
    assert s.alignment == 1
    assert s[2].name == "uint16"
    assert s[2].type == DataStruct.Type.uint16
    assert s[2].offset == 5
    assert s[2].size == 2


@pytest.mark.parametrize("param", supported_types)
def test_convert_passthrough(param: TSupportedType):
    src = DataStruct().append("val", param[1])
    dst = DataStruct().append("val", param[1])
    ss = DataStructConverter(src, dst)
    values = list(range(10))
    if DataStruct.is_signed(param[1]):
        values += list(range(-10, 0))
    check_conversion(ss, "@" + param[0] * len(values), "@" + param[0] * len(values), values)


@pytest.mark.parametrize("param", itertools.product(supported_types, repeat=2))
def test_convert_types(param: tuple[TSupportedType, TSupportedType]):
    p1, p2 = param
    values = list(range(10))
    if DataStruct.is_signed(p1[1]) and DataStruct.is_signed(p2[1]):
        values += list(range(-10, 0))

    max_range = int(min(DataStruct.type_range(p1[1])[1], DataStruct.type_range(p2[1])[1]))
    if max_range > 1024:
        values += list(range(1000, 1024))

    # Native -> Native
    s1 = DataStruct().append("val", p1[1])
    s2 = DataStruct().append("val", p2[1])
    s = DataStructConverter(s1, s2)
    check_conversion(s, "@" + p1[0] * len(values), "@" + p2[0] * len(values), values)

    # BE -> LE
    s1 = DataStruct(byte_order=DataStruct.ByteOrder.big_endian).append("val", p1[1])
    s2 = DataStruct(byte_order=DataStruct.ByteOrder.little_endian).append("val", p2[1])
    s = DataStructConverter(s1, s2)
    check_conversion(s, ">" + p1[0] * len(values), "<" + p2[0] * len(values), values)

    # LE -> BE
    s1 = DataStruct(byte_order=DataStruct.ByteOrder.little_endian).append("val", p1[1])
    s2 = DataStruct(byte_order=DataStruct.ByteOrder.big_endian).append("val", p2[1])
    s = DataStructConverter(s1, s2)
    check_conversion(s, "<" + p1[0] * len(values), ">" + p2[0] * len(values), values)


@pytest.mark.parametrize("param", supported_types)
def test_default_value(param: TSupportedType):
    s1 = DataStruct().append("val1", param[1]).append("val3", param[1])
    s2 = (
        DataStruct()
        .append("val1", param[1])
        .append("val2", param[1], DataStruct.Flags.default, 123)
        .append("val3", param[1])
    )
    s = DataStructConverter(s1, s2)

    values = list(range(10))
    if DataStruct.is_signed(param[1]):
        values += list(range(-10, 0))
    output = []
    for k in range(len(values) // 2):
        output += [values[k * 2], 123, values[k * 2 + 1]]

    check_conversion(
        s,
        "@" + param[0] * len(values),
        "@" + param[0] * (len(values) + len(values) // 2),
        values,
        output,
    )


def test_missing_field_error():
    s1 = DataStruct().append("val1", DataStruct.Type.uint32)
    s2 = DataStruct().append("val2", DataStruct.Type.uint32)
    with pytest.raises(RuntimeError, match='Field "val2" not found in source struct.'):
        s = DataStructConverter(s1, s2)
        check_conversion(s, "@I", "@I", [1], [1])


def test_round_and_saturation():
    s1 = DataStruct().append("val", DataStruct.Type.float32)
    s2 = DataStruct().append("val", DataStruct.Type.int8)
    s = DataStructConverter(s1, s2)
    values = [-0.55, -0.45, 0, 0.45, 0.55, 127, 128, -127, -200]
    check_conversion(s, "@fffffffff", "@bbbbbbbbb", values, [-1, 0, 0, 0, 1, 127, 127, -127, -128])


@pytest.mark.parametrize("param", supported_types)
def test_roundtrip_normalization(param: TSupportedType):
    s1 = DataStruct().append("val", param[1], DataStruct.Flags.normalized)
    s2 = DataStruct().append("val", DataStruct.Type.float32)
    s = DataStructConverter(s1, s2)
    max_range = 1.0
    if DataStruct.is_integer(param[1]):
        max_range = float(DataStruct.type_range(param[1])[1])
    values_in = list(range(10))
    values_out = [i / max_range for i in range(10)]
    check_conversion(s, "@" + (param[0] * 10), "@" + ("f" * 10), values_in, values_out)

    s = DataStructConverter(s2, s1)
    check_conversion(s, "@" + ("f" * 10), "@" + (param[0] * 10), values_out, values_in)


@pytest.mark.parametrize("param", supported_types)
def test_roundtrip_normalization_int2int(param: TSupportedType):
    if DataStruct.is_float(param[1]):
        return
    s1_type = DataStruct.Type.int8 if DataStruct.is_signed(param[1]) else DataStruct.Type.uint8
    s1_dtype = "b" if DataStruct.is_signed(param[1]) else "B"
    s1_range = DataStruct.type_range(s1_type)
    s2_range = DataStruct.type_range(param[1])
    s1 = DataStruct().append("val", s1_type, DataStruct.Flags.normalized)
    s2 = DataStruct().append("val", param[1], DataStruct.Flags.normalized)
    s = DataStructConverter(s1, s2)
    values_in = list(range(int(s1_range[0]), int(s1_range[1] + 1)))
    values_out = np.array(values_in, dtype=np.float64)
    values_out *= s2_range[1] / s1_range[1]
    values_out = np.rint(values_out)
    values_out = np.maximum(values_out, s2_range[0])
    values_out = np.minimum(values_out, s2_range[1])
    values_out = np.array(values_out, param[2])
    check_conversion(
        s,
        "@" + (s1_dtype * len(values_in)),
        "@" + (param[0] * len(values_in)),
        values_in,
        values_out,
    )


def test_gamma_1():
    s = DataStructConverter(
        DataStruct().append(
            "v",
            DataStruct.Type.uint8,
            DataStruct.Flags.normalized | DataStruct.Flags.srgb_gamma,
        ),
        DataStruct().append("v", DataStruct.Type.float32),
    )

    src_data = list(range(256))
    dest_data = [from_srgb(x / 255.0) for x in src_data]

    check_conversion(s, "@" + ("B" * 256), "@" + ("f" * 256), src_data, dest_data, err_thresh=1e-5)


def test_gamma_2():
    s = DataStructConverter(
        DataStruct().append("v", DataStruct.Type.float32),
        DataStruct().append(
            "v",
            DataStruct.Type.uint8,
            DataStruct.Flags.normalized | DataStruct.Flags.srgb_gamma,
        ),
    )

    src_data = list(np.linspace(0, 1, 256))
    dest_data = [np.uint8(np.round(to_srgb(x) * 255)) for x in src_data]

    check_conversion(s, "@" + ("f" * 256), "@" + ("B" * 256), src_data, dest_data)


def test_blend():
    src = DataStruct()
    src.append("a", DataStruct.Type.float32)
    src.append("b", DataStruct.Type.float32)

    target = DataStruct()
    target.append("v", DataStruct.Type.float32, blend=[(3.0, "a"), (4.0, "b")])

    s = DataStructConverter(src, target)
    check_conversion(s, "@ff", "@f", (1.0, 2.0), (3.0 + 8.0,))

    src = DataStruct()
    src.append("a", DataStruct.Type.uint8, DataStruct.Flags.normalized)
    src.append("b", DataStruct.Type.uint8, DataStruct.Flags.normalized)

    target = DataStruct()
    target.append("v", DataStruct.Type.float32, blend=[(3.0, "a"), (4.0, "b")])

    s = DataStructConverter(src, target)

    check_conversion(s, "@BB", "@f", (255, 127), (3.0 + 4.0 * (127.0 / 255.0),))


def test_blend_gamma():
    src = DataStruct()
    src.append(
        "a",
        DataStruct.Type.uint8,
        DataStruct.Flags.normalized | DataStruct.Flags.srgb_gamma,
    )
    src.append(
        "b",
        DataStruct.Type.uint8,
        DataStruct.Flags.normalized | DataStruct.Flags.srgb_gamma,
    )

    target = DataStruct()
    target.append(
        "v",
        DataStruct.Type.uint8,
        DataStruct.Flags.normalized | DataStruct.Flags.srgb_gamma,
        blend=[(1, "a"), (1, "b")],
    )

    s = DataStructConverter(src, target)
    ref = int(np.round(to_srgb(from_srgb(100 / 255.0) + from_srgb(200 / 255.0)) * 255))

    check_conversion(s, "@BB", "@B", (100, 200), (ref,))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
