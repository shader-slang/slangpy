import sys
import enum
import struct
from typing import Any


class PrintValueType(enum.IntEnum):
    Bool = 1
    Int8 = 2
    Int16 = 3
    Int = 4
    Uint8 = 5
    Uint16 = 6
    Uint = 7
    Float = 8
    String = 9
    Array = 10
    Vector = 11
    Format = 12


def print_slang_buffer(buf: bytes, hashed_strings: dict[int, str]):
    idx = 0
    while values_left(buf, idx) > 0:
        idx, val = next_element(buf, idx, hashed_strings)
        sys.stdout.write(str(val))


def next_element(
    buf: bytes, idx: int, hashed_strings: dict[int, str]
) -> tuple[int, Any]:
    kind = get_type(buf, idx)
    if kind == PrintValueType.Array or kind == PrintValueType.Vector:
        D = get_value(buf, idx, "<I")
        assert isinstance(D, int)
        result = []
        idx += 1
        while len(result) < D and values_left(buf, idx) > 0:
            idx, element = next_element(buf, idx, hashed_strings)
            result.append(element)
        if kind == PrintValueType.Vector:
            result = tuple(result)
        return idx, result
    elif kind == PrintValueType.Format:
        fmt = get_scalar_value(PrintValueType.String, hashed_strings, buf, idx)
        assert isinstance(fmt, str)
        idx += 2
        if values_left(buf, idx) > 0:
            cnt = get_scalar_value(PrintValueType.Uint, hashed_strings, buf, idx - 1)
            assert isinstance(cnt, int)
            values = []
            while len(values) < cnt and values_left(buf, idx) > 0:
                idx, element = next_element(buf, idx, hashed_strings)
                if isinstance(element, tuple) or isinstance(element, list):
                    element = str(element)
                values.append(element)
            fmt = fmt.format(*values)
        return idx, fmt
    else:
        return idx + 1, get_scalar_value(kind, hashed_strings, buf, idx)


def values_left(buf: bytes, idx: int) -> int:
    return (len(buf) // 8) - idx


def get_type(buf: bytes, idx: int) -> PrintValueType:
    return PrintValueType(struct.unpack_from("<I", buf, idx * 8)[0])


def get_value(buf: bytes, idx: int, fmt: str) -> int | float | str:
    return struct.unpack_from(fmt, buf, idx * 8 + 4)[0]


def get_scalar_value(
    type: PrintValueType, hashed_strings: dict[int, str], buf: bytes, idx: int
) -> int | float | str:
    if type == PrintValueType.Bool:
        return get_value(buf, idx, "<I") != 0
    elif type == PrintValueType.Int8:
        return get_value(buf, idx, "<b")
    elif type == PrintValueType.Uint8:
        return get_value(buf, idx, "<B")
    elif type == PrintValueType.Int16:
        return get_value(buf, idx, "<h")
    elif type == PrintValueType.Uint16:
        return get_value(buf, idx, "<H")
    elif type == PrintValueType.Int:
        return get_value(buf, idx, "<i")
    elif type == PrintValueType.Uint:
        return get_value(buf, idx, "<I")
    elif type == PrintValueType.Float:
        return get_value(buf, idx, "<f")
    elif type == PrintValueType.String:
        str_hash = get_value(buf, idx, "<I")
        assert isinstance(str_hash, int)
        return hashed_strings[str_hash]
    else:
        raise ValueError("{type} is not a scalar PrintValueType")
