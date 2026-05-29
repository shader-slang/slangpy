# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

import slangpy as spy
import slangpy.testing.helpers as helpers
from slangpy.bindings import cursor_writer_type
from slangpy.bindings.typeregistry import PYTHON_SIGNATURES, PYTHON_TYPES
from slangpy.core.native import (
    NativeCallDataCache,
    SignatureBuilder,
    _unregister_python_cursor_writer_types,
)


@cursor_writer_type(
    slang_type_name="PythonCursorValue",
    signature="[PythonCursorValue]",
)
class PythonCursorValue:
    def __init__(self, item_id: int, weight: float) -> None:
        self.item_id = item_id
        self.weight = weight

    def write_to_cursor(self, cursor: spy.BufferElementCursor) -> None:
        cursor["item_id"] = self.item_id
        cursor["weight"] = self.weight


DEVICE_TYPES = helpers.DEFAULT_DEVICE_TYPES[0:1]


def unregister_python_cursor_writer_type(python_type: type) -> None:
    _unregister_python_cursor_writer_types()
    PYTHON_TYPES.pop(python_type, None)
    PYTHON_SIGNATURES.pop(python_type, None)


def native_value_signature(value: object) -> str:
    cache = NativeCallDataCache()
    signature = SignatureBuilder()
    cache.get_value_signature(signature, value)
    return signature.str


def python_cursor_value_layout(device_type: spy.DeviceType) -> object:
    device = helpers.get_device(type=device_type)
    module = device.load_module_from_source(
        "test_python_cursor_writer",
        r"""
struct PythonCursorValue
{
    uint item_id;
    float weight;
};
""",
    )
    return module.layout.get_type_layout(module.layout.find_type_by_name("PythonCursorValue"))


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_python_cursor_writer_callback_writes_buffer_struct(device_type: spy.DeviceType) -> None:
    try:
        value = PythonCursorValue(17, 2.5)
        cursor = spy.BufferCursor(device_type, python_cursor_value_layout(device_type), 1)
        cursor[0].write(value)

        result = cursor[0].read()
        assert result["item_id"] == 17
        assert result["weight"] == pytest.approx(2.5)
        assert native_value_signature(value) == "[PythonCursorValue]"
    finally:
        unregister_python_cursor_writer_type(PythonCursorValue)
