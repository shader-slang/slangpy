# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re

import pytest

import slangpy as spy
import slangpy.testing.helpers as helpers
from slangpy.bindings import (
    WriteToCursorMarshall,
    WriteToCursorMarshallInfo,
    register_write_to_cursor_type,
)
from slangpy.bindings.typeregistry import (
    PYTHON_SIGNATURES,
    PYTHON_TYPES,
    get_or_create_type,
    lookup_signature_callback,
    lookup_type_callback,
)
from slangpy.core.native import NativeCallDataCache, NativeValueMarshall, SignatureBuilder


class CursorValueObject:
    pass


class DuplicateCursorValueObject:
    pass


class NativeCursorValueObject:
    pass


class FakeSlangType:
    def __init__(self, full_name: str) -> None:
        self.full_name = full_name


DEVICE_TYPES = helpers.DEFAULT_DEVICE_TYPES[0:1]


def cursor_value_layout(device_type: spy.DeviceType) -> object:
    device = helpers.get_device(type=device_type)
    module = helpers.create_module(
        device,
        """
struct CursorValue
{
    uint id;
};
""",
    )
    return module.layout


def unregister_write_to_cursor_type(python_type: type) -> None:
    PYTHON_TYPES.pop(python_type, None)
    PYTHON_SIGNATURES.pop(python_type, None)


def native_value_signature(value: object) -> str:
    cache = NativeCallDataCache()
    signature = SignatureBuilder()
    cache.get_value_signature(signature, value)
    return signature.str


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_registered_write_to_cursor_marshall_type_callback(device_type: spy.DeviceType) -> None:
    try:
        register_write_to_cursor_type(
            CursorValueObject,
            slang_type_name="CursorValue",
            signature="[CursorValueObject]",
        )
        type_callback = lookup_type_callback(CursorValueObject)

        assert type_callback is not None
        marshall = type_callback(cursor_value_layout(device_type), CursorValueObject())
        assert isinstance(marshall, WriteToCursorMarshall)
        assert isinstance(marshall, NativeValueMarshall)
    finally:
        unregister_write_to_cursor_type(CursorValueObject)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_get_or_create_type_uses_native_cursor_writer_metadata(
    monkeypatch: pytest.MonkeyPatch,
    device_type: spy.DeviceType,
) -> None:
    value = NativeCursorValueObject()

    def get_native_cursor_writer_type_info(obj: object) -> dict[str, object] | None:
        if obj is value:
            return {
                "slang_type_name": "CursorValue",
                "signature": "[NativeCursorValueObject]",
                "imports": ("cursor_value.slang",),
            }
        return None

    monkeypatch.setattr(
        "slangpy.core.native._get_native_cursor_writer_type_info",
        get_native_cursor_writer_type_info,
    )

    marshall = get_or_create_type(cursor_value_layout(device_type), NativeCursorValueObject, value)

    assert isinstance(marshall, WriteToCursorMarshall)
    assert isinstance(marshall, NativeValueMarshall)
    assert marshall.m_info.signature == "[NativeCursorValueObject]"
    assert marshall.m_info.imports == ("cursor_value.slang",)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_get_or_create_type_prefers_python_registration_over_native_cursor_writer_metadata(
    monkeypatch: pytest.MonkeyPatch,
    device_type: spy.DeviceType,
) -> None:
    def get_native_cursor_writer_type_info(obj: object) -> dict[str, object] | None:
        raise AssertionError("native cursor-writer fallback should not be consulted")

    monkeypatch.setattr(
        "slangpy.core.native._get_native_cursor_writer_type_info",
        get_native_cursor_writer_type_info,
    )

    try:
        register_write_to_cursor_type(
            CursorValueObject,
            slang_type_name="CursorValue",
            signature="[PythonRegisteredCursorValueObject]",
        )

        marshall = get_or_create_type(
            cursor_value_layout(device_type),
            CursorValueObject,
            CursorValueObject(),
        )

        assert isinstance(marshall, WriteToCursorMarshall)
        assert marshall.m_info.signature == "[PythonRegisteredCursorValueObject]"
    finally:
        unregister_write_to_cursor_type(CursorValueObject)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_native_signatures_use_cursor_writer_registry(
    device_type: spy.DeviceType,
) -> None:
    device = helpers.get_device(type=device_type)

    buffer = device.create_buffer(
        size=4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    assert native_value_signature(buffer) == f"[{int(buffer.desc.usage)}]"

    tensor = spy.Tensor.empty(device, shape=(2, 3), dtype=float)
    assert (
        native_value_signature(tensor) == "Tensor\n[float,2,(shader_resource | unordered_access)]"
    )

    texture = device.create_texture(
        width=4,
        height=4,
        format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.shader_resource,
    )
    assert native_value_signature(texture) == (
        f"[{texture.desc.type.value},"
        f"{int(texture.desc.usage)},"
        f"{texture.desc.format.value},"
        f"{int(texture.desc.array_length)}]"
    )


def test_registered_write_to_cursor_marshall_signature() -> None:
    try:
        register_write_to_cursor_type(
            CursorValueObject,
            slang_type_name="CursorValue",
            signature="[CursorValueObject]",
        )
        found, signature_callback = lookup_signature_callback(CursorValueObject)

        assert found is True
        assert signature_callback is not None
        assert signature_callback(CursorValueObject()) == "[CursorValueObject]"
    finally:
        unregister_write_to_cursor_type(CursorValueObject)


def test_register_write_to_cursor_marshall_duplicate_rejected() -> None:
    try:
        register_write_to_cursor_type(DuplicateCursorValueObject, slang_type_name="CursorValue")

        with pytest.raises(ValueError, match="already registered"):
            register_write_to_cursor_type(DuplicateCursorValueObject, slang_type_name="CursorValue")
    finally:
        unregister_write_to_cursor_type(DuplicateCursorValueObject)


def test_write_to_cursor_marshall_info_accepts_fullmatch_regex() -> None:
    info = WriteToCursorMarshallInfo(
        slang_type_name="CursorValue",
        signature="[CursorValueObject]",
        imports=(),
        accepted_type_regex=re.compile(r"CursorValue(?:<.*>)?"),
    )

    assert info.accepts_type(FakeSlangType("CursorValue")) is True
    assert info.accepts_type(FakeSlangType("CursorValue<float4>")) is True
    assert info.accepts_type(FakeSlangType("Nested.CursorValue")) is False
