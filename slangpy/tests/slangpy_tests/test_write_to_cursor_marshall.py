# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re

import pytest

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
from slangpy.core.native import NativeSlangType, NativeValueMarshall


class CursorValueObject:
    pass


class DuplicateCursorValueObject:
    pass


class NativeCursorValueObject:
    pass


class FakeSlangType:
    def __init__(self, full_name: str) -> None:
        self.full_name = full_name


class FakeLayout:
    def find_type_by_name(self, name: str) -> NativeSlangType | None:
        if name == "CursorValue":
            return NativeSlangType()
        return None


def unregister_write_to_cursor_type(python_type: type) -> None:
    PYTHON_TYPES.pop(python_type, None)
    PYTHON_SIGNATURES.pop(python_type, None)


def test_registered_write_to_cursor_marshall_type_callback() -> None:
    try:
        register_write_to_cursor_type(
            CursorValueObject,
            slang_type_name="CursorValue",
            signature="[CursorValueObject]",
        )
        type_callback = lookup_type_callback(CursorValueObject)

        assert type_callback is not None
        marshall = type_callback(FakeLayout(), CursorValueObject())
        assert isinstance(marshall, WriteToCursorMarshall)
        assert isinstance(marshall, NativeValueMarshall)
    finally:
        unregister_write_to_cursor_type(CursorValueObject)


def test_get_or_create_type_uses_native_cursor_writer_metadata(
    monkeypatch: pytest.MonkeyPatch,
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

    marshall = get_or_create_type(FakeLayout(), NativeCursorValueObject, value)

    assert isinstance(marshall, WriteToCursorMarshall)
    assert isinstance(marshall, NativeValueMarshall)
    assert marshall.m_info.signature == "[NativeCursorValueObject]"
    assert marshall.m_info.imports == ("cursor_value.slang",)


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
