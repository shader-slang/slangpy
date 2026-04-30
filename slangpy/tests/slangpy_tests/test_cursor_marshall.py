# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re

import pytest

from slangpy.bindings import CursorMarshallInfo, register_cursor_type
from slangpy.bindings.typeregistry import (
    PYTHON_SIGNATURES,
    PYTHON_TYPES,
    lookup_signature_callback,
    lookup_type_callback,
)


class CursorValueObject:
    pass


class DuplicateCursorValueObject:
    pass


class FakeSlangType:
    def __init__(self, full_name: str) -> None:
        self.full_name = full_name


def unregister_cursor_type(python_type: type) -> None:
    PYTHON_TYPES.pop(python_type, None)
    PYTHON_SIGNATURES.pop(python_type, None)


def test_registered_cursor_marshall_type_callback() -> None:
    try:
        register_cursor_type(
            CursorValueObject,
            slang_type_name="CursorValue",
            signature="[CursorValueObject]",
        )
        type_callback = lookup_type_callback(CursorValueObject)

        assert type_callback is not None
    finally:
        unregister_cursor_type(CursorValueObject)


def test_registered_cursor_marshall_signature() -> None:
    try:
        register_cursor_type(
            CursorValueObject,
            slang_type_name="CursorValue",
            signature="[CursorValueObject]",
        )
        found, signature_callback = lookup_signature_callback(CursorValueObject)

        assert found is True
        assert signature_callback is not None
        assert signature_callback(CursorValueObject()) == "[CursorValueObject]"
    finally:
        unregister_cursor_type(CursorValueObject)


def test_register_cursor_marshall_duplicate_rejected() -> None:
    try:
        register_cursor_type(DuplicateCursorValueObject, slang_type_name="CursorValue")

        with pytest.raises(ValueError, match="already registered"):
            register_cursor_type(DuplicateCursorValueObject, slang_type_name="CursorValue")
    finally:
        unregister_cursor_type(DuplicateCursorValueObject)


def test_cursor_marshall_info_accepts_fullmatch_regex() -> None:
    info = CursorMarshallInfo(
        slang_type_name="CursorValue",
        signature="[CursorValueObject]",
        imports=(),
        accepted_type_regex=re.compile(r"CursorValue(?:<.*>)?"),
    )

    assert info.accepts_type(FakeSlangType("CursorValue")) is True
    assert info.accepts_type(FakeSlangType("CursorValue<float4>")) is True
    assert info.accepts_type(FakeSlangType("Nested.CursorValue")) is False
