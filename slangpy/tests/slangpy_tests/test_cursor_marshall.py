# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from slangpy.bindings import register_cursor_type
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
