# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, get_args, get_type_hints

import slangpy.reflection as spyref
from slangpy.bindings.boundvariable import (
    BoundVariable,
    BoundVariableException,
    can_direct_bind_common,
)
from slangpy.bindings.codegen import CodeGenBlock
from slangpy.bindings.marshall import BindContext
from slangpy.bindings.typeregistry import PYTHON_SIGNATURES, PYTHON_TYPES
from slangpy.core.native import AccessType, NativeValueMarshall


@dataclass(frozen=True)
class WriteToCursorMarshallInfo:
    """Metadata needed to marshal values through native cursor-writer registration."""

    slang_type_name: str
    signature: str
    imports: tuple[str, ...]
    accepted_type_regex: re.Pattern[str]

    def accepts_type(self, slang_type: spyref.SlangType) -> bool:
        """Return True when a reflected Slang type is compatible with this writer."""
        return self.accepted_type_regex.fullmatch(slang_type.full_name) is not None


class WriteToCursorMarshall(NativeValueMarshall):
    """Marshall for scalar values that are written through the native cursor fast path."""

    def __init__(self, layout: spyref.SlangProgramLayout, info: WriteToCursorMarshallInfo):
        """Resolve the registered Slang type in the current composed program layout."""
        super().__init__()
        self.m_info = info
        slang_type = layout.find_type_by_name(info.slang_type_name)
        if slang_type is None:
            raise ValueError(
                f"Cursor-bindable Slang type '{info.slang_type_name}' is not visible. "
                "Import or link the module that defines it before passing this value to SlangPy."
            )
        self.slang_type = slang_type

    def resolve_types(
        self, context: BindContext, bound_type: spyref.SlangType
    ) -> list[spyref.SlangType]:
        """Allow binding only to the exact Slang type family accepted by this writer."""
        if self.m_info.accepts_type(bound_type):
            return [bound_type]
        return []

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: spyref.SlangType,
    ) -> int | None:
        """Native cursor-writer values bind as scalar values with no vectorized dimensions."""
        if self.m_info.accepts_type(vector_target_type):
            return 0
        return None

    def gen_calldata(
        self,
        cgb: CodeGenBlock,
        context: BindContext,
        binding: BoundVariable,
    ) -> None:
        """Emit imports and use the target Slang type directly as the call-data type."""
        assert binding.vector_type is not None
        for import_name in self.m_info.imports:
            cgb.add_import(import_name)
        binding.gen_calldata_type_name(cgb, binding.vector_type.full_name)

    def can_direct_bind(self, binding: BoundVariable) -> bool:
        """Use direct binding only for read-only scalar values."""
        return can_direct_bind_common(binding) and binding.access[0] == AccessType.read

    def gen_trampoline_load(
        self,
        cgb: CodeGenBlock,
        binding: BoundVariable,
        data_name: str,
        value_name: str,
    ) -> bool:
        """Load the entry-point value directly; writable cursor writers have no readback path."""
        if not binding.direct_bind:
            raise BoundVariableException(
                "WriteToCursorMarshall only supports read-only scalar direct binding.",
                binding,
            )
        cgb.append_statement(f"{value_name} = {data_name}")
        return True


def register_write_to_cursor_type(
    python_type: type,
    *,
    slang_type_name: str,
    signature: str | None = None,
    imports: Iterable[str] = (),
    accepted_type_regex: str | re.Pattern[str] | None = None,
) -> None:
    """
    Register a Python-visible value type for WriteToCursorMarshall construction.

    This is the Python registry side of the cursor-writer contract. Native values normally
    arrive through ``register_cursor_writer<T>()`` and are discovered from the native registry;
    this helper keeps Python-only/future registrations on the same metadata shape.
    """
    if python_type in PYTHON_TYPES or python_type in PYTHON_SIGNATURES:
        raise ValueError(
            f"Python type '{python_type.__name__}' is already registered with SlangPy."
        )

    if accepted_type_regex is None:
        type_regex = re.compile(re.escape(slang_type_name))
    elif isinstance(accepted_type_regex, str):
        type_regex = re.compile(accepted_type_regex)
    else:
        type_regex = accepted_type_regex

    type_signature = (
        signature
        if signature is not None
        else f"[WriteToCursorMarshall,{python_type.__module__}.{python_type.__qualname__},{slang_type_name}]"
    )
    info = WriteToCursorMarshallInfo(
        slang_type_name=slang_type_name,
        signature=type_signature,
        imports=tuple(imports),
        accepted_type_regex=type_regex,
    )

    def create_cursor_marshall(
        layout: spyref.SlangProgramLayout,
        _value: Any,
    ) -> WriteToCursorMarshall:
        return WriteToCursorMarshall(layout, info)

    def cursor_signature(_value: Any) -> str:
        return info.signature

    PYTHON_TYPES[python_type] = create_cursor_marshall
    PYTHON_SIGNATURES[python_type] = cursor_signature


def register_cursor_writer_type(
    python_type: type,
    *,
    write_shader_cursor: Callable[[Any, Any], None] | None = None,
    write_buffer_cursor: Callable[[Any, Any], None] | None = None,
    slang_type_name: str | None = None,
    signature: str | None = None,
    imports: Iterable[str] = (),
    accepted_type_regex: str | re.Pattern[str] | None = None,
) -> None:
    """
    Register a pure Python type with the native cursor-writer registry.

    The callback receives ``(cursor, value)``. If ``slang_type_name`` is supplied, the
    type is also registered for ``WriteToCursorMarshall`` construction.
    """
    if write_shader_cursor is None and write_buffer_cursor is None:
        raise ValueError("At least one cursor writer callback must be provided.")

    type_signature = (
        signature
        if signature is not None
        else f"[CursorWriter,{python_type.__module__}.{python_type.__qualname__}]"
    )
    import_tuple = tuple(imports)

    if slang_type_name is None and accepted_type_regex is not None:
        raise ValueError("accepted_type_regex requires slang_type_name.")
    if slang_type_name is None and import_tuple:
        raise ValueError("imports requires slang_type_name.")

    if slang_type_name is not None and (
        python_type in PYTHON_TYPES or python_type in PYTHON_SIGNATURES
    ):
        raise ValueError(
            f"Python type '{python_type.__name__}' is already registered with SlangPy."
        )

    from slangpy.core.native import register_python_cursor_writer_type

    register_python_cursor_writer_type(
        python_type,
        write_shader_cursor,
        write_buffer_cursor,
        type_signature,
        slang_type_name or "",
        list(import_tuple),
    )

    if slang_type_name is not None:
        register_write_to_cursor_type(
            python_type,
            slang_type_name=slang_type_name,
            signature=type_signature,
            imports=import_tuple,
            accepted_type_regex=accepted_type_regex,
        )


def _cursor_writer_method_cursor_kinds(method: Callable[..., Any]) -> set[str]:
    signature = inspect.signature(method)
    params = list(signature.parameters.values())
    if params and params[0].name in ("self", "cls"):
        params = params[1:]
    if not params:
        raise ValueError("Cursor writer method must take a cursor argument.")

    cursor_param = params[0]
    hints = get_type_hints(method)
    cursor_annotation = hints.get(cursor_param.name, cursor_param.annotation)
    if cursor_annotation is inspect.Signature.empty:
        raise ValueError(
            f"Cursor writer method '{method.__name__}' must annotate its cursor argument."
        )

    annotations = get_args(cursor_annotation) or (cursor_annotation,)

    import slangpy as spy

    kinds: set[str] = set()
    for annotation in annotations:
        if annotation is spy.ShaderCursor:
            kinds.add("shader")
        elif annotation is spy.BufferElementCursor:
            kinds.add("buffer")

    if not kinds:
        raise ValueError(
            f"Cursor writer method '{method.__name__}' cursor argument must be annotated "
            "as ShaderCursor, BufferElementCursor, or a union of those cursor types."
        )

    return kinds


def _instance_cursor_writer(method_name: str) -> Callable[[Any, Any], None]:
    def write(cursor: Any, value: Any) -> None:
        getattr(value, method_name)(cursor)

    return write


def cursor_writer_type(
    *,
    slang_type_name: str | None = None,
    signature: str | None = None,
    imports: Iterable[str] = (),
    accepted_type_regex: str | re.Pattern[str] | None = None,
) -> Callable[[type], type]:
    """
    Decorate a class whose methods write instances into Slang cursors.

    Supported method names are ``write_to_cursor`` with an annotated cursor
    argument, plus explicit ``write_to_shader_cursor`` and
    ``write_to_buffer_cursor`` methods.
    """

    def decorate(python_type: type) -> type:
        write_shader_cursor: Callable[[Any, Any], None] | None = None
        write_buffer_cursor: Callable[[Any, Any], None] | None = None

        def set_writer(kind: str, method_name: str) -> None:
            nonlocal write_shader_cursor, write_buffer_cursor
            writer = _instance_cursor_writer(method_name)
            if kind == "shader":
                if write_shader_cursor is not None:
                    raise ValueError(
                        f"Python type '{python_type.__name__}' has multiple shader cursor writers."
                    )
                write_shader_cursor = writer
            elif kind == "buffer":
                if write_buffer_cursor is not None:
                    raise ValueError(
                        f"Python type '{python_type.__name__}' has multiple buffer cursor writers."
                    )
                write_buffer_cursor = writer

        if hasattr(python_type, "write_to_shader_cursor"):
            set_writer("shader", "write_to_shader_cursor")
        if hasattr(python_type, "write_to_buffer_cursor"):
            set_writer("buffer", "write_to_buffer_cursor")

        if hasattr(python_type, "write_to_cursor"):
            method = getattr(python_type, "write_to_cursor")
            for kind in _cursor_writer_method_cursor_kinds(method):
                set_writer(kind, "write_to_cursor")

        register_cursor_writer_type(
            python_type,
            write_shader_cursor=write_shader_cursor,
            write_buffer_cursor=write_buffer_cursor,
            slang_type_name=slang_type_name,
            signature=signature,
            imports=imports,
            accepted_type_regex=accepted_type_regex,
        )
        return python_type

    return decorate
