# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

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
    slang_type_name: str
    signature: str
    imports: tuple[str, ...]
    accepted_type_regex: re.Pattern[str]

    def accepts_type(self, slang_type: spyref.SlangType) -> bool:
        return self.accepted_type_regex.fullmatch(slang_type.full_name) is not None


class WriteToCursorMarshall(NativeValueMarshall):
    def __init__(self, layout: spyref.SlangProgramLayout, info: WriteToCursorMarshallInfo):
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
        if self.m_info.accepts_type(bound_type):
            return [bound_type]
        return []

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: spyref.SlangType,
    ) -> int | None:
        if self.m_info.accepts_type(vector_target_type):
            return 0
        return None

    def gen_calldata(
        self,
        cgb: CodeGenBlock,
        context: BindContext,
        binding: BoundVariable,
    ) -> None:
        assert binding.vector_type is not None
        for import_name in self.m_info.imports:
            cgb.add_import(import_name)
        binding.gen_calldata_type_name(cgb, binding.vector_type.full_name)

    def can_direct_bind(self, binding: BoundVariable) -> bool:
        return can_direct_bind_common(binding) and binding.access[0] == AccessType.read

    def gen_trampoline_load(
        self,
        cgb: CodeGenBlock,
        binding: BoundVariable,
        data_name: str,
        value_name: str,
    ) -> bool:
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
