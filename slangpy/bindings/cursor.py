# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import slangpy.reflection as spyref
from slangpy.bindings.boundvariable import BoundVariable, can_direct_bind_common
from slangpy.bindings.codegen import CodeGenBlock
from slangpy.bindings.marshall import BindContext, Marshall
from slangpy.bindings.typeregistry import PYTHON_SIGNATURES, PYTHON_TYPES
from slangpy.core.native import AccessType, CallContext


@dataclass(frozen=True)
class CursorMarshallInfo:
    slang_type_name: str
    signature: str
    imports: tuple[str, ...]
    accepted_type_names: tuple[str, ...]
    accepted_type_name_prefixes: tuple[str, ...]
    direct_bind: bool
    call_dimensionality: int

    def accepts_type(self, slang_type: spyref.SlangType) -> bool:
        full_name = slang_type.full_name
        if full_name in self.accepted_type_names:
            return True
        return any(
            full_name.startswith(prefix) for prefix in self.accepted_type_name_prefixes
        )


class CursorMarshall(Marshall):
    def __init__(self, layout: spyref.SlangProgramLayout, info: CursorMarshallInfo):
        super().__init__(layout)
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
            return self.m_info.call_dimensionality
        return None

    def create_calldata(
        self,
        context: CallContext,
        binding: Any,
        data: Any,
    ) -> Any:
        return data

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
        return (
            self.m_info.direct_bind
            and can_direct_bind_common(binding)
            and binding.access[0] == AccessType.read
        )

    def gen_trampoline_load(
        self,
        cgb: CodeGenBlock,
        binding: BoundVariable,
        data_name: str,
        value_name: str,
    ) -> bool:
        if not binding.direct_bind:
            return False
        cgb.append_statement(f"{value_name} = {data_name}")
        return True


def register_cursor_type(
    python_type: type,
    *,
    slang_type_name: str,
    signature: str | None = None,
    imports: Iterable[str] = (),
    accepted_type_names: Iterable[str] = (),
    accepted_type_name_prefixes: Iterable[str] = (),
    direct_bind: bool = True,
    call_dimensionality: int = 0,
) -> None:
    if python_type in PYTHON_TYPES or python_type in PYTHON_SIGNATURES:
        raise ValueError(
            f"Python type '{python_type.__name__}' is already registered with SlangPy."
        )

    accepted_names = tuple(accepted_type_names)
    if len(accepted_names) == 0:
        accepted_names = (slang_type_name,)

    type_signature = (
        signature
        if signature is not None
        else f"[CursorMarshall,{python_type.__module__}.{python_type.__qualname__},{slang_type_name}]"
    )
    info = CursorMarshallInfo(
        slang_type_name=slang_type_name,
        signature=type_signature,
        imports=tuple(imports),
        accepted_type_names=accepted_names,
        accepted_type_name_prefixes=tuple(accepted_type_name_prefixes),
        direct_bind=direct_bind,
        call_dimensionality=call_dimensionality,
    )

    def create_cursor_marshall(
        layout: spyref.SlangProgramLayout,
        _value: Any,
    ) -> CursorMarshall:
        return CursorMarshall(layout, info)

    def cursor_signature(_value: Any) -> str:
        return info.signature

    PYTHON_TYPES[python_type] = create_cursor_marshall
    PYTHON_SIGNATURES[python_type] = cursor_signature
