from __future__ import annotations

from reflection import SlangType, InterfaceType

from collections import defaultdict
import enum

from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from codegen import SlangCodeGen
    from layers import ShaderCursor


class DontCare:
    pass


class IOFlag(enum.IntFlag):
    Read = enum.auto()
    Write = enum.auto()

    def read(self) -> bool:
        return IOFlag.Read in self

    def write(self) -> bool:
        return IOFlag.Write in self


class CodegenType(SlangType):
    unique_names: dict[SlangType, str] = {}
    class_count: dict[type, int] = defaultdict(int)

    def define(self, codegen: SlangCodeGen):
        pass

    def __str__(self) -> str:
        if self not in CodegenType.unique_names:
            count = CodegenType.class_count[type(self)]
            CodegenType.class_count[type(self)] += 1
            CodegenType.unique_names[self] = f"{type(self).__name__}_{count}"

        return CodegenType.unique_names[self]


class IVariable(InterfaceType, CodegenType):
    def __init__(self, type: SlangType):
        super().__init__()
        self.type = type

    def __str__(self) -> str:
        raise NotImplementedError


class Wrapper:
    def batch_size(self, value: Any) -> tuple[int, ...]:
        raise NotImplementedError

    def broadcast(self, value: Any, batch_size: tuple[int, ...]) -> tuple[IVariable, Any]:
        raise NotImplementedError


class Variable:
    unique_names: dict[Any, str] = {}
    class_count: dict[type, int] = defaultdict(int)

    def __init__(self, type: SlangType):
        super().__init__()
        self.type = type

    def define(self, codegen: SlangCodeGen):
        pass

    def assign(self, shader_var: ShaderCursor, value: Any, batch_size: tuple[int, ...]):
        pass

    def batch_size(self, value: Any) -> tuple[int, ...]:
        raise NotImplementedError

    def key(self) -> Any:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    @staticmethod
    def unique_name(var: Variable):
        key = var.key()
        if key not in Variable.unique_names:
            count = Variable.class_count[type(var)]
            Variable.class_count[type(var)] += 1
            Variable.unique_names[key] = f"{type(var).__name__}_{count}"

        return Variable.unique_names[key]


class Wildcard(SlangType):
    pass


_wrappers: dict[type, dict[type, Any]] = defaultdict(dict)


def register_wrapper(value_type: type, wrapper: Any):
    """if slang_type is not Wildcard and value_type is not Wildcard:
        _wrappers[value_type][slang_type] = wrapper
    elif slang_type is not Wildcard:
        _wrappers[object][slang_type] = wrapper
    elif value_type is not Wildcard:
        _wrappers[value_type][object] = wrapper
    else:
        raise ValueError("Double wild card wrapper is not allowed")"""
    # TODO
    pass


def register_assignment(slang_type: type[SlangType], assigner: Any):
    pass


def wrap(type: SlangType, value: Any, flag: IOFlag) -> Wrapper:
    for vtype in value.mro():
        if vtype in _wrappers:
            if type.__class__ in _wrappers[vtype]:
                return _wrappers[vtype][type.__class__](type, value, flag)
            elif object in _wrappers[vtype]:
                return _wrappers[vtype][type.__class__](type, value, flag)

    raise ValueError(
        f"Found no valid conversion from {value.__class__.__name__} to {type}")


def assign(type: SlangType, var: ShaderCursor, value: Any):
    pass
