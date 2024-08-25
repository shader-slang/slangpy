from __future__ import annotations

from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..variable import AssignmentCursor, Variable
    from ..codegen import SlangCodeGen


class SlangType:
    opaque_types = {}

    def to_slang_type(self) -> str:
        raise NotImplementedError

    def differentiable(self) -> bool:
        return False

    def differentiate(self) -> Optional[SlangType]:
        return None

    def assign_vars(self, var: AssignmentCursor, val: Any):
        raise NotImplementedError


def opaque_type(*names: str):
    def register_opaque_type(cls: type):
        for name in names:
            SlangType.opaque_types[name] = cls
        return cls

    return register_opaque_type


class DontCare:
    pass


class TypedValue:
    def get_type(self) -> SlangType:
        raise NotImplementedError


class CodegenType(SlangType):
    def define_type(self, gen: SlangCodeGen):
        raise NotImplementedError


class BatchedType(SlangType):
    def infer_batch_size(self, value: Any) -> tuple[int, ...]:
        raise NotImplementedError

    def broadcast(self, value: Any, batch_size: tuple[int, ...]):
        raise NotImplementedError


class TypeTranslator:
    def declare_variable(self, var: Variable, gen: SlangCodeGen):
        raise NotImplementedError

    def read_variable(self, var: Variable, var_name: str) -> str:
        raise NotImplementedError

    def write_variable(
        self, var: Variable, var_name: str, value: str, gen: SlangCodeGen
    ):
        raise NotImplementedError
