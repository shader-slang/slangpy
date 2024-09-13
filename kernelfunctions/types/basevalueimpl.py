
from typing import Optional
from kernelfunctions.codegen import CodeGenBlock
from kernelfunctions.types.basevalue import BaseValue
from kernelfunctions.types.enums import AccessType, PrimType


class BaseValueImpl(BaseValue):
    def __init__(self):
        super().__init__()

    def is_compatible(self, other: 'BaseValue') -> bool:
        return True

    @property
    def primal_type_name(self):
        return self.primal.name()

    @property
    def derivative_type_name(self):
        return self.derivative.name() if self.derivative is not None else None

    @property
    def argument_declaration(self):
        return f"{self.primal_type_name} {self.name}"

    def get(self, t: PrimType):
        if t == PrimType.primal:
            return self.primal
        else:
            assert self.derivative is not None
            return self.derivative

    def _recurse_str(self, depth: int) -> str:
        if self.fields is not None:
            child_strs = [
                f"{'  ' * depth}{name}: {child._recurse_str(depth + 1)}" for name, child in self.fields.items()]
            return "\n" + "\n".join(child_strs)
        else:
            if self.name == "":
                return f"{self.primal.name()}"
            else:
                return f"{self.primal.name()} ({self.name})"
