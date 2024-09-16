
from typing import TYPE_CHECKING, Any
from kernelfunctions.core.basevariable import BaseVariable
from kernelfunctions.core.enums import PrimType

if TYPE_CHECKING:
    from kernelfunctions.core.basetype import BaseType


class BaseVariableImpl(BaseVariable):
    def __init__(self):
        super().__init__()

    def is_compatible(self, other: 'BaseVariable') -> bool:
        return True

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

    def _find_bottom_level_element(self, value: Any = None) -> 'BaseType':
        t = self.primal
        while True:
            c = t.element_type(value)
            if c is None or c is t:
                return c
            t = c
        return t
