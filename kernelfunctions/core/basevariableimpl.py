
from typing import TYPE_CHECKING, Any, Optional

from .basevariable import BaseVariable
from .enums import PrimType

if TYPE_CHECKING:
    from .basetype import BaseType


def _get_name(el_type: Optional['BaseType'], value: Any, default: Any = None):
    return el_type.name if el_type is not None else default


class BaseVariableImpl(BaseVariable):
    def __init__(self):
        super().__init__()

    def is_compatible(self, other: 'BaseVariable') -> bool:
        return True

    @property
    def primal_type_name(self):
        return self.primal.name

    @property
    def derivative_type_name(self):
        return _get_name(self.derivative, None)

    @property
    def primal_element_name(self):
        return _get_name(self.primal.element_type, None)

    @property
    def derivative_element_name(self):
        if self.derivative is not None:
            return _get_name(self.derivative.element_type, None)
        else:
            return None

    @property
    def root_element_name(self):
        return _get_name(self._find_bottom_level_element(), None)

    @property
    def differentiable(self):
        return self.primal.differentiable and self.primal.has_derivative

    @property
    def writable(self):
        return self.primal.is_writable

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
                return f"{self.primal.name}"
            else:
                return f"{self.primal.name} ({self.name})"

    def _find_bottom_level_element(self, value: Any = None) -> 'BaseType':
        t = self.primal
        while True:
            c = t.element_type
            if c is None or c is t:
                return c
            t = c
        return t
