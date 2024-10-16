
import re
from typing import TYPE_CHECKING, Any, Optional

from .basevariable import BaseVariable
from .enums import PrimType

if TYPE_CHECKING:
    from .native import NativeType


def _get_name(el_type: Optional['NativeType'], value: Any, default: Any = None):
    return el_type.name if el_type is not None else default


class BaseVariableImpl(BaseVariable):
    def __init__(self):
        super().__init__()

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

    def _find_bottom_level_element(self, value: Any = None) -> Optional['NativeType']:
        t = self.primal
        while True:
            c = t.element_type
            if c is None or c is t:
                return c
            t = c
        return t

    def is_compatible(self, other: 'BaseVariable') -> bool:
        if self.fields is not None:
            if other.fields is None:
                return False
            for field in self.fields:
                if field not in other.fields:
                    return False
                if not self.fields[field].is_compatible(other.fields[field]):
                    return False
            return True

        el_name = self.root_element_name
        other_name = other.root_element_name

        # None is 'wildcard'
        if el_name is None or other_name is None:
            return True

        if el_name == other_name:
            return True
        if el_name == 'none' or other_name == 'none':
            return True

        stripped_primal_name = re.sub(
            r"\d+_t", "", el_name).replace("uint", "int")
        stripped_other_name = re.sub(
            r"\d+_t", "", other_name).replace("uint", "int")

        if stripped_primal_name == stripped_other_name:
            return True

        if stripped_primal_name == f"vector<{stripped_other_name},1>":
            return True
        if f"vector<{stripped_primal_name},1>" == stripped_other_name:
            return True

        return False
