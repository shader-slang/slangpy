
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
    def writable(self):
        return self.primal.is_writable

    def _recurse_str(self, depth: int) -> str:
        if self.fields is not None:
            child_strs = [
                f"{'  ' * depth}{name}: {child._recurse_str(depth + 1)}" for name, child in self.fields.items()]
            return "\n" + "\n".join(child_strs)
        else:
            return f"{self.primal.name}"

    def _find_bottom_level_element(self, value: Any = None) -> Optional['NativeType']:
        t = self.primal
        while True:
            c = t.element_type
            if c is None or c is t:
                return c
            t = c
        return t
