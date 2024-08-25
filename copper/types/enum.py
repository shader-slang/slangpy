from __future__ import annotations

from .interfaces import SlangType

import enum

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import SlangName
    from ..variable import AssignmentCursor


class EnumType(SlangType):
    def __init__(
        self,
        name: SlangName,
        dtype: SlangType,
        is_flags: bool,
        cases: list[tuple[str, int]],
    ):
        super().__init__()
        self.name = name
        self.dtype = dtype
        self.is_flags = is_flags
        self.cases = cases

    def to_slang_type(self) -> str:
        return self.name.declared

    def assign_vars(self, var: AssignmentCursor, val: Any):
        value_type = val.__class__.__name__
        if isinstance(val, enum.Enum):
            if value_type == self.name.base:
                self.dtype.assign_vars(var, val.value)
            else:
                var.error(
                    f"Expected a value from enum '{self.name.declared}', "
                    f"but received a value from enum {value_type}"
                )
        else:
            var.error(
                f"Expected a value from enum '{self.name.declared}', but "
                f"received a {value_type} instead"
            )

    def __eq__(self, other: Any):
        if not isinstance(other, EnumType):
            return NotImplemented
        return self.name == other.name and self.dtype == other.dtype
