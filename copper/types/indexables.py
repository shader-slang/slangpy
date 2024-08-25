from __future__ import annotations

from .base import SlangType

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..variable import AssignmentCursor


class IntIndexableType(SlangType):
    def length(self) -> int:
        raise NotImplementedError

    def element_type(self) -> SlangType:
        raise NotImplementedError

    def assign_vars(self, var: AssignmentCursor, val: Any):
        var.is_leaf = False
        if isinstance(val, list) or isinstance(val, tuple):
            if len(val) != self.length():
                var.error(
                    f"List/tuple not compatible with {self.to_slang_type()}: Expected "
                    f"{self.length()} elements, received {len(val)} instead"
                )
            for i in range(self.length()):
                var[i] = val[i]
        else:
            var.error(
                f"Argument of type {val.__class__.__name__} can't be "
                f"cast to {self.to_slang_type()}; expecting list/tuple"
            )


class StrIndexableType(SlangType):
    def fields(self) -> dict[str, SlangType]:
        raise NotImplementedError

    def assign_vars(self, var: AssignmentCursor, val: Any):
        var.is_leaf = False
        if isinstance(val, dict):
            fields = self.fields()
            for k in val.keys():
                if k not in fields:
                    var.error(f"Type {self.to_slang_type()} has no member {k}")

            for k, v in fields.items():
                if k not in val:
                    var.error(
                        f"Argument is missing required struct "
                        f"member {self.to_slang_type()}::{k}"
                    )
                var[k] = val[k]
        else:
            var.error(
                f"Expected dictionary for argument of type {self.to_slang_type()}; "
                f"received {val.__class__.__name__} instead"
            )
