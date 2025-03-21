# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from slangpy.reflection import SlangType, SlangProgramLayout, TYPE_OVERRIDES
from slangpy.core.native import Shape
from slangpy.backend import TypeReflection


class CoopVecType(SlangType):
    """
    This type represents a reflected CoopVec type.

    This type should not be created directly; it will be instantiated by slangpy
    instead of a generic SlangType when reflecting CoopVec or DiffCoopVec
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        args = program.get_resolved_generic_args(refl)
        assert args is not None
        assert len(args) == 2
        assert isinstance(args[0], SlangType)
        assert isinstance(args[1], int)
        super().__init__(program, refl, element_type=args[0], local_shape=Shape((args[1], )))
        self.element_type: SlangType
        self._dims = args[1]

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def dtype(self) -> SlangType:
        return self.element_type


TYPE_OVERRIDES["CoopVec"] = CoopVecType
TYPE_OVERRIDES["DiffCoopVec"] = CoopVecType
