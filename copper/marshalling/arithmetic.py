from __future__ import annotations

from .base import IOFlag, IVariable, Wrapper, register_wrapper, register_assignment, assign

from ..reflection import ScalarKind, SlangType, ScalarType, opaque_type

from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from ..layers import ShaderCursor


@opaque_type("Constant", IVariable, [SlangType])
class Constant(IVariable):
    def __init__(self, type: SlangType):
        super().__init__(type)


class ConstantWrapper(Wrapper):
    def __init__(self, type: ScalarType, value: int | float, flag: IOFlag):
        super().__init__()
        assert not flag.write()  # TODO
        self.type = type

    def batch_size(self, value: Any) -> tuple[int, ...]:
        return ()

    def broadcast(self, value: Any, batch_size: tuple[int, ...]) -> tuple[IVariable, Any]:
        value = float(value) if self.type.kind == ScalarKind.Float else int(value)
        return Constant(self.type), value


def assign_constant(type: Constant, var: ShaderCursor, value: int | float):
    assign(type.type, var["value"], value)


register_wrapper(int, ConstantWrapper)
register_wrapper(float, ConstantWrapper)

register_assignment(Constant, assign_constant)
