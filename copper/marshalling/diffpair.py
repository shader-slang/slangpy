from __future__ import annotations

from .base import Variable, register_wrapper, wrap

from ..reflection import Modifier, DifferentialPairType
from ..util import broadcast_shapes

from typing import Any, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ..codegen import SlangCodeGen
    from ..layers import ShaderCursor


class DifferentialPair:
    def __init__(self, p: Any, d: Any = None):
        super().__init__()
        self.p = p
        self.d = d


class DiffPairVariable(Variable):
    def __init__(self, type: DifferentialPairType, primal: Variable, differential: Optional[Variable]):
        assert type.primal == primal.type
        assert differential is None or type.differential == differential.type
        super().__init__(type)
        self.primal = primal
        self.differential = differential

    def define(self, codegen: SlangCodeGen):
        self.primal.define(codegen)
        if self.differential:
            self.differential.define(codegen)

    def assign(self, shader_var: ShaderCursor, value: DifferentialPair, batch_size: tuple[int, ...]):
        self.primal.assign(shader_var['p'], value.p, batch_size)
        if self.differential:
            self.differential.assign(shader_var['d'], value.p, batch_size)

    def batch_size(self, value: Any) -> tuple[int, ...]:
        shape = self.primal.batch_size(value.p)
        if self.differential:
            shape = broadcast_shapes([shape, self.differential.batch_size(value.d)])
            assert shape  # TODO
        return shape

    def key(self) -> Any:
        if self.differential:
            return (DiffPairVariable, self.primal.key(), self.differential.key())
        else:
            return (DiffPairVariable, self.primal.key())

    def __str__(self):
        if self.differential:
            return f"DiffPair<{self.type.primal}, {self.primal}, {self.differential}>"


def wrap_diffpair(type: DifferentialPairType, value: DifferentialPair, mod: Modifier):
    var_p, val_p = wrap(type.primal, value.p, mod)
    var_d, val_d = None, None
    if value.d is not None:
        var_d, val_d = wrap(type.differential, value.d, mod)

    return DiffPairVariable(type, var_p, var_d), DifferentialPair(val_p, val_d)


register_wrapper(DifferentialPairType, DifferentialPair, wrap_diffpair)
