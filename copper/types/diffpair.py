from __future__ import annotations

from .interfaces import SlangType, TypeTranslator, opaque_type
from .indexables import StrIndexableType

from collections import OrderedDict

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..variable import AssignmentCursor, Variable
    from ..codegen import SlangCodeGen
    from ..layers import ReflectedType
    from .base import SlangName


class DifferentialPair:
    def __init__(self, p: Any, d: Any = None):
        super().__init__()
        self.p = p
        self.d = d


@opaque_type("DifferentialPair")
class DifferentialPairType(StrIndexableType):
    def __init__(self, primal: SlangType):
        super().__init__()
        self.primal = primal
        differential = primal.differentiate()
        assert differential is not None
        self.differential = differential

    def differentiable(self) -> bool:
        return True

    def differentiate(self) -> SlangType:
        return (
            self
            if self.primal is self.differential
            else DifferentialPairType(self.differential)
        )

    def to_slang_type(self):
        return f"DifferentialPair<{self.primal.to_slang_type()}>"

    def fields(self) -> dict[str, SlangType]:
        return OrderedDict([("p", self.primal), ("d", self.differential)])

    def assign_vars(self, var: AssignmentCursor, val: Any):
        if not isinstance(val, DifferentialPair):
            var.error("Expected a DifferentialPair argument")

        var.set_translator(
            DifferentialPairTranslator(
                self.primal, self.differential, val.d is not None
            )
        )

        var["p"] = val.p
        if val.d:
            var["d"] = val.d

    @staticmethod
    def from_reflection(name: SlangName, type: ReflectedType) -> DifferentialPairType:
        return DifferentialPairType(type.generic_args()[0])


class DifferentialPairTranslator(TypeTranslator):
    def __init__(
        self, primal: SlangType, differential: SlangType, have_differential: bool
    ):
        super().__init__()
        self.primal = primal
        self.differential = differential
        self.have_differential = have_differential

    def declare_variable(self, var: Variable, gen: SlangCodeGen):
        gen.begin_block("struct ")
        var["p"].declare(gen)
        if self.have_differential:
            var["d"].declare(gen)
        gen.end_block(f" {var.name};")

    def read_variable(self, var: Variable, var_name: str) -> str:
        args = var["p"].read(f"{var_name}.p")
        if self.have_differential:
            args += ", " + var["d"].read(f"{var_name}.d")
        return f"diffPair({args})"

    def write_variable(
        self, var: Variable, var_name: str, value: str, gen: SlangCodeGen
    ):
        assert self.have_differential
        var["d"].write(f"{var_name}.d", f"{value}.d", gen)

    def __repr__(self):
        args = self.primal.to_slang_type()
        if self.have_differential:
            args += ", " + self.differential.to_slang_type()
        return f"diffPair({args})"
