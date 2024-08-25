from __future__ import annotations

from .base import Modifier, VoidType
from .diffpair import DifferentialPairType

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interfaces import SlangType


class SlangFuncParam:
    def __init__(
        self,
        name: str,
        type: SlangType,
        has_default: bool,
        modifiers: Modifier,
        derivative_degree: int,
    ):
        super().__init__()
        self.name = name
        self.type = type
        self.has_default = has_default
        self.modifiers = modifiers

        # How many times has this parameter been differentiated?
        self.derivative_degree = derivative_degree

    def __repr__(self):
        out = ""
        pieces = []
        if Modifier.NoDiff in self.modifiers:
            pieces.append("no_diff")
        if Modifier.InOut in self.modifiers:
            pieces.append("inout")
        if Modifier.Out in self.modifiers:
            pieces.append("out")
        pieces.append(self.type.to_slang_type())
        pieces.append(self.name)
        # if self.has_default:
        #    pieces.append('= ...')
        return " ".join(pieces)

    def is_in(self) -> bool:
        return Modifier.In in self.modifiers or Modifier.InOut in self.modifiers

    def is_out(self) -> bool:
        return Modifier.Out in self.modifiers or Modifier.InOut in self.modifiers

    def is_inout(self) -> bool:
        return Modifier.InOut in self.modifiers

    def is_nodiff(self) -> bool:
        return Modifier.NoDiff in self.modifiers


class SlangFunc:
    def __init__(
        self,
        name: str,
        params: list[SlangFuncParam],
        return_type: SlangType,
        func_modifiers: Modifier,
    ):
        super().__init__()
        self.name = name
        self.return_type = return_type
        self.func_modifiers = func_modifiers
        self.params = params

    def differentiate(self) -> SlangFunc:
        params = self.params[:]
        if not isinstance(self.return_type, VoidType):
            params.append(self.get_return_param())

        diff_params = []
        for p in params:
            differentiable = not p.is_nodiff() and p.type.differentiable()
            pure_out = p.is_out() and not p.is_in()

            if pure_out and not differentiable:
                continue
            elif pure_out and differentiable:
                differential = p.type.differentiate()
                assert differential is not None
                p = SlangFuncParam(
                    p.name, differential, False, Modifier.In, p.derivative_degree + 1
                )
            elif p.is_in() and not differentiable:
                p = SlangFuncParam(
                    p.name, p.type, False, Modifier.In, p.derivative_degree
                )
            elif p.is_in() and differentiable:
                p = SlangFuncParam(
                    p.name,
                    DifferentialPairType(p.type),
                    False,
                    Modifier.InOut,
                    p.derivative_degree + 1,
                )

            diff_params.append(p)

        return SlangFunc(
            f"bwd_diff({self.name})", diff_params, VoidType(), Modifier.Nothing
        )

    def get_return_param(self) -> SlangFuncParam:
        if isinstance(self.return_type, VoidType):
            raise RuntimeError(
                "Trying to create a return parameter for a void function"
            )

        return_modifiers = Modifier.Out
        if Modifier.NoDiff in self.func_modifiers:
            return_modifiers |= Modifier.NoDiff

        return_name = self.unique_param_name("return_value")
        return SlangFuncParam(return_name, self.return_type, False, return_modifiers, 0)

    def unique_param_name(self, suggestion: str) -> str:
        if any(p.name == suggestion for p in self.params):
            return self.unique_param_name("_" + suggestion)
        return suggestion

    def __repr__(self):
        mods = ""
        fdiff = Modifier.ForwardDifferentiable in self.func_modifiers
        bdiff = Modifier.BackwardDifferentiable in self.func_modifiers
        if fdiff:
            mods += "[ForwardDifferentiable] "
        elif bdiff:
            mods += "[BackwardDifferentiable] "
        if Modifier.NoDiffThis in self.func_modifiers:
            mods += "[NoDiffThis] "
        if Modifier.Mutating in self.func_modifiers:
            mods += "[mutating] "
        if Modifier.NoDiff in self.func_modifiers:
            mods += "nodiff "

        return f'{mods}{self.return_type.to_slang_type()} {self.name}({", ".join(repr(p) for p in self.params)})'
