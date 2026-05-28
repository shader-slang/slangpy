# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any, Optional

from slangpy.core.enums import PrimType


class DiffPair:
    """
    A pair of values, one representing the primal value and the other representing the gradient value.
    Typically only required when wanting to output gradients from scalar calls to a function.
    """

    def __init__(self, p: Optional[Any], d: Optional[Any], needs_grad: bool = True):
        super().__init__()
        self.primal = p if p is not None else 0.0
        self.grad = d if d is not None else type(self.primal)()
        self.needs_grad = needs_grad

    def get(self, type: PrimType):
        """
        Get the primal or gradient value.
        """
        return self.primal if type == PrimType.primal else self.grad

    def set(self, type: PrimType, value: Any):
        """
        Set the primal or gradient value.
        """
        if type == PrimType.primal:
            self.primal = value
        else:
            self.grad = value

    @property
    def slangpy_signature(self) -> str:
        """
        Get the unique type signature of the DiffPair.
        """
        return f"[{type(self.primal).__name__},{type(self.grad).__name__},{self.needs_grad}]"


def diffPair(p: Optional[Any] = None, d: Optional[Any] = None, needs_grad: bool = True) -> DiffPair:
    """
    Create a DiffPair.
    """
    return DiffPair(p, d, needs_grad)


def floatDiffPair(p: float = 0.0, d: float = 1.0, needs_grad: bool = True) -> DiffPair:
    """
    Helper to create a DiffPair with float values.
    """
    return diffPair(p, d, needs_grad)
