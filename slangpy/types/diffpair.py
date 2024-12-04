
from typing import Any, Optional

from kernelfunctions.core import PrimType


class DiffPair:
    def __init__(self, p: Optional[Any], d: Optional[Any], needs_grad: bool = True):
        super().__init__()
        self.primal = p if p is not None else 0.0
        self.grad = d if d is not None else type(self.primal)()
        self.needs_grad = needs_grad

    def get(self, type: PrimType):
        return self.primal if type == PrimType.primal else self.grad

    def set(self, type: PrimType, value: Any):
        if type == PrimType.primal:
            self.primal = value
        else:
            self.grad = value

    @property
    def slangpy_signature(self) -> str:
        return f"[{type(self.primal).__name__},{type(self.grad).__name__},{self.needs_grad}]"


def diffPair(
    p: Optional[Any] = None, d: Optional[Any] = None, needs_grad: bool = True
) -> DiffPair:
    return DiffPair(p, d, needs_grad)


def floatDiffPair(
    p: float = 0.0, d: float = 1.0, needs_grad: bool = True
) -> DiffPair:
    return diffPair(p, d, needs_grad)
