
from typing import Any, Optional


class DiffPair:
    def __init__(self, p: Optional[Any], d: Optional[Any], needs_grad: bool = True):
        super().__init__()
        self.primal = p if p is not None else 0.0
        self.grad = d if d is not None else 0.0
        self.needs_grad = needs_grad


def diffPair(
    p: Optional[Any] = None, d: Optional[Any] = None, needs_grad: bool = True
) -> DiffPair:
    return DiffPair(p, d, needs_grad)


def floatDiffPair(
    p: float = 0.0, d: float = 1.0, needs_grad: bool = True
) -> DiffPair:
    return diffPair(p, d, needs_grad)
