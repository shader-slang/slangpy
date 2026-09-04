# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any, Optional

from slangpy.core.enums import PrimType


class DiffPair:
    """
    Pair a primal Python value with its derivative seed or gradient result.

    ``DiffPair`` is primarily useful for differentiable scalar calls. Tensor
    workflows normally use :meth:`slangpy.Tensor.with_grads` instead. When a
    value is omitted, the primal defaults to ``0.0`` and the derivative defaults
    to the primal value's zero-initialized Python type.

    :param p: Primal value, or ``None`` to use ``0.0``.
    :param d: Derivative value, or ``None`` to construct a zero value matching
        the primal type.
    :param needs_grad: Whether functional dispatch should track a derivative.

    Example:

    .. code-block:: python

        import slangpy as spy

        pair = spy.DiffPair(2.0, 1.0)
        assert pair.primal == 2.0
        assert pair.grad == 1.0
    """

    def __init__(self, p: Optional[Any], d: Optional[Any], needs_grad: bool = True) -> None:
        """
        Create a primal and derivative pair.

        :param p: Primal value, or ``None`` to use ``0.0``.
        :param d: Derivative value, or ``None`` to construct a zero value matching
            the primal type.
        :param needs_grad: Whether functional dispatch should track a derivative.
        """
        super().__init__()
        self.primal = p if p is not None else 0.0
        """Primal value supplied to or returned from a differentiable call."""
        self.grad = d if d is not None else type(self.primal)()
        """Derivative seed supplied to or gradient returned from a call."""
        self.needs_grad = needs_grad
        """Whether functional dispatch should track this value's derivative."""

    def get(self, type: PrimType) -> Any:
        """
        Return the primal or derivative component.

        :param type: Component to return.
        :return: The selected Python value.
        """
        return self.primal if type == PrimType.primal else self.grad

    def set(self, type: PrimType, value: Any) -> None:
        """
        Replace the primal or derivative component.

        :param type: Component to replace.
        :param value: New Python value.
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
    Create a :class:`DiffPair`, inferring defaults from the primal value.

    :param p: Primal value, defaulting to ``0.0``.
    :param d: Derivative value, defaulting to a zero value of the primal type.
    :param needs_grad: Whether functional dispatch should track a derivative.
    :return: The new differential pair.

    Example:

    .. code-block:: python

        import slangpy as spy

        value = spy.diffPair(p=3.0)
        assert value.grad == 0.0
    """
    return DiffPair(p, d, needs_grad)


def floatDiffPair(p: float = 0.0, d: float = 1.0, needs_grad: bool = True) -> DiffPair:
    """
    Create a floating-point :class:`DiffPair` with a unit derivative by default.

    :param p: Primal floating-point value.
    :param d: Derivative seed, defaulting to ``1.0``.
    :param needs_grad: Whether functional dispatch should track a derivative.
    :return: The new differential pair.

    Example:

    .. code-block:: python

        import slangpy as spy

        seed = spy.floatDiffPair()
        assert seed.primal == 0.0
        assert seed.grad == 1.0
    """
    return diffPair(p, d, needs_grad)
