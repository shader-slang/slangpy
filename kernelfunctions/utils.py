from typing import Any, Optional
from numpy import ndarray

import sgl

from kernelfunctions.callsignature import BasePythonTypeMarshal, register_python_type
from kernelfunctions.typemappings import is_valid_scalar_type_conversion


class ScalarRef:
    def __init__(self, init_value: Any):
        super().__init__()
        self.value = init_value


class ScalarRefMarshall(BasePythonTypeMarshal):
    def __init__(self, val: ScalarRef):
        super().__init__(ScalarRef)
        self.element_type = type(val.value)
        self.shape = (1,)

    def is_compatible(self, slang_type: sgl.TypeReflection) -> bool:
        return slang_type.kind == sgl.TypeReflection.Kind.scalar and is_valid_scalar_type_conversion(slang_type.scalar_type, self.element_type)


register_python_type(ScalarRef, lambda x: ScalarRefMarshall(x),
                     lambda stream, x: stream.write(type(x.value).__name + "\n"))


class ScalarDiffPair:
    def __init__(self, p: Optional[Any], d: Optional[Any], needs_grad: bool = True):
        super().__init__()
        self.primal = p if p is not None else 0.0
        self.grad = d if d is not None else 0.0
        self.needs_grad = needs_grad


def intRef(init_value: int = 0) -> ScalarRef:
    return ScalarRef(int(init_value))


def floatRef(init_value: float = 0.0) -> ScalarRef:
    return ScalarRef(float(init_value))


def diffPair(
    p: Optional[Any] = None, d: Optional[Any] = None, needs_grad: bool = True
) -> ScalarDiffPair:
    return ScalarDiffPair(p, d, needs_grad)


def is_differentiable_buffer(val: Any):
    grad = getattr(val, "grad", None)
    if grad is None:
        return False
    needs_grad = getattr(val, "needs_grad", None)
    if needs_grad is None:
        return False
    if not isinstance(needs_grad, bool):
        return False
    return needs_grad


def to_numpy(buffer: sgl.Buffer):
    np = buffer.to_numpy()
    if isinstance(np, ndarray):
        return np
    else:
        raise ValueError("Buffer did not return an ndarray")
