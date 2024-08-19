from typing import Any
from numpy import ndarray

import sgl


def is_differentiable_buffer(val: Any):
    grad = val.getattr("grad", None)
    if grad is None:
        return False
    needs_grad = val.getattr("needs_grad", None)
    if needs_grad is None:
        return False
    if not isinstance(needs_grad, bool):
        return False
    return True


def to_numpy(buffer: sgl.Buffer):
    np = buffer.to_numpy()
    if isinstance(np, ndarray):
        return np
    else:
        raise ValueError("Buffer did not return an ndarray")
