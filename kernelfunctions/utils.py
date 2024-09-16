from typing import Any, Union
from numpy import ndarray
from sgl import ProgramLayout, TypeLayoutReflection, TypeReflection

from kernelfunctions.backend import Buffer


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


def to_numpy(buffer: Buffer):
    np = buffer.to_numpy()
    if isinstance(np, ndarray):
        return np
    else:
        raise ValueError("Buffer did not return an ndarray")


def find_type_layout_for_buffer(program_layout: ProgramLayout, slang_type: Union[str, TypeReflection, TypeLayoutReflection]):
    if isinstance(slang_type, str):
        slang_type_name = slang_type
    elif isinstance(slang_type, (TypeReflection, TypeLayoutReflection)):
        slang_type_name = slang_type.name
    buffer_type = program_layout.find_type_by_name(f"StructuredBuffer<{slang_type_name}>")
    buffer_layout = program_layout.get_type_layout(buffer_type)
    return buffer_layout.element_type_layout
