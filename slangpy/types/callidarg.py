# SPDX-License-Identifier: Apache-2.0

from slangpy.bindings import (PYTHON_TYPES, AccessType, Marshall, BindContext,
                              BoundVariable,
                              CodeGenBlock, Shape)
from slangpy.reflection import SlangProgramLayout, SlangType, TypeReflection

from typing import Any


class CallIdArg:
    """
    Passes the thread id as an argument to a SlangPy function.
    """

    def __init__(self, dims_or_shape: int | tuple[int, ...] = 3):
        super().__init__()
        if isinstance(dims_or_shape, tuple):
            self.shape = Shape(dims_or_shape)
            self.dims = len(dims_or_shape)
        elif isinstance(dims_or_shape, int):
            self.shape = None
            self.dims = dims_or_shape
        else:
            raise ValueError("Argument to thread_id must be int or tuple")

    @property
    def slangpy_signature(self) -> str:
        return f"[{self.dims}]"


def call_id(dims_or_shape: int | tuple[int, ...]):
    """
    Create a ThreadIdArg to pass to a SlangPy function, which passes the thread id.
    """
    return CallIdArg(dims_or_shape)


class CallIdArgMarshall(Marshall):
    def __init__(self, layout: SlangProgramLayout, dims: int):
        super().__init__(layout)
        self.dims = dims
        st = layout.find_type_by_name(f"CallIdArg<{self.dims}>")
        if st is None:
            raise ValueError(
                f"Could not find CallIdArg slang type. This usually indicates the callidarg module has not been imported.")
        self.slang_type = st

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        return context.layout.vector_type(TypeReflection.ScalarType.uint32, self.dims)

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        return self.dims

    def get_shape(self, value: Any = None) -> Shape:
        if isinstance(value, CallIdArg) and value.shape:
            return value.shape
        else:
            return self.concrete_shape


PYTHON_TYPES[CallIdArg] = lambda l, x: CallIdArgMarshall(l, x.dims)
