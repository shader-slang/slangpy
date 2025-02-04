# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union
from slangpy.bindings import (PYTHON_TYPES, AccessType, Marshall, BindContext,
                              BoundVariable,
                              CodeGenBlock, Shape)
from slangpy.bindings.boundvariableruntime import BoundVariableRuntime
from slangpy.reflection import SlangProgramLayout, SlangType, TypeReflection
from slangpy.core.shapes import TShapeOrTuple
from slangpy.core.native import NativeObject, CallContext
from slangpy.reflection.reflectiontypes import TYPE_OVERRIDES, ArrayType, VectorType


class GridArg(NativeObject):
    """
    Passes the thread id as an argument to a SlangPy function.
    """

    def __init__(self, shape: Union[int, TShapeOrTuple], offset: Optional[TShapeOrTuple] = None, stride: Optional[TShapeOrTuple] = None):
        super().__init__()
        if isinstance(shape, int):
            shape = (-1,) * shape
        self.shape = Shape(shape)
        self.stride = Shape(stride) if stride is not None else Shape(tuple([1] * len(self.shape)))
        self.offset = Shape(offset) if offset is not None else Shape(tuple([0] * len(self.shape)))
        if not self.stride.concrete:
            raise ValueError("GridArg stride must be concrete.")
        if len(self.shape) != len(self.stride):
            raise ValueError("GridArg shape and stride must have the same length.")
        self.slangpy_signature = f"[{len(self.shape)}]"

    @property
    def dims(self) -> int:
        return len(self.shape)


def grid(shape: Union[int, TShapeOrTuple], offset: Optional[TShapeOrTuple] = None, stride: Optional[TShapeOrTuple] = None) -> GridArg:
    """
    Create a ThreadIdArg to pass to a SlangPy function, which passes the thread id.
    """
    return GridArg(shape, offset, stride)


class GridArgType(SlangType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        args = program.get_resolved_generic_args(refl)
        assert args is not None
        assert len(args) == 1
        assert isinstance(args[0], int)
        super().__init__(program, refl,
                         local_shape=Shape((-1,)*args[0]))


class GridArgMarshall(Marshall):
    def __init__(self, layout: SlangProgramLayout, dims: int):
        super().__init__(layout)
        self.dims = dims
        st = layout.find_type_by_name(f"GridArg<{self.dims}>")
        if st is None:
            raise ValueError(
                f"Could not find GridArgType slang type. This usually indicates the gradarg module has not been imported.")
        self.slang_type = st

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def create_calldata(self, context: CallContext, binding: BoundVariableRuntime, data: GridArg) -> Any:
        access = binding.access
        if access[0] == AccessType.read:
            return {
                'offset': data.offset,
                'stride': data.stride
            }

    def get_shape(self, data: GridArg):
        # For each dimension, if a concrete size is known, shape is size/stride, otherwise it is
        # left as 1 and broadcast to every dimension
        t = [data.shape[i]//data.stride[i] if data.shape[i] >= 0 else 1 for i in range(self.dims)]
        return Shape(tuple(t))

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        if isinstance(bound_type, ArrayType):
            return context.layout.array_type(context.layout.scalar_type(TypeReflection.ScalarType.int32), self.dims)
        elif isinstance(bound_type, VectorType):
            return context.layout.vector_type(TypeReflection.ScalarType.int32, self.dims)
        else:
            raise ValueError(
                f"Grid arguments must be passed to a function with an array or vector type, not {bound_type}")

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        return self.dims


TYPE_OVERRIDES["GridArg"] = GridArgType
PYTHON_TYPES[GridArg] = lambda l, x: GridArgMarshall(l, x.dims)
