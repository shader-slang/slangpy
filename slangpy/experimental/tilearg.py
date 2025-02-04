# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

from slangpy.backend import Texture
from slangpy.bindings import (PYTHON_TYPES, AccessType, Marshall, BindContext,
                              BoundVariable,
                              CodeGenBlock, Shape)
from slangpy.bindings.boundvariableruntime import BoundVariableRuntime
from slangpy.reflection import SlangProgramLayout, SlangType, TypeReflection
from slangpy.core.shapes import TShapeOrTuple
from slangpy.core.native import NativeObject, CallContext
from slangpy.reflection.reflectiontypes import TYPE_OVERRIDES, ArrayType, VectorType


class TileArg(NativeObject):
    """
    Passes the thread id as an argument to a SlangPy function.
    """

    def __init__(self, input: Any, tile_size: Union[TShapeOrTuple, int] = 1, stride: Union[TShapeOrTuple, int] = 1):
        super().__init__()
        if isinstance(input, Texture):
            shape = (input.height, input.width)
        else:
            raise ValueError("TileArg input must be a Texture.")

        self.input = input
        self.shape = Shape(shape)

        if isinstance(tile_size, int):
            tile_size = (tile_size,)*len(self.shape)
        if isinstance(stride, int):
            stride = (stride,)*len(self.shape)

        self.stride = Shape(stride) if stride is not None else Shape(tuple([1] * len(self.shape)))
        if not self.stride.concrete:
            raise ValueError("GridArg stride must be concrete.")
        if len(self.shape) != len(self.stride):
            raise ValueError("GridArg shape and stride must have the same length.")
        self.slangpy_signature = f"[{len(self.shape)},{type(self.input).__name__}]"

    @property
    def dims(self) -> int:
        return len(self.shape)

    @property
    def type(self):
        return "Texture2DTile"


def tile(input: Any, tile_size: Union[TShapeOrTuple, int] = 1, stride: Union[TShapeOrTuple, int] = 1) -> TileArg:
    """
    Create a ThreadIdArg to pass to a SlangPy function, which passes the thread id.
    """
    return TileArg(input, tile_size, stride)


class TileArgType(SlangType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        if refl.name == "Texture2DTile":
            dims = 2
        else:
            raise ValueError(f"TileArgType does not support type {refl.name}")
        super().__init__(program, refl,
                         local_shape=Shape((-1,)*dims))


class TileArgMarshall(Marshall):
    def __init__(self, layout: SlangProgramLayout, dims: int, input_type: str):
        super().__init__(layout)
        self.dims = dims
        st = layout.find_type_by_name(f"Texture2DTileType<float4>")
        if st is None:
            raise ValueError(
                f"Could not find GridArgType slang type. This usually indicates the gradarg module has not been imported.")
        self.slang_type = st

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def create_calldata(self, context: CallContext, binding: BoundVariableRuntime, data: TileArg) -> Any:
        access = binding.access
        if access[0] == AccessType.read:
            return {
                'texture': data.input,
                'stride': data.stride
            }

    def get_shape(self, data: TileArg):
        return data.shape

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):
        return bound_type

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        return 2


TYPE_OVERRIDES["Texture2DTile"] = TileArgType
PYTHON_TYPES[TileArg] = lambda l, x: TileArgMarshall(l, x.dims, x.type)
