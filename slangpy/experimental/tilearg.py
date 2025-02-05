# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

from slangpy.backend import Texture, ResourceUsage
from slangpy.bindings import (PYTHON_TYPES, AccessType, Marshall, BindContext,
                              BoundVariable,
                              CodeGenBlock, Shape)
from slangpy.bindings.boundvariableruntime import BoundVariableRuntime
from slangpy.bindings.typeregistry import PYTHON_SIGNATURES, get_or_create_type
from slangpy.builtin.texture import TextureMarshall
from slangpy.reflection import SlangProgramLayout, SlangType, TypeReflection
from slangpy.core.shapes import TShapeOrTuple
from slangpy.core.native import NativeObject, CallContext
from slangpy.reflection.reflectiontypes import TYPE_OVERRIDES, ArrayType, VectorType


def _build_tile_arg_info(input: Any):
    if isinstance(input, Texture):
        return {'dims': 2}
    else:
        return None


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
        return type(self.input)


def tile(input: Any, tile_size: Union[TShapeOrTuple, int] = 1, stride: Union[TShapeOrTuple, int] = 1) -> TileArg:
    """
    Create a ThreadIdArg to pass to a SlangPy function, which passes the thread id.
    """
    return TileArg(input, tile_size, stride)


tile_arg_info = {
    "Texture2DTile": {'dims': 2},
    "RWTexture2DTile": {'dims': 2}
}


class TileArgType(SlangType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        info = tile_arg_info.get(refl.name)
        if info is None:
            raise ValueError(f"TileArgType does not support type {refl.name}")

        dims = info['dims']

        super().__init__(program, refl,
                         local_shape=Shape((-1,)*dims))


class TileArgMarshall(Marshall):
    def __init__(self, layout: SlangProgramLayout, input_marshall: Marshall):
        super().__init__(layout)
        st = None
        if isinstance(input_marshall, TextureMarshall):
            self.dims = input_marshall.texture_dims
            self.writable = input_marshall.is_writable

            st = layout.find_type_by_name(
                input_marshall.slang_type.full_name.replace("<", "Tile<", 1))
        else:
            raise ValueError("TileArgMarshall input must be a TextureMarshall.")

        if st is None:
            raise ValueError(
                f"Could not find Tile slang type. This usually indicates the tile module has not been imported.")
        self.slang_type = st

        at = layout.find_type_by_name(self.slang_type.full_name.replace("Tile<", "TileType<", 1))
        if at is None:
            raise ValueError(
                f"Could not find TileType slang type. This usually indicates the tile module has not been imported.")

        self.accessor_type = at

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name

        if binding.call_dimensionality != self.dims:
            raise ValueError(
                f"TileArg call dimensionality {binding.call_dimensionality} does not match expected {self.dims}")

        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", self.accessor_type.full_name)

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
TYPE_OVERRIDES["RWTexture2DTile"] = TileArgType
PYTHON_TYPES[TileArg] = lambda l, x: TileArgMarshall(l, x.dims, x.type)


def _get_or_create_python_type(layout: SlangProgramLayout, value: Any):
    assert isinstance(value, TileArg)
    input_marshall = get_or_create_type(layout, type(value.input), value.input)
    return TileArgMarshall(layout, input_marshall)


def _get_or_create_python_signature(value: Any):
    assert isinstance(value, TileArg)
    # TODO: Need to expose getting signature of input to do this properly
    return f"[{type(value.input).__name__}]"


PYTHON_TYPES[TileArg] = _get_or_create_python_type
PYTHON_SIGNATURES[TileArg] = _get_or_create_python_signature
