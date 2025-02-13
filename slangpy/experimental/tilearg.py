# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional, Union

from slangpy.backend import Texture, ResourceUsage, ShaderCursor
from slangpy.bindings import (PYTHON_TYPES, AccessType, Marshall, BindContext,
                              BoundVariable,
                              CodeGenBlock, Shape)
from slangpy.bindings.boundvariableruntime import BoundVariableRuntime
from slangpy.bindings.typeregistry import PYTHON_SIGNATURES, get_or_create_type
from slangpy.builtin.ndbuffer import NDBufferMarshall
from slangpy.builtin.texture import TextureMarshall
from slangpy.reflection import SlangProgramLayout, SlangType, TypeReflection
from slangpy.core.shapes import TShapeOrTuple
from slangpy.core.native import NativeObject, CallContext, get_value_signature, get_texture_shape, NativeNDBuffer
from slangpy.reflection.reflectiontypes import TYPE_OVERRIDES, ArrayType, VectorType, get_type_descriptor
from slangpy.types.buffer import NDBuffer


def _build_tile_arg_info(input: Any):
    if isinstance(input, Texture):
        return {'dims': 2}
    else:
        return None


class TileArg(NativeObject):
    """
    Passes the thread id as an argument to a SlangPy function.
    """

    def __init__(self, input: Any, stride: Union[TShapeOrTuple, int] = 1):
        super().__init__()
        if isinstance(input, Texture):
            shape = get_texture_shape(input, 0)
        elif isinstance(input, (NDBuffer, NativeNDBuffer)):
            shape = input.shape
        else:
            raise ValueError("TileArg input must be a Texture.")

        self.input = input
        self.shape = Shape(shape)

        if isinstance(stride, int):
            stride = (stride,)*len(self.shape)

        self.stride = Shape(stride) if stride is not None else Shape(tuple([1] * len(self.shape)))
        if not self.stride.concrete:
            raise ValueError("GridArg stride must be concrete.")
        if len(self.shape) != len(self.stride):
            raise ValueError("GridArg shape and stride must have the same length.")

        # Signature is based on the input signature
        self.slangpy_signature = f"[Tile-{get_value_signature(input)}]"

    @property
    def dims(self) -> int:
        return len(self.shape)

    @property
    def type(self):
        return type(self.input)


def tile(input: Any, stride: Union[TShapeOrTuple, int] = 1) -> TileArg:
    """
    Create a ThreadIdArg to pass to a SlangPy function, which passes the thread id.
    """
    return TileArg(input, stride)


class TileArgType(SlangType):
    """
    Slang type representation for a TileArg, typically used as ITile or IRWTile.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        args = program.get_resolved_generic_args(refl)
        assert args is not None
        assert len(args) == 2
        assert isinstance(args[0], SlangType)
        assert isinstance(args[1], int)
        super().__init__(program, refl,
                         element_type=args[0], local_shape=Shape((-1,)*args[1]))
        self.element_type: SlangType
        self._writable = "RW" in refl.name
        self._dims = args[1]


class TileArgMarshall(Marshall):
    """
    Marshall for tile arguments.
    """

    def __init__(self, layout: SlangProgramLayout, input_marshall: Marshall):
        super().__init__(layout)
        self.input_marshall = input_marshall
        self.slang_type = layout.require_type_by_name("Tile")

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name

        # Call data uses the fact that Tile<ContainerT> exposes an 'SPType' alias,
        # which maps to the corresponding accessor TileType<ContainerT>.
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", f"{binding.vector_type.full_name}.SPType")

    def write_shader_cursor_pre_dispatch(self, context: CallContext, binding: BoundVariableRuntime, cursor: ShaderCursor, value: TileArg, read_back: list):
        field = cursor[binding.variable_name]

        # Hack! we want the marshall to write to the TileType's data field, but it attempts to write to
        # a field based on its variable name. For now, temporarilly set it to 'data' and then restore it.
        n = binding.variable_name
        binding.variable_name = "data"
        try:
            self.input_marshall.write_shader_cursor_pre_dispatch(
                context, binding, field, value.input, read_back)
        finally:
            binding.variable_name = n

        # Also store stride+shape uniforms
        field["stride"] = value.stride.as_tuple()
        field["shape"] = value.shape.as_tuple()

    def get_shape(self, data: TileArg):
        # Shape is data shape divided by stride.
        return Shape(tuple([data.shape[i] // data.stride[i] for i in range(len(data.shape))]))

    def resolve_type(self, context: BindContext, bound_type: 'SlangType'):

        # Currently tile args only support being passed directly to an ITile or IRWTile parameter
        if not isinstance(bound_type, TileArgType):
            raise ValueError(
                f"Tile arg must be passed to a Tile parameter, but is being passed to {bound_type}")

        # Resolve to Tile<AccessorName>. Ideally would have a more general way to resolve this.
        if isinstance(self.input_marshall, TextureMarshall):
            nm = self.input_marshall.build_accessor_name(bound_type._writable)
        elif isinstance(self.input_marshall, NDBufferMarshall):
            nm = self.input_marshall.build_accessor_name(bound_type._writable)
        else:
            raise ValueError(f"Unsupported data type: {bound_type}")
        return bound_type.program.find_type_by_name(f"Tile<{nm}>")

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: 'SlangType'):
        # Resolve dimensionality by reading the 'SPDims' value from the type, which has been confirmed as a 'Tile' during resolution
        dims = get_type_descriptor(vector_target_type, "SPDims")
        assert isinstance(dims, int)
        return dims


TYPE_OVERRIDES["ITile"] = TileArgType
TYPE_OVERRIDES["IRWTile"] = TileArgType


def _get_or_create_python_type(layout: SlangProgramLayout, value: Any):
    assert isinstance(value, TileArg)
    input_marshall = get_or_create_type(layout, type(value.input), value.input)
    return TileArgMarshall(layout, input_marshall)


PYTHON_TYPES[TileArg] = _get_or_create_python_type
