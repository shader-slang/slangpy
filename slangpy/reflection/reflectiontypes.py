# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Callable, Optional, Union, cast

import numpy as np

from slangpy.core.enums import IOType
from slangpy.core.native import NativeSlangType, Shape

from slangpy.backend import (FunctionReflection, ModifierID, ProgramLayout,
                             ResourceUsage, TypeLayoutReflection)
from slangpy.backend import TypeReflection
from slangpy.backend import TypeReflection as TR
from slangpy.backend import VariableReflection

scalar_names = {
    TR.ScalarType.void: "void",
    TR.ScalarType.bool: "bool",
    TR.ScalarType.int8: "int8_t",
    TR.ScalarType.int16: "int16_t",
    TR.ScalarType.int32: "int",
    TR.ScalarType.int64: "int64_t",
    TR.ScalarType.uint8: "uint8_t",
    TR.ScalarType.uint16: "uint16_t",
    TR.ScalarType.uint32: "uint",
    TR.ScalarType.uint64: "uint64_t",
    TR.ScalarType.float16: "half",
    TR.ScalarType.float32: "float",
    TR.ScalarType.float64: "double",
}

SIGNED_INT_TYPES = {TR.ScalarType.int8, TR.ScalarType.int16,
                    TR.ScalarType.int32, TR.ScalarType.int64}
UNSIGNED_INT_TYPES = {TR.ScalarType.uint8, TR.ScalarType.uint16,
                      TR.ScalarType.uint32, TR.ScalarType.uint64}
FLOAT_TYPES = {TR.ScalarType.float16, TR.ScalarType.float32, TR.ScalarType.float64}
BOOL_TYPES = {TR.ScalarType.bool}
INT_TYPES = SIGNED_INT_TYPES | UNSIGNED_INT_TYPES

SCALAR_TYPE_TO_NUMPY_TYPE = {
    TR.ScalarType.int8: np.int8,
    TR.ScalarType.int16: np.int16,
    TR.ScalarType.int32: np.int32,
    TR.ScalarType.int64: np.int64,
    TR.ScalarType.uint8: np.uint8,
    TR.ScalarType.uint16: np.uint16,
    TR.ScalarType.uint32: np.uint32,
    TR.ScalarType.uint64: np.uint64,
    TR.ScalarType.float16: np.float16,
    TR.ScalarType.float32: np.float32,
    TR.ScalarType.float64: np.float64,
    TR.ScalarType.bool: np.int8,
}

NUMPY_TYPE_TO_SCALAR_TYPE = {np.dtype(v): k for k, v in SCALAR_TYPE_TO_NUMPY_TYPE.items()}

texture_names = {
    TR.ResourceShape.texture_1d: "Texture1D",
    TR.ResourceShape.texture_2d: "Texture2D",
    TR.ResourceShape.texture_3d: "Texture3D",
    TR.ResourceShape.texture_cube: "TextureCube",
    TR.ResourceShape.texture_1d_array: "Texture1DArray",
    TR.ResourceShape.texture_2d_array: "Texture2DArray",
    TR.ResourceShape.texture_cube_array: "TextureCubeArray",
    TR.ResourceShape.texture_2d_multisample: "Texture2DMS",
    TR.ResourceShape.texture_2d_multisample_array: "Texture2DMSArray",
}
texture_dims = {
    TR.ResourceShape.texture_1d: 1,
    TR.ResourceShape.texture_2d: 2,
    TR.ResourceShape.texture_3d: 3,
    TR.ResourceShape.texture_cube: 3,
    TR.ResourceShape.texture_1d_array: 2,
    TR.ResourceShape.texture_2d_array: 3,
    TR.ResourceShape.texture_cube_array: 4,
    TR.ResourceShape.texture_2d_multisample: 2,
    TR.ResourceShape.texture_2d_multisample_array: 3,
}


def is_float(kind: TR.ScalarType):
    return kind in (TR.ScalarType.float16, TR.ScalarType.float32, TR.ScalarType.float64)


class SlangLayout:
    """
    Size, alignment and stride of a type.
    """

    def __init__(self, tlr: TypeLayoutReflection):
        super().__init__()
        self._tlr = tlr

    @property
    def reflection(self) -> TypeLayoutReflection:
        """
        Underlying SGL TypeLayoutReflection for this layout.
        """
        return self._tlr

    @property
    def size(self) -> int:
        """
        Size in bytes of the type. Note: when calculating size in
        a buffer, use the `stride` property instead.
        """
        return self._tlr.size

    @property
    def alignment(self) -> int:
        """
        Alignment in bytes of the type.
        """
        return self._tlr.alignment

    @property
    def stride(self) -> int:
        """
        Stride in bytes of the type.
        """
        return self._tlr.stride


class SlangType(NativeSlangType):
    """
    Base class for all Slang types.
    """

    def __init__(self,
                 program: SlangProgramLayout,
                 refl: TypeReflection,
                 element_type: Optional[SlangType] = None,
                 local_shape: Shape = Shape(None)):
        super().__init__()

        #: Underlying SGL TypeReflection for this type.
        self.type_reflection = refl

        self._program = program
        self._element_type = element_type

        self._cached_fields: Optional[dict[str, SlangField]] = None
        self._cached_differential: Optional[SlangType] = None
        self._cached_uniform_layout: Optional[SlangLayout] = None
        self._cached_buffer_layout: Optional[SlangLayout] = None

        # Native shape storage
        if self._element_type == self:
            self.shape = local_shape
        elif local_shape.valid and self._element_type is not None:
            self.shape = local_shape + self._element_type.shape
        else:
            self.shape = local_shape

    def on_hot_reload(self, refl: TypeReflection):
        """
        Called when the type reflection is hot reloaded. Stores updated reflection and clears 
        cached data.
        """
        self.type_reflection = refl
        self._cached_fields = None
        self._cached_differential = None
        self._cached_uniform_layout = None
        self._cached_buffer_layout = None

    @property
    def program(self) -> SlangProgramLayout:
        """
        Program layout this type is part of.
        """
        return self._program

    @property
    def name(self) -> str:
        """
        Short name of this type. For generics, this
        will not include the generic arguments.
        """
        return self.type_reflection.name

    @property
    def full_name(self) -> str:
        """
        Fully qualified name of this type.
        """
        return self.type_reflection.full_name

    @property
    def element_type(self) -> Optional[SlangType]:
        """
        Element type for arrays, vectors, matrices, etc.
        """
        return self._element_type

    @property
    def fields(self) -> dict[str, SlangField]:
        """
        Fields of this type. For non-struct types, this will be empty.
        """
        return self._get_fields()

    @property
    def differentiable(self) -> bool:
        """
        Whether this type is differentiable.
        """
        return self._get_differential() is not None

    @property
    def derivative(self) -> SlangType:
        """
        Get derivative type of this type.
        """
        if self.differentiable:
            res = self._get_differential()
            assert res is not None
            return res
        else:
            raise ValueError(f"Type {self.full_name} is not differentiable")

    @property
    def num_dims(self) -> int:
        """
        Number of dimensions of this type.
        """
        return len(self.shape)

    @property
    def uniform_layout(self) -> SlangLayout:
        """
        Get the layout of this type when used as a uniform / in a constant buffer.
        """
        if self._cached_uniform_layout is None:
            sl = self._program.program_layout.get_type_layout(self.type_reflection)
            if sl is None:
                raise ValueError(
                    f"Unable to get layout for {self.full_name}. This can happen if the type is defined in a module that isn't accesible during type resolution.")
            self._cached_uniform_layout = SlangLayout(sl)
        return self._cached_uniform_layout

    @property
    def buffer_layout(self) -> SlangLayout:
        """
        Get the layout of this type when used in a structured buffer.
        """
        if self._cached_buffer_layout is None:
            buffer_type = self._program.program_layout.find_type_by_name(
                f"StructuredBuffer<{self.full_name}>")
            if buffer_type is None:
                raise ValueError(
                    f"Unable to get layout for {self.full_name}. This can happen if the type is defined in a module that isn't accesible during type resolution.")
            buffer_layout = self._program.program_layout.get_type_layout(buffer_type)
            self._cached_buffer_layout = SlangLayout(buffer_layout.element_type_layout)
        return self._cached_buffer_layout

    def build_differential_type(self) -> Optional[SlangType]:
        """
        Overridable function to build the differential type for this type.
        """
        return self._program.find_type_by_name(self.full_name + ".Differential")

    def build_fields(self) -> dict[str, Union[SlangType, SlangField]]:
        """
        Overridable function to build fields for this type.
        """
        return {}

    def _get_differential(self) -> Optional[SlangType]:
        if self._cached_differential is None:
            self._cached_differential = self.build_differential_type()
        return self._cached_differential

    def _get_fields(self) -> dict[str, SlangField]:
        if self._cached_fields is None:
            def make_field(field_name: str, field_val: Union[SlangType, SlangField, VariableReflection]) -> SlangField:
                if isinstance(field_val, SlangType):
                    return SlangField(self._program, field_val, field_name, set())
                elif isinstance(field_val, VariableReflection):
                    return SlangField(self._program, refl=field_val)
                else:
                    return field_val
            fields = self.build_fields()
            self._cached_fields = {name: make_field(
                name, value) for name, value in fields.items()}
        return self._cached_fields


class VoidType(SlangType):
    """
    Represents the void type.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl)


class ScalarType(SlangType):
    """
    Represents any scalar type such as int/float/bool. See `sgl.TypeReflection.ScalarType`.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        assert refl.scalar_type not in (TR.ScalarType.none, TR.ScalarType.void)
        super().__init__(program, refl, element_type=self, local_shape=Shape())

    @property
    def slang_scalar_type(self) -> TR.ScalarType:
        """
        Slang scalar type id.
        """
        return self.type_reflection.scalar_type


class VectorType(SlangType):
    """
    Represents a vector type such as int3/float3/vector<float,3> etc.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        element_type = program.scalar_type(refl.scalar_type)
        dims = refl.col_count

        super().__init__(program, refl,
                         element_type=element_type,
                         local_shape=Shape((dims,)))

    @property
    def num_elements(self) -> int:
        """
        Number of elements in the vector.
        """
        return self.shape[0]

    @property
    def scalar_type(self) -> ScalarType:
        """
        Scalar element type of the vector.
        """
        return cast(ScalarType, self.element_type)

    @property
    def slang_scalar_type(self) -> TR.ScalarType:
        """
        Slang scalar element type id.
        """
        assert isinstance(self.element_type, ScalarType)
        return self.element_type.slang_scalar_type

    def build_fields(self):
        """
        Build fields for this vector type generates the x/y/z/w fields.
        """
        names = ['x', 'y', 'z', 'w']
        return {names[i]: self.scalar_type for i in range(self.num_elements)}


class MatrixType(SlangType):
    """
    Represents a matrix type such as float3x3/matrix<float,3,3> etc.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl,
                         element_type=program.vector_type(
                             refl.scalar_type, refl.col_count),
                         local_shape=Shape((refl.row_count,)))

    @property
    def rows(self) -> int:
        """
        Number of rows in the matrix.
        """
        return self.shape[0]

    @property
    def cols(self) -> int:
        """
        Number of columns in the matrix.
        """
        return self.shape[1]

    @property
    def scalar_type(self) -> ScalarType:
        """
        Scalar element type of the matrix.
        """
        assert isinstance(self.element_type, VectorType)
        return cast(ScalarType, self.element_type.scalar_type)

    @property
    def slang_scalar_type(self) -> TR.ScalarType:
        """
        Slang scalar element type id.
        """
        return self.scalar_type.slang_scalar_type


class ArrayType(SlangType):
    """
    Represents an array type such as float[3]/array<float,3> etc.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl, program.find_type(
            refl.element_type), local_shape=Shape((refl.element_count,)))

    @property
    def num_elements(self) -> int:
        """
        Number of elements in the array.
        """
        return self.shape[0]


def is_matching_array_type(a: SlangType, b: SlangType) -> bool:
    """
    Helper to check if 2 array types are compatible. This handles
    the situation in which one or both of the array types have
    unknown dimensions. In this case, the dimensions are considered
    compatible.
    """
    if not isinstance(a, ArrayType) or not isinstance(b, ArrayType):
        return False
    if a.element_type != b.element_type:
        return False
    if a.num_elements > 0 and b.num_elements > 0:
        return a.num_elements == b.num_elements
    return True


class StructType(SlangType):
    """
    Represents a struct type.They are treated as opaque types
    with no element type and 0D local shape.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        # An opaque struct has no element type, but like a normal scalar has a 0D local shape
        super().__init__(program, refl, local_shape=Shape())

    def build_fields(self):
        return {field.name: field for field in self.type_reflection.fields}


class InterfaceType(SlangType):
    """
    Represents an interface type.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl)


class ResourceType(SlangType):
    """
    Base class for all resource types such as textures, buffers, etc.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @property
    def resource_shape(self) -> TR.ResourceShape:
        """
        Resource shape of this resource. See `sgl.TypeReflection.ResourceShape`.
        """
        return self.type_reflection.resource_shape

    @property
    def resource_access(self) -> TR.ResourceAccess:
        """
        Resource access of this resource. See `sgl.TypeReflection.ResourceAccess`.
        """
        return self.type_reflection.resource_access

    @property
    def writable(self) -> bool:
        """
        Whether this resource is writable.
        """
        if self.resource_access == TR.ResourceAccess.read_write:
            return True
        elif self.resource_access == TR.ResourceAccess.read:
            return False
        else:
            raise ValueError("Resource is neither read_write or read")

    @property
    def usage(self) -> ResourceUsage:
        """
        Supported shader resource usage.
        """
        if self.writable:
            return ResourceUsage.unordered_access
        else:
            return ResourceUsage.shader_resource


class TextureType(ResourceType):
    """
    Represents one of the texture types, including textures, texture arrays and cube maps.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):

        self.texture_dims = texture_dims[refl.resource_shape]

        super().__init__(program, refl,
                         element_type=program.find_type(refl.resource_result_type),
                         local_shape=Shape((-1,)*self.texture_dims,))


class StructuredBufferType(ResourceType):
    """
    Represents a structured buffer type.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):

        super().__init__(program, refl,
                         element_type=program.find_type(refl.resource_result_type),
                         local_shape=Shape((-1,)))


class ByteAddressBufferType(ResourceType):
    """
    Represents a byte address buffer type.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl,
                         element_type=program.scalar_type(TR.ScalarType.uint8),
                         local_shape=Shape((-1,)))


class DifferentialPairType(SlangType):
    """
    Represents a Slang differential pair.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl, local_shape=Shape())

        args = program.get_resolved_generic_args(refl)
        assert args is not None
        assert len(args) == 1
        assert isinstance(args[0], SlangType)
        assert args[0].differentiable
        self.primal = args[0]

    def build_differential_type(self):
        """
        Differential type for a differential pair is `DifferentialPair<Primal.Derivative>`.
        """
        return self._program.find_type_by_name("DifferentialPair<" + self.primal.derivative.full_name + ">")


class RaytracingAccelerationStructureType(SlangType):
    """
    Represents a raytracing acceleration structure type.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl, local_shape=Shape())


class SamplerStateType(SlangType):
    """
    Represents a sampler type.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl, local_shape=Shape())


class UnhandledType(SlangType):
    """
    Represents an unhandled type.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl)

    @property
    def kind(self) -> TR.Kind:
        return self.type_reflection.kind


class SlangFunction:
    """
    Represents a Slang function.
    """

    def __init__(self, program: SlangProgramLayout, refl: FunctionReflection, this: Optional[SlangType], full_name: Optional[str]):
        super().__init__()
        self._this = this
        self._reflection = refl
        self._program = program
        self._cached_parameters: Optional[tuple[SlangParameter, ...]] = None
        self._cached_return_type: Optional[SlangType] = None

        if full_name is None:
            full_name = refl.name
        self._full_name = full_name

    def on_hot_reload(self, refl: FunctionReflection):
        self._reflection = refl
        self._cached_parameters = None
        self._cached_return_type = None

    @property
    def reflection(self) -> FunctionReflection:
        """
        Underlying SGL FunctionReflection for this function.
        """
        return self._reflection

    @property
    def name(self) -> str:
        """
        Name of this function.
        """
        return self._reflection.name

    @property
    def full_name(self) -> str:
        """
        Fully qualified name of this function, including generic arguments (if any).
        """
        return self._full_name

    @property
    def this(self) -> Optional[SlangType]:
        """
        Type that this function is a method of, or None if it is a global function.
        """
        return self._this

    @property
    def return_type(self) -> Optional[SlangType]:
        """
        Return type of this function.
        """
        if self._cached_return_type is None and self._reflection.return_type is not None:
            self._cached_return_type = self._program.find_type(
                self._reflection.return_type)
        return self._cached_return_type

    @property
    def parameters(self) -> tuple[SlangParameter, ...]:
        """
        Parameters of this function.
        """
        if self._cached_parameters is None:
            ref_params = [x for x in self._reflection.parameters]
            self._cached_parameters = tuple([
                SlangParameter(self._program, param, i) for i, param in enumerate(ref_params)])
        return self._cached_parameters

    @property
    def have_return_value(self) -> bool:
        """
        Return true if this function doesn't return void.
        """
        return not isinstance(self.return_type, VoidType)

    @property
    def differentiable(self) -> bool:
        """
        Whether this function is differentiable - i.e. does it have the differentiable
        attribute in slang.
        """
        return self.reflection.has_modifier(ModifierID.differentiable)

    @property
    def mutating(self) -> bool:
        """
        Whether this function is mutating - i.e. does it have the mutating
        attribute in slang. Only relevant for type methods.
        """
        return self.reflection.has_modifier(ModifierID.mutating)


class BaseSlangVariable:
    """
    Base class for slang variables (fields and parameters).
    """

    def __init__(self, program: SlangProgramLayout, slang_type: SlangType, name: str, modifiers: set[ModifierID]):
        super().__init__()
        self._program = program
        self._type = slang_type
        self._name = name
        self._modifiers = modifiers

    @property
    def type(self) -> SlangType:
        """
        Type of this variable.
        """
        return self._type

    @property
    def name(self) -> str:
        """
        Name of this variable.
        """
        return self._name

    @property
    def modifiers(self) -> set[ModifierID]:
        """
        Slang modifiers for this variable.
        """
        return self._modifiers

    @property
    def declaration(self) -> str:
        """
        String representation of the declaration of this variable.
        """
        mods = [str(mod) for mod in self.modifiers]
        return " ".join(mods + [f"{self.type.full_name} {self.name}"])

    @property
    def io_type(self) -> IOType:
        """
        Calculate IOType of this variable (in/inout/out) based on modifiers.
        """
        have_in = ModifierID.inn in self.modifiers
        have_out = ModifierID.out in self.modifiers
        have_inout = ModifierID.inout in self.modifiers

        if (have_in and have_out) or have_inout:
            return IOType.inout
        elif have_out:
            return IOType.out
        else:
            return IOType.inn

    @property
    def no_diff(self) -> bool:
        """
        Whether this variable has the no_diff modifier.
        """
        return ModifierID.nodiff in self.modifiers

    @property
    def differentiable(self) -> bool:
        """
        Whether this variable is differentiable. Requires type
        to be differentiable + not have the no_diff modifier.
        """
        if self.no_diff:
            return False
        return self.type.differentiable

    @property
    def derivative(self) -> SlangType:
        """
        Get derivative type of this variable.
        """
        if self.differentiable:
            return self.type.derivative
        else:
            raise ValueError(f"Variable {self.name} is not differentiable")


class SlangField(BaseSlangVariable):
    """
    Variable that represents a field in a struct, typically constructed when a type's
    fields are enumerated.
    """

    def __init__(self, program: SlangProgramLayout, slang_type: Optional[SlangType] = None, name: Optional[str] = None, modifiers: Optional[set[ModifierID]] = None, refl: Optional[VariableReflection] = None):

        if not ((slang_type is not None) ^ (refl is not None)):
            raise ValueError("Must specify either type+name OR refl")

        if refl is not None:
            assert name is None
            assert slang_type is None
            assert modifiers is None
            slang_type = program.find_type(refl.type)
            name = refl.name
            modifiers = {mod for mod in ModifierID if refl.has_modifier(mod)}
        else:
            assert name is not None
            assert slang_type is not None
            if modifiers is None:
                modifiers = set()

        super().__init__(program, slang_type, name, modifiers)
        self._reflection = refl


class SlangParameter(BaseSlangVariable):
    """
    Variable that represents a parameter in a function, typically constructed when a function's
    parameters are enumerated.
    """

    def __init__(self, program: SlangProgramLayout, refl: VariableReflection, index: int):
        slang_type = program.find_type(refl.type)
        name = refl.name
        modifiers = {mod for mod in ModifierID if refl.has_modifier(mod)}
        super().__init__(program, slang_type, name, modifiers)
        self._reflection = refl

        self._index = index
        self._has_default = False  # TODO: Work out defaults

    @property
    def index(self) -> int:
        """
        Index of this parameter in the function.
        """
        return self._index

    @property
    def has_default(self) -> bool:
        """
        Whether this parameter has a default value.
        """
        return self._has_default


class SlangProgramLayout:
    """
    Program layout for a module. This is the main entry point for any reflection queries,
    and provides a way to look up types, functions, etc. Typically this is accessed via
    the loaded module with `module.layout`, however it can be constructed explicitly 
    from an sgl ProgramLayout.
    """

    def __init__(self, program_layout: ProgramLayout, slangpy_program_layout: Optional[ProgramLayout] = None):
        super().__init__()
        assert isinstance(program_layout, ProgramLayout)
        self.program_layout = program_layout
        self.slangpy_program_layout = slangpy_program_layout
        self._types_by_name: dict[str, SlangType] = {}
        self._types_by_reflection: dict[TypeReflection, SlangType] = {}
        self._functions_by_name: dict[str, SlangFunction] = {}
        self._functions_by_reflection: dict[FunctionReflection, SlangFunction] = {}

    def on_hot_reload(self, program_layout: ProgramLayout):
        if program_layout == self.program_layout:
            return
        self.program_layout = program_layout

        new_types_by_name: dict[str, SlangType] = {}
        new_types_by_reflection: dict[TypeReflection, SlangType] = {}

        # Re-lookup all types.
        for name, type in self._types_by_name.items():
            trefl = program_layout.find_type_by_name(name)
            assert trefl is not None
            type.on_hot_reload(trefl)
            new_types_by_name[name] = type
            new_types_by_reflection[trefl] = type

        self._types_by_name = new_types_by_name
        self._types_by_reflection = new_types_by_reflection

        new_functions_by_name: dict[str, SlangFunction] = {}
        new_functions_by_reflection: dict[FunctionReflection, SlangFunction] = {}

        # Re-lookup all functions.
        for name, func in self._functions_by_name.items():
            if '::' in name:
                idx = name.index('::')
                type_name = name[:idx]
                func_name = name[idx+2:]
                type = self.find_type_by_name(type_name)
                assert type is not None
                frefl = program_layout.find_function_by_name_in_type(
                    type.type_reflection, func_name)
                assert frefl is not None
                func.on_hot_reload(frefl)
            else:
                frefl = program_layout.find_function_by_name(name)
                assert frefl is not None
                func.on_hot_reload(frefl)
            new_functions_by_name[name] = func
            new_functions_by_reflection[frefl] = func

        self._functions_by_name = new_functions_by_name
        self._functions_by_reflection = new_functions_by_reflection

    def find_type(self, refl: TypeReflection) -> SlangType:
        """
        Find slangpy reflection for a given slang TypeReflection.
        """
        return self._get_or_create_type(refl)

    def find_function(self, refl: FunctionReflection, this_refl: Optional[TypeReflection]) -> SlangFunction:
        """
        Find slangpy reflection for a given slang FunctionReflection, optionally as a method of a type.
        """
        if this_refl is None:
            return self._get_or_create_function(refl, None, None)
        else:
            return self._get_or_create_function(refl, self._get_or_create_type(this_refl), None)

    def find_type_by_name(self, name: str) -> Optional[SlangType]:
        """
        Find a type by name.
        """
        existing = self._types_by_name.get(name)
        if existing is not None:
            return existing
        type_refl = self.program_layout.find_type_by_name(name)
        if type_refl is None and self.slangpy_program_layout is not None:
            type_refl = self.slangpy_program_layout.find_type_by_name(name)
        if type_refl is None:
            return None
        res = self._get_or_create_type(type_refl)
        return res

    def require_type_by_name(self, name: str) -> SlangType:
        """
        Require a type by name, raising an error if it is not found.
        """
        res = self.find_type_by_name(name)
        if res is None:
            raise ValueError(f"Type {name} not found")
        return res

    def find_function_by_name(self, name: str) -> Optional[SlangFunction]:
        """
        Find a function by name.
        """
        existing = self._functions_by_name.get(name)
        if existing is not None:
            return existing
        func_refl = self.program_layout.find_function_by_name(name)
        if func_refl is None:
            return None
        res = self._get_or_create_function(func_refl, None, name)
        return res

    def require_function_by_name(self, name: str) -> SlangFunction:
        """
        Require a function by name, raising an error if it is not found.
        """
        res = self.find_function_by_name(name)
        if res is None:
            raise ValueError(f"Function {name} not found")
        return res

    def find_function_by_name_in_type(self, type: SlangType, name: str) -> Optional[SlangFunction]:
        """
        Find a function by name in an already loaded type.
        """
        qualified_name = f"{type.full_name}::{name}"
        existing = self._functions_by_name.get(qualified_name)
        if existing is not None:
            return existing
        type_refl = self.program_layout.find_type_by_name(type.full_name)
        if type_refl is None:
            raise ValueError(f"Type {type.full_name} not found")
        func_refl = self.program_layout.find_function_by_name_in_type(type_refl, name)
        if func_refl is None:
            return None
        res = self._get_or_create_function(
            self.program_layout.find_function_by_name_in_type(type_refl, name), self._get_or_create_type(type_refl), name)
        return res

    def require_function_by_name_in_type(self, type: SlangType, name: str) -> SlangFunction:
        """
        Require a function by name in an already loaded type, raising an error if it is not found.
        """
        res = self.find_function_by_name_in_type(type, name)
        if res is None:
            raise ValueError(f"Function {name} not found in type {type.full_name}")
        return res

    def scalar_type(self, scalar_type: TR.ScalarType) -> ScalarType:
        """
        Helper to get a scalar type given a Slang scalar type id.
        """
        return cast(ScalarType, self.find_type_by_name(scalar_names[scalar_type]))

    def vector_type(self, scalar_type: TR.ScalarType, size: int) -> VectorType:
        """
        Helper to get a vector type given a Slang scalar type id and size.
        """
        return cast(VectorType, self.find_type_by_name(f"vector<{scalar_names[scalar_type]},{size}>"))

    def matrix_type(self, scalar_type: TR.ScalarType, rows: int, cols: int) -> MatrixType:
        """
        Helper to get a matrix type given a Slang scalar type id and rows/cols.
        """
        return cast(MatrixType, self.find_type_by_name(f"matrix<{scalar_names[scalar_type]},{rows},{cols}>"))

    def array_type(self, element_type: SlangType, count: int) -> ArrayType:
        """
        Helper to get an array type given an element type and count.
        """
        if count > 0:
            return cast(ArrayType, self.find_type_by_name(f"{element_type.full_name}[{count}]"))
        else:
            return cast(ArrayType, self.find_type_by_name(f"{element_type.full_name}[]"))

    def _get_or_create_type(self, refl: TypeReflection):
        existing = self._types_by_reflection.get(refl)
        if existing is not None:
            return existing
        res = self._reflect_type(refl)
        self._types_by_reflection[refl] = res
        self._types_by_name[res.full_name] = res
        return res

    def _get_or_create_function(self, refl: FunctionReflection, this: Optional[SlangType], full_name: Optional[str]):
        existing = self._functions_by_reflection.get(refl)
        if existing is not None:
            return existing
        res = self._reflect_function(refl, this, full_name)
        self._functions_by_reflection[refl] = res

        if this is not None:
            self._functions_by_name[f"{this.full_name}::{res.name}"] = res
        else:
            self._functions_by_name[res.name] = res
        return res

    def _reflect_type(self, refl: TypeReflection):
        if refl.kind == TR.Kind.scalar:
            return self._reflect_scalar(refl)
        elif refl.kind == TR.Kind.vector:
            return self._reflect_vector(refl)
        elif refl.kind == TR.Kind.matrix:
            return self._reflect_matrix(refl)
        elif refl.kind == TR.Kind.array:
            return self._reflect_array(refl)
        elif refl.kind == TR.Kind.resource:
            return self._reflect_resource(refl)
        elif refl.kind == TR.Kind.sampler_state:
            return SamplerStateType(self, refl)

        # It's not any of the fundamental types. Check if a custom handler was defined,
        # giving precedence to handlers that match the fully specialized name
        full_name = refl.full_name
        handler = TYPE_OVERRIDES.get(refl.name)
        handler = TYPE_OVERRIDES.get(full_name, handler)
        if handler is not None:
            return handler(self, refl)

        # Catch the remaining types
        if refl.kind == TR.Kind.struct:
            return StructType(self, refl)
        elif refl.kind == TR.Kind.interface:
            return InterfaceType(self, refl)
        else:
            # This type is not represented by its own class - just store the basic info
            return UnhandledType(self, refl)

    def _reflect_scalar(self, refl: TypeReflection) -> SlangType:
        if refl.scalar_type == TR.ScalarType.void:
            return VoidType(self, refl)
        else:
            return ScalarType(self, refl)

    def _reflect_vector(self, refl: TypeReflection) -> SlangType:
        return VectorType(self, refl)

    def _reflect_matrix(self, refl: TypeReflection) -> SlangType:
        return MatrixType(self, refl)

    def _reflect_array(self, refl: TypeReflection) -> SlangType:
        return ArrayType(self, refl)

    def _reflect_resource(self, refl: TypeReflection) -> SlangType:
        if refl.resource_shape == TR.ResourceShape.structured_buffer:
            return StructuredBufferType(self, refl)
        elif refl.resource_shape == TR.ResourceShape.byte_address_buffer:
            return ByteAddressBufferType(self, refl)
        elif refl.resource_shape in texture_names:
            return TextureType(self, refl)
        elif refl.resource_shape == TR.ResourceShape.acceleration_structure:
            return RaytracingAccelerationStructureType(self, refl)
        else:
            return ResourceType(self, refl)

    def _reflect_function(self, function: FunctionReflection, this: Optional[SlangType], full_name: Optional[str]) -> SlangFunction:
        return SlangFunction(self, function, this, full_name)

    def get_resolved_generic_args(self, slang_type: TypeReflection) -> TGenericArgs:
        """ 
        Parse the arguments of a generic and resolve them into value args (i.e. ints) or slang types.
        """
        # TODO: This should really be extracted from the reflection API, but this is not
        # currently implemented in SGL, and we do it via string processing for now until this is fixed

        full = slang_type.full_name
        # If full name does not end in >, this is not a generic
        if full[-1] != ">":
            return None

        # Parse backwards from right to left
        # (because full_name could be e.g. OuterStruct<float>::InnerType<int>)
        # Keep track of the current nesting level
        # (because generics could be nested, e.g. vector<vector<float, 2>, 2>)
        # Retrieve a list of generic args as string
        head = full
        idx = len(head) - 1
        level = 0
        pieces: list[str] = []
        while idx > 0:
            idx -= 1
            if level > 0:
                if head[idx] == "<":
                    level -= 1
            else:
                if head[idx] == ">":
                    level += 1
                elif head[idx] == "," or head[idx] == "<":
                    pieces.append(head[idx+1:-1].strip())
                    head = head[:idx+1]
                if head[idx] == "<":
                    break
        if head[idx] != "<":
            raise ValueError(f"Unable to parse generic '{full}'")

        # Now resolve generics into ints or types
        # Note: avoiding using exception as it makes things hard to debug
        result = []
        for piece in reversed(pieces):
            if can_convert_to_int(piece):
                x = int(piece)
            else:
                x = self.find_type_by_name(piece)
            result.append(x)

        return tuple(result)


def can_convert_to_int(value: Any):

    # Check if it's an integer or a float that can be cast to an int
    if isinstance(value, int):
        return True
    elif isinstance(value, float) and value.is_integer():
        return True
    elif isinstance(value, str) and value.lstrip('+-').isdigit():
        return True
    else:
        return False


TGenericArgs = Optional[tuple[int | SlangType, ...]]

#: Mapping from a type name to a callable that creates a SlangType from a TypeReflection.
#: This can be used to extend the type system and wrap custom types in their own reflection types.
TYPE_OVERRIDES: dict[str, Callable[[
    SlangProgramLayout, TypeReflection], SlangType]] = {}


def create_differential_pair(layout: SlangProgramLayout, refl: TypeReflection) -> SlangType:
    return DifferentialPairType(layout, refl)


TYPE_OVERRIDES['DifferentialPair'] = create_differential_pair
