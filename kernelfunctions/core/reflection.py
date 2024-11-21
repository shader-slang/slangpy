from __future__ import annotations

from kernelfunctions.core.native import Shape, NativeSlangType

from .enums import IOType
from kernelfunctions.backend import TypeReflection as TR
from kernelfunctions.backend import ModifierID, VariableReflection, TypeReflection, FunctionReflection, SlangModule, TypeLayoutReflection, ProgramLayout, ResourceUsage

from typing import Optional, Callable, Any, Union, cast

import numpy as np

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
    def __init__(self, tlr: TypeLayoutReflection):
        super().__init__()
        self._tlr = tlr

    @property
    def reflection(self) -> TypeLayoutReflection:
        return self._tlr

    @property
    def size(self) -> int:
        return self._tlr.size

    @property
    def alignment(self) -> int:
        return self._tlr.alignment

    @property
    def stride(self) -> int:
        return self._tlr.stride


class SlangType(NativeSlangType):
    def __init__(self,
                 program: SlangProgramLayout,
                 refl: TypeReflection,
                 element_type: Optional[SlangType] = None,
                 local_shape: Shape = Shape(None)):
        super().__init__()

        self._program = program
        self._reflection = refl
        self._element_type = element_type

        self._cached_fields: Optional[dict[str, SlangField]] = None
        self._cached_differential: Optional[SlangType] = None
        self._cached_uniform_layout: Optional[SlangLayout] = None
        self._cached_buffer_layout: Optional[SlangLayout] = None

        if self._element_type == self:
            self._cached_shape = local_shape
        elif local_shape.valid and self._element_type is not None:
            self._cached_shape = local_shape + self._element_type.shape
        else:
            self._cached_shape = local_shape

    @property
    def program(self) -> SlangProgramLayout:
        return self._program

    @property
    def type_reflection(self) -> TypeReflection:
        return self._reflection

    @property
    def name(self) -> str:
        return self._reflection.name

    @property
    def full_name(self) -> str:
        return self._reflection.full_name

    @property
    def element_type(self) -> Optional[SlangType]:
        return self._element_type

    @property
    def fields(self) -> dict[str, SlangField]:
        return self._get_fields()

    @property
    def shape(self) -> Shape:
        return self._cached_shape

    @property
    def differentiable(self) -> bool:
        return self._get_differential() is not None

    @property
    def derivative(self) -> SlangType:
        if self.differentiable:
            res = self._get_differential()
            assert res is not None
            return res
        else:
            raise ValueError(f"Type {self.full_name} is not differentiable")

    @property
    def num_dims(self) -> int:
        return len(self.shape)

    @property
    def uniform_layout(self) -> SlangLayout:
        if self._cached_uniform_layout is None:
            sl = self._program.program_layout.get_type_layout(self.type_reflection)
            if sl is None:
                raise ValueError(
                    f"Unable to get layout for {self.full_name}. This can happen if the type is defined in a module that isn't accesible during type resolution.")
            self._cached_uniform_layout = SlangLayout(sl)
        return self._cached_uniform_layout

    @property
    def buffer_layout(self) -> SlangLayout:
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
        return self._program.find_type_by_name(self.full_name + ".Differential")

    def build_fields(self) -> dict[str, Union[SlangType, SlangField]]:
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
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl)


class ScalarType(SlangType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        assert refl.scalar_type not in (TR.ScalarType.none, TR.ScalarType.void)
        super().__init__(program, refl, element_type=self, local_shape=Shape())

    @property
    def slang_scalar_type(self) -> TR.ScalarType:
        return self._reflection.scalar_type


class VectorType(SlangType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        element_type = program.scalar_type(refl.scalar_type)
        dims = refl.col_count

        super().__init__(program, refl,
                         element_type=element_type,
                         local_shape=Shape((dims,)))

    @property
    def num_elements(self) -> int:
        return self.shape[0]

    @property
    def scalar_type(self) -> ScalarType:
        return cast(ScalarType, self.element_type)

    @property
    def slang_scalar_type(self) -> TR.ScalarType:
        assert isinstance(self.element_type, ScalarType)
        return self.element_type.slang_scalar_type

    def build_fields(self):
        names = ['x', 'y', 'z', 'w']
        return {names[i]: self.scalar_type for i in range(self.num_elements)}


class MatrixType(SlangType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl,
                         element_type=program.vector_type(
                             refl.scalar_type, refl.col_count),
                         local_shape=Shape((refl.row_count,)))

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    @property
    def scalar_type(self) -> ScalarType:
        assert isinstance(self.element_type, VectorType)
        return cast(ScalarType, self.element_type.scalar_type)

    @property
    def slang_scalar_type(self) -> TR.ScalarType:
        return self.scalar_type.slang_scalar_type


class ArrayType(SlangType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl, program.find_type(
            refl.element_type), local_shape=Shape((refl.element_count,)))

    @property
    def num_elements(self) -> int:
        return self.shape[0]


class StructType(SlangType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        # An opaque struct has no element type, but like a normal scalar has a 0D local shape
        super().__init__(program, refl, local_shape=Shape())

    def build_fields(self):
        return {field.name: field for field in self.type_reflection.fields}


class InterfaceType(SlangType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl)


class ResourceType(SlangType):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @property
    def resource_shape(self) -> TR.ResourceShape:
        return self.type_reflection.resource_shape

    @property
    def resource_access(self) -> TR.ResourceAccess:
        return self.type_reflection.resource_access

    @property
    def writable(self) -> bool:
        if self.resource_access == TR.ResourceAccess.read_write:
            return True
        elif self.resource_access == TR.ResourceAccess.read:
            return False
        else:
            raise ValueError("Resource is neither read_write or read")

    @property
    def usage(self) -> ResourceUsage:
        if self.writable:
            return ResourceUsage.unordered_access
        else:
            return ResourceUsage.shader_resource


class TextureType(ResourceType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):

        self.texture_dims = texture_dims[refl.resource_shape]

        super().__init__(program, refl,
                         element_type=program.find_type(refl.resource_result_type),
                         local_shape=Shape((-1,)*self.texture_dims,))


class StructuredBufferType(ResourceType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):

        super().__init__(program, refl,
                         element_type=program.find_type(refl.resource_result_type),
                         local_shape=Shape((-1,)))


class ByteAddressBufferType(ResourceType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl,
                         element_type=program.scalar_type(TR.ScalarType.uint8),
                         local_shape=Shape((-1,)))


class DifferentialPairType(SlangType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl, local_shape=Shape())

        args = program.get_resolved_generic_args(refl)
        assert args is not None
        assert len(args) == 1
        assert isinstance(args[0], SlangType)
        assert args[0].differentiable
        self.primal = args[0]

    def build_differential_type(self):
        return self._program.find_type_by_name("DifferentialPair<" + self.primal.derivative.full_name + ">")


class UnhandledType(SlangType):
    def __init__(self, program: SlangProgramLayout, refl: TypeReflection):
        super().__init__(program, refl)

    @property
    def kind(self) -> TR.Kind:
        return self.type_reflection.kind


class SlangFunction:
    def __init__(self, program: SlangProgramLayout, refl: FunctionReflection, this: Optional[SlangType]):
        super().__init__()
        self._this = this
        self._reflection = refl
        self._program = program
        func_params = [x for x in refl.parameters]
        self._cached_parameters = tuple(SlangParameter(
            program, param, i) for i, param in enumerate(func_params))
        self._cached_return_type: Optional[SlangType] = None

    @property
    def reflection(self) -> FunctionReflection:
        return self._reflection

    @property
    def name(self) -> str:
        return self._reflection.name

    @property
    def this(self) -> Optional[SlangType]:
        return self._this

    @property
    def return_type(self) -> Optional[SlangType]:
        if self._cached_return_type is None and self._reflection.return_type is not None:
            self._cached_return_type = self._program.find_type(
                self._reflection.return_type)
        return self._cached_return_type

    @property
    def parameters(self) -> tuple[SlangParameter, ...]:
        return self._cached_parameters

    @property
    def have_return_value(self) -> bool:
        return not isinstance(self.return_type, VoidType)

    @property
    def differentiable(self) -> bool:
        return self.reflection.has_modifier(ModifierID.differentiable)

    @property
    def mutating(self) -> bool:
        return self.reflection.has_modifier(ModifierID.mutating)


class BaseSlangVariable:
    def __init__(self, program: SlangProgramLayout, slang_type: SlangType, name: str, modifiers: set[ModifierID]):
        super().__init__()
        self._program = program
        self._type = slang_type
        self._name = name
        self._modifiers = modifiers

    @property
    def type(self) -> SlangType:
        return self._type

    @property
    def name(self) -> str:
        return self._name

    @property
    def modifiers(self) -> set[ModifierID]:
        return self._modifiers

    @property
    def declaration(self) -> str:
        mods = [str(mod) for mod in self.modifiers]
        return " ".join(mods + [f"{self.type.full_name} {self.name}"])

    @property
    def io_type(self) -> IOType:
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
        return ModifierID.nodiff in self.modifiers

    @property
    def differentiable(self) -> bool:
        if self.no_diff:
            return False
        return self.type.differentiable

    @property
    def derivative(self) -> SlangType:
        if self.differentiable:
            return self.type.derivative
        else:
            raise ValueError(f"Variable {self.name} is not differentiable")


class SlangField(BaseSlangVariable):
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
        return self._index

    @property
    def has_default(self) -> bool:
        return self._has_default


class SlangProgramLayout:
    def __init__(self, program_layout: ProgramLayout):
        super().__init__()
        assert isinstance(program_layout, ProgramLayout)
        self.program_layout = program_layout
        self._types_by_name: dict[str, SlangType] = {}
        self._types_by_reflection: dict[TypeReflection, SlangType] = {}
        self._functions_by_name: dict[str, SlangFunction] = {}
        self._functions_by_reflection: dict[FunctionReflection, SlangFunction] = {}

    def find_type(self, refl: TypeReflection) -> SlangType:
        return self._get_or_create_type(refl)

    def find_function(self, refl: FunctionReflection, this_refl: Optional[TypeReflection]) -> SlangFunction:
        if this_refl is None:
            return self._get_or_create_function(refl, None)
        else:
            return self._get_or_create_function(refl, self._get_or_create_type(this_refl))

    def find_type_by_name(self, name: str) -> Optional[SlangType]:
        existing = self._types_by_name.get(name)
        if existing is not None:
            return existing
        type_refl = self.program_layout.find_type_by_name(name)
        if type_refl is None:
            return None
        res = self._get_or_create_type(type_refl)
        self._types_by_name[name] = res
        return res

    def find_function_by_name(self, name: str) -> Optional[SlangFunction]:
        existing = self._functions_by_name.get(name)
        if existing is not None:
            return existing
        func_refl = self.program_layout.find_function_by_name(name)
        if func_refl is None:
            return None
        res = self._get_or_create_function(func_refl, None)
        self._functions_by_name[name] = res
        return res

    def find_function_by_name_in_type(self, type: SlangType, name: str) -> Optional[SlangFunction]:
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
            self.program_layout.find_function_by_name_in_type(type_refl, name), self._get_or_create_type(type_refl))
        self._functions_by_name[qualified_name] = res
        return res

    def scalar_type(self, scalar_type: TR.ScalarType) -> ScalarType:
        return cast(ScalarType, self.find_type_by_name(scalar_names[scalar_type]))

    def vector_type(self, scalar_type: TR.ScalarType, size: int) -> VectorType:
        return cast(VectorType, self.find_type_by_name(f"vector<{scalar_names[scalar_type]},{size}>"))

    def matrix_type(self, scalar_type: TR.ScalarType, rows: int, cols: int) -> MatrixType:
        return cast(MatrixType, self.find_type_by_name(f"matrix<{scalar_names[scalar_type]},{rows},{cols}>"))

    def array_type(self, element_type: SlangType, count: int) -> ArrayType:
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
        return res

    def _get_or_create_function(self, refl: FunctionReflection, this: Optional[SlangType]):
        existing = self._functions_by_reflection.get(refl)
        if existing is not None:
            return existing
        res = self._reflect_function(refl, this)
        self._functions_by_reflection[refl] = res
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
        else:
            return ResourceType(self, refl)

    def _reflect_function(self, function: FunctionReflection, this: Optional[SlangType]) -> SlangFunction:
        return SlangFunction(self, function, this)

    # Parse the arguments of a generic and resolve them into value args (i.e. ints) or slang types
    # This should really be extracted from the reflection API, but this is not currently implemented in SGL,
    # and we do it via string processing for now until this is fixed
    def get_resolved_generic_args(self, slang_type: TypeReflection) -> TGenericArgs:
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
TYPE_OVERRIDES: dict[str, Callable[[
    SlangProgramLayout, TypeReflection], SlangType]] = {}


def create_differential_pair(layout: SlangProgramLayout, refl: TypeReflection) -> SlangType:
    return DifferentialPairType(layout, refl)


TYPE_OVERRIDES['DifferentialPair'] = create_differential_pair

# There is not currently a way to go from TypeReflection to the enclosing scope,
# so we need this global state to retain it for now. The reflection API should be
# changed to allow removing this in the future
_cur_module: Optional[SlangModule] = None


class scope:
    def __init__(self, module: SlangModule):
        super().__init__()
        self.module = module

    def __enter__(self):
        global _cur_module
        _cur_module = self.module

    def __exit__(self, exception_type: Any, exception_value: Any, exception_traceback: Any):
        global _cur_module
        _cur_module = None
