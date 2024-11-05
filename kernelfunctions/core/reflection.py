from __future__ import annotations

from .native import NativeSlangType, NativeTypeLayout
from .enums import IOType
from kernelfunctions.backend import TypeReflection as TR
from kernelfunctions.backend import ModifierID, VariableReflection, TypeReflection, TypeLayoutReflection, VariableLayoutReflection, FunctionReflection, SlangModule

from collections import OrderedDict
from typing import Optional, Callable, Any


scalar_names = {
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


class SlangType(NativeSlangType):
    def __init__(self):
        super().__init__()

        self.element_type: Optional[SlangType]
        self.differential: Optional[SlangType]
        self.fields: dict[str, SlangType]

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
        return self.differential is not None and not self.no_diff

    @property
    def derivative(self) -> SlangType:
        assert self.differential is not None and self.differentiable
        return self.differential

    def __eq__(self, other: Any):
        if not isinstance(other, SlangType):
            return NotImplemented
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class VoidType(SlangType):
    def __init__(self):
        super().__init__()

        self.name = "void"


class ScalarType(SlangType):
    def __init__(self, scalar_type: TR.ScalarType):
        super().__init__()

        assert scalar_type not in (TR.ScalarType.none, TR.ScalarType.void)
        self.name = scalar_names[scalar_type]
        self.scalar_type = scalar_type
        if is_float(scalar_type):
            self.differential = self


class VectorType(SlangType):
    def __init__(self, element_type: ScalarType, num_elements: int):
        super().__init__()

        self.name = f"vector<{element_type.name},{num_elements}>"
        self.element_type: ScalarType = element_type
        self.num_elements = num_elements
        if element_type.differential is not None:
            assert element_type.differential is element_type
            self.differential = self


class MatrixType(SlangType):
    def __init__(self, element_type: ScalarType, rows: int, cols: int):
        super().__init__()

        self.name = f"matrix<{element_type.name},{rows},{cols}>"
        self.element_type: SlangType = VectorType(element_type, cols)

        self.scalar_et = element_type
        self.rows = rows
        self.cols = cols
        self.num_elements = rows

        if element_type.differential is not None:
            assert element_type.differential is element_type
            self.differential = self


class ArrayType(SlangType):
    def __init__(self, element_type: SlangType, num_elements: int):
        super().__init__()

        if num_elements > 0:
            self.name = f"{element_type.name}[{num_elements}]"
        else:
            self.name = f"{element_type.name}[]"

        self.element_type: SlangType = element_type
        self.num_elements = num_elements
        if element_type.differential is not None:
            if element_type.differential is element_type:
                self.differential = self
            else:
                self.differential = ArrayType(element_type.differential, num_elements)


class StructType(SlangType):
    def __init__(self, name: str, members: OrderedDict[str, SlangType]):
        super().__init__()
        self.name = name
        self.fields = members


class InterfaceType(SlangType):
    def __init__(self, name: str):
        super().__init__()
        self.name = name


class ResourceType(SlangType):
    def __init__(self, name: str, resource_shape: TR.ResourceShape, resource_access: TR.ResourceAccess):
        super().__init__()
        self.name = name
        self.resource_shape = resource_shape
        self.resource_access = resource_access


class TextureType(ResourceType):
    def __init__(self, texture_resource_shape: TR.ResourceShape, resource_access: TR.ResourceAccess, element_type: SlangType):
        assert texture_resource_shape in texture_names
        assert resource_access in (TR.ResourceAccess.read, TR.ResourceAccess.read_write)

        prefix = "RW" if resource_access == TR.ResourceAccess.read_write else ""
        name = f"{prefix}{texture_names[texture_resource_shape]}<{element_type.name}>"
        super().__init__(name, texture_resource_shape, resource_access)

        self.element_type = element_type
        self.num_dims = texture_dims[texture_resource_shape]


class StructuredBufferType(ResourceType):
    def __init__(self, resource_access: TR.ResourceAccess, element_type: SlangType):
        assert resource_access in (TR.ResourceAccess.read, TR.ResourceAccess.read_write)

        prefix = "RW" if resource_access == TR.ResourceAccess.read_write else ""
        name = f"{prefix}StructuredBuffer<{element_type.name}>"
        super().__init__(name, TR.ResourceShape.structured_buffer, resource_access)

        self.element_type = element_type


class ByteAddressBufferType(ResourceType):
    def __init__(self, resource_access: TR.ResourceAccess):
        assert resource_access in (TR.ResourceAccess.read, TR.ResourceAccess.read_write)

        prefix = "RW" if resource_access == TR.ResourceAccess.read_write else ""
        name = f"{prefix}ByteAddressBuffer"
        super().__init__(name, TR.ResourceShape.byte_address_buffer, resource_access)


class DifferentialPairType(SlangType):
    def __init__(self, primal: SlangType):
        super().__init__()

        assert primal.differential is not None
        self.primal = primal
        self.name = f"DifferentialPair<{primal.name}>"
        self.fields = {"p": self.primal, "d": primal.differential}
        if primal.differential.differential is not None:
            if primal.differential is primal:
                self.differential = self
            else:
                DifferentialPairType(primal.differential)


class UnhandledType(SlangType):
    def __init__(self, name: str, kind: TR.Kind):
        super().__init__()

        self.name = name
        self.kind = kind


class SlangFunction:
    def __init__(self, name: str, parameters: tuple[SlangParameter, ...], return_param: SlangParameter, modifiers: set[ModifierID]):
        super().__init__()
        self.name = name
        self.parameters = parameters
        self.return_param = return_param
        self.modifiers = modifiers

    @property
    def have_return_value(self) -> bool:
        return not isinstance(self.return_param.type, VoidType)

    @property
    def differentiable(self) -> bool:
        return ModifierID.differentiable in self.modifiers


class SlangParameter:
    def __init__(self, slang_type: SlangType, name: str, index: int, has_default: bool):
        super().__init__()

        self.type = slang_type
        self.name = name
        self.index = index
        self.has_default = has_default

    def cast(self, new_type: SlangType) -> SlangParameter:
        return SlangParameter(new_type, self.name, self.index, self.has_default)

    @property
    def declaration(self) -> str:
        mods = [str(mod) for mod in self.type.modifiers]

        return " ".join(mods + [f"{self.type.name} {self.name}"])


SCALAR: dict[TR.ScalarType, ScalarType] = {}
VECTOR: dict[TR.ScalarType | ScalarType, tuple[VectorType, ...]] = {}
MATRIX: dict[TR.ScalarType | ScalarType, tuple[tuple[MatrixType, ...], ...]] = {}
for scalar_type in scalar_names.keys():
    st = ScalarType(scalar_type)
    SCALAR[scalar_type] = st
    VECTOR[scalar_type] = (None,) + tuple(VectorType(st, dim)
                                          for dim in (1, 2, 3, 4))  # type: ignore
    MATRIX[scalar_type] = (None,) + tuple((None,) + tuple(MatrixType(st, r, c)  # type: ignore
                                                          for c in (1, 2, 3, 4)) for r in (1, 2, 3, 4))

    VECTOR[st] = VECTOR[scalar_type]
    MATRIX[st] = MATRIX[scalar_type]

bool_ = SCALAR[TR.ScalarType.bool]
int8 = SCALAR[TR.ScalarType.int8]
int16 = SCALAR[TR.ScalarType.int16]
int32 = SCALAR[TR.ScalarType.int32]
int64 = SCALAR[TR.ScalarType.int64]
uint8 = SCALAR[TR.ScalarType.uint8]
uint16 = SCALAR[TR.ScalarType.uint16]
uint32 = SCALAR[TR.ScalarType.uint32]
uint64 = SCALAR[TR.ScalarType.uint64]
float16 = SCALAR[TR.ScalarType.float16]
float32 = SCALAR[TR.ScalarType.float32]
float64 = SCALAR[TR.ScalarType.float64]
half = float16
double = float64

float2 = VECTOR[float32][2]
float3 = VECTOR[float32][3]
float4 = VECTOR[float32][4]
int2 = VECTOR[int32][2]
int3 = VECTOR[int32][3]
int4 = VECTOR[int32][4]
uint2 = VECTOR[uint32][2]
uint3 = VECTOR[uint32][3]
uint4 = VECTOR[uint32][4]
half2 = VECTOR[half][2]
half3 = VECTOR[half][3]
half4 = VECTOR[half][4]

TGenericArgs = Optional[tuple[int | SlangType, ...]]
TYPE_OVERRIDES: dict[str, Callable[[
    TypeReflection, Optional[TGenericArgs]], SlangType]] = {}


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


def _add_modifiers(slang_type: SlangType, modifiers: set[ModifierID]):
    if len(modifiers) == 0:
        return

    slang_type.modifiers |= modifiers
    if slang_type.element_type is not None:
        _add_modifiers(slang_type.element_type, modifiers)
    for v in slang_type.fields.values():
        _add_modifiers(v, modifiers)


def _add_layout(slang_type: SlangType, layout: TypeLayoutReflection, offset: int = 0):
    slang_type.layout = NativeTypeLayout(layout.size, layout.stride, offset)

    if slang_type.element_type is not None:
        _add_layout(slang_type.element_type, layout.element_type_layout)
    for f in layout.fields:
        assert f.name in slang_type.fields
        _add_layout(slang_type.fields[f.name], f.type_layout, f.offset)


def reflect(t: TypeReflection | TypeLayoutReflection | VariableReflection | VariableLayoutReflection | TR.ScalarType) -> SlangType:
    if isinstance(t, TypeReflection):
        return reflect_type(t)
    elif isinstance(t, TypeLayoutReflection):
        return reflect_type_layout(t)
    elif isinstance(t, VariableReflection):
        return reflect_variable(t)
    elif isinstance(t, VariableLayoutReflection):
        return reflect_variable_layout(t)
    elif isinstance(t, TR.ScalarType):
        return reflect_scalar(t)

    assert False, f"Argument type {t} can't be used for reflection"


def reflect_variable_layout(var: VariableLayoutReflection) -> SlangType:
    slang_type = reflect_variable(var.variable)
    _add_layout(slang_type, var.type_layout, var.offset)
    return slang_type


def reflect_variable(var: VariableReflection) -> SlangType:
    result = reflect_type(var.type)

    modifiers = {mod for mod in ModifierID if var.has_modifier(mod)}
    if modifiers:
        _add_modifiers(result, modifiers)

    return result


def reflect_type_layout(layout: TypeLayoutReflection) -> SlangType:
    result = reflect_type(layout.type)

    _add_layout(result, layout)

    return result


def reflect_type(type: TypeReflection) -> SlangType:
    if type.kind == TR.Kind.scalar:
        return reflect_scalar(type.scalar_type)
    elif type.kind == TR.Kind.vector:
        return reflect_vector(type)
    elif type.kind == TR.Kind.matrix:
        return reflect_matrix(type)
    elif type.kind == TR.Kind.array:
        return reflect_array(type)
    elif type.kind == TR.Kind.resource:
        return reflect_resource(type)

    # It's not any of the fundamental types. Check if a custom handler was defined,
    # giving precedence to handlers that match the fully specialized name
    full_name = type.full_name
    handler = TYPE_OVERRIDES.get(type.name)
    handler = TYPE_OVERRIDES.get(full_name, handler)
    if handler is not None:
        result = handler(type, _get_resolved_generic_args(type))
        result.name = type.full_name
        result.needs_specialization = type.kind == TR.Kind.interface
        _set_differential(result, type)

    # Catch the remaining types
    if type.kind == TR.Kind.struct:
        return reflect_struct(type)
    elif type.kind == TR.Kind.interface:
        return reflect_interface(type)
    else:
        # This type is not represented by its own class - just store the basic info
        return UnhandledType(type.full_name, type.kind)


def reflect_scalar(kind: TR.ScalarType) -> SlangType:
    if kind == TR.ScalarType.void:
        return VoidType()
    else:
        return SCALAR[kind]


def reflect_vector(type: TypeReflection) -> SlangType:
    return VECTOR[type.scalar_type][type.col_count]


def reflect_matrix(type: TypeReflection) -> SlangType:
    return MATRIX[type.scalar_type][type.row_count][type.col_count]


def reflect_array(type: TypeReflection) -> SlangType:
    return ArrayType(reflect(type.element_type), type.element_count)


def reflect_resource(type: TypeReflection) -> SlangType:
    if type.resource_shape == TR.ResourceShape.structured_buffer:
        return StructuredBufferType(type.resource_access, reflect(type.resource_result_type))
    elif type.resource_shape == TR.ResourceShape.byte_address_buffer:
        return ByteAddressBufferType(type.resource_access)
    elif type.resource_shape in texture_names:
        return TextureType(type.resource_shape, type.resource_access, reflect(type.resource_result_type))
    else:
        return ResourceType(type.full_name, type.resource_shape, type.resource_access)


def reflect_struct(type: TypeReflection) -> SlangType:
    members = OrderedDict()
    for field in type.fields:
        members[field.name] = reflect_variable(field)

    result = StructType(type.full_name, members)
    _set_differential(result, type)
    return result


def reflect_interface(type: TypeReflection) -> SlangType:
    result = InterfaceType(type.full_name)
    _set_differential(result, type)
    result.needs_specialization = True

    return result


def _set_differential(slang_type: SlangType, type: TypeReflection):
    if _cur_module is None:
        return

    full_name = type.full_name
    differential = _cur_module.layout.find_type_by_name(full_name + ".Differential")
    if differential is None:
        return

    if differential.full_name == full_name:
        slang_type.differential = slang_type
    else:
        slang_type.differential = reflect_type(differential)


def reflect_function(function: FunctionReflection, this: Optional[TypeReflection]) -> SlangFunction:
    parameters = []
    for param in function.parameters:
        type = reflect_variable(param)
        # TODO: Get actual value of has_default from reflection API
        has_default = False
        parameters.append(SlangParameter(
            type, param.name, len(parameters), has_default))

    return_type = reflect_type(function.return_type)
    return_modifiers = {ModifierID.out}
    # TODO: Test that  function no_diff actually works
    if function.has_modifier(ModifierID.nodiff):
        return_modifiers.add(ModifierID.nodiff)
    _add_modifiers(return_type, return_modifiers)
    return_param = SlangParameter(return_type, "_result", len(parameters), True)

    modifiers = {mod for mod in ModifierID if function.has_modifier(mod)}

    if this is not None and function.name != "$init":
        this_type = reflect_type(this)
        # TODO: Check for [NoDiffThis]
        if ModifierID.mutating in modifiers:
            _add_modifiers(this_type, {ModifierID.inout})

        this_param = SlangParameter(this_type, "_this", -1, False)

        parameters.insert(0, this_param)

    return SlangFunction(function.name, tuple(parameters), return_param, modifiers)


# Parse the arguments of a generic and resolve them into value args (i.e. ints) or slang types
# This should really be extracted from the reflection API, but this is not currently implemented in SGL,
# and we do it via string processing for now until this is fixed
def _get_resolved_generic_args(slang_type: TypeReflection) -> TGenericArgs:
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
    result = []
    for piece in reversed(pieces):
        try:
            # Try int first; if it fails, try a type instead
            x = int(piece)
        except ValueError:
            if _cur_module is None:
                raise RuntimeError(
                    "Trying to reflect type without setting current module")
            x = reflect_type(_cur_module.layout.find_type_by_name(piece))
        result.append(x)

    return tuple(result)
