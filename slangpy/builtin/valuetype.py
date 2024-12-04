from typing import Any
from slangpy.backend import math
from slangpy.core.enums import PrimType, IOType
from slangpy.core.native import Shape, CallContext, AccessType, TypeReflection
from slangpy.backend import ResourceUsage, Buffer, ResourceView, ResourceViewType, Texture, ResourceType, FormatType, get_format_info
from slangpy.bindings import CodeGenBlock, BindContext, ReturnContext, BaseType, BaseTypeImpl, BoundVariable, PYTHON_TYPES, PYTHON_SIGNATURES, BoundVariableRuntime, get_or_create_type
from slangpy.reflection import SlangProgramLayout, SlangType, TYPE_OVERRIDES, is_matching_array_type
from slangpy.types import NDBuffer, NDDifferentiableBuffer
import slangpy.bindings.typeregistry as tr
import slangpy.reflection as kfr
from slangpy.reflection.reflectiontypes import FLOAT_TYPES, INT_TYPES, BOOL_TYPES, SIGNED_INT_TYPES, UNSIGNED_INT_TYPES
import slangpy.backend as kfbackend

"""
Common functionality for basic value types such as int, float, vector, matrix etc that aren't
writable and don't store an additional derivative.
"""


def slang_type_to_return_type(slang_type: kfr.SlangType) -> Any:
    if isinstance(slang_type, kfr.ScalarType):
        if slang_type.slang_scalar_type in FLOAT_TYPES:
            return float
        elif slang_type.slang_scalar_type in INT_TYPES:
            return int
        elif slang_type.slang_scalar_type in BOOL_TYPES:
            return bool
    elif isinstance(slang_type, kfr.VectorType):
        if slang_type.slang_scalar_type in FLOAT_TYPES:
            return getattr(kfbackend, f'float{slang_type.num_elements}')
        elif slang_type.slang_scalar_type in SIGNED_INT_TYPES:
            return getattr(kfbackend, f'int{slang_type.num_elements}')
        elif slang_type.slang_scalar_type in UNSIGNED_INT_TYPES:
            return getattr(kfbackend, f'uint{slang_type.num_elements}')
        elif slang_type.slang_scalar_type in BOOL_TYPES:
            return getattr(kfbackend, f'bool{slang_type.num_elements}')
    elif isinstance(slang_type, kfr.MatrixType):
        if slang_type.slang_scalar_type in FLOAT_TYPES:
            return getattr(kfbackend, f'float{slang_type.rows}x{slang_type.cols}')
        elif slang_type.slang_scalar_type in SIGNED_INT_TYPES:
            return getattr(kfbackend, f'int{slang_type.rows}x{slang_type.cols}')
        elif slang_type.slang_scalar_type in UNSIGNED_INT_TYPES:
            return getattr(kfbackend, f'uint{slang_type.rows}x{slang_type.cols}')
        elif slang_type.slang_scalar_type in BOOL_TYPES:
            return getattr(kfbackend, f'bool{slang_type.rows}x{slang_type.cols}')
    else:
        raise ValueError(f"Slang type {slang_type} has no associated python value type")


class ValueType(BaseTypeImpl):
    def __init__(self, layout: kfr.SlangProgramLayout):
        super().__init__(layout)

    # Values don't store a derivative - they're just a value
    @property
    def has_derivative(self) -> bool:
        return False

    # Values are readonly
    @property
    def is_writable(self) -> bool:
        return False

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access
        name = binding.variable_name
        if access[0] in [AccessType.read, AccessType.readwrite]:
            cgb.type_alias(
                f"_t_{name}", f"ValueType<{self.slang_type.full_name}>")
        else:
            cgb.type_alias(f"_t_{name}", f"NoneType")

    # Call data just returns the primal
    def create_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: Any) -> Any:
        access = binding.access
        if access[0] in [AccessType.read, AccessType.readwrite]:
            return {
                'value': data
            }

    # Values just return themselves for raw dispatch
    def create_dispatchdata(self, data: Any) -> Any:
        return data

    # No need to create any buffers for output data, as we're read only!
    def create_output(self, context: CallContext, binding: BoundVariableRuntime) -> Any:
        pass

    # Return the input as output, as it was by definition not changed
    def read_output(self, context: CallContext, binding: BoundVariableRuntime, data: Any) -> Any:
        return data


"""
Mapping of type reflection enum to slang type name
"""
SCALAR_TYPE_NAMES: dict[TypeReflection.ScalarType, str] = {
    TypeReflection.ScalarType.none: "none",
    TypeReflection.ScalarType.void: "void",
    TypeReflection.ScalarType.bool: "bool",
    TypeReflection.ScalarType.int32: "int",
    TypeReflection.ScalarType.uint32: "uint",
    TypeReflection.ScalarType.int64: "int64_t",
    TypeReflection.ScalarType.uint64: "uint64_t",
    TypeReflection.ScalarType.float16: "float16_t",
    TypeReflection.ScalarType.float32: "float",
    TypeReflection.ScalarType.float64: "float64_t",
    TypeReflection.ScalarType.int8: "int8_t",
    TypeReflection.ScalarType.uint8: "uint8_t",
    TypeReflection.ScalarType.int16: "int16_t",
    TypeReflection.ScalarType.uint16: "uint16_t",
}
SCALAR_TYPE_TO_PYTHON_TYPE: dict[TypeReflection.ScalarType, type] = {
    TypeReflection.ScalarType.none: type(None),
    TypeReflection.ScalarType.void: type(None),
    TypeReflection.ScalarType.bool: bool,
    TypeReflection.ScalarType.int32: int,
    TypeReflection.ScalarType.uint32: int,
    TypeReflection.ScalarType.int64: int,
    TypeReflection.ScalarType.uint64: int,
    TypeReflection.ScalarType.float16: float,
    TypeReflection.ScalarType.float32: float,
    TypeReflection.ScalarType.float64: float,
    TypeReflection.ScalarType.int8: int,
    TypeReflection.ScalarType.uint8: int,
    TypeReflection.ScalarType.int16: int,
    TypeReflection.ScalarType.uint16: int,
}
SCALAR_TYPE_SIZES: dict[TypeReflection.ScalarType, int] = {
    TypeReflection.ScalarType.none: 1,
    TypeReflection.ScalarType.void: 1,
    TypeReflection.ScalarType.bool: 4,
    TypeReflection.ScalarType.int32: 4,
    TypeReflection.ScalarType.uint32: 4,
    TypeReflection.ScalarType.int64: 8,
    TypeReflection.ScalarType.uint64: 8,
    TypeReflection.ScalarType.float16: 2,
    TypeReflection.ScalarType.float32: 4,
    TypeReflection.ScalarType.float64: 8,
    TypeReflection.ScalarType.int8: 1,
    TypeReflection.ScalarType.uint8: 1,
    TypeReflection.ScalarType.int16: 2,
    TypeReflection.ScalarType.uint16: 2,
}


class ScalarType(ValueType):
    def __init__(self, layout: kfr.SlangProgramLayout, scalar_type: TypeReflection.ScalarType):
        super().__init__(layout)
        self.slang_type = layout.scalar_type(scalar_type)
        self.concrete_shape = self.slang_type.shape

    def reduce_type(self, context: BindContext, dimensions: int):
        if dimensions > 0:
            raise ValueError("Cannot reduce scalar type")
        return self.slang_type


class NoneValueType(ValueType):
    def __init__(self, layout: kfr.SlangProgramLayout):
        super().__init__(layout)
        self.slang_type = layout.scalar_type(TypeReflection.ScalarType.void)

    def resolve_dimensionality(self, context: BindContext, binding: BoundVariable, vector_target_type: BaseType):
        # None type can't resolve dimensionality
        return None


class VectorType(ValueType):
    def __init__(self, layout: kfr.SlangProgramLayout, scalar_type: TypeReflection.ScalarType, num_elements: int):
        super().__init__(layout)
        self.slang_type = layout.vector_type(scalar_type, num_elements)
        self.concrete_shape = self.slang_type.shape

    def reduce_type(self, context: 'BindContext', dimensions: int):
        self_type = self.slang_type
        if dimensions == 1:
            return self_type.element_type
        elif dimensions == 0:
            return self_type
        else:
            raise ValueError("Cannot reduce vector type by more than one dimension")


class MatrixType(ValueType):
    def __init__(self, layout: kfr.SlangProgramLayout, scalar_type: TypeReflection.ScalarType, rows: int, cols: int):
        super().__init__(layout)
        self.slang_type = layout.matrix_type(scalar_type, rows, cols)
        self.concrete_shape = self.slang_type.shape

    def reduce_type(self, context: 'BindContext', dimensions: int):
        self_type = self.slang_type
        if dimensions == 2:
            assert self_type.element_type is not None
            return self_type.element_type.element_type
        elif dimensions == 1:
            return self_type.element_type
        elif dimensions == 0:
            return self_type


# Point built in python types at their slang equivalents
PYTHON_TYPES[type(None)] = lambda layout, pytype: NoneValueType(layout)
PYTHON_TYPES[bool] = lambda layout, pytype: ScalarType(
    layout, TypeReflection.ScalarType.bool)
PYTHON_TYPES[float] = lambda layout, pytype: ScalarType(
    layout, TypeReflection.ScalarType.float32)
PYTHON_TYPES[int] = lambda layout, pytype: ScalarType(
    layout, TypeReflection.ScalarType.int32)
PYTHON_SIGNATURES[type(None)] = None
PYTHON_SIGNATURES[bool] = None
PYTHON_SIGNATURES[float] = None
PYTHON_SIGNATURES[int] = None


# Python quaternion type
PYTHON_TYPES[math.quatf] = lambda layout, pytype: VectorType(
    layout, TypeReflection.ScalarType.float32, 4)
PYTHON_SIGNATURES[math.quatf] = None

# Python versions of vector and matrix types
for pair in zip(["int", "float", "bool", "uint", "float16_t"], [TypeReflection.ScalarType.int32, TypeReflection.ScalarType.float32, TypeReflection.ScalarType.bool, TypeReflection.ScalarType.uint32, TypeReflection.ScalarType.float16]):
    base_name = pair[0]
    slang_scalar_type = pair[1]

    for dim in range(1, 5):
        vec_type: type = getattr(math, f"{base_name}{dim}")
        if vec_type is not None:
            t = lambda layout, pytype, dim=dim, st=slang_scalar_type: VectorType(
                layout, st, dim)
            PYTHON_TYPES[vec_type] = t
            PYTHON_SIGNATURES[vec_type] = None

    for row in range(2, 5):
        for col in range(2, 5):
            mat_type: type = getattr(math, f"float{row}x{col}", None)  # type: ignore
            if mat_type is not None:
                t = lambda layout, pytype, row=row, st=slang_scalar_type, col=col: MatrixType(
                    layout, st, row, col)
                t.python_type = mat_type  # type: ignore
                PYTHON_TYPES[mat_type] = t
                PYTHON_SIGNATURES[mat_type] = None