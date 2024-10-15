

from types import NoneType
from typing import Any
import numpy.typing as npt
import numpy as np

from kernelfunctions.core import CodeGenBlock, BindContext, BaseType, BaseTypeImpl, BoundVariable, AccessType, BoundVariableRuntime, CallContext, Shape

from kernelfunctions.backend import TypeReflection, math
from kernelfunctions.typeregistry import PYTHON_SIGNATURES, PYTHON_TYPES, SLANG_MATRIX_TYPES, SLANG_SCALAR_TYPES, SLANG_STRUCT_TYPES_BY_NAME, SLANG_VECTOR_TYPES, get_or_create_type
from kernelfunctions.utils import parse_generic_signature

"""
Common functionality for basic value types such as int, float, vector, matrix etc that aren't
writable and don't store an additional derivative.
"""


class ValueType(BaseTypeImpl):
    def __init__(self):
        super().__init__()
        self.element_type = self

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
                f"_{name}", f"ValueType<{self.name}>")
        else:
            cgb.type_alias(f"_{name}", f"NoneType")

    # Call data just returns the primal
    def create_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: Any) -> Any:
        access = binding.access
        if access[0] in [AccessType.read, AccessType.readwrite]:
            return {
                'value': data
            }

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
    def __init__(self, slang_type: TypeReflection.ScalarType):
        super().__init__()
        self.slang_type = slang_type
        self.diff = self.slang_type in [TypeReflection.ScalarType.float16,
                                        TypeReflection.ScalarType.float32, TypeReflection.ScalarType.float64]
        self.python_type = SCALAR_TYPE_TO_PYTHON_TYPE[self.slang_type]
        self.bytes = SCALAR_TYPE_SIZES[self.slang_type]
        self.name = SCALAR_TYPE_NAMES[self.slang_type]
        self.concrete_shape = Shape()

    def get_byte_size(self, value: Any = None) -> int:
        return self.bytes

    @property
    def differentiable(self):
        return self.diff

    @property
    def derivative(self):
        return self if self.diff else None

    def to_numpy(self, value: Any) -> npt.NDArray[Any]:
        if self.python_type == int:
            if value is None:
                return np.array([0], dtype=np.int32)
            else:
                return np.array([value], dtype=np.int32)
        elif self.python_type == float:
            if value is None:
                return np.array([0], dtype=np.float32)
            else:
                return np.array([value], dtype=np.float32)
        elif self.python_type == bool:
            if value is None:
                return np.array([0], dtype=np.uint8)
            else:
                return np.array([1 if value else 0], dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported scalar type: {type(value)}")

    def from_numpy(self, array: npt.NDArray[Any]) -> Any:
        if self.python_type == int:
            return int(array.view(dtype=np.int32)[0])
        elif self.python_type == float:
            return float(array.view(dtype=np.float32)[0])
        elif self.python_type == bool:
            return bool(array[0] == 1)
        else:
            raise ValueError(f"Unsupported scalar type: {array.dtype}")

    @property
    def python_return_value_type(self) -> type:
        return self.python_type


class NoneValueType(ValueType):
    def __init__(self, slang_type: TypeReflection.ScalarType):
        super().__init__()
        self.name = "none"

    def get_shape(self, value: Any = None) -> Shape:
        return Shape(None)

    @property
    def python_return_value_type(self) -> type:
        return NoneType


class VectorType(ValueType):
    def __init__(self, element_type: BaseType, size: int):
        super().__init__()
        self.element_type = element_type
        self.size = size
        self.python_type: type = NoneType
        self.name = f"vector<{self.element_type.name},{self.size}>"
        self.concrete_shape = Shape(self.size)

    def get_byte_size(self, value: Any = None) -> int:
        assert self.element_type is not None
        return self.size * self.element_type.get_byte_size()

    @property
    def differentiable(self):
        return self.element_type.differentiable

    @property
    def fields(self):
        axes = ("x", "r", "y", "g", "z", "b", "w", "a")
        return {a: self.et for a in axes[:self.size * 2]}

    @property
    def derivative(self):
        et = self.element_type.derivative
        if et is not None:
            return VectorType(et, self.size)
        else:
            return None

    def to_numpy(self, value: Any) -> npt.NDArray[Any]:
        vals = [x for x in value]
        if value.element_type == int:
            return np.array(vals, dtype=np.int32)
        elif value.element_type == float:
            return np.array(vals, dtype=np.float32)
        elif value.element_type == bool:
            return np.array([1 if x else 0 for x in vals], dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported scalar type: {type(value)}")

    def from_numpy(self, array: npt.NDArray[Any]) -> Any:
        value = self.python_type()
        if value.element_type == int:
            array = array.view(dtype=np.int32)
        elif value.element_type == float:
            array = array.view(dtype=np.float32)
        elif value.element_type == bool:
            array = array.view(dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported scalar type: {type(value)}")
        return self.python_type(list(array))

    @property
    def python_return_value_type(self) -> type:
        return self.python_type


class MatrixType(ValueType):
    def __init__(self, element_type: BaseType, rows: int, cols: int):
        super().__init__()
        self.element_type = element_type
        self.rows = rows
        self.cols = cols
        self.python_type: type = NoneType
        self.name = f"matrix<{self.element_type.name},{self.rows},{self.cols}>"
        self.concrete_shape = Shape(self.rows, self.cols)

    def get_byte_size(self, value: Any = None) -> int:
        assert self.element_type is not None
        return self.rows * self.cols * self.element_type.get_byte_size()

    @property
    def differentiable(self):
        return self.element_type.differentiable

    @property
    def derivative(self):
        et = self.element_type.derivative
        if et is not None:
            return MatrixType(et, self.rows, self.cols)
        else:
            return None

    def to_numpy(self, value: Any) -> npt.NDArray[Any]:
        return value.to_numpy()

    def from_numpy(self, array: npt.NDArray[Any]) -> Any:
        return self.python_type(array)

    @property
    def python_return_value_type(self) -> type:
        return self.python_type


# Hook up all the basic slang scalar, vector and matrix types
for x in TypeReflection.ScalarType:
    SLANG_SCALAR_TYPES[x] = ScalarType(x)
    SLANG_STRUCT_TYPES_BY_NAME[SCALAR_TYPE_NAMES[x]] = SLANG_SCALAR_TYPES[x]
    SLANG_VECTOR_TYPES[x] = [VectorType(SLANG_SCALAR_TYPES[x], i) for i in range(0, 5)]
    SLANG_MATRIX_TYPES[x] = []
    for rows in range(0, 5):
        row = []
        for cols in range(0, 5):
            row.append(MatrixType(SLANG_SCALAR_TYPES[x], rows, cols))
        SLANG_MATRIX_TYPES[x].append(row)

# Overwrite void and none with none type
SLANG_SCALAR_TYPES[TypeReflection.ScalarType.none] = NoneValueType(
    TypeReflection.ScalarType.none)
SLANG_SCALAR_TYPES[TypeReflection.ScalarType.void] = NoneValueType(
    TypeReflection.ScalarType.void)

# Point built in python types at their slang equivalents
PYTHON_TYPES[type(None)] = SLANG_SCALAR_TYPES[TypeReflection.ScalarType.none]
PYTHON_TYPES[bool] = SLANG_SCALAR_TYPES[TypeReflection.ScalarType.bool]
PYTHON_TYPES[float] = SLANG_SCALAR_TYPES[TypeReflection.ScalarType.float32]
PYTHON_TYPES[int] = SLANG_SCALAR_TYPES[TypeReflection.ScalarType.int32]
PYTHON_SIGNATURES[type(None)] = None
PYTHON_SIGNATURES[bool] = None
PYTHON_SIGNATURES[float] = None
PYTHON_SIGNATURES[int] = None


# Python quaternion type
PYTHON_TYPES[math.quatf] = SLANG_VECTOR_TYPES[TypeReflection.ScalarType.float32][4]
PYTHON_SIGNATURES[math.quatf] = None

# Python versions of vector and matrix types
for pair in zip(["int", "float", "bool", "uint", "float16_t"], [TypeReflection.ScalarType.int32, TypeReflection.ScalarType.float32, TypeReflection.ScalarType.bool, TypeReflection.ScalarType.uint32, TypeReflection.ScalarType.float16]):
    base_name = pair[0]
    slang_scalar_type = pair[1]

    for dim in range(1, 5):
        vec_type: type = getattr(math, f"{base_name}{dim}")
        if vec_type is not None:
            t = SLANG_VECTOR_TYPES[slang_scalar_type][dim]
            t.python_type = vec_type  # type: ignore
            PYTHON_TYPES[vec_type] = t
            PYTHON_SIGNATURES[vec_type] = None

    for row in range(2, 5):
        for col in range(2, 5):
            mat_type: type = getattr(math, f"float{row}x{col}", None)  # type: ignore
            if mat_type is not None:
                t = SLANG_MATRIX_TYPES[slang_scalar_type][row][col]
                t.python_type = mat_type  # type: ignore
                PYTHON_TYPES[mat_type] = t
                PYTHON_SIGNATURES[mat_type] = None

# Map the 1D vectors to scalars
PYTHON_TYPES[math.int1] = SLANG_SCALAR_TYPES[TypeReflection.ScalarType.int32]
PYTHON_TYPES[math.uint1] = SLANG_SCALAR_TYPES[TypeReflection.ScalarType.uint32]
PYTHON_TYPES[math.bool1] = SLANG_SCALAR_TYPES[TypeReflection.ScalarType.bool]
PYTHON_TYPES[math.float1] = SLANG_SCALAR_TYPES[TypeReflection.ScalarType.float32]


def create_vector_type_for_slang(value: TypeReflection):
    assert isinstance(value, str)
    name, args = parse_generic_signature(value)
    assert name == "vector"
    el_type = get_or_create_type(args[0])
    assert isinstance(el_type, ScalarType)
    dim = int(args[1])
    return SLANG_VECTOR_TYPES[el_type.slang_type][dim]


SLANG_STRUCT_TYPES_BY_NAME["vector"] = create_vector_type_for_slang
