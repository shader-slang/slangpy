

from types import NoneType
from typing import Any, Optional
import numpy.typing as npt
import numpy as np

from kernelfunctions.backend import TypeReflection, math, Device
from kernelfunctions.codegen import CodeGenBlock
from kernelfunctions.typeregistry import PYTHON_TYPES, SLANG_MATRIX_TYPES, SLANG_SCALAR_TYPES, SLANG_VECTOR_TYPES
from kernelfunctions.types.basetype import BaseType
from kernelfunctions.types.basetypeimpl import BaseTypeImpl
from kernelfunctions.types.basevalue import BaseVariable
from kernelfunctions.types.enums import AccessType

"""
Common functionality for basic value types such as int, float, vector, matrix etc that aren't
writable and don't store an additional derivative.
"""


class ValueTypeImpl(BaseTypeImpl):
    def __init__(self):
        super().__init__()

    # A value is its own element
    def element_type(self, value: Any = None):
        return self

    # Values don't store a derivative - they're just a value
    def has_derivative(self, value: Any = None) -> bool:
        return False

    # Values are readonly
    def is_writable(self, value: Any = None) -> bool:
        return False

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, input_value: 'BaseVariable', name: str, transform: list[Optional[int]], access: tuple[AccessType, AccessType]):
        assert not access[0] in [AccessType.readwrite, AccessType.write]
        assert access[1] == AccessType.none
        cgb.type_alias(f"_{name}", f"ValueType<{input_value.primal_type_name}>")

    # Call data just returns the primal
    def create_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: Any) -> Any:
        assert not access[0] in [AccessType.readwrite, AccessType.write]
        assert access[1] == AccessType.none
        if access[0] == AccessType.read:
            return {
                'value': data
            }

    # Read back from call data does nothing
    def read_calldata(self, device: Device, input_value: 'BaseVariable', access: tuple[AccessType, AccessType], data: Any, result: Any) -> None:
        pass


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


class ScalarType(ValueTypeImpl):
    def __init__(self, slang_type: TypeReflection.ScalarType):
        super().__init__()
        self.slang_type = slang_type
        self.diff = self.slang_type in [TypeReflection.ScalarType.float16,
                                        TypeReflection.ScalarType.float32, TypeReflection.ScalarType.float64]
        self.python_type = SCALAR_TYPE_TO_PYTHON_TYPE[self.slang_type]
        self.bytes = SCALAR_TYPE_SIZES[self.slang_type]

    def name(self, value: Any = None) -> str:
        return SCALAR_TYPE_NAMES[self.slang_type]

    def byte_size(self, value: Any = None) -> int:
        return self.bytes

    def shape(self, value: Any = None):
        return (1,)

    def differentiable(self, value: Any = None):
        return self.diff

    def differentiate(self, value: Any = None):
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

    def python_return_value_type(self, value: Any = None) -> type:
        return self.python_type


class NoneValueType(ValueTypeImpl):
    def __init__(self, slang_type: TypeReflection.ScalarType):
        super().__init__()

    def shape(self, value: Any = None):
        return None

    def name(self, value: Any = None) -> str:
        return "none"

    def python_return_value_type(self, value: Any = None) -> type:
        return NoneType


class VectorType(ValueTypeImpl):
    def __init__(self, element_type: BaseType, size: int):
        super().__init__()
        self.et = element_type
        self.size = size
        self.python_type: type = NoneType

    def name(self, value: Any = None) -> str:
        return f"vector<{self.et.name()},{self.size}>"

    def byte_size(self, value: Any = None) -> int:
        return self.size * self.et.byte_size()

    def shape(self, value: Any = None):
        return (self.size,)

    def differentiable(self, value: Any = None):
        return self.et.differentiable(value)

    def differentiate(self, value: Any = None):
        et = self.et.differentiate(value)
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
        return self.python_type(list(array))

    def python_return_value_type(self, value: Any = None) -> type:
        return self.python_type


class MatrixType(ValueTypeImpl):
    def __init__(self, element_type: BaseType, rows: int, cols: int):
        super().__init__()
        self.et = element_type
        self.rows = rows
        self.cols = cols
        self.python_type: type = NoneType

    def name(self, value: Any = None) -> str:
        return f"matrix<{self.et.name()},{self.rows},{self.cols}>"

    def byte_size(self, value: Any = None) -> int:
        return self.rows * self.cols * self.et.byte_size()

    def shape(self, value: Any = None):
        return (self.rows, self.cols)

    def differentiable(self, value: Any = None):
        return self.et.differentiable(value)

    def differentiate(self, value: Any = None):
        et = self.et.differentiate(value)
        if et is not None:
            return MatrixType(et, self.rows, self.cols)
        else:
            return None

    def to_numpy(self, value: Any) -> npt.NDArray[Any]:
        return value.to_numpy()

    def from_numpy(self, array: npt.NDArray[Any]) -> Any:
        return self.python_type(array)

    def python_return_value_type(self, value: Any = None) -> type:
        return self.python_type


# Hook up all the basic slang scalar, vector and matrix types
for x in TypeReflection.ScalarType:
    SLANG_SCALAR_TYPES[x] = ScalarType(x)
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

# Python quaternion type
PYTHON_TYPES[math.quatf] = SLANG_VECTOR_TYPES[TypeReflection.ScalarType.float32][4]

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

    for row in range(2, 5):
        for col in range(2, 5):
            mat_type: type = getattr(math, f"float{row}x{col}", None)  # type: ignore
            if mat_type is not None:
                t = SLANG_MATRIX_TYPES[slang_scalar_type][row][col]
                t.python_type = mat_type  # type: ignore
                PYTHON_TYPES[mat_type] = t
