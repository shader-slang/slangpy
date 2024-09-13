

from typing import Any, Optional
import numpy.typing as npt
import numpy as np

from kernelfunctions.backend import TypeReflection, math, Device
from kernelfunctions.codegen import CodeGenBlock
from kernelfunctions.typeregistry import PYTHON_TYPES, SLANG_MATRIX_TYPES, SLANG_SCALAR_TYPES, SLANG_VECTOR_TYPES
from kernelfunctions.types.basetype import BaseType
from kernelfunctions.types.basetypeimpl import BaseTypeImpl
from kernelfunctions.types.basevalue import BaseValue
from kernelfunctions.types.enums import AccessType, PrimType

"""
Common functionality for basic value types such as int, float, vector, matrix etc that aren't
writable and don't store an additional derivative.
"""


class ValueTypeImpl(BaseTypeImpl):
    def __init__(self):
        super().__init__()

    # Values don't store a derivative - they're just a value
    def has_derivative(self) -> bool:
        return False

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, input_value: 'BaseValue', name: str, access: tuple[AccessType, AccessType]):
        assert access[0] == AccessType.read
        assert access[1] == AccessType.none
        cgb.declare(input_value.primal_type_name, f"{name}_primal")

    # Load should only ever be reading the primal directly from the call data
    def gen_load(self, cgb: CodeGenBlock, input_value: 'BaseValue', from_call_data: str, to_variable: str, transform: list[Optional[int]], prim: PrimType, access: AccessType):
        assert prim == PrimType.primal
        assert access == AccessType.read
        cgb.assign(to_variable, from_call_data)

    # Never store anything
    def gen_store(self, cgb: CodeGenBlock, input_value: 'BaseValue', from_variable: str, to_call_data: str, transform: list[Optional[int]], prim: PrimType, access: AccessType):
        pass

    # Call data just returns the primal
    def create_calldata(self, device: Device, input_value: 'BaseValue', access: tuple[AccessType, AccessType], data: Any) -> Any:
        assert access[0] == AccessType.read
        assert access[1] == AccessType.none
        return data

    # Read back from call data does nothing
    def read_calldata(self, device: Device, input_value: 'BaseValue', access: tuple[AccessType, AccessType], data: Any, result: Any) -> None:
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


class ScalarType(ValueTypeImpl):
    def __init__(self, slang_type: TypeReflection.ScalarType):
        super().__init__()
        self.slang_type = slang_type
        self.diff = self.slang_type in [TypeReflection.ScalarType.float16,
                                        TypeReflection.ScalarType.float32, TypeReflection.ScalarType.float64]
        self.python_type = SCALAR_TYPE_TO_PYTHON_TYPE[self.slang_type]

    def name(self) -> str:
        return SCALAR_TYPE_NAMES[self.slang_type]

    def element_type(self, value: Any = None):
        return self

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


class VectorType(ValueTypeImpl):
    def __init__(self, element_type: BaseType, size: int):
        super().__init__()
        self.et = element_type
        self.size = size

    def element_type(self, value: Any = None):
        return self.et

    def shape(self, value: Any = None):
        return (self.size,)

    def differentiable(self, value: Any = None):
        return self.et.differentiable(value)

    def differentiate(self, value: Any = None):
        return self.et.differentiate(value)


class MatrixType(ValueTypeImpl):
    def __init__(self, element_type: BaseType, rows: int, cols: int):
        super().__init__()
        self.et = element_type
        self.rows = rows
        self.cols = cols

    def element_type(self, value: Any = None):
        return self.et

    def shape(self, value: Any = None):
        return (self.rows, self.cols)

    def differentiable(self, value: Any = None):
        return self.et.differentiable(value)

    def differentiate(self, value: Any = None):
        return self.et.differentiate(value)


# Hook up all the basic slang scalar, vector and matrix types
for x in TypeReflection.ScalarType:
    SLANG_SCALAR_TYPES[x] = ScalarType(x)
    SLANG_VECTOR_TYPES[x] = [VectorType(SLANG_SCALAR_TYPES[x], i) for i in range(0, 5)]
    SLANG_MATRIX_TYPES[x] = []
    for rows in range(0, 5):
        row: list[BaseType] = []
        for cols in range(0, 5):
            row.append(MatrixType(SLANG_SCALAR_TYPES[x], rows, cols))
        SLANG_MATRIX_TYPES[x].append(row)

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
            PYTHON_TYPES[vec_type] = SLANG_VECTOR_TYPES[slang_scalar_type][dim]

    for row in range(2, 5):
        for col in range(2, 5):
            mat_type: type = getattr(math, f"float{row}x{col}", None)  # type: ignore
            if mat_type is not None:
                PYTHON_TYPES[mat_type] = SLANG_MATRIX_TYPES[slang_scalar_type][row][col]
