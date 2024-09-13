

from typing import Any
from kernelfunctions.backend import TypeReflection, math
from kernelfunctions.typeregistry import PYTHON_TYPES, SLANG_MATRIX_TYPES, SLANG_SCALAR_TYPES, SLANG_VECTOR_TYPES
from kernelfunctions.types.basetype import BaseType
from kernelfunctions.types.basetypeimpl import BaseTypeImpl


class ValueTypeImpl(BaseTypeImpl):
    def __init__(self):
        super().__init__()

    def is_writable(self):
        raise NotImplementedError()

    def is_readable(self):
        raise NotImplementedError()

    def gen_calldata(self, cgb: CodeGenBlock, input_value: 'BaseValue', name: str, access: tuple[AccessType, AccessType]):
        """
        Declare the call data for this value. By default, read only values are stored as uniforms, and read-write
        values are stored as structured buffers with a single element.
        """
        if access == AccessType.read:
            cgb.declare(type_name, variable_name)
        else:
            cgb.declare(f"RWStructuredBuffer<{type_name}>", variable_name)


class ScalarType(ValueTypeImpl):
    def __init__(self, slang_type: TypeReflection.ScalarType):
        super().__init__()
        self.slang_type = slang_type
        self.diff = self.slang_type in [TypeReflection.ScalarType.float16,
                                        TypeReflection.ScalarType.float32, TypeReflection.ScalarType.float64]

    def element_type(self, value: Any = None):
        return self

    def shape(self, value: Any = None):
        return (1,)

    def differentiable(self, value: Any = None):
        return self.diff

    def differentiate(self, value: Any = None):
        return self if self.diff else None


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
