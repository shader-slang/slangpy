# Base class for marshalling slang types


from typing import Optional, Union

from sgl import TypeReflection, VariableReflection

from kernelfunctions.shapes import TLooseShape

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

SCALAR_TYPE_TO_PYTHON: dict[TypeReflection.ScalarType, type] = {
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


class SlangMarshall:
    def __init__(self, slang_type: Union[TypeReflection, TypeReflection.ScalarType]):
        super().__init__()
        if isinstance(slang_type, TypeReflection):
            self.name = slang_type.full_name
            self.kind = slang_type.kind
            self.scalar_type = slang_type.scalar_type
            self.value_shape: TLooseShape = ()
        else:
            self.name = SCALAR_TYPE_NAMES[slang_type]
            self.kind = TypeReflection.Kind.scalar
            self.scalar_type = slang_type
            self.value_shape: TLooseShape = (1,)
        self.container_shape: TLooseShape = ()

    @property
    def differentiable(self) -> bool:
        return False

    def differentiate(self) -> Optional['SlangMarshall']:
        return None

    @property
    def shape(self):
        return self.container_shape + self.value_shape

    def load_fields(self, slang_type: TypeReflection) -> dict[str, Union[VariableReflection, TypeReflection.ScalarType]]:
        raise NotImplementedError()

    @property
    def python_return_value_type(self) -> type:
        raise NotImplementedError()

    def __repr__(self):
        return self.name
