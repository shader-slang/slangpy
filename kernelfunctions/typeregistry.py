from enum import Enum
from io import StringIO
from typing import Any, Callable, Optional, Union
from sgl import TypeLayoutReflection, TypeReflection, VariableReflection
from kernelfunctions.shapes import TLooseOrUndefinedShape, TLooseShape
from kernelfunctions.typemappings import TPythonScalar, TSGLVector


class AccessType(Enum):
    none = 0
    read = 1
    write = 2
    readwrite = 3


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

# Base class for marshalling slang types


class BaseSlangTypeMarshal:
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

    def differentiate(self) -> Optional['BaseSlangTypeMarshal']:
        return None

    @property
    def shape(self):
        return self.container_shape + self.value_shape

    def load_fields(self, slang_type: TypeReflection) -> dict[str, Union[VariableReflection, TypeReflection.ScalarType]]:
        raise NotImplementedError()

    def __repr__(self):
        return self.name


# Base class for marshalling python types
class BasePythonTypeMarshal:
    def __init__(self, python_type: type):
        super().__init__()
        self.type = python_type

    def get_element_shape(self, value: Any) -> TLooseOrUndefinedShape:
        return ()

    def get_container_shape(self, value: Any) -> TLooseOrUndefinedShape:
        return ()

    def get_element_type(self, value: Any) -> Optional[Union[type[TSGLVector], type[TPythonScalar], TypeLayoutReflection]]:
        return type(value)

    def is_writable(self, value: Any) -> bool:
        return False

    def is_differentiable(self, value: Any) -> bool:
        return False

    def get_calldata_typename(self, typename: str, shape: TLooseOrUndefinedShape, access: AccessType):
        if access == AccessType.read:
            return typename
        else:
            return f"RWStructuredBuffer<{typename}>"

    def get_indexer(self, call_transform: list[Optional[int]], access: AccessType):
        if access == AccessType.read:
            return ""
        else:
            return "[0]"

    @property
    def name(self):
        return self.type.__name__

    def __repr__(self):
        return self.type.__name__


# Dictionary of python types to corresponding marshall
PYTHON_TYPE_MARSHAL: dict[type, BasePythonTypeMarshal] = {}

# Dictionary of python types to corresponding hash functions
PYTHON_SIGNATURE_HASH: dict[type, Optional[Callable[[StringIO, Any], Any]]] = {
    int: None,
    float: None,
    bool: None,
    str: None,
    list: None,
    dict: None,
    tuple: None,
}

# Register a mapping from type to marshall


def register_python_type(
    python_type: type,
    marshall: BasePythonTypeMarshal,
    hash_fn: Optional[Callable[[StringIO, Any], Any]],
):
    PYTHON_TYPE_MARSHAL[python_type] = marshall
    PYTHON_SIGNATURE_HASH[python_type] = hash_fn


def get_python_type_marshall(value: Any) -> BasePythonTypeMarshal:
    if isinstance(value, type):
        return PYTHON_TYPE_MARSHAL[value]
    else:
        return PYTHON_TYPE_MARSHAL[type(value)]


# Create slang marshall for reflection type
SLANG_MARSHALS_BY_FULL_NAME: dict[str, type[BaseSlangTypeMarshal]] = {}
SLANG_MARSHALS_BY_NAME: dict[str, type[BaseSlangTypeMarshal]] = {}
SLANG_MARSHALS_BY_KIND: dict[TypeReflection.Kind, type[BaseSlangTypeMarshal]] = {}
SLANG_MARSHALS_BY_SCALAR_TYPE: dict[TypeReflection.ScalarType,
                                    type[BaseSlangTypeMarshal]] = {}


def create_slang_type_marshal(slang_type: Union[TypeReflection, TypeReflection.ScalarType]) -> BaseSlangTypeMarshal:
    """
    Looks up correct marshall for a given slang type using
    first full name search, then base name search, then kind
    """
    if isinstance(slang_type, TypeReflection):
        marshal = SLANG_MARSHALS_BY_FULL_NAME.get(slang_type.full_name, None)
        if marshal is not None:
            return marshal(slang_type)
        marshal = SLANG_MARSHALS_BY_NAME.get(slang_type.name, None)
        if marshal is not None:
            return marshal(slang_type)
        marshal = SLANG_MARSHALS_BY_KIND.get(slang_type.kind, None)
        if marshal is not None:
            return marshal(slang_type)
        raise ValueError(f"Unsupported slang type {slang_type.full_name}")
    else:
        marshal = SLANG_MARSHALS_BY_SCALAR_TYPE.get(slang_type, None)
        if marshal is not None:
            return marshal(slang_type)
        raise ValueError(f"Unsupported slang type {slang_type}")
