import hashlib
from io import StringIO
from typing import Any, Callable, Optional, Union
from sgl import FunctionReflection, TypeReflection, VariableReflection
import kernelfunctions.slangmarshalls  # type: ignore
import kernelfunctions.pythonmarshalls  # type: ignore


# Base class for marshalling slang types
class BaseSlangTypeMarshal:
    def __init__(self, slang_type: TypeReflection):
        super().__init__()
        self.name = slang_type.full_name

    def __repr__(self):
        return self.name


# Base class for marshalling python types
class BasePythonTypeMarshal:
    def __init__(self, python_type: type):
        super().__init__()
        self.type = python_type

    def is_compatible(self, slang_type: TypeReflection) -> bool:
        raise NotImplementedError()

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


def register_python_type(
    python_type: type,
    marshall: BasePythonTypeMarshal,
    hash_fn: Optional[Callable[[StringIO, Any], Any]],
):
    PYTHON_TYPE_MARSHAL[python_type] = marshall
    PYTHON_SIGNATURE_HASH[python_type] = hash_fn


# Node in a built signature tree, maintains a pairing of python+slang marshall,
# and a potential set of child nodes
class SignatureNode:
    def __init__(self, value: Any):
        super().__init__()
        self.python_marshal = PYTHON_TYPE_MARSHAL[(type(value))]
        self.children: Optional[dict[str, SignatureNode]] = None
        if isinstance(value, dict):
            self.children = {x: SignatureNode(y) for x, y in value.items()}

    def is_compatible(
        self, slang_reflection: Union[VariableReflection, FunctionReflection]
    ) -> bool:
        if isinstance(slang_reflection, VariableReflection):
            return self.python_marshal.is_compatible(slang_reflection.type)
        elif isinstance(slang_reflection, FunctionReflection):
            return self.python_marshal.is_compatible(slang_reflection.return_type)
        else:
            raise ValueError(f"Unsupported reflection type {type(slang_reflection)}")

    def __repr__(self):
        return self.python_marshal.__repr__()


# Efficient function to build a sha256 hash that uniquiely identifies
# a function call signature (by types and shapes).
def build_signature_hash(*args: Any, **kwargs: Any) -> str:
    stream = StringIO()
    stream.write("args\n")
    for x in args:
        stream.write("-\n")
        _recurse_build_signature_hash(stream, x)
    stream.write("kwargs\n")
    for x, y in kwargs.items():
        stream.write(x)
        stream.write("\n")
        _recurse_build_signature_hash(stream, y)
    return hashlib.sha256(stream.getvalue().encode()).hexdigest()


# Internal function that walks the value tree to build a hashable
# buffer containing the types and shapes of the values.
def _recurse_build_signature_hash(stream: StringIO, python_value: Any):
    val_type = type(python_value)
    stream.write(val_type.__name__)
    stream.write("\n")

    hasher = PYTHON_SIGNATURE_HASH.get(val_type, None)
    if hasher is not None:
        hasher(stream, python_value)
        stream.write("\n")

    if isinstance(python_value, dict):
        for key, value in sorted(python_value.items()):
            stream.write(key)
            stream.write("\n")
            _recurse_build_signature_hash(stream, value)
    else:
        raise ValueError(f"Unsupported type {val_type.__name__}")


# Builds the initial trees of signature nodes for a given set of
# python arguments and keyword arguments.
def build_signature(*args: Any, **kwargs: Any):
    arg_signatures = [SignatureNode(x) for x in args]
    kwarg_signatures = {k: SignatureNode(v) for k, v in kwargs.items()}
    return (arg_signatures, kwarg_signatures)


# Tests if a signature is compatible with a slang function
def test_signature(
    signature: tuple[list[SignatureNode], dict[str, SignatureNode]],
    function_reflection: FunctionReflection,
) -> bool:
    if len(signature[0]) != len(slang_signature):
        return False
    for x, y in zip(signature[0], slang_signature):
        if x.python_marshal.name != y.name:
            return False
    return True
