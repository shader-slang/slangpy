from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Optional

from slangpy.bindings.basetype import BaseType
from slangpy.backend import SlangModule

if TYPE_CHECKING:
    from slangpy.reflection import SlangProgramLayout

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

TTypeLookup = Callable[['SlangProgramLayout', Any], BaseType]

# Dictionary of python types to corresponding base type
PYTHON_TYPES: dict[type, TTypeLookup] = {}

# Dictionary of python types to custom function that returns a signature
# Note: preferred mechanism is to provide a slangpy_signature attribute
PYTHON_SIGNATURES: dict[type, Optional[Callable[[Any], str]]] = {}

# There is not currently a way to go from TypeReflection to the enclosing scope,
# so we need this global state to retain it for now. The reflection API should be
# changed to allow removing this in the future
_cur_module: list[SlangModule] = []


def cur_scope() -> Optional[SlangModule]:
    return _cur_module[-1] if len(_cur_module) > 0 else None


class scope:
    def __init__(self, module: SlangModule):
        super().__init__()
        self.module = module

    def __enter__(self):
        _cur_module.append(self.module)

    def __exit__(self, exception_type: Any, exception_value: Any, exception_traceback: Any):
        _cur_module.pop()


def get_or_create_type(layout: 'SlangProgramLayout', python_type: Any, value: Any = None) -> BaseType:
    if isinstance(python_type, type):
        cb = PYTHON_TYPES.get(python_type)
        if cb is None:
            raise ValueError(f"Unsupported type {python_type}")
        res = cb(layout, value)
        if res is None:
            raise ValueError(f"Unsupported type {python_type}")
        return res
    elif isinstance(python_type, BaseType):
        return python_type
    else:
        raise ValueError(f"Unsupported type {python_type}")
