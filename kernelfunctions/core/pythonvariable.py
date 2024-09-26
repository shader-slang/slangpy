from types import NoneType
from typing import Any, Optional
from kernelfunctions.typeregistry import get_or_create_type
from .basevariableimpl import BaseVariableImpl
from .basetype import BaseType


class PythonFunctionCall:
    def __init__(self, *args: Any, **kwargs: Any) -> NoneType:
        super().__init__()
        self.args = [PythonVariable(x, None, None) for x in args]
        self.kwargs = {n: PythonVariable(v, None, n) for n, v in kwargs.items()}


class PythonVariable(BaseVariableImpl):
    def __init__(self,
                 value: Any,
                 parent: Optional['PythonVariable'],
                 name: Optional[str]):
        super().__init__()

        self.name = name if name is not None else ""
        self.set_type(get_or_create_type(type(value), value), value)

        if isinstance(value, dict):
            self.fields = {n: PythonVariable(v, self, n) for n, v in value.items()}
        else:
            self.fields = None

    def set_type(self, new_type: BaseType, value: Any = None):
        self.primal = new_type
        self.derivative = self.primal.derivative
        primal_shape = self.primal.get_shape(value)
        self.dimensionality = len(primal_shape) if primal_shape is not None else None

    def update_from_slang_type(self, slang_type: BaseType):
        if self.dimensionality is None:
            self.primal.update_from_bound_type(slang_type)
            primal_shape = self.primal.get_shape()
            self.dimensionality = len(primal_shape) if primal_shape is not None else None
