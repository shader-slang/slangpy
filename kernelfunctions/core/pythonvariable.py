from types import NoneType
from typing import Any, Optional
from kernelfunctions.backend.slangpynativeemulation import Shape
from kernelfunctions.typeregistry import get_or_create_type
from .basevariableimpl import BaseVariableImpl
from .basetype import BaseType


class PythonFunctionCall:
    def __init__(self, *args: Any, **kwargs: Any) -> NoneType:
        super().__init__()
        self.args = [PythonVariable(x, None, None) for x in args]
        self.kwargs = {n: PythonVariable(v, None, n) for n, v in kwargs.items()}

    @property
    def num_function_args(self) -> int:
        total = len(self.args) + self.num_function_kwargs
        return total

    @property
    def num_function_kwargs(self) -> int:
        total = len(self.kwargs)
        if "_this" in self.kwargs:
            total -= 1
        if "_result" in self.kwargs:
            total -= 1
        return total

    @property
    def has_implicit_args(self) -> bool:
        return any(x.vector_type is None for x in self.args)

    @property
    def has_implicit_mappings(self) -> bool:
        return any(not x.vector_mapping.valid for x in self.args)

    def apply_explicit_vectorization(self, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if len(args) > len(self.args):
            raise ValueError("Too many arguments supplied for explicit vectorization")
        if len(kwargs) > len(self.kwargs):
            raise ValueError(
                "Too many keyword arguments supplied for explicit vectorization")

        for i, arg in enumerate(args):
            self.args[i].apply_explicit_vectorization(arg)

        for name, arg in kwargs.items():
            if not name in self.kwargs:
                raise ValueError(f"Unknown keyword argument {name}")
            self.kwargs[name].apply_explicit_vectorization(arg)


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

        self.parameter_index = -1
        self.vector_mapping: Shape = Shape(None)
        self.vector_type: Optional[BaseType] = None

    def set_type(self, new_type: BaseType, value: Any = None):
        self.primal = new_type
        self.derivative = self.primal.derivative

    def apply_explicit_vectorization(self, mapping: Any):
        """
        Apply explicit vectorization to this variable and children.
        This will result in any explicit mapping or typing provided
        by the caller being stored on the corresponding bound variable.
        """
        if self.fields is not None:
            assert isinstance(mapping, dict)
            for name, child in self.fields.items():
                child_mapping = mapping.get(name)
                if child_mapping is not None:
                    assert isinstance(child, PythonVariable)
                    child.apply_explicit_vectorization(child_mapping)

            type_mapping = mapping.get("$type")
            if type_mapping is not None:
                self._apply_explicit_vectorization(type_mapping)
        else:
            self._apply_explicit_vectorization(mapping)

    def _apply_explicit_vectorization(self, mapping: Any):
        if isinstance(mapping, tuple):
            self.vector_mapping = Shape(*mapping)
            self.vector_type = self.primal.reduce_type(len(mapping))
        elif mapping is not None:
            self.vector_type = get_or_create_type(mapping)
