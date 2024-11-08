from types import NoneType
from typing import TYPE_CHECKING, Any, Optional
from kernelfunctions.typeregistry import get_or_create_type
from .basevariableimpl import BaseVariableImpl
from .basetype import BaseType
from .native import Shape
from .reflection import SlangType, SlangProgramLayout

if TYPE_CHECKING:
    from .basetype import BindContext


class PythonVariableException(Exception):
    def __init__(self, message: str, variable: 'PythonVariable') -> NoneType:
        super().__init__(message)
        self.message = message
        self.variable = variable


class PythonFunctionCall:
    def __init__(self,
                 context: 'BindContext', *args: Any, **kwargs: Any) -> NoneType:
        super().__init__()
        self.args = [PythonVariable(context, x, None, None) for x in args]
        self.kwargs = {n: PythonVariable(context, v, None, n) for n, v in kwargs.items()}

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

    def apply_explicit_vectorization(self, context: 'BindContext', args: tuple[Any, ...], kwargs: dict[str, Any]):
        if len(args) > len(self.args):
            raise ValueError("Too many arguments supplied for explicit vectorization")
        if len(kwargs) > len(self.kwargs):
            raise ValueError(
                "Too many keyword arguments supplied for explicit vectorization")

        for i, arg in enumerate(args):
            self.args[i].apply_explicit_vectorization(context, arg)

        for name, arg in kwargs.items():
            if not name in self.kwargs:
                raise ValueError(f"Unknown keyword argument {name}")
            self.kwargs[name].apply_explicit_vectorization(context, arg)


class PythonVariable(BaseVariableImpl):
    def __init__(self,
                 context: 'BindContext',
                 value: Any,
                 parent: Optional['PythonVariable'],
                 name: Optional[str]):
        super().__init__()

        self.name = name if name is not None else ""
        self.set_type(get_or_create_type(context.layout, type(value), value), value)

        if isinstance(value, dict):
            self.fields = {n: PythonVariable(context, v, self, n)
                           for n, v in value.items()}
        else:
            self.fields = None

        self.parameter_index = -1
        self.vector_mapping: Shape = Shape(None)
        self.vector_type: Optional[SlangType] = None
        self.explicitly_vectorized = False

    @property
    def has_derivative(self):
        return self.primal.has_derivative

    def set_type(self, new_type: BaseType, value: Any = None):
        self.primal = new_type

    def apply_explicit_vectorization(self, context: 'BindContext', mapping: Any):
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
                    child.apply_explicit_vectorization(context, child_mapping)

            type_mapping = mapping.get("$type")
            if type_mapping is not None:
                self._apply_explicit_vectorization(context, type_mapping)
        else:
            self._apply_explicit_vectorization(context, mapping)

    def _apply_explicit_vectorization(self, context: 'BindContext', mapping: Any):
        try:
            if isinstance(mapping, tuple):
                self.vector_mapping = Shape(*mapping)
                self.vector_type = self.primal.reduce_type(context, len(mapping))
                self.explicitly_vectorized = True
            elif isinstance(mapping, SlangType):
                self.vector_type = mapping
                self.explicitly_vectorized = True
            elif isinstance(mapping, str):
                self.vector_type = context.layout.find_type_by_name(mapping)
                self.explicitly_vectorized = True
            elif isinstance(mapping, type):
                marshall = get_or_create_type(context.layout, mapping)
                if not marshall:
                    raise PythonVariableException(
                        f"Invalid explicit type: {mapping}", self)
                self.vector_type = marshall.slang_type
                self.explicitly_vectorized = True
            else:
                raise PythonVariableException(
                    f"Invalid explicit type: {mapping}", self)
        except Exception as e:
            raise PythonVariableException(
                f"Explicit vectorization raised exception: {e.__repr__()}", self)
