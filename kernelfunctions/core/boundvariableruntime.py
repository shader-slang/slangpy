from typing import TYPE_CHECKING

from .native import NativeBoundCallRuntime, NativeBoundVariableRuntime, Shape
from kernelfunctions.shapes import check_concrete

if TYPE_CHECKING:
    from .boundvariable import BoundVariable, BoundCall


class BoundCallRuntime(NativeBoundCallRuntime):
    def __init__(self, call: 'BoundCall'):
        super().__init__()
        self.args = [BoundVariableRuntime(arg) for arg in call.args]
        self.kwargs = {name: BoundVariableRuntime(
            arg) for name, arg in call.kwargs.items()}


class BoundVariableRuntime(NativeBoundVariableRuntime):
    def __init__(self, source: 'BoundVariable'):
        super().__init__()

        # Data potentially used by type marshalls
        self.access = source.access
        self.transform = check_concrete(
            source.transform) if source.transform.valid else Shape(None)
        self.slang_shape = source.slang.primal.get_shape()
        self.python_type = source.python.primal

        # Temp data stored / updated each call
        self.shape = Shape(None)

        # Internal data
        self._source_for_exceptions = source
        self._name = source.python.name
        self._variable_name = source.variable_name
        self._children = {
            name: BoundVariableRuntime(child) for name, child in source.children.items()
        } if source.children is not None else None
