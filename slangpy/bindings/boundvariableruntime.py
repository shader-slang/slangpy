from typing import TYPE_CHECKING

from slangpy.core.native import NativeBoundCallRuntime, NativeBoundVariableRuntime, Shape

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
        self.transform = source.vector_mapping
        self.python_type = source.python
        self.vector_type = source.vector_type
        self.call_dimensionality = source.call_dimensionality

        # Temp data stored / updated each call
        self.shape = Shape(None)

        # Internal data
        self._source_for_exceptions = source
        self.variable_name = source.variable_name

        if source.children is not None:
            self.children = {
                name: BoundVariableRuntime(child) for name, child in source.children.items()
            }
