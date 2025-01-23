# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

from slangpy.core.native import (NativeBoundCallRuntime,
                                 NativeBoundVariableRuntime, Shape)

if TYPE_CHECKING:
    from .boundvariable import BoundCall, BoundVariable
    from slangpy.reflection.reflectiontypes import SlangType
    from slangpy.bindings.marshall import Marshall


class BoundCallRuntime(NativeBoundCallRuntime):
    """
    Minimal call data stored after kernel generation required to
    dispatch a call to a SlangPy kernel.
    """

    def __init__(self, call: 'BoundCall'):
        super().__init__()

        #: Positional arguments.
        self.args = [BoundVariableRuntime(arg) for arg in call.args]

        #: Keyword arguments.
        self.kwargs = {name: BoundVariableRuntime(
            arg) for name, arg in call.kwargs.items()}


class BoundVariableRuntime(NativeBoundVariableRuntime):
    """
    Minimal variable data stored after kernel generation required to
    dispatch a call to a SlangPy kernel.
    """

    def __init__(self, source: 'BoundVariable'):
        super().__init__()

        #: Access type (in/out/inout).
        self.access = source.access

        #: Mapping of dimensions.
        self.transform = source.vector_mapping

        #: Python type of variable.
        if source.python is not None:
            self.python_type: 'Marshall' = source.python

        #: Slang type being passed to.
        if source.vector_type is not None:
            self.vector_type: 'SlangType' = source.vector_type  # type: ignore

        #: Call dimensionality of variable.
        self.call_dimensionality = source.call_dimensionality

        # Temp data stored / updated each call.
        self.shape = Shape(None)

        #: Reference to original bound variable for use during exception
        #: handling.
        self._source_for_exceptions = source

        #: Name of variable.
        self.variable_name = source.variable_name

        if source.children is not None:
            #: Child variables.
            self.children = {
                name: BoundVariableRuntime(child) for name, child in source.children.items()
            }
