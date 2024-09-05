from typing import Any, Optional
from kernelfunctions.codegen import CodeGen, declare
from kernelfunctions.shapes import TConcreteShape
from kernelfunctions.typeregistry import AccessType, BasePythonTypeMarshal, register_python_type


class BuiltInScalarMarshal(BasePythonTypeMarshal):
    def __init__(self, python_type: type):
        super().__init__(python_type)

    def get_element_shape(self, value: Any):
        return (1,)


class IntMarshal(BuiltInScalarMarshal):
    def __init__(self):
        super().__init__(int)


class FloatMarshal(BuiltInScalarMarshal):
    def __init__(self):
        super().__init__(float)


class BoolMarshal(BuiltInScalarMarshal):
    def __init__(self):
        super().__init__(bool)


class DictMarshall(BasePythonTypeMarshal):
    def __init__(self):
        super().__init__(dict)

    def is_differentiable(self, value: Any) -> bool:
        # a python struct always assumes it could have a differentiable version,
        # which would be another struct
        return True

    def get_shape(self, value: Any):
        return (1,)


class NoneTypeMarshal(BasePythonTypeMarshal):
    """
    None type occurs for basic return values when user hasn't provided a destination
    """

    def __init__(self):
        super().__init__(type(None))

    def is_differentiable(self, value: Any) -> bool:
        # a None type is a request to auto-allocate a buffer, so it is by definition
        # possible to differentiate, as we'd auto allocate a differentiable buffer!
        return True

    def get_element_shape(self, value: Any):
        return None

    def is_writable(self, value: Any) -> bool:
        return True


register_python_type(int, IntMarshal(), None)
register_python_type(float, FloatMarshal(), None)
register_python_type(bool, BoolMarshal(), None)
register_python_type(dict, DictMarshall(), None)
register_python_type(type(None), NoneTypeMarshal(), None)
