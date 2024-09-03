from typing import Any
from sgl import TypeReflection
from kernelfunctions.callsignature import BasePythonTypeMarshal, register_python_type


class BuiltInScalarMarshal(BasePythonTypeMarshal):
    def __init__(self, python_type: type):
        super().__init__(python_type)

    def get_shape(self, value: Any):
        return (1,)

    def get_element_type(self, value: Any):
        return type(value)


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

    def get_shape(self, value: Any):
        return (1,)

    def get_element_type(self, value: Any):
        return type(value)


class NoneTypeMarshal(BasePythonTypeMarshal):
    def __init__(self):
        super().__init__(type(None))

    def get_shape(self, value: Any):
        return None

    def get_element_type(self, value: Any):
        return type(value)


register_python_type(int, IntMarshal(), None)
register_python_type(float, FloatMarshal(), None)
register_python_type(bool, BoolMarshal(), None)
register_python_type(dict, DictMarshall(), None)
register_python_type(type(None), NoneTypeMarshal(), None)
