from typing import Any
from kernelfunctions.typeregistry import register_python_type
from kernelfunctions.types import PythonMarshal
import numpy as np
import numpy.typing as npt


class BuiltInScalarMarshal(PythonMarshal):
    def __init__(self, python_type: type):
        super().__init__(python_type)

    def get_element_shape(self, value: Any):
        return (1,)


class IntMarshal(BuiltInScalarMarshal):
    def __init__(self):
        super().__init__(int)

    def primal_to_numpy(self, value: Any):
        return np.array([value], dtype=np.int32)

    def primal_from_numpy(self, value: npt.NDArray[np.int32]):
        return value.view(np.int32)[0]


class FloatMarshal(BuiltInScalarMarshal):
    def __init__(self):
        super().__init__(float)

    def primal_to_numpy(self, value: Any):
        return np.array([value], dtype=np.float32)

    def primal_from_numpy(self, value: npt.NDArray[np.float32]):
        return value.view(np.float32)[0]


class BoolMarshal(BuiltInScalarMarshal):
    def __init__(self):
        super().__init__(bool)

    def primal_to_numpy(self, value: Any):
        return np.array([1 if value else 0], dtype=np.int32)

    def primal_from_numpy(self, value: npt.NDArray[np.int32]):
        return value.view(np.int32)[0] == 0


class DictMarshall(PythonMarshal):
    def __init__(self):
        super().__init__(dict)

    def is_differentiable(self, value: Any) -> bool:
        # a python struct always assumes it could have a differentiable version,
        # which would be another struct
        return True

    def get_shape(self, value: Any):
        return (1,)


class NoneTypeMarshal(PythonMarshal):
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
