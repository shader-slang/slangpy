from typing import Any, Optional
from kernelfunctions.codegen import declare
from kernelfunctions.shapes import TConcreteShape
from kernelfunctions.typeregistry import AccessType, BasePythonTypeMarshal, register_python_type


class BuiltInScalarMarshal(BasePythonTypeMarshal):
    def __init__(self, python_type: type):
        super().__init__(python_type)

    def get_shape(self, value: Any):
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

    def get_shape(self, value: Any):
        return (1,)


class NoneTypeMarshal(BasePythonTypeMarshal):
    """
    None type occurs for basic return values when user hasn't provided a destination
    """

    def __init__(self):
        super().__init__(type(None))

    def get_shape(self, value: Any):
        return None

    def is_writable(self, value: Any) -> bool:
        return True

    def declare_inputs(self,
                       name: str, shape: TConcreteShape,
                       primal_type: Optional[str], primal_access: AccessType,
                       derivative_type: Optional[str], derivative_access: AccessType,
                       out_inputs: list[Any]):
        if primal_access == AccessType.write:
            out_inputs.append(
                declare(f"RWTensorBuffer<{primal_type},{max(len(shape),1)}>", f"{name}_primal"))


register_python_type(int, IntMarshal(), None)
register_python_type(float, FloatMarshal(), None)
register_python_type(bool, BoolMarshal(), None)
register_python_type(dict, DictMarshall(), None)
register_python_type(type(None), NoneTypeMarshal(), None)
