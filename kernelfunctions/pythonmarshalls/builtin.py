from sgl import TypeReflection
from kernelfunctions.callsignature import BasePythonTypeMarshal, register_python_type
from kernelfunctions.typemappings import is_valid_scalar_type_conversion


class BuiltInScalarMarshal(BasePythonTypeMarshal):
    def __init__(self, python_type: type):
        super().__init__(python_type)
        self.shape = (1,)

    def is_compatible(self, slang_type: TypeReflection) -> bool:
        return slang_type.kind == TypeReflection.Kind.scalar and is_valid_scalar_type_conversion(slang_type.scalar_type, self.type)


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

    def is_compatible(self, slang_type: TypeReflection) -> bool:
        return slang_type.kind == TypeReflection.Kind.struct


class NoneTypeMarshal(BasePythonTypeMarshal):
    def __init__(self):
        super().__init__(type(None))

    def is_compatible(self, slang_type: TypeReflection) -> bool:
        return slang_type.kind == TypeReflection.Kind.scalar


register_python_type(int, IntMarshal(), None)
register_python_type(float, FloatMarshal(), None)
register_python_type(bool, BoolMarshal(), None)
register_python_type(dict, DictMarshall(), None)
register_python_type(type(None), NoneTypeMarshal(), None)
