from typing import Any, Optional, Union

import sgl
from kernelfunctions.buffer import StructuredBuffer
import kernelfunctions.typemappings as tm

N = object()

TTypeType = Optional[Union["BaseType", sgl.TypeReflection.ScalarType]]


def scalar_type_to_param_string(scalar_type: sgl.TypeReflection.ScalarType) -> str:
    return f"{str(scalar_type)[11:]}_t"


class BaseType:
    def __init__(
        self,
        reflection: sgl.TypeReflection,
    ):
        super().__init__()
        self.reflection = reflection

    @property
    def shape(self) -> tuple[Union[int, None], ...]:
        raise NotImplementedError()

    @property
    def dimensionality(self) -> int:
        return len(self.shape)

    @property
    def param_def_string(self) -> str:
        raise NotImplementedError()

    def is_compatible_with_python_type(self, python_type: type) -> bool:
        raise NotImplementedError()

    def is_compatible_with_python_value(self, python_value: Any, deep: bool) -> bool:
        return self.is_compatible_with_python_type(type(python_value))


# Fixed size scaler value such as a float, 1D with fixed length of 1
class ScalarType(BaseType):
    def __init__(self, reflection: sgl.TypeReflection):
        super().__init__(reflection)

    @property
    def shape(self) -> tuple[Union[int, None], ...]:
        return (1,)

    @property
    def slang_scalar_type(self) -> sgl.TypeReflection.ScalarType:
        return self.reflection.scalar_type

    @property
    def param_def_string(self) -> str:
        return f"{scalar_type_to_param_string(self.slang_scalar_type)}"

    def is_compatible_with_python_type(self, python_type: type) -> bool:
        return tm.is_valid_scalar_type_conversion(self.slang_scalar_type, python_type)


# Fixed size vector value such as a float3, 1D with fixed length of num components
class VectorType(BaseType):
    def __init__(self, reflection: sgl.TypeReflection):
        super().__init__(reflection)

    @property
    def shape(self) -> tuple[Union[int, None], ...]:
        return (self.reflection.col_count,)

    @property
    def slang_scalar_type(self) -> sgl.TypeReflection.ScalarType:
        return self.reflection.scalar_type

    @property
    def param_def_string(self) -> str:
        return f"vector<{scalar_type_to_param_string(self.slang_scalar_type)},{self.shape[0]}>"

    def is_compatible_with_python_type(self, python_type: type) -> bool:
        return tm.is_valid_vector_type_conversion(
            self.slang_scalar_type, python_type, self.reflection.col_count
        )


# Fixed size matrix value such as a float3x3, 2D with fixed length of num rows and num columns
class MatrixType(BaseType):
    def __init__(self, reflection: sgl.TypeReflection):
        super().__init__(reflection)

    @property
    def shape(self) -> tuple[Union[int, None], ...]:
        return (self.reflection.col_count, self.reflection.row_count)

    @property
    def slang_scalar_type(self) -> sgl.TypeReflection.ScalarType:
        return self.reflection.scalar_type

    @property
    def param_def_string(self) -> str:
        s = self.shape
        return f"matrix<{scalar_type_to_param_string(self.slang_scalar_type)},{s[0]},{s[1]}>"


# Fixed size array value, 1D with an optionall defined number of elements
class ArrayType(BaseType):
    def __init__(self, reflection: sgl.TypeReflection):
        super().__init__(reflection)

    @property
    def shape(self) -> tuple[Union[int, None], ...]:
        ec = self.reflection.element_count
        if ec >= 1:
            return (self.reflection.element_count,)
        else:
            return (None,)

    @property
    def param_def_string(self) -> str:
        s = self.shape
        if s[0] is not None:
            return f"{self.reflection.name}[{self.shape[0]}]"
        else:
            return f"{self.reflection.name}[]"

    @property
    def element_type(self) -> "BaseType":
        return convert_type(self.reflection.element_type)


# Structured value such as a struct { float a; float b; }, effectively a scalar with fields
class StructType(BaseType):
    def __init__(self, reflection: sgl.TypeReflection):
        super().__init__(reflection)

    @property
    def shape(self) -> tuple[Union[int, None], ...]:
        return (1,)

    @property
    def param_def_string(self) -> str:
        return self.reflection.name

    @property
    def fields(self) -> dict[str, "BaseType"]:
        return {
            field.name: convert_type(field.type) for field in self.reflection.fields
        }

    def is_compatible_with_python_type(self, python_type: type) -> bool:
        return python_type == dict

    def is_compatible_with_python_value(self, python_value: Any, deep: bool) -> bool:
        if not isinstance(python_value, dict):
            return False
        if deep:
            for field_name in self.fields:
                if field_name not in python_value:
                    return False
                if not self.fields[field_name].is_compatible_with_python_value(
                    python_value[field_name], deep
                ):
                    return False
        return True


# # Buffer value such as a StructuredBuffer<float>, 1D with undefined length
# class BufferType(BaseType):
#     def __init__(self, reflection: sgl.TypeReflection, value_type: TTypeType):
#         super().__init__(value_type, dimensionality=1)
#
#
# # Texture value such as a Texture2D<float>, 2D with undefined length
# class TextureType(BaseType):
#     def __init__(self, reflection: sgl.TypeReflection, value_type: TTypeType):
#         super().__init__(value_type, dimensionality=2)
#
#
# # Tensor value such as a Tensor<float>, ND with undefined length
# class TensorType(BaseType):
#     def __init__(self, reflection: sgl.TypeReflection, value_type: TTypeType, dimensionality: int):
#         super().__init__(value_type, dimensionality=dimensionality)


def convert_type(reflection: sgl.TypeReflection) -> BaseType:
    if reflection.kind == sgl.TypeReflection.Kind.scalar:
        return ScalarType(reflection)
    elif reflection.kind == sgl.TypeReflection.Kind.vector:
        return VectorType(reflection)
    elif reflection.kind == sgl.TypeReflection.Kind.matrix:
        return MatrixType(reflection)
    elif reflection.kind == sgl.TypeReflection.Kind.array:
        return ArrayType(reflection)
    elif reflection.kind == sgl.TypeReflection.Kind.struct:
        return StructType(reflection)
    else:
        raise TypeError(f"Unsupported type {reflection.kind}")


def try_convert_type(reflection: sgl.TypeReflection) -> Optional[BaseType]:
    try:
        return convert_type(reflection)
    except TypeError:
        return None


class Argument:
    def __init__(
        self, reflection: sgl.VariableReflection, value: Any, translation_type: BaseType
    ):
        super().__init__()
        self.value = value
        self.reflection = reflection
        self.translation_type = translation_type

    @property
    def input_def_string(self) -> str:
        if isinstance(self.value, StructuredBuffer):
            return f"RWStructuredBuffer<{self.translation_type.param_def_string}>"
        else:
            return self.translation_type.param_def_string

    @property
    def python_shape(self):
        if isinstance(self.value, StructuredBuffer):
            return (self.value.element_count,)
        else:
            return (1,)

    @property
    def name(self) -> str:
        return self.reflection.name

    @property
    def param_string(self) -> str:
        modifiers: list[str] = []
        for x in sgl.ModifierID:
            if self.reflection.has_modifier(x):
                modifiers.append(str(x)[11:])
        return f"{' '.join(modifiers)} {self.translation_type.param_def_string} {self.name}"

    def get_variable_access_string(self, indexer: str = "i"):
        if isinstance(self.value, StructuredBuffer):
            return f"{self.name}[{indexer}]"
        else:
            return f"{self.name}"


def try_create_argument(
    reflection: sgl.VariableReflection, value: Any, deep_check: bool = False
) -> Optional[Argument]:
    base_type = try_convert_type(reflection.type)
    if base_type is None:
        return None

    if base_type.is_compatible_with_python_value(value, deep_check):
        return Argument(reflection, value, base_type)

    if isinstance(value, StructuredBuffer):
        if base_type.is_compatible_with_python_type(value.element_type):
            return Argument(reflection, value, base_type)

    return None
