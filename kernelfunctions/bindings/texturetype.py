

from typing import Optional

from kernelfunctions.core import BaseType, Shape

from kernelfunctions.backend import Texture, TypeReflection
from kernelfunctions.typeregistry import SLANG_STRUCT_TYPES_BY_NAME, get_or_create_type

from .valuetype import ValueType


class TextureType(ValueType):

    def __init__(self, element_type: BaseType, writable: bool, base_texture_type_name: str, texture_shape: int):
        super().__init__()
        self._writable = writable
        self._texture_shape = texture_shape
        self._base_texture_type_name = base_texture_type_name
        self.element_type = element_type
        self.name = f"{self._prefix()}{self._base_texture_type_name}<{self.element_type.name}>"

    def _prefix(self):
        return "RW" if self._writable else ""

    @property
    def differentiable(self):
        return self.element_type.differentiable


class Texture2DType(TextureType):
    def __init__(self, element_type: BaseType, writable: bool):
        super().__init__(element_type=element_type, writable=writable,
                         base_texture_type_name="Texture2D", texture_shape=2)

    def get_container_shape(self, value: Optional[Texture] = None) -> Shape:
        if value is not None:
            return Shape(value.width, value.height)
        else:
            return Shape(-1, -1)

    @property
    def derivative(self):
        el_diff = self.element_type.derivative
        if el_diff is not None:
            return Texture2DType(el_diff, self._writable)
        else:
            return None


def _get_or_create_slang_type_reflection(slang_type: TypeReflection) -> BaseType:
    assert isinstance(slang_type, TypeReflection)
    assert slang_type.kind == TypeReflection.Kind.resource
    if slang_type.resource_shape.name == "texture_2d":
        return Texture2DType(
            element_type=get_or_create_type(slang_type.resource_result_type),
            writable=slang_type.resource_access == TypeReflection.ResourceAccess.read_write)
    else:
        raise ValueError(f"Unsupported slang type {slang_type}")


SLANG_STRUCT_TYPES_BY_NAME['__TextureImpl'] = _get_or_create_slang_type_reflection
