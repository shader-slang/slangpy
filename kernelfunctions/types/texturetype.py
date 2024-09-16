

from typing import Any, Optional

from sgl import Buffer, Texture, TypeReflection
import kernelfunctions.typeregistry as tr
from kernelfunctions.types.basetype import BaseType
from kernelfunctions.types.valuetypeimpl import ValueTypeImpl


class TextureType(ValueTypeImpl):

    def __init__(self, element_type: BaseType, writable: bool, base_texture_type_name: str, texture_shape: int):
        super().__init__()
        self._el_type = element_type
        self._writable = writable
        self._texture_shape = texture_shape
        self._base_texture_type_name = base_texture_type_name
        self._tex_type_name = f"{self._prefix()}{self._base_texture_type_name}{self._el_type.name()}"

    def _prefix(self):
        return "RW" if self._writable else ""

    def name(self, value: Optional[Buffer] = None) -> str:
        return self._tex_type_name

    def element_type(self, value: Any = None):
        return self._el_type

    def differentiable(self, value: Optional[Buffer] = None):
        return self.element_type(value).differentiable()


class Texture2DType(TextureType):
    def __init__(self, element_type: BaseType, writable: bool):
        super().__init__(element_type=element_type, writable=writable,
                         base_texture_type_name="Texture2D", texture_shape=2)

    def container_shape(self, value: Optional[Texture] = None):
        if value is not None:
            return (value.width, value.height)
        else:
            return (None, None)

    def differentiate(self, value: Optional[Buffer] = None):
        el_diff = self.element_type(value).differentiate()
        if el_diff is not None:
            return Texture2DType(el_diff, self._writable)
        else:
            return None


def _get_or_create_slang_type_reflection(slang_type: TypeReflection) -> BaseType:
    assert isinstance(slang_type, TypeReflection)
    assert slang_type.kind == TypeReflection.Kind.resource
    if slang_type.resource_shape.name == "texture_2d":
        return Texture2DType(
            element_type=tr.get_or_create_type(slang_type.resource_result_type),
            writable=slang_type.resource_access == TypeReflection.ResourceAccess.read_write)
    else:
        raise ValueError(f"Unsupported slang type {slang_type}")


tr.SLANG_STRUCT_TYPES_BY_NAME['__TextureImpl'] = _get_or_create_slang_type_reflection
