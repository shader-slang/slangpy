

from typing import Any, Optional, Union

from kernelfunctions.backend.slangpynativeemulation import CallContext
from kernelfunctions.core import BaseType, Shape, AccessType, BindContext, BoundVariable, CodeGenBlock

from kernelfunctions.backend import Texture, TypeReflection, ResourceUsage, ResourceType, get_format_info, FormatType, ResourceView

from kernelfunctions.core.boundvariableruntime import BoundVariableRuntime
from kernelfunctions.typeregistry import PYTHON_SIGNATURES, PYTHON_TYPES, SLANG_STRUCT_TYPES_BY_NAME, SLANG_VECTOR_TYPES, get_or_create_type

from .valuetype import ValueType


class TextureType(ValueType):

    def __init__(self, element_type: BaseType, writable: bool, base_texture_type_name: str, texture_dims: int):
        super().__init__()
        self._writable = writable
        self._texture_dims = texture_dims
        self._base_texture_type_name = base_texture_type_name
        self.element_type = element_type
        self.name = f"{self._prefix()}{self._base_texture_type_name}<{self.element_type.name}>"

    def is_writable(self):
        return self._writable

    def _prefix(self):
        return "RW" if self._writable else ""

    def build_accessor_name(self, writable: Optional[bool] = None):
        if writable is None:
            writable = self._writable
        assert self.element_type is not None
        prefix = "RW" if writable else ""
        return f"{prefix}{self._base_texture_type_name}Type<{self.element_type.name}>"

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        assert self.element_type is not None
        access = binding.access
        name = binding.variable_name

        if binding.call_dimensionality == 0:
            # If broadcast directly, function is just taking the texture argument directly, so use the slang type
            assert not access[0] in [AccessType.write, AccessType.readwrite]
            assert isinstance(binding.slang.primal, TextureType)
            if binding.slang.primal._writable and not self._writable:
                raise ValueError(
                    f"Cannot bind read-only texture to writable texture {name}")
            cgb.type_alias(
                f"_{name}", binding.slang.primal.build_accessor_name())
        elif binding.call_dimensionality == self._texture_dims:
            # If broadcast is the same shape as the texture, this is loading from pixels, so use the
            # type required to support the required access
            if not self._writable and access[0] in [AccessType.write, AccessType.readwrite]:
                raise ValueError(f"Cannot write to read-only texture {name}")
            if access[0] in [AccessType.write, AccessType.readwrite]:
                cgb.type_alias(
                    f"_{name}", self.build_accessor_name(True))
            elif access[0] == AccessType.read:
                cgb.type_alias(
                    f"_{name}", self.build_accessor_name(False))
            else:
                cgb.type_alias(f"_{name}", f"NoneType")
        else:
            raise ValueError(
                f"Texture {name} has invalid transform {binding.transform}")

    # Call data just returns the primal
    def create_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: Any) -> Any:
        access = binding.access
        if access[0] != AccessType.none:
            return {
                'value': data
            }

    # Container shape internally handles both textures or views onto textures,
    # which lets it deal with views onto none-zero mip levels of a texture.
    def get_container_shape(self, value: Optional[Union[Texture, ResourceView]] = None) -> Shape:
        mip = 0
        if isinstance(value, ResourceView):
            mip = value.subresource_range.mip_level
            assert isinstance(value.resource, Texture)
            value = value.resource
        if value is not None:
            res = self.get_texture_shape(value, mip)
            assert len(res) == self._texture_dims
            return res
        else:
            return Shape((-1,)*self._texture_dims)

    def get_texture_shape(self, value: Texture, mip: int) -> Shape:
        raise NotImplementedError()

    @property
    def differentiable(self):
        return self.element_type.differentiable

    @property
    def derivative(self):
        el_diff = self.element_type.derivative
        if el_diff is not None:
            # Note: all subtypes of TextureType take just element type + writable
            return type(self)(el_diff, self._writable)  # type: ignore
        else:
            return None


class Texture1DType(TextureType):
    def __init__(self, element_type: BaseType, writable: bool):
        super().__init__(element_type=element_type, writable=writable,
                         base_texture_type_name="Texture1D", texture_dims=1)

    def get_texture_shape(self, value: Texture, mip: int) -> Shape:
        return Shape(value.width >> mip)


class Texture2DType(TextureType):
    def __init__(self, element_type: BaseType, writable: bool):
        super().__init__(element_type=element_type, writable=writable,
                         base_texture_type_name="Texture2D", texture_dims=2)

    def get_texture_shape(self, value: Texture, mip: int) -> Shape:
        return Shape(value.width >> mip, value.height >> mip)


class Texture1DArrayType(TextureType):
    def __init__(self, element_type: BaseType, writable: bool):
        super().__init__(element_type=element_type, writable=writable,
                         base_texture_type_name="Texture1DArray", texture_dims=2)

    def get_texture_shape(self, value: Texture, mip: int) -> Shape:
        return Shape(value.array_size, value.width >> mip)


class Texture2DArrayType(TextureType):
    def __init__(self, element_type: BaseType, writable: bool):
        super().__init__(element_type=element_type, writable=writable,
                         base_texture_type_name="Texture2DArray", texture_dims=3)

    def get_texture_shape(self, value: Texture, mip: int) -> Shape:
        return Shape(value.array_size, value.width >> mip, value.height >> mip)


class Texture3DType(TextureType):
    def __init__(self, element_type: BaseType, writable: bool):
        super().__init__(element_type=element_type, writable=writable,
                         base_texture_type_name="Texture3D", texture_dims=3)

    def get_texture_shape(self, value: Texture, mip: int) -> Shape:
        return Shape(value.width >> mip, value.height >> mip, value.depth >> mip)


class TextureCubeType(TextureType):
    def __init__(self, element_type: BaseType, writable: bool):
        super().__init__(element_type=element_type, writable=writable,
                         base_texture_type_name="TextureCube", texture_dims=3)

    def get_texture_shape(self, value: Texture, mip: int) -> Shape:
        return Shape(6, value.width >> mip, value.height >> mip)


class TextureCubeArrayType(TextureType):
    def __init__(self, element_type: BaseType, writable: bool):
        super().__init__(element_type=element_type, writable=writable,
                         base_texture_type_name="TextureCubeArray", texture_dims=4)

    def get_texture_shape(self, value: Texture, mip: int) -> Shape:
        return Shape(value.array_size, 6, value.width >> mip, value.height >> mip)


def _get_or_create_slang_type_reflection(slang_type: TypeReflection) -> BaseType:
    assert isinstance(slang_type, TypeReflection)
    assert slang_type.kind == TypeReflection.Kind.resource
    et = get_or_create_type(slang_type.resource_result_type)
    writable = slang_type.resource_access == TypeReflection.ResourceAccess.read_write
    if slang_type.resource_shape == TypeReflection.ResourceShape.texture_1d:
        return Texture1DType(element_type=et, writable=writable)
    elif slang_type.resource_shape == TypeReflection.ResourceShape.texture_2d:
        return Texture2DType(element_type=et, writable=writable)
    elif slang_type.resource_shape == TypeReflection.ResourceShape.texture_3d:
        return Texture3DType(element_type=et, writable=writable)
    elif slang_type.resource_shape == TypeReflection.ResourceShape.texture_cube:
        return TextureCubeType(element_type=et, writable=writable)
    elif slang_type.resource_shape == TypeReflection.ResourceShape.texture_1d_array:
        return Texture1DArrayType(element_type=et, writable=writable)
    elif slang_type.resource_shape == TypeReflection.ResourceShape.texture_2d_array:
        return Texture2DArrayType(element_type=et, writable=writable)
    elif slang_type.resource_shape == TypeReflection.ResourceShape.texture_cube_array:
        return TextureCubeArrayType(element_type=et, writable=writable)
    else:
        raise ValueError(f"Unsupported slang type {slang_type}")


SLANG_STRUCT_TYPES_BY_NAME['__TextureImpl'] = _get_or_create_slang_type_reflection
SLANG_STRUCT_TYPES_BY_NAME['_Texture'] = _get_or_create_slang_type_reflection


def get_or_create_python_texture_type(value: Texture, usage: ResourceUsage):
    writable = (usage & ResourceUsage.unordered_access.value) != 0

    fmt_info = get_format_info(value.desc.format)
    if fmt_info.type in [FormatType.float, FormatType.unorm, FormatType.snorm, FormatType.unorm_srgb]:
        scalar_type = TypeReflection.ScalarType.float32
    elif fmt_info.type == FormatType.uint:
        scalar_type = TypeReflection.ScalarType.uint32
    elif fmt_info.type == FormatType.sint:
        scalar_type = TypeReflection.ScalarType.int32
    else:
        raise ValueError(f"Unsupported format {value.desc.format}")
    element_type = SLANG_VECTOR_TYPES[scalar_type][fmt_info.channel_count]

    if value.array_size == 1:
        if value.desc.type == ResourceType.texture_1d:
            return Texture1DType(element_type, writable)
        elif value.desc.type == ResourceType.texture_2d:
            return Texture2DType(element_type, writable)
        elif value.desc.type == ResourceType.texture_3d:
            return Texture3DType(element_type, writable)
        elif value.desc.type == ResourceType.texture_cube:
            return TextureCubeType(element_type, writable)
        else:
            raise ValueError(f"Unsupported texture type {value.desc.type}")
    else:
        if value.desc.type == ResourceType.texture_1d:
            return Texture1DArrayType(element_type, writable)
        elif value.desc.type == ResourceType.texture_2d:
            return Texture2DArrayType(element_type, writable)
        elif value.desc.type == ResourceType.texture_cube:
            return TextureCubeArrayType(element_type, writable)
        else:
            raise ValueError(f"Unsupported texture type {value.desc.type}")


def _get_or_create_python_type(value: Any):
    assert isinstance(value, Texture)
    usage = value.desc.usage
    return get_or_create_python_texture_type(value, usage)


PYTHON_TYPES[Texture] = _get_or_create_python_type
PYTHON_SIGNATURES[Texture] = lambda x: f"[{x.desc.type},{x.desc.usage},{x.desc.format},{x.array_size>1}]"
