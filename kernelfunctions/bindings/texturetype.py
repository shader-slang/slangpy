

from typing import Any, Optional, Union

from kernelfunctions.backend.slangpynativeemulation import CallContext
from kernelfunctions.core import BaseType, Shape, AccessType, BindContext, BoundVariable, CodeGenBlock

from kernelfunctions.backend import Texture, TypeReflection, ResourceUsage, ResourceType, get_format_info, FormatType, ResourceView

from kernelfunctions.core.boundvariableruntime import BoundVariableRuntime
from kernelfunctions.typeregistry import PYTHON_SIGNATURES, PYTHON_TYPES, SLANG_STRUCT_TYPES_BY_NAME, SLANG_VECTOR_TYPES, get_or_create_type

from .valuetype import ValueType


def has_uav(usage: ResourceUsage):
    return (usage & ResourceUsage.unordered_access.value) != 0


def prefix(usage: ResourceUsage):
    return "RW" if has_uav(usage) else ""


class TextureType(ValueType):

    def __init__(self, resource_shape: TypeReflection.ResourceShape, element_type: BaseType, usage: ResourceUsage):
        super().__init__()
        self._resource_shape = resource_shape
        self._usage = usage

        tex_type = ""
        tex_dims = 0

        if resource_shape == TypeReflection.ResourceShape.texture_1d:
            tex_type = "Texture1D"
            tex_dims = 1
        elif resource_shape == TypeReflection.ResourceShape.texture_2d:
            tex_type = "Texture2D"
            tex_dims = 2
        elif resource_shape == TypeReflection.ResourceShape.texture_3d:
            tex_type = "Texture3D"
            tex_dims = 3
        elif resource_shape == TypeReflection.ResourceShape.texture_cube:
            tex_type = "TextureCube"
            tex_dims = 3
        elif resource_shape == TypeReflection.ResourceShape.texture_1d_array:
            tex_type = "Texture1DArray"
            tex_dims = 2
        elif resource_shape == TypeReflection.ResourceShape.texture_2d_array:
            tex_type = "Texture2DArray"
            tex_dims = 3
        elif resource_shape == TypeReflection.ResourceShape.texture_cube_array:
            tex_type = "TextureCubeArray"
            tex_dims = 4
        elif resource_shape == TypeReflection.ResourceShape.texture_2d_multisample:
            tex_type = "Texture2DMS"
            tex_dims = 2
        elif resource_shape == TypeReflection.ResourceShape.texture_2d_multisample_array:
            tex_type = "Texture2DMSArray"
            tex_dims = 3
        else:
            raise ValueError(f"Unsupported resource shape {resource_shape}")

        self._texture_dims = tex_dims
        self._base_texture_type_name = tex_type
        self.element_type = element_type
        self.name = f"{prefix(self._usage)}{self._base_texture_type_name}<{self.element_type.name}>"

    def resolve_type(self, context: BindContext, bound_type: 'BaseType'):
        if isinstance(bound_type, TextureType):
            if self._usage & bound_type._usage == 0:
                raise ValueError(
                    f"Cannot bind texture view {self.name} with usage {bound_type._usage}")
            if self._resource_shape != bound_type._resource_shape:
                raise ValueError(
                    f"Cannot bind texture view {self.name} with different shape {bound_type._resource_shape}")
            if self.element_type.name != bound_type.element_type.name:
                raise ValueError(
                    f"Cannot bind texture view {self.name} with different element type {bound_type.element_type.name}")
            return bound_type
        else:
            return super().resolve_type(context, bound_type)

    # Texture is writable if it has unordered access view.
    def is_writable(self):
        return has_uav(self._usage)

    # Generate the slangpy accessor type name (eg Texture2DType<float4>).
    def build_accessor_name(self, usage: Optional[ResourceUsage] = None):
        if usage is None:
            usage = self._usage
        assert self.element_type is not None
        return f"{prefix(usage)}{self._base_texture_type_name}Type<{self.element_type.name}>"

    # Call data can only be read access to primal, and simply declares it as a variable.
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        assert self.element_type is not None
        access = binding.access[0]
        name = binding.variable_name

        if access == AccessType.none:
            cgb.type_alias(f"_{name}", f"NoneType")
            return

        if binding.call_dimensionality == 0:
            # If broadcast directly, function is just taking the texture argument directly, so use the slang type
            assert access == AccessType.read
            assert isinstance(binding.slang.primal, TextureType)
            if self._usage & binding.slang.primal._usage == 0:
                raise ValueError(
                    f"Cannot bind texture view {name} with usage {binding.slang.primal._usage}")
            cgb.type_alias(
                f"_{name}", binding.slang.primal.build_accessor_name())
        elif binding.call_dimensionality == self._texture_dims:
            # If broadcast is the same shape as the texture, this is loading from pixels, so use the
            # type required to support the required access
            if access == AccessType.read:
                # Read access can be either shader resource or UAV, so just bind the correct type
                # for this resource view
                cgb.type_alias(
                    f"_{name}", self.build_accessor_name())
            else:
                # Write access requires a UAV so check it and bind RW type
                if not has_uav(self._usage):
                    raise ValueError(
                        f"Cannot write to read-only texture {name}")
                cgb.type_alias(
                    f"_{name}", self.build_accessor_name(ResourceUsage.unordered_access))
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
        resource_shape = self._resource_shape
        if resource_shape == TypeReflection.ResourceShape.texture_1d:
            return Shape(value.width >> mip)
        elif resource_shape == TypeReflection.ResourceShape.texture_2d or resource_shape == TypeReflection.ResourceShape.texture_2d_multisample:
            return Shape(value.width >> mip, value.height >> mip)
        elif resource_shape == TypeReflection.ResourceShape.texture_3d:
            return Shape(value.width >> mip, value.height >> mip, value.depth >> mip)
        elif resource_shape == TypeReflection.ResourceShape.texture_cube:
            return Shape(6, value.width >> mip, value.height >> mip)
        elif resource_shape == TypeReflection.ResourceShape.texture_1d_array:
            return Shape(value.array_size, value.width >> mip)
        elif resource_shape == TypeReflection.ResourceShape.texture_2d_array or resource_shape == TypeReflection.ResourceShape.texture_2d_multisample_array:
            return Shape(value.array_size, value.width >> mip, value.height >> mip)
        elif resource_shape == TypeReflection.ResourceShape.texture_cube_array:
            return Shape(value.array_size, 6, value.width >> mip, value.height >> mip)
        else:
            raise ValueError(f"Unsupported resource shape {resource_shape}")

    @property
    def differentiable(self):
        return self.element_type.differentiable

    @property
    def derivative(self):
        el_diff = self.element_type.derivative
        if el_diff is not None:
            return TextureType(self._resource_shape, el_diff, self._usage)
        else:
            return None


def _get_or_create_slang_type_reflection(slang_type: TypeReflection) -> BaseType:
    assert isinstance(slang_type, TypeReflection)
    assert slang_type.kind == TypeReflection.Kind.resource
    et = get_or_create_type(slang_type.resource_result_type)

    # A slang texture requires a specific usage to be bound to. a Texture
    # must be a shader resource, and an RWTexture must be a UAV.
    if slang_type.resource_access == TypeReflection.ResourceAccess.read:
        usage = ResourceUsage.shader_resource
    elif slang_type.resource_access == TypeReflection.ResourceAccess.read_write:
        usage = ResourceUsage.unordered_access
    else:
        raise ValueError(f"Unsupported resource access {slang_type.resource_access}")
    return TextureType(slang_type.resource_shape, et, usage)


SLANG_STRUCT_TYPES_BY_NAME['__TextureImpl'] = _get_or_create_slang_type_reflection
SLANG_STRUCT_TYPES_BY_NAME['_Texture'] = _get_or_create_slang_type_reflection


def get_or_create_python_texture_type(resource: Texture, usage: ResourceUsage):

    # Translate format into slang scalar type + channel count, which allows
    # us to build the element type of the texture.
    fmt_info = get_format_info(resource.desc.format)
    if fmt_info.type in [FormatType.float, FormatType.unorm, FormatType.snorm, FormatType.unorm_srgb]:
        scalar_type = TypeReflection.ScalarType.float32
    elif fmt_info.type == FormatType.uint:
        scalar_type = TypeReflection.ScalarType.uint32
    elif fmt_info.type == FormatType.sint:
        scalar_type = TypeReflection.ScalarType.int32
    else:
        raise ValueError(f"Unsupported format {resource.desc.format}")
    element_type = SLANG_VECTOR_TYPES[scalar_type][fmt_info.channel_count]

    # Translate resource type + array size into a slang resource shape.
    resource_shape = TypeReflection.ResourceShape.none
    if resource.array_size == 1:
        if resource.desc.type == ResourceType.texture_1d:
            resource_shape = TypeReflection.ResourceShape.texture_1d
        elif resource.desc.type == ResourceType.texture_2d:
            if resource.desc.sample_count == 1:
                resource_shape = TypeReflection.ResourceShape.texture_2d
            else:
                resource_shape = TypeReflection.ResourceShape.texture_2d_multisample
        elif resource.desc.type == ResourceType.texture_3d:
            resource_shape = TypeReflection.ResourceShape.texture_3d
        elif resource.desc.type == ResourceType.texture_cube:
            resource_shape = TypeReflection.ResourceShape.texture_cube
        else:
            raise ValueError(f"Unsupported texture type {resource.desc.type}")
    else:
        if resource.desc.type == ResourceType.texture_1d:
            resource_shape = TypeReflection.ResourceShape.texture_1d_array
        elif resource.desc.type == ResourceType.texture_2d:
            if resource.desc.sample_count == 1:
                resource_shape = TypeReflection.ResourceShape.texture_2d_array
            else:
                resource_shape = TypeReflection.ResourceShape.texture_2d_multisample_array
        elif resource.desc.type == ResourceType.texture_cube:
            resource_shape = TypeReflection.ResourceShape.texture_cube_array
        else:
            raise ValueError(f"Unsupported texture type {resource.desc.type}")

    return TextureType(resource_shape, element_type, usage)


def _get_or_create_python_type(value: Any):
    assert isinstance(value, Texture)
    usage = value.desc.usage
    return get_or_create_python_texture_type(value, usage)


PYTHON_TYPES[Texture] = _get_or_create_python_type
PYTHON_SIGNATURES[Texture] = lambda x: f"[{x.desc.type},{x.desc.usage},{x.desc.format},{x.array_size>1}]"
