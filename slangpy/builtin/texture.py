# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Any

from slangpy.core.native import AccessType, CallContext, Shape, NativeTextureMarshall
from slangpy.backend import TypeReflection

from slangpy.backend import (FormatType, TextureType, TextureUsage, Sampler,
                             Texture, Format, get_format_info)
from slangpy.bindings import (PYTHON_SIGNATURES, PYTHON_TYPES, Marshall,
                              BindContext, BoundVariable, BoundVariableRuntime,
                              CodeGenBlock, ReturnContext)
from typing import Optional

import slangpy.reflection as refl

SCALARTYPE_TO_TEXTURE_FORMAT = {
    TypeReflection.ScalarType.float32: (Format.r32_float, Format.rg32_float, Format.rgb32_float, Format.rgba32_float),
    TypeReflection.ScalarType.float16: (Format.r16_float, Format.rg16_float, None, Format.rgba16_float),
    TypeReflection.ScalarType.uint32: (Format.r32_uint, Format.rg32_uint, Format.rgb32_uint, Format.rgba32_uint),
    TypeReflection.ScalarType.uint16: (Format.r16_unorm, Format.rg16_unorm, None, Format.rgba16_unorm),
    TypeReflection.ScalarType.uint8: (Format.r8_unorm, Format.rg8_unorm, None, Format.rgba8_unorm),
    TypeReflection.ScalarType.int16: (Format.r16_snorm, Format.rg16_snorm, None, Format.rgba16_snorm),
    TypeReflection.ScalarType.int8: (Format.r8_snorm, Format.rg8_snorm, None, Format.rgba8_snorm),
}


def has_uav(usage: TextureUsage):
    return (usage & TextureUsage.unordered_access.value) != 0


def prefix(usage: TextureUsage):
    return "RW" if has_uav(usage) else ""


class TextureMarshall(NativeTextureMarshall):

    def __init__(self, layout: refl.SlangProgramLayout, resource_shape: TypeReflection.ResourceShape, element_type: refl.SlangType, format: Format, usage: TextureUsage):
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

        self._base_texture_type_name = tex_type

        st = layout.find_type_by_name(self.build_type_name(usage, element_type))
        assert st is not None

        # tell type system slang types are Python types, not just native
        self.slang_type: 'refl.SlangType'
        self.slang_element_type: 'refl.SlangType'

        super().__init__(st, element_type, resource_shape, format, usage, tex_dims)

    def reduce_type(self, context: BindContext, dimensions: int):
        return super().reduce_type(context, dimensions)

    def resolve_type(self, context: BindContext, bound_type: refl.SlangType):
        # Handle being passed to a texture
        if isinstance(bound_type, refl.TextureType):
            if self.usage & bound_type.usage == 0:
                raise ValueError(
                    f"Cannot bind texture view {self.slang_type.name} with usage {bound_type.usage}")
            if self.resource_shape != bound_type.resource_shape:
                raise ValueError(
                    f"Cannot bind texture view {self.slang_type.name} with different shape {bound_type.resource_shape}")
            # TODO: Check element types match
            # if self.element_type.name != bound_type.element_type.name:
            #    raise ValueError(
            #        f"Cannot bind texture view {self.name} with different element type {bound_type.element_type.name}")
            return bound_type

        # If implicit element casts enabled, allow conversion from type to element type
        if context.options['implicit_element_casts']:
            if self.slang_element_type == bound_type:
                return bound_type

        # Otherwise, use default behaviour from marshall
        return super().resolve_type(context, bound_type)

    # Texture is writable if it has unordered access view.
    @property
    def is_writable(self):
        return has_uav(self.usage)

    # Generate the slang type name (eg Texture2D<float4>).
    def build_type_name(self, usage: TextureUsage, el_type: refl.SlangType):
        return f"{prefix(usage)}{self._base_texture_type_name}<{el_type.full_name}>"

    # Generate the slangpy accessor type name (eg Texture2DType<float4>).
    def build_accessor_name(self, usage: TextureUsage, el_type: refl.SlangType):
        return f"{prefix(usage)}{self._base_texture_type_name}Type<{self.slang_element_type.full_name}>"

    # Call data can only be read access to primal, and simply declares it as a variable.
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        access = binding.access[0]
        name = binding.variable_name

        if access == AccessType.none:
            cgb.type_alias(f"_t_{name}", f"NoneType")
            return

        if binding.call_dimensionality == 0:
            # If broadcast directly, function is just taking the texture argument directly, so use the slang type
            assert access == AccessType.read
            assert isinstance(binding.vector_type, refl.TextureType)
            if self.usage & binding.vector_type.usage == 0:
                raise ValueError(
                    f"Cannot bind texture view {name} with usage {binding.vector_type.usage}")
            cgb.type_alias(
                f"_t_{name}", binding.vector_type.full_name.replace("<", "Type<", 1))
        elif binding.call_dimensionality == self.texture_dims:
            # If broadcast is the same shape as the texture, this is loading from pixels, so use the
            # type required to support the required access
            if access == AccessType.read:
                # Read access can be either shader resource or UAV, so just bind the correct type
                # for this resource view
                cgb.type_alias(
                    f"_t_{name}", self.build_accessor_name(self.usage, self.slang_element_type))
            else:
                # Write access requires a UAV so check it and bind RW type
                if not has_uav(self.usage):
                    raise ValueError(
                        f"Cannot write to read-only texture {name}")
                cgb.type_alias(
                    f"_t_{name}", self.build_accessor_name(TextureUsage.unordered_access, self.slang_element_type))
        else:
            raise ValueError(
                f"Texture {name} has invalid dimensionality {binding.call_dimensionality}")


TYPE_TO_RESOURCE = {
    TextureType.texture_1d: TypeReflection.ResourceShape.texture_1d,
    TextureType.texture_2d: TypeReflection.ResourceShape.texture_2d,
    TextureType.texture_2d_ms: TypeReflection.ResourceShape.texture_2d_multisample,
    TextureType.texture_3d: TypeReflection.ResourceShape.texture_3d,
    TextureType.texture_cube: TypeReflection.ResourceShape.texture_cube,
    TextureType.texture_1d_array: TypeReflection.ResourceShape.texture_1d_array,
    TextureType.texture_2d_array: TypeReflection.ResourceShape.texture_2d_array,
    TextureType.texture_2d_ms_array: TypeReflection.ResourceShape.texture_2d_multisample_array,
}


def get_or_create_python_texture_type(layout: refl.SlangProgramLayout, format: Format, type: TextureType, usage: TextureUsage, array_size: int, sample_count: int):

    # Translate format into slang scalar type + channel count, which allows
    # us to build the element type of the texture.
    fmt_info = get_format_info(format)
    if fmt_info.type in [FormatType.float, FormatType.unorm, FormatType.snorm, FormatType.unorm_srgb]:
        scalar_type = TypeReflection.ScalarType.float32
    elif fmt_info.type == FormatType.uint:
        scalar_type = TypeReflection.ScalarType.uint32
    elif fmt_info.type == FormatType.sint:
        scalar_type = TypeReflection.ScalarType.int32
    else:
        raise ValueError(f"Unsupported format {format}")
    element_type = layout.vector_type(scalar_type, fmt_info.channel_count)

    # Translate resource type + array size into a slang resource shape.
    resource_shape = TYPE_TO_RESOURCE[type]

    return TextureMarshall(layout, resource_shape, element_type, format, usage)


def slang_type_to_texture_format(st: refl.SlangType) -> Optional[Format]:
    if isinstance(st, refl.VectorType) or isinstance(st, refl.ArrayType):
        assert st.element_type

        num_channels = st.shape[0]
        channel_type = st.element_type
    else:
        num_channels = 1
        channel_type = st

    if not isinstance(channel_type, refl.ScalarType):
        return None
    if num_channels == 0 or num_channels > 4:
        return None

    scalar = channel_type.slang_scalar_type
    if scalar not in SCALARTYPE_TO_TEXTURE_FORMAT:
        return None

    return SCALARTYPE_TO_TEXTURE_FORMAT[scalar][num_channels - 1]


def _get_or_create_python_type(layout: refl.SlangProgramLayout, value: Any):
    if isinstance(value, Texture):
        desc = value.desc
        return get_or_create_python_texture_type(layout,
                                                 desc.format,
                                                 desc.type,
                                                 desc.usage,
                                                 value.array_length,
                                                 desc.sample_count)
    elif isinstance(value, ReturnContext):
        format = slang_type_to_texture_format(value.slang_type)
        if format is None:
            raise ValueError(f"Can't create output texture: Slang type "
                             f"\"{value.slang_type.full_name}\" can't be used as a texel type")

        dim = value.bind_context.call_dimensionality
        if dim == 1:
            tex_type = TextureType.texture_1d
        elif dim == 2:
            tex_type = TextureType.texture_2d
        elif dim == 3:
            tex_type = TextureType.texture_3d
        else:
            raise ValueError(
                "Can't create output texture: Call dimensionality has to be 1D, 2D or 3D")

        usage = TextureUsage.unordered_access | TextureUsage.shader_resource

        return get_or_create_python_texture_type(layout, format, tex_type, usage, 1, 1)
    else:
        raise ValueError(f"Type {type(value)} is unsupported for TextureMarshall")


PYTHON_TYPES[Texture] = _get_or_create_python_type
PYTHON_SIGNATURES[Texture] = lambda x: f"[texture,{x.desc.type},{x.desc.usage},{x.desc.format}]"


class SamplerMarshall(Marshall):

    def __init__(self, layout: refl.SlangProgramLayout):
        super().__init__(layout)
        st = layout.find_type_by_name("SamplerState")
        if st is None:
            raise ValueError(
                f"Could not find Sampler slang type. This usually indicates the slangpy module has not been imported.")
        self.slang_type = st
        self.concrete_shape = Shape()

    # Call data can only be read access to primal, and simply declares it as a variable
    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: 'BoundVariable'):
        name = binding.variable_name
        assert isinstance(binding.vector_type, refl.SamplerStateType)
        cgb.type_alias(f"_t_{name}", f"SamplerStateType")

    # Call data just returns the primal
    def create_calldata(self, context: CallContext, binding: 'BoundVariableRuntime', data: Any) -> Any:
        access = binding.access
        if access[0] != AccessType.none:
            return {
                'value': data
            }

    # Buffers just return themselves for raw dispatch
    def create_dispatchdata(self, data: Any) -> Any:
        return data


def _get_or_create_sampler_python_type(layout: refl.SlangProgramLayout, value: Sampler):
    assert isinstance(value, Sampler)
    return SamplerMarshall(layout)


PYTHON_TYPES[Sampler] = _get_or_create_sampler_python_type
