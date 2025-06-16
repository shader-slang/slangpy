// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/device.h"

#include "utils/slangpyresources.h"

namespace sgl {

} // namespace sgl

namespace sgl::slangpy {

Shape NativeBufferMarshall::get_shape(nb::object data) const
{
    const Buffer* buffer;
    if (nb::try_cast<const Buffer*>(data, buffer)) {
        if (buffer->desc().struct_size != 0) {
            std::vector<int> shape = {int(buffer->desc().size / buffer->desc().struct_size)};
            return Shape(shape);
        }
    }
    return Shape({-1});
}

void NativeBufferMarshall::write_shader_cursor_pre_dispatch(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderCursor cursor,
    nb::object value,
    nb::list read_back
) const
{
    SGL_UNUSED(read_back);
    SGL_UNUSED(context);

    AccessType primal_access = binding->get_access().first;
    if (primal_access != AccessType::none) {
        SGL_UNUSED(binding);
        SGL_UNUSED(context);
        ShaderCursor field = cursor[binding->get_variable_name()]["value"];
        ref<BufferView> view;
        if (nb::try_cast(value, view)) {
            field.set_buffer_view(view);
        } else {
            field.set_buffer(nb::cast<ref<Buffer>>(value));
        }
    }
}

void NativeTextureMarshall::write_shader_cursor_pre_dispatch(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderCursor cursor,
    nb::object value,
    nb::list read_back
) const
{
    SGL_UNUSED(context);
    SGL_UNUSED(read_back);
    AccessType primal_access = binding->get_access().first;
    if (primal_access != AccessType::none) {

        ShaderCursor field = cursor[binding->get_variable_name()]["value"];
        ref<TextureView> view;
        if (nb::try_cast(value, view)) {
            field.set_texture_view(view);
        } else {
            field.set_texture(nb::cast<ref<Texture>>(value));
        }
    }
}

Shape NativeTextureMarshall::get_shape(nb::object value) const
{
    int mip;
    const Texture* texture;

    // Get texture pointer and mip level.
    TextureView* view;
    if (value.is_none()) {
        texture = nullptr;
        mip = 0;
    } else if (nb::try_cast(value, view)) {
        texture = view->texture();
        SGL_CHECK(texture, "NativeTextureMarshall ResourceView must point at a texture");
        mip = view->subresource_range().mip;
    } else {
        texture = nb::cast<Texture*>(value);
        mip = 0;
    }

    // If we have an actual texture, compute its shape. Otherwise
    // return -1 for each texture dimension.
    if (texture) {
        Shape res = get_texture_shape(texture, mip);
        SGL_CHECK(res.size() == m_texture_dims, "Texture dimensions are incorrect");
        return res + m_slang_element_type->get_shape();
    } else {
        std::vector<int> negativeDims(m_texture_dims, -1);
        Shape unknown(negativeDims); // or use a loop
        return unknown + m_slang_element_type->get_shape();
    }
}

Shape NativeTextureMarshall::get_texture_shape(const Texture* texture, int mip) const
{
    switch (m_resource_shape) {
    case TypeReflection::ResourceShape::texture_1d:
        return Shape({(int)texture->width() >> mip});

    case TypeReflection::ResourceShape::texture_2d:
    case TypeReflection::ResourceShape::texture_2d_multisample:
        return Shape({(int)texture->height() >> mip, (int)texture->width() >> mip});

    case TypeReflection::ResourceShape::texture_3d:
        return Shape({(int)texture->depth() >> mip, (int)texture->height() >> mip, (int)texture->width() >> mip});

    case TypeReflection::ResourceShape::texture_cube:
        return Shape({6, (int)texture->height() >> mip, (int)texture->width() >> mip});

    case TypeReflection::ResourceShape::texture_1d_array:
        return Shape({(int)texture->array_length(), (int)texture->width() >> mip});

    case TypeReflection::ResourceShape::texture_2d_array:
    case TypeReflection::ResourceShape::texture_2d_multisample_array:
        return Shape({(int)texture->array_length(), (int)texture->height() >> mip, (int)texture->width() >> mip});

    case TypeReflection::ResourceShape::texture_cube_array:
        return Shape({(int)texture->array_length(), 6, (int)texture->height() >> mip, (int)texture->width() >> mip});

    default:
        SGL_THROW("Unsupported resource shape: {}", m_resource_shape);
    }
}

nb::object NativeTextureMarshall::create_output(CallContext* context, NativeBoundVariableRuntime* binding) const
{
    size_t dims = context->call_shape().size();
    SGL_CHECK(dims > 0 && dims <= 3, "Invalid call shape (must be 1D, 2D or 3D) for texture output");

    TextureType type = dims == 1 ? TextureType::texture_1d
        : dims == 2              ? TextureType::texture_2d
                                 : TextureType::texture_3d;

    SGL_UNUSED(binding);
    TextureDesc desc;
    desc.format = m_format;
    desc.usage = m_usage;
    desc.type = type;

    size_t shape_end = context->call_shape().size() - 1;
    desc.width = context->call_shape()[shape_end];
    if (dims > 1)
        desc.height = context->call_shape()[shape_end - 1];
    if (dims > 2)
        desc.depth = context->call_shape()[shape_end - 2];
    auto texture = context->device()->create_texture(desc);

    return nb::cast(texture);
}

nb::object
NativeTextureMarshall::read_output(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data) const
{
    SGL_UNUSED(context);
    SGL_UNUSED(binding);
    return data;
}

Shape get_texture_shape(const Texture* texture, int mip)
{
    switch (texture->type()) {
    case TextureType::texture_1d:
        return Shape({(int)texture->width() >> mip});
    case TextureType::texture_1d_array:
        return Shape({(int)texture->array_length(), (int)texture->width() >> mip});
    case TextureType::texture_2d:
    case TextureType::texture_2d_ms:
        return Shape({(int)texture->height() >> mip, (int)texture->width() >> mip});
    case TextureType::texture_2d_array:
    case TextureType::texture_2d_ms_array:
        return Shape({(int)texture->array_length(), (int)texture->height() >> mip, (int)texture->width() >> mip});
    case TextureType::texture_3d:
        return Shape({(int)texture->depth() >> mip, (int)texture->height() >> mip, (int)texture->width() >> mip});
    case TextureType::texture_cube:
        return Shape({6, (int)texture->height() >> mip, (int)texture->width() >> mip});
    case TextureType::texture_cube_array:
        return Shape({(int)texture->array_length(), 6, (int)texture->height() >> mip, (int)texture->width() >> mip});
    default:
        SGL_THROW("Invalid texture shape: {}", texture->type());
    }
}

} // namespace sgl::slangpy

SGL_PY_EXPORT(utils_slangpy_resources)
{
    using namespace sgl;
    using namespace sgl::slangpy;

    nb::module_ slangpy = m.attr("slangpy");

    slangpy.def("get_texture_shape", &get_texture_shape, "texture"_a, "mip"_a = 0, D_NA(get_texture_shape()));

    nb::class_<NativeBufferMarshall, NativeMarshall>(slangpy, "NativeBufferMarshall") //
        .def(
            "__init__",
            [](NativeBufferMarshall& self, ref<NativeSlangType> slang_type, BufferUsage usage)
            { new (&self) NativeBufferMarshall(slang_type, usage); },
            "slang_type"_a,
            "usage"_a,
            D_NA(NativeBufferMarshall, NativeBufferMarshall)
        )
        .def(
            "write_shader_cursor_pre_dispatch",
            &NativeBufferMarshall::write_shader_cursor_pre_dispatch,
            "context"_a,
            "binding"_a,
            "cursor"_a,
            "value"_a,
            "read_back"_a,
            D_NA(NativeBufferMarshall, write_shader_cursor_pre_dispatch)
        )
        .def("get_shape", &NativeBufferMarshall::get_shape, "value"_a, D_NA(NativeBufferMarshall, get_shape))
        .def_prop_ro("usage", &sgl::slangpy::NativeBufferMarshall::usage)
        .def_prop_ro("slang_type", &sgl::slangpy::NativeBufferMarshall::get_slang_type);

    nb::class_<NativeTextureMarshall, NativeMarshall>(slangpy, "NativeTextureMarshall") //
        .def(
            "__init__",
            [](NativeTextureMarshall& self,
               ref<NativeSlangType> slang_type,
               ref<NativeSlangType> element_type,
               TypeReflection::ResourceShape resource_shape,
               Format format,
               TextureUsage usage,
               int dims)
            { new (&self) NativeTextureMarshall(slang_type, element_type, resource_shape, format, usage, dims); },
            "slang_type"_a,
            "element_type"_a,
            "resource_shape"_a,
            "format"_a,
            "usage"_a,
            "dims"_a,
            D_NA(NativeTextureMarshall, NativeTextureMarshall)
        )
        .def(
            "write_shader_cursor_pre_dispatch",
            &NativeTextureMarshall::write_shader_cursor_pre_dispatch,
            "context"_a,
            "binding"_a,
            "cursor"_a,
            "value"_a,
            "read_back"_a,
            D_NA(NativeTextureMarshall, write_shader_cursor_pre_dispatch)
        )
        .def("get_shape", &NativeTextureMarshall::get_shape, "value"_a, D_NA(NativeTextureMarshall, get_shape))
        .def(
            "get_texture_shape",
            &NativeTextureMarshall::get_texture_shape,
            "texture"_a,
            "mip"_a,
            D_NA(NativeTextureMarshall, get_texture_shape)
        )
        .def_prop_ro(
            "resource_shape",
            &NativeTextureMarshall::resource_shape,
            D_NA(NativeTextureMarshall, resource_shape)
        )
        .def_prop_ro("usage", &NativeTextureMarshall::usage, D_NA(NativeTextureMarshall, usage))
        .def_prop_ro("texture_dims", &NativeTextureMarshall::texture_dims, D_NA(NativeTextureMarshall, texture_dims))
        .def_prop_ro(
            "slang_element_type",
            &NativeTextureMarshall::slang_element_type,
            D_NA(NativeTextureMarshall, slang_element_type)
        );
}
