// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <vector>
#include <map>

#include "nanobind.h"

#include "sgl/core/macros.h"
#include "sgl/core/fwd.h"
#include "sgl/core/object.h"

#include "sgl/device/fwd.h"
#include "sgl/device/resource.h"

#include "utils/slangpy.h"

namespace sgl::slangpy {

class NativeBufferMarshall : public NativeMarshall {
public:
    NativeBufferMarshall(ref<NativeSlangType> slang_type, BufferUsage usage)
        : NativeMarshall(slang_type)
        , m_usage(usage)
    {
    }

    BufferUsage usage() { return m_usage; }

    void write_shader_cursor_pre_dispatch(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        ShaderCursor cursor,
        nb::object value,
        nb::list read_back
    ) const override;

    Shape get_shape(nb::object data) const override;

    nb::object create_dispatchdata(nb::object data) const override { return data; }

private:
    BufferUsage m_usage;
};

class NativeTextureMarshall : public NativeMarshall {
public:
    NativeTextureMarshall(
        ref<NativeSlangType> slang_type,
        ref<NativeSlangType> element_type,
        TypeReflection::ResourceShape resource_shape,
        Format format,
        TextureUsage usage,
        int dims
    )
        : NativeMarshall(std::move(slang_type))
        , m_resource_shape(resource_shape)
        , m_format(format)
        , m_usage(usage)
        , m_texture_dims(dims)
        , m_slang_element_type(std::move(element_type))
    {
    }

    TypeReflection::ResourceShape resource_shape() const { return m_resource_shape; }
    TextureUsage usage() const { return m_usage; }
    int texture_dims() const { return m_texture_dims; }
    ref<NativeSlangType> slang_element_type() const { return m_slang_element_type; }

    void write_shader_cursor_pre_dispatch(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        ShaderCursor cursor,
        nb::object value,
        nb::list read_back
    ) const override;

    Shape get_shape(nb::object value) const override;

    Shape get_texture_shape(const Texture* texture, int mip) const;

    nb::object create_dispatchdata(nb::object data) const override { return data; }

    nb::object create_output(CallContext* context, NativeBoundVariableRuntime* binding) const override;

    nb::object read_output(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data) const override;

private:
    TypeReflection::ResourceShape m_resource_shape;
    Format m_format;
    TextureUsage m_usage;
    int m_texture_dims;
    ref<NativeSlangType> m_slang_element_type;
};

class NativeDescriptorMarshall : public NativeMarshall {
public:
    NativeDescriptorMarshall(ref<NativeSlangType> slang_type, DescriptorHandleType type)
        : NativeMarshall(slang_type)
        , m_type(type)
    {
    }

    DescriptorHandleType type() { return m_type; }

    void write_shader_cursor_pre_dispatch(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        ShaderCursor cursor,
        nb::object value,
        nb::list read_back
    ) const override;

    Shape get_shape(nb::object data) const override;

    nb::object create_dispatchdata(nb::object data) const override { return data; }

private:
    DescriptorHandleType m_type;
};


} // namespace sgl::slangpy
