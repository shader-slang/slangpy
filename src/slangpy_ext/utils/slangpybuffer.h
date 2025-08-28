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

#include "slangpystridedbufferview.h"

namespace sgl::slangpy {

struct NativeNDBufferDesc : public StridedBufferViewDesc { };

class NativeNDBuffer : public StridedBufferView {
public:
    NativeNDBuffer(Device* device, NativeNDBufferDesc desc, ref<Buffer> storage = nullptr);
    virtual ~NativeNDBuffer() { }

    virtual NativeNDBufferDesc& desc() override { return m_desc; }
    virtual const NativeNDBufferDesc& desc() const override { return m_desc; }

    ref<NativeNDBuffer> view(Shape shape, Shape strides = Shape(), int offset = 0) const;
    ref<NativeNDBuffer> broadcast_to(const Shape& shape) const;
    ref<NativeNDBuffer> index(nb::object index_arg) const;

    /// Get string representation of the NDBuffer.
    std::string to_string() const;

private:
    NativeNDBufferDesc m_desc;
};


class NativeNDBufferMarshall : public NativeMarshall {
public:
    NativeNDBufferMarshall(
        int dims,
        bool writable,
        ref<NativeSlangType> slang_type,
        ref<NativeSlangType> slang_element_type,
        ref<TypeLayoutReflection> element_layout
    )
        : NativeMarshall(slang_type)
        , m_dims(dims)
        , m_writable(writable)
        , m_slang_element_type(slang_element_type)
        , m_element_layout(element_layout)
    {
    }

    int dims() const { return m_dims; }
    bool writable() const { return m_writable; }
    ref<NativeSlangType> slang_element_type() const { return m_slang_element_type; }
    ref<TypeLayoutReflection> element_layout() const { return m_element_layout; }
    size_t element_stride() const { return m_element_layout->stride(); }

    Shape get_shape(nb::object data) const override;

    void write_shader_cursor_pre_dispatch(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        ShaderCursor cursor,
        nb::object value,
        nb::list read_back
    ) const override;

    void read_calldata(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data, nb::object result)
        const override;

    nb::object create_output(CallContext* context, NativeBoundVariableRuntime* binding) const override;

    nb::object create_dispatchdata(nb::object data) const override;

    nb::object read_output(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data) const override;

protected:
    ref<NativeNDBuffer> create_buffer(Device* device, const Shape& shape) const;

private:
    int m_dims;
    bool m_writable;
    ref<NativeSlangType> m_slang_element_type;
    ref<TypeLayoutReflection> m_element_layout;
};

class NativeNumpyMarshall : public NativeNDBufferMarshall {
public:
    NativeNumpyMarshall(
        int dims,
        ref<NativeSlangType> slang_type,
        ref<NativeSlangType> slang_element_type,
        ref<TypeLayoutReflection> element_layout,
        nb::dlpack::dtype dtype
    )
        : NativeNDBufferMarshall(dims, true, slang_type, slang_element_type, element_layout)
        , m_dtype(dtype)
    {
    }

    nb::dlpack::dtype dtype() const { return m_dtype; }

    Shape get_shape(nb::object data) const override;

    void write_shader_cursor_pre_dispatch(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        ShaderCursor cursor,
        nb::object value,
        nb::list read_back
    ) const override;

    void read_calldata(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data, nb::object result)
        const override;

    nb::object create_output(CallContext* context, NativeBoundVariableRuntime* binding) const override;

    nb::object create_dispatchdata(nb::object data) const override;

private:
    nb::dlpack::dtype m_dtype;
};

} // namespace sgl::slangpy
