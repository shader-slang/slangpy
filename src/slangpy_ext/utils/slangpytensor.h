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

class NativeTensor;

struct NativeTensorDesc : public StridedBufferViewDesc { };

class NativeTensor : public StridedBufferView {
public:
    NativeTensor(
        NativeTensorDesc desc,
        const ref<Buffer>& storage,
        const ref<NativeTensor>& grad_in,
        const ref<NativeTensor>& grad_out
    );
    virtual ~NativeTensor() { }

    virtual NativeTensorDesc& desc() override { return m_desc; }
    virtual const NativeTensorDesc& desc() const override { return m_desc; }

    ref<NativeTensor> view(Shape shape, Shape strides = Shape(), int offset = 0) const;
    ref<NativeTensor> broadcast_to(const Shape& shape) const;
    ref<NativeTensor> index(nb::args args) const;

    const ref<NativeTensor>& grad_in() const { return m_grad_in; }
    void set_grad_in(const ref<NativeTensor>& grad_in) { m_grad_in = grad_in; }

    const ref<NativeTensor>& grad_out() const { return m_grad_out; }
    void set_grad_out(const ref<NativeTensor>& grad_out) { m_grad_out = grad_out; }

    /// Helper that gets/validates the output grad.
    ref<NativeTensor> grad() const
    {
        SGL_CHECK(m_grad_out, "Tensor has no grad.");
        return m_grad_out;
    }

    /// Create a new version of this tensor with associated grads. It is valid for
    /// both input and output grads to refer to the same tensor. If neither grad_in
    /// or grad_out are provided, a single new tensor is created and used for both grads.
    ref<NativeTensor>
    with_grads(ref<NativeTensor> grad_in = nullptr, ref<NativeTensor> grad_out = nullptr, bool zero = false) const;

    /// Create a new version of this tensor without grads that refers to the same storage.
    ref<NativeTensor> detach() const;

private:
    NativeTensorDesc m_desc;
    ref<NativeTensor> m_grad_in;
    ref<NativeTensor> m_grad_out;
};


class NativeTensorMarshall : public NativeMarshall {
public:
    NativeTensorMarshall(
        int dims,
        bool writable,
        ref<NativeSlangType> slang_type,
        ref<NativeSlangType> slang_element_type,
        ref<TypeLayoutReflection> element_layout,
        ref<NativeTensorMarshall> d_in,
        ref<NativeTensorMarshall> d_out
    )
        : NativeMarshall(slang_type)
        , m_dims(dims)
        , m_writable(writable)
        , m_slang_element_type(slang_element_type)
        , m_element_layout(element_layout)
        , m_d_in(d_in)
        , m_d_out(d_out)
    {
    }

    int dims() const { return m_dims; }
    bool writable() const { return m_writable; }
    ref<NativeSlangType> slang_element_type() const { return m_slang_element_type; }
    ref<TypeLayoutReflection> element_layout() const { return m_element_layout; }
    size_t element_stride() const { return m_element_layout->stride(); }
    bool has_derivative() const { return m_d_in != nullptr || m_d_out != nullptr; }
    ref<NativeTensorMarshall> d_in() const { return m_d_in; }
    ref<NativeTensorMarshall> d_out() const { return m_d_out; }

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

private:
    int m_dims;
    bool m_writable;
    ref<NativeSlangType> m_slang_element_type;
    ref<TypeLayoutReflection> m_element_layout;
    ref<NativeTensorMarshall> m_d_in;
    ref<NativeTensorMarshall> m_d_out;

    void write_shader_cursor_fields(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        ShaderCursor field,
        NativeTensor* value,
        nb::list read_back
    ) const;
};

/// Bare minimum overridable functions to allow python marshall
/// extensions to utilize the majority of native functionality.
struct PyNativeTensorMarshall : public NativeTensorMarshall {
    NB_TRAMPOLINE(NativeTensorMarshall, 5);

    Shape get_shape(nb::object data) const override { NB_OVERRIDE(get_shape, data); }

    nb::object
    create_calldata(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data) const override
    {
        NB_OVERRIDE(create_calldata, context, binding, data);
    }

    void read_calldata(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data, nb::object result)
        const override
    {
        NB_OVERRIDE(read_calldata, context, binding, data, result);
    }


    nb::object create_output(CallContext* context, NativeBoundVariableRuntime* binding) const override
    {
        NB_OVERRIDE(create_output, context, binding);
    }
    nb::object read_output(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data) const override
    {
        NB_OVERRIDE(read_output, context, binding, data);
    }
};

} // namespace sgl::slangpy
