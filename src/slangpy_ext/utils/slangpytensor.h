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
#include "sgl/device/shader_offset.h"

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
    ref<NativeTensor> index(nb::object index_arg) const;

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
    with_grads(ref<NativeTensor> grad_in = nullptr, ref<NativeTensor> grad_out = nullptr, bool zero = true) const;

    /// Create a new version of this tensor without grads that refers to the same storage.
    ref<NativeTensor> detach() const;

    /// Get string representation of the tensor.
    std::string to_string() const override;

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

    void read_calldata(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        nb::object data,
        nb::object result
    ) const override;

    nb::object create_output(CallContext* context, NativeBoundVariableRuntime* binding) const override;

    nb::object create_dispatchdata(nb::object data) const override;

    nb::object read_output(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data) const override;

    /// Cached shader offsets for a single tensor's fields
    /// Public so NativeTorchTensorMarshall can reuse them
    struct TensorFieldOffsets {
        int array_stride;
        ShaderOffset data;                // Offset for _data field
        ShaderOffset shape;               // Offset for _shape field
        ShaderOffset strides;             // Offset for _strides field
        ShaderOffset offset;              // Offset for _offset field
        ShaderOffset element_byte_stride; // Offset for _element_byte_stride field (if present)
        bool is_valid = false;            // Whether offsets have been initialized
    };

    /// Cached binding info for all tensor variants (primal, grad_in, grad_out)
    /// Contains shader offsets plus copy-back decision flags.
    /// Public so NativeTorchTensorMarshall can reuse this structure.
    struct CachedBindingInfo {
        TensorFieldOffsets primal;    // Offsets for primal tensor fields
        TensorFieldOffsets grad_in;   // Offsets for gradient input fields (if present)
        TensorFieldOffsets grad_out;  // Offsets for gradient output fields (if present)
        bool has_grad_fields = false; // Whether tensor uses _primal wrapper (differentiated mode)
        ShaderOffset field_offset;    // Base offset of the entire field structure
        uint32_t field_size = 0;      // Total size of the field in uniform data

        // Whether to copy interop buffer back to torch tensor after dispatch.
        // Only used by NativeTorchTensorMarshall; see ensure_binding_info_cached() for details.
        // Gradient copy-back uses runtime check instead (has_derivative() is not final at cache time).
        bool needs_primal_copyback = false;
    };

    /// Extract TensorFieldOffsets from a ShaderCursor pointing to a tensor structure
    /// Public so NativeTorchTensorMarshall can reuse it
    static TensorFieldOffsets extract_tensor_field_offsets(ShaderCursor tensor_cursor);

    /// Extract all cached binding info (primal, grad_in, grad_out) from a field cursor
    /// Public so NativeTorchTensorMarshall can reuse it
    static CachedBindingInfo extract_binding_info(ShaderCursor cursor);

private:
    int m_dims;
    bool m_writable;
    ref<NativeSlangType> m_slang_element_type;
    ref<TypeLayoutReflection> m_element_layout;
    ref<NativeTensorMarshall> m_d_in;
    ref<NativeTensorMarshall> m_d_out;
    mutable CachedBindingInfo m_cached_binding_info;

    /// Initialize cached binding info if not already done
    /// This method is called on the first dispatch to cache reflection data for subsequent calls
    void ensure_binding_info_cached(ShaderCursor cursor, NativeBoundVariableRuntime* binding) const;

    //
    // High-Level Write Methods
    //

    /// Write differentiated tensor structure (handles primal, grad_in, grad_out)
    /// This method handles both flat and differentiated tensor layouts
    void write_native_tensor(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        ShaderObject* shader_object,
        void* base_address,
        NativeTensor* primal_tensor,
        nb::list read_back
    ) const;

    //
    // Core Field Writing Methods (Fast Path)
    //

    /// Write NativeTensor fields using pre-cached offsets
    /// Uses direct memory writes with pre-computed offsets for maximum performance
    /// Write NativeTensor fields using pre-cached offsets
    /// Uses direct memory writes with pre-computed offsets for maximum performance
    void write_native_tensor_fields(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        ShaderObject* shader_object,
        void* base_address,
        const TensorFieldOffsets& offsets,
        NativeTensor* buffer,
        nb::list read_back
    ) const;

    /// Write tensor fields using pre-cached offsets (Buffer version)
    /// For non-CUDA backends, binds the buffer; for CUDA, writes the device pointer
    void write_tensor_fields_from_buffer(
        ShaderObject* shader_object,
        void* base_address,
        const TensorFieldOffsets& offsets,
        const ref<Buffer>& buffer,
        const Shape& shape,
        const Shape& strides,
        int offset
    ) const;

    /// Write tensor fields using pre-cached offsets (Raw pointer version)
    /// Used for PyTorch tensors where we write the raw device pointer directly
    void write_tensor_fields_from_pointer(
        ShaderObject* shader_object,
        void* base_address,
        const TensorFieldOffsets& offsets,
        void* data_ptr,
        const Shape& shape,
        const Shape& strides,
        int offset
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

    void read_calldata(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        nb::object data,
        nb::object result
    ) const override
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
