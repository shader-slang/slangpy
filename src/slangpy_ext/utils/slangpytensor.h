// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <vector>
#include <map>
#include <optional>

#include "nanobind.h"

#include "sgl/core/macros.h"
#include "sgl/core/fwd.h"
#include "sgl/core/object.h"

#include "sgl/device/fwd.h"
#include "sgl/device/resource.h"
#include "sgl/device/shader_offset.h"
#include "sgl/func/tensor.h"

#include "utils/slangpy.h"

namespace sgl::slangpy {

using Tensor = func::Tensor;
using TensorDesc = func::TensorDesc;
using TensorViewData = func::TensorViewData;
using DiffTensorViewData = func::DiffTensorViewData;

static constexpr int kSlangPyTensorViewMaxDim = func::kSlangPyTensorViewMaxDim;

nb::dict tensor_uniforms(const Tensor& tensor);


class TensorMarshall : public NativeMarshall {
public:
    TensorMarshall(
        int dims,
        bool writable,
        ref<refl::Type> slang_type,
        ref<refl::Type> slang_element_type,
        ref<TypeLayoutReflection> element_layout,
        ref<TensorMarshall> d_in,
        ref<TensorMarshall> d_out
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
    ref<refl::Type> slang_element_type() const { return m_slang_element_type; }
    ref<TypeLayoutReflection> element_layout() const { return m_element_layout; }
    size_t element_stride() const { return m_element_layout->stride(); }
    bool has_derivative() const { return m_d_in != nullptr || m_d_out != nullptr; }
    ref<TensorMarshall> d_in() const { return m_d_in; }
    ref<TensorMarshall> d_out() const { return m_d_out; }

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
        ShaderOffset tensorview_offset;   // Base offset for TensorView (used with set_data)
        bool is_valid = false;            // Whether offsets have been initialized
        bool is_tensorview = false;
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

        // Whether to copy interop buffers back to torch tensors after dispatch.
        // Only used by NativeTorchTensorMarshall; computed in ensure_binding_info_cached()
        // from the Slang uniform type name (Tensor/WTensor/RWTensor/DiffTensor/etc.).
        bool needs_primal_copyback = false;
        bool needs_grad_copyback = false;
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
    ref<refl::Type> m_slang_element_type;
    ref<TypeLayoutReflection> m_element_layout;
    ref<TensorMarshall> m_d_in;
    ref<TensorMarshall> m_d_out;
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
        Tensor* primal_tensor,
        nb::list read_back
    ) const;

    //
    // Core Field Writing Methods (Fast Path)
    //

    /// Write Tensor fields using pre-cached offsets
    /// Uses direct memory writes with pre-computed offsets for maximum performance
    /// Write Tensor fields using pre-cached offsets
    /// Uses direct memory writes with pre-computed offsets for maximum performance
    void write_native_tensor_fields(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        ShaderObject* shader_object,
        void* base_address,
        const TensorFieldOffsets& offsets,
        Tensor* buffer,
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
struct PyTensorMarshall : public TensorMarshall {
    NB_TRAMPOLINE(TensorMarshall, 5);

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

class NativeNumpyMarshall : public TensorMarshall {
public:
    NativeNumpyMarshall(
        int dims,
        ref<refl::Type> slang_type,
        ref<refl::Type> slang_element_type,
        ref<TypeLayoutReflection> element_layout,
        nb::dlpack::dtype dtype
    )
        : TensorMarshall(dims, true, slang_type, slang_element_type, element_layout, nullptr, nullptr)
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

    void read_calldata(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        nb::object data,
        nb::object result
    ) const override;

    nb::object create_output(CallContext* context, NativeBoundVariableRuntime* binding) const override;

    nb::object create_dispatchdata(nb::object data) const override;

private:
    ref<Tensor> create_tensor(Device* device, const Shape& shape) const;

    nb::dlpack::dtype m_dtype;
};

} // namespace sgl::slangpy
