// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "nanobind.h"

#include "sgl/core/macros.h"
#include "sgl/core/object.h"

#include "sgl/device/fwd.h"
#include "sgl/device/shader_offset.h"

#include "utils/slangpy.h"
#include "utils/slangpytensor.h"
#include "utils/torch_bridge.h"

namespace sgl::slangpy {

/// Native marshall for torch.Tensor objects.
///
/// This class handles marshalling of raw PyTorch tensors (not wrapped in TensorRef)
/// to shader uniforms. It uses TorchBridge for fast tensor metadata extraction.
///
/// Key features:
/// - Native get_shape() using TorchBridge (~28ns vs ~350ns Python)
/// - Direct CUDA tensor pointer writing for CUDA devices
/// - Interop buffer handling for non-CUDA backends
///
/// This class shares the CachedOffsets and TensorFieldOffsets structures with
/// NativeTensorMarshall to ensure consistent shader data layout.
class NativeTorchTensorMarshall : public NativeMarshall {
public:
    /// Reuse the offset structures from NativeTensorMarshall
    using TensorFieldOffsets = NativeTensorMarshall::TensorFieldOffsets;
    using CachedOffsets = NativeTensorMarshall::CachedOffsets;

    NativeTorchTensorMarshall(
        int dims,
        bool writable,
        ref<NativeSlangType> slang_type,
        ref<NativeSlangType> slang_element_type,
        ref<TypeLayoutReflection> element_layout,
        ref<NativeTorchTensorMarshall> d_in,
        ref<NativeTorchTensorMarshall> d_out
    );

    virtual ~NativeTorchTensorMarshall() = default;

    // Accessors
    int dims() const { return m_dims; }
    bool writable() const { return m_writable; }
    ref<NativeSlangType> slang_element_type() const { return m_slang_element_type; }
    ref<TypeLayoutReflection> element_layout() const { return m_element_layout; }
    size_t element_stride() const { return m_element_layout->stride(); }
    bool has_derivative() const { return m_d_in != nullptr || m_d_out != nullptr; }
    ref<NativeTorchTensorMarshall> d_in() const { return m_d_in; }
    ref<NativeTorchTensorMarshall> d_out() const { return m_d_out; }

    /// Get shape from a torch.Tensor using TorchBridge (native, fast)
    Shape get_shape(nb::object data) const override;

    /// Write tensor data to shader cursor (main dispatch entry point)
    void write_shader_cursor_pre_dispatch(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        ShaderCursor cursor,
        nb::object value,
        nb::list read_back
    ) const override;

    /// Read data back after dispatch (for non-CUDA backends)
    void read_calldata(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        nb::object data,
        nb::object result
    ) const override;

    /// Create output tensor for return values
    nb::object create_output(CallContext* context, NativeBoundVariableRuntime* binding) const override;

    /// Read output tensor after dispatch
    nb::object read_output(CallContext* context, NativeBoundVariableRuntime* binding, nb::object data) const override;

    /// Create dispatch data dictionary (for Python fallback path)
    nb::object create_dispatchdata(nb::object data) const override;

private:
    int m_dims;
    bool m_writable;
    ref<NativeSlangType> m_slang_element_type;
    ref<TypeLayoutReflection> m_element_layout;
    ref<NativeTorchTensorMarshall> m_d_in;
    ref<NativeTorchTensorMarshall> m_d_out;
    mutable CachedOffsets m_cached_offsets;

    /// Initialize cached offsets if not already done
    void ensure_offsets_cached(ShaderCursor cursor, NativeBoundVariableRuntime* binding) const;

    /// Write torch tensor fields to shader uniforms
    /// If interop_buffer is provided, uses its device address instead of tensor's CUDA pointer
    void write_torch_tensor_fields(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        ShaderObject* shader_object,
        void* base_address,
        const TensorFieldOffsets& offsets,
        const TensorBridgeInfo& info,
        Buffer* interop_buffer
    ) const;

    /// Handle interop path for non-CUDA device backends (D3D12/Vulkan)
    void write_shader_cursor_with_interop(
        CallContext* context,
        NativeBoundVariableRuntime* binding,
        ShaderObject* shader_object,
        void* base_address,
        nb::object value,
        const TensorBridgeInfo& info,
        nb::list read_back
    ) const;
};

/// Python trampoline for virtual method overrides.
/// Bare minimum overridable functions to allow python marshall
/// extensions to utilize the majority of native functionality.
struct PyNativeTorchTensorMarshall : public NativeTorchTensorMarshall {
    NB_TRAMPOLINE(NativeTorchTensorMarshall, 5);

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
