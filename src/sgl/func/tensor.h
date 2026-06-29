// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/device/fwd.h"
#include "sgl/device/reflection.h"
#include "sgl/device/resource.h"
#include "sgl/refl/type.h"
#include "sgl/utils/slangpy.h"

#include <cstdint>
#include <optional>
#include <string>

namespace sgl {

class SignatureBuffer;

} // namespace sgl

namespace sgl::func {

/// Maximum dimensions for TensorView interop data.
static constexpr int kSlangPyTensorViewMaxDim = 5;

/// CPU representation of Slang's TensorView uniform payload.
struct TensorViewData {
    uint64_t data;
    uint32_t strides[kSlangPyTensorViewMaxDim];
    uint32_t sizes[kSlangPyTensorViewMaxDim];
    uint32_t dimensionCount;
};
static_assert(sizeof(TensorViewData) == 56, "TensorViewData must match Slang's TensorView layout.");

/// CPU representation of Slang's DiffTensorView uniform payload.
struct DiffTensorViewData {
    TensorViewData primal;
    TensorViewData diff;
};
static_assert(sizeof(DiffTensorViewData) == 112, "DiffTensorViewData must match Slang's DiffTensorView layout.");

/// Native Tensor descriptor shared by native code and the extension binding layer.
struct TensorDesc {
    ref<refl::Type> dtype;
    ref<TypeLayoutReflection> element_layout;
    int offset{0};
    slangpy::Shape shape;
    slangpy::Shape strides;
    BufferUsage usage{BufferUsage::shader_resource | BufferUsage::unordered_access};
    MemoryType memory_type{MemoryType::device_local};
};

/// Native tensor runtime object.
///
/// This class deliberately contains no language-binding dependency. Adapter
/// conveniences such as NumPy/Torch conversion and indexing syntax live in the
/// slangpy extension layer.
class SGL_API Tensor : public Object {
    SGL_OBJECT(Tensor)
public:
    Tensor(TensorDesc desc, ref<Buffer> storage, ref<Tensor> grad_in = nullptr, ref<Tensor> grad_out = nullptr);

    TensorDesc& desc() { return m_desc; }
    const TensorDesc& desc() const { return m_desc; }

    Device* device() const;
    const ref<refl::Type>& dtype() const { return m_desc.dtype; }
    int offset() const { return m_desc.offset; }
    const slangpy::Shape& shape() const { return m_desc.shape; }
    const slangpy::Shape& strides() const { return m_desc.strides; }
    int dims() const { return static_cast<int>(m_desc.shape.size()); }
    size_t element_count() const { return m_desc.shape.element_count(); }
    BufferUsage usage() const { return m_desc.usage; }
    MemoryType memory_type() const { return m_desc.memory_type; }
    const ref<Buffer>& storage() const { return m_storage; }
    size_t element_stride() const { return m_desc.element_layout->stride(); }

    /// Signature fragment used by SlangPy's native call-data cache.
    const std::string& signature() const { return m_signature; }

    /// Write the SlangPy cache signature used by functional dispatch.
    static void write_slangpy_signature(SignatureBuffer& signature, const Tensor* value);

    /// Write a tensor to a shader cursor.
    ///
    /// Plain Tensor/WTensor/RWTensor/PrimalTensor targets are handled by writing the
    /// cursor's tensor fields directly. Differentiable wrapper targets write the
    /// primal tensor plus any gradient fields that are present on the target type.
    static void write_to_cursor(const ShaderCursor& cursor, const Tensor* value);

    /// Write a tensor to buffer cursor storage for pointer-backed Tensor fields or TensorView payloads.
    ///
    /// Resource-backed tensor fields require a ShaderCursor and will throw when encountered.
    static void write_to_cursor(const BufferElementCursor& cursor, const Tensor* value);

    /// Build the POD payload used by Slang TensorView fields.
    static TensorViewData make_tensor_view_data(
        const ref<const Buffer>& storage,
        const slangpy::Shape& shape,
        const slangpy::Shape& strides,
        int offset,
        size_t element_stride
    );

    bool is_contiguous() const;
    ref<BufferCursor> cursor(std::optional<int> start = std::nullopt, std::optional<int> count = std::nullopt) const;

    void clear(CommandEncoder* cmd = nullptr);
    void point_to(ref<Tensor> target);

    ref<Tensor> view(slangpy::Shape shape, slangpy::Shape strides = slangpy::Shape(), int offset = 0) const;
    ref<Tensor> broadcast_to(const slangpy::Shape& shape) const;

    const ref<Tensor>& grad_in() const { return m_grad_in; }
    void set_grad_in(const ref<Tensor>& grad_in) { m_grad_in = grad_in; }

    const ref<Tensor>& grad_out() const { return m_grad_out; }
    void set_grad_out(const ref<Tensor>& grad_out) { m_grad_out = grad_out; }

    ref<Tensor> grad() const;
    ref<Tensor> with_grads(ref<Tensor> grad_in = nullptr, ref<Tensor> grad_out = nullptr, bool zero = true) const;
    ref<Tensor> detach() const;

    std::string to_string() const override;

private:
    void update_signature();
    void view_inplace(slangpy::Shape shape, slangpy::Shape strides, int offset);
    void broadcast_to_inplace(const slangpy::Shape& shape);

    TensorDesc m_desc;
    ref<Buffer> m_storage;
    ref<Tensor> m_grad_in;
    ref<Tensor> m_grad_out;
    std::string m_signature;
};

} // namespace sgl::func
