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

struct StridedBufferViewDesc {
    ref<NativeSlangType> dtype;
    ref<TypeLayoutReflection> element_layout;
    int offset{0};
    Shape shape;
    Shape strides;
    BufferUsage usage{BufferUsage::shader_resource | BufferUsage::unordered_access};
    MemoryType memory_type{MemoryType::device_local};
};

class StridedBufferView : public NativeObject {
public:
    StridedBufferView(Device* device, const StridedBufferViewDesc& desc, ref<Buffer> storage);
    virtual ~StridedBufferView() { }

    // Derived classes are free to store a desc with extra information, and can provide us
    // with the desc for the buffer view by implementing these methods
    virtual StridedBufferViewDesc& desc() { SGL_THROW("desc() is not implemented"); }
    virtual const StridedBufferViewDesc& desc() const { SGL_THROW("desc() is not implemented"); }

    Device* device() const { return storage()->device(); }
    ref<NativeSlangType> dtype() const { return desc().dtype; }
    int offset() const { return desc().offset; }
    const Shape& shape() const { return desc().shape; }
    const Shape& strides() const { return desc().strides; }
    int dims() const { return (int)desc().shape.size(); }
    size_t element_count() const { return desc().shape.element_count(); }
    BufferUsage usage() const { return desc().usage; }
    MemoryType memory_type() const { return desc().memory_type; }
    ref<Buffer> storage() const { return m_storage; }
    size_t element_stride() const { return desc().element_layout->stride(); }

    bool is_contiguous() const;

    ref<BufferCursor> cursor(std::optional<int> start = std::nullopt, std::optional<int> count = std::nullopt) const;
    nb::dict uniforms() const;

    /// Clear buffer to 0s
    void clear(CommandEncoder* cmd = nullptr);

    /// Copy to CPU memory as a numpy array of correct stride/shape
    nb::ndarray<nb::numpy> to_numpy() const;
    /// Map GPU memory to torch tensor of correct stride/shape
    nb::ndarray<nb::pytorch> to_torch() const;
    /// Copy from CPU memory (as a numpy array) into GPU buffer
    void copy_from_numpy(nb::ndarray<nb::numpy> data);

    void point_to(ref<StridedBufferView> target);

protected:
    // In-place versions of view changing methods.
    // Derived classes are supposed the non-inplace versions by creating a copy
    // with the same buffer/view, call the inplace method on the copy, and return it
    void view_inplace(Shape shape, Shape strides = Shape(), int offset = 0);
    void broadcast_to_inplace(const Shape& shape);
    void index_inplace(nb::args args);

private:
    ref<Buffer> m_storage;
};

} // namespace sgl::slangpy
