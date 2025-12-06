// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/object.h"

#include "sgl/device/fwd.h"

namespace sgl::cuda {

struct TensorView {
    int device_id;
    void* data;
    size_t size;
    size_t stride;
};

class InteropBuffer : public Object {
    SGL_OBJECT(cuda::InteropBuffer)
public:
    InteropBuffer(sgl::Device* device, const TensorView tensor_view, bool is_uav);
    ~InteropBuffer();

    const ref<sgl::Buffer>& buffer() const { return m_buffer; }
    bool is_uav() const { return m_is_uav; }

    void copy_from_cuda(void* cuda_stream = 0);
    void copy_to_cuda(void* cuda_stream = 0);

private:
    sgl::Device* m_device;
    TensorView m_tensor_view;
    bool m_is_uav;
    ref<sgl::Buffer> m_buffer;
    ref<cuda::ExternalMemory> m_external_memory;
};

} // namespace sgl::cuda
