// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cuda_interop.h"

#include "sgl/device/device.h"
#include "sgl/device/command.h"
#include "sgl/device/cuda_utils.h"

namespace sgl::cuda {

InteropBuffer::InteropBuffer(sgl::Device* device, const TensorView tensor_view, bool is_uav)
    : m_device(device)
    , m_tensor_view(tensor_view)
    , m_is_uav(is_uav)
{
    m_buffer = device->create_buffer({
        .size = m_tensor_view.size,
        .struct_size = 4,
        .usage = (is_uav ? BufferUsage::unordered_access : BufferUsage::shader_resource) | BufferUsage::shared,
        .default_state = is_uav ? ResourceState::unordered_access : ResourceState::shader_resource,
    });
    m_external_memory = make_ref<cuda::ExternalMemory>(m_buffer);
}

InteropBuffer::~InteropBuffer() { }

void InteropBuffer::copy_from_cuda(void* cuda_stream)
{
    SGL_CU_SCOPE(m_device);
    SGL_CU_CHECK(cuMemcpyAsync(
        reinterpret_cast<CUdeviceptr>(m_external_memory->mapped_data()),
        reinterpret_cast<CUdeviceptr>(m_tensor_view.data),
        m_tensor_view.size,
        static_cast<CUstream>(cuda_stream)
    ));
}

void InteropBuffer::copy_to_cuda(void* cuda_stream)
{
    SGL_CU_SCOPE(m_device);
    SGL_CU_CHECK(cuMemcpyAsync(
        reinterpret_cast<CUdeviceptr>(m_tensor_view.data),
        reinterpret_cast<CUdeviceptr>(m_external_memory->mapped_data()),
        m_tensor_view.size,
        static_cast<CUstream>(cuda_stream)
    ));
}

} // namespace sgl::cuda
