// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"

#include "sgl/device/fwd.h"
#include "sgl/device/fence.h"
#include "sgl/device/types.h"

#include <slang-rhi/cuda-driver-api.h>

#define SGL_CU_CHECK(call)                                                                                             \
    do {                                                                                                               \
        CUresult result = call;                                                                                        \
        if (result != CUDA_SUCCESS) {                                                                                  \
            const char* errorName;                                                                                     \
            cuGetErrorName(result, &errorName);                                                                        \
            const char* errorString;                                                                                   \
            cuGetErrorString(result, &errorString);                                                                    \
            SGL_THROW("CUDA call {} failed with error {} ({}).", #call, errorName, errorString);                       \
        }                                                                                                              \
    } while (0)

#define SGL_CU_SCOPE(device) sgl::cuda::ContextScope _context_scope(device)


namespace sgl::cuda {

SGL_API void* malloc_device(size_t size);
SGL_API void free_device(void* ptr);

SGL_API void memcpy_device_to_device(void* dst, const void* src, size_t count);
SGL_API void memcpy_host_to_device(void* dst, const void* src, size_t count);
SGL_API void memcpy_device_to_host(void* dst, const void* src, size_t count);

SGL_API void memset_device(void* dst, uint8_t value, size_t count);

SGL_API CUexternalMemory import_external_memory(const Buffer* buffer);
SGL_API void destroy_external_memory(CUexternalMemory ext_mem);
SGL_API void* external_memory_get_mapped_buffer(CUexternalMemory ext_mem, size_t offset, size_t size);

SGL_API CUexternalSemaphore import_external_semaphore(const Fence* fence);
SGL_API void destroy_external_semaphore(CUexternalSemaphore ext_sem);
SGL_API void signal_external_semaphore(CUexternalSemaphore ext_sem, uint64_t value, CUstream stream = 0);
SGL_API void wait_external_semaphore(CUexternalSemaphore ext_sem, uint64_t value, CUstream stream = 0);


/// Wraps a CUDA device, context and stream.
class SGL_API Device : public Object {
    SGL_OBJECT(cuda::Device)
public:
    /// Constructor.
    /// Creates a CUDA device on the same adapter as the sgl device.
    explicit Device(const sgl::Device* device);
    explicit Device(CUcontext context);
    explicit Device(CUdevice device);
    ~Device();

    CUdevice device() const { return m_device; }
    CUcontext context() const { return m_context; }

    AdapterLUID adapter_luid() const;
    std::string adapter_name() const;

private:
    CUdevice m_device{-1};
    CUcontext m_context{nullptr};
    bool m_owns_context{false};
};

/// Wraps an external memory resource.
class SGL_API ExternalMemory : public Object {
    SGL_OBJECT(cuda::ExternalMemory)
public:
    explicit ExternalMemory(const Buffer* buffer);
    ~ExternalMemory();

    size_t size() const { return m_size; }

    void* mapped_data() const;

private:
    /// Non-owning pointer to the resource.
    const Resource* m_resource;
    CUexternalMemory m_external_memory;
    size_t m_size;
    mutable void* m_mapped_data{nullptr};
};

/// Wraps an external semaphore.
class SGL_API ExternalSemaphore : public Object {
    SGL_OBJECT(cuda::ExternalSemaphore)
public:
    explicit ExternalSemaphore(Fence* fence);
    ~ExternalSemaphore();

    void signal(uint64_t value, CUstream stream = 0);
    void wait(uint64_t value, CUstream stream = 0);

private:
    /// Non-owning pointer to the fence.
    Fence* m_fence;
    CUexternalSemaphore m_external_semaphore;
};

class SGL_API ContextScope {
public:
    explicit ContextScope(const Device* device);
    explicit ContextScope(const sgl::Device* device);
    ~ContextScope();
};

} // namespace sgl::cuda
