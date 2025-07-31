// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cuda_utils.h"

#include "sgl/core/error.h"

#include "sgl/device/device.h"
#include "sgl/device/resource.h"
#include "sgl/device/fence.h"
#include "sgl/device/command.h"

#include <span>

namespace sgl::cuda {

void* malloc_device(size_t size)
{
    CUdeviceptr ptr;
    SGL_CU_CHECK(cuMemAlloc(&ptr, size));
    return reinterpret_cast<void*>(ptr);
}

void free_device(void* ptr)
{
    if (!ptr)
        return;
    SGL_CU_CHECK(cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)));
}

void memcpy_device_to_device(void* dst, const void* src, size_t count)
{
    SGL_CU_CHECK(cuMemcpyDtoD(reinterpret_cast<CUdeviceptr>(dst), reinterpret_cast<CUdeviceptr>(src), count));
}

void memcpy_host_to_device(void* dst, const void* src, size_t count)
{
    SGL_CU_CHECK(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst), src, count));
}

void memcpy_device_to_host(void* dst, const void* src, size_t count)
{
    SGL_CU_CHECK(cuMemcpyDtoH(dst, reinterpret_cast<CUdeviceptr>(src), count));
}

void memset_device(void* dst, uint8_t value, size_t count)
{
    SGL_CU_CHECK(cuMemsetD8(reinterpret_cast<CUdeviceptr>(dst), value, count));
}

CUexternalMemory import_external_memory(const Buffer* buffer)
{
    SGL_CU_SCOPE(buffer->device());

    SGL_CHECK_NOT_NULL(buffer);
    SGL_CHECK(is_set(buffer->desc().usage, BufferUsage::shared), "Buffer was not created with shared usage.");
    NativeHandle shared_handle = buffer->shared_handle();
    SGL_CHECK(shared_handle, "Buffer shared handle creation failed.");

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC desc = {};
    switch (buffer->device()->type()) {
#if SGL_WINDOWS
    case DeviceType::d3d12:
        desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
        desc.handle.win32.handle = (void*)shared_handle.value();
        break;
    case DeviceType::vulkan:
        desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
        desc.handle.win32.handle = (void*)shared_handle.value();
        break;
#elif SGL_LINUX
    case DeviceType::vulkan:
        desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        desc.handle.fd = (int)(shared_handle.value());
        break;
#endif
    default:
        SGL_THROW("Unsupported device type {}.", buffer->device()->type());
    }
    desc.size = buffer->size();
    desc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;

    CUexternalMemory ext_mem;
    SGL_CU_CHECK(cuImportExternalMemory(&ext_mem, &desc));

    return ext_mem;
}

void destroy_external_memory(CUexternalMemory ext_mem)
{
    SGL_CU_CHECK(cuDestroyExternalMemory(ext_mem));
}

void* external_memory_get_mapped_buffer(CUexternalMemory ext_mem, size_t offset, size_t size)
{
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC desc = {};
    desc.offset = offset;
    desc.size = size;

    CUdeviceptr ptr = 0;
    SGL_CU_CHECK(cuExternalMemoryGetMappedBuffer(&ptr, ext_mem, &desc));
    return reinterpret_cast<void*>(ptr);
}

CUexternalSemaphore import_external_semaphore(const Fence* fence)
{
    SGL_CHECK_NOT_NULL(fence);
    SGL_CHECK(fence->desc().shared, "Fence was not created with shared flag.");
    NativeHandle shared_handle = fence->shared_handle();
    SGL_CHECK(shared_handle, "Fence shared handle creation failed.");

    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc = {};
    switch (fence->device()->type()) {
#if SGL_WINDOWS
    case DeviceType::d3d12:
        desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE;
        desc.handle.win32.handle = (void*)shared_handle.value();
        break;
    case DeviceType::vulkan:
        desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32;
        desc.handle.win32.handle = (void*)shared_handle.value();
        break;
#elif SGL_LINUX
    case DeviceType::vulkan:
        desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD;
        desc.handle.fd = (int)shared_handle.value();
        break;
#endif
    default:
        SGL_THROW("Unsupported device type {}.", fence->device()->type());
    }

    CUexternalSemaphore ext_sem;
    SGL_CU_CHECK(cuImportExternalSemaphore(&ext_sem, &desc));
    return ext_sem;
}

void destroy_external_semaphore(CUexternalSemaphore ext_sem)
{
    SGL_CU_CHECK(cuDestroyExternalSemaphore(ext_sem));
}

void signal_external_semaphore(CUexternalSemaphore ext_sem, uint64_t value, CUstream stream)
{
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS params = {};
    params.params.fence.value = value;
    SGL_CU_CHECK(cuSignalExternalSemaphoresAsync(&ext_sem, &params, 1, stream));
}

void wait_external_semaphore(CUexternalSemaphore ext_sem, uint64_t value, CUstream stream)
{
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params = {};
    params.params.fence.value = value;
    SGL_CU_CHECK(cuWaitExternalSemaphoresAsync(&ext_sem, &params, 1, stream));
}


inline int find_device_by_luid(int device_count, const AdapterLUID& luid)
{
    for (int i = 0; i < device_count; ++i) {
        CUdevice device;
        SGL_CU_CHECK(cuDeviceGet(&device, i));
#if SGL_WINDOWS
        // On Windows, we compare the 8-byte LUID. The LUID is the same for
        // D3D12, Vulkan and CUDA.
        std::array<char, 8> device_luid;
        unsigned int device_node_mask;
        SGL_CU_CHECK(cuDeviceGetLuid(device_luid.data(), &device_node_mask, device));
        static_assert(device_luid.size() <= sizeof(AdapterLUID));
        if (std::memcmp(device_luid.data(), luid.data(), device_luid.size()) == 0)
            return i;
#elif SGL_LINUX
        // On Linux, the LUID is not supported. Instead we compare the 16-byte
        // UUID which GFX conveniently returns in-place of the LUID.
        CUuuid device_uuid;
        SGL_CU_CHECK(cuDeviceGetUuid(&device_uuid, device));
        static_assert(sizeof(device_uuid) <= sizeof(AdapterLUID));
        if (std::memcmp(&device_uuid, luid.data(), sizeof(CUuuid)) == 0)
            return i;
#endif
    }
    return -1;
}

inline int find_device_by_name(int device_count, std::string_view name)
{
    for (int i = 0; i < device_count; ++i) {
        char device_name[256];
        SGL_CU_CHECK(cuDeviceGetName(device_name, sizeof(device_name), i));
        if (name == device_name)
            return i;
    }
    return -1;
}

inline int get_device_attribute(CUdevice device, CUdevice_attribute attribute)
{
    int value;
    SGL_CU_CHECK(cuDeviceGetAttribute(&value, attribute, device));
    return value;
}

Device::Device(const sgl::Device* device)
{
    SGL_CHECK_NOT_NULL(device);
    SGL_CHECK(rhiCudaDriverApiInit(), "Failed to load CUDA driver API.");
    SGL_CU_CHECK(cuInit(0));

    // Get number of available CUDA devices.
    int device_count;
    SGL_CU_CHECK(cuDeviceGetCount(&device_count));

    // First we try to find the matching CUDA device by LUID.
    int selected_device = find_device_by_luid(device_count, device->info().adapter_luid);
    if (selected_device < 0) {
        log_warn("Failed to find CUDA device by LUID. Falling back to device name.");
        // Next we try to find the matching CUDA device by name.
        selected_device = find_device_by_name(device_count, device->info().adapter_name);
        if (selected_device < 0) {
            log_warn("Failed to find CUDA device by name. Falling back to first compatible device.");
            // Finally we try to find the first compatible CUDA device.
            for (int i = 0; i < device_count; ++i) {
                if (get_device_attribute(i, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) >= 7) {
                    selected_device = i;
                    break;
                }
            }
        }
    }

    if (selected_device < 0)
        SGL_THROW("No compatible CUDA device found.");

    SGL_CU_CHECK(cuDeviceGet(&m_device, selected_device));
    SGL_CU_CHECK(cuDevicePrimaryCtxRetain(&m_context, m_device));
    m_owns_context = true;

    char name[256];
    SGL_CU_CHECK(cuDeviceGetName(name, sizeof(name), m_device));
    int major = get_device_attribute(m_device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    int minor = get_device_attribute(m_device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    log_debug("Created CUDA context on device \"{}\" (architecture {}.{}).", name, major, minor);
}

Device::Device(CUcontext context)
{
    m_context = context;
    SGL_CU_SCOPE(this);
    SGL_CU_CHECK(cuCtxGetDevice(&m_device));
    m_owns_context = false;

    char name[256];
    SGL_CU_CHECK(cuDeviceGetName(name, sizeof(name), m_device));
    int major = get_device_attribute(m_device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    int minor = get_device_attribute(m_device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    log_debug("Sharing CUDA context on device \"{}\" (architecture {}.{}).", name, major, minor);
}

Device::Device(CUdevice device)
{
    m_device = device;

    SGL_CU_CHECK(cuDevicePrimaryCtxRetain(&m_context, m_device));
    m_owns_context = true;
    char name[256];

    SGL_CU_CHECK(cuDeviceGetName(name, sizeof(name), m_device));
    int major = get_device_attribute(m_device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    int minor = get_device_attribute(m_device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    log_debug("Created CUDA context on device \"{}\" (architecture {}.{}).", name, major, minor);
}

Device::~Device()
{
    if (m_owns_context) {
        SGL_CU_CHECK(cuDevicePrimaryCtxRelease(m_device));
    }
}

AdapterLUID Device::adapter_luid() const
{
    AdapterLUID luid;
    memset(luid.data(), 0, luid.size());

#if SGL_WINDOWS
    // On Windows, we compare the 8-byte LUID. The LUID is the same for
    // D3D12, Vulkan and CUDA.
    std::array<char, 8> device_luid;
    unsigned int device_node_mask;
    SGL_CU_CHECK(cuDeviceGetLuid(device_luid.data(), &device_node_mask, m_device));
    static_assert(device_luid.size() <= sizeof(AdapterLUID));
    std::memcpy(luid.data(), device_luid.data(), device_luid.size());
#elif SGL_LINUX
    // On Linux, the LUID is not supported. Instead we compare the 16-byte
    // UUID which GFX conveniently returns in-place of the LUID.
    CUuuid device_uuid;
    SGL_CU_CHECK(cuDeviceGetUuid(&device_uuid, m_device));
    static_assert(sizeof(device_uuid) <= sizeof(AdapterLUID));
    std::memcpy(luid.data(), &device_uuid, sizeof(CUuuid));
#endif
    return luid;
}

std::string Device::adapter_name() const
{
    char name[256];
    SGL_CU_CHECK(cuDeviceGetName(name, sizeof(name), m_device));
    return std::string(name);
}

ExternalMemory::ExternalMemory(const Buffer* buffer)
    : m_resource(buffer)
    , m_external_memory(import_external_memory(buffer))
    , m_size(buffer->size())
{
}

ExternalMemory::~ExternalMemory()
{
    free_device(m_mapped_data);
    destroy_external_memory(m_external_memory);
}

void* ExternalMemory::mapped_data() const
{
    if (!m_mapped_data)
        m_mapped_data = external_memory_get_mapped_buffer(m_external_memory, 0, m_size);
    return m_mapped_data;
}

ExternalSemaphore::ExternalSemaphore(Fence* fence)
    : m_fence(fence)
    , m_external_semaphore(import_external_semaphore(fence))
{
}

ExternalSemaphore::~ExternalSemaphore()
{
    destroy_external_semaphore(m_external_semaphore);
}

void ExternalSemaphore::signal(uint64_t value, CUstream stream)
{
    signal_external_semaphore(m_external_semaphore, value, stream);
}

void ExternalSemaphore::wait(uint64_t value, CUstream stream)
{
    wait_external_semaphore(m_external_semaphore, value, stream);
}

ContextScope::ContextScope(const Device* device)
{
    SGL_CU_CHECK(cuCtxPushCurrent(device->context()));
}

ContextScope::ContextScope(const sgl::Device* device)
{
    SGL_CU_CHECK(cuCtxPushCurrent(device->cuda_device()->context()));
}

ContextScope::~ContextScope()
{
    CUcontext p;
    SGL_CU_CHECK(cuCtxPopCurrent(&p));
}

} // namespace sgl::cuda
