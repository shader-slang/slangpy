// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "native_handle.h"

#include "sgl/core/config.h"

#if SGL_HAS_D3D12
#include <d3d12.h>
#endif

#if SGL_HAS_VULKAN
#include <vulkan/vulkan.h>
#endif

#include <slang-rhi/cuda-driver-api.h>

#include "sgl/stl/bit.h" // Replace with <bit> when available on all platforms.

namespace sgl {

template<typename T>
struct NativeHandleTrait { };

#define SGL_NATIVE_HANDLE(T, TYPE)                                                                                     \
    template<>                                                                                                         \
    struct NativeHandleTrait<T> {                                                                                      \
        static const NativeHandleType type = TYPE;                                                                     \
        static uint64_t pack(T native)                                                                                 \
        {                                                                                                              \
            return stdx::bit_cast<uint64_t>(native);                                                                   \
        }                                                                                                              \
        static T unpack(uint64_t value)                                                                                \
        {                                                                                                              \
            return stdx::bit_cast<T>(value);                                                                           \
        }                                                                                                              \
    };

#define SGL_NATIVE_HANDLE_32(T, TYPE)                                                                                  \
    template<>                                                                                                         \
    struct NativeHandleTrait<T> {                                                                                      \
        static const NativeHandleType type = TYPE;                                                                     \
        static uint64_t pack(T native)                                                                                 \
        {                                                                                                              \
            return stdx::bit_cast<uint32_t>(native);                                                                   \
        }                                                                                                              \
        static T unpack(uint64_t value)                                                                                \
        {                                                                                                              \
            return stdx::bit_cast<T>(narrow_cast<uint32_t>(value));                                                    \
        }                                                                                                              \
    };

#if SGL_HAS_D3D12
SGL_NATIVE_HANDLE(ID3D12Device*, NativeHandleType::D3D12Device);
SGL_NATIVE_HANDLE(ID3D12Resource*, NativeHandleType::D3D12Resource);
SGL_NATIVE_HANDLE(ID3D12PipelineState*, NativeHandleType::D3D12PipelineState);
SGL_NATIVE_HANDLE(ID3D12Fence*, NativeHandleType::D3D12Fence);
SGL_NATIVE_HANDLE(ID3D12CommandQueue*, NativeHandleType::D3D12CommandQueue);
SGL_NATIVE_HANDLE(ID3D12GraphicsCommandList*, NativeHandleType::D3D12GraphicsCommandList);
SGL_NATIVE_HANDLE(D3D12_CPU_DESCRIPTOR_HANDLE, NativeHandleType::D3D12CpuDescriptorHandle);
#endif // SGL_HAS_D3D12

#if SGL_HAS_VULKAN
SGL_NATIVE_HANDLE(VkInstance, NativeHandleType::VkInstance);
SGL_NATIVE_HANDLE(VkPhysicalDevice, NativeHandleType::VkPhysicalDevice);
SGL_NATIVE_HANDLE(VkDevice, NativeHandleType::VkDevice);
SGL_NATIVE_HANDLE(VkImage, NativeHandleType::VkImage);
SGL_NATIVE_HANDLE(VkImageView, NativeHandleType::VkImageView);
SGL_NATIVE_HANDLE(VkBuffer, NativeHandleType::VkBuffer);
// SGL_NATIVE_HANDLE(VkBufferView, NativeHandleType::VkBufferView);
SGL_NATIVE_HANDLE(VkPipeline, NativeHandleType::VkPipeline);
// SGL_NATIVE_HANDLE(VkFence, NativeHandleType::VkFence);
SGL_NATIVE_HANDLE(VkQueue, NativeHandleType::VkQueue);
SGL_NATIVE_HANDLE(VkCommandBuffer, NativeHandleType::VkCommandBuffer);
SGL_NATIVE_HANDLE(VkSampler, NativeHandleType::VkSampler);
#endif // SGL_HAS_VULKAN

SGL_NATIVE_HANDLE_32(CUdevice, NativeHandleType::CUdevice);
SGL_NATIVE_HANDLE(CUcontext, NativeHandleType::CUcontext);
SGL_NATIVE_HANDLE(CUstream, NativeHandleType::CUstream);

#undef SGL_NATIVE_HANDLE
#undef SGL_NATIVE_HANDLE_32

} // namespace sgl
