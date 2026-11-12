// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/device/buffer_ptr.h"

#include "sgl/device/device.h"

namespace sgl {

namespace detail {
    uint64_t encode_buffer_ptr(DeviceType backend, uint64_t base, uint64_t byte_offset)
    {
        constexpr auto MAX = std::numeric_limits<uint64_t>::max();
        if (backend == DeviceType::d3d12) {
            SGL_CHECK_LT(base, uint64_t(std::numeric_limits<uint32_t>::max()));
            SGL_CHECK_LE(byte_offset, uint64_t(std::numeric_limits<uint32_t>::max()));
            return base | (byte_offset << 32);
        }
        SGL_CHECK(
            backend == DeviceType::vulkan || backend == DeviceType::cuda || backend == DeviceType::metal,
            "BufferPtr does not support this backend"
        );
        SGL_CHECK_LT(base, MAX);
        SGL_CHECK_LT(byte_offset, MAX - base);
        return base + byte_offset;
    }
} // namespace detail

template<bool Writable>
BufferPtr<Writable>
BufferPtr<Writable>::from_buffer(const ref<Buffer>& buffer, uint64_t byte_offset, uint32_t alignment)
{
    SGL_CHECK(alignment == 4 || alignment == 8 || alignment == 16, "Expected alignment 4, 8 or 16");
    if (!buffer) {
        SGL_CHECK_EQ(byte_offset, 0);
        return {};
    }
    const auto backend = buffer->device()->type();
    SGL_CHECK(
        backend == DeviceType::d3d12 || backend == DeviceType::vulkan || backend == DeviceType::cuda
            || backend == DeviceType::metal,
        "BufferPtr supports D3D12, Vulkan, CUDA and Metal"
    );
    if (backend == DeviceType::metal) {
        // Reflection cannot discover resources reached through GPU pointer tables.
        SGL_CHECK(
            buffer->device()->has_feature(Feature::residency_set),
            "Metal BufferPtr requires device-wide residency sets"
        );
    }
    SGL_CHECK_LE(byte_offset, buffer->size());
    const uint64_t address = buffer->device_address();
    SGL_CHECK_LE(byte_offset, std::numeric_limits<uint64_t>::max() - address);
    SGL_CHECK_EQ((address + byte_offset) % alignment, 0);
    if (backend == DeviceType::d3d12) {
        SGL_CHECK_EQ(byte_offset % alignment, 0);
        const auto descriptor = Writable ? buffer->descriptor_handle_rw() : buffer->descriptor_handle_ro();
        return BufferPtr(detail::encode_buffer_ptr(backend, descriptor.value, byte_offset));
    }
    return BufferPtr(detail::encode_buffer_ptr(backend, address, byte_offset));
}

template<bool Writable>
void BufferPtr<Writable>::check_cursor_type(slang::TypeLayoutReflection* type_layout)
{
    Slang::ComPtr<slang::IBlob> name;
    type_layout->getType()->getFullName(name.writeRef());
    SGL_CHECK(
        name
            && std::string_view(static_cast<const char*>(name->getBufferPointer()), name->getBufferSize())
                == slang_type_name,
        "BufferPtr cursor type must be {}",
        slang_type_name
    );
}

template class BufferPtr<false>;
template class BufferPtr<true>;


} // namespace sgl
