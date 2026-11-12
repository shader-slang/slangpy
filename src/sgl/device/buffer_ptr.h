// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/buffer_cursor.h"
#include "sgl/device/shader_cursor.h"

#include "sgl/device/resource.h"

#include <array>
#include <limits>
#include <string_view>

namespace sgl {

namespace detail {
    SGL_API uint64_t encode_buffer_ptr(DeviceType backend, uint64_t base, uint64_t byte_offset);
} // namespace detail

/// Non-owning, untyped GPU reference for the corresponding Slang BufferPtr.
/// The owner must retain the buffer through GPU use and declare resource states
/// and barriers. The eight-byte payload is backend-specific; it is not a CPU pointer.
/// Typed/aligned shader views are obtained in Slang. Default construction is null.
/// GPU allocation replacement invalidates existing values.
/// Metal requires device-wide residency sets for indirectly referenced buffers.
template<bool Writable = false>
class BufferPtr {
public:
    BufferPtr() = default;

    static void write_to_cursor(const ShaderCursor& cursor, const BufferPtr* value)
    {
        SGL_CHECK_NOT_NULL(value);
        value->write_to_cursor(cursor);
    }

    static void write_to_cursor(const BufferElementCursor& cursor, const BufferPtr* value)
    {
        SGL_CHECK_NOT_NULL(value);
        value->write_to_cursor(cursor);
    }

    static constexpr std::string_view slang_type_name
        = Writable ? "BufferPtr<void, Access.ReadWrite, 0>" : "BufferPtr<void, Access.Read, 0>";
    static constexpr std::array<std::string_view, 1> slangpy_imports() { return {"bufferptr"}; }

    /// Create a non-owning shader pointer into the buffer's current allocation.
    /// @param byte_offset Displacement in bytes from the buffer start. May equal buffer size for an end pointer,
    /// which cannot be dereferenced. A null buffer requires zero and produces a null pointer.
    /// @param alignment Required initial address alignment in bytes (4, 8 or 16). Checked without rounding;
    /// this does not set the shader's load_alignment, which defaults to plain access.
    /// @remark Rejects unsupported backends, out-of-range offsets, encoding overflow and misalignment.
    /// The caller retains the buffer and provides resource states and synchronization through GPU use.
    static SGL_API BufferPtr from_buffer(const ref<Buffer>& buffer, uint64_t byte_offset = 0, uint32_t alignment = 4);

    /// Backend-specific transport bits. On D3D12 these encode a descriptor and byte offset, not a device address.
    uint64_t data() const { return m_data; }
    /// Test for the null sentinel only; does not check resource lifetime, bounds or alignment.
    bool is_valid() const { return m_data != std::numeric_limits<uint64_t>::max(); }

    /// Write to the matching BufferPtr<void, access, 0> shader field.
    /// Typed, nonzero-load_alignment and different-access destinations are rejected; create typed/aligned
    /// views in the shader after binding. Writing the value does not retain its resource.
    template<typename TCursor>
    void write_to_cursor(const TCursor& cursor) const
    {
        SGL_CHECK(cursor.is_valid(), "Cannot write BufferPtr to an invalid cursor");
        check_cursor_type(cursor.slang_type_layout());
        cursor.set_data(&m_data, sizeof(m_data));
    }

private:
    static SGL_API void check_cursor_type(slang::TypeLayoutReflection* type_layout);

    explicit BufferPtr(uint64_t data)
        : m_data(data)
    {
    }
    uint64_t m_data = std::numeric_limits<uint64_t>::max();
};

using RWBufferPtr = BufferPtr<true>;

static_assert(sizeof(BufferPtr<>) == 8 && alignof(BufferPtr<>) == 8);
static_assert(sizeof(RWBufferPtr) == 8 && alignof(RWBufferPtr) == 8);

/// Resolve a non-owning read-only shader reference. See BufferPtr::from_buffer.
inline BufferPtr<> to_buffer_ptr(const ref<Buffer>& buffer, uint64_t byte_offset = 0, uint32_t alignment = 4)
{
    return BufferPtr<>::from_buffer(buffer, byte_offset, alignment);
}

/// Resolve a non-owning writable shader reference. See BufferPtr::from_buffer.
inline RWBufferPtr to_rw_buffer_ptr(const ref<Buffer>& buffer, uint64_t byte_offset = 0, uint32_t alignment = 4)
{
    return RWBufferPtr::from_buffer(buffer, byte_offset, alignment);
}

} // namespace sgl
