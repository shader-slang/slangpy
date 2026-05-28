// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/core/memory_stream.h"
#include "sgl/core/error.h"

namespace sgl {

MemoryStream::MemoryStream(size_t capacity)
    : m_capacity(0)
    , m_size(0)
    , m_pos(0)
    , m_data(nullptr)
    , m_owns_data(true)
    , m_is_open(true)
    , m_is_writable(true)
{
    resize(capacity);
}

MemoryStream::MemoryStream(const void* data, size_t size)
    : m_capacity(size)
    , m_size(size)
    , m_pos(0)
    , m_data((uint8_t*)data)
    , m_owns_data(false)
    , m_is_open(true)
    , m_is_writable(false)
{
}

MemoryStream::MemoryStream(void* data, size_t size)
    : m_capacity(size)
    , m_size(size)
    , m_pos(0)
    , m_data(reinterpret_cast<uint8_t*>(data))
    , m_owns_data(false)
    , m_is_open(true)
    , m_is_writable(true)
{
}

MemoryStream::~MemoryStream()
{
    if (m_owns_data)
        std::free(m_data);
}

void MemoryStream::close()
{
    m_is_open = false;
}

void MemoryStream::read(void* p, size_t size)
{
    if (!is_open())
        SGL_THROW("Attempted to read from a closed memory stream");

    size_t gcount = size;
    if (m_pos + size > m_size) {
        gcount = m_size - m_pos;
    }

    std::memcpy(p, m_data + m_pos, gcount);
    m_pos += gcount;

    if (gcount < size)
        throw EOFException(fmt::format("Memory stream: read {} out of {} bytes", gcount, size), gcount);
}

void MemoryStream::write(const void* p, size_t size)
{
    if (!is_open())
        SGL_THROW("Attempted to write to a closed memory stream");

    if (!is_writable())
        SGL_THROW("Attempted to write to a read-only memory stream");

    size_t end_pos = m_pos + size;
    if (end_pos > m_size) {
        if (end_pos > m_capacity) {
            // Amortized reallocation
            auto new_capacity = m_capacity;
            while (end_pos > new_capacity)
                new_capacity *= 2;
            resize(new_capacity);
        }
        m_size = end_pos;
    }
    std::memcpy(m_data + m_pos, p, size);
    m_pos = end_pos;
}

void MemoryStream::truncate(size_t size)
{
    resize(size);
    m_size = size;
    if (m_pos > m_size)
        m_pos = m_size;
}

void MemoryStream::resize(size_t new_size)
{
    if (!m_owns_data)
        SGL_THROW("Cannot resize non-owned buffer");

    if (m_data == nullptr)
        m_data = reinterpret_cast<uint8_t*>(std::malloc(new_size));
    else
        m_data = reinterpret_cast<uint8_t*>(std::realloc(m_data, new_size));

    if (new_size > m_capacity)
        std::memset(m_data + m_capacity, 0, new_size - m_capacity);

    m_capacity = new_size;
}

} // namespace sgl
