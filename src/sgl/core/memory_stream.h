// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/stream.h"

namespace sgl {

class SGL_API MemoryStream : public Stream {
    SGL_OBJECT(MemoryStream)
public:
    /// Create a read/write memory stream with the given initial capacity.
    MemoryStream(size_t capacity = 1024);

    /// Create a read-only memory stream from an existing buffer.
    MemoryStream(const void* data, size_t size);

    /// Create a read/write memory stream from an existing buffer.
    MemoryStream(void* data, size_t size);

    /// Destructor.
    virtual ~MemoryStream();

    virtual bool is_open() const override { return m_is_open; }

    virtual bool is_readable() const override { return true; }

    virtual bool is_writable() const override { return m_is_writable; }

    virtual void close() override;

    virtual void read(void* p, size_t size) override;

    virtual void write(const void* p, size_t size) override;

    virtual void seek(size_t pos) override { m_pos = pos; }

    virtual void truncate(size_t size) override;

    virtual size_t tell() const override { return m_pos; }

    virtual size_t size() const override { return m_size; }

    virtual void flush() override { }

    /// Return the current capacity of the underlying buffer.
    size_t capacity() const { return m_capacity; }

    /// Return true if stream owns the underlying buffer.
    bool owns_data() const { return m_owns_data; }

    /// Return the underlying raw buffer.
    const uint8_t* data() const { return m_data; }

protected:
    void resize(size_t new_size);

    /// Current capacity of the underlying buffer.
    size_t m_capacity;
    /// Current size of the stream.
    size_t m_size;
    /// Current position in the stream.
    size_t m_pos;
    /// Pointer to the underlying buffer.
    uint8_t* m_data;
    /// True if the stream owns the underlying buffer.
    bool m_owns_data;
    /// True if the stream is open.
    bool m_is_open;
    /// True if the stream is writable.
    bool m_is_writable;
};

} // namespace sgl
