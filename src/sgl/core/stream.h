// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/object.h"

namespace sgl {

class SGL_API EOFException : public std::runtime_error {
public:
    EOFException(const std::string& what, size_t gcount)
        : std::runtime_error(what)
        , m_gcount(gcount)
    {
    }

    size_t gcount() const { return m_gcount; }

private:
    size_t m_gcount;
};

/**
 * \brief Base class for all stream objects.
 */
class SGL_API Stream : public Object {
    SGL_OBJECT(Stream)
public:
    /// Returns true if the stream is open.
    virtual bool is_open() const = 0;

    /// Returns true if the stream is readable.
    virtual bool is_readable() const = 0;

    /// Returns true if the stream is writable.
    virtual bool is_writable() const = 0;

    /// Close the stream. No further read/writes are allowed.
    /// \note This is a no-op if the stream is already closed.
    virtual void close() = 0;

    /// Read data from the stream.
    /// Throws an exception if not all data could be read.
    virtual void read(void* p, size_t size) = 0;

    /// Write data to the stream.
    /// Throws an exception if not all data could be written.
    virtual void write(const void* p, size_t size) = 0;

    /// Seeks a position within the stream.
    virtual void seek(size_t pos) = 0;

    /// Truncate the stream to the given size.
    /// Throws an exception if stream is in read-only mode or does not support truncation.
    virtual void truncate(size_t size) = 0;

    /// Get the current position in the stream.
    virtual size_t tell() const = 0;

    /// Get the size of the stream.
    virtual size_t size() const = 0;

    /// Flush the internal buffers.
    virtual void flush() = 0;
};

} // namespace sgl
