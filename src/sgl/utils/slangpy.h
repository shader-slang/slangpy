// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/fwd.h"
#include "sgl/core/object.h"
#include "sgl/core/enum.h"
#include "sgl/device/fwd.h"
#include "sgl/device/native_handle.h"

#include <vector>
#include <map>

namespace sgl::slangpy {

enum class AccessType {
    none,
    read,
    write,
    readwrite,
};

SGL_ENUM_INFO(
    AccessType,
    {
        {AccessType::none, "none"},
        {AccessType::read, "read"},
        {AccessType::write, "write"},
        {AccessType::readwrite, "readwrite"},
    }
);
SGL_ENUM_REGISTER(AccessType);

enum class CallMode { prim = 0, bwds = 1, fwds = 2 };
SGL_ENUM_INFO(
    CallMode,
    {
        {CallMode::prim, "prim"},
        {CallMode::bwds, "bwds"},
        {CallMode::fwds, "fwds"},
    }
);
SGL_ENUM_REGISTER(CallMode);

enum class CallDataMode { global_data, entry_point };
SGL_ENUM_INFO(
    CallDataMode,
    {
        {CallDataMode::global_data, "global_data"},
        {CallDataMode::entry_point, "entry_point"},
    }
);
SGL_ENUM_REGISTER(CallDataMode);


class SGL_API Shape {
public:
    static constexpr size_t INLINE_CAPACITY = 8;

    Shape()
        : m_size(0)
        , m_valid(false)
        , m_uses_heap(false)
    {
    }

    /// Constructor from optional 'tuple'.
    Shape(const std::optional<std::vector<int>>& shape)
        : m_size(0)
        , m_valid(shape.has_value())
        , m_uses_heap(false)
    {
        if (m_valid) {
            const auto& vec = *shape;
            m_size = vec.size();
            if (m_size > INLINE_CAPACITY) {
                m_uses_heap = true;
                m_storage.heap_data = std::make_unique<int[]>(m_size);
                for (size_t i = 0; i < m_size; ++i) {
                    m_storage.heap_data[i] = vec[i];
                }
            } else {
                for (size_t i = 0; i < m_size; ++i) {
                    m_storage.inline_data[i] = vec[i];
                }
            }
        }
    }

    /// Constructor that creates a Shape of a given size with uninitialized values
    /// Use this when you need to populate the shape manually
    explicit Shape(size_t size)
        : m_size(size)
        , m_valid(true)
        , m_uses_heap(size > INLINE_CAPACITY)
    {
        if (m_uses_heap) {
            m_storage.heap_data = std::make_unique<int[]>(m_size);
            // Values are uninitialized - caller must populate them
        }
        // For inline storage, values are also uninitialized
    }

    /// Constructor that creates a Shape of a given size with all elements set to a value
    Shape(size_t size, int fill_value)
        : m_size(size)
        , m_valid(true)
        , m_uses_heap(size > INLINE_CAPACITY)
    {
        if (m_uses_heap) {
            m_storage.heap_data = std::make_unique<int[]>(m_size);
            for (size_t i = 0; i < m_size; ++i) {
                m_storage.heap_data[i] = fill_value;
            }
        } else {
            for (size_t i = 0; i < m_size; ++i) {
                m_storage.inline_data[i] = fill_value;
            }
        }
    }

    /// Static helper to create a Shape filled with ones
    static Shape ones(size_t size) { return Shape(size, 1); }

    /// Static helper to create a Shape filled with zeros
    static Shape zeros(size_t size) { return Shape(size, 0); }

    /// Constructor from initializer list
    Shape(std::initializer_list<int> shape)
        : m_size(shape.size())
        , m_valid(true)
        , m_uses_heap(shape.size() > INLINE_CAPACITY)
    {
        if (m_uses_heap) {
            m_storage.heap_data = std::make_unique<int[]>(m_size);
            size_t i = 0;
            for (int val : shape) {
                m_storage.heap_data[i++] = val;
            }
        } else {
            size_t i = 0;
            for (int val : shape) {
                m_storage.inline_data[i++] = val;
            }
        }
    }

    /// Copy constructor.
    Shape(const Shape& other)
        : m_size(other.m_size)
        , m_valid(other.m_valid)
        , m_uses_heap(other.m_uses_heap)
    {
        if (m_uses_heap) {
            m_storage.heap_data = std::make_unique<int[]>(m_size);
            for (size_t i = 0; i < m_size; ++i) {
                m_storage.heap_data[i] = other.m_storage.heap_data[i];
            }
        } else {
            for (size_t i = 0; i < m_size; ++i) {
                m_storage.inline_data[i] = other.m_storage.inline_data[i];
            }
        }
    }

    /// Move constructor.
    Shape(Shape&& other) noexcept
        : m_size(other.m_size)
        , m_valid(other.m_valid)
        , m_uses_heap(other.m_uses_heap)
    {
        if (m_uses_heap) {
            m_storage.heap_data = std::move(other.m_storage.heap_data);
            other.m_uses_heap = false;
            other.m_valid = false;
            other.m_size = 0;
        } else {
            for (size_t i = 0; i < m_size; ++i) {
                m_storage.inline_data[i] = other.m_storage.inline_data[i];
            }
        }
    }

    /// Destructor (default is fine now that we use struct instead of union)
    ~Shape() = default;

    /// Add operator combines the 2 shapes (optimized to avoid temporary allocations).
    Shape operator+(const Shape& other) const
    {
        Shape result;
        result.m_size = m_size + other.m_size;
        result.m_valid = true;
        result.m_uses_heap = result.m_size > INLINE_CAPACITY;

        if (result.m_uses_heap) {
            result.m_storage.heap_data = std::make_unique<int[]>(result.m_size);
            // Copy from this
            for (size_t i = 0; i < m_size; ++i) {
                result.m_storage.heap_data[i] = (*this)[i];
            }
            // Copy from other
            for (size_t i = 0; i < other.m_size; ++i) {
                result.m_storage.heap_data[m_size + i] = other[i];
            }
        } else {
            // Copy from this
            for (size_t i = 0; i < m_size; ++i) {
                result.m_storage.inline_data[i] = (*this)[i];
            }
            // Copy from other
            for (size_t i = 0; i < other.m_size; ++i) {
                result.m_storage.inline_data[m_size + i] = other[i];
            }
        }

        return result;
    }

    /// Assignment operator.
    Shape& operator=(const Shape& other)
    {
        if (this != &other) {
            m_size = other.m_size;
            m_valid = other.m_valid;
            m_uses_heap = other.m_uses_heap;

            if (m_uses_heap) {
                m_storage.heap_data = std::make_unique<int[]>(m_size);
                for (size_t i = 0; i < m_size; ++i) {
                    m_storage.heap_data[i] = other.m_storage.heap_data[i];
                }
            } else {
                for (size_t i = 0; i < m_size; ++i) {
                    m_storage.inline_data[i] = other.m_storage.inline_data[i];
                }
            }
        }
        return *this;
    }

    /// Indexers.
    int operator[](size_t i) const
    {
        SGL_ASSERT(i < m_size);
        return m_uses_heap ? m_storage.heap_data[i] : m_storage.inline_data[i];
    }

    int& operator[](size_t i)
    {
        SGL_ASSERT(i < m_size);
        return m_uses_heap ? m_storage.heap_data[i] : m_storage.inline_data[i];
    }

    /// Access to internal data as pointer (const version).
    const int* data() const
    {
        if (!m_valid) {
            SGL_THROW("Shape is invalid");
        }
        return m_uses_heap ? m_storage.heap_data.get() : m_storage.inline_data;
    }

    /// Access to internal data as pointer (mutable version).
    /// Use this in hot paths to avoid per-element branching on m_uses_heap.
    int* data()
    {
        if (!m_valid) {
            SGL_THROW("Shape is invalid");
        }
        return m_uses_heap ? m_storage.heap_data.get() : m_storage.inline_data;
    }

    /// Access to internal vector (creates a copy for compatibility).
    /// NOTE: This method allocates memory. Prefer using data() + size() or direct indexing.
    std::vector<int> as_vector() const
    {
        if (!m_valid) {
            SGL_THROW("Shape is invalid");
        }
        const int* ptr = m_uses_heap ? m_storage.heap_data.get() : m_storage.inline_data;
        return std::vector<int>(ptr, ptr + m_size);
    }

    /// Check if shape is valid (if the std::optional has a value).
    bool valid() const { return m_valid; }

    /// Get size (i.e. number of dimensions) of shape.
    size_t size() const { return m_size; }

    /// Iterator support for range-based for loops and algorithms
    const int* begin() const { return data(); }
    const int* end() const { return data() + m_size; }
    int* begin() { return data(); }
    int* end() { return data() + m_size; }

    /// Check if concrete shape (no dimensions are -1).
    bool concrete() const
    {
        const int* ptr = m_uses_heap ? m_storage.heap_data.get() : m_storage.inline_data;
        for (size_t i = 0; i < m_size; ++i) {
            if (ptr[i] == -1) {
                return false;
            }
        }
        return true;
    }

    /// Convert to string
    std::string to_string() const
    {
        if (!m_valid) {
            return "[invalid]";
        }
        const int* ptr = m_uses_heap ? m_storage.heap_data.get() : m_storage.inline_data;
        std::string result = "[";
        for (size_t i = 0; i < m_size; ++i) {
            if (i > 0)
                result += ", ";
            result += std::to_string(ptr[i]);
        }
        result += "]";
        return result;
    }

    /// Total element count (if this represented contiguous array)
    size_t element_count() const
    {
        const int* ptr = m_uses_heap ? m_storage.heap_data.get() : m_storage.inline_data;
        size_t result = 1;
        for (size_t i = 0; i < m_size; ++i) {
            result *= ptr[i];
        }
        return result;
    }

    /// Calculate the strides of a buffer of this shape, assuming it is contiguous.
    /// Optimized to avoid temporary allocations.
    Shape calc_contiguous_strides() const
    {
        if (!valid()) {
            return Shape();
        }

        const int* src_ptr = m_uses_heap ? m_storage.heap_data.get() : m_storage.inline_data;

        Shape result;
        result.m_size = m_size;
        result.m_valid = true;
        result.m_uses_heap = m_size > INLINE_CAPACITY;

        int total = 1;
        if (result.m_uses_heap) {
            result.m_storage.heap_data = std::make_unique<int[]>(m_size);
            for (int i = (int)m_size - 1; i >= 0; --i) {
                result.m_storage.heap_data[i] = total;
                total *= src_ptr[i];
            }
        } else {
            for (int i = (int)m_size - 1; i >= 0; --i) {
                result.m_storage.inline_data[i] = total;
                total *= src_ptr[i];
            }
        }

        return result;
    }

    bool operator==(const Shape& o) const
    {
        if (valid() != o.valid())
            return false;
        if (!valid() && !o.valid())
            return true;
        if (m_size != o.m_size)
            return false;

        const int* this_ptr = m_uses_heap ? m_storage.heap_data.get() : m_storage.inline_data;
        const int* other_ptr = o.m_uses_heap ? o.m_storage.heap_data.get() : o.m_storage.inline_data;
        for (size_t i = 0; i < m_size; ++i) {
            if (this_ptr[i] != other_ptr[i])
                return false;
        }
        return true;
    }

private:
    struct Storage {
        int inline_data[INLINE_CAPACITY];
        std::unique_ptr<int[]> heap_data;
    };

    Storage m_storage;
    size_t m_size;
    bool m_valid;
    bool m_uses_heap;
};

class SGL_API CallContext : Object {
public:
    CallContext(ref<Device> device, const Shape& call_shape, CallMode call_mode, NativeHandle cuda_stream)
        : m_device(std::move(device))
        , m_call_shape(call_shape)
        , m_call_mode(call_mode)
        , m_cuda_stream(cuda_stream)
    {
    }

    Device* device() const { return m_device.get(); }
    const Shape& call_shape() const { return m_call_shape; }
    CallMode call_mode() const { return m_call_mode; }

    const NativeHandle& cuda_stream() const { return m_cuda_stream; }

private:
    ref<Device> m_device;
    Shape m_call_shape;
    CallMode m_call_mode;
    NativeHandle m_cuda_stream;
};

} // namespace sgl::slangpy
