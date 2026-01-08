// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/fwd.h"
#include "sgl/core/object.h"
#include "sgl/core/enum.h"
#include "sgl/device/fwd.h"

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

    /// Add operator combines the 2 shapes.
    Shape operator+(const Shape& other) const
    {
        std::vector<int> combined;
        combined.reserve(m_size + other.m_size);

        for (size_t i = 0; i < m_size; ++i) {
            combined.push_back((*this)[i]);
        }
        for (size_t i = 0; i < other.m_size; ++i) {
            combined.push_back(other[i]);
        }

        return Shape(std::optional<std::vector<int>>(combined));
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

    /// Access to internal data as pointer.
    const int* data() const
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

    /// Check if concrete shape (no dimensions are -1).
    bool concrete() const
    {
        for (size_t i = 0; i < m_size; ++i) {
            if ((*this)[i] == -1) {
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
        std::string result = "[";
        for (size_t i = 0; i < m_size; ++i) {
            if (i > 0)
                result += ", ";
            result += std::to_string((*this)[i]);
        }
        result += "]";
        return result;
    }

    /// Total element count (if this represented contiguous array)
    size_t element_count() const
    {
        size_t result = 1;
        for (size_t i = 0; i < m_size; ++i) {
            result *= (*this)[i];
        }
        return result;
    }

    /// Calculate the strides of a buffer of this shape, assuming it is contiguous.
    Shape calc_contiguous_strides() const
    {
        if (valid()) {
            int total = 1;
            std::vector<int> strides(m_size, 1);
            for (int i = (int)m_size - 1; i >= 0; --i) {
                strides[i] = total;
                total *= (*this)[i];
            }
            return Shape(std::optional<std::vector<int>>(strides));
        } else {
            return Shape();
        }
    }

    bool operator==(const Shape& o) const
    {
        if (valid() != o.valid())
            return false;
        if (!valid() && !o.valid())
            return true;
        if (m_size != o.m_size)
            return false;

        for (size_t i = 0; i < m_size; ++i) {
            if ((*this)[i] != o[i])
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
    CallContext(ref<Device> device, const Shape& call_shape, CallMode call_mode)
        : m_device(std::move(device))
        , m_call_shape(call_shape)
        , m_call_mode(call_mode)
    {
    }

    Device* device() const { return m_device.get(); }
    const Shape& call_shape() const { return m_call_shape; }
    CallMode call_mode() const { return m_call_mode; }

private:
    ref<Device> m_device;
    Shape m_call_shape;
    CallMode m_call_mode;
};

} // namespace sgl::slangpy
