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
    Shape() = default;

    /// Constructor from optional 'tuple'.
    Shape(const std::optional<std::vector<int>>& shape)
        : m_shape(shape)
    {
    }

    /// Constructor from initializer list
    Shape(std::initializer_list<int> shape)
        : m_shape(shape)
    {
    }

    /// Copy constructor.
    Shape(const Shape& other)
        : m_shape(other.m_shape)
    {
    }

    /// Move constructor.
    Shape(Shape&& other) noexcept
        : m_shape(std::move(other.m_shape))
    {
    }

    /// Add operator combines the 2 shapes.
    Shape operator+(const Shape& other) const
    {
        auto& this_vec = as_vector();
        auto& other_vec = other.as_vector();
        std::vector<int> combined = this_vec;
        combined.insert(combined.end(), other_vec.begin(), other_vec.end());
        return Shape(combined);
    }

    /// Assignment operator.
    Shape& operator=(const Shape& other)
    {
        m_shape = other.m_shape;
        return *this;
    }

    /// Indexers.
    int operator[](size_t i) const { return as_vector()[i]; }
    int& operator[](size_t i) { return as_vector()[i]; }

    /// Access to internal vector.
    std::vector<int>& as_vector()
    {
        if (!m_shape) {
            SGL_THROW("Shape is invalid");
        }
        return *m_shape;
    }

    /// Const access to internal vector.
    const std::vector<int>& as_vector() const
    {
        if (!m_shape) {
            SGL_THROW("Shape is invalid");
        }
        return *m_shape;
    }

    /// Check if shape is valid (if the std::optional has a value).
    bool valid() const { return m_shape.has_value(); }

    /// Get size (i.e. number of dimensions) of shape.
    size_t size() const { return as_vector().size(); }

    /// Check if concrete shape (no dimensions are -1).
    bool concrete() const
    {
        for (auto dim : as_vector()) {
            if (dim == -1) {
                return false;
            }
        }
        return true;
    }

    /// Convert to string
    std::string to_string() const
    {
        if (!m_shape) {
            return "[invalid]";
        }
        return fmt::format("[{}]", fmt::join(as_vector(), ", "));
    }

    /// Total element count (if this represented contiguous array)
    size_t element_count() const
    {
        size_t result = 1;
        for (auto dim : as_vector()) {
            result *= dim;
        }
        return result;
    }

    /// Calculate the strides of a buffer of this shape, assuming it is contiguous.
    Shape calc_contiguous_strides() const
    {
        if (valid()) {
            auto& shape = as_vector();
            int total = 1;
            std::vector<int> strides(shape.size(), 1);
            for (int i = (int)shape.size() - 1; i >= 0; --i) {
                strides[i] = total;
                total *= shape[i];
            }
            return Shape(strides);
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

        return *m_shape == *o.m_shape;
    }

private:
    std::optional<std::vector<int>> m_shape;
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
