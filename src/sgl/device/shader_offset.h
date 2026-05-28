// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <compare>

namespace sgl {

/**
 * Represents the offset of a shader variable relative to its enclosing type/buffer/block.
 *
 * A `ShaderOffset` can be used to store the offset of a shader variable that might use
 * ordinary/uniform data, resources like textures/buffers/samplers, or some combination.
 *
 * A `ShaderOffset` can also encode an invalid offset, to indicate that a particular
 * shader variable is not present.
 */
struct ShaderOffset {
    uint32_t uniform_offset{INVALID};
    uint32_t binding_range_index{INVALID};
    uint32_t binding_array_index{INVALID};

    /// Create an invalid offset.
    static ShaderOffset invalid() { return {}; }

    /// Create a zero offset.
    static ShaderOffset zero() { return ShaderOffset(0, 0, 0); }

    /// Default constructor. Creates an invalid offset.
    ShaderOffset() = default;

    /// Constructor.
    ShaderOffset(uint32_t uniform_offset_, uint32_t binding_range_index_, uint32_t binding_array_index_)
        : uniform_offset(uniform_offset_)
        , binding_range_index(binding_range_index_)
        , binding_array_index(binding_array_index_)
    {
    }

    /// Check whether this offset is valid.
    bool is_valid() const { return uniform_offset != INVALID; }

    /// Check whether this has valid uniform offset
    bool has_uniform_offset() const { return uniform_offset != INVALID; }

    /// Check whether this has valid binding offset
    bool has_binding_offset() const { return binding_range_index != INVALID; }

    /// Adds another offset to this offset.
    /// Returns an invalid offset if either offset is invalid.
    ShaderOffset operator+(const ShaderOffset& other) const
    {
        if (is_valid() && other.is_valid()) {
            return ShaderOffset(
                uniform_offset + other.uniform_offset,
                binding_range_index + other.binding_range_index,
                binding_array_index + other.binding_array_index
            );
        } else {
            return {};
        }
    }

    /// Comparison.
    auto operator<=>(const ShaderOffset&) const = default;

private:
    /// Invalid offset value.
    static constexpr uint32_t INVALID = uint32_t(-1);
};

} // namespace sgl
